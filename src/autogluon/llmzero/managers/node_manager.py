"""
Node-based manager using pure Monte Carlo Tree Search. It implements a tree-based
search strategy that allows for more flexible exploration and exploitation of solution
space. It also ensures all available tools are tried during the exploration process.
"""

import logging
import math
import os
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Literal, Optional, Set

from ..llm import ChatLLMFactory
from ..utils import extract_hp_decisions

logger = logging.getLogger(__name__)


# ── ExTS scoring primitives ────────────────────────────────────────────────
# Ported from exts0516.py. These operate on normalized-and-bounded scores so the
# same temperature/constants work regardless of the framework's absolute range.


def _phi(x: float, temperature: float) -> float:
    """Exponential shaping: phi(x) = (e^(x/T) - 1) / (e^(1/T) - 1), x in [0, 1].

    Convex for T > 0 (favors high scores); identity for T <= 0.
    """
    if temperature <= 0:
        return x
    exp_inv_t = math.exp(1.0 / temperature)
    return (math.exp(x / temperature) - 1.0) / (exp_inv_t - 1.0)


def _shaped_mean(history: List[float], score_min: float, score_max: float, temperature: float) -> float:
    """Mean of individually min-max-normalized, phi-shaped observations.

    `history` holds raw scores already routed through the framework's
    normalize_score (higher-is-better space). Empty history -> 0.0; a degenerate
    range (score_max <= score_min) -> 1.0 (mirrors the single-scored-node case).
    """
    if not history:
        return 0.0
    if score_max <= score_min:
        return 1.0
    span = score_max - score_min
    total = 0.0
    for r in history:
        x = (r - score_min) / span
        x = max(0.0, min(1.0, x))
        total += _phi(x, temperature)
    return total / len(history)


def _success_rate(validated_visits: int, visits: int, non_success: int, failure_offset: int) -> float:
    """ExTS success rate with the failure-forgiveness offset folded into the denominator.

    success_rate = validated_visits / max(1, visits - min(non_success, offset)), clamped [0,1].
    `non_success` = failures + no-score runs. offset=0 => plain validated_visits / visits.
    """
    forgiven = min(non_success, failure_offset)
    effective_visits = max(1, visits - forgiven)
    return min(1.0, validated_visits / effective_visits)


def _exploration_term(
    exploration_type: str,
    exploration_constant: float,
    parent_visits: float,
    node_visits: float,
    puct_parent_exp: float,
) -> float:
    """Exploration bonus.

    UCT:  C * sqrt(ln(n_parent) / n_v)          — logarithmic decay
    PUCT: C * n_parent^p / (1 + n_v)            — linear decay, faster convergence
    """
    if exploration_type == "puct":
        return exploration_constant * (parent_visits**puct_parent_exp) / (1 + node_visits)
    return exploration_constant * math.sqrt(math.log(parent_visits) / node_visits)


def _generate_node_timestamp() -> str:
    """
    Generate a timestamp for the run in the format YYMMDD_HHMM_<2_digit_uuid>.

    This timestamp is generated once per run and shared across all nodes.

    Returns:
        str: Timestamp string
    """
    from datetime import datetime

    # Get current date/time in YYMMDD_HHMM format
    datetime_str = datetime.now().strftime("%y%m%d_%H%M")

    # Generate a 2-digit hex UUID
    uuid_str = uuid.uuid4().hex[:2]

    return f"{datetime_str}_{uuid_str}"


@dataclass
class Node:
    """
    A node in the solution tree representing a single iteration.
    Stores code, execution results, and evaluation information.
    """

    # Node creation time
    ctime: float = field(default_factory=lambda: time.time())

    # Tree structure
    parent: Optional["Node"] = None
    children: Set["Node"] = field(default_factory=set)

    # Node position in tree
    time_step: int = None  # Corresponds to the global time step when created
    depth: int = 0  # Depth in the tree (root=0, increases with each level)
    timestamp: str = ""  # Deprecated: timestamp is now run-level, stored in NodeManager.run_timestamp

    # Solution stage
    stage: Literal["root", "debug", "evolve"] = "root"

    # MCTS statistics
    visits: int = 0
    validated_visits: int = 0  # Number of successful runs with validation scores
    failure_visits: int = 0  # Number of failed runs
    unvalidated_visits: int = 0  # Number of successful runs without validation scores
    validated_reward: float = 0.0  # Total reward from validated runs
    # Raw validated scores, one entry per validated visit (this node + backprop from
    # descendants). ExTS shapes each observation individually rather than shaping the
    # mean, so the full history is required. Populated in update().
    reward_history: List[float] = field(default_factory=list)
    # total_reward: float = 0.0  # Replaced by separate reward tracking

    # Node state tracking
    is_successful: bool = False  # Did the execution succeed?
    is_debug_successful: bool = False  # Did the debug in the subtree succeed?
    is_terminal: bool = False  # Should this node not be expanded further?
    debug_attempts: int = 0  # Number of debug attempts on this node
    early_stopped: bool = False  # Was this execution early stopped by the early stopper?
    early_stop_reason: str = ""  # Reason for early stopping (if applicable)

    # Solution artifacts
    entry_point_script: str = ""  # Environment-agnostic training script (entry point)
    execution_script: str = ""  # Environment-specific execution wrapper (e.g., Ray job submission)
    remote_execution_wrapper: str = (
        ""  # Wrapper script that calls run.sh with execution_script to support remote execution
    )
    tool_used: str = ""  # The primary tool used for this solution
    tools_available: List[str] = field(
        default_factory=list
    )  # All tools available for this solution, in priority order
    tutorial_retrieval: str = ""  # Retrieved tutorials for this node
    tutorial_prompt: str = ""  # Processed tutorial prompt for this node

    # Execution results
    stdout: str = ""
    stderr: str = ""
    execution_time: float = 0.0
    error_message: str = ""
    error_analysis: str = ""

    # Evaluation metrics
    validation_score: Optional[float] = None

    # Observer results
    observer_results: dict = field(default_factory=dict)

    # Checkpoint tracking — ALL checkpoints produced during this node's training run
    checkpoints: List[dict] = field(default_factory=list)  # [{step, path, timestamp}]

    # Checkpoint resume — the specific checkpoint this node resumed from
    resume_from_checkpoint: bool = False
    resume_checkpoint_path: Optional[str] = None
    resume_checkpoint_step: Optional[int] = None
    # Number of NEW gradient steps this node should train, capped by
    # checkpoint_optimization.max_steps_per_node. For from-scratch nodes this is the full
    # run length; for resumed nodes it is the additional steps beyond resume_checkpoint_step.
    # Enforced via trainer.total_training_steps (= resume_checkpoint_step + training_steps_budget).
    training_steps_budget: Optional[int] = None

    # Execution strategy tracking — whether this node trained from scratch or resumed
    from_scratch: bool = True  # True if training from scratch, False if resuming from checkpoint
    forced_from_scratch: bool = False  # True if forced from-scratch by exploration mechanism

    # Progressive-widening latch: once an evolve node's score clears the widening quantile
    # its widening is unlocked permanently. The visit-driven widening bonus is monotonic in
    # visits, so latching the (tree-relative, non-monotonic) score gate keeps the effective
    # child cap non-decreasing. This prevents a transient score dip from shrinking the cap
    # and causing _check_ancestors_terminal to permanently terminalize the node.
    widening_unlocked: bool = False

    # MCTS stats for early-stopped runs
    early_stopped_visits: int = 0

    # Locking for thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock)
    expected_child_count: int = 0

    @property
    def id(
        self,
    ):
        return self.time_step

    def __post_init__(self):
        """
        Initialize a node, adding it to parent's children if parent exists.
        Set depth based on parent's depth.
        """
        if self.parent is not None:
            self.parent.add_child(self)
            self.depth = self.parent.depth + 1

    def add_child(self, child: "Node") -> None:
        """
        Add a child node to this node.
        """
        logger.detail(f"Node {child.id} is added to children of Node {self.id}.")
        self.children.add(child)

    def remove_child(self, child: "Node") -> None:
        """
        Remove a child node of this node.
        """
        logger.detail(f"Node {child.id} is removed from children of Node {self.id}.")
        self.children.remove(child)

    @property
    def is_leaf(self) -> bool:
        """
        Check if the node is a leaf node (has no children).
        """
        return len(self.children) == 0

    @property
    def num_children(self) -> int:
        """
        Get the number of child nodes.
        """
        return len(self.children)

    @property
    def prev_tutorial_prompt(self) -> str:
        if self.parent and self.parent.tutorial_prompt:
            return self.parent.tutorial_prompt

    def update(
        self, reward: float, is_validated: bool = False, is_failure: bool = False, is_early_stopped: bool = False
    ) -> None:
        """
        Update the node's statistics with a new reward.

        Args:
            reward: The raw validation score (for validated runs) or None
            is_validated: Whether this reward comes from a validated run
            is_failure: Whether this run was a failure
            is_early_stopped: Whether this run was early stopped (incomplete)
        """
        with self._lock:
            self.visits += 1

            if is_failure:
                self.failure_visits += 1
            elif is_early_stopped:
                self.early_stopped_visits += 1
                # An early-stopped run with a validation score is treated as a normal
                # validated success (ExTS: its score drives exploitation like any other;
                # there is no separate early-stop penalty term).
                if is_validated and reward is not None:
                    self.validated_visits += 1
                    self.validated_reward += reward
                    self.reward_history.append(reward)
            elif is_validated and reward is not None:
                # For validated runs, store the raw validation score
                self.validated_visits += 1
                self.validated_reward += reward  # Sum up raw scores (mean used as fallback bound)
                self.reward_history.append(reward)  # Per-observation history for ExTS phi-shaping
            else:
                # For successful runs without validation
                self.unvalidated_visits += 1

    def uct_value(
        self,
        exploration_constant: float = 1.414,
        failure_offset: float = 0,
        score_min: float = 0.0,
        score_max: float = 0.0,
        score_temperature: float = 0.3,
        success_rate_alpha: float = 1.414,
        exploration_type: str = "uct",
        puct_parent_exp: float = 0.5,
        normalized_history: Optional[List[float]] = None,
    ) -> float:
        """Compute the ExTS score of this node (exploitation + exploration).

        Exploitation (ExTS):
            success_rate = validated_visits / max(1, visits - min(failures, offset))
            exploitation = success_rate ** alpha * shaped_mean(observations)

        `success_rate` folds the failure-forgiveness offset into the denominator
        (offset default 0 => plain validated_visits / visits). Runs that produced no
        usable validation score (failures + unvalidated) live in the denominator only,
        so they depress success_rate exactly like a failure. There is no additive
        failure or early-stop penalty term.

        `shaped_mean` shapes each observation individually (see module `_shaped_mean`),
        so the scale-invariant temperature applies per-observation, not to the mean.

        Args:
            exploration_constant: Exploration weight C.
            failure_offset: Non-success runs to forgive in the success_rate denominator.
            score_min / score_max: Bounds for per-observation min-max normalization
                (already in the framework-normalized, higher-is-better space).
            score_temperature: phi shaping temperature (lower = sharper).
            success_rate_alpha: Exponent on success_rate (>1 amplifies failures).
            exploration_type: "uct" or "puct".
            puct_parent_exp: Exponent p on parent visits for PUCT.
            normalized_history: Per-observation scores already routed through the
                framework's normalize_score. When None, falls back to this node's raw
                reward_history, and to a single-observation [avg] when that is empty
                (keeps direct-call tests that set only validated_reward working).

        Returns:
            The node's selection score.
        """
        # For unvisited nodes, return infinity to ensure they are visited first.
        if self.visits == 0:
            return float("inf")

        parent_visits = max(1, self.parent.visits) if self.parent else 1

        # Non-success runs (failures + no-score) that a positive offset forgives.
        non_success = self.failure_visits + self.unvalidated_visits
        self.success_rate = _success_rate(self.validated_visits, self.visits, non_success, failure_offset)

        if self.validated_visits > 0:
            history = normalized_history if normalized_history is not None else list(self.reward_history)
            if not history:
                # Cold path only: update() keeps validated_visits and reward_history in
                # lockstep, so in production a validated node always has history and this
                # branch is unreachable. Direct/test calls that set validated_reward without
                # reward_history land here; the raw avg is NOT normalize_score-routed, so this
                # is direction-correct only for maximize frameworks (acceptable for tests).
                history = [self.validated_reward / self.validated_visits]
            self.shaped_mean = _shaped_mean(history, score_min, score_max, score_temperature)
            self.exploitation = (self.success_rate**success_rate_alpha) * self.shaped_mean
        else:
            self.shaped_mean = 0.0
            self.exploitation = 0.0

        self.exploration = _exploration_term(
            exploration_type, exploration_constant, parent_visits, self.visits, puct_parent_exp
        )

        return self.exploitation + self.exploration

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)


class NodeManager:
    """
    Manages a tree of nodes representing different iterations of solution development.
    Uses Monte Carlo Tree Search (MCTS) to explore the solution space more effectively.
    """

    def __init__(
        self,
        input_data_folder: str,
        output_folder: str,
        config: Any,
        initial_user_input: str,
        enable_per_iteration_instruction: bool,
        wandb_api_key: Optional[str] = None,
        wandb_project_postfix: str = "",
        run_postfix: str = "",
    ):
        """
        Initialize the NodeManager with required paths and configuration.

        Args:
            input_data_folder: Path to input data directory
            output_folder: Path to output directory
            config: Configuration object
            initial_user_input: Initial user instruction
            enable_per_iteration_instruction: If asking for per iteration user input
            wandb_api_key: Optional WandB API key for experiment tracking
            wandb_project_postfix: Optional postfix for wandb project names
            run_postfix: Optional postfix appended to node output folder names
        """
        # Store required paths
        self.input_data_folder = input_data_folder
        self.output_folder = output_folder

        # Validate paths
        for path, name in [(input_data_folder, "input_data_folder")]:
            if not Path(path).exists():
                raise FileNotFoundError(f"{name} not found: {path}")

        # Create output folder if it doesn't exist
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        self.config = config
        self.enable_per_iteration_instruction = enable_per_iteration_instruction
        self.initial_user_input = initial_user_input
        self.wandb_api_key = wandb_api_key
        self.wandb_project_postfix = wandb_project_postfix
        self.run_postfix = run_postfix

        # Training-framework handle drives score-direction-aware comparisons and
        # UCT normalization (defaults to "maximize", an identity pass-through).
        from ..training_frameworks import get_framework_from_config

        self._score_framework = get_framework_from_config(config)

        # Generate single timestamp for entire run (shared across all nodes)
        self.run_timestamp = _generate_node_timestamp()

        # Track time_step
        self.time_step = -1
        # Create root node
        self.root_node = Node(stage="root", time_step=self.time_step, depth=0)
        self.current_node = self.root_node

        # Track best nodes and metrics
        self._best_node = None
        self._best_validation_score = None
        self._worst_validation_score = None
        self.last_successful_node = None

        # Best run metrics for early stopping comparison
        self._best_run_metrics_history: Optional[List] = None
        self._best_run_observer_results: Optional[dict] = None

        # Key node tracking
        self.best_step = -1
        self.last_successful_step = -1

        # MCTS parameters
        self.exploration_constant = self.config.exploration_constant
        self.max_debug_depth = self.config.max_debug_depth
        self.failure_offset = getattr(self.config, "failure_offset", 0)
        self.score_temperature = getattr(self.config, "score_temperature", 0.3)

        # ExTS parameters (see docs/exts-node-search-plan.md)
        self.success_rate_alpha = getattr(self.config, "success_rate_alpha", 1.414)
        self.exploration_type = getattr(self.config, "exploration_type", "uct")
        self.puct_parent_exp = getattr(self.config, "puct_parent_exp", 0.5)
        self.use_gated_expansion = getattr(self.config, "use_gated_expansion", True)
        self.expansion_gate_quantile = getattr(self.config, "expansion_gate_quantile", 0.25)
        self.use_progressive_widening = getattr(self.config, "use_progressive_widening", True)
        self.widening_base = getattr(self.config, "widening_base", 2.0)
        self.widening_visit_threshold = getattr(self.config, "widening_visit_threshold", 32)
        self.widening_score_quantile = getattr(self.config, "widening_score_quantile", 0.75)

        # Seeded RNG for ExTS random tie-breaking and Thompson-style virtual-child sampling.
        self.rng = random.Random(getattr(self.config, "random_seed", 0))

        # Global raw-observation score bounds (framework-normalized space), maintained
        # incrementally on each validated backprop (ExTS-faithful). Used for per-observation
        # phi-shaping. Init to +/-inf; the degenerate-range guard covers the empty case.
        self._score_min = float("inf")
        self._score_max = float("-inf")

        # Checkpoint exploration tracking
        self.from_scratch_count = 0
        self.resume_count = 0

        # Checkpoint exploration parameters
        ckpt_config = getattr(self.config, "checkpoint_optimization", None)
        self.min_from_scratch_ratio = getattr(ckpt_config, "min_from_scratch_ratio", 0.2) if ckpt_config else 0.2

        # Tracking for thread safety
        self._node_lock = threading.Lock()
        self.search_start_time = time.time()

        # User inputs storage
        self.user_inputs = []

        self.description_file_contents = None

        # Error analysis storage
        self._all_error_analyses = []

        # Sibling config tracking for deduplication (parent_id -> list of config deltas)
        self._sibling_configs = {}

        # Tool tracking

        # Target prompt instance for meta-prompting
        self.target_prompt_instance = None

        # Initialize the agent components
        self._init_agents()

    def _init_agents(self):
        """Initialize all required agents."""
        from ..agents import (
            ErrorAnalyzerAgent,
            ExecuterAgent,
            ExecutionCoderAgent,
            PostTrainingCoderAgent,
        )
        from ..early_stoppers import AgenticEarlyStopper
        from ..proposers import AgenticProposer

        # Error analyzer
        self.error_analyzer = ErrorAnalyzerAgent(
            config=self.config,
            manager=self,
            llm_config=self.config.error_analyzer,
            prompt_template=None,
        )

        # Initialize proposer (unified text/vision support via config)
        logger.info("Initializing AgenticProposer (unified text/vision)")

        # Get proposer LLM config (with higher temperature for creativity)
        proposer_llm_config = getattr(self.config, "proposer_llm", self.config.bash_coder)

        proposer = AgenticProposer(
            config=self.config,
            manager=self,
            agent_llm_config=proposer_llm_config,
            agent_prompt_template=None,
        )
        self.proposers = [("unified", proposer)]

        # Initialize early stopper (unified text/vision support via config)
        early_stopper_config = getattr(self.config, "early_stopper", {})
        early_stopper_enabled = getattr(early_stopper_config, "enabled", False)

        self.early_stoppers = []
        if early_stopper_enabled:
            logger.info("Initializing AgenticEarlyStopper (unified text/vision)")
            stopper = AgenticEarlyStopper(
                config=self.config,
                manager=self,
                agent_llm_config=self.config.bash_coder,
                agent_prompt_template=None,
            )
            self.early_stoppers.append(("unified", stopper))
        else:
            logger.info("Early stopping is disabled")

        # Post-training script coder (entry point generation)
        self.post_training_coder = PostTrainingCoderAgent(
            config=self.config,
            manager=self,
            llm_config=self.config.bash_coder,  # Reuse Bash coder config
            prompt_template=None,
        )

        # Execution script coder (environment-specific wrapper generation)
        self.execution_coder = ExecutionCoderAgent(
            config=self.config,
            manager=self,
            llm_config=self.config.bash_coder,  # Reuse Bash coder config
            prompt_template=None,
        )

        # Executer
        self.executer = ExecuterAgent(
            config=self.config,
            manager=self,
            language="bash",
            timeout=self.config.per_execution_timeout,
            executer_llm_config=self.config.executer,
            executer_prompt_template=None,
        )

    def initialize(self):
        """Initialize the manager: read description files directly."""
        self.description_file_contents = self._read_description_files()
        self.task_description = self.initial_user_input
        self.data_prompt = self.description_file_contents
        self.available_tools = []

    def _read_description_files(self) -> str:
        """Read all .txt and .md files from input_data_folder directly."""
        from pathlib import Path

        contents = []
        input_path = Path(self.input_data_folder)
        for ext in ("*.txt", "*.md"):
            for filepath in sorted(input_path.glob(ext)):
                try:
                    content = filepath.read_text()
                    contents.append(content)
                except Exception as e:
                    logger.warning(f"Could not read {filepath}: {e}")
        return "\n\n".join(contents) if contents else ""

    def get_iteration_folder(self, node: Node) -> str:
        """
        Get the folder for storing iteration artifacts.

        Args:
            node: The node to get the folder for

        Returns:
            Path to the iteration folder
        """
        if node.id < 0:
            iter_folder = os.path.join(self.output_folder, "node_init")
        else:
            iter_folder = os.path.join(self.output_folder, f"node_{node.id}")
        os.makedirs(iter_folder, exist_ok=True)
        return iter_folder

    def get_per_iteration_output_folder(self, node: Node) -> str:
        """
        Get the folder for storing iteration output artifacts.

        Args:
            node: The node to get the output folder for

        Returns:
            Path to the iteration output folder
        """
        iter_output_folder = os.path.join(self.get_iteration_folder(node), "output")
        os.makedirs(iter_output_folder, exist_ok=True)
        return iter_output_folder

    def save_and_log_states(self, content, save_name, per_iteration=False, add_uuid=False, node=None):
        """
        Save states to a file and log them.

        Args:
            content: Content to save
            save_name: Name for the saved file
            per_iteration: Whether this is for a specific iteration (backward compatibility)
            add_uuid: Whether to add a UUID to the filename
            node: Node to associate with the saved content (required if per_iteration is False)
        """
        if add_uuid:
            # Split filename and extension
            name, ext = os.path.splitext(save_name)
            # Generate 4-digit UUID (using first 4 characters of hex)
            uuid_suffix = str(uuid.uuid4()).replace("-", "")[:4]
            save_name = f"{name}_{uuid_suffix}{ext}"

        # Determine the save directory
        if per_iteration and self.current_node:
            states_dir = os.path.join(self.get_iteration_folder(self.current_node), "states")
        elif node:
            states_dir = os.path.join(self.get_iteration_folder(node), "states")
        else:
            states_dir = os.path.join(self.output_folder, "states")

        os.makedirs(states_dir, exist_ok=True)
        output_file = os.path.join(states_dir, save_name)

        logger.info(f"Saving {output_file}...")
        with open(output_file, "w") as file:
            if content is not None:
                if isinstance(content, list):
                    # Join list elements with newlines
                    file.write("\n".join(str(item) for item in content))
                else:
                    # Handle as string (original behavior)
                    file.write(content)
            else:
                file.write("<None>")

    def _save_observer_results(self, observer_results: dict):
        """
        Save observer results to separate files.

        Args:
            observer_results: Dictionary mapping observer names to their results
        """
        import json

        if not observer_results:
            logger.debug("No observer results to save")
            return

        # Save step metrics if available
        if "step_metrics" in observer_results:
            metrics_result = observer_results["step_metrics"]
            metrics_content = json.dumps(metrics_result, indent=2)
            self.save_and_log_states(
                metrics_content,
                "step_metrics.json",
                node=self.current_node,
                add_uuid=False,
            )

        # Save validation metrics if available
        if "validation" in observer_results:
            val_result = observer_results["validation"]
            val_content = json.dumps(val_result, indent=2)
            self.save_and_log_states(
                val_content,
                "validation_metrics.json",
                node=self.current_node,
                add_uuid=False,
            )

        # Save checkpoint paths if available
        if "checkpoint_paths" in observer_results:
            ckpt_result = observer_results["checkpoint_paths"]
            ckpt_content = json.dumps(ckpt_result, indent=2)
            self.save_and_log_states(
                ckpt_content,
                "checkpoint_paths.json",
                node=self.current_node,
                add_uuid=False,
            )

        # Save complete observer results as one file
        all_results_content = json.dumps(observer_results, indent=2)
        self.save_and_log_states(
            all_results_content,
            "observer_results.json",
            node=self.current_node,
            add_uuid=False,
        )

    def _save_timing_log(self):
        """Save a cumulative timing log to the output folder."""
        import json

        total_elapsed = time.time() - self.search_start_time
        nodes = []
        for node in self._get_all_nodes():
            if node.id < 0:
                continue
            nodes.append(
                {
                    "node_id": node.id,
                    "stage": node.stage,
                    "execution_time_seconds": round(node.execution_time, 2),
                    "is_successful": node.is_successful,
                    "validation_score": node.validation_score,
                }
            )

        timing = {
            "total_elapsed_seconds": round(total_elapsed, 2),
            "total_elapsed_hours": round(total_elapsed / 3600, 4),
            "num_nodes": len(nodes),
            "nodes": nodes,
        }

        timing_path = os.path.join(self.output_folder, "timing_log.json")
        with open(timing_path, "w") as f:
            json.dump(timing, f, indent=2)

        # Write per-node timing into the node's own folder
        node_timing = {
            "node_id": self.current_node.id,
            "execution_time_seconds": round(self.current_node.execution_time, 2),
            "execution_time_hours": round(self.current_node.execution_time / 3600, 4),
            "total_elapsed_seconds": round(total_elapsed, 2),
            "total_elapsed_hours": round(total_elapsed / 3600, 4),
        }
        node_timing_path = os.path.join(self.get_iteration_folder(self.current_node), "timing.json")
        with open(node_timing_path, "w") as f:
            json.dump(node_timing, f, indent=2)

        logger.info(
            f"Timing: node {self.current_node.id} took {self.current_node.execution_time:.1f}s, "
            f"total elapsed {total_elapsed:.1f}s"
        )

    def log_agent_start(self, message: str):
        """Log agent start message."""
        logger.info(message)

    def log_agent_end(self, message: str):
        """Log agent end message."""
        logger.info(message)

    def select_node(self) -> Node:
        """
        Select a node for expansion using the ExTS policy.

        Descends from the root. At each internal node a virtual child (ExTS
        Thompson-style prior with fair-share visits) competes with the best existing
        child. Expansion happens when the virtual child wins AND the node clears the soft
        expansion gate AND there is structural room; otherwise the search descends into
        the best child. Deepening thus happens when children improve, expansion when a
        fresh sibling looks more promising than the incumbents.

        Returns a node to expand, or None when the reachable tree is exhausted (all
        children terminal and no room to expand — the caller ends the search).
        """
        # Ensure score range is cached for this step (used by _virtual_child_uct too)
        if not hasattr(self, "_cached_score_range") or self._cached_score_range_step != self.time_step:
            self._cached_score_range = self._compute_score_range()
            self._cached_score_range_step = self.time_step

        node = self.root_node

        while node is not None:
            # Leaf nodes: always expand
            if node.is_leaf:
                return node

            # Structural room for a new child (hard cap; terminal-safe)
            can_expand = self._can_add_child(node)

            non_terminal_children = [child for child in node.children if not child.is_terminal]

            if not non_terminal_children and not can_expand:
                # All children terminal and no room for new ones
                if node.is_terminal:
                    return None
                # Mark terminal and return None
                self.mark_node_terminal(node)
                return None

            if not non_terminal_children:
                # All existing children are terminal, but we can still expand
                return node

            # Best existing child by ExTS score, with random tie-break so equally-scored
            # (e.g. unvisited, score=inf) siblings are not starved by insertion order.
            best_child = max(non_terminal_children, key=lambda c: (self.compute_uct_value(c), self.rng.random()))
            best_child_uct = self.compute_uct_value(best_child)

            if can_expand and self._passes_expansion_gate(node):
                virtual_uct = self._virtual_child_uct(node)
                logger.detail(
                    f"Node {node.id}: virtual_child UCT={virtual_uct:.3f} vs "
                    f"best_child (Node {best_child.id}) UCT={best_child_uct:.3f}"
                )
                if virtual_uct > best_child_uct:
                    # Virtual child wins → expand at this node
                    return node

            # Descend into the best existing child
            node = best_child

        return None

    def _score_quantile(self, quantile: float) -> Optional[float]:
        """Quantile over the framework-normalized validation scores of all scored nodes.

        Used by ExTS gated expansion and progressive widening. Returns None when no node
        has a validation score yet.
        """
        norm = self._score_framework.normalize_score
        scores = sorted(norm(n.validation_score) for n in self._get_all_nodes() if n.validation_score is not None)
        if not scores:
            return None
        idx = min(int(len(scores) * quantile), len(scores) - 1)
        return scores[idx]

    def _max_children_for_node(self, node: Node) -> int:
        """Hard child cap for a node, including ExTS progressive widening.

        root:  M_0 = initial_root_children; widens unconditionally past the visit
               threshold (ExTS-faithful — the root is the trunk).
        debug: max_debug_children; never widens (bounded fix-retry loop).
        evolve: M_0 = max_evolve_children; widens only for high-scoring nodes
               (score >= widening_score_quantile).
        """
        if node.stage == "root":
            base = self.config.initial_root_children
        elif node.stage == "debug":
            return self.config.max_debug_children  # never widens
        else:  # evolve
            base = self.config.max_evolve_children

        if not self.use_progressive_widening or node.visits < self.widening_visit_threshold:
            return base

        bonus = int(math.log(node.visits / self.widening_visit_threshold) / math.log(self.widening_base))

        if node.stage == "root":
            return base + bonus

        # evolve: gate widening on the score quantile, but LATCH the unlock so the cap is
        # monotonic in visits (a later score dip cannot shrink it — see Node.widening_unlocked).
        if not node.widening_unlocked:
            q_gamma = self._score_quantile(self.widening_score_quantile)
            if (
                q_gamma is not None
                and node.validation_score is not None
                and self._score_framework.normalize_score(node.validation_score) >= q_gamma
            ):
                node.widening_unlocked = True
        return base + bonus if node.widening_unlocked else base

    def _can_add_child(self, node: Node) -> bool:
        """
        Check if a node can accept a new child (hard structural cap only).

        This is the terminal-safety gate — it feeds `mark_node_terminal` / ancestor
        termination, so it must NOT depend on the soft ExTS quantile gate (that lives in
        `_passes_expansion_gate`); otherwise a temporarily-gated node would be marked
        permanently terminal.

        Args:
            node: The node to check

        Returns:
            True if a new child can be added, False otherwise
        """
        if node.is_terminal:
            return False

        # Root node
        if node.stage == "root":
            return node.num_children < self._max_children_for_node(node)

        # For debug nodes, stop expanding after getting a successful node
        if node.stage == "debug":
            if node.is_debug_successful:
                return False
            return node.num_children < self._max_children_for_node(node)

        # For evolve nodes — count only non-forced children against the limit
        if node.stage == "evolve":
            normal_children = sum(1 for c in node.children if not c.forced_from_scratch)
            return normal_children < self._max_children_for_node(node)

        return False

    def _passes_expansion_gate(self, node: Node) -> bool:
        """Soft, selection-only ExTS expansion gate (never marks terminal).

        A new child competes with existing children only if the node's score clears the
        expansion-gate quantile. Root, debug, and failed/unscored nodes are exempt (their
        expansion is bounded structurally by `_can_add_child` + terminal marking; blocking
        them here would freeze the debug fix-retry loop). Net effect: only a "succeeded but
        produced no usable score" evolve node is blocked from widening.
        """
        if not self.use_gated_expansion:
            return True
        if node is self.root_node or node.stage == "debug":
            return True
        if not node.is_successful:
            return True
        if node.validation_score is None:
            return False  # succeeded but no score -> dead branch, block widening
        q_alpha = self._score_quantile(self.expansion_gate_quantile)
        if q_alpha is None:
            return True
        return self._score_framework.normalize_score(node.validation_score) >= q_alpha

    def _virtual_child_uct(self, node: Node) -> float:
        """ExTS virtual-child score: what a fresh child of `node` would look like.

        Models a young subtree with `n_fair` visits (the average visit count among
        existing children) and a Thompson-style prior sampled from the node's reward pool
        (its observations plus its own validation score). Compared against the best real
        child to decide expand-vs-descend.
        """
        parent_visits = max(1, node.visits)
        m = len(node.children)
        children_visits = sum(c.visits for c in node.children)
        n_fair = max(1, children_visits / m) if m > 0 else 1

        score_min, score_max = self._cached_score_range
        norm = self._score_framework.normalize_score

        # Grace-folded success rate (matches _node_score / Node.uct_value); no-op at the
        # default failure_offset=0.
        non_success = node.failure_visits + node.unvalidated_visits
        success_rate = (
            _success_rate(node.validated_visits, node.visits, non_success, self.failure_offset)
            if node.visits > 0
            else 1.0
        )

        q_prior = 0.0
        if node.reward_history and score_max > score_min:
            k = max(1, round(n_fair * success_rate))
            # Pool: this node's observations plus its own tracked quality (validation_score),
            # all in normalized space. Sampling with replacement preserves stochasticity.
            pool = [norm(r) for r in node.reward_history]
            if node.validation_score is not None:
                pool.append(norm(node.validation_score))
            samples = self.rng.choices(pool, k=k)
            sampled_mean = _shaped_mean(samples, score_min, score_max, self.score_temperature)
            q_prior = (success_rate**self.success_rate_alpha) * sampled_mean
        elif node.validated_visits > 0 and node.visits > 0:
            # Degenerate range (all scores equal): mirror the real-child formula so the
            # virtual child does not automatically lose.
            q_prior = success_rate**self.success_rate_alpha

        return q_prior + _exploration_term(
            self.exploration_type, self.exploration_constant, parent_visits, n_fair, self.puct_parent_exp
        )

    def expand(self) -> Node:
        """
        Expand the current node by creating a child node.

        Returns:
            The newly created child node
        """
        if self.current_node.stage == "root":
            return self._create_evolve_node()
        elif self.current_node.is_successful:
            return self._create_evolve_node()
        else:
            return self._create_debug_node()

    def _create_debug_node(
        self,
    ) -> Node:
        """
        Create a debug node to fix issues in a failed node.

        Returns:
            The newly created debug node
        """
        # Increment global time step for this new node
        self.time_step += 1

        # Create a new node
        self.current_node = Node(
            parent=self.current_node,
            stage="debug",
            time_step=self.time_step,
            debug_attempts=self.current_node.debug_attempts + 1,
        )

        # Check if we've exceeded the maximum debug attempts for this node
        if self.current_node.debug_attempts >= self.max_debug_depth:
            logger.warning(
                f"Node {self.current_node.id} has reached the maximum debug depth ({self.max_debug_depth}). Marking as terminal."
            )
            self.mark_node_terminal(self.current_node)

        # Generate code for the node
        self._generate_code()

    def _create_evolve_node(
        self,
    ) -> Node:
        """
        Create an evolve node to improve a successful node.

        Returns:
            The newly created evolve node
        """
        # Increment global time step for this new node
        self.time_step += 1

        self.current_node = Node(
            parent=self.current_node,
            stage="evolve",
            time_step=self.time_step,
        )

        # Generate code for the node
        self._generate_code()

    def _combine_proposer_suggestions(self, suggestions_dict):
        """
        Combine suggestions from multiple proposers into a formatted string.

        Args:
            suggestions_dict: Dictionary mapping proposer type to suggestion string

        Returns:
            str: Combined suggestions with clear attribution
        """
        if not suggestions_dict:
            return ""

        combined_parts = []

        # Map proposer types to user-friendly names
        proposer_names = {
            "text": "Text-based Analysis",
            "vision": "Visual Analysis",
            "multimodal": "Multimodal Analysis (Text + Vision)",
        }

        for proposer_type, suggestions in suggestions_dict.items():
            if suggestions and suggestions.strip():
                name = proposer_names.get(proposer_type, proposer_type.capitalize())
                combined_parts.append(f"## {name}\n\n{suggestions.strip()}")

        if not combined_parts:
            return ""

        return "\n\n" + "\n\n---\n\n".join(combined_parts)

    # --- Sibling Config Deduplication ---

    @staticmethod
    def _extract_hp_decisions(text: str, max_chars: int = 600) -> str:
        return extract_hp_decisions(text, max_chars)

    def _record_sibling_config(self, node: Node):
        """Record this node's proposed config delta for sibling deduplication."""
        if node.parent is None:
            return

        parent_id = node.parent.id
        if parent_id not in self._sibling_configs:
            self._sibling_configs[parent_id] = []

        config_delta = {
            "node_id": node.id,
            "suggestions": self._extract_hp_decisions(self._proposer_suggestions_combined),
        }
        self._sibling_configs[parent_id].append(config_delta)

    def get_sibling_configs(self, node: Node) -> List["Node"]:
        """Get previously tried config deltas from siblings of this node's parent."""
        if node.parent is None:
            return []
        return self._sibling_configs.get(node.parent.id, [])

    @property
    def sibling_config_summary(self) -> str:
        """Formatted summary of sibling configs for proposer prompt injection.

        Only surfaces deltas from siblings recorded BEFORE the current node, so the
        proposer sees what its siblings already tried and is steered to diverge.
        """
        if self.current_node is None or self.current_node.parent is None:
            return ""

        configs = self.get_sibling_configs(self.current_node)
        # Exclude the current node's own (already-recorded) delta, if present.
        configs = [c for c in configs if c.get("node_id") != self.current_node.id]
        if not configs:
            return ""

        lines = ["Previously tried modifications from this parent node:"]
        for i, cfg in enumerate(configs):
            suggestions = (cfg.get("suggestions", "") or "").strip()
            if suggestions:
                lines.append(f"  Sibling {i + 1} (node {cfg['node_id']}): {suggestions}")

        if len(lines) == 1:
            return ""

        lines.append(
            "\nPropose a configuration that is MEANINGFULLY DIFFERENT from every sibling above — "
            "change different hyperparameters and/or move them in a different direction, not the "
            "same levers with similar values."
        )
        return "\n".join(lines)

    def _resolve_checkpoint_selection(self):
        """
        Resolve checkpoint selection from the proposer's response.

        If checkpoint optimization is enabled and the parent has checkpoints,
        parses the proposer suggestions for RESUME_FROM_CHECKPOINT directive
        and sets the current node's resume fields accordingly.
        Also updates from_scratch/resume counters for exploration tracking.
        """

        ckpt_config = getattr(self.config, "checkpoint_optimization", None)
        if not ckpt_config or not getattr(ckpt_config, "enabled", False):
            self.current_node.from_scratch = True
            self.from_scratch_count += 1
            return

        # The per-node step budget applies to ALL nodes once checkpoint optimization is enabled.
        # From-scratch nodes default to the full cap; resumed nodes use the clamped additional steps.
        self.current_node.training_steps_budget = self._resolve_step_budget(ckpt_config)

        if not self.current_node.parent or not self.current_node.parent.checkpoints:
            self.current_node.from_scratch = True
            self.from_scratch_count += 1
            return

        selected_step = self._resolve_checkpoint_from_proposer()
        if selected_step is not None:
            # Find nearest checkpoint at or before the requested step
            candidates = [c for c in self.current_node.parent.checkpoints if c["step"] <= selected_step]
            if candidates:
                best = max(candidates, key=lambda c: c["step"])
                self.current_node.resume_from_checkpoint = True
                self.current_node.from_scratch = False
                self.current_node.resume_checkpoint_path = best["path"]
                self.current_node.resume_checkpoint_step = best["step"]
                self.resume_count += 1
                if best["step"] != selected_step:
                    logger.info(f"Proposer requested step {selected_step}, nearest checkpoint at step {best['step']}")
                logger.info(
                    f"Node {self.current_node.id}: will resume from checkpoint at step {best['step']} "
                    f"(path: {best['path']})"
                )
                # Clean sibling checkpoints from parent if enabled
                if getattr(ckpt_config, "clean_ckpt", False):
                    self._delete_checkpoints(self.current_node.parent, keep_steps={best["step"]})
            else:
                logger.warning(f"No checkpoint at or before step {selected_step}. Training from scratch.")
                self.current_node.from_scratch = True
                self.from_scratch_count += 1
        else:
            # Proposer chose TRAIN_FROM_SCRATCH or didn't specify
            self.current_node.from_scratch = True
            self.from_scratch_count += 1

    def _resolve_checkpoint_from_proposer(self):
        """
        Parse the combined proposer suggestions for a RESUME_FROM_CHECKPOINT directive.

        Returns:
            The selected checkpoint step (int), or None for from-scratch training.
        """
        import re as _re

        if not self._proposer_suggestions_combined:
            return None

        match = _re.search(r"RESUME_FROM_CHECKPOINT:\s*(\d+)", self._proposer_suggestions_combined)
        if match:
            return int(match.group(1))
        return None

    def _resolve_step_budget(self, ckpt_config) -> int:
        """Resolve this node's NEW-gradient-step budget, clamped to the configured max.

        The budget caps how many new gradient steps the node trains (the full run length for
        from-scratch nodes, or the additional steps beyond the resume point for resumed nodes).
        The proposer requests it via ADDITIONAL_TRAINING_STEPS; that LLM-produced number is
        clamped to checkpoint_optimization.max_steps_per_node so every node stays bounded. If
        the proposer omits the directive (e.g. from-scratch nodes), the configured max is used
        as the default budget. Returns a value in [1, max_steps_per_node].
        """
        import re as _re

        max_steps = getattr(ckpt_config, "max_steps_per_node", 100) or 100

        requested = None
        if self._proposer_suggestions_combined:
            match = _re.search(r"ADDITIONAL_TRAINING_STEPS:\s*(\d+)", self._proposer_suggestions_combined)
            if match:
                requested = int(match.group(1))

        if requested is None:
            logger.info(
                f"Node {self.current_node.id}: no ADDITIONAL_TRAINING_STEPS specified; "
                f"defaulting step budget to max_steps_per_node={max_steps}"
            )
            return max_steps

        clamped = max(1, min(requested, max_steps))
        if clamped != requested:
            logger.info(
                f"Node {self.current_node.id}: clamped step budget "
                f"{requested} -> {clamped} (max_steps_per_node={max_steps})"
            )
        return clamped

    def _delete_checkpoints(self, node: Node, keep_steps: Optional[set] = None):
        """Delete checkpoint directories from a node's checkpoint list.

        Args:
            node: The node whose checkpoints to clean.
            keep_steps: If provided, keep checkpoints at these steps and delete the rest.
                        If None, delete all checkpoints.
        """
        import shutil

        if not node.checkpoints:
            return

        to_keep = []
        to_delete = []
        for ckpt in node.checkpoints:
            if keep_steps and ckpt["step"] in keep_steps:
                to_keep.append(ckpt)
            else:
                to_delete.append(ckpt)

        for ckpt in to_delete:
            path = ckpt.get("path", "")
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                    logger.info(f"Deleted checkpoint: {path} (step {ckpt['step']}) from node {node.id}")
                else:
                    logger.debug(f"Checkpoint path already removed: {path}")
            except Exception as e:
                logger.warning(f"Failed to delete checkpoint {path}: {e}")

        node.checkpoints = to_keep
        if to_delete:
            logger.info(
                f"Cleaned {len(to_delete)} checkpoint(s) from node {node.id}, " f"kept {len(to_keep)} checkpoint(s)"
            )

    def cleanup_all_checkpoints(self):
        """Delete all checkpoints except the best node's.

        Called after the search completes to reclaim disk space.
        """
        ckpt_config = getattr(self.config, "checkpoint_optimization", None)
        if not ckpt_config or not getattr(ckpt_config, "clean_ckpt", False):
            return

        all_nodes = self._get_all_nodes()
        total_deleted = 0
        for node in all_nodes:
            if node == self._best_node or not node.checkpoints:
                continue
            count = len(node.checkpoints)
            self._delete_checkpoints(node)
            total_deleted += count

        if total_deleted > 0:
            best_kept = len(self._best_node.checkpoints) if self._best_node and self._best_node.checkpoints else 0
            logger.brief(
                f"Checkpoint cleanup: deleted {total_deleted} checkpoint(s) from non-best nodes, "
                f"kept {best_kept} checkpoint(s) on best node {self._best_node.id if self._best_node else 'N/A'}"
            )

    def _validate_checkpoint(self, ckpt: dict) -> bool:
        """Delegate checkpoint validation to the training framework."""
        from ..training_frameworks import get_framework_from_config

        framework = get_framework_from_config(self.config)
        return framework.validate_checkpoint(ckpt)

    def _generate_code(self):
        """
        Generate both entry point script and execution wrapper script.
        1. PostTrainingCoderAgent generates the environment-agnostic entry point
        2. ExecutionCoderAgent generates the environment-specific execution wrapper
        """
        # Get user input for this step if enabled
        if (
            self.enable_per_iteration_instruction
        ):  # TODO: refine the logic to store user inputs (currently they are not in nodes)
            self._get_user_input_for_step()

        self.tool_prompt = ""

        # Step 0: Get hyperparameter suggestions from all proposers (if we have training history)
        if self.current_node.parent is not None and self.current_node.parent.observer_results:
            self._proposer_suggestions = {}  # Dict mapping proposer type to suggestions
            for proposer_type, proposer in self.proposers:
                try:
                    suggestions = proposer.propose()
                    self._proposer_suggestions[proposer_type] = suggestions
                    logger.info(f"Generated {proposer_type} proposer suggestions: {len(suggestions)} characters")
                except Exception as e:
                    logger.warning(f"Failed to generate {proposer_type} proposer suggestions: {e}")
                    self._proposer_suggestions[proposer_type] = ""

            # For backward compatibility, create a combined string
            self._proposer_suggestions_combined = self._combine_proposer_suggestions(self._proposer_suggestions)

            # Record this node's proposed config delta so the next same-parent
            # sibling is told to diverge from it.
            self._record_sibling_config(self.current_node)
        else:
            self._proposer_suggestions = {}
            self._proposer_suggestions_combined = ""

        # Resolve checkpoint selection from proposer response
        self._resolve_checkpoint_selection()

        # Step 1: Generate entry point script (environment-agnostic training logic)
        self.current_node.entry_point_script = self.post_training_coder()

        # Write the entry point script to a file in the iteration folder
        entry_point_path = os.path.join(self.get_iteration_folder(self.current_node), "run.sh")
        with open(entry_point_path, "w") as file:
            file.write(self.current_node.entry_point_script)

        # Make the script executable
        Path(entry_point_path).chmod(0o755)

        # Step 2: Generate execution script (environment-specific wrapper)
        # Store the entry point path for use in the execution script prompt
        self.current_node_entry_point_path = str(entry_point_path)
        self.current_node.execution_script = self.execution_coder()

        # Write the execution script to a file in the iteration folder
        execution_script_path = os.path.join(self.get_iteration_folder(self.current_node), "execution_script.sh")
        with open(execution_script_path, "w") as file:
            file.write(self.current_node.execution_script)

        # Make the execution script executable
        Path(execution_script_path).chmod(0o755)

    def _get_user_input_for_step(self):
        """Get user input for the current step."""
        # TODO: refine the logic to store user inputs (currently they are not in nodes)
        if self.time_step == -1:
            user_input = self.initial_user_input or ""
        else:
            logger.info(f"Previous iteration info is stored in: {self.get_iteration_folder(self.current_node)}")
            user_input = self.initial_user_input or ""
            user_input += "\n" + input(
                f"Enter your inputs for current node (step {self.time_step}) (press Enter to skip): "
            )

        self.user_inputs.append(user_input)

    def simulate(self) -> tuple:
        """
        Simulate execution of current node and evaluate the result.

        Returns:
            Tuple containing: (validation_score, is_validated, is_failure, is_early_stopped)
                validation_score: The raw validation score (or None if not available)
                is_validated: True if this run has a validation score
                is_failure: True if this run failed
                is_early_stopped: True if this run was early stopped
        """
        # Use the execution script directly
        code_to_execute = self.current_node.execution_script
        code_to_analyze = self.current_node.entry_point_script

        planner_decision, error_summary, validation_score, planner_prompt, stderr, stdout, observer_results = (
            self.executer(
                code_to_execute=code_to_execute,
                code_to_analyze=code_to_analyze,
                execution_task=self.task_description,
                execution_data=self.data_prompt,
            )
        )

        # Store execution results
        self.current_node.stdout = stdout
        self.current_node.stderr = stderr
        self.current_node.observer_results = observer_results

        # Save execution outputs
        self.save_and_log_states(stderr, "stderr", node=self.current_node, add_uuid=False)
        self.save_and_log_states(stdout, "stdout", node=self.current_node, add_uuid=False)

        # Save observer results
        self._save_observer_results(observer_results)

        # Extract checkpoint data from observer results, filtering out incomplete saves
        ckpt_results = observer_results.get("checkpoint_paths", {})
        if ckpt_results.get("checkpoints"):
            all_ckpts = ckpt_results["checkpoints"]
            valid_ckpts = [c for c in all_ckpts if self._validate_checkpoint(c)]
            skipped = len(all_ckpts) - len(valid_ckpts)
            if skipped:
                logger.warning(f"Node {self.current_node.id}: filtered out {skipped} incomplete checkpoint(s)")
            self.current_node.checkpoints = valid_ckpts
            logger.info(f"Node {self.current_node.id}: recorded {len(valid_ckpts)} valid checkpoints")

        # Update validation score
        self.current_node.validation_score = validation_score

        # Track the best and worst validation scores for scaling in UCT calculation
        # Check if training was early stopped (don't update best run for early stopped runs)
        was_early_stopped = observer_results.get("checkpoint_early_stop", {}).get("early_stopped", False)
        early_stop_reason = observer_results.get("checkpoint_early_stop", {}).get("stop_reason", "")

        if validation_score is not None and not was_early_stopped:
            # Update best validation score (only for completed runs, not early stopped)
            if self._best_node is None or self._score_framework.better(validation_score, self._best_validation_score):
                self._best_node = self.current_node
                self._best_validation_score = validation_score
                self.best_step = self.time_step

                # Store best run metrics for early stopping comparison
                if observer_results and "step_metrics" in observer_results:
                    self._best_run_metrics_history = observer_results["step_metrics"].get("metrics_history", [])
                    self._best_run_observer_results = observer_results.copy()
                    logger.info(
                        f"Updated best run metrics: {len(self._best_run_metrics_history)} steps, "
                        f"validation_score={validation_score:.4f}"
                    )

            # Track worst validation score (initialize if not set yet)
            if self._worst_validation_score is None:
                self._worst_validation_score = validation_score
            elif self._score_framework.better(self._worst_validation_score, validation_score):
                # "worst" = least-good in normalized space
                self._worst_validation_score = validation_score
        elif was_early_stopped:
            logger.info(f"Training was early stopped, not updating best run tracking. Reason: {early_stop_reason}")

        # Determine if the execution was successful
        # Early stopped runs are treated as successful (not errors), but marked for visualization
        if planner_decision == "SUCCESS" or was_early_stopped:
            self.current_node.is_successful = True
            self.last_successful_node = self.current_node
            self.last_successful_step = self.time_step
            self.current_node.error_message = ""

            # Store early stopped status and reason for visualization and proposer access
            if was_early_stopped:
                self.current_node.early_stopped = True
                self.current_node.early_stop_reason = early_stop_reason
                logger.info(
                    f"Node {self.current_node.id} marked as successful but early stopped. Reason: {early_stop_reason}"
                )

            # If this is a debug node, find the origin of the debug chain
            if self.current_node.stage == "debug":
                # Find the original node that started this debugging chain
                debug_origin = self._find_debug_origin(self.current_node)

                # Add this successful node as a sibling to the original buggy node
                self.current_node.parent.remove_child(self.current_node)
                self.current_node.parent = debug_origin.parent
                debug_origin.parent.add_child(self.current_node)

                self.mark_node_terminal(debug_origin)

                logger.info(
                    f"Replaced debug origin node {debug_origin.id} with successful debug node {self.current_node.id}"
                )

            # Return the raw validation score (for tracking), is_validated flag, is_failure flag, and early_stopped flag
            return (validation_score, validation_score is not None, False, was_early_stopped)
        else:
            self.current_node.is_successful = False
            self.current_node.error_message = f"stdout + stderr: {stdout + stderr}\n\n" if stdout or stderr else ""
            self.current_node.error_message += f"Error summary: {error_summary}"

            # Get error analysis
            self.current_node.error_analysis = self.error_analyzer()

            self._all_error_analyses.append(self.current_node.error_analysis)

            # If this is a debug node and it failed, check parent's debug attempts
            if self.current_node.stage == "debug" and self.current_node.parent:
                self.current_node.parent.debug_attempts += 1
                logger.warning(
                    f"Debug attempt failed. Debug attempts on parent node {self.current_node.parent.id}: {self.current_node.parent.debug_attempts}/{self.max_debug_depth}"
                )

                # If parent has reached max debug attempts, mark it as terminal
                if self.current_node.parent.debug_attempts >= self.max_debug_depth:
                    logger.warning(
                        f"Parent node {self.current_node.parent.id} has reached the maximum debug depth. Marking as terminal."
                    )
                    self.mark_node_terminal(self.current_node.parent)

            # For failures, we return None score, not validated, is_failure=True, not early stopped
            return (None, False, True, False)

    def _compute_score_range(self) -> tuple:
        """Return the global raw-observation score bounds (framework-normalized space).

        ExTS shapes each observation individually against these bounds:
        - normalized = (obs - min) / (max - min)   → [0, 1]
        - shaped = (exp(normalized/T) - 1) / (exp(1/T) - 1)  → [0, 1]

        The bounds are maintained incrementally in backpropagate() over every validated
        observation (not per-node averages). When no validated observation has been seen
        yet, the bounds are still +/-inf; fall back to recomputing from the tree so callers
        (and tests) that populate nodes directly without going through backpropagate still
        get sensible bounds.

        Returns:
            (min_score, max_score) tuple, or (0.0, 0.0) if no scored nodes exist.
        """
        if self._score_min <= self._score_max:  # at least one observation recorded
            return (self._score_min, self._score_max)

        # Fallback: derive bounds from the tree (covers direct-populated nodes / tests).
        scored_nodes = [n for n in self._get_all_nodes() if n.validated_visits > 0]
        if not scored_nodes:
            return (0.0, 0.0)
        norm = self._score_framework.normalize_score
        obs = []
        for n in scored_nodes:
            if n.reward_history:
                obs.extend(norm(r) for r in n.reward_history)
            else:
                obs.append(norm(n.validated_reward / n.validated_visits))
        return (min(obs), max(obs))

    def backpropagate(self, simulation_result):
        """
        Backpropagate the reward up the tree and update terminal status.

        Args:
            simulation_result: Tuple of (validation_score, is_validated, is_failure, is_early_stopped)
        """
        # Extract simulation results
        validation_score, is_validated, is_failure, is_early_stopped = simulation_result

        # Maintain the global raw-observation bounds (framework-normalized space) for
        # per-observation phi-shaping. Every validated observation (incl. early-stopped
        # with a score) widens the bounds — ExTS-faithful, O(1).
        if is_validated and validation_score is not None:
            normalized = self._score_framework.normalize_score(validation_score)
            self._score_min = min(self._score_min, normalized)
            self._score_max = max(self._score_max, normalized)

        node = self.current_node
        while node is not None:
            node.update(validation_score, is_validated, is_failure, is_early_stopped)
            node = node.parent

    def _should_force_from_scratch(self) -> bool:
        """Check if we should force a from-scratch exploration run.

        Returns True when the from-scratch ratio is below the configured minimum
        and we have a best node to branch from.
        """
        total = self.from_scratch_count + self.resume_count
        if total == 0 or self._best_node is None:
            return False
        current_ratio = self.from_scratch_count / total
        return current_ratio < self.min_from_scratch_ratio

    def _create_from_scratch_child(self) -> None:
        """Create a from-scratch evolve child on the best node.

        This is a separate exploration slot that does NOT count against
        max_evolve_children, allowing the normal MCTS expansion to proceed
        independently.
        """
        self.time_step += 1
        self.current_node = Node(
            parent=self._best_node,
            stage="evolve",
            time_step=self.time_step,
            from_scratch=True,
            forced_from_scratch=True,
        )
        logger.brief(
            f"Forcing from-scratch exploration: creating node {self.current_node.id} "
            f"as child of best node {self._best_node.id} "
            f"(from_scratch ratio: {self.from_scratch_count}/{self.from_scratch_count + self.resume_count})"
        )
        # Generate code -- proposer won't see checkpoint info since from_scratch=True
        self._generate_code()

    def step(self):
        """
        Perform one step of the Monte Carlo Tree Search.

        Returns:
            True if a successful node was found, False otherwise
        """
        node_start_time = time.time()

        # Check if we need to force a from-scratch exploration run
        if self._should_force_from_scratch():
            self._create_from_scratch_child()
        else:
            # Selection: select a node to expand
            self.current_node = self.select_node()
            if self.current_node is None:
                return None

            # Expansion: create a new child node
            # Note: time_step is now incremented in the creation methods
            self.expand()

        # Simulation: execute the code and get results
        simulation_result = self.simulate()

        # Backpropagation: update node statistics
        self.backpropagate(simulation_result)

        # Record timing
        self.current_node.execution_time = time.time() - node_start_time
        self._save_timing_log()

        # Generate a visualization of the node tree after each iteration
        from .node_visualizer import visualize_tree_only

        visualize_tree_only(self)

        return self.current_node.is_successful

    def mark_node_terminal(self, node):
        """
        Mark a node and all its descendants as terminal.
        Then check if any ancestors should be marked terminal.

        Args:
            node: The node to mark as terminal
        """
        # Mark the node itself and all descendants as terminal
        self._mark_subtree_terminal(node)

        # Check if any ancestors should be marked terminal
        self._check_ancestors_terminal(node.parent)

    def _mark_subtree_terminal(self, node):
        """
        Recursively mark a node and all its descendants as terminal.

        Args:
            node: The node to mark as terminal
        """
        if node.is_terminal:
            return

        node.is_terminal = True
        logger.info(f"Marking node {node.id} as terminal")

        # Recursively mark all children
        for child in node.children:
            self._mark_subtree_terminal(child)

    def _check_ancestors_terminal(self, node):
        """
        Recursively check if ancestors should be marked as terminal.
        An ancestor is terminal if fully expanded and all children are terminal.

        Args:
            node: The ancestor node to check
        """
        if node is None:
            return

        if not self._can_add_child(node) and all(child.is_terminal for child in node.children):
            node.is_terminal = True
            logger.info(f"Marking ancestor node {node.id} as terminal (all children terminal)")

            # Continue checking up the tree
            self._check_ancestors_terminal(node.parent)

    def _get_all_nodes(self) -> List[Node]:
        """
        Get all nodes in the tree.

        Returns:
            List of all nodes
        """
        all_nodes = []

        def _collect_nodes(node):
            all_nodes.append(node)
            for child in node.children:
                _collect_nodes(child)

        _collect_nodes(self.root_node)
        return all_nodes

    def _save_all_observer_results_summary(self):
        """
        Save a summary of observer results from all nodes.
        """
        import json

        all_nodes = self._get_all_nodes()

        # Collect observer results from all nodes
        summary = {
            "total_nodes": len(all_nodes),
            "nodes_with_observer_results": 0,
            "node_summaries": [],
        }

        for node in all_nodes:
            if hasattr(node, "observer_results") and node.observer_results:
                summary["nodes_with_observer_results"] += 1

                node_summary = {
                    "node_id": node.id,
                    "stage": node.stage,
                    "is_successful": node.is_successful,
                    "validation_score": node.validation_score,
                    "tool_used": node.tool_used,
                }
                node_summary.update(node.observer_results)

                summary["node_summaries"].append(node_summary)

        # Save summary to output folder
        summary_content = json.dumps(summary, indent=2)
        summary_path = os.path.join(self.output_folder, "observer_results_summary.json")
        logger.info(f"Saving observer results summary to {summary_path}")

        with open(summary_path, "w") as f:
            f.write(summary_content)

        # Also save a human-readable version
        readable_summary = [
            "Observer Results Summary",
            "=" * 80,
            f"Total nodes: {summary['total_nodes']}",
            f"Nodes with observer results: {summary['nodes_with_observer_results']}",
            "",
            "Per-Node Summary:",
            "-" * 80,
        ]

        for node_sum in summary["node_summaries"]:
            readable_summary.append(f"\nNode {node_sum['node_id']} ({node_sum['stage']}):")
            readable_summary.append(f"  Success: {node_sum['is_successful']}")
            readable_summary.append(f"  Validation Score: {node_sum.get('validation_score', 'N/A')}")
            readable_summary.append(f"  Tool: {node_sum.get('tool_used', 'N/A')}")

            if "step_metrics" in node_sum:
                sm = node_sum["step_metrics"]
                readable_summary.append(
                    f"  Step Metrics: {sm['num_entries']} step entries, "
                    f"{len(sm['metrics'])} metrics at latest step {sm['latest_step']}"
                )

        readable_path = os.path.join(self.output_folder, "observer_results_summary.txt")
        with open(readable_path, "w") as f:
            f.write("\n".join(readable_summary))

    def create_best_run_copy(self):
        """Create a 'best_run' folder that copies the best node folder."""
        # Save observer results summary before creating best_run copy
        self._save_all_observer_results_summary()

        # Determine which node to copy
        target_node = None
        copy_reason = ""

        if self._best_node:
            target_node = self._best_node
            copy_reason = f"best validation score ({self._best_validation_score:.4f})"
        elif self.last_successful_node:
            target_node = self.last_successful_node
            copy_reason = "last successful execution"
        else:
            logger.warning("No best node or successful node found. Cannot create best_run copy.")
            return

        # Create paths
        source_folder = self.get_iteration_folder(target_node)
        best_run_folder = os.path.join(self.output_folder, "best_run")

        # Verify source folder exists
        if not os.path.exists(source_folder):
            logger.warning(f"Source folder does not exist: {source_folder}")
            return

        # Check if source folder has an 'output' subdirectory
        source_output_folder = os.path.join(source_folder, "output")
        if not os.path.exists(source_output_folder):
            logger.warning(f"Source output folder does not exist: {source_output_folder}")
            return

        # Remove existing best_run folder if it exists
        if os.path.exists(best_run_folder):
            import shutil

            try:
                shutil.rmtree(best_run_folder)
                logger.info("Removed existing best_run folder")
            except Exception as e:
                logger.error(f"Failed to remove existing best_run folder: {e}")
                return

        try:
            # Copy all files from source_output_folder to self.output_folder
            import shutil

            for item in os.listdir(source_output_folder):
                source_item = os.path.join(source_output_folder, item)
                dest_item = os.path.join(self.output_folder, item)

                if os.path.isfile(source_item):
                    shutil.copy2(source_item, dest_item)
                elif os.path.isdir(source_item):
                    shutil.copytree(source_item, dest_item, dirs_exist_ok=True)

            # Copy the entire source folder to best_run folder
            shutil.copytree(source_folder, best_run_folder, dirs_exist_ok=True)

            logger.info(f"Created best_run folder (copied from node {target_node.id} - {copy_reason})")

            # Save summary information in the best_run folder
            summary_content = [
                "Best Run Summary",
                "================",
                f"Copied from: node_{target_node.id}",
                f"Reason: {copy_reason}",
                f"Copy created at: {os.path.basename(best_run_folder)}",
                "",
                self.get_validation_score_summary(),
            ]

            # Save summary in both the main output folder and the best_run folder
            summary_text = "\n".join(summary_content)

            self.save_and_log_states(
                content=summary_text, save_name="best_run_summary.txt", node=target_node, add_uuid=False
            )

        except Exception as e:
            logger.error(f"Failed to copy folder: {e}")

    def get_validation_score_summary(self) -> str:
        """
        Get a summary of all validation scores.

        Returns:
            A summary string
        """
        all_nodes = self._get_all_nodes()
        nodes_with_scores = [node for node in all_nodes if node.validation_score is not None]

        if not nodes_with_scores:
            return "No validation scores available."

        summary = ["Validation Score Summary:"]
        for node in nodes_with_scores:
            marker = " (BEST)" if node == self._best_node else ""
            summary.append(f"Node {node.id}: {node.validation_score}{marker}")

        if self._best_node:
            summary.append(f"\nBest score: {self._best_validation_score:.4f} from node {self._best_node.id}")

        return "\n".join(summary)

    def cleanup(self):
        """Clean up resources."""
        pass

    def _find_debug_origin(self, node: Node) -> Optional[Node]:
        """
        Find the original node that started this debugging chain.

        Args:
            node: The current node in the debug chain

        Returns:
            The original node that started the debug chain
        """
        # Go up the tree until we find a non-debug node
        current = node
        while current.parent and current.parent.stage == "debug":
            current = current.parent

        debug_origin = current.parent
        assert not debug_origin.is_successful

        return debug_origin

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()

    def visualize_results(self, output_path: Optional[str] = None) -> str:
        """
        Generate a PDF visualization of the node structure.

        Args:
            output_path: Path to save the PDF. If not provided, it will be saved to
                        the output folder.

        Returns:
            The path to the generated PDF file
        """
        from .node_visualizer import visualize_results

        return visualize_results(self, output_path)

    def report_token_usage(self):
        token_usage_path = os.path.join(self.output_folder, "token_usage.json")
        usage = ChatLLMFactory.get_total_token_usage(save_path=token_usage_path)
        total = usage["total"]
        logger.brief(
            f"Total tokens — input: {total['total_input_tokens']}, "
            f"output: {total['total_output_tokens']}, "
            f"sum: {total['total_tokens']}"
        )

        logger.info(f"Full token usage detail:\n{usage}")

    def _normalized_history(self, node) -> List[float]:
        """Node's raw observations routed through the framework's normalize_score.

        Places every observation in the same higher-is-better space as the score bounds.
        """
        norm = self._score_framework.normalize_score
        return [norm(r) for r in node.reward_history]

    def compute_uct_value(self, node):
        # Cache score range per step to avoid recomputing for every node
        if not hasattr(self, "_cached_score_range") or self._cached_score_range_step != self.time_step:
            self._cached_score_range = self._compute_score_range()
            self._cached_score_range_step = self.time_step
        score_min, score_max = self._cached_score_range
        return node.uct_value(
            self.exploration_constant,
            failure_offset=self.failure_offset,
            score_min=score_min,
            score_max=score_max,
            score_temperature=self.score_temperature,
            success_rate_alpha=self.success_rate_alpha,
            exploration_type=self.exploration_type,
            puct_parent_exp=self.puct_parent_exp,
            normalized_history=self._normalized_history(node),
        )

    # Properties to maintain compatibility with Manager API
    @property
    def user_input(self) -> str:
        """Get the user input for the current step."""
        if self.time_step < 0 or self.time_step >= len(self.user_inputs):
            return self.initial_user_input
        return "\n".join([self.initial_user_input] + [self.user_inputs])

    @property
    def best_validation_score(self) -> float:
        """Get the best validation score."""
        return self._best_validation_score if self._best_validation_score is not None else 0.0

    @property
    def best_node(self) -> Node:
        """Get the best node."""
        return self._best_node

    def get_early_stop_decision(self, current_metrics_history: List, elapsed_time: float) -> tuple:
        """
        Get combined decision from all early stoppers.

        Args:
            current_metrics_history: Metrics history from current run up to checkpoint
            elapsed_time: Time elapsed since training started (in seconds)

        Returns:
            Tuple of (should_stop: bool, reason: str)
        """
        if not self.early_stoppers:
            return False, "No early stoppers configured"

        for stopper_type, stopper in self.early_stoppers:
            try:
                should_stop, reason = stopper.should_stop(
                    current_metrics_history=current_metrics_history,
                    best_run_metrics_history=self._best_run_metrics_history,
                    best_run_validation_score=self._best_validation_score,
                    elapsed_time=elapsed_time,
                )
                if should_stop:
                    logger.info(f"Early stopper [{stopper_type}] recommends STOP: {reason[:100]}...")
                    return True, f"[{stopper_type}] {reason}"
                else:
                    logger.info(f"Early stopper [{stopper_type}] recommends CONTINUE")
            except Exception as e:
                logger.error(f"Error in early stopper [{stopper_type}]: {e}", exc_info=True)
                # Continue to next stopper on error

        return False, "All early stoppers recommend continuing"

    @property
    def entry_point_script(self) -> str:
        """Get the entry point script from the current node."""
        return self.current_node.entry_point_script if self.current_node else ""

    @property
    def previous_entry_point_script(self) -> str:
        """Get the entry point script from the previous node."""
        if self.current_node and self.current_node.parent:
            return self.current_node.parent.entry_point_script
        return ""

    @property
    def execution_script(self) -> str:
        """Get the execution script from the current node."""
        return self.current_node.execution_script if self.current_node else ""

    @property
    def proposer_suggestions(self) -> str:
        """Get the combined proposer suggestions for hyperparameter changes from all proposers."""
        return getattr(self, "_proposer_suggestions_combined", "")

    @property
    def previous_execution_script(self) -> str:
        """Get the execution script from the previous node."""
        if self.current_node and self.current_node.parent:
            return self.current_node.parent.execution_script
        return ""

    @property
    def entry_point_script_path(self) -> str:
        """Get the entry point script path for the current node."""
        if hasattr(self, "current_node_entry_point_path"):
            return self.current_node_entry_point_path
        return ""

    @property
    def error_message(self) -> str:
        """Get the error message from the current node."""
        return self.current_node.error_message if self.current_node else ""

    @property
    def previous_error_message(self) -> str:
        """Get the error message from the previous node."""
        if self.current_node and self.current_node.parent:
            return self.current_node.parent.error_message
        return ""

    @property
    def error_analysis(self) -> str:
        """Get the error analysis from the current node."""
        return self.current_node.error_analysis if self.current_node else ""

    @property
    def all_previous_error_analyses(self) -> str:
        """Get all error analyses from previous nodes."""
        return "\n\n".join(self._all_error_analyses)

    @property
    def previous_error_analysis(self) -> str:
        """Get error analysis from the immediate parent node only if it failed.

        Returns empty string if:
        - No parent node exists
        - Parent node succeeded (is_successful=True)
        - Parent node has no error analysis
        """
        parent_node = self.current_node.parent if self.current_node and self.current_node.parent else None

        if not parent_node:
            return ""

        # Only return error analysis if the parent failed
        if parent_node.is_successful:
            return ""

        return parent_node.error_analysis or ""

    @property
    def other_previous_error_analyses(self) -> str:
        """Get all error analyses except from the immediate parent node.

        This is useful when you want to show historical errors but treat the
        most recent error separately based on whether it succeeded or failed.
        """
        if not self._all_error_analyses:
            return ""

        # If we have fewer than 2 analyses, there are no "other" analyses
        if len(self._all_error_analyses) < 2:
            return ""

        # Return all but the last (most recent) error analysis
        return "\n\n".join(self._all_error_analyses[:-1])

    @property
    def per_iteration_output_folder(self) -> str:
        """Get the output folder for the current iteration."""
        if not self.current_node:
            return os.path.join(self.output_folder, "initialization", "output")
        return self.get_per_iteration_output_folder(self.current_node)

    @property
    def iteration_folder(self) -> str:
        """Get the folder for the current iteration."""
        if not self.current_node:
            return os.path.join(self.output_folder, "initialization")
        return self.get_iteration_folder(self.current_node)

    @property
    def node_output_folder_name(self) -> str:
        """Get the output folder name for the current node based on its ID and run timestamp."""
        if not self.current_node:
            base_name = "llmzero_default"
        else:
            base_name = f"llmzero_N{self.current_node.id}_{self.run_timestamp}"

        # Append postfix if provided
        if self.run_postfix:
            return f"{base_name}{self.run_postfix}"
        return base_name

    @property
    def tutorial_prompt(self) -> str:
        return ""

    @property
    def selected_tool(self) -> str:
        return ""

    @property
    def configure_env(self):
        return self.config.configure_env

    @property
    def wandb_enabled(self) -> bool:
        """Check if WandB is enabled (i.e., API key is provided)."""
        return self.wandb_api_key is not None and self.wandb_api_key.strip() != ""

    @property
    def wandb_api_key_value(self) -> str:
        """Get the WandB API key value, or empty string if not set."""
        return self.wandb_api_key if self.wandb_api_key else ""
