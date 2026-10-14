"""
Async Node Manager implementing Top-K Weighted Deepening search strategy.

Instead of sequential MCTS (select -> expand -> simulate -> backpropagate),
this manager runs N jobs concurrently with an asyncio-based event loop:

1. Virtual root (no execution) with N initial children launched in parallel
2. On job completion, select parent from top-ceil(sqrt(N)) best nodes (weighted by score)
3. Proposer avoids duplicate/near-identical proposals via sibling config tracking

Key properties:
- Within-branch: sequential (full trajectory info before deepening)
- Cross-branch: parallel (independent exploration)
- Orchestration runs in asyncio event loop (single-threaded, no race conditions)
- Blocking execution (training jobs) offloaded to thread pool via run_in_executor
"""

import asyncio
import logging
import math
import os
import random
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..llm import ChatLLMFactory
from ..utils import extract_hp_decisions
from .node_manager import Node, _generate_node_timestamp

logger = logging.getLogger(__name__)


class AsyncNodeManager:
    """
    Manages concurrent MCTS exploration using Top-K Weighted Deepening.

    Algorithm:
    - Bootstrap: Launch N jobs as children of a virtual root
    - On completion: Select from top-K best nodes (K=ceil(sqrt(N))), weighted by score
    - Deepen selected node with new proposals avoiding sibling duplicates
    - Maintain N concurrent jobs until budget exhausted

    Uses asyncio event loop for orchestration (race-condition-free) with
    thread pool for blocking execution (training jobs that run for hours).
    """

    def __init__(
        self,
        input_data_folder: str,
        output_folder: str,
        config: Any,
        initial_user_input: str,
        enable_per_iteration_instruction: bool,
        max_concurrent: int = 4,
        selection_temperature: float = 0.5,
        wandb_api_key: Optional[str] = None,
        wandb_project_postfix: str = "",
        run_postfix: str = "",
    ):
        self.input_data_folder = input_data_folder
        self.output_folder = output_folder

        for path, name in [(input_data_folder, "input_data_folder")]:
            if not Path(path).exists():
                raise FileNotFoundError(f"{name} not found: {path}")

        Path(output_folder).mkdir(parents=True, exist_ok=True)

        self.config = config
        self.enable_per_iteration_instruction = enable_per_iteration_instruction
        self.initial_user_input = initial_user_input
        self.wandb_api_key = wandb_api_key
        self.wandb_project_postfix = wandb_project_postfix
        self.run_postfix = run_postfix

        # Training-framework handle drives score-direction-aware best/worst tracking
        # (defaults to "maximize", an identity pass-through).
        from ..training_frameworks import get_framework_from_config

        self._score_framework = get_framework_from_config(config)

        # Async-specific parameters
        self.max_concurrent = max_concurrent
        self.selection_temperature = selection_temperature

        # Generate single timestamp for entire run
        self.run_timestamp = _generate_node_timestamp()

        # Tree structure — virtual root never executes
        self.time_step = -1
        self.root_node = Node(stage="root", time_step=self.time_step, depth=0)
        self._current_node = self.root_node

        # Thread-local storage for per-execution-thread current_node
        self._thread_local = threading.local()

        # Job tracking: asyncio.Task -> node_id
        self._pending_tasks: Dict[asyncio.Task, int] = {}
        self._node_by_id: Dict[int, Node] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self._iterations_spent = 0

        # Node tracking
        self._best_node: Optional[Node] = None
        self._best_validation_score: Optional[float] = None
        self._worst_validation_score: Optional[float] = None
        self.last_successful_node: Optional[Node] = None

        # Best run metrics for early stopping
        self._best_run_metrics_history: Optional[List] = None
        self._best_run_observer_results: Optional[dict] = None

        # Key node tracking
        self.best_step = -1
        self.last_successful_step = -1

        # MCTS parameters (reuse from config)
        self.max_debug_depth = self.config.max_debug_depth
        self.failure_offset = getattr(self.config, "failure_offset", 0)

        # Checkpoint exploration tracking
        self.from_scratch_count = 0
        self.resume_count = 0

        ckpt_config = getattr(self.config, "checkpoint_optimization", None)
        self.min_from_scratch_ratio = getattr(ckpt_config, "min_from_scratch_ratio", 0.2) if ckpt_config else 0.2

        # Timing
        self.search_start_time = time.time()

        # Description file contents
        self.description_file_contents = None

        # Error analysis
        self._all_error_analyses: List[str] = []

        # Sibling config tracking for deduplication
        self._sibling_configs: Dict[int, List[Dict[str, Any]]] = {}

        # Root relaunch tracking. Async search has no debug path — a failed node can
        # never be deepened (validation_score is None, is_successful is False). Without
        # a relaunch mechanism, a failed bootstrap child permanently wastes its slot, and
        # if every bootstrap child fails the search dies with budget remaining. Each failed
        # child of the virtual root earns one credit to launch a fresh root child in a
        # diverging direction (cold-start proposer), keeping the root frontier alive.
        self._failed_root_child_count = 0
        self._root_relaunch_count = 0

        # Initialize agents
        self._init_agents()

    @property
    def current_node(self) -> Node:
        """Thread-safe current_node: execution threads get their own node, event loop gets main."""
        thread_node = getattr(self._thread_local, "node", None)
        if thread_node is not None:
            return thread_node
        return self._current_node

    @current_node.setter
    def current_node(self, value: Node):
        self._current_node = value

    def _init_agents(self):
        """Initialize all required agents."""
        from ..agents import (
            ErrorAnalyzerAgent,
            ExecutionCoderAgent,
            PostTrainingCoderAgent,
        )
        from ..early_stoppers import AgenticEarlyStopper
        from ..proposers import AgenticProposer

        self.error_analyzer = ErrorAnalyzerAgent(
            config=self.config,
            manager=self,
            llm_config=self.config.error_analyzer,
            prompt_template=None,
        )

        proposer_llm_config = getattr(self.config, "proposer_llm", self.config.bash_coder)
        proposer = AgenticProposer(
            config=self.config,
            manager=self,
            agent_llm_config=proposer_llm_config,
            agent_prompt_template=None,
        )
        self.proposers = [("unified", proposer)]

        early_stopper_config = getattr(self.config, "early_stopper", {})
        early_stopper_enabled = getattr(early_stopper_config, "enabled", False)

        self.early_stoppers = []
        if early_stopper_enabled:
            stopper = AgenticEarlyStopper(
                config=self.config,
                manager=self,
                agent_llm_config=self.config.bash_coder,
                agent_prompt_template=None,
            )
            self.early_stoppers.append(("unified", stopper))

        self.post_training_coder = PostTrainingCoderAgent(
            config=self.config,
            manager=self,
            llm_config=self.config.bash_coder,
            prompt_template=None,
        )

        self.execution_coder = ExecutionCoderAgent(
            config=self.config,
            manager=self,
            llm_config=self.config.bash_coder,
            prompt_template=None,
        )

        # NOTE: executer is created fresh per call in _create_executer()
        # to avoid thread-safety issues (no shared LLM state across concurrent jobs)
        self._executer_llm_config = self.config.executer.copy()
        self._executer_llm_config.multi_turn = False

    def initialize(self):
        """Initialize: read description files directly (no LLM agents needed)."""
        self.description_file_contents = self._read_description_files()
        self.task_description = self.initial_user_input
        self.data_prompt = self.description_file_contents

    def _read_description_files(self) -> str:
        """Read all .txt and .md files from input_data_folder directly."""
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

    # --- Async Search Loop ---

    def run_search(self, max_iterations: int) -> None:
        """
        Entry point: run the async search loop.

        Creates an asyncio event loop and runs the search coroutine.
        Safe to call from synchronous code.
        """
        asyncio.run(self._async_search(max_iterations))

    async def _async_search(self, max_iterations: int) -> None:
        """
        Main async search coroutine.

        All orchestration (tree updates, parent selection, slot filling) runs here
        in the event loop — single-threaded, no race conditions. Blocking execution
        (training jobs) is offloaded to the thread pool.
        """
        logger.brief(
            f"Starting async search: budget={max_iterations}, "
            f"max_concurrent={self.max_concurrent}, "
            f"K=ceil(sqrt({self.max_concurrent}))={math.ceil(math.sqrt(self.max_concurrent))}"
        )

        # Phase 1: Bootstrap — launch N initial children
        await self._bootstrap_initial_jobs()

        # Phase 2: Event loop — wait for completions, fill slots
        while self._iterations_spent < max_iterations:
            if not self._pending_tasks:
                logger.brief("No jobs in flight and budget remaining — all branches terminal.")
                break

            # Wait for any task to complete (with timeout)
            timeout = getattr(self.config, "per_execution_timeout", 86400) + 600
            done, _ = await asyncio.wait(
                self._pending_tasks.keys(),
                return_when=asyncio.FIRST_COMPLETED,
                timeout=timeout,
            )

            if not done:
                logger.warning(f"Timeout waiting for job completion ({timeout}s). Cancelling oldest task.")
                self._cancel_oldest_task()
                continue

            for task in done:
                node_id = self._pending_tasks.pop(task)
                node = self._node_by_id[node_id]

                try:
                    result = task.result()
                except Exception as e:
                    logger.error(f"Node {node_id} task raised exception: {e}")
                    result = self._make_error_result(node, str(e))

                self._process_completion(node, result)
                self._iterations_spent += 1

                logger.brief(
                    f"Iteration {self._iterations_spent}/{max_iterations} complete. "
                    f"In-flight: {len(self._pending_tasks)}. "
                    f"Best score: {self._best_validation_score}"
                )

                if self._iterations_spent >= max_iterations:
                    break

            # Fill slots with new jobs
            if self._iterations_spent < max_iterations:
                await self._fill_slots(max_iterations)

        # Shutdown thread pool
        self._executor.shutdown(wait=True)
        logger.brief(f"Async search complete. Iterations spent: {self._iterations_spent}")

    async def _bootstrap_initial_jobs(self):
        """Launch N initial children of the virtual root."""
        for i in range(self.max_concurrent):
            self.time_step += 1
            child = Node(
                parent=self.root_node,
                stage="evolve",
                time_step=self.time_step,
            )
            await self._submit_job(child, is_initial=(i == 0))

    async def _fill_slots(self, max_iterations: int):
        """Fill available slots with new deepening jobs."""
        while (
            len(self._pending_tasks) < self.max_concurrent
            and self._iterations_spent + len(self._pending_tasks) < max_iterations
        ):
            parent = self._select_parent_for_deepening()
            if parent is None:
                # Nothing deepenable. If failed root children left unspent relaunch
                # credits, launch a fresh root child so the search keeps making
                # progress instead of stalling with budget remaining.
                if self._root_relaunch_count < self._failed_root_child_count:
                    await self._relaunch_root_child()
                    continue
                break

            self.time_step += 1
            child = Node(
                parent=parent,
                stage="evolve",
                time_step=self.time_step,
            )

            await self._submit_job(child, is_initial=False)

    async def _relaunch_root_child(self):
        """Launch a fresh child of the virtual root to replace a failed bootstrap child.

        The new node is a cold-start bootstrap sibling: it hangs off the virtual root,
        so _generate_code_for_node takes the cold-start path and the proposer diverges
        from every prior root sibling (baseline + previous relaunches), keeping the root
        frontier diverse. Counted against the failed-root-child credit pool.
        """
        self._root_relaunch_count += 1
        self.time_step += 1
        child = Node(
            parent=self.root_node,
            stage="evolve",
            time_step=self.time_step,
        )
        logger.brief(
            f"Relaunching a fresh root child (node {child.id}) after root-child failure "
            f"[{self._root_relaunch_count}/{self._failed_root_child_count} credits used]"
        )
        await self._submit_job(child, is_initial=False)

    def _select_parent_for_deepening(self) -> Optional[Node]:
        """
        Select a parent node using epsilon-greedy strategy.

        With probability 0.8: softmax-weighted from top-K nodes (exploitation).
        With probability 0.2: uniform random from remaining nodes (exploration).
        K = ceil(sqrt(N)) where N = max_concurrent.
        """
        completed_nodes = [n for n in self._get_all_nodes() if n.validation_score is not None and not n.is_terminal]

        if not completed_nodes:
            successful = [n for n in self._get_all_nodes() if n.is_successful and not n.is_terminal]
            if successful:
                return random.choice(successful)
            return None

        if len(completed_nodes) == 1:
            return completed_nodes[0]

        K = math.ceil(math.sqrt(self.max_concurrent))
        # Rank by framework-normalized score so "best" respects score direction
        # (identity for the default "maximize"). top_k = the K best-normalized nodes.
        sorted_nodes = sorted(completed_nodes, key=lambda n: self._score_framework.normalize_score(n.validation_score))
        top_k = sorted_nodes[-K:]
        rest = sorted_nodes[:-K]

        epsilon = 0.8

        if random.random() < epsilon:
            scores = [self._score_framework.normalize_score(n.validation_score) for n in top_k]
            weights = self._softmax(scores, temperature=self.selection_temperature)
            selected = random.choices(top_k, weights=weights, k=1)[0]
            logger.info(
                f"Epsilon-greedy EXPLOIT: selected node {selected.id} "
                f"(score={selected.validation_score:.4f}) from top-{K} via softmax"
            )
        else:
            if rest:
                selected = random.choice(rest)
                logger.info(
                    f"Epsilon-greedy EXPLORE: selected node {selected.id} "
                    f"(score={selected.validation_score:.4f}) uniformly from {len(rest)} non-top-K nodes"
                )
            else:
                selected = random.choice(top_k)
                logger.info(
                    f"Epsilon-greedy EXPLORE (fallback): selected node {selected.id} "
                    f"(score={selected.validation_score:.4f}) — all nodes within top-K"
                )

        return selected

    @staticmethod
    def _softmax(scores: List[float], temperature: float) -> List[float]:
        """Compute softmax weights with temperature scaling."""
        if temperature <= 0:
            weights = [0.0] * len(scores)
            weights[scores.index(max(scores))] = 1.0
            return weights

        max_score = max(scores)
        exp_scores = [math.exp((s - max_score) / temperature) for s in scores]
        total = sum(exp_scores)
        return [e / total for e in exp_scores]

    # --- Job Submission ---

    async def _submit_job(self, node: Node, is_initial: bool):
        """
        Generate code (synchronous in event loop) then submit execution to thread pool.

        Code generation happens here in the event loop — safe to set current_node.
        Execution runs in a thread — uses thread-local current_node.
        """
        # Set current_node for agent compatibility during code generation
        self.current_node = node
        self._node_by_id[node.id] = node

        # Generate code (synchronous — LLM calls are fast relative to training)
        self._generate_code_for_node(node, is_initial=is_initial)

        # Submit execution to thread pool
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self._executor, self._execute_node_in_thread, node)
        task = asyncio.ensure_future(future)
        self._pending_tasks[task] = node.id

        logger.info(f"Submitted job for node {node.id} (parent={node.parent.id if node.parent else None})")

    def _generate_code_for_node(self, node: Node, is_initial: bool = False):
        """
        Generate training and execution scripts for a node.

        Runs in the event loop — current_node is safe.
        """
        self.current_node = node

        parent = node.parent
        parent_has_history = parent is not None and bool(parent.observer_results)
        # Bootstrap children hang off the virtual root, which never executes and so
        # never has observer_results. Without a cold-start path here, every bootstrap
        # child would skip the proposer and generate the IDENTICAL default config.
        is_bootstrap = parent is self.root_node
        cold_start = is_bootstrap and not parent_has_history

        if is_initial:
            # Node 0 is the baseline: package-default config, no proposer call.
            # Record it as a sibling so later bootstrap children diverge from it.
            self._proposer_suggestions = {}
            self._proposer_suggestions_combined = ""
            self._record_baseline_sibling(node)
        elif parent_has_history or cold_start:
            # Normal deepening (parent has metrics) OR cold-start bootstrap sibling
            # (no metrics yet, but must differ from the baseline and prior siblings).
            self._proposer_suggestions = {}
            for proposer_type, proposer in self.proposers:
                try:
                    suggestions = proposer.propose(cold_start=cold_start)
                    self._proposer_suggestions[proposer_type] = suggestions
                except Exception as e:
                    logger.warning(f"Failed to generate {proposer_type} proposer suggestions: {e}")
                    self._proposer_suggestions[proposer_type] = ""

            self._proposer_suggestions_combined = self._combine_proposer_suggestions(self._proposer_suggestions)

            # Record sibling config delta so the next same-parent sibling diverges.
            self._record_sibling_config(node)
        else:
            self._proposer_suggestions = {}
            self._proposer_suggestions_combined = ""

        # Resolve checkpoint selection
        self._resolve_checkpoint_selection()

        # Generate entry point script
        node.entry_point_script = self.post_training_coder()

        entry_point_path = os.path.join(self.get_iteration_folder(node), "run.sh")
        with open(entry_point_path, "w") as f:
            f.write(node.entry_point_script)
        Path(entry_point_path).chmod(0o755)

        # Generate execution script
        self.current_node_entry_point_path = str(entry_point_path)
        node.execution_script = self.execution_coder()

        execution_script_path = os.path.join(self.get_iteration_folder(node), "execution_script.sh")
        with open(execution_script_path, "w") as f:
            f.write(node.execution_script)
        Path(execution_script_path).chmod(0o755)

    def _create_executer(self):
        """Create a fresh ExecuterAgent instance (thread-safe, no shared LLM state)."""
        from ..agents import ExecuterAgent

        return ExecuterAgent(
            config=self.config,
            manager=self,
            language="bash",
            timeout=self.config.per_execution_timeout,
            executer_llm_config=self._executer_llm_config,
            executer_prompt_template=None,
        )

    def _execute_node_in_thread(self, node: Node) -> Tuple[Node, Any]:
        """
        Execute a node's training job (runs in thread pool).

        Sets thread-local current_node so agents reading self.manager.current_node
        see the correct node for THIS execution, not another thread's.
        """
        # Set thread-local current_node for agent compatibility
        self._thread_local.node = node
        try:
            return self._execute_node(node)
        finally:
            self._thread_local.node = None

    def _execute_node(self, node: Node) -> Tuple[Node, Any]:
        """Execute a node's training job. Returns (node, simulation_data)."""
        node_start_time = time.time()

        code_to_execute = node.execution_script
        code_to_analyze = node.entry_point_script

        try:
            executer = self._create_executer()
            planner_decision, error_summary, validation_score, planner_prompt, stderr, stdout, observer_results = (
                executer(
                    code_to_execute=code_to_execute,
                    code_to_analyze=code_to_analyze,
                    execution_task=self.task_description,
                    execution_data=self.data_prompt,
                )
            )
        except Exception as e:
            logger.error(f"Node {node.id} execution failed with exception: {e}")
            node.execution_time = time.time() - node_start_time
            return (node, (None, False, True, False, "", str(e), {}))

        node.execution_time = time.time() - node_start_time

        return (
            node,
            (validation_score, planner_decision, error_summary, stderr, stdout, observer_results),
        )

    # --- Completion Processing ---

    def _process_completion(self, node: Node, result: Tuple):
        """
        Process a completed job — update tree, backpropagate, save artifacts.

        Runs in the event loop — single-threaded, no race conditions.
        """
        if result is None:
            logger.error(f"Node {node.id} returned None result")
            return

        returned_node, sim_data = result

        # Transfer execution_time from returned_node (set in thread)
        node.execution_time = returned_node.execution_time

        # Unpack simulation data
        if len(sim_data) == 7:
            validation_score, is_validated, is_failure, is_early_stopped, stdout, stderr, observer_results = sim_data
            planner_decision = "FAILURE"
            error_summary = stderr
        else:
            validation_score, planner_decision, error_summary, stderr, stdout, observer_results = sim_data
            is_failure = planner_decision != "SUCCESS"
            is_early_stopped = observer_results.get("checkpoint_early_stop", {}).get("early_stopped", False)
            is_validated = validation_score is not None

        # Set current_node for save operations (safe — in event loop)
        self.current_node = node

        # Store execution results on node
        node.stdout = stdout
        node.stderr = stderr
        node.observer_results = observer_results
        node.validation_score = validation_score

        # Save artifacts
        self.save_and_log_states(stderr, "stderr", node=node, add_uuid=False)
        self.save_and_log_states(stdout, "stdout", node=node, add_uuid=False)
        self._save_observer_results(observer_results)

        # Extract checkpoints
        ckpt_results = observer_results.get("checkpoint_paths", {})
        if ckpt_results.get("checkpoints"):
            all_ckpts = ckpt_results["checkpoints"]
            valid_ckpts = [c for c in all_ckpts if self._validate_checkpoint(c)]
            node.checkpoints = valid_ckpts

        # Update best/worst tracking (single-threaded here — no races)
        was_early_stopped = is_early_stopped
        early_stop_reason = observer_results.get("checkpoint_early_stop", {}).get("stop_reason", "")

        if validation_score is not None and not was_early_stopped:
            if self._best_node is None or self._score_framework.better(validation_score, self._best_validation_score):
                self._best_node = node
                self._best_validation_score = validation_score
                self.best_step = node.time_step

                if observer_results and "step_metrics" in observer_results:
                    self._best_run_metrics_history = observer_results["step_metrics"].get("metrics_history", [])
                    self._best_run_observer_results = observer_results.copy()

            if self._worst_validation_score is None:
                self._worst_validation_score = validation_score
            elif self._score_framework.better(self._worst_validation_score, validation_score):
                # "worst" = least-good in normalized space
                self._worst_validation_score = validation_score

        # Update node success state
        if planner_decision == "SUCCESS" or was_early_stopped:
            node.is_successful = True
            self.last_successful_node = node
            self.last_successful_step = node.time_step
            node.error_message = ""

            if was_early_stopped:
                node.early_stopped = True
                node.early_stop_reason = early_stop_reason
        else:
            node.is_successful = False
            node.error_message = f"stdout + stderr: {stdout + stderr}\n\n" if stdout or stderr else ""
            node.error_message += f"Error summary: {error_summary}"

            # Error analysis
            node.error_analysis = self.error_analyzer()
            self._all_error_analyses.append(node.error_analysis)

            # A failed direct child of the virtual root cannot be deepened (no score),
            # so grant a credit to relaunch a fresh root child (see _fill_slots).
            if node.parent is self.root_node:
                self._failed_root_child_count += 1

            # Mark terminal after too many failures in branch
            if node.depth >= self.max_debug_depth:
                self.mark_node_terminal(node)

        # Backpropagate (single-threaded — safe)
        self._backpropagate(node, validation_score, is_validated, is_failure, is_early_stopped)

        # Save timing
        self._save_timing_log()

        # Visualize
        from .node_visualizer import visualize_tree_only

        visualize_tree_only(self)

    def _make_error_result(self, node: Node, error_msg: str) -> Tuple:
        """Create an error result tuple when a task raises an unhandled exception."""
        node.execution_time = time.time() - node.ctime
        return (node, (None, False, True, False, "", error_msg, {}))

    def _cancel_oldest_task(self):
        """Cancel the oldest pending task on timeout."""
        if not self._pending_tasks:
            return

        oldest_task = next(iter(self._pending_tasks))
        node_id = self._pending_tasks.pop(oldest_task)
        oldest_task.cancel()
        logger.warning(f"Cancelled timed-out task for node {node_id}")

    def _backpropagate(
        self,
        node: Node,
        validation_score: Optional[float],
        is_validated: bool,
        is_failure: bool,
        is_early_stopped: bool,
    ):
        """Backpropagate results up the tree."""
        current = node
        while current is not None:
            current.update(validation_score, is_validated, is_failure, is_early_stopped)
            current = current.parent

    # --- Sibling Config Deduplication ---

    def _record_baseline_sibling(self, node: Node):
        """Record the baseline (node 0) so later bootstrap siblings diverge from it.

        Node 0 uses the package-default config with no proposer call, so it has no
        proposer delta. We still register it under the parent's sibling list with an
        explicit baseline marker so the cold-start proposer for siblings 1..N-1 is
        told 'a baseline-default run already exists — propose something different'.
        """
        if node.parent is None:
            return
        parent_id = node.parent.id
        self._sibling_configs.setdefault(parent_id, []).append(
            {
                "node_id": node.id,
                "suggestions": "Baseline run using the package-default hyperparameter configuration (no modifications).",
            }
        )

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

    def get_sibling_configs(self, node: Node) -> List[Dict[str, Any]]:
        """Get previously tried config deltas from siblings of this node's parent."""
        if node.parent is None:
            return []
        return self._sibling_configs.get(node.parent.id, [])

    @property
    def sibling_config_summary(self) -> str:
        """Formatted summary of sibling configs for proposer prompt injection."""
        if self.current_node is None or self.current_node.parent is None:
            return ""

        configs = self.get_sibling_configs(self.current_node)
        if not configs:
            return ""

        lines = ["Previously tried modifications from this parent node:"]
        for i, cfg in enumerate(configs):
            # Already distilled to concrete decisions in _record_sibling_config; show
            # them in full so the sibling can pick a genuinely different region.
            suggestions = (cfg.get("suggestions", "") or "").strip()
            if suggestions:
                lines.append(f"  Sibling {i + 1} (node {cfg['node_id']}): {suggestions}")

        lines.append(
            "\nPropose a configuration that is MEANINGFULLY DIFFERENT from every sibling above — "
            "change different hyperparameters and/or move them in a different direction, not the "
            "same levers with similar values."
        )
        return "\n".join(lines)

    # --- Code Generation Helpers ---

    def _combine_proposer_suggestions(self, suggestions_dict):
        """Combine suggestions from multiple proposers."""
        if not suggestions_dict:
            return ""

        proposer_names = {
            "text": "Text-based Analysis",
            "vision": "Visual Analysis",
            "multimodal": "Multimodal Analysis (Text + Vision)",
        }

        combined_parts = []
        for proposer_type, suggestions in suggestions_dict.items():
            if suggestions and suggestions.strip():
                name = proposer_names.get(proposer_type, proposer_type.capitalize())
                combined_parts.append(f"## {name}\n\n{suggestions.strip()}")

        if not combined_parts:
            return ""

        return "\n\n" + "\n\n---\n\n".join(combined_parts)

    def _resolve_checkpoint_selection(self):
        """Resolve checkpoint selection from proposer response."""
        ckpt_config = getattr(self.config, "checkpoint_optimization", None)
        if not ckpt_config or not getattr(ckpt_config, "enabled", False):
            self.current_node.from_scratch = True
            self.from_scratch_count += 1
            return

        if not self.current_node.parent or not self.current_node.parent.checkpoints:
            self.current_node.from_scratch = True
            self.from_scratch_count += 1
            return

        selected_step = self._resolve_checkpoint_from_proposer()
        if selected_step is not None:
            candidates = [c for c in self.current_node.parent.checkpoints if c["step"] <= selected_step]
            if candidates:
                best = max(candidates, key=lambda c: c["step"])
                self.current_node.resume_from_checkpoint = True
                self.current_node.from_scratch = False
                self.current_node.resume_checkpoint_path = best["path"]
                self.current_node.resume_checkpoint_step = best["step"]
                self.resume_count += 1
            else:
                self.current_node.from_scratch = True
                self.from_scratch_count += 1
        else:
            self.current_node.from_scratch = True
            self.from_scratch_count += 1

    def _resolve_checkpoint_from_proposer(self):
        """Parse proposer suggestions for RESUME_FROM_CHECKPOINT directive."""
        import re

        if not self._proposer_suggestions_combined:
            return None

        match = re.search(r"RESUME_FROM_CHECKPOINT:\s*(\d+)", self._proposer_suggestions_combined)
        if match:
            return int(match.group(1))
        return None

    def _validate_checkpoint(self, ckpt: dict) -> bool:
        """Delegate checkpoint validation to the training framework."""
        from ..training_frameworks import get_framework_from_config

        framework = get_framework_from_config(self.config)
        return framework.validate_checkpoint(ckpt)

    # --- File/State Management ---

    def get_iteration_folder(self, node: Node) -> str:
        if node.id < 0:
            iter_folder = os.path.join(self.output_folder, "node_init")
        else:
            iter_folder = os.path.join(self.output_folder, f"node_{node.id}")
        os.makedirs(iter_folder, exist_ok=True)
        return iter_folder

    def get_per_iteration_output_folder(self, node: Node) -> str:
        iter_output_folder = os.path.join(self.get_iteration_folder(node), "output")
        os.makedirs(iter_output_folder, exist_ok=True)
        return iter_output_folder

    def save_and_log_states(self, content, save_name, per_iteration=False, add_uuid=False, node=None):
        if add_uuid:
            name, ext = os.path.splitext(save_name)
            uuid_suffix = str(uuid.uuid4()).replace("-", "")[:4]
            save_name = f"{name}_{uuid_suffix}{ext}"

        if per_iteration and self.current_node:
            states_dir = os.path.join(self.get_iteration_folder(self.current_node), "states")
        elif node:
            states_dir = os.path.join(self.get_iteration_folder(node), "states")
        else:
            states_dir = os.path.join(self.output_folder, "states")

        os.makedirs(states_dir, exist_ok=True)
        output_file = os.path.join(states_dir, save_name)

        logger.info(f"Saving {output_file}...")
        with open(output_file, "w") as f:
            if content is not None:
                if isinstance(content, list):
                    f.write("\n".join(str(item) for item in content))
                else:
                    f.write(content)
            else:
                f.write("<None>")

    def _save_observer_results(self, observer_results: dict):
        """Save observer results to files."""
        import json

        if not observer_results:
            return

        if "step_metrics" in observer_results:
            self.save_and_log_states(
                json.dumps(observer_results["step_metrics"], indent=2),
                "step_metrics.json",
                node=self.current_node,
                add_uuid=False,
            )

        if "validation" in observer_results:
            self.save_and_log_states(
                json.dumps(observer_results["validation"], indent=2),
                "validation_metrics.json",
                node=self.current_node,
                add_uuid=False,
            )

        if "checkpoint_paths" in observer_results:
            self.save_and_log_states(
                json.dumps(observer_results["checkpoint_paths"], indent=2),
                "checkpoint_paths.json",
                node=self.current_node,
                add_uuid=False,
            )

        self.save_and_log_states(
            json.dumps(observer_results, indent=2),
            "observer_results.json",
            node=self.current_node,
            add_uuid=False,
        )

    def _save_timing_log(self):
        """Save timing log."""
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

    # --- Terminal/Tree Management ---

    def mark_node_terminal(self, node: Node):
        """Mark a node and descendants as terminal."""
        self._mark_subtree_terminal(node)
        self._check_ancestors_terminal(node.parent)

    def _mark_subtree_terminal(self, node: Node):
        if node.is_terminal:
            return
        node.is_terminal = True
        for child in node.children:
            self._mark_subtree_terminal(child)

    def _check_ancestors_terminal(self, node: Optional[Node]):
        if node is None:
            return
        if all(child.is_terminal for child in node.children) and len(node.children) > 0:
            node.is_terminal = True
            self._check_ancestors_terminal(node.parent)

    def _get_all_nodes(self) -> List[Node]:
        all_nodes = []

        def _collect(node):
            all_nodes.append(node)
            for child in node.children:
                _collect(child)

        _collect(self.root_node)
        return all_nodes

    # --- Results / Cleanup ---

    def create_best_run_copy(self):
        """Create best_run folder (same as NodeManager)."""
        target_node = self._best_node or self.last_successful_node
        if not target_node:
            logger.warning("No best or successful node found.")
            return

        import shutil

        source_folder = self.get_iteration_folder(target_node)
        best_run_folder = os.path.join(self.output_folder, "best_run")
        source_output_folder = os.path.join(source_folder, "output")

        if not os.path.exists(source_output_folder):
            return

        if os.path.exists(best_run_folder):
            shutil.rmtree(best_run_folder)

        shutil.copytree(source_folder, best_run_folder, dirs_exist_ok=True)
        logger.info(f"Created best_run folder from node {target_node.id}")

    def cleanup_all_checkpoints(self):
        """Delete all checkpoints except best node's."""
        ckpt_config = getattr(self.config, "checkpoint_optimization", None)
        if not ckpt_config or not getattr(ckpt_config, "clean_ckpt", False):
            return

        for node in self._get_all_nodes():
            if node == self._best_node or not node.checkpoints:
                continue
            self._delete_checkpoints(node)

    def _delete_checkpoints(self, node: Node, keep_steps: Optional[set] = None):
        """Delete checkpoint directories."""
        import shutil

        if not node.checkpoints:
            return

        to_delete = [c for c in node.checkpoints if not keep_steps or c["step"] not in keep_steps]
        for ckpt in to_delete:
            path = ckpt.get("path", "")
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
            except Exception as e:
                logger.warning(f"Failed to delete checkpoint {path}: {e}")

        if keep_steps:
            node.checkpoints = [c for c in node.checkpoints if c["step"] in keep_steps]
        else:
            node.checkpoints = []

    def get_validation_score_summary(self) -> str:
        nodes_with_scores = [n for n in self._get_all_nodes() if n.validation_score is not None]
        if not nodes_with_scores:
            return "No validation scores available."

        summary = ["Validation Score Summary:"]
        for node in nodes_with_scores:
            marker = " (BEST)" if node == self._best_node else ""
            summary.append(f"Node {node.id} ({node.tool_used}): {node.validation_score}{marker}")

        if self._best_node:
            summary.append(f"\nBest: {self._best_validation_score:.4f} (node {self._best_node.id})")
        return "\n".join(summary)

    def visualize_results(self, output_path: Optional[str] = None) -> str:
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

    def cleanup(self):
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)

    def log_agent_start(self, message: str):
        logger.info(message)

    def log_agent_end(self, message: str):
        logger.info(message)

    def get_early_stop_decision(self, current_metrics_history: List, elapsed_time: float) -> tuple:
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
                    return True, f"[{stopper_type}] {reason}"
            except Exception as e:
                logger.error(f"Error in early stopper [{stopper_type}]: {e}")

        return False, "All early stoppers recommend continuing"

    # --- Properties for Agent Compatibility ---

    @property
    def user_input(self) -> str:
        return self.initial_user_input or ""

    @property
    def best_validation_score(self) -> float:
        return self._best_validation_score if self._best_validation_score is not None else 0.0

    @property
    def best_node(self) -> Optional[Node]:
        return self._best_node

    @property
    def entry_point_script(self) -> str:
        return self.current_node.entry_point_script if self.current_node else ""

    @property
    def previous_entry_point_script(self) -> str:
        if self.current_node and self.current_node.parent:
            return self.current_node.parent.entry_point_script
        return ""

    @property
    def execution_script(self) -> str:
        return self.current_node.execution_script if self.current_node else ""

    @property
    def proposer_suggestions(self) -> str:
        return getattr(self, "_proposer_suggestions_combined", "")

    @property
    def previous_execution_script(self) -> str:
        if self.current_node and self.current_node.parent:
            return self.current_node.parent.execution_script
        return ""

    @property
    def entry_point_script_path(self) -> str:
        if hasattr(self, "current_node_entry_point_path"):
            return self.current_node_entry_point_path
        return ""

    @property
    def error_message(self) -> str:
        return self.current_node.error_message if self.current_node else ""

    @property
    def previous_error_message(self) -> str:
        if self.current_node and self.current_node.parent:
            return self.current_node.parent.error_message
        return ""

    @property
    def error_analysis(self) -> str:
        return self.current_node.error_analysis if self.current_node else ""

    @property
    def all_previous_error_analyses(self) -> str:
        return "\n\n".join(self._all_error_analyses)

    @property
    def previous_error_analysis(self) -> str:
        parent = self.current_node.parent if self.current_node and self.current_node.parent else None
        if not parent or parent.is_successful:
            return ""
        return parent.error_analysis or ""

    @property
    def other_previous_error_analyses(self) -> str:
        if len(self._all_error_analyses) < 2:
            return ""
        return "\n\n".join(self._all_error_analyses[:-1])

    @property
    def per_iteration_output_folder(self) -> str:
        if not self.current_node:
            return os.path.join(self.output_folder, "initialization", "output")
        return self.get_per_iteration_output_folder(self.current_node)

    @property
    def iteration_folder(self) -> str:
        if not self.current_node:
            return os.path.join(self.output_folder, "initialization")
        return self.get_iteration_folder(self.current_node)

    @property
    def node_output_folder_name(self) -> str:
        if not self.current_node:
            base_name = "llmzero_default"
        else:
            base_name = f"llmzero_N{self.current_node.id}_{self.run_timestamp}"

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
        return self.wandb_api_key is not None and self.wandb_api_key.strip() != ""

    @property
    def wandb_api_key_value(self) -> str:
        return self.wandb_api_key if self.wandb_api_key else ""

    def __del__(self):
        self.cleanup()
