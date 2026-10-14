"""
Early stopper agent that uses LLM to decide whether to stop training early.

This agent compares current training progress with the best previous run
and makes a data-driven decision on whether to continue or stop.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

from ..prompts.early_stopper_agent_prompt import EarlyStopperAgentPrompt
from .base_agent import BaseAgent
from .utils import init_llm

logger = logging.getLogger(__name__)


class EarlyStopperAgent(BaseAgent):
    """
    Analyzes current training progress and compares with best run to decide whether to stop.

    This agent examines:
    - Current training metrics up to checkpoint
    - Best run training metrics (if available)
    - Best run validation score (if available)

    And makes a decision:
    - CONTINUE: Current run shows promise, let it complete
    - STOP: Current run is not better than best, stop early

    Agent Input:
    - Current metrics history (up to checkpoint)
    - Best run metrics history (or None if first run)
    - Best run validation score (or None if first run)

    Agent Output:
    - Tuple of (should_stop: bool, reason: str)
    """

    def __init__(self, config, manager, llm_config, prompt_template=None):
        super().__init__(config=config, manager=manager)

        self.early_stopper_llm_config = llm_config
        self.early_stopper_prompt_template = prompt_template

        self.early_stopper_prompt = EarlyStopperAgentPrompt(
            llm_config=self.early_stopper_llm_config,
            manager=self.manager,
            template=self.early_stopper_prompt_template,
        )

        if self.early_stopper_llm_config and self.early_stopper_llm_config.multi_turn:
            self.early_stopper_llm = init_llm(
                llm_config=self.early_stopper_llm_config,
                agent_name="early_stopper_agent",
                multi_turn=self.early_stopper_llm_config.multi_turn,
            )

    def __call__(
        self,
        current_metrics_history: List[Dict],
        best_run_metrics_history: Optional[List[Dict]],
        best_run_validation_score: Optional[float],
        elapsed_time: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Evaluate whether training should be stopped.

        Args:
            current_metrics_history: Metrics from current run up to checkpoint
            best_run_metrics_history: Metrics from best run (or None if first run)
            best_run_validation_score: Validation score of best run (or None)
            elapsed_time: Time elapsed since training started (in seconds)

        Returns:
            Tuple of (should_stop, reason)
        """
        self.manager.log_agent_start("EarlyStopperAgent: analyzing training progress to decide whether to stop early.")

        # Get config parameters for early stopper
        early_stopper_config = getattr(self.config, "early_stopper", {})
        num_samples = getattr(early_stopper_config, "num_samples", 10)
        use_average = getattr(early_stopper_config, "use_average", True)
        text_metrics = getattr(early_stopper_config, "text_metrics", [])
        plot_metrics = getattr(early_stopper_config, "plot_metrics", [])

        # Fall back to framework defaults when metric lists are not configured
        if not text_metrics or not plot_metrics:
            fw = self.training_framework
            if not text_metrics:
                text_metrics = fw.default_text_metrics()
            if not plot_metrics:
                plot_metrics = fw.default_plot_metrics()

        # Get validation metric pattern from main config
        validation_metric_pattern = getattr(self.config, "validation_metric_pattern", None)

        # Build prompt for making early stop decision
        prompt = self.early_stopper_prompt.build(
            current_metrics_history=current_metrics_history,
            best_run_metrics_history=best_run_metrics_history,
            best_run_validation_score=best_run_validation_score,
            num_samples=num_samples,
            use_average=use_average,
            text_metrics=text_metrics,
            plot_metrics=plot_metrics,
            validation_metric_pattern=validation_metric_pattern,
            elapsed_time=elapsed_time,
        )

        if not self.early_stopper_llm_config.multi_turn:
            self.early_stopper_llm = init_llm(
                llm_config=self.early_stopper_llm_config,
                agent_name="early_stopper_agent",
                multi_turn=self.early_stopper_llm_config.multi_turn,
            )

        response = self.early_stopper_llm.assistant_chat(prompt)

        should_stop, reason = self.early_stopper_prompt.parse(response, elapsed_time=elapsed_time)

        # Save prompt and response in early_stopper/<timestamp>/ folder
        self._save_early_stopper_outputs(prompt, response, elapsed_time)

        self.manager.log_agent_end(f"EarlyStopperAgent: decision made - {'STOP' if should_stop else 'CONTINUE'}")

        return should_stop, reason

    def _save_early_stopper_outputs(self, prompt, response, elapsed_time: Optional[float]):
        """Save early stopper prompt and response to timestamped folder."""
        elapsed_str = f"{int(elapsed_time)}s" if elapsed_time is not None else "unknown"

        # Create early_stopper/<timestamp>/ directory
        iter_folder = self.manager.get_iteration_folder(self.manager.current_node)
        early_stopper_dir = os.path.join(iter_folder, "early_stopper", elapsed_str)
        os.makedirs(early_stopper_dir, exist_ok=True)

        # Save prompt (text only for this agent)
        prompt_text = prompt if isinstance(prompt, str) else str(prompt)
        prompt_path = os.path.join(early_stopper_dir, "prompt.txt")
        with open(prompt_path, "w") as f:
            f.write(prompt_text)
        logger.info(f"Saved early stopper prompt to {prompt_path}")

        # Save response
        response_text = response if isinstance(response, str) else str(response)
        response_path = os.path.join(early_stopper_dir, "response.txt")
        with open(response_path, "w") as f:
            f.write(response_text)
        logger.info(f"Saved early stopper response to {response_path}")
