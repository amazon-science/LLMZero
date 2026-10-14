"""
Base early stopper class for comparing current training with best run.

This module provides the abstract base class for all early stopper implementations.
Early stoppers analyze current training progress and compare with the historical best
run to decide whether to continue or stop training early.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class BaseEarlyStopper(ABC):
    """
    Abstract base class for early stoppers that decide whether to continue training.

    An early stopper analyzes the current training run's metrics and compares
    them against the historical best run to decide whether to continue or stop.

    This provides a common abstraction for different early stopper implementations:
    - Agentic early stoppers (LLM-based, text only)
    - Multimodal early stoppers (LLM with plots)
    - Vision-only early stoppers (LLM with plots only)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        manager,
        agent_llm_config=None,
        agent_prompt_template=None,
    ):
        """
        Initialize the early stopper.

        Args:
            config: Configuration dictionary for the early stopper
            manager: NodeManager instance for accessing state and history
            agent_llm_config: LLM configuration for the early stopper agent
            agent_prompt_template: Optional custom prompt template for the agent
        """
        self.config = config
        self.manager = manager
        self.agent_llm_config = agent_llm_config
        self.agent_prompt_template = agent_prompt_template

        # Create the specific agent implementation
        self.early_stopper_agent = self._create_agent()

    @abstractmethod
    def _create_agent(self):
        """
        Create the specific agent implementation for this early stopper.

        Subclasses must implement this to return their specific agent type
        (e.g., EarlyStopperAgent with configurable text/visual modalities).

        Returns:
            Agent instance
        """
        pass

    @abstractmethod
    def _get_logger_prefix(self) -> str:
        """
        Get the logger prefix for this early stopper type.

        Returns:
            String prefix for logging (e.g., "AgenticEarlyStopper")
        """
        pass

    def should_stop(
        self,
        current_metrics_history: List[Dict],
        best_run_metrics_history: Optional[List[Dict]],
        best_run_validation_score: Optional[float],
        elapsed_time: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Decide whether to stop training early.

        Args:
            current_metrics_history: Metrics from current run up to checkpoint.
                Each entry is a dict with "step" and "metrics" keys.
            best_run_metrics_history: Metrics from the best previous run (or None if first run).
                Same format as current_metrics_history.
            best_run_validation_score: Best validation score of best run (or None if first run).
            elapsed_time: Time elapsed since training started (in seconds).

        Returns:
            Tuple of (should_stop, reason)
            - should_stop: True if training should be stopped
            - reason: Explanation of the decision
        """
        prefix = self._get_logger_prefix()
        logger.info(f"{prefix}: evaluating whether to stop training")

        # Check if we have any current metrics
        if not current_metrics_history:
            logger.warning(f"{prefix}: No current metrics available")
            return False, "No current metrics available, continuing training"

        # Call the early stopper agent to make a decision
        try:
            should_stop, reason = self.early_stopper_agent(
                current_metrics_history=current_metrics_history,
                best_run_metrics_history=best_run_metrics_history,
                best_run_validation_score=best_run_validation_score,
                elapsed_time=elapsed_time,
            )

            decision = "STOP" if should_stop else "CONTINUE"
            logger.info(f"{prefix}: decision={decision}, reason={reason[:100]}...")
            return should_stop, reason

        except Exception as e:
            logger.error(f"{prefix}: Error during evaluation: {e}", exc_info=True)
            return False, f"Error during early stop evaluation: {e}. Continuing training."

    def get_metrics_at_step_count(self, metrics_history: List[Dict], target_step_count: int) -> List[Dict]:
        """
        Get metrics history truncated to a target step count.

        Useful for comparing current run (at checkpoint) with best run
        at the same point in training.

        Args:
            metrics_history: Full metrics history
            target_step_count: Number of steps to include

        Returns:
            Truncated metrics history
        """
        if not metrics_history or target_step_count <= 0:
            return []

        # Find entries up to target_step_count
        truncated = []
        for entry in metrics_history:
            if entry.get("step", 0) < target_step_count:
                truncated.append(entry)
            else:
                break

        return truncated if truncated else metrics_history[:target_step_count]

    def compute_metric_trend(self, metrics_history: List[Dict], metric_key: str) -> Optional[Dict[str, float]]:
        """
        Compute trend statistics for a specific metric.

        Args:
            metrics_history: Metrics history
            metric_key: Key of the metric to analyze

        Returns:
            Dict with trend statistics or None if insufficient data
        """
        if not metrics_history or len(metrics_history) < 2:
            return None

        values = []
        for entry in metrics_history:
            if metric_key in entry.get("metrics", {}):
                val = entry["metrics"][metric_key]
                if isinstance(val, (int, float)):
                    values.append(float(val))

        if len(values) < 2:
            return None

        # Compute basic statistics
        first_half = values[: len(values) // 2]
        second_half = values[len(values) // 2 :]

        return {
            "start": values[0],
            "end": values[-1],
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "first_half_mean": sum(first_half) / len(first_half) if first_half else 0,
            "second_half_mean": sum(second_half) / len(second_half) if second_half else 0,
            "improvement": values[0] - values[-1],  # Positive if decreasing (good for loss)
            "improvement_pct": ((values[0] - values[-1]) / abs(values[0])) * 100 if values[0] != 0 else 0,
        }
