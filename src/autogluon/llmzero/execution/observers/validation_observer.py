"""Observer for tracking the validation score during training (best or final)."""

import logging
import re
from typing import Dict, List, Optional, Tuple

from .base_observer import BaseExecutionObserver

logger = logging.getLogger(__name__)


class ValidationObserver(BaseExecutionObserver):
    """
    Observer that tracks the validation score during training (best or final).

    This observer monitors training output for validation metrics and:
    - Extracts validation metrics from step metrics lines (same format as StepMetricsObserver)
    - Selects the first metric matching a REQUIRED pattern (e.g., "val-core.*__dev")
    - Records the selected metric name and uses it consistently for all validations
    - Tracks the best score across all validation checkpoints
    - Uses the best score for MCTS node evaluation

    The reported score follows the configured ``reduction``: it reports either the
    direction-aware best score across the run ("best") or the last recorded
    validation score ("final").

    The metric selection happens once at the first validation step, and the same
    metric name is used for all subsequent validations to ensure consistency.

    The validation_metric_pattern is REQUIRED - without it, nodes cannot be evaluated
    and compared in the MCTS search.
    """

    def __init__(
        self,
        parser,
        validation_metric_pattern: Optional[str] = None,
        score_direction: str = "maximize",
        reduction: str = "best",
        name: str = "validation",
        log_stream: str = "stdout",
    ):
        """
        Initialize the validation observer.

        Args:
            parser: A BaseStdoutParser the framework supplies (required).
            validation_metric_pattern: Regex pattern to match the validation metric name (REQUIRED).
            score_direction: "maximize" (default) or "minimize" — which checkpoint within a
                    run is the run's best (highest vs lowest metric value).
            reduction: "best" (direction-aware best across the run) or "final" (last recorded).
            name: Observer name
            log_stream: Which stream to ingest — "stdout", "stderr", or "both".
        """
        super().__init__(name, log_stream=log_stream)
        self.parser = parser
        self.minimize = score_direction == "minimize"
        self.reduction = reduction

        if not validation_metric_pattern:
            raise ValueError(
                "validation_metric_pattern is required for ValidationObserver. "
                "Please configure validation_metric_pattern in your config YAML file."
            )

        self.validation_metric_pattern = re.compile(validation_metric_pattern)

        # State tracking
        self.best_validation_score: Optional[float] = None
        self.best_validation_step: Optional[int] = None
        self.best_validation_metrics: Optional[Dict[str, float]] = None  # Full metrics at best step
        self.first_validation_score: Optional[float] = None  # The "val" score
        self.all_validation_scores: List[Tuple[int, float]] = []  # (step, score) pairs
        self.validation_count = 0
        self.selected_validation_metric_name: Optional[str] = None  # The specific metric name selected
        self.final_validation_score: Optional[float] = None
        self.final_validation_step: Optional[int] = None
        self.final_validation_metrics: Optional[Dict[str, float]] = None

    def _ingest(self, line: str) -> None:
        """Process a line from the selected stream to detect and track validation scores."""
        result = self.parser.parse_step_metrics(line)
        if result is None:
            return

        current_step, all_metrics = result

        # Select validation metric from all metrics (delegated to the parser).
        previously_selected = self.selected_validation_metric_name
        score, self.selected_validation_metric_name = self.parser.select_validation_metric(
            all_metrics, self.validation_metric_pattern, self.selected_validation_metric_name
        )

        # Log once when a new metric name gets locked in (None -> name transition).
        if previously_selected is None and self.selected_validation_metric_name is not None:
            logger.info(
                f"Selected validation metric: '{self.selected_validation_metric_name}' "
                f"(value: {all_metrics[self.selected_validation_metric_name]:.4f}). "
                f"This metric will be used for all subsequent validations."
            )

        if score is not None:
            # Record validation score
            self.all_validation_scores.append((current_step, score))
            self.validation_count += 1

            self.final_validation_score = score
            self.final_validation_step = current_step
            self.final_validation_metrics = all_metrics.copy()

            if self.first_validation_score is None:
                # First validation - initialize best
                self.first_validation_score = score
                self.best_validation_score = score
                self.best_validation_step = current_step
                self.best_validation_metrics = all_metrics.copy()
                logger.info(f"First validation score: {score} at step {current_step} with {len(all_metrics)} metrics")
            else:
                logger.debug(f"Validation score at step {current_step}: {score}")

            # Update best if better (direction-aware: lower is better when minimizing)
            is_better = score < self.best_validation_score if self.minimize else score > self.best_validation_score
            if is_better:
                self.best_validation_score = score
                self.best_validation_step = current_step
                self.best_validation_metrics = all_metrics.copy()
                logger.info(
                    f"New best validation score: {score} at step {current_step} with {len(all_metrics)} metrics"
                )

    def on_execution_complete(self, success: bool, stdout: str, stderr: str) -> None:
        """Called when execution completes."""
        if self.validation_count == 0:
            logger.warning("Did not find any validation scores during training")
        else:
            logger.info(
                f"Tracked {self.validation_count} validation scores using metric: '{self.selected_validation_metric_name}'. "
                f"First (val): {self.first_validation_score}, "
                f"Best: {self.best_validation_score} at step {self.best_validation_step}"
            )

    def should_terminate(self) -> Tuple[bool, Optional[str]]:
        """Validation observer does not trigger early termination."""
        return False, None

    def get_result(self) -> dict:
        """
        Get the observer's results.

        The reported validation_score/step/metrics follow the configured reduction:
        - "best": the direction-aware best across the run.
        - "final": the last recorded validation score.

        Returns a dict with:
            - validation_score / validation_step / validation_metrics (per reduction)
            - first_validation_score, all_validation_scores, validation_count
            - selected_validation_metric_name, reduction
        """
        if self.reduction == "final":
            score = self.final_validation_score
            step = self.final_validation_step
            metrics = self.final_validation_metrics
        else:
            score = self.best_validation_score
            step = self.best_validation_step
            metrics = self.best_validation_metrics
        return {
            "validation_score": score,
            "validation_step": step,
            "validation_metrics": metrics,
            "first_validation_score": self.first_validation_score,
            "all_validation_scores": self.all_validation_scores,
            "validation_count": self.validation_count,
            "selected_validation_metric_name": self.selected_validation_metric_name,
            "reduction": self.reduction,
        }
