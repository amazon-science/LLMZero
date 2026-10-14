"""Observer for checkpoint-based early stopping with LLM comparison."""

import logging
import time
from typing import Callable, Dict, List, Optional, Tuple

from .base_observer import BaseExecutionObserver

logger = logging.getLogger(__name__)


class CheckpointEarlyStopObserver(BaseExecutionObserver):
    """
    Observer that triggers early stopping decisions at checkpoint intervals.

    This observer:
    1. Monitors elapsed training time
    2. Collects step metrics in real-time (similar to StepMetricsObserver)
    3. At checkpoint intervals, invokes the early stopper callback to compare
       current training with the best previous run
    4. Signals termination if early stopper decides to stop

    The early stopper callback receives current metrics and returns (should_stop, reason).
    """

    def __init__(
        self,
        checkpoint_interval_seconds: float,
        early_stopper_callback: Callable[[List[Dict], Optional[List[Dict]], Optional[float]], Tuple[bool, str]],
        parser,
        best_run_metrics: Optional[List[Dict]] = None,
        best_run_validation_score: Optional[float] = None,
        min_steps_before_early_stop: int = 0,
        name: str = "checkpoint_early_stop",
        log_stream: str = "stdout",
    ):
        """
        Initialize the checkpoint early stop observer.

        Args:
            checkpoint_interval_seconds: Time interval between checkpoint evaluations (e.g., 1800 for 30 min)
            early_stopper_callback: Function that takes (current_metrics, best_metrics, best_score)
                                   and returns (should_stop, reason)
            parser: A BaseStdoutParser the framework supplies (required) — owns the stdout grammar.
            best_run_metrics: Metrics history from the best previous run (or None if first run)
            best_run_validation_score: Validation score of the best previous run (or None)
            min_steps_before_early_stop: Number of steps to skip early stopping at the beginning of each run
            name: Observer name
            log_stream: Which stream to ingest — "stdout", "stderr", or "both".
        """
        super().__init__(name, log_stream=log_stream)
        self.checkpoint_interval = checkpoint_interval_seconds
        self.early_stopper_callback = early_stopper_callback
        self.best_run_metrics = best_run_metrics
        self.best_run_validation_score = best_run_validation_score
        self.parser = parser

        # Timing state
        self.start_time = time.time()
        self.last_checkpoint_time = self.start_time
        self.checkpoint_count = 0

        # Decision state
        self.should_stop = False
        self.stop_reason: Optional[str] = None
        self.decision_made_at_checkpoint: Optional[int] = None

        # Early stop grace period
        self.min_steps_before_early_stop = min_steps_before_early_stop

        # Metrics collection (similar to StepMetricsObserver)
        self.metrics_history: List[Dict] = []
        self.latest_step: Optional[int] = None

    @property
    def steps_completed(self) -> int:
        """Number of training steps observed in the current run."""
        return len(self.metrics_history)

    def _ingest(self, line: str) -> None:
        """Process a line from the selected stream: collect metrics and check for checkpoint."""
        # Collect metrics (reusing pattern from StepMetricsObserver)
        self._collect_metrics(line)

        # Check if checkpoint interval has passed
        if not self.should_stop:
            current_time = time.time()
            elapsed_since_checkpoint = current_time - self.last_checkpoint_time

            if elapsed_since_checkpoint >= self.checkpoint_interval:
                self._evaluate_at_checkpoint()
                self.last_checkpoint_time = current_time
                self.checkpoint_count += 1

    def _collect_metrics(self, line: str) -> None:
        """Collect step metrics from output line (delegated to the parser)."""
        result = self.parser.parse_step_metrics(line)
        if result is None:
            return
        current_step, parsed_metrics = result
        self.latest_step = current_step
        self.metrics_history.append({"step": current_step, "metrics": parsed_metrics})

    def _evaluate_at_checkpoint(self) -> None:
        """Evaluate current training progress at checkpoint using early stopper."""
        if not self.metrics_history:
            logger.info("No metrics collected yet, skipping checkpoint evaluation")
            return

        if self.steps_completed < self.min_steps_before_early_stop:
            logger.info(
                f"Only {self.steps_completed} steps completed (need {self.min_steps_before_early_stop}), "
                f"skipping early stop check"
            )
            return

        elapsed_time = time.time() - self.start_time
        logger.info(
            f"Checkpoint #{self.checkpoint_count + 1} at {elapsed_time:.1f}s "
            f"({elapsed_time/60:.1f} min) with {len(self.metrics_history)} metrics entries"
        )

        try:
            should_stop, reason = self.early_stopper_callback(
                self.metrics_history,
                self.best_run_metrics,
                self.best_run_validation_score,
                elapsed_time,
            )

            if should_stop:
                self.should_stop = True
                self.stop_reason = reason
                self.decision_made_at_checkpoint = self.checkpoint_count + 1
                logger.info(f"Early stopper decision: STOP - {reason}")
            else:
                logger.info(f"Early stopper decision: CONTINUE - {reason}")

        except Exception as e:
            logger.error(f"Error during early stopper evaluation: {e}", exc_info=True)
            # Don't stop on evaluation error - let training continue

    def on_execution_complete(self, success: bool, stdout: str, stderr: str) -> None:
        """Called when execution completes."""
        elapsed_time = time.time() - self.start_time
        if self.should_stop:
            logger.info(
                f"Training was early stopped at checkpoint #{self.decision_made_at_checkpoint} "
                f"after {elapsed_time:.1f}s ({elapsed_time/60:.1f} min). Reason: {self.stop_reason}"
            )
        else:
            logger.info(
                f"Training completed normally after {elapsed_time:.1f}s ({elapsed_time/60:.1f} min). "
                f"Evaluated at {self.checkpoint_count} checkpoints."
            )

    def should_terminate(self) -> Tuple[bool, Optional[str]]:
        """Return termination decision from checkpoint evaluation."""
        return self.should_stop, self.stop_reason

    def get_result(self) -> dict:
        """
        Get the observer's results.

        Returns:
            Dictionary containing:
            - early_stopped: Whether training was early stopped
            - stop_reason: Reason for early stopping (if applicable)
            - decision_checkpoint: Checkpoint number where stop decision was made
            - checkpoint_count: Total number of checkpoints evaluated
            - elapsed_time: Total elapsed time in seconds
            - metrics_at_stop: Metrics history up to the stop point
            - latest_step: Latest training step observed
        """
        return {
            "early_stopped": self.should_stop,
            "stop_reason": self.stop_reason,
            "decision_checkpoint": self.decision_made_at_checkpoint,
            "checkpoint_count": self.checkpoint_count,
            "elapsed_time": time.time() - self.start_time,
            "metrics_at_stop": self.metrics_history.copy(),
            "latest_step": self.latest_step,
        }
