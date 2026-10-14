"""Observer for extracting all training step metrics from execution output."""

import logging
from typing import Dict, List, Optional, Tuple

from .base_observer import BaseExecutionObserver

logger = logging.getLogger(__name__)


class StepMetricsObserver(BaseExecutionObserver):
    """
    Observer that extracts all metrics from each training step.

    This observer:
    - Monitors for lines containing "step:" pattern with metrics
    - Parses all key-value pairs from the metrics line (training dynamics, validation metrics, etc.)
    - Stores metrics history for all training steps
    - Does not trigger early termination

    The metrics captured include both training dynamics (actor/*, critic/*, etc.) and
    validation-related information at each training step.
    """

    def __init__(
        self,
        parser,
        name: str = "step_metrics",
        log_stream: str = "stdout",
    ):
        """
        Initialize the step metrics observer.

        Args:
            parser: A BaseStdoutParser the framework supplies. Required — the
                    framework owns the stdout grammar.
            name: Observer name
            log_stream: Which stream to ingest — "stdout", "stderr", or "both".
        """
        super().__init__(name, log_stream=log_stream)
        self.parser = parser

        # State tracking
        self.metrics_history: List[Dict[str, any]] = []
        self.latest_metrics: Dict[str, float] = {}
        self.latest_step: Optional[int] = None

    def _ingest(self, line: str) -> None:
        """Process a line from the selected stream to detect and parse metrics."""
        result = self.parser.parse_step_metrics(line)
        if result is None:
            return

        current_step, parsed_metrics = result
        self.latest_step = current_step

        # Add to metrics history
        metrics_entry = {
            "step": current_step,
            "metrics": parsed_metrics,
        }
        self.metrics_history.append(metrics_entry)

        # Update latest metrics (for backward compatibility)
        self.latest_metrics = parsed_metrics
        logger.info(f"Extracted {len(parsed_metrics)} metrics at step {current_step}")

    def on_execution_complete(self, success: bool, stdout: str, stderr: str) -> None:
        """Called when execution completes."""
        if not self.metrics_history:
            logger.warning("Did not find any step metrics in output")
        else:
            logger.info(
                f"Collected {len(self.metrics_history)} step metrics entries. "
                f"Final metrics at step {self.latest_step}: {len(self.latest_metrics)} metrics"
            )

    def should_terminate(self) -> Tuple[bool, Optional[str]]:
        """Step metrics observer does not trigger early termination."""
        return False, None

    def get_result(self) -> dict:
        """
        Get the observer's results.

        Returns:
            Dictionary containing:
            - metrics_history: List of all metrics entries, each with step and metrics dict
            - num_entries: Number of metrics entries collected
            - metrics: Dictionary of all extracted metrics (most recent step, for backward compatibility)
            - latest_step: The step number of the latest metrics (for backward compatibility)
        """
        return {
            "metrics_history": self.metrics_history,
            "num_entries": len(self.metrics_history),
            "metrics": self.latest_metrics.copy(),
            "latest_step": self.latest_step,
        }
