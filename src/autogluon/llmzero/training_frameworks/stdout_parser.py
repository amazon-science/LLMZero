"""Stdout parsing abstraction.

A framework supplies a parser so the observers stay framework-agnostic
mechanisms. There is no default parser — every framework owns its stdout
grammar (see ``verl_grpo/parser.py`` for the worked example).
"""

import re
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple


class BaseStdoutParser(ABC):
    """Parse training stdout lines into structured signals for observers."""

    @abstractmethod
    def parse_step_metrics(self, line: str) -> Optional[Tuple[int, Dict[str, float]]]:
        """Return (step, {metric_name: value}) for a metrics line, else None."""

    def parse_checkpoint(self, line: str, pending_step: Optional[int] = None) -> Optional[dict]:
        """Return {'step': int, 'path': str} for a checkpoint-save line, else None.

        A framework that emits a separate save marker first may instead return
        {'pending_step': int} to prime a step paired with the following path line.
        """
        return None

    def parse_progress(self, line: str) -> Optional[Tuple[int, float]]:
        """Return (iteration, seconds_per_iter) for a progress line, else None."""
        return None

    def select_validation_metric(
        self,
        metrics: Dict[str, float],
        pattern: "re.Pattern",
        selected_name: Optional[str],
    ) -> Tuple[Optional[float], Optional[str]]:
        """Pick the validation score from a metrics dict.

        Returns (score, selected_name). Once a name is selected it is reused on
        later calls for consistency. Returns (None, selected_name) when this step
        has no validation metric.
        """
        if not metrics:
            return None, selected_name
        if selected_name:
            return (metrics.get(selected_name), selected_name)
        for key in metrics:
            if pattern.search(key):
                return metrics[key], key
        return None, selected_name
