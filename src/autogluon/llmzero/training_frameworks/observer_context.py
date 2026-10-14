"""Inputs to BaseTrainingFramework.build_observers().

Carries exactly what the executer agent knows at execution time so the
framework can construct the right observer set without importing agent code.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass
class ObserverContext:
    validation_score_mode: str = "best_validation"
    validation_metric_pattern: Optional[str] = None

    # checkpoint path tracking
    checkpoint_optimization_enabled: bool = False

    # LLM early stopper
    early_stopper_enabled: bool = False
    early_stopper_callback: Optional[Callable] = None
    best_run_metrics: Optional[List[dict]] = None
    best_run_validation_score: Optional[float] = None
    min_steps_before_early_stop: int = 0
    checkpoint_interval_seconds: float = 1800.0
    is_from_scratch: bool = True
