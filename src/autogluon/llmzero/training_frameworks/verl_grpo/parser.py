"""VERL stdout parser — the exact grammar verl 0.9.0 logs.

Self-contained: carries its own step-metrics grammar plus verl's checkpoint and
progress grammar. This module is the single source of truth for verl's
``global_step_(\\d+)`` checkpoint regex.
"""

import logging
import re
from typing import Dict, Optional, Tuple

from ..stdout_parser import BaseStdoutParser

logger = logging.getLogger(__name__)

# Tight verl step anchor: a metrics line begins "step:N - global_seqlen/...".
_VERL_STEP_PATTERN = re.compile(r"step:\d+\s+-\s+global_seqlen/")
# Generic key:value metric token, tolerating an np.float64(...) wrapper.
_METRIC_TOKEN = re.compile(
    r"([\w/@-]+):(?:np\.(?:float64|float32|int32|int64)\()?([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\)?"
)
# Step number embedded in a verl checkpoint path: ".../global_step_120/...".
_PATH_STEP_RE = re.compile(r"global_step_(\d+)")
# verl 0.9.0 announces the actor checkpoint on STDERR as:
#   INFO:<ts>:Saving checkpoint to /fsx/.../global_step_N/actor
# (older builds printed "local_global_step_folder: <path>" on stdout; match both.)
_PATH_RE = re.compile(r"(?:Saving checkpoint to|local_global_step_folder:)\s*(\S+)")
_SAVE_MARKER_RE = re.compile(r"\[step:\s*(\d+)\]\s*saving checkpoint")
_PROGRESS_RE = re.compile(r"Training Progress:\s*\d+%\|[^\|]+\| (\d+)/\d+ \[\d+:\d+<[^\]]+, ([0-9.]+)s/it\]")


class VerlStdoutParser(BaseStdoutParser):
    """verl 0.9.0 stdout grammar: step metrics, checkpoint marker/path, tqdm progress."""

    step_pattern = _VERL_STEP_PATTERN

    def parse_step_metrics(self, line: str) -> Optional[Tuple[int, Dict[str, float]]]:
        if not self.step_pattern.search(line):
            return None
        step_match = re.search(r"step:(\d+)", line)
        if not step_match:
            return None
        current_step = int(step_match.group(1))
        after = re.search(r"step:\d+(.*)$", line)
        if not after:
            return None
        matches = _METRIC_TOKEN.findall(after.group(1))
        if not matches:
            return None
        metrics: Dict[str, float] = {}
        for key, value in matches:
            try:
                metrics[key] = float(value)
            except ValueError:
                logger.debug(f"Could not convert metric value to float: {key}={value}")
                continue
        return (current_step, metrics) if metrics else None

    def parse_checkpoint(self, line: str, pending_step: Optional[int] = None) -> Optional[dict]:
        save_match = _SAVE_MARKER_RE.search(line)
        if save_match:
            return {"pending_step": int(save_match.group(1))}
        path_match = _PATH_RE.search(line)
        if path_match:
            path = path_match.group(1)
            embedded = _PATH_STEP_RE.search(path)
            step = int(embedded.group(1)) if embedded else pending_step
            if step is not None:
                return {"step": step, "path": path}
        return None

    def parse_progress(self, line: str) -> Optional[Tuple[int, float]]:
        m = _PROGRESS_RE.search(line)
        if not m:
            return None
        return int(m.group(1)), float(m.group(2))
