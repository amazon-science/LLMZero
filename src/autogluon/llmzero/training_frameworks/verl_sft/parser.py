# src/autogluon/llmzero/training_frameworks/verl_sft/parser.py
"""verl 0.9.0 SFT stdout parser.

verl SFT logs metrics via the console LocalLogger (print -> stdout) in the
grammar ``step:N - key:value - key:value``. Train lines carry train/* and perf/*
metrics; validation lines carry a single ``val/loss``. Unlike GRPO, SFT lines have
NO ``global_seqlen/`` token, so the step anchor differs. Checkpoints are saved flat
under ``global_step_N/`` and announced with ``Saving checkpoint to: <path>``.

NOTE(verl-0.9.0): the first metric token after ``step:N - `` is version-dependent.
In 0.9.0 train-step lines LEAD with ``perf/*`` (e.g. ``step:3 - perf/max_memory_...
- train/loss:...``), not ``train/``; validation lines still lead with ``val/loss``.
The step anchor therefore matches any ``step:N - <namespaced-key>:`` token rather than
requiring a literal ``train/``/``val/`` prefix, so 0.9.0 train lines are not dropped.
"""

import logging
import re
from typing import Dict, Optional, Tuple

from ..stdout_parser import BaseStdoutParser

logger = logging.getLogger(__name__)

# SFT step line begins "step:N - <namespaced-key>:...". In verl 0.9.0 the leading
# metric namespace is version-dependent (train lines lead with perf/*, val lines with
# val/loss), so anchor on any "step:N - <key>/<...>:<value>" token instead of a literal
# train//val/ prefix. Require a "/" in the leading key to avoid matching prose that
# happens to contain "step:N - ".
_SFT_STEP_PATTERN = re.compile(r"step:\d+\s+-\s+[\w@-]+/[\w/@-]*:")
# Generic key:value metric token (reused verbatim from verl_grpo), tolerating an
# np.float64(...) wrapper and slashes in keys.
_METRIC_TOKEN = re.compile(
    r"([\w/@-]+):(?:np\.(?:float64|float32|int32|int64)\()?([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\)?"
)
# Step embedded in a verl checkpoint path: ".../global_step_120".
_PATH_STEP_RE = re.compile(r"global_step_(\d+)")
# SFT save marker + path on one line: "Saving checkpoint to: <dir>/global_step_N".
_SAVE_PATH_RE = re.compile(r"Saving checkpoint to:\s*(\S+)")
# tqdm-style progress line (best-effort): "... 12/240 [00:30<10:00, 2.50s/it]".
_PROGRESS_RE = re.compile(r"\| (\d+)/\d+ \[\d+:\d+<[^\]]+, ([0-9.]+)s/it\]")


class VerlSftStdoutParser(BaseStdoutParser):
    """verl 0.9.0 SFT grammar: step train/val/perf metrics, flat checkpoint path, tqdm progress."""

    step_pattern = _SFT_STEP_PATTERN

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
        path_match = _SAVE_PATH_RE.search(line)
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
