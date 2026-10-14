"""Observer for tracking checkpoint save events during training."""

import logging
import time
from typing import Dict, List, Optional, Tuple

from .base_observer import BaseExecutionObserver

logger = logging.getLogger(__name__)


class CheckpointPathObserver(BaseExecutionObserver):
    """
    Stateful observer that records checkpoint save events and their paths.

    The framework-supplied parser (a ``BaseStdoutParser``) owns all checkpoint
    grammar; this observer is generic. Per line it asks the parser to
    ``parse_checkpoint`` and acts on the result:

    - A completed checkpoint ``{'step', 'path'}`` (e.g. when the path embeds the
      step) is recorded immediately.
    - A ``{'pending_step'}`` from a save-marker line primes a step that is paired
      with the following path line. When both a pending step and a path-embedded
      step are available, the parser decides which wins.

    The parser is stateless; this observer owns the pending-step state machine
    (priming, pairing, and discarding a pending step on an unrelated line).
    """

    def __init__(
        self,
        parser,
        name: str = "checkpoint_paths",
        log_stream: str = "stdout",
    ):
        super().__init__(name, log_stream=log_stream)
        self.parser = parser
        self._pending_step: Optional[int] = None
        self.checkpoints: List[Dict] = []  # [{step, path, timestamp}]

    def _ingest(self, line: str) -> None:
        result = self.parser.parse_checkpoint(line, self._pending_step)
        if result is None:
            # Non-checkpoint line: if a save marker was pending, the original
            # observer discarded it when a non-path line arrived. Preserve that.
            self._pending_step = None
            return

        if "pending_step" in result:
            self._pending_step = result["pending_step"]
            return

        # result has {'step', 'path'}: record and clear pending.
        self.checkpoints.append(
            {
                "step": result["step"],
                "path": result["path"],
                "timestamp": time.time(),
            }
        )
        logger.info(f"Checkpoint recorded: step {result['step']} -> {result['path']}")
        self._pending_step = None

    def on_execution_complete(self, success: bool, stdout: str, stderr: str) -> None:
        logger.info(f"CheckpointPathObserver: execution complete, recorded {len(self.checkpoints)} checkpoints")

    def should_terminate(self) -> Tuple[bool, Optional[str]]:
        return False, None

    def get_result(self) -> dict:
        return {
            "checkpoints": self.checkpoints,
            "num_checkpoints": len(self.checkpoints),
        }
