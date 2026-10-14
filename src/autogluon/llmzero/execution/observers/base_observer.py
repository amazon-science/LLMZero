"""Base class for execution observers."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple


class BaseExecutionObserver(ABC):
    """
    Base class for observers that monitor code execution.

    Observers process output lines in real-time, collect metrics, and optionally
    signal early termination. Which subprocess stream feeds an observer is chosen
    by ``log_stream`` ("stdout" (default), "stderr", or "both"); the framework
    declares it via ``BaseTrainingFramework.log_stream()``. Subclasses implement
    the stream-agnostic ``_ingest(line)`` hook; the base routes stdout/stderr
    lines to it based on ``log_stream``.
    """

    def __init__(self, name: str, log_stream: str = "stdout"):
        """
        Args:
            name: Unique identifier for this observer.
            log_stream: Which stream to ingest — "stdout", "stderr", or "both".
        """
        self.name = name
        self.log_stream = log_stream

    def _ingest(self, line: str) -> None:
        """Process one output line from the selected stream. Default: no-op."""
        pass

    def on_stdout_line(self, line: str) -> None:
        """Route a stdout line to ``_ingest`` when the selected stream includes stdout."""
        if self.log_stream in ("stdout", "both"):
            self._ingest(line)

    def on_stderr_line(self, line: str) -> None:
        """Route a stderr line to ``_ingest`` when the selected stream includes stderr."""
        if self.log_stream in ("stderr", "both"):
            self._ingest(line)

    @abstractmethod
    def on_execution_complete(self, success: bool, stdout: str, stderr: str) -> None:
        """Called when execution completes."""
        pass

    @abstractmethod
    def get_result(self) -> dict:
        """Return the observer's collected results as a dict."""
        pass

    def should_terminate(self) -> Tuple[bool, Optional[str]]:
        """Return (should_stop, reason); default never terminates early."""
        return False, None
