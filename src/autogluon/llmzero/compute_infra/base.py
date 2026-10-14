"""Abstract compute backend interface for LLMZero.

This module defines the abstract base class that all compute backends must implement.
Currently supported backends: raycluster, conda, docker, slurmdocker.
"""

import logging
import os
import re
import signal
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ComputeJobResult:
    """Result of a compute job submission."""

    returncode: int
    stdout: str
    stderr: str


class BaseComputeBackend(ABC):
    """Abstract interface for compute infrastructure backends.

    Every backend must implement these methods so that CodeExecutor can
    submit, monitor, and terminate jobs in a backend-agnostic way.

    Key assumption
    --------------
    ``CodeExecutor.execute()`` runs the generated script as a **local**
    subprocess (``bash -c <script>``) and reads training logs from its
    stdout/stderr in real-time.  The submission command produced by the
    backend's ``execution_script_template()`` must therefore **block and
    stream the remote job's output** to the local process's stdout/stderr.
    Ray satisfies this via ``ray job submit`` (without ``--no-wait``).
    A backend whose submission command returns immediately (e.g. Slurm
    ``sbatch``) must arrange for log streaming itself (e.g. by tailing
    the log file after submission) or the execution model will need to
    be extended.
    """

    # -- Properties ----------------------------------------------------------

    @property
    @abstractmethod
    def address(self) -> str:
        """Human-readable address string for logging (e.g. the dashboard URL)."""

    @property
    @abstractmethod
    def backend_type(self) -> str:
        """Short identifier for this backend type (e.g. 'raycluster')."""

    @property
    def last_job_id(self) -> Optional[str]:
        """The ID of the most recently submitted job, or None.

        Only meaningful for backends with remote jobs (Ray cluster) or
        container IDs (Docker).  The conda backend does not use this.
        """
        return getattr(self, "_last_job_id", None)

    @last_job_id.setter
    def last_job_id(self, value: Optional[str]) -> None:
        self._last_job_id = value

    # -- Script manipulation -------------------------------------------------

    @abstractmethod
    def prepare_script(self, script_content: str) -> str:
        """Rewrite backend-specific addresses/paths in *script_content*.

        For Ray this replaces localhost addresses with K8s service DNS.
        """

    def extract_job_id(self, stdout: str) -> Optional[str]:
        """Extract a job ID from the stdout of a submission command.

        Returns ``None`` by default.  Only backends with remote jobs
        (Ray cluster) need to override this.
        """
        return None

    @staticmethod
    def _strip_ray_submit_wrapper(script_content: str) -> str:
        """Best-effort extraction of the inner script from a ray job submit wrapper.

        If the script contains ``ray job submit ... -- bash -lc '...'``,
        extract the inner part.  Otherwise return the original content unchanged.
        """
        match = re.search(
            r"ray\s+job\s+submit\b.*?--\s*\n?\s*bash\s+-lc\s+'(.*)'",
            script_content,
            re.DOTALL,
        )
        if match:
            logger.debug("Stripped ray job submit wrapper from script")
            return match.group(1)
        return script_content

    # -- Job lifecycle -------------------------------------------------------

    @abstractmethod
    def submit_job(self, script_content: str) -> ComputeJobResult:
        """Submit *script_content* as a job and return the result.

        .. note::

            This method is currently only used by the CLI
            ``llmzero submit-script`` path (via
            ``CodeExecutor.submit_script``).  The MCTS training loop uses
            ``CodeExecutor.execute()`` instead, which calls
            ``prepare_script()`` and spawns its own subprocess with
            observer support, timeout, and early-stop.  The two paths
            should be unified in the future so that ``submit_job`` also
            benefits from observer notifications and process-group
            management.
        """

    def terminate(
        self,
        pid: Optional[int] = None,
        job_id: Optional[str] = None,
        stdout_so_far: str = "",
    ) -> bool:
        """Unified termination entry point.

        Handles both the MCTS execution path (where we have a local *pid*)
        and the CLI ``stop-job`` path (where we only have a *job_id*).
        Either or both of *pid* / *job_id* may be provided.

        Steps:
        1. Resolve *job_id* from *stdout_so_far* if not provided.
        2. Stop the remote / external job via ``_stop_remote_job`` if we
           have a *job_id*.
        3. Kill the local process group via ``_kill_process_group`` if we
           have a *pid*.
        """
        # 1. Try to resolve job_id from stdout if not provided
        if not job_id and stdout_so_far:
            job_id = self.extract_job_id(stdout_so_far)

        # 2. Stop remote/external job if we have an ID
        if job_id:
            self._stop_remote_job(job_id)
            if pid is not None:
                # Give remote side time to wind down before killing local client
                time.sleep(3)

        # 3. Kill local process group if we have a PID
        if pid is not None:
            self._kill_process_group(pid)

        return True

    def _kill_process_group(self, pid: int) -> None:
        """Send SIGTERM then SIGKILL to the process group rooted at *pid*."""
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
            logger.info(f"Sent SIGTERM to process group {pgid}")
        except (ProcessLookupError, PermissionError) as e:
            logger.warning(f"Failed to SIGTERM process group for PID {pid}: {e}")
            return

        time.sleep(3)

        try:
            if os.waitpid(pid, os.WNOHANG) == (0, 0):
                pgid = os.getpgid(pid)
                os.killpg(pgid, signal.SIGKILL)
                logger.info(f"Sent SIGKILL to process group {pgid}")
        except (ProcessLookupError, ChildProcessError, PermissionError):
            pass  # Already dead — that's fine

    def _stop_remote_job(self, job_id: str) -> bool:
        """Stop a backend-specific remote job by *job_id*.

        No-op by default.  Backends with remote execution (Ray cluster)
        or container isolation (Docker) override this to call
        ``ray job stop`` or ``docker stop`` respectively.

        Called by ``terminate()`` when a *job_id* is available.
        """
        return True

    @abstractmethod
    def cleanup(self) -> bool:
        """Clean up backend-specific caches/state (e.g. /tmp/ray on workers)."""

    # -- Factory & template --------------------------------------------------

    @classmethod
    @abstractmethod
    def validate_config(cls, config_dict: Dict) -> None:
        """Validate that *config_dict* contains all required keys.

        Raise ``ValueError`` with a helpful message (including CLI hints
        where applicable) if anything is missing.
        """

    @classmethod
    @abstractmethod
    def from_config(cls, config_dict: Dict) -> "BaseComputeBackend":
        """Instantiate a backend from a flat configuration dictionary."""

    @classmethod
    @abstractmethod
    def execution_script_template(cls, config_dict: Optional[Dict] = None) -> str:
        """Return the backend-specific portion of the execution script prompt.

        This is injected into ExecutionCoderPrompt so the LLM generates
        a submission wrapper appropriate for the backend.

        The generated script will be executed as a local subprocess by
        ``CodeExecutor.execute()``.  The template must instruct the LLM
        to produce a script whose submission command **blocks and streams
        the remote job's stdout/stderr** to the local process so that
        observers can parse output in real-time.

        Parameters
        ----------
        config_dict : dict, optional
            Backend configuration dictionary. Passed so that configurable
            flags (e.g. ``entrypoint_num_gpus``) can be injected into the
            template conditionally.
        """
