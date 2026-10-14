"""Code execution with observer pattern support."""

import logging
import os
import select
import subprocess
import time
from typing import List, Optional, Tuple

from rich.progress import Progress, TextColumn

from ..rich_logging import show_progress_bar
from .observers import BaseExecutionObserver

logger = logging.getLogger(__name__)


class CodeExecutor:
    """
    Executes code with real-time output streaming, timeout, and observer notifications.

    The executor manages subprocess execution and notifies registered observers
    of output lines in real-time. Observers can signal early termination.

    Important assumption: the generated script is executed as a local subprocess
    (``bash -c <script>``), and the executor reads training logs directly from
    the subprocess's stdout/stderr.  This means the submission command must
    **block and stream the remote job's output** to the local process.  Ray's
    ``ray job submit`` (without ``--no-wait``) satisfies this — it forwards the
    remote job's stdout/stderr to the caller.  Backends where the submission
    command returns immediately (e.g. Slurm ``sbatch``) would need to wrap the
    call so that logs are still streamed to stdout/stderr, or the execution
    model here would need to be extended to support polling / log-tailing.
    """

    def __init__(
        self,
        language: str,
        timeout: float,
        compute_backend_config: Optional[dict] = None,
    ):
        """
        Initialize the code executor.

        Args:
            language: The language to execute ("python" or "bash")
            timeout: Maximum execution time in seconds
            compute_backend_config: Optional compute backend configuration dict.
                                    Must contain a "type" key (default "raycluster").
                                    Required for bash scripts that submit compute jobs.
        """
        assert language in ["bash", "python"], f"Unsupported language: {language}"
        self.language = language
        self.timeout = timeout
        self.observers: List[BaseExecutionObserver] = []
        self._last_process: Optional[subprocess.Popen] = None
        self._last_stdout_chunks: List[str] = []

        self.compute_backend = None
        if compute_backend_config:
            try:
                from ..compute_infra import create_backend

                self.compute_backend = create_backend(compute_backend_config)
            except Exception as e:
                logger.warning(f"Could not create compute backend: {e}")

    def attach_observer(self, observer: BaseExecutionObserver) -> None:
        """
        Attach an observer to monitor execution.

        Args:
            observer: Observer instance to attach
        """
        self.observers.append(observer)
        logger.info(f"Attached observer: {observer.name}")

    # -- Generic job lifecycle methods ---------------------------------------

    def terminate_last_job(self) -> bool:
        """Terminate the last job launched by ``execute()``."""
        if self._last_process is None:
            logger.warning("No last process to terminate")
            return False
        self._terminate(self._last_process, self._last_stdout_chunks)
        return True

    def cleanup_backend(self) -> bool:
        """
        Clean up backend-specific caches/state.

        Returns:
            True if successful, False otherwise
        """
        if not self.compute_backend:
            logger.warning("No compute backend available")
            return False
        return self.compute_backend.cleanup()

    def submit_script(self, script_path: str) -> Tuple[bool, Optional[str], str]:
        """
        Submit a bash script as a compute job.

        This is the entry point for the ``llmzero submit-script`` CLI command.
        It delegates to ``compute_backend.submit_job()``, which calls
        ``prepare_script()`` and ``_execute_script()`` internally.

        .. note::

            The MCTS training loop does **not** use this method.  It calls
            ``execute()`` instead, which spawns its own subprocess with
            observer support (real-time metric tracking, early-stop,
            checkpoint monitoring) and process-group management
            (``os.setsid`` + ``backend.terminate()``).  The ``submit_job``
            path lacks these features.  The two paths should be unified
            in the future.

        Args:
            script_path: Path to the bash script to submit

        Returns:
            Tuple of (success, job_id, message)
        """
        if not self.compute_backend:
            return False, None, "No compute backend available - backend config not provided"

        try:
            with open(script_path, "r") as f:
                script_content = f.read()

            logger.info(f"Submitting script {script_path} as {self.compute_backend.backend_type} job")

            result = self.compute_backend.submit_job(script_content)

            if result.returncode == 0:
                job_id = self.compute_backend.last_job_id
                if job_id:
                    message = f"Successfully submitted job: {job_id}"
                    logger.info(message)
                    return True, job_id, message
                else:
                    message = "Job submitted but could not extract job ID"
                    logger.warning(message)
                    return True, None, message
            else:
                message = f"Failed to submit job: {result.stderr}"
                logger.error(message)
                return False, None, message

        except FileNotFoundError:
            message = f"Script file not found: {script_path}"
            logger.error(message)
            return False, None, message
        except Exception as e:
            message = f"Error submitting job: {str(e)}"
            logger.error(message)
            return False, None, message

    def execute(self, code: str) -> Tuple[bool, str, str, dict]:
        """
        Execute code with observer notifications.

        Args:
            code: The code to execute

        Returns:
            Tuple of (success, stdout, stderr, observer_results)
            - success: Whether execution completed successfully
            - stdout: Complete stdout output
            - stderr: Complete stderr output
            - observer_results: Dictionary mapping observer names to their results
        """
        try:
            # Rewrite backend-specific addresses in bash scripts
            if self.language.lower() == "bash" and self.compute_backend:
                try:
                    code = self.compute_backend.prepare_script(code)
                    logger.info(
                        f"Prepared script for {self.compute_backend.backend_type} backend ({self.compute_backend.address})"
                    )
                except Exception as e:
                    logger.debug(f"Could not prepare script for backend: {e}")

            # Set up the command based on language
            if self.language.lower() == "python":
                cmd = ["python", "-c", code]
            elif self.language.lower() == "bash":
                cmd = ["bash", "-c", code]
            else:
                return False, "", f"Unsupported language: {self.language}", {}

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                preexec_fn=os.setsid,
            )

            self._last_process = process
            stdout_chunks, stderr_chunks = [], []
            self._last_stdout_chunks = stdout_chunks  # shared reference

            # Set up tracking of both output streams
            streams = [process.stdout, process.stderr]

            # Track start time for timeout
            start_time = time.time()

            with Progress(
                TextColumn(f"[bold cyan]Executing {self.language}:"),
                TextColumn(
                    "[bold green]{task.completed:.1f}s[/bold green] [dim](time limit: {task.total:.0f}s)[/dim]"
                ),
                refresh_per_second=2,
                transient=False,
                disable=not show_progress_bar(),
            ) as progress_context:

                task = progress_context.add_task("", total=self.timeout)

                while streams:
                    # Calculate remaining time
                    elapsed_time = time.time() - start_time
                    progress_context.update(task, completed=elapsed_time)
                    remaining_time = max(0, self.timeout - elapsed_time)

                    # Check if we've exceeded timeout
                    if remaining_time <= 0:
                        self._terminate(process, stdout_chunks)
                        stdout_chunks.append(f"\nProcess reached time limit after {self.timeout} seconds.\n")
                        logger.info(f"\nProcess reached time limit after {self.timeout} seconds.\n")
                        break

                    # Check if any observer signals termination
                    should_stop, stop_reason = self._check_observers_for_termination()
                    if should_stop:
                        logger.brief(f"Observer requested termination: {stop_reason}")
                        self._terminate(process, stdout_chunks)
                        stdout_chunks.append(f"\nExecution terminated by observer: {stop_reason}\n")
                        break

                    # Wait for output on either stream with timeout
                    readable, _, _ = select.select(streams, [], [], min(1, remaining_time))

                    # If nothing was read but process is still running, continue the loop
                    if not readable and process.poll() is None:
                        continue

                    # If nothing was read and process exited, exit loop
                    if not readable and process.poll() is not None:
                        break

                    for stream in readable:
                        line = stream.readline()
                        if not line:  # EOF
                            streams.remove(stream)
                            continue

                        # Handle stdout
                        if stream == process.stdout:
                            stdout_chunks.append(line)
                            logger.detail(line.rstrip())
                            # Notify observers
                            self._notify_observers_stdout(line)

                        # Handle stderr
                        else:
                            stderr_chunks.append(line)
                            logger.detail(line.rstrip())
                            # Notify observers
                            self._notify_observers_stderr(line)

                elapsed_time = time.time() - start_time
                progress_context.update(task, completed=elapsed_time)

            # Wait for process to complete (should already be done, but just in case)
            if process.poll() is None:
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    stderr_chunks.append("Process forcibly terminated after timeout\n")

            success = process.returncode == 0
            stdout = "".join(stdout_chunks)
            stderr = "".join(stderr_chunks)

            # Notify observers that execution is complete
            self._notify_observers_complete(success, stdout, stderr)

            # Collect observer results
            observer_results = self._collect_observer_results()

            return success, stdout, stderr, observer_results

        except Exception as e:
            return False, "", f"Error executing {self.language} code: {str(e)}", {}

    def _notify_observers_stdout(self, line: str) -> None:
        """Notify all observers of a stdout line."""
        for observer in self.observers:
            try:
                observer.on_stdout_line(line)
            except Exception as e:
                logger.error(f"Observer {observer.name} error on stdout: {e}")

    def _notify_observers_stderr(self, line: str) -> None:
        """Notify all observers of a stderr line."""
        for observer in self.observers:
            try:
                observer.on_stderr_line(line)
            except Exception as e:
                logger.error(f"Observer {observer.name} error on stderr: {e}")

    def _notify_observers_complete(self, success: bool, stdout: str, stderr: str) -> None:
        """Notify all observers that execution is complete."""
        for observer in self.observers:
            try:
                observer.on_execution_complete(success, stdout, stderr)
            except Exception as e:
                logger.error(f"Observer {observer.name} error on complete: {e}")

    def _check_observers_for_termination(self) -> Tuple[bool, Optional[str]]:
        """
        Check if any observer signals termination.

        Returns:
            Tuple of (should_stop, reason)
        """
        for observer in self.observers:
            try:
                should_stop, reason = observer.should_terminate()
                if should_stop:
                    return True, reason or f"Observer {observer.name} requested stop"
            except Exception as e:
                logger.error(f"Observer {observer.name} error checking termination: {e}")
        return False, None

    def _terminate(self, process: subprocess.Popen, stdout_chunks: list) -> None:
        """Terminate a running job, delegating to the compute backend."""
        self.compute_backend.terminate(pid=process.pid, stdout_so_far="".join(stdout_chunks))

    def _collect_observer_results(self) -> dict:
        """
        Collect results from all observers.

        Returns:
            Dictionary mapping observer names to their results
        """
        results = {}
        for observer in self.observers:
            try:
                results[observer.name] = observer.get_result()
            except Exception as e:
                logger.error(f"Observer {observer.name} error getting result: {e}")
                results[observer.name] = {"error": str(e)}
        return results
