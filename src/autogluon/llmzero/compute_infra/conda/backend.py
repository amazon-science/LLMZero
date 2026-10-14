"""Conda environment compute backend for LLMZero.

Runs training jobs on the local machine inside a specified conda environment
instead of submitting to a remote Ray cluster.
"""

import logging
import os
import select
import subprocess
from typing import Dict, Optional

from ..base import BaseComputeBackend, ComputeJobResult

logger = logging.getLogger(__name__)


class CondaBackend(BaseComputeBackend):
    """Compute backend that runs jobs locally in a conda environment.

    Parameters
    ----------
    conda_env : str
        Name of the conda environment to activate for training jobs.
    num_gpus : int, optional
        Number of GPUs available on the local machine (informational, for prompt).
        Default 0 (auto-detect not performed; user should specify).
    """

    def __init__(self, conda_env: str, num_gpus: int = 0):
        self.conda_env = conda_env
        self.num_gpus = num_gpus
        self._last_job_id: Optional[str] = None  # stores PID as string
        self._last_process: Optional[subprocess.Popen] = None
        logger.info(f"Initialized CondaBackend with conda_env={self.conda_env}, num_gpus={self.num_gpus}")

    # -- Properties ----------------------------------------------------------

    @property
    def address(self) -> str:
        return f"local (conda: {self.conda_env})"

    @property
    def backend_type(self) -> str:
        return "conda"

    # -- Script manipulation -------------------------------------------------

    def prepare_script(self, script_content: str) -> str:
        """Wrap the script so it runs inside the target conda environment.

        Strips any ``ray job submit`` wrapper if present (e.g. from a
        previously generated script) and ensures the commands execute
        under the conda env.
        """
        # Strip ray job submit wrapper if present — extract the inner command
        inner = self._strip_ray_submit_wrapper(script_content)

        # Build a script that activates conda and runs the inner commands
        wrapped = (
            "#!/bin/bash\n"
            "set -e\n\n"
            "# Activate conda environment\n"
            f'eval "$(conda shell.bash hook)"\n'
            f"conda activate {self.conda_env}\n\n"
            "# Ensure conda env libraries take precedence over system libs\n"
            'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"\n\n'
            f"{inner}"
        )
        logger.debug(f"Prepared script for local execution in conda env '{self.conda_env}'")
        return wrapped

    # -- Job lifecycle -------------------------------------------------------

    def submit_job(self, script_content: str) -> ComputeJobResult:
        """Run *script_content* locally inside the conda environment.

        Only used by ``llmzero submit-script`` (CLI).  The MCTS loop uses
        ``CodeExecutor.execute()`` which calls ``prepare_script()`` and
        manages its own subprocess with observers and process-group cleanup.
        """
        logger.info(f"Submitting local job in conda env '{self.conda_env}'")

        modified_script = self.prepare_script(script_content)
        result = self._execute_script(modified_script)

        if result.returncode == 0:
            logger.info("Local conda job completed successfully")

        return result

    def _stop_remote_job(self, job_id: str) -> bool:
        """Stop a local process by PID.

        For conda, the "job_id" from the CLI is a PID string.
        Kill its process group so the training job is fully stopped.
        """
        try:
            pid = int(job_id)
            self._kill_process_group(pid)
            return True
        except ValueError:
            logger.error(f"Invalid PID for conda backend: {job_id}")
            return False
        except (ProcessLookupError, PermissionError) as e:
            logger.warning(f"Could not stop process {job_id}: {e}")
            return False

    def cleanup(self) -> bool:
        """No-op for conda backend — nothing to clean up."""
        logger.info("CondaBackend cleanup: nothing to do")
        return True

    # -- Factory & template --------------------------------------------------

    _CLI_ARG_MAP = {
        "conda_env": "--conda-env",
    }

    @classmethod
    def validate_config(cls, config_dict: Dict) -> None:
        required = {"conda_env"}
        present = {k for k in config_dict if k != "type"}
        missing = required - present
        if missing:
            cli_hints = ", ".join(cls._CLI_ARG_MAP[k] for k in sorted(missing) if k in cls._CLI_ARG_MAP)
            raise ValueError(
                f"Incomplete conda backend configuration. Missing: {', '.join(sorted(missing))}. "
                f"Provide via CLI: {cli_hints}"
            )

    @classmethod
    def from_config(cls, config_dict: Dict) -> "CondaBackend":
        cls.validate_config(config_dict)
        return cls(
            conda_env=config_dict["conda_env"],
            num_gpus=config_dict.get("num_gpus", 0),
        )

    @classmethod
    def execution_script_template(cls, config_dict: Optional[Dict] = None) -> str:
        """Return the conda-specific execution script skeleton for prompt injection."""
        return """Generate execution script structure:

```bash
#!/bin/bash
set -e

# The conda environment will be activated automatically — do NOT add conda activate commands.
# Simply run the training commands directly.

<copy all lines from entry point script here>
```

Requirements:
- Start with #!/bin/bash and set -e
- Do NOT use `ray job submit`. This runs locally, not on a Ray cluster.
- Do NOT add conda activate commands — the environment is already activated by the executor.
- Copy the entry point script content directly into this script.
- If "WandB Enabled" is True above, export the WandB API key:
  export WANDB_API_KEY="<use the WandB API Key value shown above>"
- If "Previous Run Error Analysis" section above has content, the previous run FAILED - focus on fixing execution/environment issues
- If "Previous Run Error Analysis" is empty, the previous run SUCCEEDED - maintain the working execution configuration"""

    # -- Internal helpers ----------------------------------------------------

    def _execute_script(self, script_content: str) -> ComputeJobResult:
        """Execute a bash script locally with streaming output."""
        logger.debug("Executing local conda job script")

        process = subprocess.Popen(
            ["bash", "-c", script_content],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid,
        )

        self._last_process = process
        self._last_job_id = str(process.pid)
        logger.info(f"Local job started with PID {process.pid}")

        stdout_chunks, stderr_chunks = [], []
        streams = [process.stdout, process.stderr]

        while streams:
            readable, _, _ = select.select(streams, [], [], 1)

            if not readable and process.poll() is not None:
                break

            for stream in readable:
                line = stream.readline()
                if not line:
                    streams.remove(stream)
                    continue
                if stream == process.stdout:
                    stdout_chunks.append(line)
                    print(line.rstrip())
                else:
                    stderr_chunks.append(line)
                    print(line.rstrip())

        process.wait()

        return ComputeJobResult(
            returncode=process.returncode,
            stdout="".join(stdout_chunks),
            stderr="".join(stderr_chunks),
        )
