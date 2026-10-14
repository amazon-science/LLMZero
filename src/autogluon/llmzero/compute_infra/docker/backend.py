"""Docker container compute backend for LLMZero.

Runs training jobs inside a Docker container on the local machine.
Supports images from local builds, Docker Hub, and Amazon ECR.
"""

import logging
import re
import select
import subprocess
from typing import Dict, List, Optional

from ..base import BaseComputeBackend, ComputeJobResult

logger = logging.getLogger(__name__)


class DockerBackend(BaseComputeBackend):
    """Compute backend that runs jobs inside a Docker container.

    Parameters
    ----------
    image : str
        Docker image to use (e.g. ``"my-training:latest"``,
        ``"123456789.dkr.ecr.us-east-1.amazonaws.com/repo:tag"``).
    gpu : str, optional
        GPU device request passed to ``docker run --gpus``.
        Common values: ``"all"``, ``"'\"device=0,1\"'"``.
        Default ``"all"``.
    runtime : str, optional
        Docker runtime (e.g. ``"nvidia"``).  If *None*, not passed.
    volumes : list[str], optional
        Explicit volume mount strings (``"host:container[:opts]"``).
        E.g. ``["/data:/workspace/data", "/fsx:/fsx:ro"]``.
    mount_dirs : list[str], optional
        List of host directories to **identity-mount** (``-v /path:/path``)
        so that host paths work unchanged inside the container.
        E.g. ``["/fsx", "/home/ubuntu/runs"]``.
        This is the recommended way to give the container access to data,
        models, reward functions, and output directories.
    shm_size : str, optional
        Shared-memory size (e.g. ``"16g"``).  Default ``"16g"``.
    network : str, optional
        Docker network mode.  Default ``"host"``.
    extra_args : list[str], optional
        Any extra arguments passed verbatim to ``docker run``.
    ecr_auto_login : bool, optional
        If *True* and the image looks like an ECR URI, run
        ``aws ecr get-login-password | docker login`` before pulling.
        Default *True*.
    """

    def __init__(
        self,
        image: str,
        gpu: str = "all",
        runtime: Optional[str] = None,
        volumes: Optional[List[str]] = None,
        mount_dirs: Optional[List[str]] = None,
        shm_size: str = "16g",
        network: str = "host",
        extra_args: Optional[List[str]] = None,
        ecr_auto_login: bool = True,
    ):
        self.image = image
        self.gpu = gpu
        self.runtime = runtime
        self.shm_size = shm_size
        self.network = network
        self.extra_args = extra_args or []
        self.ecr_auto_login = ecr_auto_login
        self._last_job_id: Optional[str] = None  # stores container ID

        # Build the final volume list: explicit volumes + identity mounts
        self.volumes: List[str] = list(volumes or [])
        for d in mount_dirs or []:
            mount = f"{d}:{d}"
            if mount not in self.volumes and f"{d}:{d}:" not in " ".join(self.volumes):
                self.volumes.append(mount)

        logger.info(f"Initialized DockerBackend with image={self.image}, " f"{len(self.volumes)} volume mount(s)")

    # -- Properties ----------------------------------------------------------

    @property
    def address(self) -> str:
        return f"docker ({self.image})"

    @property
    def backend_type(self) -> str:
        return "docker"

    # -- Script manipulation -------------------------------------------------

    def prepare_script(self, script_content: str) -> str:
        """Wrap the script to run inside a docker container.

        Strips any ``ray job submit`` wrapper if present, then wraps
        the inner commands in a ``docker run`` invocation.
        """
        inner = self._strip_ray_submit_wrapper(script_content)

        # Build docker run command
        docker_cmd = self._build_docker_run_cmd(inner)

        wrapped = "#!/bin/bash\nset -e\n\n"

        # ECR login if needed
        if self.ecr_auto_login and self._is_ecr_image(self.image):
            region = self._ecr_region(self.image)
            registry = self.image.split("/")[0]
            wrapped += (
                "# ECR login\n"
                f"aws ecr get-login-password --region {region} | "
                f"docker login --username AWS --password-stdin {registry}\n\n"
            )

        wrapped += docker_cmd + "\n"
        logger.debug(f"Prepared script for docker execution with image '{self.image}'")
        return wrapped

    def _build_docker_run_cmd(self, script_body: str) -> str:
        """Build the ``docker run`` command string."""
        parts = ["docker run --rm"]

        if self.gpu:
            parts.append(f"--gpus {self.gpu}")
        if self.runtime:
            parts.append(f"--runtime {self.runtime}")
        if self.shm_size:
            parts.append(f"--shm-size {self.shm_size}")
        if self.network:
            parts.append(f"--network {self.network}")

        # Always add ipc=host and ulimits for PyTorch distributed training
        parts.append("--ipc=host")
        parts.append("--ulimit memlock=-1")
        parts.append("--ulimit stack=67108864")

        for vol in self.volumes:
            parts.append(f"-v {vol}")

        parts.extend(self.extra_args)
        parts.append(self.image)

        # Escape single quotes in the script body for safe embedding
        escaped = script_body.replace("'", "'\"'\"'")
        parts.append(f"bash -c '{escaped}'")

        return " \\\n  ".join(parts)

    # -- Job lifecycle -------------------------------------------------------

    def submit_job(self, script_content: str) -> ComputeJobResult:
        """Run *script_content* inside a Docker container.

        Only used by ``llmzero submit-script`` (CLI).  The MCTS loop uses
        ``CodeExecutor.execute()`` which calls ``prepare_script()`` and
        manages its own subprocess with observers and process-group cleanup.
        """
        logger.info(f"Submitting Docker job with image '{self.image}'")

        modified_script = self.prepare_script(script_content)
        result = self._execute_script(modified_script)

        if result.returncode == 0:
            logger.info("Docker job completed successfully")

        return result

    def _stop_remote_job(self, job_id: str) -> bool:
        """Stop a Docker container by ID or name."""
        logger.info(f"Stopping Docker container: {job_id}")
        try:
            result = subprocess.run(
                ["docker", "stop", job_id],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                logger.info(f"Successfully stopped container: {job_id}")
                return True
            else:
                logger.warning(f"Failed to stop container {job_id}: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error stopping Docker container {job_id}: {e}")
            return False

    def cleanup(self) -> bool:
        """Remove stopped containers from this image."""
        logger.info(f"Cleaning up stopped containers for image {self.image}")
        try:
            result = subprocess.run(
                ["docker", "container", "prune", "-f", "--filter", f"ancestor={self.image}"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                logger.info("Docker cleanup completed")
                return True
            else:
                logger.warning(f"Docker cleanup failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error during Docker cleanup: {e}")
            return False

    # -- Factory & template --------------------------------------------------

    _CLI_ARG_MAP = {
        "image": "--docker-image",
    }

    @classmethod
    def validate_config(cls, config_dict: Dict) -> None:
        required = {"image"}
        present = {k for k in config_dict if k != "type"}
        missing = required - present
        if missing:
            cli_hints = ", ".join(cls._CLI_ARG_MAP[k] for k in sorted(missing) if k in cls._CLI_ARG_MAP)
            raise ValueError(
                f"Incomplete docker backend configuration. Missing: {', '.join(sorted(missing))}. "
                f"Provide via CLI: {cli_hints}"
            )

    @classmethod
    def from_config(cls, config_dict: Dict) -> "DockerBackend":
        cls.validate_config(config_dict)
        return cls(
            image=config_dict["image"],
            gpu=config_dict.get("gpu", "all"),
            runtime=config_dict.get("runtime"),
            volumes=config_dict.get("volumes"),
            mount_dirs=config_dict.get("mount_dirs"),
            shm_size=config_dict.get("shm_size", "16g"),
            network=config_dict.get("network", "host"),
            extra_args=config_dict.get("extra_args"),
            ecr_auto_login=config_dict.get("ecr_auto_login", True),
        )

    @classmethod
    def execution_script_template(cls, config_dict: Optional[Dict] = None) -> str:
        """Return the docker-specific execution script skeleton for prompt injection."""
        return """Generate execution script structure:

```bash
#!/bin/bash
set -e

# This script runs inside a Docker container — the image and GPU setup are handled automatically.
# Host directories are mounted into the container at the same paths, so use absolute host paths directly.
# Simply run the training commands directly.

<copy all lines from entry point script here>
```

Requirements:
- Start with #!/bin/bash and set -e
- Do NOT use `ray job submit`. This runs inside a Docker container, not on a Ray cluster.
- Do NOT add docker run commands — the container is managed by the executor.
- Use absolute host paths for data, model, and output directories — they are identity-mounted into the container.
- Copy the entry point script content directly into this script.
- If "WandB Enabled" is True above, export the WandB API key:
  export WANDB_API_KEY="<use the WandB API Key value shown above>"
- If "Previous Run Error Analysis" section above has content, the previous run FAILED - focus on fixing execution/environment issues
- If "Previous Run Error Analysis" is empty, the previous run SUCCEEDED - maintain the working execution configuration"""

    # -- Internal helpers ----------------------------------------------------

    @staticmethod
    def _is_ecr_image(image: str) -> bool:
        """Check if an image URI looks like an ECR image."""
        return bool(re.match(r"\d+\.dkr\.ecr\.\S+\.amazonaws\.com/", image))

    @staticmethod
    def _ecr_region(image: str) -> str:
        """Extract the AWS region from an ECR image URI."""
        match = re.search(r"\.ecr\.([a-z0-9-]+)\.amazonaws\.com", image)
        return match.group(1) if match else "us-east-1"

    def _execute_script(self, script_content: str) -> ComputeJobResult:
        """Execute a bash script with streaming output."""
        logger.debug("Executing docker job script")

        process = subprocess.Popen(
            ["bash", "-c", script_content],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # Try to capture the container ID from the docker run output
        self._last_job_id = None

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
