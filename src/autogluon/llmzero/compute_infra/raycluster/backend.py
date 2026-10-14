"""Ray cluster compute backend for LLMZero."""

import logging
import re
import select
import subprocess
from typing import Dict, Optional

from ..base import BaseComputeBackend, ComputeJobResult

logger = logging.getLogger(__name__)


class RayClusterBackend(BaseComputeBackend):
    """Compute backend targeting a Ray cluster reachable via Kubernetes service DNS.

    Parameters
    ----------
    kubernetes_namespace : str
        Kubernetes namespace where the Ray cluster is deployed.
    ray_service_name : str
        Name of the Ray head service (used to build the DNS address).
    ray_dashboard_port : int, optional
        Port for the Ray dashboard (default 8265).
    """

    def __init__(
        self,
        kubernetes_namespace: str,
        ray_service_name: str,
        ray_dashboard_port: int = 8265,
        entrypoint_num_gpus: Optional[int] = None,
    ):
        self.kubernetes_namespace = kubernetes_namespace
        self.ray_service_name = ray_service_name
        self.ray_dashboard_port = ray_dashboard_port
        self.entrypoint_num_gpus = entrypoint_num_gpus
        self._last_job_id: Optional[str] = None
        logger.info(f"Initialized RayClusterBackend with address: {self.address}")

    # -- Properties ----------------------------------------------------------

    @property
    def address(self) -> str:
        return (
            f"http://{self.ray_service_name}.{self.kubernetes_namespace}"
            f".svc.cluster.local:{self.ray_dashboard_port}"
        )

    @property
    def backend_type(self) -> str:
        return "raycluster"

    # -- Script manipulation -------------------------------------------------

    def prepare_script(self, script_content: str) -> str:
        """Rewrite localhost / 127.0.0.1 Ray addresses to the K8s service DNS."""
        patterns = [
            (r"--address\s+http://localhost:\d+", f"--address {self.address}"),
            (r"--address\s+http://127\.0\.0\.1:\d+", f"--address {self.address}"),
            (r"RAY_ADDRESS=http://localhost:\d+", f"RAY_ADDRESS={self.address}"),
            (r"RAY_ADDRESS=http://127\.0\.0\.1:\d+", f"RAY_ADDRESS={self.address}"),
        ]
        modified = script_content
        for pattern, replacement in patterns:
            modified = re.sub(pattern, replacement, modified)
        logger.debug(f"Updated Ray address in script to {self.address}")
        return modified

    def extract_job_id(self, stdout: str) -> Optional[str]:
        """Extract a ``raysubmit_*`` job ID from *stdout*."""
        match = re.search(r"raysubmit_[A-Za-z0-9]+", stdout)
        if match:
            job_id = match.group(0)
            logger.info(f"Extracted job ID: {job_id}")
            return job_id
        logger.warning("Could not extract job ID from output")
        return None

    # -- Job lifecycle -------------------------------------------------------

    def submit_job(self, script_content: str) -> ComputeJobResult:
        """Submit a Ray job via ``bash -c`` with streaming output.

        Only used by ``llmzero submit-script`` (CLI).  The MCTS loop uses
        ``CodeExecutor.execute()`` which calls ``prepare_script()`` and
        manages its own subprocess with observers and process-group cleanup.
        """
        logger.info(f"Submitting Ray job to {self.address}")

        modified_script = self.prepare_script(script_content)
        result = self._execute_script(modified_script)

        if result.returncode == 0:
            self._last_job_id = self.extract_job_id(result.stdout)

        return result

    def _stop_remote_job(self, job_id: str) -> bool:
        """Stop a specific Ray job on the cluster via ``ray job stop``."""
        logger.info(f"Stopping remote Ray job: {job_id}")
        try:
            stop_result = subprocess.run(
                ["ray", "job", "stop", "--address", self.address, job_id],
                capture_output=True,
                text=True,
                check=False,
            )
            if stop_result.returncode == 0:
                logger.info(f"Successfully stopped job: {job_id}")
                return True
            else:
                logger.warning(f"Failed to stop job {job_id}: {stop_result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error stopping Ray job {job_id}: {e}")
            return False

    def cleanup(self) -> bool:
        """Clear ``/tmp/ray`` on all Ray worker pods in the namespace."""
        logger.info("Clearing Ray cache from worker pods")
        try:
            list_pods_result = subprocess.run(
                [
                    "kubectl",
                    "get",
                    "pods",
                    "-n",
                    self.kubernetes_namespace,
                    "-l",
                    "ray.io/node-type=worker",
                    "-o",
                    "jsonpath={.items[*].metadata.name}",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if list_pods_result.returncode != 0:
                logger.warning(f"Failed to list worker pods: {list_pods_result.stderr}")
                return False

            worker_pods = list_pods_result.stdout.strip().split()
            if not worker_pods:
                logger.info("No worker pods found")
                return True

            logger.info(f"Found {len(worker_pods)} worker pods: {worker_pods}")
            success = True
            for pod_name in worker_pods:
                logger.info(f"Clearing Ray cache on pod: {pod_name}")
                clear_result = subprocess.run(
                    [
                        "kubectl",
                        "exec",
                        "-n",
                        self.kubernetes_namespace,
                        pod_name,
                        "--",
                        "rm",
                        "-rf",
                        "/tmp/ray",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if clear_result.returncode == 0:
                    logger.info(f"Successfully cleared cache on pod: {pod_name}")
                else:
                    logger.warning(f"Failed to clear cache on pod {pod_name}: {clear_result.stderr}")
                    success = False
            return success
        except Exception as e:
            logger.error(f"Error clearing Ray cache: {e}")
            return False

    # -- Factory & template --------------------------------------------------

    # -- Config validation & factory -----------------------------------------

    # Maps config keys to their corresponding CLI arguments for error messages
    _CLI_ARG_MAP = {
        "kubernetes_namespace": "--ray-namespace",
        "ray_service_name": "--ray-service",
        "ray_dashboard_port": "--ray-port",
    }

    @classmethod
    def validate_config(cls, config_dict: Dict) -> None:
        required = {"kubernetes_namespace", "ray_service_name"}
        present = {k for k in config_dict if k != "type"}
        missing = required - present
        if missing:
            cli_hints = ", ".join(cls._CLI_ARG_MAP[k] for k in sorted(missing) if k in cls._CLI_ARG_MAP)
            raise ValueError(
                f"Incomplete raycluster configuration. Missing: {', '.join(sorted(missing))}. "
                f"Add to config file or provide via CLI: {cli_hints}"
            )

    @classmethod
    def from_config(cls, config_dict: Dict) -> "RayClusterBackend":
        cls.validate_config(config_dict)
        return cls(
            kubernetes_namespace=config_dict["kubernetes_namespace"],
            ray_service_name=config_dict["ray_service_name"],
            ray_dashboard_port=config_dict.get("ray_dashboard_port", 8265),
            entrypoint_num_gpus=config_dict.get("entrypoint_num_gpus"),
        )

    @classmethod
    def execution_script_template(cls, config_dict: Optional[Dict] = None) -> str:
        """Return the Ray-specific execution script skeleton for prompt injection."""
        config_dict = config_dict or {}
        entrypoint_num_gpus = config_dict.get("entrypoint_num_gpus")

        if entrypoint_num_gpus:
            gpu_line = "  --entrypoint-num-gpus " + str(entrypoint_num_gpus) + " \\\n"
        else:
            gpu_line = ""

        return (
            """Generate execution script structure:

```bash
#!/bin/bash
set -e

ray job submit \\
  --address http://localhost:8265 \\
  --runtime-env-json '<runtime_env_json_here>' \\
"""
            + gpu_line
            + """  -- \\
  bash -lc '
<copy all lines from entry point script here>
'
```

Requirements:
- Start with #!/bin/bash and set -e
- Embed complete entry point script inside bash -lc '...'
- Use single quotes to wrap the script
- This will be executed by run.sh (environment already set up)
- For runtime-env-json:
  * If "WandB Enabled" is True above, use: '{{"env_vars":{{"WANDB_API_KEY":"<use the WandB API Key value shown above>"}}}}'
  * If "WandB Enabled" is False above, use: '{{"env_vars":{{}}}}'
- If "Previous Run Error Analysis" section above has content, the previous run FAILED - focus on fixing execution/environment issues
- If "Previous Run Error Analysis" is empty, the previous run SUCCEEDED - maintain the working execution configuration"""
        )

    # -- Internal helpers ----------------------------------------------------

    def _execute_script(self, script_content: str) -> ComputeJobResult:
        """Execute a bash script with streaming output."""
        logger.debug("Executing ray job submit script")

        process = subprocess.Popen(
            ["bash", "-c", script_content],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

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
