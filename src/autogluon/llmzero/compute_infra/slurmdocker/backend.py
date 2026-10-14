"""Slurm + Docker (pyxis/enroot) compute backend for LLMZero.

Uses Slurm's pyxis plugin (``#SBATCH --container-image``) to run training
inside a Docker/enroot container on HyperPod compute nodes.  The container
image is specified as a ``.sqsh`` file or registry URI.

Combines the Slurm two-part wrapper pattern (sbatch + tail -f) with
container execution via pyxis directives.
"""

import logging
import os
import re
import select
import subprocess
from importlib import resources as importlib_resources
from typing import Dict, List, Optional

from ..base import BaseComputeBackend, ComputeJobResult

logger = logging.getLogger(__name__)

# Outer wrapper template (sbatch + tail -f streaming) lives alongside this backend
_TEMPLATE_DIR = importlib_resources.files("autogluon.llmzero.compute_infra.slurmdocker")
_OUTER_WRAPPER_TEMPLATE = (_TEMPLATE_DIR / "outer_wrapper_template.sh").read_text()


class SlurmDockerBackend(BaseComputeBackend):
    """Compute backend that submits containerized jobs to Slurm via pyxis.

    Uses ``#SBATCH --container-image`` and ``--container-mounts`` to run
    training commands inside a Docker container on Slurm compute nodes.

    Parameters
    ----------
    partition : str
        Slurm partition.
    container_image : str
        Path to ``.sqsh`` file or Docker registry URI.
    container_mounts : list[str]
        Mount specifications (e.g. ``["/data:/data", "/models:/models"]``).
    container_workdir : str, optional
        Working directory inside the container.
    num_nodes : int
        Number of nodes. Default ``1``.
    gpus_per_node : int
        GPUs per node. Default ``8``.
    ntasks_per_node : int
        Tasks per node. Default ``1``.
    time_limit : str
        Slurm time limit. Default ``"30-00:00:00"``.
    account : str, optional
        Slurm account.
    exclusive : bool
        Request exclusive access. Default ``True``.
    setup_ray : bool
        Bootstrap Ray cluster inside containers. Default ``True``.
    slurm_log_dir : str, optional
        Shared directory for sbatch scripts and logs.
    """

    def __init__(
        self,
        partition: str,
        container_image: str,
        container_mounts: Optional[List[str]] = None,
        container_workdir: Optional[str] = None,
        container_env: Optional[Dict[str, str]] = None,
        no_container_mount_home: bool = False,
        num_nodes: int = 1,
        gpus_per_node: int = 8,
        ntasks_per_node: int = 1,
        time_limit: str = "30-00:00:00",
        account: Optional[str] = None,
        exclusive: bool = True,
        slurm_log_dir: Optional[str] = None,
    ):
        self.partition = partition
        self.container_image = container_image
        self.container_mounts = container_mounts or []
        self.container_workdir = container_workdir
        self.container_env = container_env or {}
        self.no_container_mount_home = no_container_mount_home
        self.num_nodes = num_nodes
        self.gpus_per_node = gpus_per_node
        self.ntasks_per_node = ntasks_per_node
        self.time_limit = time_limit
        self.account = account
        self.exclusive = exclusive
        self.slurm_log_dir = slurm_log_dir or os.path.join(os.path.expanduser("~"), ".llmzero", "slurm_logs")
        self._last_job_id: Optional[str] = None
        self._last_process: Optional[subprocess.Popen] = None
        logger.info(
            f"Initialized SlurmDockerBackend: partition={self.partition}, "
            f"image={self.container_image}, nodes={self.num_nodes}"
        )

    @property
    def address(self) -> str:
        return f"slurmdocker://{self.partition}"

    @property
    def backend_type(self) -> str:
        return "slurmdocker"

    def prepare_script(self, script_content: str) -> str:
        """Produce a wrapper script that submits via sbatch with container and streams logs."""
        inner = self._strip_ray_submit_wrapper(script_content)
        sbatch_script = self._build_sbatch_script(inner)
        outer_wrapper = self._build_outer_wrapper(sbatch_script)
        logger.debug(f"Prepared SlurmDocker wrapper for partition '{self.partition}'")
        return outer_wrapper

    def extract_job_id(self, stdout: str) -> Optional[str]:
        match = re.search(r"Submitted batch job (\d+)", stdout)
        if match:
            job_id = match.group(1)
            logger.info(f"Extracted Slurm job ID: {job_id}")
            return job_id
        logger.warning("Could not extract Slurm job ID from output")
        return None

    def submit_job(self, script_content: str) -> ComputeJobResult:
        logger.info(f"Submitting SlurmDocker job to partition '{self.partition}'")
        modified_script = self.prepare_script(script_content)
        result = self._execute_script(modified_script)
        if result.returncode == 0:
            self._last_job_id = self.extract_job_id(result.stdout)
        return result

    def _stop_remote_job(self, job_id: str) -> bool:
        logger.info(f"Cancelling Slurm job: {job_id}")
        try:
            result = subprocess.run(["scancel", job_id], capture_output=True, text=True, check=False)
            if result.returncode == 0:
                logger.info(f"Successfully cancelled Slurm job: {job_id}")
                return True
            logger.warning(f"Failed to cancel Slurm job {job_id}: {result.stderr}")
            return False
        except Exception as e:
            logger.error(f"Error cancelling Slurm job {job_id}: {e}")
            return False

    def cleanup(self) -> bool:
        logger.info("SlurmDockerBackend cleanup: nothing to do")
        return True

    _CLI_ARG_MAP = {
        "partition": "--slurm-partition",
        "container_image": "--docker-image",
    }

    @classmethod
    def validate_config(cls, config_dict: Dict) -> None:
        required = {"partition", "container_image"}
        present = {k for k in config_dict if k != "type"}
        missing = required - present
        if missing:
            cli_hints = ", ".join(cls._CLI_ARG_MAP.get(k, k) for k in sorted(missing))
            raise ValueError(
                f"Incomplete slurmdocker backend configuration. "
                f"Missing: {', '.join(sorted(missing))}. "
                f"Provide via CLI: {cli_hints}"
            )

    @classmethod
    def from_config(cls, config_dict: Dict) -> "SlurmDockerBackend":
        cls.validate_config(config_dict)
        return cls(
            partition=config_dict["partition"],
            container_image=config_dict["container_image"],
            container_mounts=config_dict.get("container_mounts", []),
            container_workdir=config_dict.get("container_workdir"),
            container_env=config_dict.get("container_env", {}),
            no_container_mount_home=config_dict.get("no_container_mount_home", False),
            num_nodes=config_dict.get("num_nodes", 1),
            gpus_per_node=config_dict.get("gpus_per_node", 8),
            ntasks_per_node=config_dict.get("ntasks_per_node", 1),
            time_limit=config_dict.get("time_limit", "30-00:00:00"),
            account=config_dict.get("account"),
            exclusive=config_dict.get("exclusive", True),
            slurm_log_dir=config_dict.get("slurm_log_dir"),
        )

    @classmethod
    def execution_script_template(cls, config_dict: Optional[Dict] = None) -> str:
        return """Generate execution script structure:

```bash
#!/bin/bash
set -e

# This script runs INSIDE a Docker container on Slurm compute nodes.
# The container is already set up with the correct image and mounts.
# Do NOT add #SBATCH headers — they are added automatically.
# Do NOT add conda activate — packages are in the container.
# Do NOT set up Ray (ray start) — the Ray cluster is started automatically.
# Simply run the training commands directly.

<copy all lines from entry point script here>
```

Requirements:
- Start with #!/bin/bash and set -e
- Do NOT use `ray job submit`. Ray is managed by the Slurm+Docker wrapper.
- Do NOT add #SBATCH directives.
- Do NOT add conda/docker commands — you are already inside the container.
- Copy the entry point script content directly.
- If "Previous Run Error Analysis" section above has content, the previous run FAILED - fix it
- If "Previous Run Error Analysis" is empty, the previous run SUCCEEDED - maintain configuration"""

    def _build_sbatch_script(self, training_commands: str) -> str:
        """Build the inner sbatch script with pyxis container directives.

        Pyxis/enroot containers do NOT have Slurm CLI tools (scontrol, srun).
        Node discovery and Ray bootstrap are NOT done here — the training
        framework (verl) manages its own Ray cluster internally using
        trainer.nnodes and SLURM environment variables.
        """
        mounts_str = ",".join(self.container_mounts) if self.container_mounts else ""

        extra = ""
        if self.account:
            extra += f"#SBATCH --account={self.account}\n"
        if self.exclusive:
            extra += "#SBATCH --exclusive\n"

        container_directives = f"#SBATCH --container-image={self.container_image}\n"
        if mounts_str:
            container_directives += f"#SBATCH --container-mounts={mounts_str}\n"
        if self.container_workdir:
            container_directives += f"#SBATCH --container-workdir={self.container_workdir}\n"
        if self.no_container_mount_home:
            container_directives += "#SBATCH --no-container-mount-home\n"
        for env_key, env_val in self.container_env.items():
            container_directives += f"#SBATCH --container-env={env_key}={env_val}\n"

        ray_setup = self._ray_setup_snippet() if self.num_nodes > 1 else ""

        env_exports = ""
        for env_key, env_val in self.container_env.items():
            env_exports += f"export {env_key}={env_val}\n"

        script = f"""#!/bin/bash
#SBATCH --job-name=llmzero-docker-training
#SBATCH --partition={self.partition}
#SBATCH --nodes={self.num_nodes}
#SBATCH --ntasks-per-node={self.ntasks_per_node}
#SBATCH --gpus-per-node={self.gpus_per_node}
#SBATCH --time={self.time_limit}
{extra}{container_directives}#SBATCH --output=LOG_DIR_PLACEHOLDER/slurm_%j.out
#SBATCH --error=LOG_DIR_PLACEHOLDER/slurm_%j.err

set -euo pipefail

# =======================================================
# Environment setup (inside container)
# =======================================================
unset ROCR_VISIBLE_DEVICES
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_SYMMETRIC_MEMORY=0
export TORCH_SYMMETRIC_MEMORY=0
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VLLM_USE_NCCL_SYMM_MEM=0
export VLLM_USE_TORCH_SYMMETRIC_MEMORY=0
{env_exports}

echo "Node: $(hostname)"
echo "SLURM_JOB_NODELIST: ${{SLURM_JOB_NODELIST:-unset}}"
echo "SLURM_JOB_NUM_NODES: ${{SLURM_JOB_NUM_NODES:-unset}}"
{ray_setup}
# =======================================================
# Training commands
# =======================================================
{training_commands}
"""
        return script

    def _build_outer_wrapper(self, sbatch_script: str) -> str:
        return _OUTER_WRAPPER_TEMPLATE.format(
            slurm_log_dir=self.slurm_log_dir,
            sbatch_script_content=sbatch_script,
        )

    def _ray_setup_snippet(self) -> str:
        """Ray cluster bootstrap for multi-node pyxis containers.

        Pyxis containers do NOT have scontrol/srun. Instead:
        - Parse SLURM_JOB_NODELIST with Python regex (no Slurm CLI)
        - Use ssh to start Ray workers on other nodes (pyxis shares /root/.ssh)
        - If ssh unavailable, fall back to single-node Ray with all GPUs
        """
        return f"""
# =======================================================
# Ray cluster setup (multi-node, inside container)
# =======================================================
# Parse SLURM_JOB_NODELIST without scontrol (not available in container)
IFS=',' read -ra NODE_LIST <<< "$(python3 -c "
import os, re
nodelist = os.environ.get('SLURM_JOB_NODELIST', '')
# Handle formats: ip-10-4-204-[72,179] or ip-10-4-204-72,ip-10-4-204-179
# Slurm compressed format: prefix-[suffix1,suffix2,...]
m = re.match(r'(.+-)\\[(.*?)\\]', nodelist)
if m:
    prefix = m.group(1)
    # Handle ranges like 72,179 or 1-3 or 1,3-5
    parts = []
    for item in m.group(2).split(','):
        if '-' in item:
            start, end = item.split('-', 1)
            parts.extend(str(i) for i in range(int(start), int(end)+1))
        else:
            parts.append(item)
    print(','.join(f'{{prefix}}{{p}}' for p in parts))
elif ',' in nodelist:
    print(nodelist)
else:
    print(nodelist)
")"

if [[ ${{#NODE_LIST[@]}} -eq 0 || -z "${{NODE_LIST[0]}}" ]]; then
    echo "[WARN] Could not parse SLURM_JOB_NODELIST=${{SLURM_JOB_NODELIST:-}}, using single-node" >&2
    NODE_LIST=("$(hostname)")
fi

head_node=${{NODE_LIST[0]}}
echo "Discovered nodes (${{#NODE_LIST[@]}}): ${{NODE_LIST[*]}}"
echo "Head node: $head_node"

# Get head node IP - try multiple methods
head_node_ip=$(python3 -c "
import socket, os
node = '$head_node'
try:
    ip = socket.gethostbyname(node)
    print(ip)
except:
    import subprocess
    r = subprocess.run(['hostname', '-I'], capture_output=True, text=True)
    ips = r.stdout.strip().split()
    for ip in ips:
        if '.' in ip and not ip.startswith('127.'):
            print(ip)
            break
")
echo "Head node IP: $head_node_ip"

port=${{RAY_HEAD_PORT:-6379}}
ip_head="${{head_node_ip}}:${{port}}"
export ip_head
export MASTER_ADDR=${{head_node_ip}}
export MASTER_PORT=${{port}}

# Cleanup trap
cleanup_ray() {{
    ray stop --force >/dev/null 2>&1 || true
}}
trap cleanup_ray EXIT SIGINT SIGTERM

# Stop any existing Ray
ray stop --force >/dev/null 2>&1 || true
sleep 2

# Start Ray head on this node
echo "Starting Ray HEAD on $head_node ($head_node_ip)"
ray start --head \\
    --node-ip-address="$head_node_ip" \\
    --port=$port \\
    --num-gpus={self.gpus_per_node}
sleep 8

# Start Ray workers on other nodes via ssh
worker_num=$((${{#NODE_LIST[@]}} - 1))
if [[ $worker_num -gt 0 ]]; then
    for ((i = 1; i <= worker_num; i++)); do
        node_i=${{NODE_LIST[$i]}}
        echo "Starting Ray WORKER $i on $node_i"
        ssh -o StrictHostKeyChecking=no "$node_i" \\
            "ray stop --force 2>/dev/null; ray start --address '$ip_head' --num-gpus={self.gpus_per_node}" \\
            || echo "[WARN] SSH to $node_i failed, worker not started"
        sleep 5
    done
fi
sleep 10
echo "Ray cluster ready with ${{#NODE_LIST[@]}} nodes"
ray status
"""

    def _execute_script(self, script_content: str) -> ComputeJobResult:
        """Execute the bash wrapper locally with streaming output."""
        logger.debug("Executing SlurmDocker wrapper script")

        process = subprocess.Popen(
            ["bash", "-c", script_content],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid,
        )

        self._last_process = process
        self._last_job_id = None

        stdout_chunks: List[str] = []
        stderr_chunks: List[str] = []
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
                    if self._last_job_id is None:
                        job_id = self.extract_job_id(line)
                        if job_id:
                            self._last_job_id = job_id
                else:
                    stderr_chunks.append(line)
                    print(line.rstrip())

        process.wait()

        return ComputeJobResult(
            returncode=process.returncode,
            stdout="".join(stdout_chunks),
            stderr="".join(stderr_chunks),
        )
