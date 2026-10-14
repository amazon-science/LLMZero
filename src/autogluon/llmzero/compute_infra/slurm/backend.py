"""Slurm HyperPod cluster compute backend for LLMZero.

Submits training jobs to a Slurm cluster via ``sbatch`` and streams the job's
stdout/stderr back to the calling process in real-time using ``tail -f``.
This satisfies the ``CodeExecutor`` contract that the submission command blocks
and streams output so that observers (metrics, early stopping, checkpoints)
can parse it line-by-line.
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

# Load bash templates once at import time
_TEMPLATE_DIR = importlib_resources.files(__package__)
_SBATCH_TEMPLATE = (_TEMPLATE_DIR / "sbatch_template.sh").read_text()
_OUTER_WRAPPER_TEMPLATE = (_TEMPLATE_DIR / "outer_wrapper_template.sh").read_text()


class SlurmBackend(BaseComputeBackend):
    """Compute backend that submits jobs to a Slurm HyperPod cluster.

    The backend produces a two-part wrapper script:

    1. **Inner sbatch script** — written to a temp file on the shared
       filesystem, containing ``#SBATCH`` headers, conda/CUDA setup,
       Ray cluster bootstrap, and the training commands.
    2. **Outer wrapper** — the script that ``bash -c`` actually runs.
       It ``sbatch``-submits the inner script, then ``tail -f``s the
       log file while polling ``squeue`` for completion.  This blocks
       until the Slurm job finishes and streams its output to
       stdout/stderr so that ``CodeExecutor`` observers work.

    Parameters
    ----------
    partition : str
        Slurm partition (e.g. ``"<slurm-partition>"``).
    num_nodes : int
        Number of nodes to request.  Default ``1``.
    gpus_per_node : int
        GPUs per node.  Default ``8``.
    ntasks_per_node : int
        Tasks per node.  Default ``1``.
    time_limit : str
        Slurm ``--time`` value.  Default ``"30-00:00:00"``.
    conda_env : str, optional
        Conda environment to activate on compute nodes.
    conda_base : str
        Path to the conda installation.  Default ``"/opt/conda"``.
    cuda_version : str
        CUDA toolkit version.  Default ``"12.8"``.
    account : str, optional
        Slurm account (``--account``).
    qos : str, optional
        Slurm QOS (``--qos``).
    exclusive : bool
        Request exclusive node access.  Default ``True``.
    cache_base : str, optional
        Base directory for cache files (XDG, Torch, Triton).
    job_name_prefix : str
        Prefix for Slurm job names.  Default ``"llmzero"``.
    setup_ray : bool
        Whether to bootstrap a Ray cluster inside the Slurm job.
        Default ``True``.
    slurm_log_dir : str, optional
        Directory on the **shared** filesystem for sbatch scripts and log
        files.  Must be accessible from both the login node and compute
        nodes.  Defaults to ``$HOME/.llmzero/slurm_logs``.
    """

    def __init__(
        self,
        partition: str,
        num_nodes: int = 1,
        gpus_per_node: int = 8,
        ntasks_per_node: int = 1,
        time_limit: str = "30-00:00:00",
        conda_env: Optional[str] = None,
        conda_base: str = "/opt/conda",
        cuda_version: str = "12.8",
        account: Optional[str] = None,
        qos: Optional[str] = None,
        exclusive: bool = True,
        cache_base: Optional[str] = None,
        job_name_prefix: str = "llmzero",
        setup_ray: bool = True,
        slurm_log_dir: Optional[str] = None,
    ):
        self.partition = partition
        self.num_nodes = num_nodes
        self.gpus_per_node = gpus_per_node
        self.ntasks_per_node = ntasks_per_node
        self.time_limit = time_limit
        self.conda_env = conda_env
        self.conda_base = conda_base
        self.cuda_version = cuda_version
        self.account = account
        self.qos = qos
        self.exclusive = exclusive
        self.cache_base = cache_base
        self.job_name_prefix = job_name_prefix
        self.setup_ray = setup_ray
        self.slurm_log_dir = slurm_log_dir or os.path.join(os.path.expanduser("~"), ".llmzero", "slurm_logs")
        self._last_job_id: Optional[str] = None
        self._last_process: Optional[subprocess.Popen] = None
        logger.info(
            f"Initialized SlurmBackend: partition={self.partition}, "
            f"nodes={self.num_nodes}, gpus_per_node={self.gpus_per_node}"
        )

    # -- Properties ----------------------------------------------------------

    @property
    def address(self) -> str:
        return f"slurm://{self.partition}"

    @property
    def backend_type(self) -> str:
        return "slurm"

    # -- Script manipulation -------------------------------------------------

    def prepare_script(self, script_content: str) -> str:
        """Produce a wrapper script that submits via sbatch and streams logs."""
        inner = self._strip_ray_submit_wrapper(script_content)
        sbatch_script = self._build_sbatch_script(inner)
        outer_wrapper = self._build_outer_wrapper(sbatch_script)
        logger.debug(f"Prepared Slurm wrapper script for partition '{self.partition}'")
        return outer_wrapper

    def extract_job_id(self, stdout: str) -> Optional[str]:
        """Extract a Slurm job ID from ``sbatch`` output."""
        match = re.search(r"Submitted batch job (\d+)", stdout)
        if match:
            job_id = match.group(1)
            logger.info(f"Extracted Slurm job ID: {job_id}")
            return job_id
        logger.warning("Could not extract Slurm job ID from output")
        return None

    # -- Job lifecycle -------------------------------------------------------

    def submit_job(self, script_content: str) -> ComputeJobResult:
        """Submit *script_content* to Slurm with streaming output.

        Only used by ``llmzero submit-script`` (CLI).  The MCTS loop uses
        ``CodeExecutor.execute()`` which calls ``prepare_script()`` and
        manages its own subprocess with observers and process-group cleanup.
        """
        logger.info(f"Submitting Slurm job to partition '{self.partition}'")
        modified_script = self.prepare_script(script_content)
        result = self._execute_script(modified_script)
        if result.returncode == 0:
            self._last_job_id = self.extract_job_id(result.stdout)
        return result

    def _stop_remote_job(self, job_id: str) -> bool:
        """Cancel a Slurm job via ``scancel``."""
        logger.info(f"Cancelling Slurm job: {job_id}")
        try:
            result = subprocess.run(
                ["scancel", job_id],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                logger.info(f"Successfully cancelled Slurm job: {job_id}")
                return True
            logger.warning(f"Failed to cancel Slurm job {job_id}: {result.stderr}")
            return False
        except Exception as e:
            logger.error(f"Error cancelling Slurm job {job_id}: {e}")
            return False

    def cleanup(self) -> bool:
        """No-op — Slurm handles its own resource cleanup."""
        logger.info("SlurmBackend cleanup: nothing to do")
        return True

    # -- Factory & template --------------------------------------------------

    _CLI_ARG_MAP = {
        "partition": "--slurm-partition",
    }

    @classmethod
    def validate_config(cls, config_dict: Dict) -> None:
        required = {"partition"}
        present = {k for k in config_dict if k != "type"}
        missing = required - present
        if missing:
            cli_hints = ", ".join(cls._CLI_ARG_MAP[k] for k in sorted(missing) if k in cls._CLI_ARG_MAP)
            raise ValueError(
                f"Incomplete slurm backend configuration. "
                f"Missing: {', '.join(sorted(missing))}. "
                f"Provide via CLI: {cli_hints}"
            )

    @classmethod
    def from_config(cls, config_dict: Dict) -> "SlurmBackend":
        cls.validate_config(config_dict)
        return cls(
            partition=config_dict["partition"],
            num_nodes=config_dict.get("num_nodes", 1),
            gpus_per_node=config_dict.get("gpus_per_node", 8),
            ntasks_per_node=config_dict.get("ntasks_per_node", 1),
            time_limit=config_dict.get("time_limit", "30-00:00:00"),
            conda_env=config_dict.get("conda_env"),
            conda_base=config_dict.get("conda_base", "/opt/conda"),
            cuda_version=config_dict.get("cuda_version", "12.8"),
            account=config_dict.get("account"),
            qos=config_dict.get("qos"),
            exclusive=config_dict.get("exclusive", True),
            cache_base=config_dict.get("cache_base"),
            job_name_prefix=config_dict.get("job_name_prefix", "llmzero"),
            setup_ray=config_dict.get("setup_ray", True),
            slurm_log_dir=config_dict.get("slurm_log_dir"),
        )

    @classmethod
    def execution_script_template(cls, config_dict: Optional[Dict] = None) -> str:
        """Return the Slurm-specific execution script skeleton for prompt injection."""
        return """Generate execution script structure:

```bash
#!/bin/bash
set -e

# This script runs inside a Slurm job allocation with conda, CUDA, and Ray already set up.
# Do NOT add #SBATCH headers — they are added automatically by the executor.
# Do NOT add conda activate commands — the environment is already activated.
# Do NOT set up Ray (ray start) — the Ray cluster is started automatically.
# Simply run the training commands directly.

<copy all lines from entry point script here>
```

Requirements:
- Start with #!/bin/bash and set -e
- Do NOT use `ray job submit`. The Ray cluster is managed by the Slurm job script.
- Do NOT add #SBATCH directives — Slurm submission is handled by the executor.
- Do NOT add conda activate commands — the environment is already activated.
- Do NOT set up Ray (ray start) — the Ray cluster is started automatically on Slurm nodes.
- Copy the entry point script content directly into this script.
- If "WandB Enabled" is True above, export the WandB API key:
  export WANDB_API_KEY="<use the WandB API Key value shown above>"
- If "Previous Run Error Analysis" section above has content, the previous run FAILED - focus on fixing execution/environment issues
- If "Previous Run Error Analysis" is empty, the previous run SUCCEEDED - maintain the working execution configuration"""

    # -- Internal helpers ----------------------------------------------------

    def _build_sbatch_script(self, training_commands: str) -> str:
        """Assemble the inner sbatch script from the template.

        Builds conditional snippet strings (conda setup, cache dirs, Ray
        cluster, extra SBATCH directives) and injects them into the
        ``sbatch_template.sh`` placeholders.
        """
        # Extra #SBATCH directives (account, qos, exclusive)
        extra = ""
        if self.account:
            extra += f"#SBATCH --account={self.account}\n"
        if self.qos:
            extra += f"#SBATCH --qos={self.qos}\n"
        if self.exclusive:
            extra += "#SBATCH --exclusive\n"

        # Conda activation block
        conda_setup = ""
        if self.conda_env:
            conda_setup = (
                f'\nsource "{self.conda_base}/etc/profile.d/conda.sh"\n'
                f'conda activate "{self.conda_env}"\n'
                "\n"
                "# Ensure conda env bin is first in PATH\n"
                'export PATH="${CONDA_PREFIX}/bin:${PATH}"\n'
                "\n"
                "# Prepend conda env NVIDIA libs\n"
                'NVIDIA_PKG_DIR="${CONDA_PREFIX}/lib/'
                "$(python3 -c 'import sys;print(f\"python{sys.version_info.major}.{sys.version_info.minor}\")')"
                '/site-packages/nvidia"\n'
                "export LD_LIBRARY_PATH="
                '"${NVIDIA_PKG_DIR}/nvjitlink/lib:'
                "${NVIDIA_PKG_DIR}/cudnn/lib:"
                "${NVIDIA_PKG_DIR}/nccl/lib:"
                "${NVIDIA_PKG_DIR}/cusparselt/lib:"
                "${NVIDIA_PKG_DIR}/cublas/lib:"
                "${NVIDIA_PKG_DIR}/cuda_runtime/lib:"
                "${CONDA_PREFIX}/lib:"
                '${LD_LIBRARY_PATH:-}"\n'
            )

        # Cache directories block
        cache_setup = ""
        if self.cache_base:
            cache_setup = (
                "# =======================================================\n"
                "# Cache directories\n"
                "# =======================================================\n"
                f'export XDG_CACHE_HOME="{self.cache_base}/xdg_cache"\n'
                'export TORCH_COMPILE_CACHE_DIR="$XDG_CACHE_HOME/torch_compile"\n'
                'export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"\n'
                'mkdir -p "$XDG_CACHE_HOME" "$TORCH_COMPILE_CACHE_DIR" "$TRITON_CACHE_DIR"\n'
            )

        # Ray cluster setup block
        ray_setup = ""
        if self.setup_ray:
            ray_setup = self._ray_setup_snippet()

        return _SBATCH_TEMPLATE.format(
            job_name=self.job_name_prefix,
            partition=self.partition,
            num_nodes=self.num_nodes,
            ntasks_per_node=self.ntasks_per_node,
            gpus_per_node=self.gpus_per_node,
            time_limit=self.time_limit,
            sbatch_extra_directives=extra,
            cuda_version=self.cuda_version,
            conda_setup=conda_setup,
            cache_setup=cache_setup,
            ray_setup=ray_setup,
            training_commands=training_commands,
        )

    def _build_outer_wrapper(self, sbatch_script: str) -> str:
        """Produce the outer wrapper that submits and streams."""
        return _OUTER_WRAPPER_TEMPLATE.format(
            slurm_log_dir=self.slurm_log_dir,
            sbatch_script_content=sbatch_script,
        )

    def _ray_setup_snippet(self) -> str:
        """Return the Ray cluster bootstrap bash snippet."""
        return (
            "\n"
            "# =======================================================\n"
            "# Ray cluster setup\n"
            "# =======================================================\n"
            'head_node_ip=$(srun -N1 -n1 -w "$head_node" bash -c '
            "\"hostname -I 2>/dev/null | tr ' ' '\\n' | "
            'grep -E \'^[0-9]+(\\.[0-9]+){3}$\' | head -n1") || head_node_ip=""\n'
            'if [[ -z "${head_node_ip}" ]]; then\n'
            '  head_node_ip=$(srun -N1 -n1 -w "$head_node" bash -c '
            "\"getent hosts \\$(hostname) | awk '{print \\$1}' | "
            "grep -E '^[0-9]+(\\.[0-9]+){3}$' | head -n1 || "
            "hostname --ip-address || echo ''\") || true\n"
            "fi\n"
            'if [[ -z "${head_node_ip}" || "${head_node_ip}" == *:* ]]; then\n'
            '  echo "[ERROR] Could not determine IPv4 for head node" >&2\n'
            "  exit 1\n"
            "fi\n"
            "\n"
            "port=${RAY_HEAD_PORT:-6379}\n"
            'ip_head="${head_node_ip}:${port}"\n'
            "export ip_head\n"
            "export MASTER_ADDR=${head_node_ip}\n"
            "export MASTER_PORT=${port}\n"
            'echo "Ray head will bind at: $ip_head"\n'
            "\n"
            "# --- Cleanup trap ---\n"
            "cleanup_ray() {\n"
            '    if [[ -n "${_RAY_CLEANED:-}" ]]; then return; fi\n'
            "    _RAY_CLEANED=1\n"
            '    if [[ -n "${head_node:-}" ]]; then\n'
            '        srun --nodes=1 --ntasks=1 -w "$head_node" ray stop >/dev/null 2>&1 || true\n'
            "    fi\n"
            "    if [[ ${#nodes_array[@]} -gt 1 ]]; then\n"
            '        for node_i in "${nodes_array[@]:1}"; do\n'
            '            srun --nodes=1 --ntasks=1 -w "$node_i" ray stop >/dev/null 2>&1 || true\n'
            "        done\n"
            "    fi\n"
            "}\n"
            "trap cleanup_ray EXIT SIGINT SIGTERM\n"
            "\n"
            "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-1}}\n"
            'if [[ -z "${SLURM_CPUS_PER_TASK}" ]]; then SLURM_CPUS_PER_TASK=1; fi\n'
            "\n"
            "# Stop any existing Ray on head node\n"
            'if srun --nodes=1 --ntasks=1 -w "$head_node" ray status >/dev/null 2>&1; then\n'
            '    srun --nodes=1 --ntasks=1 -w "$head_node" ray stop\n'
            "    sleep 5\n"
            "fi\n"
            'echo "Starting Ray HEAD on $head_node ($head_node_ip)"\n'
            'srun --nodes=1 --ntasks=1 -w "$head_node" \\\n'
            "    env -u ROCR_VISIBLE_DEVICES \\\n"
            "    ray start --head \\\n"
            '        --node-ip-address="$head_node_ip" \\\n'
            "        --port=$port \\\n"
            '        --num-cpus "${SLURM_CPUS_PER_TASK}" \\\n'
            f'        --num-gpus "{self.gpus_per_node}"\n'
            "sleep 8\n"
            "\n"
            "worker_num=$((SLURM_JOB_NUM_NODES - 1))\n"
            "if [[ $worker_num -gt 0 ]]; then\n"
            "    for ((i = 1; i <= worker_num; i++)); do\n"
            "        node_i=${nodes_array[$i]}\n"
            '        if [[ -z "$node_i" ]]; then continue; fi\n'
            '        if srun --nodes=1 --ntasks=1 -w "$node_i" ray status >/dev/null 2>&1; then\n'
            '            srun --nodes=1 --ntasks=1 -w "$node_i" ray stop\n'
            "            sleep 5\n"
            "        fi\n"
            '        echo "Starting Ray WORKER $i on $node_i"\n'
            '        srun --nodes=1 --ntasks=1 -w "$node_i" \\\n'
            "            env -u ROCR_VISIBLE_DEVICES \\\n"
            '            ray start --address "$ip_head" \\\n'
            '                --num-cpus "${SLURM_CPUS_PER_TASK}" \\\n'
            f'                --num-gpus "{self.gpus_per_node}"\n'
            "        sleep 5\n"
            "    done\n"
            "else\n"
            '    echo "Single-node allocation; no Ray workers to start."\n'
            "fi\n"
            "sleep 15\n"
            "\n"
            'export HEAD_NODE="$head_node"\n'
        )

    def _execute_script(self, script_content: str) -> ComputeJobResult:
        """Execute a bash script locally with streaming output."""
        logger.debug("Executing Slurm wrapper script")

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
