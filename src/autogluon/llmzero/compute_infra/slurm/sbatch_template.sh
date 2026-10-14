#!/bin/bash
#SBATCH --job-name={job_name}-training
#SBATCH --partition={partition}
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --gpus-per-node={gpus_per_node}
#SBATCH --time={time_limit}
{sbatch_extra_directives}#SBATCH --output=LOG_DIR_PLACEHOLDER/slurm_%j.out
#SBATCH --error=LOG_DIR_PLACEHOLDER/slurm_%j.err

set -euo pipefail

# =======================================================
# CUDA & Conda setup
# =======================================================
export CUDA_HOME=/usr/local/cuda-{cuda_version}
export PATH=/usr/local/cuda-{cuda_version}/bin:$PATH
{conda_setup}
{cache_setup}
# P5 symmetric memory workaround
if [[ "${{SLURM_JOB_PARTITION:-}}" == ml-p5* ]]; then
    export PYTORCH_SYMMETRIC_MEMORY=0
    export TORCH_SYMMETRIC_MEMORY=0
    export VLLM_ALLREDUCE_USE_SYMM_MEM=0
    export VLLM_USE_NCCL_SYMM_MEM=0
    export VLLM_USE_TORCH_SYMMETRIC_MEMORY=0
fi

# =======================================================
# Node discovery
# =======================================================
readarray -t nodes_array < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [[ ${{#nodes_array[@]}} -eq 0 ]]; then
    echo "[ERROR] No nodes discovered from SLURM_JOB_NODELIST" >&2
    exit 1
fi
head_node=${{nodes_array[0]}}
echo "Discovered nodes: ${{nodes_array[*]}}"
echo "Head node: $head_node"

# =======================================================
# Clean Python/GPU state on allocated nodes
# =======================================================
echo "Ensuring clean Python/GPU state on allocated nodes..."
for node_i in "${{nodes_array[@]}}"; do
    if [[ -z "${{node_i}}" ]]; then continue; fi
    echo "Cleaning node ${{node_i}}"
    srun --nodes=1 --ntasks=1 -w "$node_i" bash -c '
        set -euo pipefail
        if command -v ray >/dev/null 2>&1; then
            ray stop --force >/dev/null 2>&1 || true
        fi
        rm -rf /tmp/ray || true
        running=$(pgrep -fl "python" || true)
        if [[ -n "${{running}}" ]]; then
            pkill -9 python || true
            sleep 3
        fi
        if command -v nvidia-smi >/dev/null 2>&1; then
            for attempt in 1 2 3; do
                procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -E "[0-9]" || true)
                if [[ -z "${{procs}}" ]]; then break; fi
                for pid in ${{procs}}; do kill -9 "${{pid}}" || true; done
                sleep 2
            done
        fi
    '
    cleanup_status=$?
    if [[ ${{cleanup_status}} -ne 0 ]]; then
        echo "[ERROR] Cleanup failed on node ${{node_i}}; aborting." >&2
        exit ${{cleanup_status}}
    fi
done
{ray_setup}
# =======================================================
# Training commands
# =======================================================
{training_commands}
