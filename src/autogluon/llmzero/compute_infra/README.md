# Compute Infrastructure Backends

This package provides an abstract interface for submitting, monitoring, and
terminating training jobs on different compute backends. The executor
(`execution/executor.py`) talks only to the `BaseComputeBackend` ABC, so
adding a new backend does not require touching any core LLMZero code.

## Log Streaming Assumption

`CodeExecutor.execute()` runs the generated script as a **local subprocess**
(`bash -c <script>`) and reads training logs from its stdout/stderr in
real-time. Observers (metrics collection, early stopping, etc.) depend on
this line-by-line stream.

This means the submission command produced by `execution_script_template()`
must **block and stream the remote job's output** to the local process's
stdout/stderr. Ray satisfies this: `ray job submit` (without `--no-wait`)
forwards the remote job's stdout/stderr to the caller.

A backend whose submission command returns immediately (e.g. Slurm `sbatch`)
must arrange for log streaming itself — for example, by tailing the log file
after submission within the same script — so that the executor can still
read output line-by-line. Otherwise the observer pattern (real-time metrics,
early stopping) will not function.

## Built-in Backends

| Name | Class | Config `type` |
|------|-------|---------------|
| Ray on Kubernetes | `compute_infra.raycluster.RayClusterBackend` | `raycluster` |
| Conda (local) | `compute_infra.conda.CondaBackend` | `conda` |
| Docker (local) | `compute_infra.docker.DockerBackend` | `docker` |
| Slurm (conda, multi-node) | `compute_infra.slurm.SlurmBackend` | `slurm` |
| Slurm + Docker (pyxis/enroot) | `compute_infra.slurmdocker.SlurmDockerBackend` | `slurmdocker` |

## How to Add a New Backend

### 1. Create the backend module

```
src/autogluon/llmzero/compute_infra/<backend_name>/
├── __init__.py        # re-export the backend class
└── backend.py         # implementation
```

### 2. Implement `BaseComputeBackend`

```python
# compute_infra/<backend_name>/backend.py
from ..base import BaseComputeBackend, ComputeJobResult

class MyBackend(BaseComputeBackend):
    # Properties
    address          -> str           # dashboard / scheduler URL for logging
    backend_type     -> str           # e.g. "slurm"
    last_job_id      -> Optional[str] # getter + setter

    # Script manipulation
    prepare_script(script_content)    # rewrite addresses / paths
    extract_job_id(stdout)            # parse job ID from submission output

    # Job lifecycle
    submit_job(script_content)        # returns ComputeJobResult
    terminate(pid, job_id, stdout)    # unified termination (all args optional)
    _stop_remote_job(job_id)          # backend-specific remote stop (called by terminate)
    _kill_process_group(pid)          # SIGTERM+SIGKILL local process group (called by terminate)
    cleanup()                         # clear caches / temp files

    # Factory & template
    @classmethod from_config(config_dict)         # instantiate from YAML dict
    @classmethod execution_script_template()      # prompt snippet for LLM
```

### 3. Register in `compute_infra/__init__.py`

```python
from .<backend_name> import MyBackend
register_backend("<backend_name>", MyBackend)
```

### 4. Add a config section

In your YAML config:

```yaml
compute_infra:
  type: <backend_name>
  # backend-specific keys...
```

### 5. Add CLI flags (optional)

If your backend needs CLI-level overrides, add options to `cli/app.py`
and pass them through `compute_infra_overrides` into
`coding_agent.run_agent()`.

## Example: Slurm + Docker Backend

A Slurm backend that runs training inside Docker/enroot containers (via the
pyxis plugin) is implemented at `compute_infra/slurmdocker/backend.py`. Key design:

- **Two-part wrapper**: An inner `sbatch` script with pyxis `--container-image` directives (container setup + training commands) and an outer wrapper script that runs `sbatch`, then `tail -f`s the log file while polling `squeue` for completion.
- **Log streaming**: The outer wrapper blocks and streams the Slurm job's stdout/stderr to the local process, satisfying the `CodeExecutor` observer contract described above.
- **SIGTERM cleanup**: The outer wrapper traps SIGTERM/SIGINT and runs `scancel` so that early-stopping and MCTS cancellation propagate to the Slurm job.
- **Bash template**: The outer wrapper lives in a standalone `.sh` template file (`outer_wrapper_template.sh`) loaded via `importlib.resources`.
- **Config `type`**: `slurmdocker`. Required keys: `partition`, `container_image` (enroot `.sqsh` file or registry URI). Optional: `container_mounts`, `num_nodes`, `gpus_per_node`, `time_limit`, `account`.

## Example: Slurm Backend (conda)

A Slurm backend that runs training directly in a conda environment on HyperPod
compute nodes (no container) is implemented at `compute_infra/slurm/backend.py`.
This is the backend to use for **multi-node** training when a shared conda
environment is available, since the conda/docker backends are single-node only.
Key design:

- **Two-part wrapper**: An inner `sbatch` script (`#SBATCH` headers, conda activation, CUDA/LD_LIBRARY_PATH setup, multi-node Ray cluster bootstrap, training commands) and an outer wrapper that runs `sbatch`, then `tail -f`s the log file while polling `squeue` for completion.
- **Multi-node Ray**: The inner script starts a Ray head on the first allocated node and `ray start --address`es workers on the remaining nodes via `srun`, so verl/Ray see one cluster spanning all nodes.
- **Log streaming**: The outer wrapper blocks and streams the Slurm job's stdout/stderr to the local process, satisfying the `CodeExecutor` observer contract described above.
- **SIGTERM cleanup**: The outer wrapper traps SIGTERM/SIGINT and runs `scancel` so early-stopping and MCTS cancellation propagate to the Slurm job.
- **Bash templates**: The sbatch and outer wrapper live in standalone `.sh` template files (`sbatch_template.sh`, `outer_wrapper_template.sh`) loaded via `importlib.resources`.
- **Config `type`**: `slurm`. Required keys: `partition`. Optional: `conda_env`, `num_nodes`, `gpus_per_node`, `time_limit`, `account`, `conda_base`, `cuda_version`, `qos`, `exclusive`, `cache_base`, `setup_ray`, `slurm_log_dir`.
