---
name: General - Launch LLMZero
description: "[General] Start the llmzero MCTS training loop for a prepared task via the configured compute backend (raycluster/conda/docker/slurmdocker). Framework-agnostic engine launch."
---

# Launch LLMZero

Start the llmzero MCTS training loop for a prepared task.

## Input

- **Run folder**: Path containing data, reward function, llmzero config, and `llmzero_input/description.txt`
- **Max iterations** (optional): Number of MCTS iterations. Default: 16
- **LLMZero config** (optional): Path to llmzero YAML config. Default: `<run_folder>/llmzero_config.yaml`

## Steps

### 1. Read Constants

Load `constants/cluster.json` for compute backend info (`compute_backend`, and the relevant backend-specific fields).

### 2. Verify Prerequisites

Confirm the run folder contains:
- `llmzero_input/description.txt` (the task description)
- `data/train.parquet` and `data/val.parquet`
- `reward_fn.py`
- `llmzero_config.yaml`

If any are missing, report the error and stop.

### 3. Launch LLMZero

Build the llmzero command based on `compute_backend` from `constants/cluster.json`:

**If `compute_backend == "raycluster"` (remote Ray/HyperPod cluster):**
```bash
RUN_DIR="<run_folder>"
cd <llmzero_path> && llmzero \
  -i "${RUN_DIR}/llmzero_input" \
  -o "${RUN_DIR}/llmzero_output" \
  -c "${RUN_DIR}/llmzero_config.yaml" \
  -r <ray_service_name> \
  --ray-namespace <ray_namespace> \
  -n <max_iterations> \
  --wandb-api-key <wandb_api_key> \
  -v 2 2>&1 | tee "${RUN_DIR}/llmzero_launch.log"
```

**CRITICAL: Always use `tee` to capture output. Never pipe through `head`, `tail`, or any command that exits early.** A broken stdout pipe causes llmzero to crash silently during post-execution processing (e.g. LLM analysis after timeout), preventing it from saving node results or starting subsequent MCTS iterations.

**If `compute_backend == "conda"` (local conda environment):**
```bash
cd <llmzero_path> && llmzero \
  -i <run_folder>/llmzero_input \
  -o <run_folder>/llmzero_output \
  -c <run_folder>/llmzero_config.yaml \
  --compute-backend conda \
  --conda-env <conda_env> \
  --num-gpus <n_gpus_per_node> \
  -n <max_iterations> \
  --wandb-api-key <wandb_api_key> \
  -v 2
```

**If `compute_backend == "docker"` (local Docker container):**
```bash
cd <llmzero_path> && llmzero \
  -i <run_folder>/llmzero_input \
  -o <run_folder>/llmzero_output \
  -c <run_folder>/llmzero_config.yaml \
  --compute-backend docker \
  --docker-image <docker_image> \
  --mount-dir <runs_root> \
  --mount-dir <model_path_prefix> \
  -n <max_iterations> \
  --wandb-api-key <wandb_api_key> \
  -v 2
```
Add a `--mount-dir` for each entry in `docker_mount_dirs` from cluster.json, plus `runs_root` and `model_path_prefix`.

**If `compute_backend == "slurmdocker"` (Slurm + Docker via pyxis/enroot):**
```bash
RUN_DIR="<run_folder>"
cd <llmzero_path> && llmzero \
  -i "${RUN_DIR}/llmzero_input" \
  -o "${RUN_DIR}/llmzero_output" \
  -c "${RUN_DIR}/llmzero_config.yaml" \
  --compute-backend slurmdocker \
  --slurm-partition <slurm_partition> \
  --docker-image <docker_image> \
  --mount-dir <mount1> \
  --mount-dir <mount2> \
  --slurm-nodes <nnodes> \
  --slurm-gpus-per-node <n_gpus_per_node> \
  -n <max_iterations> \
  --wandb-api-key <wandb_api_key> \
  -v 2 2>&1 | tee "${RUN_DIR}/llmzero_launch.log"
```
Add a `--mount-dir` for each entry in `docker_mount_dirs` from cluster.json (format: `src:dst`, e.g. `/fsx:/fsx`).
Add `--slurm-account <slurm_account>` if `slurm_account` is non-empty in cluster.json.
Add `--slurm-time <slurm_time_limit>` if `slurm_time_limit` is non-empty in cluster.json.

Note: slurmdocker uses the same `--docker-image` flag as docker backend, but the value is typically a `.sqsh` file path or registry URI. Container mounts use `--mount-dir` with `src:dst` format (e.g., `/fsx:/fsx`).

**Parallel jobs (`job_parallel > 1`):**
Add `-j <job_parallel>` to any of the above commands when `job_parallel` in cluster.json is > 1. This runs N training jobs concurrently using async Top-K Weighted Deepening instead of sequential MCTS.

Key flags:
- `-i`: Input folder containing `description.txt` (`llmzero_input/` subfolder)
- `-o`: Output directory for llmzero's MCTS tree, logs, scripts
- `-c`: YAML config for the MCTS loop
- `-r`: Ray cluster head service name (raycluster only)
- `--compute-backend`: `raycluster`, `conda`, `docker`, or `slurmdocker`
- `--conda-env`: Conda environment name (conda)
- `--docker-image`: Docker image URI or .sqsh path (docker/slurmdocker)
- `--mount-dir`: Host directory to mount, repeatable (docker: identity-mount; slurmdocker: `src:dst` format)
- `--slurm-partition`: Slurm partition name (slurmdocker, required)
- `--slurm-nodes`: Number of Slurm nodes (slurmdocker)
- `--slurm-gpus-per-node`: GPUs per node (slurmdocker)
- `--slurm-time`: Slurm time limit (slurmdocker)
- `--slurm-account`: Slurm account name (slurmdocker)
- `-n`: Max MCTS iterations (default 16). LLMZero runs all iterations by default.
- `-j`: Number of parallel training jobs (default 1). If > 1, uses async search instead of sequential MCTS.
- `--stop-after-success`: Optional. Stop after the first successful run instead of running all iterations.
- `--wandb-api-key`: For experiment tracking
- `-v 2`: Verbosity level (info + file locations)

The initial training hyperparameters (epochs, lr, batch size, etc.) come from the `description.txt` reference script. LLMZero's MCTS adjusts them between iterations.

**Capture your run's Slurm job ID (slurmdocker, shared accounts).** The launch log prints
`Extracted Slurm job ID: <id>` for each node. Note these — they are how you stop ONLY your
run later. Do NOT stop a slurmdocker run with a name-based `scancel`: every llmzero run
shares the job name `llmzero-docker-training`, so a name filter cancels other users' runs on
the same account. To stop your run, `pkill -f "<your_run_folder>"` (the wrapper's trap
cancels just its own job) or `scancel`/`llmzero stop-job` a job ID you confirmed is yours.
See `/general-check-training` → "Stopping a run — SHARED-ACCOUNT SAFETY".

### 4. Monitor

Launch llmzero in the background and monitor its progress. It will:
- Run the MCTS search loop (select -> expand -> execute -> backpropagate)
- Submit training jobs for each node (via the configured compute backend)
- Track validation scores and checkpoint paths
- Apply early stopping if configured

**Estimate total runtime** to set an appropriate monitoring cadence:
- Steps per epoch ≈ `train_size / train_batch_size`
- Time per step ≈ 2-5 min (varies by model size, response length, hardware)
- One MCTS iteration ≈ steps × time_per_step
- Total ≈ num_iterations × time_per_iteration (can be days for large runs)

**Monitoring cadence:**
- **First iteration, early steps**: Check every 10-15 minutes until you see non-zero val scores across 3+ validation steps
- **First iteration confirmed healthy**: Reduce to every 1-2 hours
- **After first iteration**: llmzero MCTS runs autonomously. Check every few hours or once per day for multi-day runs.

**Early training health check (first MCTS iteration):**

Monitor `<run_folder>/llmzero_output/info_logs.txt` for validation scores. Validation happens every `test_freq` steps (default 3).

**VeRL-GRPO training-health diagnostics** (val-core metric interpretation, healthy
vs stuck-at-zero signs, JSONL diagnosis, WandB false-positive): see
/verl-grpo-validate-run (its consolidated Training-Health Reference).

### 5. Report

When llmzero completes, tell the user:
- Whether it completed successfully or with errors
- Number of iterations run
- Best validation score found (if visible in output)
- Path to llmzero output directory

## Troubleshooting

- **llmzero not found**: Run `cd <llmzero_path> && pip install -e .` to install
- **Ray connection refused** (raycluster): Verify the Ray cluster is running with `ray job list --address <ray_address>`
- **Docker permission denied** (docker): Ensure the user is in the `docker` group or run with sudo
- **Conda env not found** (conda): Verify the environment exists with `conda env list`
- **OOM errors**: Check GPU memory usage; may need to reduce batch sizes or sequence lengths in the description
- **Timeout**: llmzero's default per-execution timeout is 144000s (40h). For faster feedback, reduce `per_execution_timeout` in the config YAML
