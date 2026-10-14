---
name: General - Check Training Status
description: "[General] Check the status of a running llmzero session or its backend jobs (Ray/Slurm/conda/docker). Use to poll progress. Framework-agnostic; for VeRL-GRPO metric diagnostics see /verl-grpo-validate-run."
---

# Check Training Status

Check the status of a running llmzero training session or its underlying jobs.

## Input

- **Run folder**: Path to the run (e.g., `/fsx/runs/LLMZeroCCRuns/run_YYYYMMDD_HHMMSS/`)
- **Job ID** (optional): Specific job ID to check

## Steps

### 1. Check LLMZero Output

Look at the llmzero output directory for progress:

```bash
ls -la <run_folder>/llmzero_output/
```

Check for the MCTS node directories and their status.

### 2. Check Job Status

Read `constants/cluster.json` for `compute_backend` and backend-specific settings.

**If `compute_backend == "raycluster"`:**

Construct the Ray address and list/check jobs:
```bash
ray job list --address <ray_address> 2>&1 | head -50
```

If a specific job ID was provided:
```bash
ray job status --address <ray_address> <job_id>
ray job logs --address <ray_address> <job_id> 2>&1 | tail -100
```

**If `compute_backend == "conda"`:**

Check for running llmzero/python processes:
```bash
ps aux | grep -E "llmzero|verl|main_ppo" | grep -v grep
```

Check the llmzero output logs directly:
```bash
tail -100 <run_folder>/llmzero_output/info_logs.txt 2>/dev/null
```

**If `compute_backend == "docker"`:**

Check for running containers:
```bash
docker ps --filter "ancestor=<docker_image>" --format "table {{.ID}}\t{{.Status}}\t{{.Command}}"
```

Check container logs or the host-mounted llmzero output logs:
```bash
tail -100 <run_folder>/llmzero_output/info_logs.txt 2>/dev/null
```

**If `compute_backend == "slurmdocker"`:**

Check Slurm job status:
```bash
squeue -u $USER --format="%.18i %.9P %.30j %.8T %.10M %.9l %.6D %R" | head -20
```

If a specific job ID was provided:
```bash
scontrol show job <job_id>
```

Check the llmzero output logs:
```bash
tail -100 <run_folder>/llmzero_output/info_logs.txt 2>/dev/null
```

### 3. Analyze Logs

Look for:
- Training step progress lines (`step:<N>`)
- Validation scores (`val-core/`)
- Error tracebacks
- Completion markers (`Best checkpoint folder created`)

### 4. Report

Tell the user:
- Current job status (PENDING/RUNNING/SUCCEEDED/FAILED/STOPPED)
- Latest training step and metrics (if running)
- Best validation score so far (if available)
- Any errors encountered

## Stopping a run — SHARED-ACCOUNT SAFETY (read before any `scancel`)

**NEVER cancel Slurm jobs by name.** The slurmdocker backend gives **every** llmzero
run the SAME job name `llmzero-docker-training`. On a shared account (many users, one
`$USER`), `scancel --name llmzero-docker-training` or `squeue | grep llmzero-docker-training
| scancel` will kill **other people's runs**, not just yours. `squeue -u $USER` also lists
everyone on the shared account. This is destructive and hard to undo — a cancelled node
loses its in-progress training.

To stop only YOUR run, in order of preference:

1. **Kill the launcher process** (safest). Each slurmdocker run's outer wrapper has a
   SIGTERM/SIGINT trap that `scancel`s ONLY its own job. Killing the `llmzero` process for
   your run triggers that trap:
   ```bash
   pkill -f "<your_run_folder>"     # match the unique run-dir path, NOT the job name
   ```
2. **Stop by the exact job ID your run owns.** Capture it at launch (the launch log prints
   `Extracted Slurm job ID: <id>`), then:
   ```bash
   llmzero stop-job <job-id> [backend flags]     # or: scancel <job-id>
   ```
   Only cancel a numeric job ID you have confirmed belongs to your run.

Before any `scancel`, confirm ownership: match on the **run-folder path** or a **known job
ID**, never on the shared job name. If you cannot tie a job to your run, do NOT cancel it.

**VeRL-GRPO metrics & WandB false-positive:** for `val-core/...mean@1` interpretation,
duplicate-prompt diagnosis, and the WandB `BrokenPipeError` false-FAILED case, see
/verl-grpo-validate-run.
