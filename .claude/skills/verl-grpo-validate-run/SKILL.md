---
name: VeRL-GRPO - Validate Run
description: "[VeRL-GRPO] Health-check a GRPO run folder (parquet format, reward fn, prompt alignment, pre-flight LLM eval) and diagnose training health mid-run (val-core metrics, JSONL inspection). Use before and during training."
---

# Validate Run

Health-check a run folder before or during training. Catches common issues early.

## Input

- **Run folder**: Path to run folder (e.g., `/fsx/runs/LLMZeroCCRuns/run_YYYYMMDD_HHMMSS/`)
- **Mode** (optional): `pre-launch` (default) or `mid-training`

## Pre-launch Validation

Run these checks before launching llmzero:

### 1. Parquet Format Check

```python
import pandas as pd
for split in ["train", "val", "test"]:
    df = pd.read_parquet(f"<run_folder>/data/{split}.parquet")
    # Verify columns
    assert set(df.columns) == {"prompt", "data_source", "reward_model"}, f"Wrong columns: {df.columns}"
    # Verify types are native Python objects, not JSON strings
    row = df.iloc[0]
    assert isinstance(row["prompt"], (list, np.ndarray)), f"prompt is {type(row['prompt'])}, expected list"
    assert isinstance(row["reward_model"], dict), f"reward_model is {type(row['reward_model'])}, expected dict"
    assert "ground_truth" in row["reward_model"], "reward_model missing ground_truth"
    assert "style" in row["reward_model"], "reward_model missing style"
```

Verify:
- 3 columns: `prompt`, `data_source`, `reward_model`
- `prompt` is a list of dicts (native, not JSON string)
- `reward_model` is a dict with `ground_truth` and `style` keys
- Val split `data_source` ends with `__dev`
- No duplicate prompts in val/test (check with `df.prompt.apply(str).duplicated().sum()`)

### 2. Reward Function Syntax Check

- Import the reward function and verify `compute_score` is callable
- Test with ground truth as both input and expected output (should return 1.0)
- Test with a clearly wrong answer (should return 0.0)

### 3. Prompt Format Check

- Verify prompts have output format instructions (either in a system message or in the user message)
- If no output instructions are found, warn that the reward function may struggle to extract answers

### 4. Description Template Check

If `description.txt` exists:
- Verify no unfilled `{{PLACEHOLDER}}` values remain
- Verify file paths referenced in the description actually exist
- Verify model path exists

### 5. Pre-flight LLM Evaluation (Critical)

Run the base model on ~100 test samples and score with the reward function. This is the most important check — it validates the full prompt → model → extraction → scoring pipeline.

**Steps:**
1. Write a bash script (`<run_folder>/preflight_eval.sh`) that:
   - Installs dependencies from `install_commands.txt` (if exists)
   - Uses vLLM to generate completions for ~100 samples from `train.parquet`
   - Scores each completion with the reward function
   - Prints per-sample details and overall accuracy
   - Saves results to `<run_folder>/preflight_eval.json`

2. Submit via llmzero (command depends on `compute_backend` in `constants/cluster.json`):

   **raycluster:**
   ```bash
   cd <llmzero_path> && llmzero submit-script <run_folder>/preflight_eval.sh -r <ray_service_name>
   ```

   **conda:**
   ```bash
   cd <llmzero_path> && llmzero submit-script <run_folder>/preflight_eval.sh --compute-backend conda --conda-env <conda_env>
   ```

   **docker:**
   ```bash
   cd <llmzero_path> && llmzero submit-script <run_folder>/preflight_eval.sh --compute-backend docker --docker-image <docker_image>
   ```

   **slurmdocker:**
   ```bash
   cd <llmzero_path> && llmzero submit-script <run_folder>/preflight_eval.sh --compute-backend slurmdocker --slurm-partition <slurm_partition> --docker-image <docker_image> --mount-dir <mount1> --mount-dir <mount2>
   ```

3. Check results — **the key question is: can the base model produce any successful rollouts?** GRPO needs some positive-reward rollouts to compute advantage estimates. If none score > 0, there is no gradient signal.
   - **Some correct rollouts (even 1-5%)**: Good — GRPO can learn from the contrast between successful and failed rollouts
   - **Zero correct rollouts**: Reward function extraction is likely broken. Inspect model outputs, fix extraction, re-run.
   - Look at individual failures — are the model's answers correct but unparseable? That points to extraction bugs.

### 6. Report

Print a summary:
```
=== Run Validation: <run_folder> ===
[OK] Parquet format: train=N, val=N, test=N
[OK] No duplicate prompts in val/test
[OK] Val data_source ends with __dev
[OK] Reward function imports and passes basic tests
[OK/WARN] Prompt format includes output instructions
[OK] Description placeholders all filled
[OK] Model path exists
[OK] Pre-flight eval: X.X% baseline accuracy (N/100 correct)
```

## Mid-training Validation

When `mode=mid-training`, additionally check:

### 7. Training Dynamics Check

- Read `<run_folder>/llmzero_output/info_logs.txt` for validation scores
- Check the VERL output JSONL files for actual model outputs and scores
- Report:
  - Current MCTS iteration (node_N)
  - Latest validation score
  - Whether scores are improving, flat, or stuck at zero
  - If stuck at zero: sample some model outputs and test reward function against them to diagnose

### 8. Reward Function vs Actual Outputs

If validation scores are suspiciously low:
- Read 10-20 entries from the latest VERL JSONL output
- Run the reward function on each and show: model output (tail), extracted answer, ground truth, score
- Identify patterns: is extraction failing? Are answers correct but scored wrong?

## VeRL-GRPO Training-Health Reference (consolidated)

General engine skills (/general-launch-llmzero, /general-check-training,
/general-gather-results) point here for VeRL-GRPO-specific metric interpretation.

**Validation metric:** The tracked metric is `val-core/<task>__dev/reward/mean@1`.
If you see `mean@2` or `mean@3` under `val-core` instead of `mean@1`, the val/test
data has duplicate prompts (VERL groups identical prompts and reports pass@N).
Fix by deduplicating — see /verl-grpo-prepare-dataset.

**Healthy signs (first MCTS iteration):**
- `val-core/...__dev/reward/mean@1` is non-zero and increasing (even 1-5% is OK).
- `critic/rewards/mean` shows non-zero values.

**Unhealthy signs (investigate if persistent across 3+ val steps):**
- Val score stays at 0 or shows no improvement.
- `critic/rewards/mean` stays exactly 0.0.

**Diagnose BEFORE stopping:**
1. Find the VERL output directory from the run script (e.g. `/fsx/verl_runs/.../`).
2. Read the JSONL output files (`0.jsonl`, `3.jsonl`, ...) to see actual model
   outputs and scores.
3. Test the reward function against real model outputs to find extraction failures.
4. Only stop llmzero if you have a confident fix — stopping without one wastes the
   MCTS tree state.
5. After fixing, re-run the pre-flight LLM eval to confirm, then relaunch.

**WandB false-positive:** A Ray job may report FAILED (exit code 1) even when
training succeeded — WandB's atexit cleanup can raise `BrokenPipeError`. Check the
logs for `Best checkpoint folder created`; if present, treat the run as successful.
