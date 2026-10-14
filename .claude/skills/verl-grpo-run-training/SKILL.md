---
name: VeRL-GRPO - Run Training
description: "[VeRL-GRPO] End-to-end orchestrator for the RL post-training pipeline: dataset prep, reward fn, pre-flight eval, task generation, model download, llmzero launch, monitoring, and results. Use to run a full GRPO job from a dataset."
---

# Run End-to-End Training

Orchestrate the full pipeline: dataset preparation, reward function definition, task generation, model download, llmzero launch, and result gathering.

## Orchestrator Decision Log — MANDATORY

**You MUST maintain `<run_folder>/orchestrator_log.md` throughout the entire run.** Create it immediately after creating the run folder (Step 1), and append a timestamped entry after EVERY major decision point — do not wait until the end. This is not optional.

**When to append:** After creating the run folder, after dataset preparation, after writing the reward function, after each preflight eval (especially failures and fixes), before launching llmzero, and at each monitoring checkpoint.

**How to append:** Use `cat >> <run_folder>/orchestrator_log.md << 'EOF'` to append (not overwrite).

**What to log:** What you observed, what you decided, and why. Include metric values. Example:

```markdown
## 2026-03-19 01:15 — Preflight Failed: 0% accuracy
- Model copies template placeholder literally. Fix: simplify prompt, add substring matching.

## 2026-03-19 01:30 — Preflight Passed: 12.5% accuracy
- After prompt fix, base model gets 12.5% correct. Sufficient gradient signal for GRPO. Proceeding to launch.
```

## Input

The user provides one or more of:
- **Dataset**: HuggingFace dataset name(s) (e.g., `IDEA-AI4S/ChemCoTDataset`) or local path(s) to parquet files
- **Model** (optional): HuggingFace model name (e.g., `Qwen/Qwen3-4B`). Default: from `constants/cluster.json`
- **Training mode** (optional): `lora` or `fft`. Default: `lora`
- **User instruction** (optional): Additional guidance for reward function design or hyperparameter choices
- **LLMZero config** (optional): Path to a custom llmzero YAML config. Default: copy from `constants/cluster.json::llmzero_default_config`

## Steps

### 1. Create Run Folder

Read `constants/cluster.json` for defaults. Create a timestamped run folder:

```
/fsx/LLMZeroCCRuns/run_YYYYMMDD_HHMMSS/
```

All artifacts for this run go here: data, reward function, description, llmzero config, logs, report.

### 2. Prepare Dataset

Invoke `/verl-grpo-prepare-dataset` to:
- Download from HuggingFace (if HF name provided)
- Inspect schema and sample rows
- Convert to VERL parquet format with `prompt` and `data_source` columns
- Split into train/val/test
- **For multi-subtask datasets**: val.parquet uses a single unified `data_source` (e.g., `"<dataset_name>_agg__dev"`) so llmzero gets one aggregated validation metric. Test.parquet keeps per-subtask `data_source` values for per-subtask visibility. Train.parquet uses per-subtask `data_source` values for reward function dispatching.
- **For single-task datasets**: all splits share the same `data_source`, val gets `__dev` suffix
- Save to `<run_folder>/data/train.parquet`, `val.parquet`, and `test.parquet`

### 3. Define Reward Function

Invoke `/verl-grpo-define-reward` to:
- Analyze the dataset structure (columns, answer format, task type)
- Write a `compute_score` function in `<run_folder>/reward_fn.py`
- Include any `pip install` commands needed for dependencies

### 3b. Pre-flight LLM Evaluation

**Before launching llmzero, validate the reward function against actual model outputs.** This catches reward function bugs (especially answer extraction issues) early, before wasting GPU hours on training with zero reward signal.

**Steps:**
1. Write a bash script (`<run_folder>/preflight_eval.sh`) that:
   - Installs any dependencies from `install_commands.txt`
   - Loads the base model and generates completions for ~100 samples from `train.parquet`
   - Scores each completion using the reward function
   - Prints per-sample scores and overall accuracy
2. Submit it via `llmzero submit-script` (using the appropriate compute backend flags — see `/verl-grpo-validate-run` for backend-specific commands).
3. Check the results. **The key question: can the base model produce ANY successful rollouts?** GRPO needs some rollouts with positive reward to compute meaningful advantage estimates. If zero rollouts score > 0, there is no gradient signal and RL updates cannot improve the model.
   - **Some correct rollouts (even 1-5%)**: Good — GRPO can learn from the contrast between successful and failed rollouts
   - **Zero correct rollouts**: The reward function extraction is likely broken, or the task is too hard for the base model. Inspect actual model outputs vs what the extraction function returns. Fix the reward function's answer extraction or the prompt format, then re-run.
4. Save results to `<run_folder>/preflight_eval.json` for reference
5. **Use the results to set initial `max_response_length` for the description template's reference script.** Analyze the successful rollouts' token lengths:
   - Set `max_response_length` to ~1.5-2x the p90 response length of successful rollouts, but clamp to the bounds from `constants/cluster.json`: never below `init_response_length_floor` and never above `init_response_length_ceiling`.
   - If the pre-flight eval shows most successful answers are short (e.g., < 200 tokens), you can reduce `max_response_length` from the default, which speeds up training. But always respect the floor to leave room for thinking and exploration during RL.

**This step is critical** — it validates that the prompt format, answer extraction, and scoring logic all work together on realistic model outputs, confirms the base model can produce successful rollouts for GRPO to learn from, and informs initial hyperparameter choices. Do not skip it.

### 4. Generate Task

Invoke `/verl-grpo-generate-task` to:
- Read the appropriate description template (`constants/description_template_lora.txt` or `constants/description_template_fft.txt`)
- Fill in all placeholders (model, data paths, reward function, hardware, etc.)
- Write `<run_folder>/description.txt`

### 5. Download Model (if needed)

Invoke `/general-download-model` to:
- Check if model exists at `<model_path_prefix>/<org>/<model>`
- If not, download it via `huggingface_hub.snapshot_download` (method depends on compute backend)
- Wait for download to complete before proceeding

### 6. Copy LLMZero Config

Copy the llmzero MCTS config into the run folder:
```bash
cp <llmzero_default_config> <run_folder>/llmzero_config.yaml
```
Modify if the user provided custom settings (e.g., different `validation_metric_pattern`).

### 7. Launch LLMZero

Invoke `/general-launch-llmzero` to:
- Run llmzero with the appropriate compute backend flags (see `/general-launch-llmzero` for backend-specific commands)
- The llmzero MCTS loop handles all training iterations internally
- **Monitor the first iteration closely** (see step 7b)

### 7b. Training Monitoring

After launching llmzero, monitor training with an appropriate cadence.

**Estimate total runtime first.** A single MCTS iteration involves one full VERL training run. Rough estimation:
- Steps per epoch ≈ `train_size / train_batch_size` (e.g., 20k samples / 128 batch = ~156 steps)
- Time per step ≈ 2-5 min (depends on model size, response length, hardware)
- One iteration ≈ steps × time_per_step (e.g., 156 × 3 min ≈ 8 hours)
- Total ≈ iterations × time_per_iteration (e.g., 8 iterations × 8 hours ≈ 2-3 days)

**Monitoring cadence:**
- **First iteration, early steps (steps 1-9)**: Monitor every 10-15 minutes to confirm training launched and validation scores are non-zero. This is the critical window to catch reward function bugs.
- **First iteration confirmed healthy**: After seeing non-zero and improving val scores across 3+ validation steps, reduce monitoring to every 1-2 hours.
- **After first iteration completes successfully**: llmzero's MCTS handles subsequent iterations autonomously. Monitor every few hours or once per iteration to check progress.
- **Multi-day runs**: No need for frequent monitoring once confirmed healthy. Check once or twice per day, or use `/general-check-training`.

**Early diagnostic protocol (if metrics look bad after several val steps):**

If `val-core/...__dev/reward/mean@1` stays at 0 or shows no improvement across 3+ validation steps, or `critic/rewards/mean` stays at 0:

1. Read the VERL output JSONL files (e.g., `<output_dir>/0.jsonl`) to inspect actual model outputs
2. Test the reward function against those real outputs to identify extraction failures
3. Only stop the llmzero run if you have a **confident fix** to try
4. After fixing, re-run the pre-flight LLM evaluation (step 3b) to confirm the fix works
5. Then relaunch llmzero

### 8. Gather Results

Invoke `/general-gather-results` to parse output, generate Pareto plot, and write `<run_folder>/report.md`.

## Notes

- If any step fails, diagnose the error and attempt to fix it before retrying
- The run folder at `/fsx/LLMZeroCCRuns/run_YYYYMMDD_HHMMSS/` contains everything needed to reproduce the run
- All paths in the description.txt must be absolute and hardcoded (no shell variables)
