---
name: VeRL-GRPO - Generate Task
description: "[VeRL-GRPO] Create the description.txt llmzero reads, filling a LoRA/FFT/given-config VERL template with model, data, reward, and hardware placeholders. Use before launching a GRPO run."
---

# Generate Task Description

Create the `description.txt` file that llmzero reads to understand the training task.

## Input

- **Run folder**: Path containing prepared data and reward function
- **Model name**: e.g., `Qwen/Qwen3-4B`
- **Training mode**: `lora` or `fft`
- **Base config** (optional): Path to a full verl config YAML (e.g. an exported `training_config.yaml`) to use as the default config instead of the package-default block. When given, follow the "Given Base Config" branch in Step 2.
- **User instruction** (optional): Additional hyperparameter preferences

## Steps

### 1. Read Constants

Load `constants/cluster.json` for:
- `compute_backend` (and backend-specific fields)
- `nnodes`, `n_gpus_per_node`
- `wandb_api_key`
- `model_path_prefix`
- `verl_output_dir`

### 2. Read Description Template

**If a Base Config was provided** (a full verl config YAML), use the **Given Base Config branch**:

1. **Copy the config into the run folder**: copy the given YAML to `<run_folder>/training_config.yaml`. The run folder must own the config so the run is self-contained and reproducible. Do NOT point `--config-path` at the original external location.
2. Use template `constants/description_template_given_config.txt`.
3. Resolve these extra placeholders (in addition to the table in Step 3):
   - `{{BASE_CONFIG_YAML}}` = full text of the copied `training_config.yaml` (pasted verbatim into the "Package Default Configuration" YAML block — this REPLACES the package default, do not merge).
   - `{{BASE_CONFIG_DIR}}` = `<run_folder>` (directory holding the copied config; verl/hydra `--config-path`).
   - `{{BASE_CONFIG_NAME}}` = `training_config` (basename without `.yaml`; verl/hydra `--config-name`).
4. The reference bash script in this template is intentionally minimal: it loads the copied config via `--config-path`/`--config-name` and overrides only run-specific paths plus the pinned trainer settings (`resume_mode=disable`, `save_freq`, `test_freq`, `adv_estimator=grpo`, `val_before_train=True`). All HPs (LoRA rank/alpha, batch sizes, lr, rollout) come from the config — do NOT re-inline them. Note: `resume_mode=disable` and `total_epochs` are only defaults. When checkpoint optimization is enabled, llmzero's coder bounds **every** node by an explicit `trainer.total_training_steps` cap (with `total_epochs` set large enough to complete it); resumed nodes additionally get `resume_mode=resume_path` + `resume_from_path` to inherit the parent's global step.
5. Skip the LoRA/FFT template selection below.

**Otherwise** (no base config given), read the appropriate package-default template:
- LoRA: `constants/description_template_lora.txt`
- FFT: `constants/description_template_fft.txt`

### 3. Resolve Placeholders

Collect all values needed:

| Placeholder | Value |
|------------|-------|
| `{{MODEL_NAME}}` | e.g., `Qwen/Qwen3-4B` |
| `{{MODEL_PATH}}` | e.g., `/fsx/models/Qwen/Qwen3-4B` |
| `{{TRAIN_FILES}}` | e.g., `["/fsx/runs/LLMZeroCCRuns/run_.../data/train.parquet"]` |
| `{{VAL_FILES}}` | Val + test parquets, e.g., `["/fsx/.../data/val.parquet","/fsx/.../data/test.parquet"]` |
| `{{REWARD_FN_NAME}}` | `compute_score` |
| `{{REWARD_FN_PATH}}` | e.g., `/fsx/runs/LLMZeroCCRuns/run_.../reward_fn.py` |
| `{{PROJECT_NAME}}` | e.g., `ChemCoT_Qwen3-4B` (derived from dataset + model) |
| `{{EXPERIMENT_NAME}}` | `<RUN_NAME>_<PROJECT_NAME>` |
| `{{WANDB_API_KEY}}` | From constants or user input |
| `{{NNODES}}` | From constants |
| `{{N_GPUS_PER_NODE}}` | From constants |
| `{{OUTPUT_DIR}}` | `<verl_output_dir>/<run_name>` (from `verl_output_dir` in cluster.json) |
| `{{INSTALL_COMMANDS}}` | Read from `<run_folder>/install_commands.txt` if exists, format as shell commands, or empty |
| `{{TRAINER_LOGGER}}` | Backend-dependent (see below) |
| `{{SLURM_CONSTRAINTS}}` | Backend-dependent (see below) |

**Backend-dependent placeholders:**

- If `compute_backend == "raycluster"`: `{{TRAINER_LOGGER}}` = `["console","wandb"]` (Ray job submit propagates env vars via runtime_env)
- If `compute_backend` is `slurmdocker`, `conda`, or `docker`: `{{TRAINER_LOGGER}}` = `["console"]` (wandb API key cannot propagate to Ray worker processes spawned internally — causes immediate crash; tensorboard optional but not required)

- If `compute_backend == "slurmdocker"`: `{{SLURM_CONSTRAINTS}}` =
  ```
  **CRITICAL SLURM CONSTRAINTS (violating these causes immediate job failure):**
  - `trainer.logger` MUST be exactly `["console"]`. Adding "wandb" crashes Ray workers because WANDB_API_KEY cannot propagate to Ray actor processes on Slurm.
  - Do NOT export or reference WANDB_API_KEY anywhere in the script.
  - Do NOT use `ray job submit` wrapper — the script runs directly inside the Slurm allocation with Ray already bootstrapped.
  - For slurmdocker: Do NOT add conda activate or #SBATCH headers — the container and Slurm directives are managed by the backend wrapper.
  ```
- Otherwise: `{{SLURM_CONSTRAINTS}}` = empty string

For `{{TRAIN_FILES}}` and `{{VAL_FILES}}`, format as VERL expects:
- Single file: `["/path/to/file.parquet"]`
- Multiple files: `["/path/a.parquet","/path/b.parquet"]`

**Important**: `{{VAL_FILES}}` should include BOTH `val.parquet` and `test.parquet`. VERL evaluates all files in `val_files` during training. The val split (with `__dev` in `data_source`) is what llmzero tracks for MCTS optimization via `validation_metric_pattern: "val-core/.*__dev"`. The test split provides additional evaluation without affecting MCTS decisions.

For `{{INSTALL_COMMANDS}}`, if `<run_folder>/install_commands.txt` exists, format each line as a shell command with echo prefix, e.g.:
```bash
  echo "Installing dependencies..."
  pip install rdkit-pypi
  pip install sympy
```
If no install commands are needed, leave empty (remove the placeholder line entirely).

### 4. Adjust Thinking for Instruct Models

The template defaults to thinking enabled. If the model is an instruct variant (name contains `-Instruct`, `-instruct`, `-it`, or `-chat`), add `++data.chat_template_kwargs.enable_thinking=false` to the bash script CLI overrides to disable thinking. For non-instruct models, leave as-is (thinking stays enabled).

**Given Base Config branch:** the thinking flag key may differ by verl schema. Check the copied config: newer schema uses `data.apply_chat_template_kwargs.enable_thinking`, older uses `data.chat_template_kwargs.enable_thinking`. Override the key that already exists in the config with a plain `key=value` (no `++`). If the config already sets the desired value, no override is needed.

### 5. Write Description

Write the filled-in template to `<run_folder>/llmzero_input/description.txt`. Create the `llmzero_input/` directory if it doesn't exist.

### 6. Report

Tell the user:
- Path to generated description.txt
- Model and training mode selected
- Dataset files referenced
- Reward function referenced
- Hardware configuration
- If a base config was provided: the copied config path (`<run_folder>/training_config.yaml`) and that the script loads it via `--config-path`/`--config-name`
