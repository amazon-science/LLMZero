---
name: VeRL-GRPO - Prepare Dataset
description: "[VeRL-GRPO] Download and convert a dataset to VERL parquet format (prompt/data_source/reward_model), with train/val/test splits and the __dev val convention. Use to prepare data for a GRPO run."
---

# Prepare Dataset

Download and convert a dataset to VERL-compatible parquet format.

## Input

- **Dataset source**: HuggingFace dataset name (e.g., `IDEA-AI4S/ChemCoTDataset`) or local path
- **Run folder**: Path where processed data should be saved (e.g., `/fsx/runs/LLMZeroCCRuns/run_YYYYMMDD_HHMMSS/`)
- **User instruction** (optional): Hints about which columns to use, how to format prompts, etc.

## Steps

### 1. Download / Load Dataset

If HuggingFace name:
```python
from datasets import load_dataset
ds = load_dataset("<dataset_name>")
```

If local path, read with pandas/datasets.

### 2. Inspect Schema

Print column names, dtypes, and 2-3 sample rows. Identify:
- **Input column(s)**: The question/problem/instruction text
- **Answer column(s)**: The ground truth answer
- **Metadata columns**: Any extra info (data_source, task type, etc.)

Report findings to the user and explain your mapping decisions.

### 3. Convert to VERL Format

VERL expects parquet files with these columns stored as **native Python objects** (NOT JSON strings):
- `prompt`: A **list of dicts** (not a JSON string). Format: `[{"role": "user", "content": "<question>"}]`. Pandas stores this as an object column; parquet preserves it as a nested structure.
- `data_source`: A plain **string** used to route to the correct reward function. **See the data_source naming convention below.**
- `reward_model`: A **dict** (not a JSON string) containing at minimum `{"ground_truth": "<answer>", "style": "rule"}`. VERL accesses `data_item.non_tensor_batch["reward_model"]["ground_truth"]` directly as a dict lookup — a JSON string will cause `TypeError: string indices must be integers`.

If the dataset has additional metadata useful for scoring (e.g., multiple choice options, test cases), store them in a separate `extra_info` dict column.

### 4. Train/Val/Test Split

**Three splits are required**: train, val, and test.

**data_source naming convention** — this is critical for llmzero's MCTS metric tracking:
- **Train split**: `data_source` = `"<task_name>"` (e.g., `"gsm8k"`)
- **Val split**: `data_source` = `"<task_name>__dev"` (e.g., `"gsm8k__dev"`)
- **Test split**: `data_source` = `"<task_name>"` (e.g., `"gsm8k"`)

The `__dev` suffix on val data is **required** because llmzero's `validation_metric_pattern: "val-core/.*__dev"` uses it to identify the metric for MCTS node evaluation. Without it, llmzero cannot compare training iterations.

**Single aggregated val metric for LLMZero** — llmzero tracks only **one** validation metric (the first matching `val-core/.*__dev`). For datasets with multiple subtasks/categories, you MUST use a **single unified `data_source`** in val.parquet so that llmzero sees one aggregated score across all subtasks:
- **Val split**: ALL rows get `data_source` = `"<dataset_name>_agg__dev"` (e.g., `"<dataset_name>_agg__dev"`) — one metric for LLMZero
- **Test split**: Rows keep per-subtask `data_source` values (e.g., `"mol_und_fg_count"`, `"rxn_retro"`) — per-subtask metrics for analysis

Since `val_files` includes both val.parquet and test.parquet, VERL reports:
- `val-core/<dataset_name>_agg__dev/reward/mean@1` — the **one** aggregated metric llmzero optimizes
- `val-core/mol_und_fg_count/reward/mean@1`, etc. — per-subtask breakdown from test.parquet

The reward function must handle the unified `data_source` (e.g., `"<dataset_name>_agg__dev"`) by auto-detecting the answer type from the ground truth format rather than dispatching on the data_source name. See `/verl-grpo-define-reward` for the auto-detection pattern.

For single-task datasets (e.g., GSM8K), the train/val/test all share one `data_source` naturally, so this is only relevant for multi-subtask datasets.

**Splitting rules:**
- If the dataset has train/validation/test splits, use them directly
- If the dataset has train/test splits only, further split train into train (90%) and val (10%), keep test as-is
- If the dataset has only a train split, split into train (80%) / val (10%) / test (10%)
- Ensure splits are reproducible (use `seed=42`)

### 5. Save

```python
import pandas as pd
train_df.to_parquet("<run_folder>/data/train.parquet")
val_df.to_parquet("<run_folder>/data/val.parquet")
test_df.to_parquet("<run_folder>/data/test.parquet")
```

### 6. Report

Tell the user:
- Number of train/val/test examples
- Column mapping used (which source columns became prompt, ground_truth, etc.)
- The `data_source` values used for each split (confirm `__dev` on val)
- Sample prompt (first row) so they can verify formatting
- Any rows filtered out (and why)

## Dataset Format Reference

Columns must store **native Python objects**, not JSON strings:

```python
# Correct: native Python objects
row = {
    "prompt": [{"role": "user", "content": "What is 2+2?"}],       # list of dicts
    "data_source": "math_qa__dev",                                   # string
    "reward_model": {"ground_truth": "4", "style": "rule"},          # dict
}

# WRONG: JSON strings — will cause TypeError at runtime
row = {
    "prompt": '[{"role": "user", "content": "What is 2+2?"}]',      # string — BREAKS
    "reward_model": '{"ground_truth": "4"}',                         # string — BREAKS
}
```

Use `pd.DataFrame(rows).to_parquet(...)` — pandas preserves nested dicts/lists as object columns in parquet.

## Prompt Format and Output Instructions

**The prompt must include clear output format instructions** so the model produces answers the reward function can parse. This is critical for reward signal quality.

If the dataset's raw queries already include output format instructions (e.g., "respond with JSON", "put your answer in \boxed{}"), use them as-is.

If the dataset has **no output format instructions**, you MUST add them. Options:
- **System prompt**: Add a system message like `{"role": "system", "content": "You are an expert. Answer precisely and concisely. Output only the final answer."}` as the first message
- **User prompt suffix**: Append instructions to the user message (e.g., "Put your final answer in \boxed{}")

The choice depends on the task:
- For tasks with short answers (numbers, yes/no, SMILES): system prompt with "Output only the final answer" works well
- For tasks where the model needs to show work: suffix with `\boxed{}` or `#### answer` format works better
- For tasks expecting JSON output: include the JSON schema in the prompt

**Important**: The output format instructions in the prompt and the reward function's extraction logic must be aligned. This alignment is validated by the pre-flight LLM evaluation step in `/verl-grpo-run-training`.

## Notes

- Write the conversion as a Python script at `<run_folder>/prepare_data.py` and execute it, so the process is reproducible
- If the dataset is very large (>100k rows), consider subsampling for the validation and test sets
- Handle edge cases: missing values, empty strings, non-UTF8 text
- **Deduplicate all splits by prompt** — VERL groups responses by identical prompts and computes pass@N where N = number of duplicates. The `val-core` prefix only includes the max @N metric. If even one prompt is duplicated (N>1), `val-core` will report `mean@2` or `mean@3` instead of `mean@1`, which is near-zero and breaks llmzero's MCTS metric tracking. **Group by prompt first, merge ground truths**: if duplicate prompts have the same answer, keep one row; if they have different valid answers (e.g., multiple valid molecular optimizations), store as a JSON list in `ground_truth` (e.g., `'["ans1", "ans2"]'`). The reward function must then accept either a single string or a JSON list and score correct if any answer matches.
- Avoid adding extra columns with mixed types (e.g., `extra_info` dicts with varying nested structures) — pyarrow can fail with `ArrowInvalid: cannot mix list and non-list values`. Stick to the three required columns: `prompt`, `data_source`, `reward_model`
- For gated HuggingFace datasets, use `load_dataset(..., token='...')` or `huggingface-cli login` first. The HF token is stored in `constants/cluster.json` as `hf_token`
