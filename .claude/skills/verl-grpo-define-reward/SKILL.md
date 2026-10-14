---
name: VeRL-GRPO - Define Reward
description: "[VeRL-GRPO] Write a compute_score reward function VERL calls to score rollouts during GRPO training. Use after preparing the dataset. Covers think-tag stripping, extraction, and prompt-reward alignment."
---

# Define Reward Function

Write a custom reward function for the dataset that VERL will use to score model outputs during training.

## Input

- **Run folder**: Path containing the prepared dataset (e.g., `/fsx/runs/LLMZeroCCRuns/run_YYYYMMDD_HHMMSS/`)
- **Dataset info**: Column schema, sample rows, task type (from prepare-dataset step)
- **User instruction** (optional): Guidance on scoring logic

## Steps

### 1. Analyze the Task

Read a few sample rows from the prepared parquet file to understand:
- What kind of answers are expected (numeric, text, multiple choice, code, etc.)
- How ground truth is formatted
- Whether partial credit makes sense

### 2. Write the Reward Function

Create `<run_folder>/reward_fn.py` with the following structure:

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    Score model output against ground truth.

    Args:
        data_source: The data source identifier (e.g., "gsm8k", "gsm8k__dev")
        solution_str: The model's generated response (string)
        ground_truth: The expected answer (string, from reward_model.ground_truth)
        extra_info: Optional dict with additional metadata
        **kwargs: Additional keyword arguments (ignored)

    Returns:
        float: Reward score (typically 0.0 for wrong, 1.0 for correct)
    """
    # 1. Extract the answer from the model's response
    # 2. Compare with ground truth
    # 3. Return score
```

### 3. Answer Extraction

The model response may contain thinking (`<think>...</think>` tags) and formatting.

**CRITICAL: Strip `<think>` tags first, then extract from the visible output.** The thinking process is internal reasoning, not the model's actual output to the user. GRPO rewards the output quality, so the reward function must evaluate only what the user would see (the content after `</think>`).

```python
import re
# Remove thinking tags — the reward evaluates the MODEL OUTPUT, not internal reasoning
cleaned = re.sub(r'<think>.*?</think>', '', solution_str, flags=re.DOTALL).strip()
```

**After stripping think tags, extract the answer from the remaining content.** The extraction logic must match the output format the model was instructed to produce (see "Prompt-Reward Alignment" below). Common extraction patterns:

- **Numeric/math**: Look for `\boxed{answer}` or `#### answer` patterns
- **Multiple choice**: Look for `(A)`, `(B)`, etc. or `Answer: X`
- **Free text**: Look for `answer is X` / `Answer: X` patterns, then fall back to last non-empty line
- **JSON/structured**: Parse the JSON structure and extract the answer field (e.g., `json.loads(cleaned)["count"]`)

**Robust extraction order** (apply in sequence, return first match):
1. `\boxed{...}` pattern
2. `answer is X` / `answer: X` / `Answer: X` natural language patterns
3. JSON field extraction if output is JSON-formatted
4. Last non-empty line (fallback)

**Common pitfall**: If the model outputs structured JSON like `{"count": 1}`, a naive "last line" fallback extracts `}` instead of `1`. Always try JSON parsing before falling back to last line.

### 3b. Prompt-Reward Alignment

**The prompt's output format instructions and the reward function's extraction logic MUST be co-designed.** If they are misaligned, the model may produce correct answers that the reward function cannot parse, resulting in zero reward signal and failed training.

Rules:
- If the dataset has no output format instructions, add a system prompt specifying the expected format (e.g., "Output only the final answer", "Put your answer in \\boxed{}", "Respond with JSON: {\"answer\": ...}")
- The reward function's `_extract_answer` must handle the format the model will actually produce given those instructions
- Document the expected output format in a comment at the top of `reward_fn.py`
- **This alignment is validated by the pre-flight LLM evaluation** (see run-training step 3b). If the pre-flight eval shows near-zero accuracy, the extraction logic or prompt format likely needs adjustment

### 4. Scoring Logic

Common patterns:
- **Exact match**: `1.0 if extracted == ground_truth else 0.0`
- **Numeric tolerance**: `1.0 if abs(float(extracted) - float(ground_truth)) < 1e-6 else 0.0`
- **Normalized match**: Lowercase, strip whitespace, then compare
- **Partial credit**: For multi-part answers, score proportionally

### 5. Dependency Handling

If the reward function needs additional packages (e.g., `rdkit` for chemistry, `sympy` for math):
- Note the pip install commands needed
- These will be added to the description.txt as install commands that run before training
- Write the install commands to `<run_folder>/install_commands.txt`, one per line:
  ```
  pip install rdkit-pypi
  pip install sympy
  ```

### 6. Test the Function

Test the reward function locally with 2-3 sample rows from the dataset:
```python
# Load a few examples
import pandas as pd
df = pd.read_parquet("<run_folder>/data/val.parquet")
row = df.iloc[0]

# Simulate a correct answer
score = compute_score(
    solution_str=row["reward_model"]["ground_truth"],  # pretend model got it right
    ground_truth=row["reward_model"]["ground_truth"]
)
print(f"Score for correct answer: {score}")  # Should be 1.0
```

### 7. Report

Tell the user:
- Task type identified (math, QA, code, etc.)
- Scoring approach (exact match, numeric, etc.)
- Any dependencies needed
- Test results on sample data

## Reward Function Conventions

- Function MUST be named `compute_score` (this is what VERL calls)
- MUST accept `data_source` as the **first** argument, then `solution_str`, `ground_truth`, `extra_info=None`, `**kwargs`
- VERL calls: `compute_score(data_source=..., solution_str=..., ground_truth=..., extra_info=...)`
- MUST return a `float` (or a dict with a `"score"` key)
- Handle exceptions gracefully (return 0.0 on parse errors, not crash)
- The `ground_truth` comes from `reward_model["ground_truth"]` in the parquet (a native dict, not JSON string). The `reward_model` dict must also include `"style": "rule"`.
- The `extra_info` comes from `extra_info` column in the parquet (or empty dict if absent)
- `ground_truth` can be a single string or a JSON list of valid answers (e.g., `'["ans1", "ans2"]'`). If a list, score correct if any answer matches. This handles prompts with multiple valid solutions (common in optimization tasks).

## Dispatcher Pattern

If training on multiple datasets with different scoring logic:

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if "gsm8k" in data_source:
        return _score_math(solution_str, ground_truth)
    elif "code" in data_source:
        return _score_code(solution_str, ground_truth)
    return 0.0
```

Note: Both train (`"gsm8k"`) and val (`"gsm8k__dev"`) data_sources are passed to the same function, so use `in` or `startswith` checks rather than exact matches.

## Unified Val data_source and Auto-Detection Pattern

For multi-subtask datasets, val.parquet uses a single unified `data_source` (e.g., `"<dataset_name>_agg__dev"`) so llmzero sees one aggregated validation metric. The reward function cannot dispatch by subtask for these val samples — it must **auto-detect the answer type from the ground truth format**.

Train samples still use per-subtask `data_source` values, so the dispatcher pattern above still applies for training. The auto-detection is a fallback for the unified val `data_source`.

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    answer = _extract_answer(solution_str)

    # Route by subtask for train data (per-subtask data_source)
    ds = data_source.replace("__dev", "")
    if ds in SMILES_SUBTASKS:
        return _score_smiles(answer, ground_truth)
    elif ds in NUMERIC_SUBTASKS:
        return _score_numeric(answer, ground_truth)
    elif ds in TEXT_SUBTASKS:
        return _score_exact_text(answer, ground_truth)

    # Unified val data_source (e.g., "<dataset_name>_agg__dev") — auto-detect from ground truth
    return _auto_detect_and_score(answer, ground_truth)

def _auto_detect_and_score(answer, ground_truth):
    """Auto-detect answer type from ground truth format and score accordingly."""
    gt = str(ground_truth).strip()

    # Text classification: Yes/No, short text labels
    if gt.lower() in ("yes", "no", "true", "false"):
        return _score_exact_text(answer, ground_truth)

    # Numeric: integer or float
    try:
        float(gt)
        return _score_numeric(answer, ground_truth)
    except (ValueError, TypeError):
        pass

    # Default: SMILES comparison (covers molecules, reactions, scaffolds)
    # Fall back to text match if SMILES comparison fails
    score = _score_smiles(answer, ground_truth)
    if score > 0:
        return score
    return _score_exact_text(answer, ground_truth)
```

This pattern ensures the reward function works correctly for both per-subtask train data and unified val data.
