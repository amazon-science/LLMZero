---
name: General - Gather Results
description: "[General] Parse llmzero output and write a summary report after training completes. Framework-agnostic run summary; RFT/GRPO-specific report sections apply when the run is a VeRL-GRPO run."
---

# Gather Results

Parse llmzero output and write a summary report after training completes.

## Input

- **Run folder**: Path to the completed run (e.g., `/fsx/LLMZeroCCRuns/run_YYYYMMDD_HHMMSS/`)

## Steps

### 1. Find LLMZero Output

Look for the llmzero output directory:
```bash
ls <run_folder>/llmzero_output/
```

### 2. Extract Results

From the llmzero output, find:
- **Best validation score**: Search for validation metric values across all nodes
- **Best checkpoint path**: The path to the best model checkpoint
- **Number of iterations**: How many MCTS nodes were explored
- **Training time**: Total wall clock time

Check the llmzero logs:
```bash
find <run_folder>/llmzero_output/ -name "*.log" -o -name "*.txt" | head -20
```

Look through node directories for validation scores and checkpoint paths.

### 3. Check Ray Job Outputs

For each training run that was submitted, check the final metrics:
```bash
# Look for validation score patterns in logs
grep -r "val-core/" <run_folder>/llmzero_output/ 2>/dev/null | tail -20
grep -r "Best checkpoint" <run_folder>/llmzero_output/ 2>/dev/null
grep -r "best_validation" <run_folder>/llmzero_output/ 2>/dev/null
```

### 4. Generate Pareto Plot

Invoke `/general-plot-pareto` on the run to generate `<run_folder>/pareto_curves.png`.

### 5. Write Report

Note: sections 3 (Reward Function) and the base-vs-trained accuracy table in
section 4 apply to RFT/GRPO runs (see /verl-grpo-run-training). For non-RFT
frameworks, populate only the sections that apply.

Create `<run_folder>/report.md` with these sections:

1. **Run Details** — model, dataset name, hardware, training mode (lora/fft), total wall-clock time
2. **Dataset Description** — what the task is (e.g., "biomedical factoid QA"), input/output format, evaluation metric, 1-2 example question+answer pairs, data splits and sizes, notable properties (multi-answer, categories, domain)
3. **Reward Function** — how `compute_score` works: answer extraction logic (think-tag stripping, pattern matching, fallbacks), matching strategy (exact match, substring, normalization), edge cases handled, and why those choices were made given the dataset
4. **Results** — base model vs best trained accuracy table, per-iteration scores table with best val and duration, best checkpoint path
5. **Pareto Plot** — `![Pareto](pareto_curves.png)` with cost assumptions (GPU $/hr, API $/MTok)
6. **Key Design Decisions** — prompt design, hyperparameter choices, issues encountered and fixes applied
7. **Files** — list of all run artifacts with brief descriptions

### 6. Report to User

Print the report path, best checkpoint path, and overall success status.

**Note (VeRL-GRPO):** a Ray job may show FAILED despite success (WandB atexit
BrokenPipeError). See /verl-grpo-validate-run for the `Best checkpoint folder
created` check.
