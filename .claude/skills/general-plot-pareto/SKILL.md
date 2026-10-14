---
name: General - Plot Pareto
description: "[General] Plot Time/Cost vs Performance Pareto curves across one or more llmzero runs. Framework-agnostic analysis of MCTS node scores."
---

# Plot Pareto Graph

Plot Pareto-style curves showing Time/Cost vs Performance across one or more LLMZero runs.

## Input

- **Run paths**: One or more llmzero output directories (e.g., `/fsx/runs/LLMZeroCCRuns/run_.../llmzero_output`)
- **Run names** (optional): Display names for each run. Default: folder name
- **Output path** (optional): Where to save the PNG. Default: `<first_run>/pareto_curves.png`
- **GPU cost** (optional): GPU cost per hour in $. Default: `4 * 21.957642` (4 nodes)
- **API input cost** (optional): API input cost per MTok in $. Default: 5.0
- **API output cost** (optional): API output cost per MTok in $. Default: 25.0
- **Smoothing** (optional): Smoothing factor in (0,1]. Default: 0.3

## Steps

### 1. Resolve Run Paths

If the user provides a run folder (e.g., `/fsx/runs/LLMZeroCCRuns/run_YYYYMMDD_HHMMSS`), the llmzero output directory is `<run_folder>/llmzero_output/`. The tool needs the llmzero output directory specifically — the one containing `timing_log.json`, `token_usage.json`, and `node_*` directories.

Verify the required files exist:
- `<run_path>/timing_log.json`
- `<run_path>/token_usage.json`
- At least one `<run_path>/node_*/states/best_validation_metrics.json`

If any are missing, the run may still be in progress or hasn't completed enough nodes. Warn the user.

### 2. Adjust GPU Cost for Actual Node Count

The default GPU cost assumes 4 nodes. If the cluster config (`constants/cluster.json`) shows a different `nnodes` value, scale accordingly:
```
gpu_cost_per_hour = nnodes * 21.957642
```

### 3. Run the Plot Tool

```bash
python3 <llmzero_path>/tools/plot_pareto.py \
  --run-paths <path1> [<path2> ...] \
  --run-names <name1> [<name2> ...] \
  --gpu-cost <gpu_cost_per_hour> \
  --api-input-cost <input_cost> \
  --api-output-cost <output_cost> \
  --smooth <smoothing_factor> \
  --output <output_path>
```

### 4. Report

Tell the user:
- Path to the saved plot PNG
- Display the plot image so the user can see it
- Summary of what's shown: number of runs compared, metrics used, cost assumptions
