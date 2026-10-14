# Claude Code Skills

Skills that orchestrate the RL post-training pipeline on top of the llmzero engine.

**Naming convention:** the directory-name prefix encodes the category —
`general-*` skills are framework-agnostic (engine, compute backends, model store,
run folder); `verl-grpo-*` skills assume the VERL parquet schema, a `compute_score`
reward function, GRPO rollouts, and LoRA/FFT description templates. Invocation name
= directory name (e.g. `/general-download-model`). Category subdirectories are NOT
used: Claude Code discovers only direct children of `.claude/skills/`, so the prefix
carries the grouping.

## General (framework-agnostic)

| Skill | Purpose |
|-------|---------|
| `/general-download-model` | Download an HF model to `<model_path_prefix>/`. |
| `/general-launch-llmzero` | Start the llmzero MCTS loop via the configured compute backend. |
| `/general-check-training` | Check job/session status across backends. |
| `/general-gather-results` | Parse output and write a summary report. |
| `/general-plot-pareto` | Plot Time/Cost vs Performance Pareto curves across runs. |
| `/general-add-training-framework` | Scaffold a new training-framework plugin from a real log. |

## VeRL-GRPO (RL post-training)

| Skill | Purpose |
|-------|---------|
| `/verl-grpo-run-training` | End-to-end orchestrator: dataset → report. |
| `/verl-grpo-prepare-dataset` | Download HF dataset, convert to VERL parquet. |
| `/verl-grpo-define-reward` | Write a `compute_score` reward function. |
| `/verl-grpo-generate-task` | Create `description.txt` from a template + run config. |
| `/verl-grpo-validate-run` | Health-check run folder + pre-flight eval + mid-training diagnostics. |

`/verl-grpo-run-training` drives the VeRL-GRPO pipeline end to end, invoking the
other verl-grpo skills plus the general download/launch/check/gather skills.
