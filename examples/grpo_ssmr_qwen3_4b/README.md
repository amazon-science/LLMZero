# Example: GRPO LoRA post-training of Qwen3-4B on SSMR-Bench

This is a self-contained, reproducible example of **agentic test-time scaling for
RL post-training**: LLMZero spends inference-time compute on an LLM-agent search over
training runs to discover a stronger training strategy. The concrete job is an
**LLMZero GRPO run**: LoRA
(rank = alpha = 64) reinforcement fine-tuning of `Qwen/Qwen3-4B` on
**SSMR-Bench**, a music-theory multiple-choice benchmark (beat / chord /
interval / scale, in ABC notation). This run uses the **textual** (ABC-notation)
subset. The reward is a rule-based exact match on a `\boxed{A|B|C|D}` letter.
(Obtain SSMR-Bench from its upstream source and prepare the splits as described
in [Prepare the data](#prepare-the-data).)

LLMZero runs a Monte-Carlo Tree Search (MCTS) over training runs: each tree node is
one verl training run (from scratch or continuing from a parent node's checkpoint),
and an LLM-powered proposer evolves the full **training
strategy** node to node to maximize validation accuracy (`val-core/ssmr_agg__dev`).
The strategy is the complete recipe the agents co-optimize: the RL/optimization
hyperparameters (lr, `rollout.n`, KL penalty, LoRA rank) plus early-stopping,
checkpoint selection, and when to warm-start a strong node from its parent's
checkpoint. This example ships the exact inputs that drove a real run so you can read
and reproduce it.

- **Framework:** `verl_grpo` (verl 0.9.0 GRPO, `python3 -m verl.trainer.main_ppo`).
- **Model:** `Qwen/Qwen3-4B` (text, thinking-enabled). Rollouts via vLLM; only the
  LoRA adapter is updated with GRPO advantages, base weights frozen.
- **Compute:** 1 × `ml-p4d-24xlarge` (8 × A100-40GB), Slurm + enroot
  (`slurmdocker` backend), verl 0.9.0 container (`docker/llmzero-verl-0.9.0.sqsh`).
- **Checkpointing:** ENABLED: verl-0.9.0 native **LoRA-only** checkpoint
  save/resume, so the strategy the proposer discovers can warm-start (deepen) a
  strong node from its parent's checkpoint instead of retraining from scratch.

## Results

From the benchmark run that produced these inputs (see
[`docs/benchmarks.md`](../../docs/benchmarks.md) for full detail):

| | val (`ssmr_agg__dev`) acc | test (per-category avg) acc |
|---|---|---|
| Base Qwen3-4B (untrained, step 0) | 0.504 | 0.484 |
| No search (`node_0`, verl-default-style HP, 100 steps) | 0.736 | 0.682 |
| **LLMZero MCTS search (best node)** | **0.840** | **0.842** |

The strategy the search discovered lowered the learning rate (3.5e-5), doubled
rollout samples (`rollout.n=16`), tightened the KL penalty (0.0008), and
**warm-started from the parent node's step-200 checkpoint** before training on to
step 300, exercising the LoRA-only checkpoint-chaining path end to end.

## Files in this folder

| File | What it is |
|------|------------|
| `description.txt` | The LLMZero task description. Carries the TUNED LoRA GRPO reference command (the MCTS starting point), the fixed-vs-tunable hyperparameter contract, the LoRA-only checkpoint save/resume protocol, and verl-0.9.0 schema notes. This is the single "sacred" file LLMZero reads from `llmzero_input/`. |
| `reward_fn.py` | The `compute_score(data_source, solution_str, ground_truth, ...)` reward function: strips `<think>…</think>`, extracts the last `\boxed{X}`, returns 1.0 if the letter matches ground truth else 0.0. |
| `llmzero_config.yaml` | MCTS parameters, per-agent LLM config (Claude Opus 4.8 on Bedrock), compute-backend (`slurmdocker`) config, checkpoint-optimization and early-stopper settings. |
| `install_commands.sh` | Extra pip dependencies for this task (none, since SSMR-Bench needs no extra packages beyond the verl image). |
| `verl090_default_config.yaml` | Reference only: the exact default config verl 0.9.0 resolves (`main_ppo --cfg job --resolve` with no overrides). Shows each key's exact name / default / nesting so you can see what the reference command overrides. NOT loaded at runtime. |

> **Paths are genericized.** Absolute paths in `description.txt` and
> `llmzero_config.yaml` use `/fsx/users/example/...` and `${HF_TOKEN}`. Replace them
> with your own run folder, model prefix, container image path, and HF token before
> launching.

## Prepare the data

The `data/*.parquet` files are **not bundled** (keep the example compact). Regenerate
them from the public dataset. The run used:

- **Train:** 5000 rows, per-category `data_source`
  (`ssmr_interval`, `ssmr_beat`, `ssmr_scale`, `ssmr_chord`).
- **Val:** 500 rows, single aggregated `data_source` `ssmr_agg__dev`
  (LLMZero tracks ONE validation metric; the `__dev` suffix matches
  `validation_metric_pattern: "val-core/.*__dev"`).
- **Test:** 500 rows, kept per-category (`ssmr_scale`, `ssmr_chord`,
  `ssmr_interval`, `ssmr_beat`) for per-category test visibility. Both `val.parquet`
  and `test.parquet` go in verl's `val_files`, so held-out test accuracy is read
  directly from each node's validation logs.

Each parquet row has exactly three columns: `prompt` (list of `{role, content}`
chat dicts: a system message instructing the model to reason and emit a single
letter in `\boxed{X}`, plus a user message with the ABC-notation context and four
options), `data_source` (string), and `reward_model`
(`{"ground_truth": "<A|B|C|D>", "style": "rule"}`). Values must be native Python
objects, not JSON strings.

The easiest way to regenerate is the dataset-prep skill, which downloads the HF
dataset and converts it to the verl parquet format described above:

```
/verl-grpo-prepare-dataset
```

Point it at the SSMR-Bench source (its textual / ABC-notation subset), cap val and
test at ~500 rows each (stratified), suffix the val `data_source` with `__dev`, and
use the single aggregated `ssmr_agg__dev` for val so LLMZero sees one validation
score. Write the output to `<run_folder>/data/{train,val,test}.parquet`.

You also need the model downloaded once (to the model prefix referenced in
`description.txt`):

```
/general-download-model          # fetches Qwen/Qwen3-4B
```

## Launch

Assemble a run folder with the layout LLMZero expects (`llmzero_input/` must contain
ONLY `description.txt`):

```
<run_folder>/
  data/{train,val,test}.parquet   # from /verl-grpo-prepare-dataset
  reward_fn.py                    # this file
  llmzero_config.yaml             # this file
  llmzero_input/
    description.txt               # this file (the ONLY file in llmzero_input/)
  llmzero_output/                 # created by the run
```

Edit the genericized paths in `description.txt` (data / model / checkpoint-output
dirs) and in `llmzero_config.yaml` (`container_image`, `HF_TOKEN`) to match your
environment.

### Via the skills pipeline (recommended)

The end-to-end orchestrator drives dataset prep → reward → task description → model
download → launch → report:

```
/verl-grpo-run-training
```

Or run the individual skills (`/verl-grpo-prepare-dataset`,
`/verl-grpo-define-reward`, `/verl-grpo-generate-task`, `/general-download-model`,
`/verl-grpo-validate-run`, `/general-launch-llmzero`, `/general-gather-results`).

### Via the `llmzero` CLI directly

```bash
llmzero -i <run_folder>/llmzero_input \
        -o <run_folder>/llmzero_output \
        -c <run_folder>/llmzero_config.yaml \
        --compute-backend slurmdocker \
        --slurm-partition <partition> \
        --docker-image /path/to/llmzero-verl-0.9.0.sqsh \
        --mount-dir /fsx:/fsx \
        --slurm-nodes 1 --slurm-gpus-per-node 8 \
        -n 16 -v 2
```

(The `slurmdocker` compute block is also fully specified in `llmzero_config.yaml`;
the backend requires `container_image`/`container_mounts`, which the config sets
explicitly.)

Per-node validation scores land in
`<run_folder>/llmzero_output/observer_results_summary.txt` (and `.json`); each node's
raw training stdout/stderr is under `<run_folder>/llmzero_output/node_<i>/states/`.
