# LLMZero Benchmarks

LLMZero applies **agentic test-time scaling** to RL post-training: it spends inference-time
compute, an LLM-agent Monte-Carlo Tree Search (MCTS) over many training runs, to discover a
better **training strategy** on top of a training framework. Each tree node is one training
run (trained from scratch or continuing from a parent node's checkpoint), and the search's
LLM-powered proposer evolves the strategy (the config and
hyperparameters, plus early-stopping decisions, checkpoint selection, and when to warm-start
from a parent checkpoint) from node to node to maximize a validation metric. These benchmarks
measure what that search buys you, end to end, on two public tasks and two training frameworks
(verl 0.9.0 GRPO and SFT).

This repo is the official implementation of the EMNLP 2026 (Findings) paper
([arXiv:2606.18388](https://arxiv.org/abs/2606.18388)), with the tree search upgraded to
**[ExTS](https://github.com/amazon-science/ExTS)** ([arXiv:2608.23848](https://arxiv.org/abs/2608.23848)).

The paper's broader results ran on an internal edition of verl that cannot be open-sourced.
For this release we integrated the public **verl 0.9.0** stack and ran two new benchmarking
runs on it (GRPO on SSMR-Bench and SFT on GovReport). The two tables below are from those
fresh verl-0.9.0 runs, distinct from the paper's larger benchmark suite.

Every search starts from the **verl-default strategy** (verl's default hyperparameters in the
reference entry-point script written into `description.txt`) and lets the MCTS proposer
discover better strategies from that baseline. We report:

- **Base**: the untrained `Qwen/Qwen3-4B` model (step 0, before any training).
- **No search**: the root node (`node_0`), i.e. the verl-default strategy trained once.
- **LLMZero MCTS search**: the best node the search found.

## Setup (common)

- **Engine:** LLMZero MCTS, sync mode (GRPO search: 10 nodes; SFT search: 16 nodes).
- **Training stack:** verl 0.9.0 (Docker/enroot image `llmzero-verl-0.9.0.sqsh`,
  base `verlai/verl` stable vLLM image, vendored from verl's `Dockerfile.stable.vllm`).
- **Model:** `Qwen/Qwen3-4B` (text, thinking-enabled). Chosen over the newer Qwen3.5-4B,
  which is MoE + multimodal + hybrid-attention and does not cleanly load/train on the
  verl-0.9.0-pinned stack (SFT loss-masking hits the strict multimodal chat template).
  Qwen3-4B is a clean, reproducible text model that matches both tasks.
- **Search LLM:** Claude Opus 4.8 (Bedrock), extended thinking on.
- **Compute:** 1× `ml-p4d-24xlarge` (8× A100-40GB), Slurm + enroot (`slurmdocker` backend),
  `trainer.nnodes=1`, `trainer.n_gpus_per_node=8`.

> **Note on node count.** These runs use a single compute node. The `slurmdocker` backend's
> multi-node path (Ray-worker bootstrap for GRPO; torchrun rendezvous for SFT) is a backend
> item that was not exercised here, so both benchmarks were pinned to one node. This is a
> backend-support detail, not a limitation of the frameworks or of LLMZero.

## GRPO on SSMR-Bench (music-theory multiple choice)

- **Framework:** `verl_grpo` (RL post-training, `python -m verl.trainer.main_ppo`).
- **Data:** SSMR-Bench subsample: train 5000, val 500 (`ssmr_agg__dev`, a single
  aggregated validation `data_source`), test 500 (kept per-category: `ssmr_scale`,
  `ssmr_chord`, `ssmr_interval`, `ssmr_beat`). Reward = exact match on a `\boxed{A|B|C|D}`
  letter (1.0 correct / 0.0 otherwise), after stripping `<think>…</think>`.
- **Adapter / lengths:** LoRA rank 64, `max_prompt_length=4096`, `max_response_length=8192`
  (long enough for the thinking trace to finish; a too-short response cap truncates
  thinking and depresses accuracy).
- **Checkpointing:** ENABLED. verl-0.9.0 native **LoRA-only** checkpoint save/resume; the
  strategy can deepen a strong node by warm-starting from a parent node's checkpoint.
- **Validation metric:** `val-core/ssmr_agg__dev/acc/mean@1` (accuracy, maximize).

Both `val.parquet` (aggregated `ssmr_agg__dev`) and `test.parquet` (per-category) are in
the GRPO `val_files`, so the held-out test accuracy is read directly from each node's
validation logs at its best-val step, with no separate offline eval needed.

This search ran to **10 completed nodes** (LoRA rank 64, verl 0.9.0, with working checkpoint
warm-start). The best-node results are below.

| | val (`ssmr_agg__dev`) acc | test (per-category avg) acc |
|---|---|---|
| Base Qwen3-4B (untrained, step 0) | 0.504 | 0.484 |
| No search (`node_0`, verl-default strategy, 100 steps) | 0.736 | 0.682 |
| **LLMZero MCTS search (best node)** | **0.840** | **0.842** |

Per-category test accuracy at the best node: scale 0.928, beat 0.936, chord 0.792,
interval 0.712 (untrained base: scale 0.552, beat 0.400, chord 0.488, interval 0.496). The
search lifts aggregated validation accuracy from 50.4% (untrained) to **84.0%**: +33.6
points over the untrained base and +10.4 points over the verl-default strategy (no search),
with a matching lift on the held-out test split (48.4% → 84.2%).

The training strategy the search discovered departed from the verl default (lr 5e-5,
`rollout.n=8`, `kl_loss_coef=0.001`, 100 steps) by lowering the learning rate (3.5e-5),
doubling rollout samples (`rollout.n=16`), tightening the KL penalty (0.0008), and
**warm-starting from the parent node's step-200 checkpoint** before training on to step 300.
This exercises the LoRA-only checkpoint-chaining path end to end: a strong node is deepened
rather than retrained from scratch.

## SFT on GovReport (long-context summarization)

- **Framework:** `verl_sft` (supervised fine-tuning, `torchrun -m verl.trainer.sft_trainer`).
- **Data:** GovReport-SFT: train 17,457, val 500 (`govreport_sft__dev`), test 500.
  `messages` schema; cross-entropy loss on assistant (summary) turns only.
- **Long-context:** `data.max_length≈32768`, Ulysses sequence-parallel SP=8, dynamic
  batching, set so the gold summary is never truncated under right-truncation (p95 input
  ≈ 24.5k tokens).
- **Checkpointing:** DISABLED (single-shot; every node trains from base weights).
- **Validation metric:** `val/loss` (validation cross-entropy, minimize). Generation
  quality on the held-out test set is scored offline with `scrolls_score` (ROUGE geo-mean,
  0 to 100).

This search completed all **16 iterations**.

**Validation loss** (lower is better):

| | val/loss |
|---|---|
| Base Qwen3-4B (untrained) | 1.1338 |
| No search (`node_0`, verl-default strategy) | 1.0420 |
| **LLMZero MCTS search (best node)** | **1.0072** |

**Held-out TEST scrolls_score** (offline vLLM generation on the test split, scored with the
dataset's `compute_score`; the LoRA best-val checkpoint merged + fused to a standalone model
first). n=493 of 500 (7 prompts over the 30k-token eval-context cap were excluded, the same
exclusion for all three models):

| | test scrolls_score |
|---|---|
| Base Qwen3-4B (untrained) | 22.86 |
| No search (`node_0`, verl-default strategy) | 31.74 |
| **LLMZero MCTS search (best node)** | **34.66** |

The checkpoint the search selected improves held-out test scrolls_score from 22.86 (untrained
base) to **34.66**: +11.8 points over the untrained base and +2.9 over the verl-default
strategy (no search), and the val/loss ordering tracks it (1.1338 → 1.0420 → 1.0072). The
training strategy the search discovered reshaped lr / scheduler / epochs / LoRA rank from
verl's defaults. (The offline generation eval used the best-val node at eval time, val/loss
1.0072; the search's overall lowest val/loss was fractionally better, 1.0046, on a later node
that was not separately test-evaluated.)

## Reproducing

Each benchmark is a self-contained **run folder** produced by the LLMZero RL-post-training
skills pipeline. Rather than assembling data / reward / config / description by hand, use
the skills:

- `/verl-grpo-run-training`: end-to-end orchestrator (dataset → report).
- `/verl-grpo-prepare-dataset`: download a HF dataset and convert to verl parquet format
  (`prompt`, `data_source`, `reward_model`; val `data_source` suffixed `__dev`).
- `/verl-grpo-define-reward`: write the `compute_score` reward function.
- `/verl-grpo-generate-task`: render `description.txt` from the template + run config.
- `/general-download-model`: fetch `Qwen/Qwen3-4B` to the model prefix.
- `/verl-grpo-validate-run`: pre-flight LLM eval + mid-training health checks.
- `/general-gather-results` / `/general-plot-pareto`: parse output, write the report, plot
  the score progression across MCTS nodes.

The resulting run folder follows this layout:

```
<run_folder>/
  data/                      # train.parquet, val.parquet, test.parquet, prepare_data.py
  reward_fn.py               # compute_score reward function
  install_commands.txt       # extra pip deps (e.g. rouge_score, nltk for GovReport)
  llmzero_config.yaml         # MCTS + per-agent LLM + compute-backend config
  llmzero_input/
    description.txt           # task description + verl-default reference command (SACRED: only this file)
  llmzero_output/             # MCTS tree, per-node logs/scripts, observer_results_summary.*
  report.md                   # final training report
```

Point the `llmzero` CLI at the run folder (the `description.txt` carries the verl-default
reference command and the tunable/fixed hyperparameter contract that bounds the strategy
space; the MCTS proposer discovers strategies from there):

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

Per-node validation scores land in `<run_folder>/llmzero_output/observer_results_summary.txt`
(and `.json`); each node's raw training stdout/stderr is under
`<run_folder>/llmzero_output/node_<i>/states/`.
