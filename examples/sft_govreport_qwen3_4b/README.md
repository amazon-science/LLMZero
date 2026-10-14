# SFT: GovReport (long-context summarization), Qwen3-4B

A self-contained, reproducible example of **agentic test-time scaling for post-training**:
LLMZero runs an MCTS over training runs that spends inference-time compute to discover a
stronger **training strategy** for a **verl 0.9.0 supervised fine-tuning (SFT)** job: LoRA
fine-tuning of `Qwen/Qwen3-4B` on **long-context U.S. government-report summarization**
(GovReport-SFT), minimizing validation cross-entropy (`val/loss`).

Each MCTS tree node is one full SFT run (LLMZero in general supports continuous runs that
warm-start from a parent node's checkpoint, but this example runs single-shot: every node
trains from the base model); an LLM-powered proposer evolves the training
strategy from node to node, starting from verl's **default** SFT hyperparameters. The
strategy spans the recipe the agents co-optimize (learning rate, scheduler, epochs, LoRA
rank) together with early-stopping and checkpoint selection. See
[`docs/benchmarks.md`](../../docs/benchmarks.md) for the measured results on this task.

> **Fully automated, no human intervention.** The `verl_sft` integration itself (the
> training-framework plugin, data/reward/task scaffolding, and launch/reporting glue) was
> built end to end by Claude Code with Opus 4.8 driving the Agent Skills, with **no human
> edits** to any generated artifact. GovReport was then run on that integration as a
> subsequent launch supplying only the dataset and base model. It demonstrates the
> out-of-the-box plug-in performance of the system. Careful prompt/description adjustment
> informed by human priors is expected to improve on these numbers further.

## What this example is

- **Framework:** `verl_sft`: supervised fine-tuning via
  `torchrun -m verl.trainer.sft_trainer --config-name=sft_trainer_engine`.
- **Model:** `Qwen/Qwen3-4B` (text, thinking-enabled). FIXED: do not substitute.
- **Adapter:** LoRA, rank 32 / alpha 32, `target_modules=all-linear`.
- **Data schema:** `messages`-schema parquet (system + user document + assistant summary).
  There is **no reward function and no rollouts** during training; cross-entropy loss is
  computed on the **assistant (summary) turns only**.
- **Validation metric:** `val/loss` (validation cross-entropy). **Lower is better**
  (`score_direction = minimize`). LLMZero tracks this single metric to compare nodes.
- **Checkpointing: DISABLED** (`checkpoint_optimization.enabled: False`). This is a
  **single-shot** search: every node (root, evolve, debug) trains its **own** complete
  config **from the base model** with a fresh optimizer. No continuous training, no
  checkpoint chaining, no resume. Run length is governed by `trainer.total_epochs` in each
  node's own training script.
- **Offline test metric:** generation quality on the held-out test split is scored
  separately with `scrolls_score`: the SCROLLS `gov_report` ROUGE **geometric mean**
  `(rouge1 · rouge2 · rougeL)^(1/3)` on a 0–100 scale (see `reward_fn.py`). This is an
  **offline** evaluation, not used during the search.

### Long-context (the defining property)

GovReport documents are long: the rendered conversation (system + full report + gold
summary) has a **median ~9.6k tokens, p95 ~24.5k, max ~257k**. To fine-tune without
truncating the gold summary (which sits at the END, where `truncation=right` would clip
it), the following knobs are **FIXED** (memory-safety, not tunable):

- `data.max_length=32768`: covers ~98–99% of rows end-to-end.
- `engine.ulysses_sequence_parallel_size=8`: Ulysses **sequence parallelism** splits each
  long sequence across the 8 GPUs of one node so a long document's activations fit in 80GB.
- `data.use_dynamic_bsz=true`, `data.max_token_len_per_gpu=8192`: dynamic batching (verl
  asserts `max_token_len_per_gpu * sp_size >= longest sequence in a micro-batch`).
- `data.micro_batch_size_per_gpu=1`.

Do **not** lower `max_length` (clips the target), remove SP, or raise
`micro_batch_size_per_gpu`. Everything else (learning rate, scheduler, weight decay,
warmup, batch size, epochs) starts at verl's default and is **tunable** by the proposer.

## Files in this folder

| File | Purpose |
|------|---------|
| `description.txt` | The LLMZero task description. Carries the task framing, the FIXED long-context contract, the slurmdocker launch gotchas, and the **verl-default reference training command** the proposer tunes from. This is the only file that goes in the sacred `llmzero_input/` folder. |
| `llmzero_config.yaml` | MCTS parameters, per-agent LLM config (Bedrock Claude Opus 4.8), the `slurmdocker` compute backend, `validation_metric_pattern: "val/loss"`, and `checkpoint_optimization.enabled: False` (single-shot). |
| `reward_fn.py` | The `scrolls_score` reward (ROUGE geometric mean) used for the **offline** test-split evaluation. Strips `<think>…</think>`, sentence-splits (nltk), computes rouge1/rouge2/rougeL F-measures ×100, returns their geometric mean. |
| `install_commands.sh` | Extra pip deps for the reward (`pip install -q rouge_score nltk`). |
| `README.md` | This file. |

> **Note:** the data parquets (`train`/`val`/`test`) are **not** bundled; `train.parquet`
> alone is ~468 MB. Regenerate them with the steps below. Paths in `description.txt` and
> `llmzero_config.yaml` are generic placeholders (`/fsx/users/example/...`,
> `<slurm_partition>`, `${HF_TOKEN}`); point them at your own run folder, model, partition,
> container image, and HF token before launching.

## Regenerating the data

The data derives from the public **GovReport** long-document summarization dataset: the
SCROLLS `gov_report` task (`tau/scrolls`, config `gov_report`) or the original
`ccdv/govreport-summarization`. Build the three parquet splits as follows:

1. **Download** the GovReport split from HuggingFace (e.g. `datasets.load_dataset`).
2. **Train / val** (`messages`-schema, SFT):
   - For each example, build a `messages` list of three turns:
     - `system`: `"You are an expert at summarizing long government reports. Write a
       concise executive summary of the document below. Output only the summary text."`
     - `user`: the **full report text** (NO truncation, long context is the point).
     - `assistant`: the gold executive summary.
   - Columns: `messages` (list of `{role, content}` dicts, native Python objects, not JSON
     strings) and `data_source`.
   - `data_source` = `"govreport_sft"` for train, and **`"govreport_sft__dev"`** for val
     (the `__dev` suffix is what `validation_metric_pattern` / the framework keys on).
   - **Cap val at ~500 rows** (stratified/random subsample) so each validation pass stays
     fast under SP=8 on long docs.
   - Do **not** truncate inputs at prep time; the long-context handling is done at train
     time via `data.max_length=32768` + sequence parallelism. p95 input ≈ 24.5k tokens.
3. **Test** (GRPO/prompt-schema, for the **offline** `scrolls_score` eval):
   - Columns: `prompt` (list of `{role, content}` dicts: the same system + user document,
     **no** assistant turn), `data_source` = `"govreport_sft"`, and `reward_model`
     (`{"style": "rule", "ground_truth": <gold summary string or JSON list of references>}`).
   - **Cap test at ~500 rows.**
4. Write `data/{train,val,test}.parquet` into your run folder.

Reference row counts from this benchmark: **train 17,457, val 500, test 500.**

## How to launch

### Option A: skills pipeline (recommended)

The LLMZero RL-post-training skills assemble the whole run folder (data → reward → config →
description) and drive the MCTS loop. See the top-level README / `docs/benchmarks.md`:

- `/verl-grpo-prepare-dataset`: download + convert to parquet (adapt for the `messages`
  SFT schema above).
- `/verl-grpo-define-reward`: write/verify the `compute_score` reward (`reward_fn.py`).
- `/verl-grpo-generate-task`: render `description.txt` from the template + run config.
- `/general-download-model`: fetch `Qwen/Qwen3-4B` to your model prefix.
- `/general-launch-llmzero`, `/general-check-training`, `/general-gather-results`,
  `/general-plot-pareto`: launch, monitor, report.

### Option B: `llmzero` CLI directly

Place **only** `description.txt` into a sacred `llmzero_input/` folder, keep the data /
reward / config elsewhere in the run folder, then:

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

Notes:

- Install the reward deps first: `bash install_commands.sh` (`rouge_score`, `nltk`).
- The `slurmdocker` backend needs `container_image` / `container_mounts` set in
  `llmzero_config.yaml` (CLI flags alone fail its validation), already written in full in
  the config here; set the real `.sqsh` path and partition.
- Set your `HF_TOKEN` (the config has a `${HF_TOKEN}` placeholder in `container_env`) and,
  if using Bedrock for the search LLM, AWS credentials for Claude Opus 4.8.
- This benchmark used **1× `ml-p4d-24xlarge`** (8× A100-40GB), `trainer.nnodes=1`. SP=8
  requires 8 GPUs on one node.

Per-node validation scores land in
`<run_folder>/llmzero_output/observer_results_summary.{txt,json}`; each node's raw training
stdout/stderr is under `<run_folder>/llmzero_output/node_<i>/states/`.

## Results

See [`docs/benchmarks.md`](../../docs/benchmarks.md). Summary: the search completed all 16
iterations. **val/loss** improved 1.1338 (untrained base) → 1.0420 (no-search `node_0`,
verl-default HP) → **1.0072** (best search node). Held-out **test scrolls_score** (offline
vLLM generation on the test split, LoRA best-val checkpoint merged + fused) improved
22.86 → 31.74 → **34.66**: search > no-search > base on both metrics. The training strategy
the search discovered evolved lr / scheduler / epochs / LoRA rank from verl's defaults.
