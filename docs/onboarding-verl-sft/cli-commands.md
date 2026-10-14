# verl_sft LLMZero CLI Commands

`<RUN>` is the LLMZero run folder (e.g. `/fsx/users/example/LLMZeroCCRuns/run_YYYYMMDD_HHMMSS_verl_sft_bespoke`).
`<MODEL>` is `/fsx/users/example/models/Qwen/Qwen3-4B`.

## Prerequisites

```bash
cd /fsx/users/example/LLMZero
```

- **Base model present**: Qwen3-4B at `/fsx/users/example/models/Qwen/Qwen3-4B` (download first, separate step).
- **Container image**: `docker/llmzero-verl-0.9.0.sqsh` (verl 0.9.0 + torch + vllm). Passed as `--docker-image`.
- **HF token for gated GPQA**: the before/after EVAL step reads GPQA-Diamond (gated). Put the token in
  `constants/cluster.json` and export it as `HF_TOKEN`. Training itself does NOT need it — only the standalone eval does.
  Never paste the literal token into scripts or docs; always reference `${HF_TOKEN}`.

## Standalone run (data prep → before-eval → SFT → resume test → LoRA merge → after-eval)

`<RUN>/run_all.sh` is the full end-to-end recipe (see [description.txt](description.txt) for the SFT block).
Submit it as a single Slurm+Docker job:

```bash
llmzero submit-script <RUN>/run_all.sh \
  --compute-backend slurmdocker \
  --slurm-partition <slurm_partition> \
  --docker-image /fsx/users/example/LLMZero/docker/llmzero-verl-0.9.0.sqsh \
  --mount-dir /fsx:/fsx \
  --mount-dir /opt/dlami/nvme:/opt/dlami/nvme \
  --slurm-nodes 1 \
  --slurm-gpus-per-node 8 \
  --slurm-time 1-00:00:00 \
  --container-env HF_TOKEN=${HF_TOKEN} \
  -v 2
```

`--container-env HF_TOKEN=${HF_TOKEN}` propagates the token into the container for the gated GPQA eval only.
The script `pip install`s `math_verify latex2sympy2_extended` at its eval step; the SFT step needs nothing extra.

## Fresh llmzero MCTS run

Runs the MCTS hyperparameter search over the SFT training step. The sacred input folder
`<RUN>/llmzero_input/` must contain ONLY `description.txt`.

```bash
llmzero \
  -i <RUN>/llmzero_input \
  -o <RUN>/llmzero_output \
  -c <RUN>/llmzero_config.yaml \
  --compute-backend slurmdocker \
  --slurm-partition <slurm_partition> \
  --docker-image /fsx/users/example/LLMZero/docker/llmzero-verl-0.9.0.sqsh \
  --mount-dir /fsx:/fsx \
  --mount-dir /opt/dlami/nvme:/opt/dlami/nvme \
  --slurm-nodes 1 \
  --slurm-gpus-per-node 8 \
  -n 16 \
  -v 2
```

Bound cost with a smaller `-n` (e.g. `-n 4`) for a smoke run; add `--stop-after-success` to stop after the
first successful node instead of continuing to optimize. Add `--slurm-time 1-00:00:00` to cap wall-clock.

## Continuous training (resume from best checkpoint of a previous run)

LLMZero chains checkpoints: a parent node's best (lowest-val/loss) checkpoint becomes the warm start for a
child node, restoring the FULL training state (model + optimizer momentum/second-moment + LR schedule + RNG).
This is the key contrast with verl_grpo (weights-only warm start).

### Step 1: Find the best checkpoint from a previous run

```bash
cat <RUN>/llmzero_output/tree_state.json | python3 -c "
import json, sys
tree = json.load(sys.stdin)
nodes = tree.get('nodes', {})
# val/loss is MINIMIZED -> best = lowest score
best_node, best_score = None, float('inf')
for nid, info in nodes.items():
    score = info.get('validation_score')
    if score is not None and score < best_score:
        best_score, best_node = score, nid
info = nodes[best_node]
print(f'Best node: {best_node}, val/loss: {best_score}')
print(f'Best step: {info.get(\"validation_step\", \"unknown\")}')
# verl SFT checkpoints are flat: <default_local_dir>/global_step_N/  (no actor/ subdir)
"
```

Then locate the flat checkpoint dir the parent wrote: `.../verl_ckpts/global_step_N/`. Confirm it holds both
`model_world_size_*_rank_*.pt` and `optim_world_size_*_rank_*.pt` (the framework's `validate_checkpoint`
rejects weights-only dirs).

### Step 2: Resume-block overrides

Add these Hydra overrides to the SFT training command (LLMZero appends them automatically for MCTS-chained
children; add them by hand for a manual continuation):

```
trainer.resume_mode=resume_path
trainer.resume_from_path=<CKPTS>/global_step_N
checkpoint.save_contents=["model","optimizer","extra"]
checkpoint.load_contents=["model","optimizer","extra"]
```

Rules:
- Keep `model.path`, `model.lora_rank`, `model.lora_alpha` IDENTICAL to the checkpoint or the load fails.
- verl inherits the global step from the `global_step_N` folder name; set `trainer.total_training_steps`
  GREATER than N or the run makes zero progress.
- If you changed `data.train_batch_size` / `data.num_workers` (or the worker count), delete the stale
  dataloader state first: `rm -f <CKPTS>/global_step_N/data_*.pt`.

### Step 3: Checkpoint chain

Same CLI as the fresh run; only the resume-block/description differs.

```
base Qwen3-4B  →  run_1 (best = node_3 @ global_step_120)
                     →  run_2 resumes global_step_120, total_training_steps=240 (best = node_7 @ global_step_200)
                          →  run_3 resumes global_step_200, total_training_steps=360 ...
```

Each stage restores optimizer state and continues the optimization trajectory. Consider lowering LR for
later stages (the schedule already decays, but a lower peak stabilizes fine refinement).

## Monitoring

```bash
# Live Slurm job stdout (job dir is named by the slurm job id):
tail -f ~/.llmzero/slurm_logs/job_*/slurm_<JOBID>.out

# Per-node LLMZero state (stdout capture + parsed metrics):
ls <RUN>/llmzero_output/node_*/states/
tail -50 <RUN>/llmzero_output/node_*/states/stdout
cat <RUN>/llmzero_output/node_*/states/step_metrics.json   # parsed train/loss, val/loss, train/lr per step

# Running slurm jobs (all slurmdocker runs share the job name 'llmzero-docker-training'):
squeue -u $USER --format="%.10i %.30j %.8T %.10M"
```

## Stopping

**HARD RULE: NEVER `scancel llmzero-docker-training`.** Every slurmdocker run — across ALL concurrent
LLMZero runs — shares that single job name, so cancelling by name kills every running job, not just yours.
Stop only by numeric job id or by run-folder path:

```bash
# Preferred: stop by numeric Slurm job id
llmzero stop-job <JOBID> \
  --compute-backend slurmdocker \
  --slurm-partition <slurm_partition>

# Or kill the local driver process scoped to this run folder (does NOT touch other runs):
pkill -f "run_YYYYMMDD_HHMMSS_verl_sft_bespoke"
```

## Key config fields (llmzero_config.yaml)

| Field | Value | Purpose |
|-------|-------|---------|
| training_framework | verl_sft | Selects parser, metrics, HP policy, checkpoint logic |
| validation_metric_pattern | "val/loss" | Regex observers use to extract the score |
| validation_score_mode | best_validation | Track the best (lowest) val/loss across steps |
| checkpoint_optimization.enabled | True | Auto-chain best node's checkpoint (with optimizer state) to children |
