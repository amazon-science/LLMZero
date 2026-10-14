# Onboarding verl_sft into LLMZero

`verl_sft` is verl 0.9.0 **supervised fine-tuning** (SFT) — LoRA fine-tuning of a base LLM on a
`messages`-schema dataset, optimizing validation cross-entropy (`val/loss`). It is onboarded as a
general (non-RFT) framework and is the worked SFT counterpart to the RL/tabular examples.

How it differs from the other onboarded frameworks:

| Framework | Kind | Warm start | Target | Direction |
|-----------|------|-----------|--------|-----------|
| **verl_sft** | Supervised fine-tuning (LoRA) | **Full training state** (model + optimizer + LR/RNG) | `val/loss` | minimize |
| verl_grpo | RL post-training (GRPO) | LoRA-adapter-only (`save_lora_only=True`, `load_contents=["model"]`, fresh optimizer) | reward / val-core | maximize |

The defining feature of verl_sft is **optimizer-state continuous training**: each checkpoint saves model +
optimizer momentum/second-moment + LR schedule + RNG, so an MCTS-chained child node resumes the exact
optimization trajectory of its parent rather than restarting the optimizer.

## Files in this folder

| File | Purpose |
|------|---------|
| [README.md](README.md) | This file — overview, key decisions, gotchas, checklist |
| [cli-commands.md](cli-commands.md) | CLI invocations: standalone run, fresh MCTS run, continuous training (ckpt chain), monitoring, stopping |
| [description.txt](description.txt) | Reference `llmzero_input/description.txt` for verl_sft runs (reference command + fixed/tunable HPs + notes) |

## Framework plugin

Source: `src/autogluon/llmzero/training_frameworks/verl_sft/`

| Module | Role |
|--------|------|
| `framework.py` | Registration + wiring; `score_direction()="minimize"`, `validation_metric_pattern()="val/loss"`, `log_stream()="stdout"` |
| `parser.py` | Stdout parser for verl's `step:N - key:value` grammar; train/val metrics, flat `global_step_N/` checkpoint path, tqdm progress |
| `checkpoints.py` | Optimizer-state resume block (`resume_mode=resume_path` + `*_contents=["model","optimizer","extra"]`); `validate_checkpoint` requires BOTH model and optim shards |
| `hyperparameters.py` | Tunable HP categories + descriptions (optimizer/schedule, batch/sequence, LoRA, training length) |
| `metrics.py` | Metric descriptions; default text metrics `[train/loss, val/loss, train/lr]`, plot metrics `[train/loss, val/loss, train/grad_norm, train/lr]` |
| `config_filter.py` | Drops infra-only keys (`ray_init`, `profiler`, `hydra`, `_target_`) from the proposer's config view |
| `prompts/` | Proposer + early-stopper prompt fragments (minimize-val/loss objective, locked-HP constraints) |

## Key decisions

1. **Optimization target = `val/loss`, direction = minimize.** Validation cross-entropy on the val set's
   assistant turns. `score_direction()` returns `"minimize"` and `validation_metric_pattern()` is `r"val/loss"`.
2. **`messages` data schema, loss on assistant turns.** Parquet rows carry a `messages` list of
   `{role, content}` dicts (`data.messages_key=messages`); verl masks the loss to assistant turns only. No
   reward function, no rollouts.
3. **Flat `global_step_N/` checkpoint layout.** verl SFT saves checkpoints flat under
   `default_local_dir/global_step_N/` — NO `actor/` subdir (unlike GRPO). The parser's `_SAVE_PATH_RE` /
   `_PATH_STEP_RE` and `validate_checkpoint`'s glob both assume this flat layout.
4. **Optimizer-state save/load.** `checkpoint.save_contents` and `checkpoint.load_contents` are both
   `["model","optimizer","extra"]`, and resume uses `trainer.resume_mode=resume_path` +
   `trainer.resume_from_path=.../global_step_N`. `validate_checkpoint` rejects a directory that lacks
   `optim_world_size_*_rank_*.pt`, so the engine never silently resumes without optimizer momentum.
5. **Tier-1 onboarding (reuse base observers), stdout stream.** verl SFT prints metrics via the console
   LocalLogger to **stdout**, so `log_stream()="stdout"` and the standard `CodeExecutor` observers ingest
   them; only a framework-specific stdout parser was needed (no custom observer).
6. **LoRA merge for eval (two-step).** Evaluation needs a standalone HF model. `verl.model_merger` (fsdp
   backend) reconstructs the base weights into `merged_base/` and writes the LoRA adapter SEPARATELY under
   `merged_base/lora_adapter/` (it does NOT fuse them). A second PEFT step
   (`PeftModel.from_pretrained(...).merge_and_unload()`) fuses the adapter onto the base to produce the
   final standalone `merged_ckpt/` for vLLM eval.

## Gotchas

- **`data.max_length` / `data.truncation`.** Bespoke-Stratos distilled CoT is long (~8k–16k tokens). verl's
  default `max_length=1024` with `truncation=error` will either truncate away the answer or crash. Set
  `data.max_length=16384` and `data.truncation=right`.
- **Qwen3 thinking-template input-ids mismatch.** Qwen3's chat template can produce an input-ids/labels
  length mismatch during SFT tokenization; set `data.ignore_input_ids_mismatch=true`.
- **Dynamic-bsz token budget.** With the engine (dynamic batching) trainer, `data.max_token_len_per_gpu`
  must be `>= data.max_length` (both `16384` here) or long samples can't be packed.
- **No bare `python` on the login node.** The agent runs on a Slurm login node without the verl env. Do NOT
  run `python`/`torchrun` directly on the login node — run inside the container (via `llmzero submit-script`
  / the slurmdocker backend) or use an explicit env interpreter path.
- **WANDB does not propagate on slurmdocker.** verl spawns its own workers; env vars (including
  `WANDB_API_KEY`) don't reach them, so `trainer.logger=["console","wandb"]` crashes with
  "No API key configured". Use `trainer.logger=["console"]`.
- **GPQA-Diamond is gated.** The before/after eval reads GPQA-Diamond, which needs an HF token
  (`--container-env HF_TOKEN=${HF_TOKEN}`). Training does not; if the token is absent the eval script
  wires-but-skips GPQA.
- **`math_verify` is pip-installed at eval.** The container has verl/torch/vllm but not the math scorer;
  the eval step runs `pip install math_verify latex2sympy2_extended`. Not needed for MCTS training runs.
- **Shared Slurm job name.** All slurmdocker jobs share the name `llmzero-docker-training`. NEVER
  `scancel` by name — it kills every concurrent run. Stop by numeric job id (`llmzero stop-job <id>`) or
  `pkill -f <run-folder>` (see cli-commands.md).
- **`RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES` breaks the torchrun SFT path (root-caused on a real run).**
  The slurmdocker backend exports this var (`=1`) because the GRPO path runs verl workers as Ray actors and
  reads their assigned GPU from Ray's runtime context. The SFT trainer (`verl.trainer.sft_trainer`) runs under
  **torchrun/SPMD** — the worker is NOT a Ray actor — so with the var set verl calls
  `ray.get_runtime_context().get_accelerator_ids()["GPU"][0]` on an EMPTY list and dies with
  `IndexError: list index out of range` at Worker init. FIX: prefix the torchrun command with
  `env -u RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES` so verl falls back to torchrun's `LOCAL_RANK`. The
  reference command in `description.txt` already carries this prefix; do not remove it. (Do NOT edit the shared
  backend template — GRPO still needs the export.)
- **verl 0.9.0 `model_merger` crashes on a LoRA checkpoint (`task_type` bug).** `save_lora_adapter` does
  `peft_config["task_type"].value`, assuming a PEFT enum, but the SFT trainer writes `task_type` as the STRING
  `"CAUSAL_LM"` into `global_step_N/lora_train_meta.json` → `AttributeError: 'str' object has no attribute
  'value'`. FIX: drop the `task_type` key from `lora_train_meta.json` before running the merger (PEFT then
  defaults to `CAUSAL_LM` for a causal LM anyway). `run_all.sh` does this automatically in the MERGE step.

## Generic checklist (onboard any framework)

1. **Collect artifacts:** a completed real training log, a sample/base config, the exact launch command, the
   checkpoint scheme (paths + what's inside), the docker/conda env, and compute params.
2. **Decide the log stream** (stdout / stderr / both) and set `log_stream()`.
3. **Derive + verify parser regexes** against the real log (step-metric grammar, checkpoint-save marker,
   progress line). Use a stateful parser if eval lines lack a step number.
4. **Pick the validation metric + `score_direction`** and set `validation_metric_pattern()`.
5. **Ask the human for the HP tuning policy:** tunable categories, ranges/constraints, and LOCKED HPs;
   encode them in `hyperparameters.py` + the proposer prompt fragment.
6. **Write metric descriptions + default text/plot metrics** in `metrics.py` (feeds early-stopper/proposer plots).
7. **Implement checkpoint logic:** `validate_checkpoint` (what makes a checkpoint valid/resumable) and
   `checkpoint_resume_info` (the resume-block overrides). If the framework does not log checkpoint paths,
   disable `checkpoint_optimization` and supply paths via the description instead.
8. **Add a `config_filter`** to strip infra-only keys from the proposer's config view.
9. **Add prompt fragments** (`prompts/proposer.py`, `prompts/early_stopper.py`) encoding the objective and constraints.
10. **Register the framework** (module import in `__init__.py`) and set `training_framework: <name>` in the run config.
11. **Pre-flight a short live run:** confirm the job submits, metrics parse from the correct stream, the score
    is recorded, and MCTS advances (a resume-chained child node's script carries the resume block).

## Worked result

Standalone run: Qwen3-4B, LoRA r64, 1 epoch of Bespoke-Stratos-17k (261 steps), val on MATH-500 (val/loss,
minimize). Best checkpoint by val/loss = `global_step_200` (val/loss = 0.8385; curve fell 0.862 → 0.850 →
0.842 → 0.839 → **0.8385** → 0.839 → 0.839 across steps 40→261). Before/after benchmark accuracy (vllm,
thinking mode, temp 0.6 / top_p 0.95; math_verify for AIME/MATH, letter-match for GPQA):

| Benchmark | Base (before) | SFT (after) | Δ |
|-----------|---------------|-------------|-----|
| AIME2024 | 46.7% (14/30) | 50.0% (15/30) | **+3.3** |
| MATH-500 | 73.4% (367/500) | 73.2% (366/500) | −0.2 |
| GPQA-Diamond | 49.5% (98/198) | 45.5% (90/198) | −4.0 |

This is a **pipeline-validation** result, not a tuned one: a single LoRA epoch on an already-strong base
gives roughly neutral movement (AIME up, MATH flat, GPQA — out-of-domain for a math/code distillation set —
down). What it validates: the `verl_sft` framework drives training end-to-end, val/loss selects the best
checkpoint, optimizer-state resume is continuous (resuming `global_step_40` gave step-42/44 val/loss
0.8618/0.8619, no spike — optimizer momentum + LR/RNG restored), and the LoRA→HF merge + eval succeed.

**LLMZero smoke run** (`training_framework: verl_sft`, `-n 4`, slurmdocker): the engine attached the standard
observers (`validation` on `val/loss`, `step_metrics`, `checkpoint_paths`), and the PostTrainingCoder-generated
`node_0/run.sh` correctly reproduced the full reference command — including the required
`env -u RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES` prefix, LoRA r64, and
`checkpoint.save_contents=["model","optimizer","extra"]` — confirming the framework integrates with the MCTS
loop and the description-driven code generation.
