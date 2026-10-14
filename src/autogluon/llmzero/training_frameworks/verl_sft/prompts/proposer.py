# src/autogluon/llmzero/training_frameworks/verl_sft/prompts/proposer.py
"""verl SFT proposer prompt fragments (supervised fine-tuning flavored)."""

OBJECTIVE = "minimize the validation loss (val/loss) of the supervised-fine-tuned model"

METRIC_INTERPRETATION = """### Metric Interpretation
**PRIMARY:** val/loss (validation cross-entropy on the val set's assistant turns) is the optimization target — LOWER IS BETTER. Judge success by val/loss.
**DIAGNOSTIC:** train/loss, train/grad_norm, train/lr provide context (over/under-fitting, instability, schedule) but do NOT by themselves define success. A widening gap between a falling train/loss and a rising val/loss means overfitting — prefer fewer epochs or more regularization.

### HARD CONSTRAINTS — DO NOT VIOLATE
**LOCKED hyperparameters — NEVER propose changes to any of these** (they break checkpoint/optimizer-state loading or change the evaluation target):
- Model identity: `model.path` (the base model is fixed).
- LoRA shape across a resume chain: `model.lora_rank`, `model.lora_alpha` MUST match the checkpoint being resumed, or optimizer/model state fails to load.
- Data schema: `data.messages_key` and the parquet `messages` format are fixed.
- Checkpoint contents: `checkpoint.save_contents` and `checkpoint.load_contents` MUST remain `["model","optimizer","extra"]` so continuous training keeps optimizer momentum + LR/RNG state. Do NOT reduce them to weights-only.
- Logging: keep `trainer.logger=["console"]` on slurmdocker (WANDB env vars do not propagate to workers).

When resuming, set `trainer.total_training_steps` greater than the resumed checkpoint's step or the run makes no progress."""


def fragments() -> dict:
    return {"objective": OBJECTIVE, "metric_interpretation": METRIC_INTERPRETATION}
