# src/autogluon/llmzero/training_frameworks/verl_sft/hyperparameters.py
"""Tunable hyperparameters for verl 0.9.0 SFT (LoRA supervised fine-tuning)."""

HYPERPARAMETER_CATEGORIES = {
    "Optimizer & schedule": [
        "optim.lr",
        "optim.lr_scheduler_type",
        "optim.weight_decay",
        "optim.lr_warmup_steps_ratio",
        "optim.clip_grad",
    ],
    "Batch & sequence": [
        "data.train_batch_size",
        "data.micro_batch_size_per_gpu",
        "data.max_length",
    ],
    "LoRA": [
        "model.lora_rank",
        "model.lora_alpha",
        "model.target_modules",
    ],
    "Training length": [
        "trainer.total_epochs",
        "trainer.total_training_steps",
    ],
}

HYPERPARAMETER_DESCRIPTIONS = {
    "optim.lr": (
        "Peak learning rate for AdamW.",
        "Too high spikes/diverges val/loss; too low wastes the budget. LoRA typically 5e-5-2e-4.",
    ),
    "optim.lr_scheduler_type": (
        "LR schedule: 'constant' or 'cosine'.",
        "Cosine (with warmup) usually gives a smoother, lower final val/loss than constant.",
    ),
    "optim.weight_decay": (
        "AdamW weight decay.",
        "Mild regularization; too high underfits the SFT target.",
    ),
    "optim.lr_warmup_steps_ratio": (
        "Fraction of total steps spent warming LR up from 0.",
        "A short warmup (0.0–0.05) stabilizes early steps; too long shortens effective training.",
    ),
    "optim.clip_grad": (
        "Gradient-norm clip threshold.",
        "Lower clamps spikes but can slow learning; 1.0 is a safe default.",
    ),
    "data.train_batch_size": (
        "Global training batch size (samples per optimizer step).",
        "Larger = smoother gradients but fewer steps per epoch; must fit memory with max_length.",
    ),
    "data.micro_batch_size_per_gpu": (
        "Per-GPU micro-batch for gradient accumulation.",
        "Raise to use the GPU better; too high OOMs at long max_length.",
    ),
    "data.max_length": (
        "Max tokens per training example (prompt+response).",
        "Must cover the distilled CoT (Bespoke-Stratos is long, ~8k–16k); higher costs memory/time.",
    ),
    "model.lora_rank": (
        "LoRA adapter rank.",
        "Higher rank = more capacity but more params/memory. LOCKED across a resume chain.",
    ),
    "model.lora_alpha": (
        "LoRA scaling (effective LR of the adapter ∝ alpha/rank).",
        "Commonly set equal to rank. LOCKED across a resume chain.",
    ),
    "model.target_modules": (
        "Which linear modules get LoRA adapters ('all-linear' or a name list).",
        "Broader coverage adds capacity + cost.",
    ),
    "trainer.total_epochs": (
        "Number of passes over the training data.",
        "More epochs can overfit the SFT distillation set (watch val/loss turning up).",
    ),
    "trainer.total_training_steps": (
        "Hard cap on optimizer steps (overrides epochs when set).",
        "Use to bound run length; on resume it must exceed the checkpoint's step to progress.",
    ),
}
