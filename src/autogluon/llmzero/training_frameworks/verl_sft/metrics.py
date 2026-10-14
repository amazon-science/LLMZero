# src/autogluon/llmzero/training_frameworks/verl_sft/metrics.py
"""verl 0.9.0 SFT metric descriptions and default text/plot metric lists."""

METRIC_DESCRIPTIONS = {
    "val/loss": "Validation cross-entropy on the val set's assistant turns. PRIMARY objective — lower is better.",
    "train/loss": "Training cross-entropy on assistant turns for the current step.",
    "train/grad_norm": "Global gradient norm before clipping; spikes indicate instability.",
    "train/lr": "Current learning rate (reflects warmup + schedule).",
    "train/mfu": "Model FLOPs utilization (throughput/efficiency signal).",
    "train/global_tokens": "Tokens processed at this step (progress signal).",
}

DEFAULT_TEXT_METRICS = ["train/loss", "val/loss", "train/lr"]
DEFAULT_PLOT_METRICS = ["train/loss", "val/loss", "train/grad_norm", "train/lr"]
