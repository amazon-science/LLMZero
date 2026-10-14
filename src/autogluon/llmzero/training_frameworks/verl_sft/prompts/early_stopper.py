# src/autogluon/llmzero/training_frameworks/verl_sft/prompts/early_stopper.py
"""verl SFT early-stopper prompt fragments."""

ROLE = "an expert at analyzing supervised fine-tuning using training and validation loss curves"

OBJECTIVE = (
    "Your ONLY goal is to minimize the validation loss (val/loss). CONTINUE if the current run's "
    "val/loss trajectory has potential to drop below the best (lowest) val/loss seen so far. STOP "
    "only when val/loss has clearly plateaued or is rising (overfitting) with no realistic chance "
    "of beating the best."
)

METRIC_INTERPRETATION = (
    "**Only the validation score (val/loss, lower is better) determines STOP/CONTINUE.** "
    "train/loss, train/grad_norm and train/lr provide context (overfitting, instability) but NEVER "
    "justify stopping on their own."
)


def fragments() -> dict:
    return {"role": ROLE, "objective": OBJECTIVE, "metric_interpretation": METRIC_INTERPRETATION}
