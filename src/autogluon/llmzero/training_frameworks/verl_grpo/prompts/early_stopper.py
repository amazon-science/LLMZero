"""VERL GRPO early-stopper prompt fragments (RL-flavored)."""

ROLE = "an expert at analyzing RL for LLM post training using visual plots"

OBJECTIVE = "Your ONLY goal is to maximize the validation score. CONTINUE if the current run's validation score trajectory has potential to exceed the best validation score seen so far. STOP only when there is no realistic chance of beating it."

METRIC_INTERPRETATION = """**Only the validation score determines STOP/CONTINUE.** All other metrics (rewards, entropy, KL, gradient norms, clip_ratio, etc.) provide context but NEVER justify stopping on their own."""


def fragments() -> dict:
    return {"role": ROLE, "objective": OBJECTIVE, "metric_interpretation": METRIC_INTERPRETATION}
