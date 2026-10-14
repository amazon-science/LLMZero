"""VERL GRPO proposer prompt fragments (RL-flavored)."""

OBJECTIVE = "maximize the validation score"

METRIC_INTERPRETATION = """### Metric Interpretation
**PRIMARY:** Rewards (critic/rewards/mean) and validation scores measure overall performance. Judge success by these metrics.
**DIAGNOSTIC:** actor/entropy, actor/kl_loss, actor/grad_norm, etc. Only relevant when primary metrics degrade. Rising KL with stable rewards is acceptable."""


def fragments() -> dict:
    return {"objective": OBJECTIVE, "metric_interpretation": METRIC_INTERPRETATION}
