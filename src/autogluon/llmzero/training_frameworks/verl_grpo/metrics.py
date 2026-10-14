"""VERL GRPO metric descriptions and default metric lists."""

METRIC_DESCRIPTIONS = {
    "actor/entropy": (
        "Policy entropy - measures exploration in the policy. "
        "**Healthy pattern:** Entropy dropping WITH rising rewards (specialization while improving). "
        "**Problem pattern:** Entropy AND rewards both dropping (losing generality without gains)."
    ),
    "actor/kl_loss": (
        "KL divergence loss - measures how much the updated policy diverges from the old policy. "
        "Used to ensure training stability by preventing too large policy updates. "
        "Should remain relatively small and stable throughout training."
    ),
    "actor/ppo_kl": (
        "PPO KL divergence - measures the actual KL divergence between old and new policy distributions. "
        "Should remain stable and within the target KL threshold throughout training. "
        "**CRITICAL WARNING**: If ppo_kl has a spike AND training rewards drop significantly afterward, "
        "it means training has COLLAPSED."
    ),
    "actor/pg_loss": (
        "Policy gradient loss - the primary training signal for the actor. "
        "Measures how well the policy is being updated based on reward signals. "
        "Should generally decrease as the policy improves, though some fluctuation is normal."
    ),
    "actor/pg_clipfrac": (
        "**PPO policy clipping fraction** (upper bound) - ratio of policy updates that hit the clip_ratio_high bound. "
        "This measures how often the PPO algorithm is constraining policy improvements. "
        "High values (>0.3) indicate many large policy updates are being prevented by clip_ratio_high. "
        "May suggest the learning rate is too high or clip_ratio_high is too conservative. "
        "NOTE: This is unrelated to response_length/clip_ratio (which tracks text truncation)."
    ),
    "actor/pg_clipfrac_lower": (
        "**PPO policy clipping fraction** (lower bound) - ratio of policy updates that hit the clip_ratio_low bound. "
        "Combined with pg_clipfrac, this shows how often PPO clipping is actively constraining updates. "
        "Very high values on both bounds suggest aggressive policy changes that need regulation. "
        "NOTE: This is unrelated to response_length/clip_ratio (which tracks text truncation)."
    ),
    "actor/grad_norm": (
        "Gradient norm for actor updates - measures the magnitude of gradients during backpropagation. "
        "Very high values may indicate instability or need for gradient clipping. "
        "Very low values may indicate vanishing gradients or convergence."
    ),
    "critic/rewards/mean": (
        "Mean predicted rewards from the critic network - estimates the expected return. "
        "Should generally increase as training progresses and the policy improves. "
        "This is a key indicator of overall training progress and performance."
    ),
    "response_length/mean": (
        "Average total response length in tokens (thinking tokens + answer tokens). "
        "The total is bounded by max_response_length. "
        "Interpret length changes alongside rewards and validation score — length alone does not indicate quality."
    ),
    "response_length/clip_ratio": (
        "**Response length truncation ratio** - fraction of generated responses that hit the max_response_length limit and were truncated. "
        "NOTE: This is NOT related to PPO clip_ratio_high/low parameters (those control policy update clipping). "
        "High values mean responses are being cut off — increase max_response_length to allow longer output. "
        "Interpret changes alongside rewards and validation score to judge impact."
    ),
    "timing_s/step": (
        "Time per training step in seconds. "
        "Tracks training efficiency and can reveal performance issues or bottlenecks. "
        "Sudden increases may indicate memory issues, inefficient batch sizes, or system problems."
    ),
    "perf/max_memory_reserved_gb": (
        "Maximum GPU memory reserved in gigabytes. "
        "Monitors memory usage to detect potential out-of-memory issues. "
        "Helps optimize batch sizes and model configurations for available hardware."
    ),
}

DEFAULT_TEXT_METRICS = ["response_length/mean", "critic/rewards/mean"]
DEFAULT_PLOT_METRICS = ["actor/kl_loss", "actor/grad_norm", "critic/rewards/mean", "response_length/clip_ratio"]
