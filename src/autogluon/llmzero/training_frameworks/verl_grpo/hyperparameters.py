"""VERL GRPO hyperparameter categories and tuning descriptions."""

HYPERPARAMETER_CATEGORIES = {
    "Learning Rates & Optimization": [
        "actor_rollout_ref.actor.optim.weight_decay",
        "actor_rollout_ref.actor.grad_clip",
    ],
    "Batch Sizes": [
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",
    ],
    "PPO/GRPO Clipping": [
        "actor_rollout_ref.actor.clip_ratio_low",
        "actor_rollout_ref.actor.clip_ratio_high",
    ],
    "KL Divergence & Regularization": [
        "actor_rollout_ref.actor.kl_loss_coef",
        "actor_rollout_ref.actor.entropy_coeff",
        "algorithm.norm_adv_by_std_in_grpo",
    ],
    "Rollout & Sampling": [
        "actor_rollout_ref.rollout.n",
        "actor_rollout_ref.rollout.temperature",
    ],
    "Sequence & Memory Management": [
        "data.max_response_length",
        "actor_rollout_ref.rollout.max_num_batched_tokens",
        "actor_rollout_ref.rollout.max_num_seqs",
        "actor_rollout_ref.rollout.tensor_model_parallel_size",
    ],
    "Training Configuration": [
        "trainer.total_epochs",
    ],
}

HYPERPARAMETER_DESCRIPTIONS = {
    "actor_rollout_ref.actor.optim.weight_decay": (
        "L2 regularization on actor model weights.",
        "Higher values prevent overfitting but may cause underfitting. Increase if large train/val gap; decrease if underfitting.",
    ),
    "actor_rollout_ref.actor.grad_clip": (
        "Maximum gradient norm for actor model (clips if exceeded). Available with the default FSDP/dp actor strategy (model_engine=dp); not exposed under the Megatron strategy.",
        "Lower values prevent exploding gradients but may slow learning. Decrease if seeing NaN losses, grad norm spikes, or training collapse; increase if gradients consistently small.",
    ),
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": (
        "Micro-batch size per GPU for gradient accumulation in PPO/GRPO updates.",
        "**CRITICAL CONSTRAINT:** ppo_mini_batch_size (after normalization) must be divisible by this value, or AssertionError occurs. Larger values reduce accumulation steps but increase memory.",
    ),
    "actor_rollout_ref.actor.clip_ratio_low": (
        "**PPO policy clipping** lower bound (prevents excessive policy degradation). NOTE: This is unrelated to response_length/clip_ratio metric (which tracks text truncation).",
        "Suggested to keep at default. Increasing it may suppress low-probability tokens to 0, causing sampling space collapse.",
    ),
    "actor_rollout_ref.actor.clip_ratio_high": (
        "**PPO policy clipping** upper bound (prevents excessive policy improvement). NOTE: This is unrelated to response_length/clip_ratio metric (which tracks text truncation).",
        "Higher values leave more room for low-probability tokens to increase, enhancing entropy and generating more diverse samples. Lower values are more conservative.",
    ),
    "actor_rollout_ref.actor.kl_loss_coef": (
        "Weight of KL divergence loss to constrain policy changes from reference (used in GRPO with use_kl_loss=True).",
        "Higher values enforce stability (slower exploration). Lower values allow larger changes (faster but riskier). **Only increase if KL spikes coincide with reward/validation collapse** - isolated KL increases often indicate necessary policy changes on complex data. If training collapses with KL spikes, reduce grad_clip first before adjusting this.",
    ),
    "actor_rollout_ref.actor.entropy_coeff": (
        "Weight of entropy bonus encouraging exploration (adds entropy to reward function).",
        "**CRITICAL: Must remain 0 for GRPO runs.** GRPO already handles exploration through advantage normalization and doesn't require entropy bonus. Do NOT modify this parameter under any circumstances.",
    ),
    "algorithm.norm_adv_by_std_in_grpo": (
        "Whether to normalize advantages by their standard deviation in GRPO.",
        "Enabling (True) makes advantage scaling consistent across batches (more stable). Disabling (False) preserves raw advantage magnitudes. Generally keep True for stability.",
    ),
    "actor_rollout_ref.rollout.n": (
        "Number of response samples generated per prompt during actor rollout.",
        "Higher n improves advantage estimates (comparing responses) but increases compute linearly. Increase for better learning signal; decrease to save compute.",
    ),
    "actor_rollout_ref.rollout.temperature": (
        "Sampling temperature during training generation (higher = more random).",
        "Higher values increase exploration and diversity during training. Lower values produce more deterministic outputs.",
    ),
    "data.max_response_length": (
        "Maximum total response length in tokens (thinking + answer).",
        "If response_length/clip_ratio is high, increase this — but increases memory/compute and step time linearly. "
        "**CRITICAL CONSTRAINT:** max_prompt_length + max_response_length MUST be <= max_num_batched_tokens, "
        "otherwise vLLM crashes. Always increase max_num_batched_tokens when increasing this.",
    ),
    "actor_rollout_ref.rollout.max_num_batched_tokens": (
        "Maximum tokens in a single vLLM prefill batch.",
        "**CRITICAL CONSTRAINT:** Must be >= max_prompt_length + max_response_length. "
        "Increase this whenever you increase max_response_length.",
    ),
    "actor_rollout_ref.rollout.max_num_seqs": (
        "Maximum concurrent sequences in vLLM generation batch.",
        "Higher values increase throughput by batching more sequences together. Decrease if OOM; increase for better GPU utilization. Interacts with memory and sequence lengths.",
    ),
    "actor_rollout_ref.rollout.tensor_model_parallel_size": (
        "Number of GPUs to shard model layers across for parallel computation.",
        "Use minimum value that fits model in memory. Must evenly divide num_attention_heads and stay within single node for NVLink speed. Lower TP means higher data parallelism and better throughput. Increase only when OOM occurs.",
    ),
    "trainer.total_epochs": (
        "Number of complete passes through the training dataset.",
        "Training runs all epochs, then we select the checkpoint with best validation score. **Increase epochs if:** validation was still improving at final epoch OR showed potential for further gains. Prioritize performance over speed - use as many epochs as needed for convergence.",
    ),
}
