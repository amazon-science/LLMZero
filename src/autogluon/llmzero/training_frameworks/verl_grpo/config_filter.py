"""VERL GRPO training-config filtering (keep only tuning-relevant keys)."""

_RELEVANT_CONFIG_KEYS = {"actor_rollout_ref", "trainer", "data", "algorithm"}

_IRRELEVANT_ACTOR_REF_KEYS = {"ref", "profiler", "hybrid_engine"}

_IRRELEVANT_ROLLOUT_KEYS = {
    "multi_turn",
    "agent",
    "trace",
    "engine_kwargs",
    "load_format",
    "layered_summon",
    "merge_lora_for_inference",
    "disable_log_stats",
    "calculate_log_probs",
    "update_weights_bucket_megabytes",
    "multi_stage_wake_up",
}

_IRRELEVANT_TRAINER_KEYS = {
    "npu_profile",
    "controller_nsight_options",
    "worker_nsight_options",
    "profile_steps",
    "device",
    "use_custom_rollouts",
    "custom_rollout",
    "use_legacy_worker_impl",
    "merge_lora_on_final_save",
    "ray_wait_register_center_timeout",
    "default_hdfs_dir",
    "del_local_ckpt_after_load",
    "max_actor_ckpt_to_keep",
    "max_critic_ckpt_to_keep",
    "critic_warmup",
    "esi_redundant_time",
    "log_val_generations",
}

_IRRELEVANT_DATA_KEYS = {
    "tokenizer",
    "use_shm",
    "return_raw_input_ids",
    "return_raw_chat",
    "return_full_prompt",
    "image_key",
    "video_key",
    "trust_remote_code",
    "custom_cls",
    "return_multi_modal_inputs",
    "sampler",
    "datagen",
    "filter_overlong_prompts_workers",
}


def filter_training_config(config: dict) -> dict:
    filtered = {k: v for k, v in config.items() if k in _RELEVANT_CONFIG_KEYS}

    actor_ref = filtered.get("actor_rollout_ref")
    if isinstance(actor_ref, dict):
        for key in _IRRELEVANT_ACTOR_REF_KEYS:
            actor_ref.pop(key, None)
        rollout = actor_ref.get("rollout")
        if isinstance(rollout, dict):
            for key in _IRRELEVANT_ROLLOUT_KEYS:
                rollout.pop(key, None)

    trainer = filtered.get("trainer")
    if isinstance(trainer, dict):
        for key in _IRRELEVANT_TRAINER_KEYS:
            trainer.pop(key, None)

    data = filtered.get("data")
    if isinstance(data, dict):
        for key in _IRRELEVANT_DATA_KEYS:
            data.pop(key, None)

    _strip_target_fields(filtered)
    return filtered


def _strip_target_fields(obj):
    """Recursively remove '_target_' keys from nested dicts (Hydra internals)."""
    if isinstance(obj, dict):
        obj.pop("_target_", None)
        for v in obj.values():
            _strip_target_fields(v)
    elif isinstance(obj, list):
        for item in obj:
            _strip_target_fields(item)
