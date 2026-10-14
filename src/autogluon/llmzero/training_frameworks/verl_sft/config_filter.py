# src/autogluon/llmzero/training_frameworks/verl_sft/config_filter.py
"""Trim a resolved verl SFT config to tuning-relevant keys for the proposer."""

# Infrastructure-only top-level keys that carry no tuning signal.
_DROP_KEYS = {
    "ray_init",
    "profiler",
    "hydra",
    "_target_",
}


def filter_training_config(config: dict) -> dict:
    if not isinstance(config, dict):
        return config
    return {k: v for k, v in config.items() if k not in _DROP_KEYS}
