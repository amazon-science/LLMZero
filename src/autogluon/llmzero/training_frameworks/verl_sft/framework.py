# src/autogluon/llmzero/training_frameworks/verl_sft/framework.py
"""verl 0.9.0 SFT training framework — thin class wiring the package pieces."""

from ..base import BaseTrainingFramework


class VerlSftFramework(BaseTrainingFramework):
    """verl 0.9.0 supervised fine-tuning framework (optimizer-state continuous training)."""

    @classmethod
    def score_direction(cls) -> str:
        return "minimize"

    @classmethod
    def validation_metric_pattern(cls) -> str:
        return r"val/loss"

    @classmethod
    def log_stream(cls) -> str:
        return "stdout"

    @classmethod
    def stdout_parser(cls):
        from .parser import VerlSftStdoutParser

        return VerlSftStdoutParser()

    @classmethod
    def checkpoint_resume_info(cls, node, ckpt_config) -> str:
        from . import checkpoints

        return checkpoints.checkpoint_resume_info(node, ckpt_config)

    @classmethod
    def validate_checkpoint(cls, checkpoint: dict) -> bool:
        from . import checkpoints

        return checkpoints.validate_checkpoint(checkpoint)

    @classmethod
    def hyperparameter_categories(cls) -> dict:
        from . import hyperparameters

        return hyperparameters.HYPERPARAMETER_CATEGORIES

    @classmethod
    def hyperparameter_descriptions(cls) -> dict:
        from . import hyperparameters

        return hyperparameters.HYPERPARAMETER_DESCRIPTIONS

    @classmethod
    def metric_descriptions(cls) -> dict:
        from . import metrics

        return metrics.METRIC_DESCRIPTIONS

    @classmethod
    def default_text_metrics(cls) -> list:
        from . import metrics

        return list(metrics.DEFAULT_TEXT_METRICS)

    @classmethod
    def default_plot_metrics(cls) -> list:
        from . import metrics

        return list(metrics.DEFAULT_PLOT_METRICS)

    @classmethod
    def filter_training_config(cls, config: dict) -> dict:
        from . import config_filter

        return config_filter.filter_training_config(config)

    @classmethod
    def prompt_fragments(cls, agent_name: str) -> dict:
        from .prompts import fragments_for

        return fragments_for(agent_name)
