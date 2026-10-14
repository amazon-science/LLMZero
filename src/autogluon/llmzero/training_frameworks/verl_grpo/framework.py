"""VERL GRPO/PPO training framework — thin class wiring the package pieces."""

from ..base import BaseTrainingFramework
from . import checkpoints, config_filter, hyperparameters, metrics


class VerlGrpoFramework(BaseTrainingFramework):
    """VERL GRPO/PPO reinforcement learning training framework."""

    @classmethod
    def hyperparameter_categories(cls) -> dict:
        return hyperparameters.HYPERPARAMETER_CATEGORIES

    @classmethod
    def hyperparameter_descriptions(cls) -> dict:
        return hyperparameters.HYPERPARAMETER_DESCRIPTIONS

    @classmethod
    def metric_descriptions(cls) -> dict:
        return metrics.METRIC_DESCRIPTIONS

    @classmethod
    def validation_metric_pattern(cls) -> str:
        return "val-core/.*__dev"

    @classmethod
    def log_stream(cls) -> str:
        # verl 0.9.0 splits its logs: step/val metrics go to STDOUT, but the
        # checkpoint-save line ("Saving checkpoint to <path>") and the tqdm
        # "Training Progress" bar go to STDERR. Ingest both so the checkpoint-path
        # observer sees the save marker (else checkpoints=[] and every MCTS node is
        # forced from-scratch, disabling warm-start deepening + the early stopper).
        return "both"

    @classmethod
    def default_text_metrics(cls) -> list:
        return list(metrics.DEFAULT_TEXT_METRICS)

    @classmethod
    def default_plot_metrics(cls) -> list:
        return list(metrics.DEFAULT_PLOT_METRICS)

    @classmethod
    def checkpoint_resume_info(cls, node, ckpt_config) -> str:
        return checkpoints.checkpoint_resume_info(node, ckpt_config)

    @classmethod
    def validate_checkpoint(cls, checkpoint: dict) -> bool:
        return checkpoints.validate_checkpoint(checkpoint)

    @classmethod
    def stdout_parser(cls):
        from .parser import VerlStdoutParser

        return VerlStdoutParser()

    @classmethod
    def filter_training_config(cls, config: dict) -> dict:
        return config_filter.filter_training_config(config)

    @classmethod
    def prompt_fragments(cls, agent_name: str) -> dict:
        from .prompts import fragments_for

        return fragments_for(agent_name)
