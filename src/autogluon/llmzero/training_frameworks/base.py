"""Base training framework interface.

A framework is a PACKAGE under `training_frameworks/<name>/` (see `verl_grpo/`
for the worked example, and `README.md` in this directory for the full guide).

To add a new training framework:

1. Create a package `training_frameworks/<name>/` with a `framework.py` defining
   a class that inherits from `BaseTrainingFramework` (split other concerns into
   `parser.py`, `hyperparameters.py`, `metrics.py`, `checkpoints.py`,
   `config_filter.py`, `observers/`, `prompts/` like `verl_grpo/`).
2. Override only what applies — every method has a safe default. Common ones:
     - hyperparameter_categories() / hyperparameter_descriptions(): Provide
       hyperparameter names grouped by category and (description, tradeoff) tuples.
     - metric_descriptions(): Map metric names logged by the framework to
       human-readable descriptions for LLM prompts.
     - default_text_metrics() / default_plot_metrics(): Which metrics to show
       as text summaries vs. visual plots in the proposer/early-stopper agents.
     - score_direction(): "maximize" (default) or "minimize"; drives best-score
       tracking and UCT normalization in the node managers.
     - validation_metric_pattern(): The framework's default regex for selecting
       the validation metric (the `validation_metric_pattern` config key overrides).
     - stdout_parser(): A BaseStdoutParser the generic observers use to read this
       framework's stdout grammar (Tier 1; required — every framework owns its grammar).
     - build_observers(ctx): Override to add/replace observers with your own
       observers/ package (Tier 2); the default wires the standard set.
     - prompt_fragments(agent_name) / prompt_template(agent_name): Customize the
       proposer/early-stopper prompts (fragments into, or a full override of, the
       neutral default template).
     - checkpoint_resume_info(node, ckpt_config): Framework-specific instructions
       for resuming training from a checkpoint.
     - validate_checkpoint(checkpoint): Given a checkpoint dict with a "path" key,
       return True if the directory contains valid model weight files.
     - filter_training_config(config) / training_config_filename(): Which resolved
       config file to show the proposer and how to trim it to tuning-relevant keys.
3. Register the framework in `__init__.py`:
     from .<name> import MyFramework
     register_framework("<name>", MyFramework)
4. Set `training_framework: <name>` in your config YAML.
"""

import os
from abc import ABC, abstractmethod
from typing import Optional


class BaseTrainingFramework(ABC):
    """Abstract base class for training frameworks.

    Most override points have safe empty defaults, so a framework only overrides
    what applies. The one exception is ``stdout_parser()``, which is abstract:
    every framework owns its stdout grammar, so there is no default parser and the
    base class cannot drive an execution on its own. A concrete framework
    (e.g. ``verl_grpo``) must be selected via ``training_framework`` in the config.
    """

    @classmethod
    def prompt_fragments(cls, agent_name: str) -> dict:
        """Text fragments injected into the neutral default templates.

        Keys (all optional, default ""): 'objective', 'role', 'metric_interpretation'.
        agent_name in {'proposer', 'early_stopper'}.
        """
        return {}

    @classmethod
    def prompt_template(cls, agent_name: str) -> Optional[str]:
        """Full template override for an agent prompt, or None for the neutral default."""
        return None

    @classmethod
    def hyperparameter_categories(cls) -> dict:
        return {}

    @classmethod
    def hyperparameter_descriptions(cls) -> dict:
        return {}

    @classmethod
    def metric_descriptions(cls) -> dict:
        return {}

    @classmethod
    def validation_metric_pattern(cls) -> Optional[str]:
        """Regex selecting the validation metric name from stdout metrics.

        Returns None by default; the config key ``validation_metric_pattern``
        overrides this when set. Subclasses return their framework's default.
        """
        return None

    @classmethod
    def score_direction(cls) -> str:
        """'maximize' (default) or 'minimize'.

        Drives best-score tracking and UCT normalization in the node managers.
        """
        return "maximize"

    @classmethod
    def log_stream(cls) -> str:
        """Which subprocess stream carries this framework's training logs.

        One of "stdout" (default), "stderr", or "both". The executor already
        separates the two streams; the observers ingest lines from the stream
        named here (see BaseExecutionObserver). Frameworks that log to stderr
        (e.g. loguru's default handler) return "stderr".
        """
        return "stdout"

    @classmethod
    def normalize_score(cls, score: float) -> float:
        """Map a raw validation score into 'higher-is-better' internal space.

        Identity for 'maximize'; negated for 'minimize'. The node managers run
        all comparisons and min-max UCT normalization on the normalized value,
        while still storing/reporting the raw score on the node.
        """
        return -score if cls.score_direction() == "minimize" else score

    @classmethod
    def better(cls, candidate: float, incumbent: float) -> bool:
        """True if ``candidate`` is strictly better than ``incumbent`` (ties are not better)."""
        return cls.normalize_score(candidate) > cls.normalize_score(incumbent)

    @classmethod
    def default_text_metrics(cls) -> list:
        return []

    @classmethod
    def default_plot_metrics(cls) -> list:
        return []

    @classmethod
    def checkpoint_resume_info(cls, node, ckpt_config) -> str:
        if not (ckpt_config and getattr(ckpt_config, "enabled", False)):
            return ""
        if not node or not node.resume_from_checkpoint:
            return "No checkpoint resumption. Train from scratch."
        return (
            f"RESUME FROM CHECKPOINT\n"
            f"Path: {node.resume_checkpoint_path}\n"
            f"Resume from step: {node.resume_checkpoint_step}\n"
        )

    @classmethod
    def format_hyperparameter_guide(cls) -> str:
        categories = cls.hyperparameter_categories()
        descriptions = cls.hyperparameter_descriptions()
        if not categories or not descriptions:
            return ""

        sections = []
        for category, params in categories.items():
            param_lines = []
            for param_name in params:
                if param_name in descriptions:
                    desc, tradeoffs = descriptions[param_name]
                    param_lines.append(f"  - **{param_name}**: {desc} *{tradeoffs}*")
            if param_lines:
                sections.append(f"**{category}:**\n" + "\n".join(param_lines))

        if not sections:
            return ""

        header = (
            "### Hyperparameter Reference Guide\n"
            "*Note: This guide covers hyperparameters that may need clarification. "
            "You can still adjust other hyperparameters not listed here.*\n\n"
        )
        return header + "\n\n".join(sections) + "\n"

    @classmethod
    def format_metric_descriptions(cls, metric_names: list) -> str:
        if not metric_names:
            return ""

        descriptions_dict = cls.metric_descriptions()
        if not descriptions_dict:
            return ""

        lines = []
        for metric_name in metric_names:
            desc = descriptions_dict.get(metric_name, "")
            if desc:
                lines.append(f"- **{metric_name}**: {desc}")

        if not lines:
            return ""

        return "### Metric Descriptions\n" + "\n".join(lines)

    @classmethod
    def training_config_filename(cls) -> str:
        """Filename of the resolved training config in the run directory."""
        return "training_config.yaml"

    @classmethod
    def filter_training_config(cls, config: dict) -> dict:
        """Filter a resolved training config dict to keep only tuning-relevant keys.

        Subclasses should override to remove infrastructure-only keys that are
        not useful for hyperparameter tuning suggestions.

        Returns a new dict with irrelevant infrastructure keys removed.
        """
        return config

    @classmethod
    @abstractmethod
    def stdout_parser(cls):
        """Return the BaseStdoutParser for this framework's stdout grammar.

        Required: every framework owns its stdout grammar. There is no default
        parser. Subclasses MUST return a BaseStdoutParser instance.
        """
        raise NotImplementedError("Subclasses must implement stdout_parser()")

    @classmethod
    def build_observers(cls, ctx) -> list:
        """Build the observer list for one execution.

        Default = the standard set wired with cls.stdout_parser(). Frameworks
        override to add/replace observers (including their own observers/ package)
        with zero edits to core execution code.
        """
        from ..execution.observers import (
            CheckpointEarlyStopObserver,
            CheckpointPathObserver,
            StepMetricsObserver,
            ValidationObserver,
        )

        parser = cls.stdout_parser()
        log_stream = cls.log_stream()
        observers = []

        if ctx.validation_score_mode in ("best_validation", "final_validation"):
            reduction = "final" if ctx.validation_score_mode == "final_validation" else "best"
            pattern = ctx.validation_metric_pattern or cls.validation_metric_pattern()
            if not pattern:
                raise ValueError(
                    "validation_metric_pattern is required in best_validation/final_validation mode "
                    "(set it in config or override validation_metric_pattern() on the framework)."
                )
            observers.append(
                ValidationObserver(
                    parser,
                    validation_metric_pattern=pattern,
                    score_direction=cls.score_direction(),
                    reduction=reduction,
                    log_stream=log_stream,
                )
            )
        else:
            raise ValueError(f"Unknown validation_score_mode: {ctx.validation_score_mode}")

        observers.append(StepMetricsObserver(parser, log_stream=log_stream))

        if ctx.early_stopper_enabled and not ctx.is_from_scratch and ctx.early_stopper_callback is not None:
            observers.append(
                CheckpointEarlyStopObserver(
                    checkpoint_interval_seconds=ctx.checkpoint_interval_seconds,
                    early_stopper_callback=ctx.early_stopper_callback,
                    parser=parser,
                    best_run_metrics=ctx.best_run_metrics,
                    best_run_validation_score=ctx.best_run_validation_score,
                    min_steps_before_early_stop=ctx.min_steps_before_early_stop,
                    log_stream=log_stream,
                )
            )

        if ctx.checkpoint_optimization_enabled:
            observers.append(CheckpointPathObserver(parser, log_stream=log_stream))

        return observers

    @classmethod
    def validate_checkpoint(cls, checkpoint: dict) -> bool:
        """Return True if the checkpoint looks usable.

        Framework-agnostic default: the path exists and is a non-empty directory.
        Frameworks with a specific on-disk layout (e.g. verl's ``actor/`` subdir)
        should override with a stricter check.
        """
        path = checkpoint.get("path", "")
        return bool(path) and os.path.isdir(path) and bool(os.listdir(path))
