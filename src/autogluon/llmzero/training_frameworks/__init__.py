"""Training framework abstraction layer.

Provides a registry/factory for training frameworks so that prompts and agents
can load framework-specific hyperparameter descriptions, metric descriptions,
and checkpoint logic without hardcoding any single framework.
"""

from .base import BaseTrainingFramework

_FRAMEWORKS: dict[str, type[BaseTrainingFramework]] = {}


def register_framework(name: str, cls: type[BaseTrainingFramework]) -> None:
    _FRAMEWORKS[name] = cls


def get_framework(name: str) -> type[BaseTrainingFramework]:
    if name not in _FRAMEWORKS:
        raise KeyError(f"Unknown training framework '{name}'. Available: {sorted(_FRAMEWORKS.keys())}")
    return _FRAMEWORKS[name]


def get_framework_from_config(config) -> type[BaseTrainingFramework]:
    """Get framework class from config, falling back to BaseTrainingFramework if unset."""
    framework_name = getattr(config, "training_framework", None)
    if not framework_name:
        return BaseTrainingFramework
    return get_framework(framework_name)


def validate_framework_config(config) -> None:
    """Validate the training framework selection without instantiating downstream.

    Mirrors compute_infra.validate_backend_config: resolves the framework name,
    raises KeyError if unknown, and runs the framework's optional validate() hook.
    Unset training_framework is a no-op (the base framework is used downstream).
    """
    framework_name = getattr(config, "training_framework", None)
    if not framework_name:
        return
    fw = get_framework(framework_name)  # raises KeyError if unknown
    validate = getattr(fw, "validate", None)
    if callable(validate):
        validate(config)


from .verl_grpo import VerlGrpoFramework  # noqa: E402

register_framework("verl_grpo", VerlGrpoFramework)

from .verl_sft import VerlSftFramework  # noqa: E402

register_framework("verl_sft", VerlSftFramework)

__all__ = [
    "BaseTrainingFramework",
    "register_framework",
    "get_framework",
    "get_framework_from_config",
    "validate_framework_config",
    "VerlGrpoFramework",
    "VerlSftFramework",
]
