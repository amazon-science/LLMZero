"""Compute infrastructure abstraction layer.

Provides a registry/factory for compute backends so that ``CodeExecutor``
and the rest of LLMZero never import backend-specific code directly.

Usage::

    from autogluon.llmzero.compute_infra import create_backend

    backend = create_backend({
        "type": "raycluster",
        "kubernetes_namespace": "my-ns",
        "ray_service_name": "my-cluster-head-svc",
        "ray_dashboard_port": 8265,
    })
"""

from .base import BaseComputeBackend, ComputeJobResult

_BACKENDS: dict[str, type[BaseComputeBackend]] = {}


def register_backend(name: str, cls: type[BaseComputeBackend]) -> None:
    """Register a compute backend class under *name*."""
    _BACKENDS[name] = cls


def get_backend(name: str) -> type[BaseComputeBackend]:
    """Return the backend class registered under *name*."""
    if name not in _BACKENDS:
        raise KeyError(f"Unknown compute backend '{name}'. " f"Available: {sorted(_BACKENDS.keys())}")
    return _BACKENDS[name]


def validate_backend_config(config_dict: dict) -> None:
    """Validate a compute backend config dict without instantiating.

    Delegates to the registered backend's ``validate_config`` classmethod.
    Raises ``ValueError`` with a helpful message if required fields are missing.
    """
    config = dict(config_dict)
    backend_type = config.pop("type", "raycluster")
    cls = get_backend(backend_type)
    cls.validate_config(config)


def create_backend(config_dict: dict) -> BaseComputeBackend:
    """Instantiate a compute backend from a configuration dict.

    The ``type`` key selects the backend (default ``"raycluster"``).
    Remaining keys are forwarded to the backend's ``from_config`` classmethod.
    """
    config = dict(config_dict)  # shallow copy to avoid mutating caller's dict
    backend_type = config.pop("type", "raycluster")
    cls = get_backend(backend_type)
    return cls.from_config(config)


from .conda import CondaBackend  # noqa: E402
from .docker import DockerBackend  # noqa: E402

# -- Auto-register built-in backends ----------------------------------------
from .raycluster import RayClusterBackend  # noqa: E402
from .slurm import SlurmBackend  # noqa: E402
from .slurmdocker import SlurmDockerBackend  # noqa: E402

register_backend("raycluster", RayClusterBackend)
register_backend("conda", CondaBackend)
register_backend("docker", DockerBackend)
register_backend("slurm", SlurmBackend)
register_backend("slurmdocker", SlurmDockerBackend)

__all__ = [
    "BaseComputeBackend",
    "ComputeJobResult",
    "register_backend",
    "get_backend",
    "validate_backend_config",
    "create_backend",
    "RayClusterBackend",
    "CondaBackend",
    "DockerBackend",
    "SlurmBackend",
    "SlurmDockerBackend",
]
