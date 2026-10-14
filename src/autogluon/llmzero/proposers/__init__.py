"""
Proposer modules for hyperparameter optimization suggestions.

This package contains base proposer abstractions and concrete implementations
for suggesting hyperparameter changes during post-training optimization.
"""

from .agentic_proposer import AgenticProposer
from .base_proposer import BaseProposer

__all__ = ["BaseProposer", "AgenticProposer"]
