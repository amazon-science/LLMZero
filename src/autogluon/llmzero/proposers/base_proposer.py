"""
Base proposer class for hyperparameter optimization suggestions.

This module provides the abstract base class for all proposer implementations.
Proposers analyze training history and suggest hyperparameter changes to try next.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class BaseProposer(ABC):
    """
    Abstract base class for proposers that suggest hyperparameter changes.

    A proposer analyzes the training history (including previous code,
    training dynamics, and validation metrics) to suggest what hyperparameters
    or training configurations should be changed next.

    This provides a common abstraction for different proposer implementations:
    - Agentic proposers (LLM-based)
    - Bayesian optimization proposers
    - Random search proposers
    - Evolutionary proposers
    - etc.
    """

    def __init__(self, config: Dict[str, Any], manager):
        """
        Initialize the proposer.

        Args:
            config: Configuration dictionary for the proposer
            manager: NodeManager instance for accessing state and history
        """
        self.config = config
        self.manager = manager

    @abstractmethod
    def propose(self) -> str:
        """
        Generate a proposal for hyperparameter changes to try next.

        This method should analyze the training history and return a string
        containing suggestions for what hyperparameters or configurations
        to change in the next iteration.

        Returns:
            str: A textual description of suggested changes to make.
                 This will be provided to the coder agent as guidance.
        """
        pass

    def get_previous_script(self, node=None) -> Optional[str]:
        """
        Get the previous entry point script from the node.

        Args:
            node: Node to get script from. If None, uses current node.

        Returns:
            str or None: The previous entry point script, or None if not available
        """
        if node is None:
            node = self.manager.current_node

        if node.parent is not None:
            return node.parent.entry_point_script
        return None

    def get_training_dynamics(self, node=None) -> Optional[Dict]:
        """
        Get training dynamics from the node's observer results.

        Args:
            node: Node to get dynamics from. If None, uses current node's parent.

        Returns:
            dict or None: Training dynamics data, or None if not available
        """
        if node is None:
            node = self.manager.current_node
            if node.parent is not None:
                node = node.parent

        if node.observer_results and "training_dynamics" in node.observer_results:
            return node.observer_results["training_dynamics"]
        return None

    def get_validation_metrics(self, node=None) -> Optional[Dict]:
        """
        Get validation metrics from the node's observer results.

        Args:
            node: Node to get metrics from. If None, uses current node's parent.

        Returns:
            dict or None: Validation metrics data, or None if not available
        """
        if node is None:
            node = self.manager.current_node
            if node.parent is not None:
                node = node.parent

        if node.observer_results and "validation_metrics" in node.observer_results:
            return node.observer_results["validation_metrics"]
        return None
