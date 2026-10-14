"""
Agentic proposer implementation that uses an LLM-based agent for suggestions.

This proposer calls the ProposerAgent to analyze training history and generate
hyperparameter optimization suggestions.
"""

import logging
from typing import Any, Dict

from ..agents.proposer_agent import ProposerAgent
from .base_proposer import BaseProposer

logger = logging.getLogger(__name__)


class AgenticProposer(BaseProposer):
    """
    Agentic proposer that uses an LLM-based agent to suggest hyperparameter changes.

    This proposer leverages the ProposerAgent to analyze:
    - Previous post-training script
    - Training dynamics signals
    - Validation metrics signals

    And generate data-driven suggestions for hyperparameter modifications.
    """

    def __init__(self, config: Dict[str, Any], manager, agent_llm_config=None, agent_prompt_template=None):
        """
        Initialize the agentic proposer.

        Args:
            config: Configuration dictionary for the proposer
            manager: NodeManager instance for accessing state and history
            agent_llm_config: LLM configuration for the proposer agent
            agent_prompt_template: Optional custom prompt template for the agent
        """
        super().__init__(config=config, manager=manager)

        # Store agent configuration
        self.agent_llm_config = agent_llm_config
        self.agent_prompt_template = agent_prompt_template

        # Initialize the proposer agent
        self.proposer_agent = ProposerAgent(
            config=config,
            manager=manager,
            llm_config=agent_llm_config,
            prompt_template=agent_prompt_template,
        )

    def propose(self, cold_start: bool = False) -> str:
        """
        Generate hyperparameter change proposals using the proposer agent.

        The agent analyzes the parent node's training history (if any) and the
        sibling configs already tried, and generates suggestions for promising
        changes to try next.

        Args:
            cold_start: When True, there is no parent training history (e.g. the
                bootstrap siblings of the async virtual root). The agent is still
                invoked so each sibling produces a DISTINCT initial configuration
                that diverges from the baseline and earlier siblings — rather than
                returning a canned string that would make every sibling identical.

        Returns:
            str: Textual suggestions for hyperparameter changes
        """
        logger.info(
            f"AgenticProposer: generating hyperparameter suggestions via ProposerAgent (cold_start={cold_start})"
        )

        parent_node = self.manager.current_node.parent

        # Without a cold-start mandate, fall back to canned guidance when there is
        # no parent history (preserves the original sequential-MCTS behavior).
        if not cold_start:
            if parent_node is None:
                logger.warning("AgenticProposer: No parent node available, cannot generate informed proposals")
                return "No previous training history available. Try initial hyperparameters."
            if not parent_node.observer_results:
                logger.warning("AgenticProposer: Parent node has no observer results")
                return "No training metrics available from previous run. Try incremental hyperparameter adjustments."

        # Call the proposer agent to generate suggestions
        try:
            suggestions = self.proposer_agent(cold_start=cold_start)
            logger.info(f"AgenticProposer: generated {len(suggestions)} characters of suggestions")
            return suggestions
        except Exception as e:
            logger.error(f"AgenticProposer: Error generating suggestions: {e}", exc_info=True)
            return "Error generating suggestions."
