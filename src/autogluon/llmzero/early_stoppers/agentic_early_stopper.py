"""
Agentic early stopper implementation that uses an LLM-based agent for decisions.

This early stopper calls the EarlyStopperAgent to analyze current training progress
and compare with the best run to decide whether to continue or stop.
"""

import logging

from ..agents.early_stopper_agent import EarlyStopperAgent
from .base_early_stopper import BaseEarlyStopper

logger = logging.getLogger(__name__)


class AgenticEarlyStopper(BaseEarlyStopper):
    """
    Agentic early stopper that uses an LLM-based agent to decide whether to stop.

    This early stopper leverages the EarlyStopperAgent to analyze:
    - Current training metrics up to checkpoint
    - Best run training metrics
    - Best run validation score

    And make a data-driven decision on whether training should continue.
    """

    def _create_agent(self):
        """Create the text-based early stopper agent."""
        return EarlyStopperAgent(
            config=self.config,
            manager=self.manager,
            llm_config=self.agent_llm_config,
            prompt_template=self.agent_prompt_template,
        )

    def _get_logger_prefix(self) -> str:
        """Get the logger prefix for this early stopper type."""
        return "AgenticEarlyStopper"
