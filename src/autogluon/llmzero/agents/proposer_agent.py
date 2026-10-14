import logging

from ..prompts.proposer_agent_prompt import ProposerAgentPrompt
from .base_agent import BaseAgent
from .utils import init_llm

logger = logging.getLogger(__name__)


class ProposerAgent(BaseAgent):
    """
    Analyzes training history and suggests hyperparameter changes to try next.

    This agent examines:
    - Previous post-training script
    - Training dynamics (losses, gradients, KL divergence, etc.)
    - Validation metrics (rewards, scores, etc.)

    And provides suggestions for promising hyperparameter changes such as:
    - Learning rate adjustments
    - Batch size changes
    - KL penalty coefficient tuning
    - Training step modifications
    - PPO clipping parameter adjustments

    Agent Input:
    - Previous post-training script
    - Training dynamics signals
    - Validation metrics signals

    Agent Output:
    - Textual suggestions for hyperparameter changes with reasoning
    """

    def __init__(self, config, manager, llm_config, prompt_template=None):
        super().__init__(config=config, manager=manager)

        self.proposer_llm_config = llm_config
        self.proposer_prompt_template = prompt_template

        self.proposer_prompt = ProposerAgentPrompt(
            llm_config=self.proposer_llm_config,
            manager=self.manager,
            template=self.proposer_prompt_template,
        )

        if self.proposer_llm_config.multi_turn:
            self.proposer_llm = init_llm(
                llm_config=self.proposer_llm_config,
                agent_name="proposer_agent",
                multi_turn=self.proposer_llm_config.multi_turn,
            )

    def __call__(self, cold_start: bool = False):
        self.manager.log_agent_start(
            "ProposerAgent: starting to analyze training history and generate hyperparameter suggestions."
        )

        # Get config parameters for proposer
        proposer_config = getattr(self.config, "agentic_proposer", {})
        num_samples = getattr(proposer_config, "num_samples", 10)
        use_average = getattr(proposer_config, "use_average", True)
        text_metrics = getattr(proposer_config, "text_metrics", [])
        plot_metrics = getattr(proposer_config, "plot_metrics", [])

        # Fall back to framework defaults when metric lists are not configured
        if not text_metrics or not plot_metrics:
            fw = self.training_framework
            if not text_metrics:
                text_metrics = fw.default_text_metrics()
            if not plot_metrics:
                plot_metrics = fw.default_plot_metrics()

        # Get validation metric pattern from main config
        validation_metric_pattern = getattr(self.config, "validation_metric_pattern", None)

        # Build prompt for generating hyperparameter suggestions
        prompt = self.proposer_prompt.build(
            num_samples=num_samples,
            use_average=use_average,
            text_metrics=text_metrics,
            plot_metrics=plot_metrics,
            validation_metric_pattern=validation_metric_pattern,
            cold_start=cold_start,
        )

        if not self.proposer_llm_config.multi_turn:
            self.proposer_llm = init_llm(
                llm_config=self.proposer_llm_config,
                agent_name="proposer_agent",
                multi_turn=self.proposer_llm_config.multi_turn,
            )

        response = self.proposer_llm.assistant_chat(prompt)

        suggestions = self.proposer_prompt.parse(response)

        self.manager.log_agent_end("ProposerAgent: hyperparameter suggestions generated and parsed from response.")

        return suggestions
