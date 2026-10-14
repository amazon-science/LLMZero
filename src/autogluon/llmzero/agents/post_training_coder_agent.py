import logging

from ..prompts import PostTrainingCoderPrompt
from .base_agent import BaseAgent
from .utils import init_llm

logger = logging.getLogger(__name__)


class PostTrainingCoderAgent(BaseAgent):
    """
    Generate environment-agnostic post-training scripts (entry points) for RL tasks.

    This agent focuses on generating the core training logic including:
    - Model downloading and preparation
    - Data preprocessing
    - Training command with all parameters

    The generated script is independent of execution environment (Ray, local, etc.)

    Agent Input:
    - RL task specifications
    - Model requirements
    - Dataset information
    - Configuration parameters

    Agent Output:
    - Bash script containing training preparation and execution logic
    """

    def __init__(self, config, manager, llm_config, prompt_template):
        super().__init__(config=config, manager=manager)

        self.coder_llm_config = llm_config
        self.coder_prompt_template = prompt_template

        self.coder_prompt = PostTrainingCoderPrompt(
            llm_config=self.coder_llm_config,
            manager=self.manager,
            template=self.coder_prompt_template,
        )

        if self.coder_llm_config.multi_turn:
            self.coder_llm = init_llm(
                llm_config=self.coder_llm_config,
                agent_name="post_training_coder",
                multi_turn=self.coder_llm_config.multi_turn,
            )

    def __call__(self):
        self.manager.log_agent_start(
            "PostTrainingCoderAgent: starting to build and send post-training script generation prompt to the LLM."
        )

        # Build prompt for generating post-training entry point script
        prompt = self.coder_prompt.build()

        if not self.coder_llm_config.multi_turn:
            self.coder_llm = init_llm(
                llm_config=self.coder_llm_config,
                agent_name="post_training_coder",
                multi_turn=self.coder_llm_config.multi_turn,
            )

        response = self.coder_llm.assistant_chat(prompt)

        generated_script = self.coder_prompt.parse(response)

        self.manager.log_agent_end(
            "PostTrainingCoderAgent: post-training script generation prompt handled and code parsed from response."
        )

        return generated_script
