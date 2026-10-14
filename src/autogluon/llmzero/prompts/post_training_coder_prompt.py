import logging
from typing import Dict

from .base_prompt import BasePrompt
from .utils import extract_code

logger = logging.getLogger(__name__)


class PostTrainingCoderPrompt(BasePrompt):
    """Handles prompts for post-training entry point script generation"""

    @classmethod
    def meta_instructions(cls) -> str:
        return """
The PostTrainingCoderPrompt generates environment-agnostic bash scripts for training.

This prompt focuses on generating the CORE training logic WITHOUT any environment-specific execution wrappers.
The generated script should be a standalone entry point that can be executed in any environment.

Considerations for rewriting this template:
1. DO NOT include Ray job submission commands - this will be handled separately
2. Ensure proper model downloading and data preprocessing steps
3. Include appropriate error handling and logging
4. Configure training parameters appropriately
5. The script should be self-contained and executable as-is
"""

    def default_template(self) -> str:
        return """Create a bash entry point script for training.

IMPORTANT: Generate ONLY the training preparation and execution logic.
DO NOT include Ray job submission commands (like "ray job submit").
This will be a standalone script that gets called by the execution environment.

### Descriptions
{description_file_contents}

### User Instruction
{user_input_truncate_end_2048}

### Previous Entry Point Script
{previous_entry_point_script}

### Previous Run Error Analysis
{previous_error_analysis}

### Other Historical Error Analyses
{other_previous_error_analyses}

### Hyperparameter Optimization Suggestions
{proposer_suggestions}

### Checkpoint Resumption
{checkpoint_resume_info}

### WandB Configuration
WandB Enabled: {wandb_enabled}

Based on the description and the user instruction, generate a new bash entry point script that:
1. Downloads/prepares the model (if needed)
2. Sets up environment variables
3. Uses the output folder name: {node_output_folder_name}
4. Executes the training command with appropriate parameters
5. Configures WandB logging based on whether it's enabled

IMPORTANT:
- AVOID bash variables like OUTPUT_DIR (they break in bash -lc embedding)
- Repeat the path literally everywhere instead of using variables
- Use the output folder name {node_output_folder_name} directly in all paths
- If "Previous Run Error Analysis" section above has content, the previous run FAILED - focus on fixing those errors
- If "Previous Run Error Analysis" is empty, the previous run SUCCEEDED - apply the proposer's hyperparameter suggestions
- **Hyperparameter changes MUST come from the proposer's suggestions above.** Select some or all of the proposer's suggested changes. Do NOT invent new hyperparameter changes beyond what the proposer listed, unless strictly necessary to fix an error or resolve a compatibility issue (e.g., adjusting micro_batch_size when batch_size changes). If you must deviate, explain why.
- If the "Checkpoint Resumption" section has content, follow ALL of its instructions exactly — including any `trainer.total_training_steps` / `trainer.total_epochs` step-cap values (these apply to both resumed AND from-scratch runs) and any `resume_mode`/`resume_from_path` settings. Use the exact values it specifies; do not substitute your own epoch count.

WandB Configuration:
- If "WandB Enabled" is True, ENABLE wandb logging in the training command (e.g., using --enable_wandb or similar flags)
- If "WandB Enabled" is False, DISABLE wandb logging (e.g., using --disable_wandb or removing wandb-related flags)
- The WANDB_API_KEY environment variable will be set automatically in the execution environment if enabled

Focus on modifying the training command arguments to match the new requirements and suggestions.
"""

    def _get_checkpoint_resume_info(self) -> str:
        """Build checkpoint resume instructions for the training script."""
        node = self.manager.current_node
        ckpt_config = getattr(self.manager.config, "checkpoint_optimization", None)
        return self.training_framework.checkpoint_resume_info(node, ckpt_config)

    def _build(self, **kwargs) -> str:
        """Build a prompt for the LLM to generate a post-training entry point script."""
        assert self.manager.time_step >= 0, "run manager.step(user_input) before retrieving the prompt"

        prompt = self.render(
            additional_vars={
                "checkpoint_resume_info": self._get_checkpoint_resume_info(),
            }
        )

        format_instruction = (
            "Please format your response with the script in a ```bash``` code block to make it easily extractable."
        )
        prompt = f"{prompt}\n\n{format_instruction}"

        if self.manager.current_node.parent is not None:
            if self.manager.current_node.parent.is_successful:
                prompt = f"{prompt}\n\nPrevious entry point script works properly. Apply the proposer's suggested changes only."

        self.manager.save_and_log_states(
            content=prompt, save_name="post_training_coder_prompt.txt", per_iteration=True, add_uuid=False
        )

        return prompt

    def parse(self, response: Dict) -> str:
        """Parse the LLM's response to extract the bash entry point script"""

        extracted_script = extract_code(response=response, language="bash")

        self.manager.save_and_log_states(
            content=response, save_name="post_training_coder_response.txt", per_iteration=True, add_uuid=False
        )
        self.manager.save_and_log_states(
            content=extracted_script, save_name="run.sh", per_iteration=True, add_uuid=False
        )

        return extracted_script
