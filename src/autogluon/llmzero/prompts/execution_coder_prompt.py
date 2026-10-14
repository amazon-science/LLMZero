import logging
from typing import Dict

from .base_prompt import BasePrompt
from .utils import extract_code

logger = logging.getLogger(__name__)


class ExecutionCoderPrompt(BasePrompt):
    """Handles prompts for execution wrapper script generation"""

    @classmethod
    def meta_instructions(cls) -> str:
        """
        Returns specific instructions for meta-prompting the execution coder template.
        """
        return """
The ExecutionCoderPrompt generates environment-specific execution wrapper scripts.

This prompt generates a Ray job submission wrapper that embeds the entry point script inline using heredoc.
This approach avoids path length issues and unnecessary directory uploads.

Considerations for rewriting this template:
1. Focus on Ray job submission format and configuration
2. Inline entry point script content using heredoc (bash <<'SCRIPT_EOF')
3. Configure runtime environment (env vars, dependencies) in runtime-env-json
4. Do NOT use --working-dir (to avoid path length issues)
5. Ensure variables in the entry point script expand correctly at runtime
6. Include appropriate error handling for job submission
"""

    def _get_backend_script_template(self) -> str:
        """Get the execution script template section from the configured compute backend."""
        try:
            from omegaconf import OmegaConf

            from ..compute_infra import get_backend

            backend_type = getattr(self.manager.config.compute_infra, "type", "raycluster")
            backend_cls = get_backend(backend_type)
            config_dict = OmegaConf.to_container(self.manager.config.compute_infra, resolve=True)
            return backend_cls.execution_script_template(config_dict)
        except Exception:
            from ..compute_infra.raycluster import RayClusterBackend

            return RayClusterBackend.execution_script_template()

    def default_template(self) -> str:
        backend_template = self._get_backend_script_template()
        return f"""Create an execution script that submits a compute job.

This script will be called by run.sh which handles environment setup (conda, AWS credentials, port forwarding).

### Entry Point Script Content
{{entry_point_script}}

### User Instruction
{{user_input_truncate_end_2048}}

### WandB Configuration
WandB Enabled: {{wandb_enabled}}
WandB API Key: {{wandb_api_key}}
WandB Project Postfix: {{wandb_project_postfix}}
Note: If WandB Project Postfix is provided, append it to the project name in training config files (e.g., "MyProject" + "_v2" = "MyProject_v2")

### Previous Execution Script
{{previous_execution_script}}

### Previous Run Error Analysis
{{previous_error_analysis}}

### Other Historical Error Analyses
{{other_previous_error_analyses}}

{backend_template}
"""

    def _build(self, **kwargs) -> str:
        """Build a prompt for the LLM to generate an execution wrapper script.

        Args:
            **kwargs: Additional keyword arguments to customize the prompt building process
        """
        assert self.manager.time_step >= 0, "run manager.step(user_input) before retrieving the prompt"

        prompt = self.render()

        # Always add format instruction for execution script
        format_instruction = (
            "Please format your response with the script in a ```bash``` code block to make it easily extractable."
        )
        prompt = f"{prompt}\n\n{format_instruction}"

        if self.manager.current_node.parent is not None:
            if self.manager.current_node.parent.is_successful:
                prompt = f"{prompt}\n\nPrevious execution script works properly. Keep the same execution configuration unless changes are needed."

        self.manager.save_and_log_states(
            content=prompt, save_name="execution_coder_prompt.txt", per_iteration=True, add_uuid=False
        )

        return prompt

    def parse(self, response: Dict) -> str:
        """Parse the LLM's response to extract the bash execution wrapper script"""

        extracted_script = extract_code(response=response, language="bash")

        self.manager.save_and_log_states(
            content=response, save_name="execution_coder_response.txt", per_iteration=True, add_uuid=False
        )
        self.manager.save_and_log_states(
            content=extracted_script, save_name="execution_script.sh", per_iteration=True, add_uuid=False
        )

        return extracted_script
