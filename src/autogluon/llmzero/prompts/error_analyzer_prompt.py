import logging
import re
from typing import Optional

from .base_prompt import BasePrompt

logger = logging.getLogger(__name__)


class ErrorAnalyzerPrompt(BasePrompt):
    """Handles prompts for error analysis"""

    @classmethod
    def meta_instructions(cls) -> str:
        """
        Returns specific instructions for meta-prompting the Error Analyzer template.
        """
        return """
The ErrorAnalyzerPrompt analyzes execution errors and provides concise, actionable feedback on how to fix them.

Considerations for rewriting this template:
1. Focus on clear diagnostic analysis of root causes rather than symptoms
2. Emphasize actionable, specific guidance for error resolution
3. Structure the output to clearly separate error diagnosis from solution steps
4. Include checks for common error patterns specific to the task domain
5. Consider the context of previous errors and solutions when providing recommendations
"""

    def default_template(self) -> str:
        """Default template for code execution evaluation"""
        return """
Analyze the error and provide your response in this exact format:

ERROR_SUMMARY: [Brief technical description of the root cause in 1-3 sentences]\nSUGGESTED_FIX: [One actionable solutions addressing different aspects of the problem. Provide specific debugging directions without code blocks]

**IMPORTANT**: Ignore BrokenPipeError during atexit (wandb/logging cleanup after training completes). If training finished successfully (checkpoints created, validation metrics logged, timing summaries present) but only cleanup failed, this is NOT a real error.

### Error Message
{error_message_truncate_start_16384}

### Early Stopping Information
Early Stop Decision: {early_stop_decision}
Early Stop Reason: {early_stop_reason}

Note: If training was early stopped, consider whether the error is related to the stopping decision or occurred before/after the stop point.

### User Instructions
{user_input}

### Current Post Training Script
{entry_point_script}

### Current Job Submission Script
{execution_script}

### Relevant Tutorials
{tutorial_prompt}
"""

    def _build(self, **kwargs) -> str:
        """Build a prompt for the LLM to analyze errors.

        Args:
            **kwargs: Additional keyword arguments to customize the prompt building process
        """

        # Render the prompt using the variable provider
        prompt = self.render()

        self.manager.save_and_log_states(
            content=prompt, save_name="error_analyzer_prompt.txt", per_iteration=True, add_uuid=False
        )

        return prompt

    def parse(self, response: str) -> Optional[str]:
        analysis_match = re.search(r"ERROR_SUMMARY:\s*(.*)", response, re.DOTALL)
        if analysis_match:
            error_analysis = f"ERROR_SUMMARY: {analysis_match.group(1).strip()}"
        else:
            error_analysis = "Failed to extract error analysis from LLM response."

        self.manager.save_and_log_states(
            content=response, save_name="error_analyzer_response.txt", per_iteration=True, add_uuid=False
        )
        self.manager.save_and_log_states(
            content=error_analysis, save_name="error_analysis.txt", per_iteration=True, add_uuid=False
        )
        return error_analysis
