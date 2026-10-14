import logging
import re
from typing import Dict, Optional, Tuple

from .base_prompt import BasePrompt

logger = logging.getLogger(__name__)


class ExecuterPrompt(BasePrompt):
    """Handles prompts for code execution evaluation"""

    @classmethod
    def meta_instructions(cls) -> str:
        """
        Returns specific instructions for meta-prompting the Executer template.
        """
        return """
The ExecuterPrompt evaluates code execution results to determine success or failure, extract validation scores, and identify errors.

Considerations for rewriting this template:
1. Focus on robust error detection and classification in execution outputs
2. Include domain-specific success criteria relevant to the machine learning task
3. Improve detection and extraction of performance metrics and validation scores
4. Add specific checks for common failure modes in the given task context
5. Ensure clear decision boundaries between success and failure states
"""

    def default_template(self) -> str:
        """Default template for code execution evaluation"""
        return """You are an expert code evaluator. Analyze the execution results of the following code and determine if the execution was successful or if issues need to be fixed.

### Task Descriptions
{execution_task}

### Data Structure
{execution_data}

### Code to Analyze
{code_to_analyze}

## Execution Results
### Standard Output (stdout)

{stdout_truncate_start_8192}

### Standard Error (stderr)

{stderr_truncate_start_8192}

Evaluate the execution results and decide on one of the following actions:
1. SUCCESS - If the execution was completely successful and met all requirements.
2. FIX - If there were errors, issues, or performance problems that need to be addressed.

**Important**: Ignore cleanup/teardown errors and infrastructure messages that occur during or after training:
- Wandb BrokenPipeError during atexit (logging cleanup after training finished)
- Distributed teardown errors: broken pipe / TCPStore connection closed during worker shutdown
- Ray autoscaler messages (e.g., "Removing N nodes of type gpu-group", "Resized to X CPUs, Y GPUs") — these are normal cluster scaling operations, NOT errors
- These are harmless artifacts of cluster management and process shutdown order, not real failures

Check for **training completion indicators** in stdout:
- "Best checkpoint folder created" or similar checkpoint messages
- "training took X.XX minutes" timing summaries
- Validation metrics logged (val-core/*, val-aux/*)

If training completed successfully but only cleanup failed → DECISION: SUCCESS

Provide your decision in the following format:
DECISION: [SUCCESS or FIX]
ERROR_SUMMARY: [Brief summary of errors if any, or "None" if no errors]
VALIDATION_SCORE: [If there is a validation score for the solution, provide it as a number, otherwise "None"]

The error summary should be brief but informative enough for another agent to understand what needs to be fixed.
Even if the code executed without throwing errors, it might still have issues with logic or not meet all requirements.

For validation scores:
- If there is a validation score present in the execution results, extract it (e.g. the last validation score reported in the training process).
- Convert the score to ensure higher values indicate better performance (multiply "lower is better" metrics like RMSE, MAE, or loss by -1)
- Return the converted score that follows the "higher is better" convention"""

    @staticmethod
    def _clean_output(output: str, output_type: str = "output") -> str:
        """Clean output by removing verbose noise from distributed training logs.

        Filters:
        1. ANSI escape codes (color codes from Ray worker prefixes)
        2. NCCL INFO lines (channel setup, ring/tree topology — extremely verbose)
        3. C++ stack traces (frame #N: ... lines from libtorch)
        4. Distributed teardown errors (TCPStore, ProcessGroupNCCL shutdown noise)
        5. Wandb BrokenPipeError tracebacks
        6. Ray "Status message" duplicate logs (repeats the same error tail)

        Leaves concise summary markers so the executer can still identify what happened.
        """
        if not output:
            return output

        cleaned = output

        # --- Step 1: Strip ANSI escape codes ---
        # Matches sequences like \x1b[36m, \x1b[0m, \x1b[31m, etc.
        ansi_pattern = r"\x1b\[[0-9;]*m"
        cleaned = re.sub(ansi_pattern, "", cleaned)

        # --- Step 2: Remove Ray autoscaler messages ---
        # Lines like "(autoscaler +2h4m56s) Removing 2 nodes..." or "(autoscaler +2h4m56s) Resized to..."
        # These are normal cluster scaling operations, not errors.
        autoscaler_count = len(re.findall(r"^.*\(autoscaler\s+\+[^)]+\).*$", cleaned, re.MULTILINE))
        if autoscaler_count > 0:
            cleaned = re.sub(r"^.*\(autoscaler\s+\+[^)]+\).*\n?", "", cleaned, flags=re.MULTILINE)
            logger.debug(f"Removed {autoscaler_count} autoscaler lines from {output_type}")

        # --- Step 3: Remove NCCL INFO lines ---
        # These are extremely verbose (hundreds of lines) and not useful for debugging.
        # Lines contain "NCCL INFO" with optional Ray worker prefix like "(WorkerDict pid=...)"
        nccl_count = len(re.findall(r"^.*NCCL INFO.*$", cleaned, re.MULTILINE))
        if nccl_count > 0:
            cleaned = re.sub(r"^.*NCCL INFO.*\n?", "", cleaned, flags=re.MULTILINE)
            logger.debug(f"Removed {nccl_count} NCCL INFO lines from {output_type}")

        # --- Step 4: Remove C++ stack traces ---
        # Lines like "frame #0: c10::Error::Error(...) + 0x80 (0x... in ...libc10.so)"
        # and "Exception raised from ... at /pytorch/..."
        # and "<unknown function> + 0x..." lines
        cpp_frame_count = len(re.findall(r"^.*frame #\d+:.*$", cleaned, re.MULTILINE))
        if cpp_frame_count > 0:
            cleaned = re.sub(r"^.*frame #\d+:.*\n?", "", cleaned, flags=re.MULTILINE)
            cleaned = re.sub(r"^.*Exception raised from \w+ at /pytorch/.*\n?", "", cleaned, flags=re.MULTILINE)
            logger.debug(f"Removed {cpp_frame_count} C++ stack frame lines from {output_type}")

        # --- Step 5: Collapse distributed teardown errors ---
        # TCPStore "recvValue failed" + ProcessGroupNCCL "Failed to check" are normal
        # shutdown noise when distributed workers tear down out of order.
        # Replace all instances with a single summary marker.
        teardown_patterns = [
            r"^.*\[c10d\] recvValue failed.*Connection was likely closed.*\n?",
            r"^.*ProcessGroupNCCL.*Failed to check.*TCPStore.*\n?",
            r"^.*Failed to recv, got 0 bytes.*\n?",
        ]
        teardown_found = False
        for pattern in teardown_patterns:
            if re.search(pattern, cleaned, re.MULTILINE):
                teardown_found = True
                cleaned = re.sub(pattern, "", cleaned, flags=re.MULTILINE)

        if teardown_found:
            # Insert a single concise marker so the executer knows what happened
            # Place it before the Ray job failure message if present, otherwise at end
            ray_fail_match = re.search(r"-+\nJob '[^']+' failed\n-+", cleaned)
            marker = "[Distributed teardown error: broken pipe / TCPStore connection closed during worker shutdown]\n"
            if ray_fail_match:
                cleaned = cleaned[: ray_fail_match.start()] + marker + cleaned[ray_fail_match.start() :]
            else:
                cleaned = cleaned.rstrip() + "\n" + marker
            logger.debug(f"Collapsed distributed teardown errors in {output_type}")

        # --- Step 6: Clean wandb BrokenPipeError tracebacks ---
        # Pattern: "Exception ignored in atexit callback" ... "BrokenPipeError: [Errno 32] Broken pipe"
        # May have Ray worker prefixes on each line.
        wandb_pattern = r"Exception ignored in atexit callback:.*?BrokenPipeError: \[Errno 32\] Broken pipe"
        if re.search(wandb_pattern, cleaned, re.DOTALL):
            cleaned = re.sub(
                wandb_pattern,
                "[Wandb cleanup error: BrokenPipeError during atexit - harmless logging teardown issue]",
                cleaned,
                flags=re.DOTALL,
            )
            logger.debug(f"Cleaned wandb BrokenPipeError traceback from {output_type}")

        # --- Step 7: Remove Ray "Status message" duplicate logs ---
        # When a Ray job fails, it appends "Status message: Job entrypoint command failed with exit code 1,
        # last available logs (truncated to 20,000 chars): <repeat of tail logs>"
        # This duplicates the error we already have. Remove the duplicated tail.
        status_msg_pattern = r"Status message: Job entrypoint command failed with exit code \d+, last available logs \(truncated to [0-9,]+ chars\):.*"
        if re.search(status_msg_pattern, cleaned, re.DOTALL):
            cleaned = re.sub(
                status_msg_pattern,
                "Status message: Job entrypoint command failed with exit code 1 (duplicate logs removed, see above)",
                cleaned,
                flags=re.DOTALL,
            )
            logger.debug(f"Removed duplicate Ray status message logs from {output_type}")

        # --- Step 8: Collapse consecutive blank lines ---
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        return cleaned

    def _build(
        self, stdout: str, stderr: str, code_to_analyze: str, execution_task: str, execution_data: str, **kwargs
    ) -> str:
        """Build a prompt for the LLM to evaluate execution logs.

        Args:
            stdout: Standard output from code execution
            stderr: Standard error from code execution
            code_to_analyze: Code to analyze (Python, Bash, Ray script)
            execution_task: Description of the execution task
            execution_data: Data structure information
            **kwargs: Additional keyword arguments to customize the prompt building process
        """

        # Save originals before cleaning
        self.manager.save_and_log_states(
            content=stdout, save_name="stdout.orig.txt", per_iteration=True, add_uuid=True
        )
        self.manager.save_and_log_states(
            content=stderr, save_name="stderr.orig.txt", per_iteration=True, add_uuid=True
        )

        # Clean both stdout and stderr to reduce context length
        stdout_cleaned = self._clean_output(stdout, "stdout")
        stderr_cleaned = self._clean_output(stderr, "stderr")

        self.manager.save_and_log_states(
            content=stdout_cleaned, save_name="stdout.txt", per_iteration=True, add_uuid=True
        )
        self.manager.save_and_log_states(
            content=stderr_cleaned, save_name="stderr.txt", per_iteration=True, add_uuid=True
        )

        # Render the prompt using the variable provider with additional variables
        additional_vars = {
            "execution_task": execution_task,
            "execution_data": execution_data,
            "code_to_analyze": code_to_analyze,
            "stdout": stdout_cleaned or "No standard output",
            "stderr": stderr_cleaned or "No standard error",
        }

        prompt = self.render(additional_vars)

        self.manager.save_and_log_states(
            content=prompt, save_name="executer_prompt.txt", per_iteration=True, add_uuid=True
        )

        return prompt

    def parse(self, response: Dict) -> Tuple[str, Optional[str], Optional[float]]:
        """Parse the LLM's response to extract decision, error summary, and validation score."""

        # Extract content from LLM response
        if isinstance(response, dict) and "content" in response:
            content = response["content"]
        elif isinstance(response, str):
            content = response
        else:
            logger.warning("Unexpected response format from LLM")
            return "FIX", "Parser error", None

        # Parse the decision
        decision = "FIX"  # Default to FIX if parsing fails
        if "DECISION:" in content:
            decision_line = [line for line in content.split("\n") if "DECISION:" in line]
            if decision_line:
                decision_text = decision_line[0].split("DECISION:")[1].strip()
                if "SUCCESS" in decision_text.upper():
                    decision = "SUCCESS"
                elif "FIX" in decision_text.upper():
                    decision = "FIX"

        # Parse the error summary
        error_summary = None
        if "ERROR_SUMMARY:" in content:
            error_summary_parts = content.split("ERROR_SUMMARY:")[1].strip()
            error_summary = error_summary_parts.split("\n")[0].strip()
            if error_summary.lower() == "none" or not error_summary:
                error_summary = None

        # Parse the validation score
        validation_score = None
        if "VALIDATION_SCORE:" in content:
            validation_score_parts = content.split("VALIDATION_SCORE:")[1].strip()
            validation_score_text = validation_score_parts.split("\n")[0].strip()
            if validation_score_text.lower() != "none" and validation_score_text:
                try:
                    validation_score = float(validation_score_text)
                except ValueError:
                    logger.warning(f"Could not parse validation score: {validation_score_text}")
                    validation_score = None
        # The Validation score is only meaningful if this is a success run
        if decision != "SUCCESS":
            validation_score = None

        self.manager.save_and_log_states(
            content=response, save_name="executer_response.txt", per_iteration=True, add_uuid=True
        )
        self.manager.save_and_log_states(content=decision, save_name="decision.txt", per_iteration=True, add_uuid=True)
        self.manager.save_and_log_states(
            content=error_summary, save_name="error_summary.txt", per_iteration=True, add_uuid=True
        )
        self.manager.save_and_log_states(
            content=str(validation_score), save_name="validation_score.txt", per_iteration=True, add_uuid=True
        )

        return decision, error_summary, validation_score
