"""
Unified prompt for early stopper agent with configurable text and visual modalities.

Extends ProposerAgentPrompt for plot generation and adds comparison
plots overlaying current run vs best run metrics.
"""

import base64
import io
import logging
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt

from .proposer_agent_prompt import ProposerAgentPrompt

# Use non-interactive backend for matplotlib
matplotlib.use("Agg")

logger = logging.getLogger(__name__)


class EarlyStopperAgentPrompt(ProposerAgentPrompt):
    """
    Unified prompt for early stopper agent with configurable text and visual modalities.

    Extends ProposerAgentPrompt by generating comparison plots that overlay
    current run metrics against best run metrics. This enables visual comparison
    of training dynamics to decide whether to stop early.
    """

    FRAMEWORK_AGENT_NAME = "early_stopper"

    def default_template(self) -> str:
        return """You are {agent_role}. Your task is to decide if a run should continue or stop early.

### Objective
{objective}

### Best Run Information
{best_run_summary}

### Current Run Progress (at checkpoint)
{current_run_summary}

### Visual Comparison
The plots show training metrics over time. Blue = current run, green dashed = best run, both may start from an intermediate checkpoint.

Focus on the **validation score plot**. Ask: can the current run's trajectory eventually beat the best validation score seen so far?

{metric_descriptions}

### Decision Rules
**Default action is CONTINUE.** When uncertain → CONTINUE. Only STOP if ALL of these are true:
1. Current step ≥ 10 (too early to judge before that)
2. The validation score trajectory has no realistic chance of beating the best validation score seen so far — considering the improvement rate, not just the current value. A run behind the best can still win if its trajectory is improving faster; a run ahead can still lose if it's plateauing.

{metric_interpretation}

**Response Format:**
```
REASON: [1-2 sentences on whether the validation score trajectory can beat the best validation score seen so far]
DECISION: [CONTINUE or STOP]
```
"""

    def _build(self, **kwargs):
        """Build a multimodal prompt with comparison plots.

        Args:
            **kwargs: Keyword arguments including:
                - current_metrics_history: Current run metrics up to checkpoint
                - best_run_metrics_history: Best run metrics (or None)
                - best_run_validation_score: Best run validation score (or None)
                - num_samples: Number of samples for summary (default: 10)
                - use_average: Whether to average within segments (default: True)
                - text_metrics: List of metrics to include in text summaries (optional)
                - plot_metrics: List of metrics to visualize as plots (optional)
                - validation_metric_pattern: Regex pattern to match validation metrics (optional)
                - elapsed_time: Time elapsed since training started (in seconds)
        """
        current_metrics_history = kwargs.get("current_metrics_history", [])
        best_run_metrics_history = kwargs.get("best_run_metrics_history")
        best_run_validation_score = kwargs.get("best_run_validation_score")
        num_samples = kwargs.get("num_samples", self.DEFAULT_NUM_SAMPLES)
        use_average = kwargs.get("use_average", self.DEFAULT_USE_AVERAGE)
        text_metrics = kwargs.get("text_metrics", [])
        plot_metrics = kwargs.get("plot_metrics", [])
        validation_metric_pattern = kwargs.get("validation_metric_pattern")
        elapsed_time = kwargs.get("elapsed_time")

        # Generate comparison plots if plot_metrics specified
        image_data = []
        if plot_metrics:
            image_data = self._generate_comparison_plots(
                current_metrics_history,
                best_run_metrics_history,
                plot_metrics,
                validation_metric_pattern,
                elapsed_time,
            )

        # Build text summaries if text_metrics specified
        current_run_summary = ""
        best_run_summary = ""
        if text_metrics:
            current_run_summary = self._build_metrics_summary(
                current_metrics_history, num_samples, use_average, text_metrics
            )
            best_run_summary = self._build_best_run_summary(
                best_run_metrics_history, best_run_validation_score, num_samples, use_average, text_metrics
            )
        else:
            current_run_summary = "Text summaries disabled (no text_metrics configured)."
            best_run_summary = (
                "Text summaries disabled (no text_metrics configured)."
                if best_run_metrics_history
                else "**No previous best run exists.** This is the first training run."
            )

        # Build metric descriptions for all metrics (text + plot)
        all_metrics = list(set(text_metrics + plot_metrics))
        metric_descriptions = ""
        if all_metrics:
            metric_descriptions = self.training_framework.format_metric_descriptions(all_metrics)

        # Framework-supplied prompt fragments (neutral default template placeholders).
        # Fallbacks keep the rendered prompt sensible for frameworks with no fragments.
        fw_frags = self.training_framework.prompt_fragments("early_stopper")

        additional_vars = {
            "best_run_summary": best_run_summary,
            "current_run_summary": current_run_summary,
            "metric_descriptions": metric_descriptions,
            "agent_role": fw_frags.get("role", "an expert at analyzing training runs using visual plots"),
            "objective": fw_frags.get(
                "objective",
                "Decide whether the current run can still beat the best validation score seen so far. "
                "CONTINUE if it has potential to; STOP only when there is no realistic chance.",
            ),
            "metric_interpretation": fw_frags.get("metric_interpretation", ""),
        }

        # Render text prompt
        text_prompt = self.render(additional_vars=additional_vars)

        # Build multimodal content
        if not image_data:
            return text_prompt

        multimodal_prompt = self._build_multimodal_content(text_prompt, image_data)
        return multimodal_prompt

    def _build_metrics_summary(
        self,
        metrics_history: List[Dict],
        num_samples: int,
        use_average: bool,
        comparison_metrics: Optional[List[str]] = None,
    ) -> str:
        """Build a summary of metrics history."""
        if not metrics_history:
            return "No metrics available."

        # Create a dict structure expected by parent class methods
        step_metrics = {"metrics_history": metrics_history}

        return self._get_step_metrics_summary(step_metrics, num_samples, use_average, comparison_metrics or [])

    def _build_best_run_summary(
        self,
        best_run_metrics_history: Optional[List[Dict]],
        best_run_validation_score: Optional[float],
        num_samples: int,
        use_average: bool,
        comparison_metrics: Optional[List[str]] = None,
    ) -> str:
        """Build summary of best run."""
        if best_run_metrics_history is None:
            return "**No previous best run exists.** This is the first training run."

        parts = []

        if best_run_validation_score is not None:
            parts.append(f"**Best Run Validation Score:** {best_run_validation_score:.4f}")

        parts.append(f"**Best Run Training Steps:** {len(best_run_metrics_history)}")

        # Add metrics summary
        metrics_summary = self._build_metrics_summary(
            best_run_metrics_history, num_samples, use_average, comparison_metrics
        )
        parts.append(f"\n{metrics_summary}")

        return "\n".join(parts)

    def _generate_comparison_plots(
        self,
        current_metrics: List[Dict],
        best_metrics: Optional[List[Dict]],
        comparison_metrics: Optional[List[str]] = None,
        validation_metric_pattern: Optional[str] = None,
        elapsed_time: Optional[float] = None,
    ) -> List[Dict]:
        """
        Generate comparison plots overlaying current vs best run metrics.

        Args:
            current_metrics: Current run metrics history
            best_metrics: Best run metrics history (or None)
            comparison_metrics: List of metric keys to plot
            validation_metric_pattern: Regex pattern to match validation metrics
            elapsed_time: Time elapsed since training started (in seconds)

        Returns:
            List of dictionaries containing image data and descriptions
        """
        if not current_metrics:
            logger.warning("No current metrics available for plotting")
            return []

        # Determine metrics to plot
        # Scan ALL steps to find validation metrics (not just first step, since validation occurs at intervals)
        available_metrics = set()
        for entry in current_metrics:
            if entry.get("metrics"):
                available_metrics.update(entry["metrics"].keys())

        metrics_to_plot = comparison_metrics or self.training_framework.default_plot_metrics()
        metrics_to_plot = [m for m in metrics_to_plot if m in available_metrics]

        # Auto-include validation metrics matching the configured pattern
        if validation_metric_pattern:
            import re

            validation_regex = re.compile(validation_metric_pattern)
            validation_metrics = [m for m in available_metrics if validation_regex.search(m)]
            for val_metric in validation_metrics:
                if val_metric not in metrics_to_plot:
                    metrics_to_plot.append(val_metric)
                    logger.info(
                        f"Auto-included validation metric for plotting (matched pattern '{validation_metric_pattern}'): {val_metric}"
                    )

        if not metrics_to_plot:
            logger.warning("No comparable metrics found for plotting")
            return []

        image_data = []

        for metric_key in metrics_to_plot:
            metric_image = self._plot_comparison_metric(current_metrics, best_metrics, metric_key)
            if metric_image:
                image_data.append(
                    {
                        "image_base64": metric_image,
                        "description": f"Comparison: {metric_key}",
                    }
                )

        # Save plots to disk
        self._save_plots_to_disk(image_data, elapsed_time)

        return image_data

    def _plot_comparison_metric(
        self,
        current_metrics: List[Dict],
        best_metrics: Optional[List[Dict]],
        metric_key: str,
    ) -> Optional[str]:
        """
        Plot a single metric comparing current run vs best run, showing ALL individual points.

        If a metric has multiple values per step, all values are plotted to preserve spikes.

        Args:
            current_metrics: Current run metrics history
            best_metrics: Best run metrics history (or None)
            metric_key: Metric key to plot

        Returns:
            Base64 encoded PNG image string, or None if plotting fails
        """
        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))

            # Plot current run (blue solid line) - expand all points
            current_steps = []
            current_values = []
            for entry in current_metrics:
                if metric_key in entry.get("metrics", {}):
                    step = entry["step"]
                    value = entry["metrics"][metric_key]

                    # Expand lists/arrays into individual points
                    if hasattr(value, "ndim") and value.ndim > 0:
                        for v in value.flatten():
                            current_steps.append(step)
                            current_values.append(float(v))
                    elif isinstance(value, list):
                        for v in value:
                            current_steps.append(step)
                            current_values.append(float(v))
                    else:
                        current_steps.append(step)
                        current_values.append(float(value))

            if current_steps and current_values:
                ax.plot(
                    current_steps,
                    current_values,
                    "b-",
                    linewidth=1,
                    alpha=0.8,
                    label="Current Run",
                )

            # Plot best run (green dashed line) if available - expand all points
            if best_metrics:
                best_steps = []
                best_values = []
                for entry in best_metrics:
                    if metric_key in entry.get("metrics", {}):
                        step = entry["step"]
                        value = entry["metrics"][metric_key]

                        # Expand lists/arrays into individual points
                        if hasattr(value, "ndim") and value.ndim > 0:
                            for v in value.flatten():
                                best_steps.append(step)
                                best_values.append(float(v))
                        elif isinstance(value, list):
                            for v in value:
                                best_steps.append(step)
                                best_values.append(float(v))
                        else:
                            best_steps.append(step)
                            best_values.append(float(value))

                if best_steps and best_values:
                    ax.plot(
                        best_steps,
                        best_values,
                        "g--",
                        linewidth=1,
                        alpha=0.6,
                        label="Best Run",
                    )

            # Formatting
            metric_name = metric_key.split("/")[-1].replace("_", " ").title()
            ax.set_xlabel("Training Step", fontsize=12)
            ax.set_ylabel(metric_name, fontsize=12)
            ax.set_title(f"Comparison: {metric_key}", fontsize=14, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")

            plt.tight_layout()

            # Convert to base64
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode("utf-8")

            plt.close(fig)

            logger.debug(f"Successfully plotted comparison for {metric_key}")
            return image_base64

        except Exception as e:
            logger.error(f"Error plotting comparison for {metric_key}: {e}", exc_info=True)
            return None

    def _build_multimodal_content(self, text_prompt: str, image_data: List[Dict]) -> Dict:
        """
        Build multimodal content structure for Bedrock Claude models.

        Args:
            text_prompt: The text portion of the prompt
            image_data: List of image dictionaries with base64 data

        Returns:
            Dictionary formatted for multimodal LLM input
        """
        if not image_data:
            return text_prompt

        content_parts = []

        # Add images first
        for img in image_data:
            content_parts.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img["image_base64"],
                    },
                }
            )

        # Add text prompt
        content_parts.append({"type": "text", "text": text_prompt})

        return {"multimodal": True, "content": content_parts}

    def _save_plots_to_disk(self, image_data: List[Dict], elapsed_time: Optional[float] = None):
        """Save plots to disk in early_stopper/<timestamp>/ folder."""
        if not image_data:
            return

        # Format elapsed time for folder name
        elapsed_str = f"{int(elapsed_time)}s" if elapsed_time is not None else "unknown"

        try:
            # Create early_stopper/<timestamp>/ directory
            iter_folder = self.manager.get_iteration_folder(self.manager.current_node)
            early_stopper_dir = os.path.join(iter_folder, "early_stopper", elapsed_str)
            os.makedirs(early_stopper_dir, exist_ok=True)

            for i, img_dict in enumerate(image_data):
                try:
                    image_bytes = base64.b64decode(img_dict["image_base64"])
                    metric_name = img_dict.get("description", f"comparison_{i+1}")
                    # Clean up the "Comparison: " prefix and sanitize filename
                    metric_name = metric_name.replace("Comparison: ", "").replace("Comparison:", "")
                    safe_filename = metric_name.replace("/", "_").replace(":", "_").replace(" ", "_")
                    filename = f"{safe_filename}.png"
                    filepath = os.path.join(early_stopper_dir, filename)

                    with open(filepath, "wb") as f:
                        f.write(image_bytes)

                    logger.debug(f"Saved comparison plot to {filepath}")
                except Exception as e:
                    logger.error(f"Error saving plot {i+1} to disk: {e}")
        except Exception as e:
            logger.error(f"Error creating states directory for plots: {e}")

    def parse(self, response: Dict, elapsed_time: Optional[float] = None) -> Tuple[bool, str]:
        """Parse the LLM's response to extract decision and reason.

        Args:
            response: LLM response dict or string
            elapsed_time: Time elapsed since training started (in seconds)

        Returns:
            Tuple of (should_stop, reason)
        """
        content = response.get("content", str(response)) if isinstance(response, dict) else str(response)

        # Save response for debugging with elapsed time in filename
        filename = "early_stopper_agent_response.txt"
        if elapsed_time is not None:
            elapsed_min = int(elapsed_time / 60)
            elapsed_sec = int(elapsed_time % 60)
            filename = f"early_stopper_agent_response_{elapsed_min}m{elapsed_sec}s.txt"

        self.manager.save_and_log_states(content=content, save_name=filename, per_iteration=True, add_uuid=False)

        # Parse decision
        content_upper = content.upper()

        # Look for explicit DECISION line
        should_stop = False
        if "DECISION:" in content_upper:
            decision_line = content_upper.split("DECISION:")[1].split("\n")[0].strip()
            should_stop = "STOP" in decision_line and "CONTINUE" not in decision_line
        else:
            # Fallback: check for STOP vs CONTINUE in content
            stop_count = content_upper.count("STOP")
            continue_count = content_upper.count("CONTINUE")
            should_stop = stop_count > continue_count

        # Extract reason
        reason = content
        if "REASON:" in content:
            reason = content.split("REASON:")[1].strip()
            # Take first paragraph only
            if "\n\n" in reason:
                reason = reason.split("\n\n")[0]
        elif "DECISION:" in content:
            # Use everything after decision line as reason
            parts = content.split("DECISION:")
            if len(parts) > 1:
                after_decision = parts[1]
                lines = after_decision.strip().split("\n")
                reason = "\n".join(lines[1:]).strip() if len(lines) > 1 else lines[0]

        return should_stop, reason.strip()
