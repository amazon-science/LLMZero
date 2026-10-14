"""
Unified proposer agent prompt with flexible text/vision modalities.

This prompt supports selective inclusion of:
- Text summaries for specified metrics
- Visual plots for specified metrics
- Multimodal combination (text + plots)
"""

import base64
import io
import logging
import os
import re
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .base_prompt import BasePrompt

# Use non-interactive backend for matplotlib
matplotlib.use("Agg")

logger = logging.getLogger(__name__)


class ProposerAgentPrompt(BasePrompt):
    """
    Unified proposer prompt that supports flexible text/vision combinations.

    Configure metrics separately for text summaries vs plots:
    - text_metrics: List of metrics to show as text summaries
    - plot_metrics: List of metrics to visualize as plots
    - Metrics can appear in both (e.g., rewards as text AND plot)
    """

    # Constants
    DEFAULT_NUM_SAMPLES = 10
    DEFAULT_USE_AVERAGE = True

    FRAMEWORK_AGENT_NAME = "proposer"

    @classmethod
    def meta_instructions(cls) -> str:
        """
        Returns specific instructions for meta-prompting the proposer agent template.
        """
        return """
The ProposerAgentPrompt generates suggestions for hyperparameter changes to try next
in reinforcement learning post-training optimization.

This prompt should analyze:
1. The previous post-training script
2. Training dynamics signals (loss trends, gradient norms, etc.) as text and/or plots
3. Validation metrics signals (reward trends, score improvements, etc.) as text and/or plots

The proposer should provide actionable suggestions about what hyperparameters or
training configurations to change based on observed training behavior.

Configure which metrics appear as text vs plots via text_metrics and plot_metrics parameters.
"""

    def default_template(self) -> str:
        return """You are a hyperparameter optimization expert. Your goal is to {objective} by proposing hyperparameter changes for the next training run.

{cold_start_notice}

{previous_training_config}

{early_stop_notice}

{parent_error_context}

### Training Step Metrics Summary
The training ran for {training_steps} steps:

{step_metrics_summary}

{metric_descriptions}

{hyperparameter_guide}

{metric_interpretation}

### Your Task

Follow these steps **in order**. Do NOT suggest `total_epochs` until Step 4.

**Step 1 — Diagnosis:** Summarize the training health. What are the key trends in primary metrics? Is the run healthy, plateauing, or collapsing? Note any inflection points or anomalies.

**Step 2 — Hyperparameter suggestions:** Based on your diagnosis, propose specific hyperparameter changes for the next run. Do NOT include `total_epochs` here — that will be computed later.
{suggestion_count_guidance}
{sibling_dedup_notice}

{checkpoint_steps}

**Response format:**

{checkpoint_output_format}
"""

    @staticmethod
    def _to_scalar(value) -> float:
        """
        Convert a metric value to scalar, handling lists and numpy arrays.

        For metrics with multiple values per step (e.g., from different batches),
        this returns the maximum to preserve spikes (important for KL divergence spikes, etc.).
        """
        # Handle numpy arrays
        if hasattr(value, "ndim"):
            if value.ndim == 0:  # Scalar array
                return float(value)
            else:  # Multi-dimensional array - take max to preserve spikes
                return float(np.max(value))

        # Handle lists
        if isinstance(value, list):
            if not value:
                return 0.0
            # Check if it's a nested list
            if isinstance(value[0], (list, np.ndarray)):
                # Flatten and take max to preserve spikes
                flat = [
                    item
                    for sublist in value
                    for item in (sublist if isinstance(sublist, (list, np.ndarray)) else [sublist])
                ]
                return max(flat) if flat else 0.0
            else:
                return max(value)

        # Handle scalar values
        return float(value)

    def _build(self, **kwargs) -> str:
        """Build prompt with selective text/plot modalities.

        Args:
            **kwargs: Additional keyword arguments
                - num_samples: Number of uniform samples (default: 10)
                - use_average: Whether to average within segments (default: True)
                - text_metrics: List of metrics to include in text summaries (optional)
                - plot_metrics: List of metrics to visualize as plots (optional)
                - validation_metric_pattern: Regex pattern to match validation metrics (optional)
        """
        assert self.manager.time_step >= 0, "run manager.step(user_input) before retrieving the prompt"

        num_samples = kwargs.get("num_samples", self.DEFAULT_NUM_SAMPLES)
        use_average = kwargs.get("use_average", self.DEFAULT_USE_AVERAGE)
        text_metrics = kwargs.get("text_metrics", [])
        plot_metrics = kwargs.get("plot_metrics", [])
        validation_metric_pattern = kwargs.get("validation_metric_pattern")
        cold_start = kwargs.get("cold_start", False)

        # Get parent data
        step_metrics, training_steps = self._get_parent_data()

        # Generate text summaries if text_metrics specified
        step_metrics_summary = ""
        if text_metrics:
            step_metrics_summary = self._get_step_metrics_summary(step_metrics, num_samples, use_average, text_metrics)
        else:
            step_metrics_summary = "Text summaries disabled (no text_metrics configured)."

        # Generate plots if plot_metrics specified
        image_data = []
        if plot_metrics:
            image_data = self._generate_plots(num_samples, use_average, plot_metrics, validation_metric_pattern)

        # Build metric descriptions — only for metrics actually present in the data
        fw = self.training_framework
        all_metrics = list(set(text_metrics + plot_metrics))
        metric_descriptions = ""
        if all_metrics and step_metrics and "metrics_history" in step_metrics:
            available = set()
            for entry in step_metrics["metrics_history"]:
                if entry.get("metrics"):
                    available.update(entry["metrics"].keys())
            present_metrics = [m for m in all_metrics if m in available]
            if present_metrics:
                metric_descriptions = fw.format_metric_descriptions(present_metrics)

        # Build hyperparameter guide
        hyperparameter_guide = fw.format_hyperparameter_guide()

        # Check if parent was early stopped
        early_stop_notice = self._get_early_stop_notice()

        # Get checkpoint info for checkpoint-based optimization
        checkpoint_info = self._get_checkpoint_info()
        checkpoint_output_format = self._get_checkpoint_output_format()

        # Build checkpoint_steps: includes checkpoint decision + epoch math when enabled,
        # just epoch computation when disabled
        if checkpoint_info:
            checkpoint_steps = (
                "**Step 3 — Checkpoint decision:**\n"
                f"{checkpoint_info}\n\n"
                "**Step 4 — Training length:** Now that you have decided all hyperparameters "
                "(including `train_batch_size`) and the checkpoint strategy, set the run length. "
                "The training length of every node is enforced in code via `trainer.total_training_steps` "
                "(with `total_epochs` set large enough to complete it), so you do NOT compute `total_epochs`:\n"
                "- **If resuming:** choose `ADDITIONAL_TRAINING_STEPS` — the number of gradient steps to "
                "train beyond the resume point. The system inherits the checkpoint's global step and caps "
                "the run at `resume_step + ADDITIONAL_TRAINING_STEPS`. Stay within the stated cap.\n"
                "- **If training from scratch:** the run is automatically capped at the per-node step limit; "
                "no length input is needed from you.\n"
            )
        else:
            checkpoint_steps = (
                "**Step 3 — Epoch computation:** Choose `total_epochs` based on your target number of "
                "training steps and the batch size from Step 2. Show the math explicitly."
            )

        # Get previous training config YAML and suggestion guidance
        previous_training_config = self._get_previous_training_config()
        suggestion_count_guidance = self._get_suggestion_count_guidance()

        # Get sibling dedup notice (what siblings already tried from same parent).
        # During cold start we always surface sibling configs (independent of the
        # sibling_dedup flag) since divergence from the baseline/earlier siblings is
        # the ONLY signal available when there is no parent training history.
        sibling_dedup_notice = ""
        sibling_dedup_enabled = getattr(self.manager.config, "sibling_dedup", False)
        if (cold_start or sibling_dedup_enabled) and hasattr(self.manager, "sibling_config_summary"):
            sibling_dedup_notice = self.manager.sibling_config_summary

        # Cold-start banner: no parent metrics exist yet, so instruct the agent to
        # produce a DISTINCT initial configuration rather than tuning from history.
        cold_start_notice = self._get_cold_start_notice() if cold_start else ""

        # Framework-supplied prompt fragments (neutral default template placeholders).
        # Fallbacks keep the rendered prompt sensible for frameworks with no fragments.
        fw_frags = self.training_framework.prompt_fragments("proposer")

        additional_vars = {
            "objective": fw_frags.get("objective", "maximize the validation score"),
            "metric_interpretation": fw_frags.get("metric_interpretation", ""),
            "cold_start_notice": cold_start_notice,
            "step_metrics_summary": step_metrics_summary,
            "training_steps": training_steps,
            "metric_descriptions": metric_descriptions,
            "hyperparameter_guide": hyperparameter_guide,
            "early_stop_notice": early_stop_notice,
            "parent_error_context": self._get_parent_error_context(),
            "checkpoint_info": checkpoint_info,
            "checkpoint_steps": checkpoint_steps,
            "checkpoint_output_format": checkpoint_output_format,
            "previous_training_config": previous_training_config,
            "suggestion_count_guidance": suggestion_count_guidance,
            "sibling_dedup_notice": sibling_dedup_notice,
        }

        # Render text prompt
        text_prompt = self.render(additional_vars=additional_vars)

        # Save text for reference
        self.manager.save_and_log_states(
            content=text_prompt, save_name="proposer_agent_prompt.txt", per_iteration=True, add_uuid=False
        )

        # Return multimodal if we have plots, otherwise text-only
        if image_data:
            return self._build_multimodal_content(text_prompt, image_data)

        return text_prompt

    def _generate_plots(
        self,
        num_samples: int,
        use_average: bool,
        plot_metrics: List[str],
        validation_metric_pattern: Optional[str] = None,
    ) -> List[Dict]:
        """
        Generate plots for specified metrics.

        Note: Plots ALL steps without sampling or averaging across intervals.
        The num_samples and use_average parameters are kept for API consistency
        but are not used in plotting - they're only for text summaries.

        Args:
            num_samples: Not used in plotting (kept for API consistency)
            use_average: Not used in plotting (kept for API consistency)
            plot_metrics: List of metric keys to plot
            validation_metric_pattern: Regex pattern to match validation metrics

        Returns:
            List of dictionaries containing image data and descriptions
        """
        step_metrics, training_steps = self._get_parent_data()

        if not step_metrics or "metrics_history" not in step_metrics:
            logger.warning("No step metrics available for plotting")
            return []

        metrics_history = step_metrics["metrics_history"]
        image_data = []

        # Auto-include validation metrics matching the configured pattern
        # Scan ALL steps to find validation metrics (not just first step, since validation occurs at intervals)
        available_metrics = set()
        for entry in metrics_history:
            if entry.get("metrics"):
                available_metrics.update(entry["metrics"].keys())

        metrics_to_plot = list(plot_metrics) if plot_metrics else []
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

        # Plot each specified metric in a separate plot
        if metrics_history and metrics_to_plot:
            for metric_key in metrics_to_plot:
                metric_image = self._plot_single_metric(metrics_history, metric_key)
                if metric_image:
                    image_data.append({"image_base64": metric_image, "description": f"{metric_key}"})

        # Save plots to disk for reference
        self._save_plots_to_disk(image_data)

        return image_data

    def _plot_single_metric(self, metrics_history: List[Dict], metric_key: str) -> Optional[str]:
        """
        Plot a single metric across all training steps, showing ALL individual points.

        If a metric has multiple values per step (e.g., from different batches),
        all values are plotted at the same step to preserve spikes and variations.

        Args:
            metrics_history: List of metric entries with step and metrics data
            metric_key: Single metric key to plot (e.g., "actor/batch_loss")

        Returns:
            Base64 encoded PNG image string, or None if plotting fails
        """
        if not metrics_history or not metric_key:
            return None

        try:
            logger.debug(f"Plotting metric: {metric_key} with {len(metrics_history)} entries")

            # Create a single plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))

            # Extract all individual points (expand lists/arrays into separate points)
            # Skip entries missing this metric (e.g. validation metrics are sparse)
            all_steps = []
            all_values = []

            for entry in metrics_history:
                if metric_key not in entry.get("metrics", {}):
                    continue
                step = entry["step"]
                value = entry["metrics"][metric_key]

                # Expand lists/arrays into individual points at the same step
                if hasattr(value, "ndim") and value.ndim > 0:
                    # Numpy array with multiple values
                    for v in value.flatten():
                        all_steps.append(step)
                        all_values.append(float(v))
                elif isinstance(value, list):
                    # List of values
                    for v in value:
                        all_steps.append(step)
                        all_values.append(float(v))
                else:
                    # Single scalar value
                    all_steps.append(step)
                    all_values.append(float(value))

            if not all_steps:
                logger.warning(f"Metric '{metric_key}' not found in any step")
                return None

            steps_array = np.array(all_steps)
            values_array = np.array(all_values)

            # Generate metric display name
            metric_name = metric_key.split("/")[-1].replace("_", " ").title()

            # Plot all points
            ax.plot(steps_array, values_array, "b-", linewidth=1, alpha=0.7, label=metric_name)
            ax.set_xlabel("Training Step", fontsize=12)
            ax.set_ylabel(metric_name, fontsize=12)
            ax.set_title(f"{metric_key}", fontsize=14, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.legend()

            plt.tight_layout()

            # Save to BytesIO buffer and convert to base64
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode("utf-8")

            # Close the figure to free memory
            plt.close(fig)

            logger.debug(
                f"Successfully plotted {metric_key}: {len(all_values)} total points, range: {min(all_values):.4f} to {max(all_values):.4f}"
            )
            return image_base64

        except Exception as e:
            logger.error(f"Error plotting metric '{metric_key}': {e}", exc_info=True)
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
            # No images, return plain text
            return text_prompt

        # Build content blocks: images first, then text
        content_parts = []

        # Add images
        for img in image_data:
            content_parts.append(
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img["image_base64"]}}
            )

        # Add text prompt
        content_parts.append({"type": "text", "text": text_prompt})

        # Return as a dict that signals multimodal content
        return {"multimodal": True, "content": content_parts}

    def _save_plots_to_disk(self, image_data: List[Dict]):
        """Save plots to disk for reference."""
        if not image_data:
            return

        states_dir = os.path.join(self.manager.get_iteration_folder(self.manager.current_node), "states")
        os.makedirs(states_dir, exist_ok=True)

        for i, img_dict in enumerate(image_data):
            try:
                image_bytes = base64.b64decode(img_dict["image_base64"])
                metric_name = img_dict.get("description", f"plot_{i+1}")
                safe_filename = metric_name.replace("/", "_").replace("@", "_at_").replace(":", "_")
                filename = f"metric_{safe_filename}.png"
                filepath = os.path.join(states_dir, filename)

                with open(filepath, "wb") as f:
                    f.write(image_bytes)

                logger.info(f"Saved plot for '{metric_name}' to {filepath}")
            except Exception as e:
                logger.error(f"Error saving plot for metric {img_dict.get('description', i+1)} to disk: {e}")

    def parse(self, response: Union[Dict, str]) -> str:
        """Parse the LLM's response to extract the hyperparameter suggestions.

        Also detects checkpoint resume directives (RESUME_FROM_CHECKPOINT: N or TRAIN_FROM_SCRATCH).
        """
        suggestions = response.get("content", str(response)) if isinstance(response, dict) else str(response)

        # Detect checkpoint resume directive
        resume_match = re.search(r"RESUME_FROM_CHECKPOINT:\s*(\d+)", suggestions)
        if resume_match:
            self._checkpoint_step = int(resume_match.group(1))
            logger.info(f"Proposer selected RESUME_FROM_CHECKPOINT at step {self._checkpoint_step}")
        elif "TRAIN_FROM_SCRATCH" in suggestions:
            self._checkpoint_step = None
            logger.info("Proposer selected TRAIN_FROM_SCRATCH")
        else:
            self._checkpoint_step = None

        self.manager.save_and_log_states(
            content=suggestions, save_name="proposer_agent_response.txt", per_iteration=True, add_uuid=False
        )

        return suggestions

    @property
    def checkpoint_step(self) -> Optional[int]:
        """Get the checkpoint step selected by the proposer, or None for from-scratch."""
        return getattr(self, "_checkpoint_step", None)

    def _get_parent_data(self) -> Tuple[Optional[Dict], int]:
        """Extract step metrics from parent node."""
        parent_node = self.manager.current_node.parent if self.manager.current_node.parent else None

        if not parent_node or not parent_node.observer_results:
            return None, 0

        step_metrics_combined = {}

        # Get step metrics
        if "step_metrics" in parent_node.observer_results:
            step_metrics = parent_node.observer_results["step_metrics"]
            if "metrics_history" in step_metrics:
                step_metrics_combined["metrics_history"] = step_metrics["metrics_history"]
            training_steps = step_metrics.get("num_entries", 0)
        else:
            training_steps = 0

        # Get validation result if available
        if "validation" in parent_node.observer_results:
            step_metrics_combined["validation"] = parent_node.observer_results["validation"]

        return (step_metrics_combined if step_metrics_combined else None), training_steps

    def _get_step_metrics_summary(
        self,
        step_metrics: Optional[Dict],
        num_samples: int,
        use_average: bool,
        text_metrics: List[str],
    ) -> str:
        """Get step metrics summary for specified text metrics."""
        if not step_metrics:
            return "No step metrics available from previous run."

        return self._extract_important_step_metrics(
            step_metrics,
            num_samples=num_samples,
            use_average=use_average,
            text_metrics=text_metrics,
        )

    def _extract_important_step_metrics(
        self,
        step_metrics: Dict,
        num_samples: int = DEFAULT_NUM_SAMPLES,
        use_average: bool = DEFAULT_USE_AVERAGE,
        text_metrics: Optional[List[str]] = None,
    ) -> str:
        """Extract and summarize specified metrics as text."""
        has_metrics_history = self._has_metrics_history(step_metrics)
        has_validation = self._has_validation(step_metrics)

        if not has_metrics_history and not has_validation:
            return "No step metrics data available."

        summary_lines = []

        if has_metrics_history:
            summary_lines.extend(
                self._process_metrics_history(
                    step_metrics["metrics_history"],
                    num_samples,
                    use_average,
                    text_metrics if text_metrics else None,
                )
            )

        if has_validation:
            summary_lines.extend(self._process_validation(step_metrics["validation"]))

        return "\n".join(summary_lines) if summary_lines else "No key metrics available."

    def _process_metrics_history(
        self,
        metrics_history: List[Dict],
        num_samples: int,
        use_average: bool,
        metrics_list: Optional[List[str]] = None,
    ) -> List[str]:
        """Process step metrics history for text display."""
        if not metrics_history:
            return []

        # Use config metrics if provided, else fall back to framework defaults
        fw = self.training_framework
        default_keys = list(dict.fromkeys(fw.default_text_metrics() + fw.default_plot_metrics()))
        default_metrics = {k: k.replace("/", " / ").replace("_", " ").title() for k in default_keys}
        metrics_to_track = self._get_metrics_dict(metrics_list, default_metrics, metrics_history[0].get("metrics", {}))

        sample_indices = self._get_sample_indices(len(metrics_history), num_samples)
        step_intervals = self._calculate_step_intervals(metrics_history, sample_indices, num_samples, use_average)

        summary_lines = self._build_header(step_intervals, "Training Step Metrics")
        summary_lines.extend(
            self._extract_metric_progressions(
                metrics_history, metrics_to_track, sample_indices, num_samples, use_average
            )
        )

        return summary_lines

    def _process_validation(self, validation: Dict) -> List[str]:
        """Process validation result (best or final, per the run's reduction)."""
        summary_lines = ["\n**Validation (across training):**"]

        if "validation_score" in validation:
            score = validation["validation_score"]
            step = validation.get("validation_step")
            if score is not None:
                step_info = f" at step {step}" if step is not None else ""
                summary_lines.append(f"- Validation score: {score:.4f}{step_info}")
            else:
                summary_lines.append("- Validation score: N/A (training failed)")

        return summary_lines

    # Helper methods from original implementation
    @staticmethod
    def _get_metrics_dict(
        metric_keys: Optional[List[str]], default_metrics: Dict[str, str], available_metrics: Dict
    ) -> Dict[str, str]:
        """Convert a list of metric keys to a dict with display names."""
        if not metric_keys:
            return default_metrics

        metrics_dict = {}
        for key in metric_keys:
            if key in available_metrics:
                display_name = key.replace("/", " / ").replace("_", " ").title()
                metrics_dict[key] = display_name
            else:
                logger.warning(f"Metric '{key}' not found in available metrics, skipping")

        return metrics_dict if metrics_dict else default_metrics

    @staticmethod
    def _get_sample_indices(n_steps: int, num_samples: int) -> List[int]:
        """Get uniform sampling indices."""
        if n_steps <= num_samples:
            return list(range(n_steps))
        return np.linspace(0, n_steps - 1, num_samples, dtype=int).tolist()

    def _calculate_step_intervals(
        self, history: List[Dict], sample_indices: List[int], num_samples: int, use_average: bool
    ) -> List[str]:
        """Calculate step intervals for display."""
        step_intervals = []
        n_steps = len(history)

        if use_average:
            segment_size = max(1, n_steps // num_samples)
            for i in range(num_samples):
                start_idx = i * segment_size
                end_idx = min((i + 1) * segment_size, n_steps)
                if start_idx >= n_steps:
                    break

                start_step = history[start_idx]["step"]
                end_step = history[end_idx - 1]["step"]

                interval = f"step {start_step}" if start_step == end_step else f"steps {start_step}-{end_step}"
                step_intervals.append(interval)
        else:
            step_intervals = [f"step {history[idx]['step']}" for idx in sample_indices]

        return step_intervals

    def _build_header(self, step_intervals: List[str], header_text: str) -> List[str]:
        """Build the header for metric summaries."""
        interval_str = " | ".join(step_intervals)
        return [f"**{header_text} (averaged over {len(step_intervals)} intervals):**", f"Intervals: {interval_str}\n"]

    def _extract_metric_progressions(
        self,
        history: List[Dict],
        metrics_to_track: Dict[str, str],
        sample_indices: List[int],
        num_samples: int,
        use_average: bool,
    ) -> List[str]:
        """Extract metric progressions for display."""
        summary_lines = []

        for metric_key, metric_name in metrics_to_track.items():
            values, stds = self._calculate_metric_values(history, metric_key, sample_indices, num_samples, use_average)

            if values:
                value_str = self._format_metric_values(values, stds, use_average, metric_key)
                summary_lines.append(f"**{metric_name}:** {value_str}\n")

        return summary_lines

    def _calculate_metric_values(
        self, history: List[Dict], metric_key: str, sample_indices: List[int], num_samples: int, use_average: bool
    ) -> Tuple[List[float], List[float]]:
        """Calculate metric values and standard deviations."""
        values = []
        stds = []
        n_steps = len(history)

        if use_average:
            segment_size = max(1, n_steps // num_samples)
            for i in range(num_samples):
                start_idx = i * segment_size
                end_idx = min((i + 1) * segment_size, n_steps)
                if start_idx >= n_steps:
                    break

                segment_values = [
                    self._to_scalar(history[j]["metrics"][metric_key])
                    for j in range(start_idx, end_idx)
                    if metric_key in history[j]["metrics"]
                ]

                if segment_values:
                    values.append(np.mean(segment_values))
                    stds.append(np.std(segment_values) if len(segment_values) > 1 else 0.0)
        else:
            values = [
                self._to_scalar(history[idx]["metrics"][metric_key])
                for idx in sample_indices
                if metric_key in history[idx]["metrics"]
            ]

        return values, stds

    @staticmethod
    def _format_metric_values(values: List[float], stds: List[float], use_average: bool, metric_key: str) -> str:
        """Format metric values for display."""
        if use_average and stds:
            if metric_key == "response_length/mean":
                return " → ".join([f"{v:.1f}(±{s:.1f})" for v, s in zip(values, stds)])
            return " → ".join([f"{v:.4f}(±{s:.4f})" for v, s in zip(values, stds)])

        if metric_key == "response_length/mean":
            return " → ".join([f"{v:.1f}" for v in values])
        return " → ".join([f"{v:.4f}" for v in values])

    @staticmethod
    def _has_metrics_history(step_metrics: Dict) -> bool:
        """Check if metrics_history is available."""
        return step_metrics and "metrics_history" in step_metrics and step_metrics["metrics_history"]

    @staticmethod
    def _has_validation(step_metrics: Dict) -> bool:
        """Check if a validation result is available."""
        return step_metrics and "validation" in step_metrics and step_metrics["validation"]

    def _get_cold_start_notice(self) -> str:
        """Banner for cold-start proposals (no parent training history).

        Used by the async manager's bootstrap siblings, which all branch off the
        virtual root and therefore have no metrics to analyze. The goal is no longer
        to tune from observed dynamics but to seed a DIVERSE initial population: each
        sibling must pick a distinct, sensible starting configuration that differs
        from the baseline (node 0, package defaults) and from earlier siblings.
        """
        return (
            "### Cold-Start Proposal — No Training History Yet\n"
            "This is one of several initial runs launched in parallel; there are NO "
            "metrics from a previous run to analyze.\n\n"
            "Your job is to propose a potentially better setting that "
            "explores a different region of the hyperparameter space than the baseline and "
            "any sibling configurations listed below. Pick a coherent set of hyperparameters "
            "that is "
            "internally consistent and meaningfully different from what has already been tried. "
            "Skip the metric-diagnosis step (there is no data) and go straight to concrete "
            "hyperparameter choices with brief reasoning.\n"
        )

    def _get_early_stop_notice(self) -> str:
        """Get early stop notice if parent was early stopped."""
        parent_node = self.manager.current_node.parent if self.manager.current_node.parent else None

        if not parent_node:
            return ""

        if parent_node.early_stopped:
            return f"""### ⚠️ Previous Run Was Early Stopped
**Early Stop Reason:** {parent_node.early_stop_reason}

The previous training run was stopped early by the early stopper. This means the training dynamics or validation performance
were not meeting expectations compared to the best previous run. Your hyperparameter suggestions should address the issues
identified by the early stopper to improve the training trajectory.
"""
        return ""

    def _get_parent_error_context(self) -> str:
        """Get error analysis from parent or siblings to inform proposals."""
        if not self.manager.current_node or not self.manager.current_node.parent:
            return ""

        parent = self.manager.current_node.parent

        # If parent itself failed, show its error analysis
        if not parent.is_successful and parent.error_analysis:
            return f"""### Previous Run Failed
**Error Analysis:**
{parent.error_analysis}

Your proposals should address the issues identified above.
"""

        # Check if any siblings failed — show their errors for avoidance
        failed_siblings = [
            child
            for child in parent.children
            if not child.is_successful and child.error_analysis and child != self.manager.current_node
        ]

        if not failed_siblings:
            return ""

        lines = ["### Sibling Runs That Failed (avoid similar approaches):"]
        for sib in failed_siblings[:3]:
            lines.append(f"- Node {sib.id}: {sib.error_analysis[:300]}")
        return "\n".join(lines)

    def _get_checkpoint_info(self) -> str:
        """Get checkpoint info for checkpoint-based optimization.

        Returns instructions for the proposer to choose resume vs from-scratch,
        with a clear list of available checkpoint steps from the parent node.
        Includes exploration history to help the proposer balance strategies.
        When resuming, asks the proposer to specify ADDITIONAL_TRAINING_STEPS
        and compute total_epochs from that value and the chosen batch_size.
        """
        ckpt_config = getattr(self.manager.config, "checkpoint_optimization", None)
        if not ckpt_config or not getattr(ckpt_config, "enabled", False):
            return ""

        # If this node is forced from-scratch by the exploration mechanism,
        # don't show checkpoint info so the proposer naturally suggests bolder changes
        if self.manager.current_node and self.manager.current_node.forced_from_scratch:
            return ""

        parent = self.manager.current_node.parent if self.manager.current_node.parent else None
        if not parent or not parent.checkpoints:
            return ""

        steps = [str(ckpt["step"]) for ckpt in parent.checkpoints]
        step_list = ", ".join(steps)

        # Build exploration history context
        total = self.manager.from_scratch_count + self.manager.resume_count
        exploration_history = ""
        if total > 0:
            exploration_history = (
                f"\n**Exploration history:** {self.manager.resume_count} of {total} previous runs "
                f"resumed from checkpoints, {self.manager.from_scratch_count} trained from scratch.\n"
            )

        # Adaptive nudge when from-scratch ratio is low
        nudge = ""
        if total > 0:
            from_scratch_ratio = self.manager.from_scratch_count / total
            min_ratio = self.manager.min_from_scratch_ratio
            if from_scratch_ratio < min_ratio:
                nudge = (
                    "\n**Note:** The search has been heavily exploiting checkpoints. "
                    "Consider training from scratch to explore a fundamentally different "
                    "configuration (e.g., different batch size, learning rate schedule, or "
                    "optimizer settings) that could discover a better region of the "
                    "hyperparameter space.\n"
                )

        max_steps_per_node = getattr(ckpt_config, "max_steps_per_node", 100) or 100

        return (
            f"The previous run saved checkpoints at training steps: [{step_list}].\n\n"
            f"You have two strategies available. Both are valuable for optimization:\n"
            f"- `RESUME_FROM_CHECKPOINT: <step>` — Continue training from a checkpoint. "
            f"Best for incremental adjustments (learning rate tuning, regularization, schedule changes) "
            f"where the current trajectory is promising.\n"
            f"- `TRAIN_FROM_SCRATCH` — Start a fresh training run. "
            f"Best for exploring fundamentally different configurations (different batch size, "
            f"optimizer, architecture choices) that would conflict with the checkpoint's optimizer state.\n"
            f"{exploration_history}"
            f"{nudge}"
            f"If you choose to resume, you MUST also specify how many **additional** gradient "
            f"steps to train beyond the resume point:\n"
            f"  `ADDITIONAL_TRAINING_STEPS: <number>`\n"
            f"This MUST NOT exceed {max_steps_per_node} (the per-node step cap); larger requests are "
            f"clamped to {max_steps_per_node}. Every node — resumed or from-scratch — is capped at "
            f"{max_steps_per_node} new gradient steps, which keeps the search branching into more "
            f"configurations.\n"
        )

    def _get_checkpoint_output_format(self) -> str:
        """Get the response format section, adapting for checkpoint availability."""
        # If forced from-scratch, don't show checkpoint output format
        if self.manager.current_node and self.manager.current_node.forced_from_scratch:
            has_checkpoints = False
        else:
            ckpt_config = getattr(self.manager.config, "checkpoint_optimization", None)
            has_checkpoints = False
            if ckpt_config and getattr(ckpt_config, "enabled", False):
                parent = self.manager.current_node.parent if self.manager.current_node.parent else None
                if parent and parent.checkpoints:
                    has_checkpoints = True

        diagnosis = "**Step 1 — Diagnosis:**\n" "<2-3 sentences on training health, key trends, inflection points>\n"

        suggestions = (
            "**Step 2 — Suggestions** (do NOT include total_epochs here):\n"
            "- **learning_rate**: 5e-6 → 1e-5\n"
            "  *Reason: Loss decreasing slowly with stable gradients; can afford faster optimization*\n"
            "- **kl_coef**: 0.2 → 0.05\n"
            "  *Reason: KL divergence consistently low, indicating over-regularization*\n"
        )

        if has_checkpoints:
            checkpoint_decision = (
                "**Step 3 — Checkpoint decision:**\n"
                "`RESUME_FROM_CHECKPOINT: <step_number>` or `TRAIN_FROM_SCRATCH`\n"
                "`ADDITIONAL_TRAINING_STEPS: <number>` (if resuming)\n\n"
            )
            length = (
                "**Step 4 — Training length:**\n"
                "- If resuming: state `ADDITIONAL_TRAINING_STEPS` only (within the cap). The system "
                "inherits the checkpoint's global step and enforces the budget via "
                "`trainer.total_training_steps`; you do NOT compute `total_epochs`.\n"
                "- If training from scratch: the run is auto-capped at the per-node step limit; no "
                "length input is needed and you do NOT compute `total_epochs`.\n"
            )
            return diagnosis + "\n" + suggestions + "\n" + checkpoint_decision + length
        else:
            epoch_computation = (
                "**Step 3 — Epoch computation:**\n"
                "Based on the batch size from Step 2, compute and state `total_epochs` with the math shown.\n"
            )
            return diagnosis + "\n" + suggestions + "\n" + epoch_computation

    def _get_previous_training_config(self) -> str:
        """Show the parent node's training configuration to the proposer.

        The proposer must tune RELATIVE to the parent it is deepening from — its
        suggestions ("lr 1e-4 -> 6e-5") are meaningless without the parent's actual
        current values. We surface the parent's config two ways, best-first:

        1. The parent's ``entry_point_script`` — the literal training command it ran,
           carrying the ground-truth resolved hyperparameters. Always present on the
           node object and independent of checkpoint_optimization, so this works even
           when ``checkpoint_optimization`` is disabled (no parent.checkpoints).
        2. Fallback: the resolved config YAML from the parent's run dir (requires a
           checkpoint path to locate the run dir; only available with checkpointing).

        Without this the proposer tunes blind against the package-default reference,
        producing near-identical siblings (guessed baseline + relative edits off it).
        """
        parent = self.manager.current_node.parent if self.manager.current_node.parent else None
        if not parent:
            return ""

        # 1) Parent's actual training command (ground-truth HPs) — always available.
        parent_script = getattr(parent, "entry_point_script", "") or getattr(parent, "execution_script", "")
        if parent_script:
            return (
                "### Parent Training Configuration (the run you are tuning FROM)\n"
                "This is the parent node's training command. Its hyperparameter values are the "
                "CURRENT baseline — propose changes RELATIVE to these exact values (do NOT assume "
                "package defaults).\n"
                f"```bash\n{parent_script}\n```"
            )

        # 2) Fallback: resolved config YAML (needs a checkpoint path to find the run dir).
        if not parent.checkpoints:
            return ""

        first_ckpt_path = parent.checkpoints[0].get("path", "")
        if not first_ckpt_path:
            return ""

        run_dir = os.path.dirname(first_ckpt_path)
        config_path = os.path.join(run_dir, self.training_framework.training_config_filename())

        if not os.path.isfile(config_path):
            logger.warning(f"Training config not found at {config_path}")
            return ""

        try:
            import yaml

            with open(config_path, "r") as f:
                full_cfg = yaml.safe_load(f)

            filtered = self.training_framework.filter_training_config(full_cfg)

            config_content = yaml.dump(filtered, default_flow_style=False, sort_keys=False)
            return (
                "### Previous Training Config (resolved YAML)\n"
                "Relevant training parameters from the previous run.\n"
                f"```yaml\n{config_content}```"
            )
        except Exception as e:
            logger.error(f"Error reading training config from {config_path}: {e}")
            return ""

    def _get_suggestion_count_guidance(self) -> str:
        """Return guidance on how many suggestions to make.

        When resuming from a checkpoint, allow more suggestions since the
        proposer has richer context. Otherwise limit to 2-4.
        """
        # If forced from-scratch, use the from-scratch guidance
        if self.manager.current_node and self.manager.current_node.forced_from_scratch:
            has_checkpoints = False
        else:
            ckpt_config = getattr(self.manager.config, "checkpoint_optimization", None)
            has_checkpoints = False
            if ckpt_config and getattr(ckpt_config, "enabled", False):
                parent = self.manager.current_node.parent if self.manager.current_node.parent else None
                if parent and parent.checkpoints:
                    has_checkpoints = True

        if has_checkpoints:
            return (
                "Suggest hyperparameter changes for the next run. "
                "Since you are resuming from a checkpoint and have full training context, "
                "you may suggest as many changes as needed (typically 3-8). "
                "For each suggestion, explain clearly why the change is expected to help. "
                "Be specific: `parameter_name: old_value → new_value`."
            )
        else:
            return (
                "Suggest 2-4 diverse hyperparameter changes for the next run. "
                "For each, explain clearly why the change is expected to help. "
                "Be specific: `parameter_name: old_value → new_value`."
            )
