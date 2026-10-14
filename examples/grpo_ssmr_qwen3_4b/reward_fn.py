"""
Reward function for SSMR-Bench multiple-choice questions.

Extracts the model's answer letter (A/B/C/D) from \\boxed{X} format,
compares against ground truth letter. Returns 1.0 for correct, 0.0 otherwise.
"""
import json
import re


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    Score model output against ground truth for SSMR-Bench MCQ.

    Args:
        data_source: e.g., "ssmr_beat", "ssmr_beat__dev", "ssmr_agg__dev"
        solution_str: Model's generated response
        ground_truth: Correct letter (A/B/C/D) or JSON list of valid letters
        extra_info: Optional metadata (unused)

    Returns:
        float: 1.0 if correct, 0.0 otherwise
    """
    try:
        # Strip thinking tags
        cleaned = re.sub(r"<think>.*?</think>", "", solution_str, flags=re.DOTALL)
        cleaned = cleaned.strip()

        # Extract answer from \boxed{X}
        boxed_matches = re.findall(r"\\boxed\{([^}]*)\}", cleaned)
        if boxed_matches:
            predicted = boxed_matches[-1].strip().upper()
        else:
            # Fallback: look for standalone letter pattern at end
            # Match lines like "A", "Answer: B", "The answer is C"
            letter_match = re.search(
                r"(?:answer\s*(?:is|:)\s*)?([A-D])\s*$",
                cleaned,
                re.IGNORECASE | re.MULTILINE,
            )
            if letter_match:
                predicted = letter_match.group(1).upper()
            else:
                return 0.0

        # Normalize predicted to single letter
        if predicted not in ("A", "B", "C", "D"):
            return 0.0

        # Parse ground truth (single string or JSON list)
        valid_answers = set()
        try:
            parsed = json.loads(ground_truth)
            if isinstance(parsed, list):
                valid_answers = {str(a).strip().upper() for a in parsed}
            else:
                valid_answers = {str(ground_truth).strip().upper()}
        except (json.JSONDecodeError, TypeError):
            valid_answers = {str(ground_truth).strip().upper()}

        return 1.0 if predicted in valid_answers else 0.0

    except Exception:
        return 0.0
