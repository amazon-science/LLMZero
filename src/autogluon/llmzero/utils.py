def extract_hp_decisions(text: str, max_chars: int = 600) -> str:
    """Extract concrete hyperparameter decisions from a proposer response.

    Keeps only lines with an arrow (param: old → new) or checkpoint/step
    directives. Falls back to a whitespace-stripped slice if no structured
    decisions are found.
    """
    if not text:
        return ""

    decision_lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if (
            "→" in line
            or "->" in line
            or "RESUME_FROM_CHECKPOINT" in line
            or "ADDITIONAL_TRAINING_STEPS" in line
            or "TRAIN_FROM_SCRATCH" in line
        ):
            cleaned = line.lstrip("-* ").replace("**", "").strip()
            if cleaned:
                decision_lines.append(cleaned)

    summary = "; ".join(decision_lines)
    if not summary:
        summary = text.strip()
    return summary[:max_chars]
