"""Reward function for GovReport-SFT: SCROLLS gov_report default metric.

Replicates the SCROLLS `gov_report` per-example score:
    scrolls_score = rouge/geometric_mean = (rouge1 * rouge2 * rougeL) ** (1/3)
where each ROUGE is the F-measure * 100 (0-100 scale), use_stemmer=False, with
prediction and reference sentence-split (nltk, newline-joined) before scoring
(matching SCROLLS' postprocess_text). Display keys are rouge1/rouge2/rougeL
(rougeLsum is computed by SCROLLS but excluded from the geometric mean).

Note on aggregation: this returns the geometric mean PER SAMPLE (correct for a
per-rollout reward). The official SCROLLS leaderboard number is the geometric
mean of the corpus-averaged ROUGEs — a different order of operations — so the
mean of these per-sample rewards will not exactly equal the leaderboard score.
"""
import json
import re


def _ensure_deps():
    """Auto-install rouge_score + nltk and ensure punkt data (workers may lack them).

    Raises on install failure (loud, so an all-zero reward can't be silently
    mistaken for a bad model). Deps are normally pre-provided via
    reward/install_commands.sh, so this install path is a rarely-hit safety net.
    """
    try:
        from rouge_score import rouge_scorer  # noqa: F401
        import nltk  # noqa: F401
    except ImportError:
        import subprocess
        import sys

        print("[govreport-sft reward] rouge_score/nltk missing — installing", file=sys.stderr)
        try:
            subprocess.check_call(["pip", "install", "-q", "rouge_score", "nltk"])
        except Exception as e:
            print(f"[govreport-sft reward] FATAL: dependency install failed: {e}", file=sys.stderr)
            raise
    import nltk

    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass


def _strip_think(text):
    """Remove <think>...</think> blocks from model output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _postprocess(text):
    """SCROLLS rouge postprocess: newline after each sentence."""
    import nltk

    try:
        sents = nltk.sent_tokenize(text)
    except Exception:
        sents = [s for s in text.split(". ") if s]
    return "\n".join(sents)


def _rouge_geometric_mean(prediction, reference):
    """SCROLLS scrolls_score for one (pred, ref) pair on the 0-100 scale."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    # SCROLLS calls scorer.score(reference, prediction)
    scores = scorer.score(_postprocess(reference), _postprocess(prediction))
    r1 = scores["rouge1"].fmeasure * 100
    r2 = scores["rouge2"].fmeasure * 100
    rl = scores["rougeL"].fmeasure * 100
    # Geometric mean intentionally collapses to 0 if any single ROUGE is 0
    # (faithful to SCROLLS — do NOT "fix" this into an arithmetic mean).
    return (r1 * r2 * rl) ** (1.0 / 3.0)


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Score a summary against gold via SCROLLS gov_report scrolls_score (0-100).

    Args:
        data_source: identifier (matches "govreport_sft" / "govreport_sft__dev")
        solution_str: model's generated summary (may contain <think> tags)
        ground_truth: gold summary string, or JSON list of reference summaries
        extra_info: optional metadata (ignored)
        **kwargs: ignored

    Returns:
        float: scrolls_score (rouge geometric mean, 0-100); 0.0 on any failure.
    """
    if solution_str is None or ground_truth is None:
        return 0.0
    # Dependency check is OUTSIDE the broad try: an install failure should raise
    # loudly, not be swallowed as a 0.0 reward (which looks like a bad model).
    _ensure_deps()
    try:
        prediction = _strip_think(solution_str)
        if not prediction.strip():
            return 0.0

        # ground_truth: single string, or JSON list of references (take max per SCROLLS agg_fn=max)
        if isinstance(ground_truth, str):
            try:
                gt_list = json.loads(ground_truth)
                if not isinstance(gt_list, list):
                    gt_list = [ground_truth]
            except (json.JSONDecodeError, TypeError):
                gt_list = [ground_truth]
        elif isinstance(ground_truth, list):
            gt_list = ground_truth
        else:
            gt_list = [str(ground_truth)]

        best = 0.0
        for gt in gt_list:
            if gt is None or not str(gt).strip():
                continue
            score = _rouge_geometric_mean(prediction, str(gt))
            if score > best:
                best = score
        return float(best)
    except Exception:
        return 0.0
