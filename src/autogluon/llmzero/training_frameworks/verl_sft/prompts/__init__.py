# src/autogluon/llmzero/training_frameworks/verl_sft/prompts/__init__.py
"""Dispatch verl SFT framework prompt fragments by agent name."""

from . import early_stopper, proposer

_FRAGMENTS = {"proposer": proposer.fragments, "early_stopper": early_stopper.fragments}


def fragments_for(agent_name: str) -> dict:
    fn = _FRAGMENTS.get(agent_name)
    return fn() if fn else {}
