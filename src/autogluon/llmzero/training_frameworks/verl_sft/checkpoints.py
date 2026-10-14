# src/autogluon/llmzero/training_frameworks/verl_sft/checkpoints.py
"""verl 0.9.0 SFT checkpoint validation + resume instructions (optimizer-state).

verl SFT saves checkpoints FLAT under ``{default_local_dir}/global_step_N/``
(no ``actor/`` subdir, unlike GRPO). A resumable SFT checkpoint carries model
shards AND optimizer shards AND extra-state (LR scheduler + RNG):
    model_world_size_{WS}_rank_{R}.pt
    optim_world_size_{WS}_rank_{R}.pt
    extra_state_world_size_{WS}_rank_{R}.pt

Continuous training resumes the FULL training state (optimizer momentum +
second-moment + LR schedule + RNG) so the optimization trajectory continues
correctly. This is the key difference from verl_grpo, which forces
``load_contents=["hf_model"]`` for a weights-only warm start (fresh optimizer).
"""

import glob
import os

# The three-part checkpoint contents that make an SFT checkpoint fully resumable.
_RESUME_CONTENTS = '["model","optimizer","extra"]'


def _resume_block(path: str) -> str:
    """RESUME-FROM-CHECKPOINT override block for a verl SFT checkpoint dir."""
    return (
        "RESUME FROM CHECKPOINT (continuous training — restores model + optimizer + LR/RNG state)\n"
        f"Checkpoint dir: {path}\n"
        "Add these Hydra overrides to the verl SFT training command:\n"
        "  trainer.resume_mode=resume_path\n"
        f"  trainer.resume_from_path={path}\n"
        f"  checkpoint.save_contents={_RESUME_CONTENTS}\n"
        f"  checkpoint.load_contents={_RESUME_CONTENTS}\n"
        "NOTES:\n"
        "  - verl inherits the global step from the 'global_step_N' folder name; set\n"
        "    trainer.total_training_steps GREATER than N so the run makes progress.\n"
        "  - Keep model.path, model.lora_rank, and model.lora_alpha IDENTICAL to the\n"
        "    checkpoint, or optimizer/model state fails to load.\n"
        "  - If you changed data.train_batch_size / data.num_workers, delete the stale\n"
        f"    dataloader state first:  rm -f {path}/data_*.pt"
    )


def checkpoint_resume_info(node, ckpt_config) -> str:
    if not (ckpt_config and getattr(ckpt_config, "enabled", False)):
        return ""

    # Parent-chained node: resume from the parent's best checkpoint.
    if node and getattr(node, "resume_from_checkpoint", False) and getattr(node, "resume_checkpoint_path", None):
        return _resume_block(node.resume_checkpoint_path)

    # Root/bootstrap or forced-from-scratch node. Warm-start from a configured
    # seed checkpoint if provided (continuous-training seed), else start from the
    # base model weights with a fresh optimizer.
    base_checkpoint_path = getattr(ckpt_config, "base_checkpoint_path", None)
    if base_checkpoint_path:
        return _resume_block(base_checkpoint_path)

    return "No checkpoint resumption. Train from the base model weights (fresh optimizer)."


def validate_checkpoint(checkpoint: dict) -> bool:
    """True only if the dir holds BOTH model shards AND optimizer shards.

    A resumable continuous-training SFT checkpoint MUST carry optimizer state;
    a weights-only directory is rejected so the engine does not silently resume
    without optimizer momentum.
    """
    path = checkpoint.get("path", "")
    if not path or not os.path.isdir(path):
        return False
    has_model = bool(glob.glob(os.path.join(path, "model_world_size_*_rank_*.pt")))
    has_optim = bool(glob.glob(os.path.join(path, "optim_world_size_*_rank_*.pt")))
    return has_model and has_optim
