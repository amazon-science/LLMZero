"""VERL GRPO checkpoint resume instructions and checkpoint validation."""

import glob as _glob
import os


def checkpoint_resume_info(node, ckpt_config) -> str:
    if not (ckpt_config and getattr(ckpt_config, "enabled", False)):
        return ""

    # verl 0.9.0 checkpoint format for resumable LoRA (and full-finetune) warm-start.
    # save_contents=["model","optimizer","extra"] writes per-rank FSDP-sharded files
    # (model_world_size_*_rank_*.pt). With +save_lora_only=True AND a LoRA model, verl
    # saves ONLY the adapter tensors (~tens of MiB/rank) instead of a fused full model;
    # save_lora_only is a no-op for full-finetune (it then saves full sharded weights).
    # On resume, verl auto-detects a LoRA-only state dict and loads it strict=False.
    # This REPLACES the old save_contents=["hf_model"] path, which wrote a FUSED full
    # model with no separable adapter and made LoRA resume a silent no-op (the child
    # re-initialized a fresh adapter). Verified end-to-end on verl 0.9.0: LoRA-only save
    # (66 MiB/rank, no fused model) + "Loaded LoRA-only checkpoint (504 keys)" on resume.
    ckpt_overrides = (
        "\n\nIMPORTANT — add these checkpoint overrides to the VERL command so future runs "
        "can resume from this checkpoint (verl 0.9.0 saves a compact LoRA-only sharded "
        "checkpoint when the model has LoRA adapters, or full sharded weights otherwise):\n"
        '  actor_rollout_ref.actor.checkpoint.save_contents=["model","optimizer","extra"]\n'
        "  +actor_rollout_ref.actor.checkpoint.save_lora_only=True\n"
    )

    # The run length of EVERY node is bounded by an explicit step cap rather than total_epochs:
    # verl stops when global_steps >= trainer.total_training_steps (ray_trainer.py). We compute
    # the cap in code so it does not depend on the LLM's epoch arithmetic. trainer.total_epochs
    # is set to (total_training_steps + 1) so the epoch loop is guaranteed to outlast the step
    # cap even when there is only one gradient step per epoch (tiny datasets / large batch).
    max_steps_per_node = getattr(ckpt_config, "max_steps_per_node", 100) or 100
    node_budget = getattr(node, "training_steps_budget", None) if node else None
    step_budget = node_budget or max_steps_per_node

    # From-scratch nodes (including the root and TRAIN_FROM_SCRATCH evolve nodes): cap the run
    # at exactly step_budget new gradient steps starting from step 0.
    if not node or not node.resume_from_checkpoint:
        total_training_steps = step_budget
        total_epochs = total_training_steps + 1
        return (
            f"No checkpoint resumption. Train from scratch.\n\n"
            f"Bound the run length with an explicit step cap (do NOT rely on epoch count):\n"
            f"  trainer.total_training_steps={total_training_steps}\n"
            f"  trainer.total_epochs={total_epochs}\n"
            f"This trains exactly {total_training_steps} gradient steps. The run is bounded by "
            f"trainer.total_training_steps, NOT total_epochs — total_epochs is set large enough "
            f"({total_epochs}) that the epoch loop never ends before the step cap is reached. "
            f"Do NOT lower total_training_steps or total_epochs below these values." + ckpt_overrides
        )

    # Resumed nodes: inherit the parent's global step so the run continues from
    # resume_checkpoint_step instead of restarting the counter at 0. trainer.resume_mode=resume_path
    # makes verl read the step from the "global_step_<N>" folder name. We resume with
    # load_contents=["model"] (weights/adapter ONLY, NOT optimizer/extra): this warm-starts the
    # trained LoRA adapter (verl auto-detects the LoRA-only state dict and loads it strict=False)
    # while using a FRESH optimizer + LR schedule, so proposer HP changes (learning rate, warmup,
    # etc.) actually take effect on the resumed run instead of being overwritten by the parent's
    # optimizer state.
    additional_steps = step_budget
    resume_step = node.resume_checkpoint_step or 0
    total_training_steps = resume_step + additional_steps
    total_epochs = total_training_steps + 1

    step_cap_instructions = (
        f"  trainer.resume_mode=resume_path\n"
        f"  trainer.resume_from_path={node.resume_checkpoint_path}\n"
        f"  trainer.total_training_steps={total_training_steps}\n"
        f"  trainer.total_epochs={total_epochs}\n"
        f"This resumes at global step {resume_step} and trains {additional_steps} additional "
        f"gradient steps (stopping at global step {total_training_steps}). The run is bounded by "
        f"trainer.total_training_steps, NOT total_epochs — total_epochs is set large enough "
        f"({total_epochs}) that the epoch loop never ends before the step cap is reached. Do NOT "
        f"lower total_training_steps or total_epochs below these values."
    )

    # verl 0.9.0 resume: load the parent's FSDP-sharded weights (LoRA adapter, or full
    # model for full-finetune) with load_contents=["model"] only. verl auto-detects a
    # LoRA-only state dict from the sharded files and loads it strict=False onto the freshly
    # built adapter — there is NO separate lora_adapter/ directory and NO lora_adapter_path
    # override (that key is for loading an externally-provided PEFT adapter, never what
    # training writes). resume_from_path supplies both the weights and the global step.
    resume_load_override = '  actor_rollout_ref.actor.checkpoint.load_contents=["model"]\n'
    return (
        f"RESUME FROM CHECKPOINT\n"
        f"Path: {node.resume_checkpoint_path}\n"
        f"Resume from step: {resume_step} (+{additional_steps} additional steps)\n\n"
        f"Before the training command, remove the dataloader state file to avoid\n"
        f"num_workers mismatch when hyperparameters change across runs:\n"
        f"  rm -f {node.resume_checkpoint_path}/data.pt\n\n"
        f"Add to the VERL training command:\n"
        f"{resume_load_override}"
        f"{step_cap_instructions}"
        + ckpt_overrides
        + "\nKeep trainer.default_local_dir as the new per-run directory (new checkpoints save there).\n"
        "Only modify hyperparameters suggested by the proposer. Keep all else identical."
    )


def validate_checkpoint(checkpoint: dict) -> bool:
    """Return True if checkpoint directory contains valid model weight files.

    Checks for:
    1. FSDP sharded files (model_world_size_*_rank_0.pt)
    2. HuggingFace full-model files (model*.safetensors / pytorch_model*.bin)
    3. LoRA adapter files (adapter_model.safetensors / adapter_model*.bin)
    inside the actor sub-directory.
    """
    path = checkpoint.get("path", "")
    actor_path = os.path.join(path, "actor")
    if not os.path.isdir(actor_path):
        return False

    if _glob.glob(os.path.join(actor_path, "model_world_size_*_rank_0.pt")):
        return True

    hf_path = os.path.join(actor_path, "huggingface")
    if os.path.isdir(hf_path):
        if _glob.glob(os.path.join(hf_path, "model*.safetensors")) or _glob.glob(
            os.path.join(hf_path, "pytorch_model*.bin")
        ):
            return True

    lora_path = os.path.join(actor_path, "lora_adapter")
    if os.path.isdir(lora_path):
        if _glob.glob(os.path.join(lora_path, "adapter_model.safetensors")) or _glob.glob(
            os.path.join(lora_path, "adapter_model*.bin")
        ):
            return True

    return False
