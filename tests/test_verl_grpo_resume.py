"""Tests for VerlGrpoFramework.checkpoint_resume_info step-cap + step-inheritance logic.

The per-node step cap applies to ALL nodes: from-scratch nodes are bounded at exactly the
budget, resumed nodes inherit the parent global step and train `budget` additional steps.
In both cases trainer.total_epochs is set large enough (total_training_steps + 1) to guarantee
the step cap is reached before the epoch loop ends.
"""

from types import SimpleNamespace

from autogluon.llmzero.training_frameworks.verl_grpo import VerlGrpoFramework
from autogluon.llmzero.training_frameworks.verl_grpo.parser import VerlStdoutParser


def test_parse_checkpoint_verl090_stderr_format():
    # verl 0.9.0 announces the actor checkpoint on STDERR as "Saving checkpoint to <path>/global_step_N".
    # The path observer must extract the step (regression: prior regex only matched the 0.7.x
    # "local_global_step_folder:" form, so checkpoints=[] and every node was forced from-scratch).
    p = VerlStdoutParser()
    ck = p.parse_checkpoint("Saving checkpoint to /fsx/run/verl_ckpts/llmzero_N1/global_step_40")
    assert ck == {"step": 40, "path": "/fsx/run/verl_ckpts/llmzero_N1/global_step_40"}
    # with an INFO prefix and an /actor suffix (also a real 0.9.0 shape)
    ck2 = p.parse_checkpoint("INFO:2026-09-10:Saving checkpoint to /fsx/r/global_step_60/actor")
    assert ck2 == {"step": 60, "path": "/fsx/r/global_step_60/actor"}
    # old 0.7.x stdout form still parses
    ck3 = p.parse_checkpoint("local_global_step_folder: /fsx/r/global_step_80")
    assert ck3 == {"step": 80, "path": "/fsx/r/global_step_80"}
    # a metric line is not a checkpoint
    assert p.parse_checkpoint("step:1 - global_seqlen/min:60557.0 - actor/entropy:0.4") is None


def test_log_stream_is_both():
    # verl 0.9.0 puts metrics on stdout but the checkpoint-save line on stderr; both are needed.
    assert VerlGrpoFramework.log_stream() == "both"


def _ckpt_config(max_steps_per_node=100, enabled=True):
    return SimpleNamespace(enabled=enabled, max_steps_per_node=max_steps_per_node)


def _resume_node(step=39, budget=40, path="/tmp/run/global_step_39"):
    return SimpleNamespace(
        resume_from_checkpoint=True,
        resume_checkpoint_path=path,
        resume_checkpoint_step=step,
        training_steps_budget=budget,
    )


def _scratch_node(budget=100):
    return SimpleNamespace(resume_from_checkpoint=False, training_steps_budget=budget)


def test_disabled_returns_empty():
    assert VerlGrpoFramework.checkpoint_resume_info(_resume_node(), _ckpt_config(enabled=False)) == ""


def test_from_scratch_node_is_step_capped():
    node = _scratch_node(budget=80)
    info = VerlGrpoFramework.checkpoint_resume_info(node, _ckpt_config())
    assert "Train from scratch" in info
    assert "resume_mode=resume_path" not in info
    # Cap applies to from-scratch too: total_training_steps == budget, epochs large enough.
    assert "trainer.total_training_steps=80" in info
    assert "trainer.total_epochs=81" in info


def test_from_scratch_with_no_node_uses_cap_default():
    info = VerlGrpoFramework.checkpoint_resume_info(None, _ckpt_config(max_steps_per_node=100))
    assert "trainer.total_training_steps=100" in info
    assert "trainer.total_epochs=101" in info


def test_resume_emits_resume_path_and_inherits_step():
    info = VerlGrpoFramework.checkpoint_resume_info(_resume_node(step=39, budget=40), _ckpt_config())
    # Inherits global step via resume_path (fixes the "counter restarts at 0" bug).
    assert "trainer.resume_mode=resume_path" in info
    assert "trainer.resume_from_path=/tmp/run/global_step_39" in info
    # Bounds the run by an absolute step cap = resume_step + budget.
    assert "trainer.total_training_steps=79" in info
    # total_epochs is large enough (cap + 1) so the step cap governs, not the epoch loop.
    assert "trainer.total_epochs=80" in info


def test_resume_uses_native_lora_sharded_load_not_adapter_path():
    # verl 0.9.0 warm-start loads the FSDP-sharded LoRA-only checkpoint via load_contents=["model"]
    # (weights/adapter only, fresh optimizer). It must NOT emit the old dead lora_adapter_path
    # override (that dir is never written by training -- see the broken-resume bug this fix addresses).
    info = VerlGrpoFramework.checkpoint_resume_info(_resume_node(step=20, budget=100), _ckpt_config())
    assert "lora_adapter_path" not in info
    assert 'actor_rollout_ref.actor.checkpoint.load_contents=["model"]' in info
    assert "trainer.resume_mode=resume_path" in info
    assert "trainer.total_training_steps=120" in info  # 20 + 100
    assert "trainer.total_epochs=121" in info


def test_save_overrides_use_lora_only_not_hf_model():
    # Every node's checkpoint-save instruction must use the compact LoRA-only sharded format,
    # NOT the old fused-full-model save_contents=["hf_model"] that made LoRA resume a no-op.
    scratch = VerlGrpoFramework.checkpoint_resume_info(_scratch_node(budget=50), _ckpt_config())
    resume = VerlGrpoFramework.checkpoint_resume_info(_resume_node(step=20, budget=100), _ckpt_config())
    for info in (scratch, resume):
        assert 'save_contents=["model","optimizer","extra"]' in info
        assert "save_lora_only=True" in info
        assert '["hf_model"]' not in info


def test_resume_falls_back_to_cap_when_budget_missing():
    node = _resume_node(step=10, budget=None)
    info = VerlGrpoFramework.checkpoint_resume_info(node, _ckpt_config(max_steps_per_node=64))
    # budget defaults to max_steps_per_node -> total = 10 + 64.
    assert "trainer.total_training_steps=74" in info
    assert "trainer.total_epochs=75" in info
