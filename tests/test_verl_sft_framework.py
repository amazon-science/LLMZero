# tests/test_verl_sft_framework.py
from autogluon.llmzero.training_frameworks import get_framework
from autogluon.llmzero.training_frameworks.verl_sft import checkpoints as sft_ck
from autogluon.llmzero.training_frameworks.verl_sft.parser import VerlSftStdoutParser


def test_verl_sft_registered_and_parser():
    fw = get_framework("verl_sft")
    parser = fw.stdout_parser()
    assert parser.__class__.__name__ == "VerlSftStdoutParser"
    assert fw.score_direction() == "minimize"
    assert fw.validation_metric_pattern() == r"val/loss"
    assert fw.log_stream() == "stdout"


def test_parse_train_step_line():
    p = VerlSftStdoutParser()
    line = "step:5 - train/loss:0.8421 - train/grad_norm:1.234 - train/lr:1e-05 - train/mfu:0.31 - train/global_tokens:12345"
    step, metrics = p.parse_step_metrics(line)
    assert step == 5
    assert metrics["train/loss"] == 0.8421
    assert metrics["train/grad_norm"] == 1.234
    assert metrics["train/lr"] == 1e-05


def test_parse_val_step_line():
    p = VerlSftStdoutParser()
    step, metrics = p.parse_step_metrics("step:5 - val/loss:0.7563")
    assert step == 5
    assert metrics == {"val/loss": 0.7563}


def test_parse_np_float64_wrapper():
    p = VerlSftStdoutParser()
    step, metrics = p.parse_step_metrics("step:7 - val/loss:np.float64(0.6421)")
    assert step == 7
    assert metrics["val/loss"] == 0.6421


def test_non_metric_line_returns_none():
    p = VerlSftStdoutParser()
    assert p.parse_step_metrics("Some random log line without step") is None
    # Prose after "step:N - " with no namespaced key:value token must NOT match:
    assert p.parse_step_metrics("step:5 - starting validation now") is None


def test_parse_perf_leading_train_line():
    # verl 0.9.0 SFT train-step lines LEAD with perf/* (not train/); the parser must
    # still anchor and capture the full metric set (regression for the 0.9.0 grammar).
    p = VerlSftStdoutParser()
    line = (
        "step:3 - perf/max_memory_allocated_gb:7.49 - train/loss:1.194 "
        "- train/grad_norm:0.49 - train/lr:0.0001 - train/mfu:0.28"
    )
    step, metrics = p.parse_step_metrics(line)
    assert step == 3
    assert metrics["train/loss"] == 1.194
    assert metrics["perf/max_memory_allocated_gb"] == 7.49


def test_parse_checkpoint_line():
    p = VerlSftStdoutParser()
    ck = p.parse_checkpoint("Saving checkpoint to: /fsx/run/verl_ckpts/global_step_120")
    assert ck == {"step": 120, "path": "/fsx/run/verl_ckpts/global_step_120"}


def test_parse_progress_line():
    p = VerlSftStdoutParser()
    it, spi = p.parse_progress("Epoch 1/1:   5%|##        | 12/240 [00:30<10:00, 2.50s/it]")
    assert it == 12
    assert spi == 2.50


def _make_ckpt(tmp_path, *, model=True, optim=True):
    d = tmp_path / "global_step_100"
    d.mkdir(parents=True)
    if model:
        (d / "model_world_size_8_rank_0.pt").write_text("x")
    if optim:
        (d / "optim_world_size_8_rank_0.pt").write_text("x")
    (d / "extra_state_world_size_8_rank_0.pt").write_text("x")
    return {"step": 100, "path": str(d)}


def test_validate_checkpoint_requires_optimizer_state(tmp_path):
    good = _make_ckpt(tmp_path / "a", model=True, optim=True)
    assert sft_ck.validate_checkpoint(good) is True


def test_validate_checkpoint_rejects_weights_only(tmp_path):
    bad = _make_ckpt(tmp_path / "b", model=True, optim=False)
    assert sft_ck.validate_checkpoint(bad) is False


def test_validate_checkpoint_rejects_missing_path():
    assert sft_ck.validate_checkpoint({"path": ""}) is False


class _NodeStub:
    def __init__(self, resume, path=None, step=None):
        self.resume_from_checkpoint = resume
        self.resume_checkpoint_path = path
        self.resume_checkpoint_step = step


class _CkCfg:
    def __init__(self, enabled, base=None):
        self.enabled = enabled
        self.base_checkpoint_path = base


def test_resume_info_disabled_returns_empty():
    assert sft_ck.checkpoint_resume_info(_NodeStub(False), _CkCfg(False)) == ""


def test_resume_info_parent_chain_emits_optimizer_flags():
    node = _NodeStub(True, path="/fsx/run/verl_ckpts/global_step_100", step=100)
    txt = sft_ck.checkpoint_resume_info(node, _CkCfg(True))
    assert "trainer.resume_mode=resume_path" in txt
    assert "trainer.resume_from_path=/fsx/run/verl_ckpts/global_step_100" in txt
    assert 'checkpoint.load_contents=["model","optimizer","extra"]' in txt


def test_resume_info_from_scratch_when_no_parent_no_seed():
    txt = sft_ck.checkpoint_resume_info(_NodeStub(False), _CkCfg(True))
    assert "resume_from_path" not in txt


def test_resume_info_seed_when_no_parent_but_base_set():
    cfg = _CkCfg(True, base="/fsx/seed/global_step_500")
    txt = sft_ck.checkpoint_resume_info(_NodeStub(False), cfg)
    assert "trainer.resume_from_path=/fsx/seed/global_step_500" in txt
    assert 'checkpoint.load_contents=["model","optimizer","extra"]' in txt


def test_resume_block_handles_brace_in_path():
    node = _NodeStub(True, path="/fsx/run/{weird}/global_step_100", step=100)
    txt = sft_ck.checkpoint_resume_info(node, _CkCfg(True))
    assert "/fsx/run/{weird}/global_step_100" in txt


def test_framework_guides_present():
    fw = get_framework("verl_sft")
    guide = fw.format_hyperparameter_guide()
    assert "optim.lr" in guide
    assert "model.lora_rank" in guide
    md = fw.metric_descriptions()
    assert "val/loss" in md
    assert "val/loss" in fw.default_text_metrics()
    assert "train/loss" in fw.default_plot_metrics()


def test_filter_training_config_drops_infra_keys():
    fw = get_framework("verl_sft")
    cfg = {
        "optim": {"lr": 1e-4},
        "trainer": {"logger": ["console"], "total_epochs": 1},
        "ray_init": {"num_cpus": 8},
        "profiler": {"enable": False},
    }
    out = fw.filter_training_config(cfg)
    assert "optim" in out
    assert "ray_init" not in out
    assert "profiler" not in out


def test_prompt_fragments():
    fw = get_framework("verl_sft")
    prop = fw.prompt_fragments("proposer")
    assert "val/loss" in prop["objective"]
    assert "LOCKED" in prop["metric_interpretation"]
    es = fw.prompt_fragments("early_stopper")
    assert "role" in es and "objective" in es and "metric_interpretation" in es
    assert fw.prompt_fragments("unknown_agent") == {}


def test_build_observers_wires_standard_set():
    from types import SimpleNamespace

    from autogluon.llmzero.execution.observers import ValidationObserver

    fw = get_framework("verl_sft")
    ctx = SimpleNamespace(
        validation_score_mode="best_validation",
        validation_metric_pattern=None,  # falls back to fw.validation_metric_pattern()
        early_stopper_enabled=False,
        is_from_scratch=True,
        early_stopper_callback=None,
        checkpoint_optimization_enabled=True,
        checkpoint_interval_seconds=900,
        best_run_metrics=None,
        best_run_validation_score=None,
        min_steps_before_early_stop=20,
    )
    observers = fw.build_observers(ctx)
    types = {type(o).__name__ for o in observers}
    assert "ValidationObserver" in types
    assert "StepMetricsObserver" in types
    # score_direction propagates to the validation observer
    vo = next(o for o in observers if isinstance(o, ValidationObserver))
    assert vo.minimize is True
