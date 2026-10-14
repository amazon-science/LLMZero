"""Tests for the training framework abstraction seams."""

from autogluon.llmzero.training_frameworks import get_framework
from autogluon.llmzero.training_frameworks.base import BaseTrainingFramework


def test_base_score_direction_defaults_to_maximize():
    assert BaseTrainingFramework.score_direction() == "maximize"


def test_base_normalize_score_maximize_is_passthrough():
    assert BaseTrainingFramework.normalize_score(0.7) == 0.7
    assert BaseTrainingFramework.better(0.8, 0.5) is True
    assert BaseTrainingFramework.better(0.3, 0.5) is False


def test_minimize_framework_flips_normalization():
    class MinFramework(BaseTrainingFramework):
        @classmethod
        def score_direction(cls):
            return "minimize"

    assert MinFramework.normalize_score(0.7) == -0.7
    # lower raw is better when minimizing
    assert MinFramework.better(0.3, 0.5) is True
    assert MinFramework.better(0.8, 0.5) is False


def test_verl_is_maximize_and_has_validation_pattern():
    fw = get_framework("verl_grpo")
    assert fw.score_direction() == "maximize"
    assert fw.validation_metric_pattern() == "val-core/.*__dev"


def test_verl_parser_parses_step_metrics_with_npfloat():
    from autogluon.llmzero.training_frameworks.verl_grpo.parser import VerlStdoutParser

    p = VerlStdoutParser()
    line = "step:3 - global_seqlen/min:np.float64(23491.0) actor/entropy:2.435 val-core/foo__dev/reward:0.5"
    step, metrics = p.parse_step_metrics(line)
    assert step == 3
    assert metrics["actor/entropy"] == 2.435
    assert metrics["val-core/foo__dev/reward"] == 0.5
    assert metrics["global_seqlen/min"] == 23491.0


def test_verl_parser_parses_step_metrics_standalone():
    """VerlStdoutParser must parse step metrics on its own (no shared default parser)."""
    from autogluon.llmzero.training_frameworks.verl_grpo.parser import VerlStdoutParser

    p = VerlStdoutParser()
    step, metrics = p.parse_step_metrics("step:3 - global_seqlen/min:23491.0 - actor/entropy:2.435")
    assert step == 3
    assert metrics["global_seqlen/min"] == 23491.0
    assert metrics["actor/entropy"] == 2.435
    # np.float64(...) wrapper is stripped
    step2, m2 = p.parse_step_metrics("step:4 - global_seqlen/min:np.float64(10.0) actor/kl:1.5")
    assert step2 == 4 and m2["actor/kl"] == 1.5
    # a non-verl loose line (no global_seqlen anchor) is rejected by the tight anchor
    assert p.parse_step_metrics("step:7 loss:0.5 acc:0.9") is None


def test_verl_parser_extracts_checkpoint_path_step():
    from autogluon.llmzero.training_frameworks.verl_grpo.parser import VerlStdoutParser

    p = VerlStdoutParser()
    ckpt = p.parse_checkpoint("local_global_step_folder: /run/global_step_120", pending_step=None)
    assert ckpt == {"step": 120, "path": "/run/global_step_120"}


def test_verl_parser_progress():
    from autogluon.llmzero.training_frameworks.verl_grpo.parser import VerlStdoutParser

    p = VerlStdoutParser()
    out = p.parse_progress("Training Progress:  10%|## | 5/50 [00:10<01:30, 2.0s/it]")
    assert out == (5, 2.0)


def test_verl_framework_returns_parser():
    from autogluon.llmzero.training_frameworks import get_framework

    parser = get_framework("verl_grpo").stdout_parser()
    assert parser.parse_step_metrics("step:1 - global_seqlen/min:1.0 a/b:2.0")[0] == 1


def test_base_parser_subclass_parses_minimal_grammar():
    """A minimal BaseStdoutParser subclass implements only parse_step_metrics."""
    import re

    from autogluon.llmzero.training_frameworks.stdout_parser import BaseStdoutParser

    class _Loose(BaseStdoutParser):
        _TOKEN = re.compile(r"([\w/@-]+):([-+]?[0-9.]+)")

        def parse_step_metrics(self, line):
            if "step:" not in line:
                return None
            step = int(re.search(r"step:(\d+)", line).group(1))
            metrics = {k: float(v) for k, v in self._TOKEN.findall(line.split("step:")[1])}
            metrics.pop(str(step), None)  # drop the step:N token itself if captured
            return (step, metrics) if metrics else None

    p = _Loose()
    step, metrics = p.parse_step_metrics("step:7 loss:0.5 acc:0.9")
    assert step == 7
    assert metrics["loss"] == 0.5 and metrics["acc"] == 0.9


def test_select_validation_metric_locks_first_match():
    import re

    from autogluon.llmzero.training_frameworks.verl_grpo.parser import VerlStdoutParser

    p = VerlStdoutParser()
    pat = re.compile(r"val-core/.*__dev")
    score, name = p.select_validation_metric({"x/y": 1.0, "val-core/foo__dev/r": 0.7}, pat, None)
    assert score == 0.7 and name == "val-core/foo__dev/r"
    score2, name2 = p.select_validation_metric({"val-core/bar__dev/r": 0.9, "val-core/foo__dev/r": 0.3}, pat, name)
    assert score2 == 0.3 and name2 == "val-core/foo__dev/r"


def test_build_observers_standard_set_for_best_validation():
    from autogluon.llmzero.training_frameworks import get_framework
    from autogluon.llmzero.training_frameworks.observer_context import ObserverContext

    ctx = ObserverContext(
        validation_score_mode="best_validation",
        validation_metric_pattern="val-core/.*__dev",
        early_stopper_enabled=False,
        checkpoint_optimization_enabled=False,
    )
    observers = get_framework("verl_grpo").build_observers(ctx)
    names = {o.name for o in observers}
    assert "validation" in names
    assert "step_metrics" in names


def test_build_observers_includes_checkpoint_when_enabled():
    from autogluon.llmzero.training_frameworks import get_framework
    from autogluon.llmzero.training_frameworks.observer_context import ObserverContext

    ctx = ObserverContext(
        validation_score_mode="best_validation",
        validation_metric_pattern="val-core/.*__dev",
        early_stopper_enabled=False,
        checkpoint_optimization_enabled=True,
    )
    names = {o.name for o in get_framework("verl_grpo").build_observers(ctx)}
    assert "checkpoint_paths" in names


def test_build_observers_best_validation_requires_pattern():
    import pytest

    from autogluon.llmzero.training_frameworks.base import BaseTrainingFramework
    from autogluon.llmzero.training_frameworks.observer_context import ObserverContext
    from autogluon.llmzero.training_frameworks.stdout_parser import BaseStdoutParser

    class _NoPatternParser(BaseStdoutParser):
        def parse_step_metrics(self, line):
            return None

    class _NoPatternFramework(BaseTrainingFramework):
        # validation_metric_pattern() inherits None from base; ctx has none either
        @classmethod
        def stdout_parser(cls):
            return _NoPatternParser()

    ctx = ObserverContext(validation_score_mode="best_validation", validation_metric_pattern=None)
    with pytest.raises(ValueError):
        _NoPatternFramework.build_observers(ctx)


def test_build_observers_final_validation_mode():
    from autogluon.llmzero.training_frameworks import get_framework
    from autogluon.llmzero.training_frameworks.observer_context import ObserverContext

    ctx = ObserverContext(
        validation_score_mode="final_validation",
        validation_metric_pattern="val-core/.*__dev",
    )
    observers = get_framework("verl_grpo").build_observers(ctx)
    val = next(o for o in observers if o.name == "validation")
    assert val.reduction == "final"
    assert "step_metrics" in {o.name for o in observers}


def test_validation_observer_result_key_matches_executer_contract():
    """The validation observer's name/result keys must match what executer_agent reads."""
    from autogluon.llmzero.training_frameworks import get_framework
    from autogluon.llmzero.training_frameworks.observer_context import ObserverContext

    ctx = ObserverContext(
        validation_score_mode="best_validation",
        validation_metric_pattern=r"val-core/.*__dev",
    )
    val = next(o for o in get_framework("verl_grpo").build_observers(ctx) if o.name == "validation")
    # The verl parser requires the "global_seqlen/" anchor on metrics lines.
    val.on_stdout_line("step:1 - global_seqlen/min:1.0 val-core/x__dev/reward/mean@1:0.5")
    result = val.get_result()
    # executer_agent reads observer_results["validation"]["validation_score"]
    assert "validation_score" in result
    assert result["validation_score"] == 0.5


def test_verl_prompt_fragments_present():
    from autogluon.llmzero.training_frameworks import get_framework

    frags = get_framework("verl_grpo").prompt_fragments("proposer")
    assert "objective" in frags and frags["objective"]
    assert "metric_interpretation" in frags

    es = get_framework("verl_grpo").prompt_fragments("early_stopper")
    assert "role" in es and es["role"]


def test_base_prompt_fragments_empty_by_default():
    from autogluon.llmzero.training_frameworks.base import BaseTrainingFramework

    assert BaseTrainingFramework.prompt_fragments("proposer") == {}
    assert BaseTrainingFramework.prompt_template("proposer") is None


def test_validate_framework_config_accepts_known():
    from types import SimpleNamespace

    from autogluon.llmzero.training_frameworks import validate_framework_config

    validate_framework_config(SimpleNamespace(training_framework="verl_grpo"))  # no raise


def test_validate_framework_config_rejects_unknown():
    from types import SimpleNamespace

    import pytest

    from autogluon.llmzero.training_frameworks import validate_framework_config

    with pytest.raises(KeyError):
        validate_framework_config(SimpleNamespace(training_framework="does_not_exist"))


def test_validate_framework_config_allows_unset():
    from types import SimpleNamespace

    from autogluon.llmzero.training_frameworks import validate_framework_config

    validate_framework_config(SimpleNamespace())  # unset -> base, no raise


def test_build_observers_injects_parser_into_checkpoint_observer():
    """CheckpointPathObserver must receive the framework parser (no config patterns)."""
    from autogluon.llmzero.training_frameworks import get_framework
    from autogluon.llmzero.training_frameworks.observer_context import ObserverContext
    from autogluon.llmzero.training_frameworks.verl_grpo.parser import VerlStdoutParser

    ctx = ObserverContext(
        validation_score_mode="best_validation",
        validation_metric_pattern="val-core/.*__dev",
        checkpoint_optimization_enabled=True,
    )
    observers = get_framework("verl_grpo").build_observers(ctx)
    ckpt = next(o for o in observers if o.name == "checkpoint_paths")
    assert isinstance(ckpt.parser, VerlStdoutParser)


def test_observer_context_has_no_pattern_fields():
    """step_save_pattern / path_pattern are framework-owned now, not ctx fields."""
    from autogluon.llmzero.training_frameworks.observer_context import ObserverContext

    ctx = ObserverContext()
    assert not hasattr(ctx, "step_save_pattern")
    assert not hasattr(ctx, "path_pattern")


def test_base_framework_stdout_parser_is_abstract():
    """A framework that does not implement stdout_parser cannot be used."""
    import pytest

    from autogluon.llmzero.training_frameworks.base import BaseTrainingFramework

    class _NoParser(BaseTrainingFramework):
        @classmethod
        def validation_metric_pattern(cls):
            return r"val/x"

    # @abstractmethod blocks instantiation of a subclass that omits stdout_parser.
    with pytest.raises((TypeError, NotImplementedError)):
        _NoParser().stdout_parser()

    # Frameworks are used via classmethods (never instantiated in production), so
    # the body-raise is the contract that actually fires on the real call path.
    with pytest.raises(NotImplementedError):
        _NoParser.stdout_parser()


def test_base_log_stream_defaults_to_stdout():
    assert BaseTrainingFramework.log_stream() == "stdout"


def test_build_observers_threads_log_stream_to_all_observers():
    """A stderr framework's build_observers must set log_stream='stderr' on every observer."""
    import re

    from autogluon.llmzero.training_frameworks.observer_context import ObserverContext
    from autogluon.llmzero.training_frameworks.stdout_parser import BaseStdoutParser

    class _StderrParser(BaseStdoutParser):
        _TOKEN = re.compile(r"([\w/@-]+):([-+]?[0-9]*\.?[0-9]+)")

        def parse_step_metrics(self, line):
            if "step:" not in line:
                return None
            m = re.search(r"step:(\d+)", line)
            metrics = {k: float(v) for k, v in self._TOKEN.findall(line) if k != "step"}
            return (int(m.group(1)), metrics) if m and metrics else None

    class _StderrFramework(BaseTrainingFramework):
        @classmethod
        def log_stream(cls):
            return "stderr"

        @classmethod
        def validation_metric_pattern(cls):
            return r"val/score"

        @classmethod
        def stdout_parser(cls):
            return _StderrParser()

    ctx = ObserverContext(
        validation_score_mode="best_validation",
        validation_metric_pattern=r"val/score",
        checkpoint_optimization_enabled=True,
    )
    observers = _StderrFramework.build_observers(ctx)
    assert observers  # non-empty
    assert all(o.log_stream == "stderr" for o in observers)
