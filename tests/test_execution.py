"""Unit tests for CodeExecutor and execution observers."""

import re
from unittest.mock import MagicMock, patch

import pytest

from autogluon.llmzero.execution.executor import CodeExecutor
from autogluon.llmzero.execution.observers import (
    BaseExecutionObserver,
    CheckpointPathObserver,
    StepMetricsObserver,
    ValidationObserver,
)
from autogluon.llmzero.training_frameworks.stdout_parser import BaseStdoutParser
from autogluon.llmzero.training_frameworks.verl_grpo.parser import VerlStdoutParser


class _LooseTestParser(BaseStdoutParser):
    """Minimal Tier-1 parser: loose 'step:N key:value' grammar, no checkpoints.

    Doubles as proof a framework can drive the observers by implementing only
    parse_step_metrics.
    """

    _STEP = re.compile(r"step:\d+\s+-\s+")
    _TOKEN = re.compile(r"([\w/@-]+):([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")

    def parse_step_metrics(self, line):
        if not self._STEP.search(line):
            return None
        step_match = re.search(r"step:(\d+)", line)
        after = re.search(r"step:\d+(.*)$", line)
        if not step_match or not after:
            return None
        metrics = {k: float(v) for k, v in self._TOKEN.findall(after.group(1))}
        return (int(step_match.group(1)), metrics) if metrics else None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def executor():
    """CodeExecutor with no backend (pure init test)."""
    return CodeExecutor(language="bash", timeout=300.0)


@pytest.fixture
def executor_with_backend():
    """CodeExecutor with mocked compute backend."""
    with patch("autogluon.llmzero.compute_infra.create_backend") as mock_create:
        mock_backend = MagicMock()
        mock_backend.backend_type = "raycluster"
        mock_create.return_value = mock_backend
        exe = CodeExecutor(language="bash", timeout=600.0, compute_backend_config={"type": "raycluster"})
    return exe


@pytest.fixture
def validation_observer():
    """ValidationObserver with pattern matching reward metrics."""
    return ValidationObserver(
        _LooseTestParser(),
        validation_metric_pattern=r"val-core.*reward",
        name="test_validation",
    )


@pytest.fixture
def step_observer():
    """StepMetricsObserver with relaxed step pattern."""
    return StepMetricsObserver(_LooseTestParser(), name="test_step_metrics")


class _DummyObserver(BaseExecutionObserver):
    """Minimal concrete observer for attachment tests."""

    def __init__(self, name="dummy"):
        super().__init__(name)
        self.lines = []

    def on_stdout_line(self, line):
        self.lines.append(line)

    def on_execution_complete(self, success, stdout, stderr):
        pass

    def get_result(self):
        return {"lines": self.lines}


# ---------------------------------------------------------------------------
# TestCodeExecutorInit
# ---------------------------------------------------------------------------


class TestCodeExecutorInit:
    def test_init_bash(self):
        exe = CodeExecutor(language="bash", timeout=120.0)
        assert exe.language == "bash"
        assert exe.timeout == 120.0
        assert exe.observers == []
        assert exe.compute_backend is None

    def test_init_python(self):
        exe = CodeExecutor(language="python", timeout=60.0)
        assert exe.language == "python"

    def test_init_unsupported_language(self):
        with pytest.raises(AssertionError):
            CodeExecutor(language="ruby", timeout=10.0)

    def test_init_with_backend_config(self, executor_with_backend):
        assert executor_with_backend.compute_backend is not None
        assert executor_with_backend.compute_backend.backend_type == "raycluster"

    def test_init_without_backend(self, executor):
        """No backend config means no compute_backend."""
        assert executor.compute_backend is None


# ---------------------------------------------------------------------------
# TestObserverAttachment
# ---------------------------------------------------------------------------


class TestObserverAttachment:
    def test_attach_single(self, executor):
        obs = _DummyObserver("obs1")
        executor.attach_observer(obs)
        assert len(executor.observers) == 1
        assert executor.observers[0] is obs

    def test_attach_multiple(self, executor):
        obs1 = _DummyObserver("obs1")
        obs2 = _DummyObserver("obs2")
        executor.attach_observer(obs1)
        executor.attach_observer(obs2)
        assert len(executor.observers) == 2
        assert executor.observers[0].name == "obs1"
        assert executor.observers[1].name == "obs2"

    def test_notify_stdout(self, executor):
        obs = _DummyObserver("obs1")
        executor.attach_observer(obs)
        executor._notify_observers_stdout("hello\n")
        assert obs.lines == ["hello\n"]

    def test_collect_results(self, executor):
        obs = _DummyObserver("obs1")
        executor.attach_observer(obs)
        obs.lines.append("x")
        results = executor._collect_observer_results()
        assert results == {"obs1": {"lines": ["x"]}}


# ---------------------------------------------------------------------------
# TestValidationObserver
# ---------------------------------------------------------------------------


class TestValidationObserver:
    def test_requires_metric_pattern(self):
        with pytest.raises(ValueError, match="validation_metric_pattern is required"):
            ValidationObserver(_LooseTestParser(), validation_metric_pattern=None)

    def test_no_match_returns_none(self, validation_observer):
        validation_observer.on_stdout_line("some random log line\n")
        assert validation_observer.best_validation_score is None
        assert validation_observer.validation_count == 0

    def test_extracts_validation_score(self, validation_observer):
        line = "step:5 - val-core/ssmr_agg__dev/reward/mean@1:0.37 - actor/kl_loss:0.015\n"
        validation_observer.on_stdout_line(line)
        assert validation_observer.validation_count == 1
        assert validation_observer.best_validation_score == pytest.approx(0.37)
        assert validation_observer.best_validation_step == 5
        assert validation_observer.first_validation_score == pytest.approx(0.37)

    def test_tracks_best_across_steps(self, validation_observer):
        lines = [
            "step:5 - val-core/ssmr_agg__dev/reward/mean@1:0.37 - actor/kl_loss:0.015\n",
            "step:10 - val-core/ssmr_agg__dev/reward/mean@1:0.42 - actor/kl_loss:0.012\n",
            "step:15 - val-core/ssmr_agg__dev/reward/mean@1:0.39 - actor/kl_loss:0.010\n",
        ]
        for line in lines:
            validation_observer.on_stdout_line(line)
        assert validation_observer.validation_count == 3
        assert validation_observer.best_validation_score == pytest.approx(0.42)
        assert validation_observer.best_validation_step == 10
        assert validation_observer.first_validation_score == pytest.approx(0.37)
        assert len(validation_observer.all_validation_scores) == 3

    def test_selected_metric_name_locked(self, validation_observer):
        line1 = "step:1 - val-core/ssmr_agg__dev/reward/mean@1:0.30 - other/metric:1.0\n"
        validation_observer.on_stdout_line(line1)
        assert validation_observer.selected_validation_metric_name == "val-core/ssmr_agg__dev/reward/mean@1"
        # Second line without the selected metric -> ignored
        line2 = "step:2 - other/metric:2.0 - actor/kl_loss:0.01\n"
        validation_observer.on_stdout_line(line2)
        assert validation_observer.validation_count == 1  # still 1

    def test_get_result(self, validation_observer):
        line = "step:3 - val-core/ssmr_agg__dev/reward/mean@1:0.55 - x:1.0\n"
        validation_observer.on_stdout_line(line)
        result = validation_observer.get_result()
        assert result["validation_score"] == pytest.approx(0.55)
        assert result["validation_step"] == 3
        assert result["validation_count"] == 1
        assert result["selected_validation_metric_name"] == "val-core/ssmr_agg__dev/reward/mean@1"

    def test_should_terminate_always_false(self, validation_observer):
        assert validation_observer.should_terminate() == (False, None)


# ---------------------------------------------------------------------------
# TestStepMetricsObserver
# ---------------------------------------------------------------------------


class TestStepMetricsObserver:
    def test_no_match(self, step_observer):
        step_observer.on_stdout_line("unrelated log output\n")
        assert step_observer.latest_step is None
        assert step_observer.metrics_history == []

    def test_extracts_metrics(self, step_observer):
        line = "step:5 - val-core/ssmr_agg__dev/reward/mean@1:0.37 - actor/kl_loss:0.015\n"
        step_observer.on_stdout_line(line)
        assert step_observer.latest_step == 5
        assert len(step_observer.metrics_history) == 1
        metrics = step_observer.latest_metrics
        assert "val-core/ssmr_agg__dev/reward/mean@1" in metrics
        assert metrics["val-core/ssmr_agg__dev/reward/mean@1"] == pytest.approx(0.37)
        assert "actor/kl_loss" in metrics
        assert metrics["actor/kl_loss"] == pytest.approx(0.015)

    def test_multiple_steps(self, step_observer):
        lines = [
            "step:1 - global_seqlen/min:23491.0 - actor/entropy:2.435\n",
            "step:2 - global_seqlen/min:23500.0 - actor/entropy:2.410\n",
        ]
        for line in lines:
            step_observer.on_stdout_line(line)
        assert step_observer.latest_step == 2
        assert len(step_observer.metrics_history) == 2
        assert step_observer.metrics_history[0]["step"] == 1
        assert step_observer.metrics_history[1]["step"] == 2

    def test_scientific_notation(self, step_observer):
        line = "step:10 - loss:3.5e-4 - lr:1.0e-5\n"
        step_observer.on_stdout_line(line)
        assert step_observer.latest_metrics["loss"] == pytest.approx(3.5e-4)
        assert step_observer.latest_metrics["lr"] == pytest.approx(1.0e-5)

    def test_get_result(self, step_observer):
        line = "step:7 - metric_a:1.5 - metric_b:2.5\n"
        step_observer.on_stdout_line(line)
        result = step_observer.get_result()
        assert result["num_entries"] == 1
        assert result["latest_step"] == 7
        assert result["metrics"]["metric_a"] == pytest.approx(1.5)

    def test_should_terminate_always_false(self, step_observer):
        assert step_observer.should_terminate() == (False, None)


# ---------------------------------------------------------------------------
# TestComputeBackendFactory
# ---------------------------------------------------------------------------


class TestComputeBackendFactory:
    def test_create_raycluster(self):
        from autogluon.llmzero.compute_infra import get_backend

        cls = get_backend("raycluster")
        assert cls is not None

    def test_create_conda(self):
        from autogluon.llmzero.compute_infra import get_backend

        cls = get_backend("conda")
        assert cls is not None

    def test_create_slurmdocker(self):
        from autogluon.llmzero.compute_infra import get_backend

        cls = get_backend("slurmdocker")
        assert cls is not None

    def test_create_docker(self):
        from autogluon.llmzero.compute_infra import get_backend

        cls = get_backend("docker")
        assert cls is not None

    def test_unknown_backend_raises(self):
        from autogluon.llmzero.compute_infra import get_backend

        with pytest.raises(KeyError, match="Unknown compute backend"):
            get_backend("nonexistent_backend")


# ---------------------------------------------------------------------------
# TestCheckpointPathObserver
# ---------------------------------------------------------------------------


class TestCheckpointPathObserver:
    def test_public_verl_single_line(self):
        obs = CheckpointPathObserver(VerlStdoutParser())
        obs.on_stdout_line("local_global_step_folder: /runs/exp/global_step_120\n")
        obs.on_stdout_line("local_global_step_folder: /runs/exp/global_step_240\n")
        result = obs.get_result()
        assert result["num_checkpoints"] == 2
        assert result["checkpoints"][0]["step"] == 120
        assert result["checkpoints"][0]["path"] == "/runs/exp/global_step_120"
        assert result["checkpoints"][1]["step"] == 240

    def test_path_step_arbitrary_prefix(self):
        obs = CheckpointPathObserver(VerlStdoutParser())
        obs.on_stdout_line("(TaskRunner pid=123) local_global_step_folder: /ck/global_step_7\n")
        result = obs.get_result()
        assert result["num_checkpoints"] == 1
        assert result["checkpoints"][0]["step"] == 7

    def test_embedded_step_wins_over_pending(self):
        obs = CheckpointPathObserver(VerlStdoutParser())
        obs.on_stdout_line("[step: 99] saving checkpoint\n")
        obs.on_stdout_line("local_global_step_folder: /runs/exp/global_step_100\n")
        result = obs.get_result()
        assert result["num_checkpoints"] == 1
        assert result["checkpoints"][0]["step"] == 100

    def test_no_checkpoint_lines(self):
        obs = CheckpointPathObserver(VerlStdoutParser())
        obs.on_stdout_line("some unrelated log line\n")
        obs.on_stdout_line("Training Progress: 10%\n")
        assert obs.get_result()["num_checkpoints"] == 0


# ---------------------------------------------------------------------------
# TestParseStepMetricsLine (utility)
# ---------------------------------------------------------------------------


class TestParseStepMetricsLine:
    @staticmethod
    def _parser():
        return _LooseTestParser()

    def test_valid_line(self):
        parser = self._parser()
        line = "step:3 - global_seqlen/min:23491.0 - actor/entropy:2.435"
        result = parser.parse_step_metrics(line)
        assert result is not None
        step, metrics = result
        assert step == 3
        assert metrics["global_seqlen/min"] == pytest.approx(23491.0)
        assert metrics["actor/entropy"] == pytest.approx(2.435)

    def test_no_match(self):
        parser = self._parser()
        assert parser.parse_step_metrics("INFO: some log message") is None

    def test_negative_values(self):
        parser = self._parser()
        line = "step:1 - loss:-0.5 - grad:-1.2e-3"
        result = parser.parse_step_metrics(line)
        assert result is not None
        _, metrics = result
        assert metrics["loss"] == pytest.approx(-0.5)
        assert metrics["grad"] == pytest.approx(-1.2e-3)


# ---------------------------------------------------------------------------
# Injected-parser tests (Task 5)
# ---------------------------------------------------------------------------


def test_step_observer_uses_injected_parser():
    from autogluon.llmzero.execution.observers import StepMetricsObserver

    obs = StepMetricsObserver(VerlStdoutParser())
    obs.on_stdout_line("step:1 - global_seqlen/min:np.float64(10.0) actor/entropy:1.5")
    result = obs.get_result()
    assert result["latest_step"] == 1
    assert result["metrics"]["actor/entropy"] == 1.5


def test_checkpoint_path_observer_pending_then_path():
    from autogluon.llmzero.execution.observers import CheckpointPathObserver

    obs = CheckpointPathObserver(VerlStdoutParser())
    # save marker primes pending step; path line without embedded step uses it
    obs.on_stdout_line("[step: 50] saving checkpoint")
    obs.on_stdout_line("local_global_step_folder: /run/ckpt_dir")
    ckpts = obs.get_result()["checkpoints"]
    assert len(ckpts) == 1 and ckpts[0]["step"] == 50 and ckpts[0]["path"] == "/run/ckpt_dir"


def test_checkpoint_path_observer_embedded_step_wins():
    from autogluon.llmzero.execution.observers import CheckpointPathObserver

    obs = CheckpointPathObserver(VerlStdoutParser())
    obs.on_stdout_line("[step: 50] saving checkpoint")
    obs.on_stdout_line("local_global_step_folder: /run/global_step_120")
    ckpts = obs.get_result()["checkpoints"]
    assert len(ckpts) == 1 and ckpts[0]["step"] == 120


def test_checkpoint_path_observer_pending_reset_on_non_path_line():
    from autogluon.llmzero.execution.observers import CheckpointPathObserver

    obs = CheckpointPathObserver(VerlStdoutParser())
    obs.on_stdout_line("[step: 50] saving checkpoint")
    obs.on_stdout_line("some unrelated log line")  # must discard pending
    obs.on_stdout_line("local_global_step_folder: /run/ckpt_dir")  # no embedded step, no pending
    assert obs.get_result()["checkpoints"] == []


def test_best_validation_observer_maximize_picks_highest():
    from autogluon.llmzero.execution.observers import ValidationObserver

    obs = ValidationObserver(_LooseTestParser(), validation_metric_pattern=r"val/score", score_direction="maximize")
    for step, val in [(1, 0.4), (2, 0.7), (3, 0.5)]:
        obs.on_stdout_line(f"step:{step} - val/score:{val}")
    result = obs.get_result()
    assert result["validation_score"] == pytest.approx(0.7)
    assert result["validation_step"] == 2


def test_best_validation_observer_minimize_picks_lowest():
    from autogluon.llmzero.execution.observers import ValidationObserver

    obs = ValidationObserver(_LooseTestParser(), validation_metric_pattern=r"val/loss", score_direction="minimize")
    for step, loss in [(1, 0.9), (2, 0.3), (3, 0.6)]:
        obs.on_stdout_line(f"step:{step} - val/loss:{loss}")
    result = obs.get_result()
    assert result["validation_score"] == pytest.approx(0.3)
    assert result["validation_step"] == 2


def test_validation_observer_final_reports_last_score():
    from autogluon.llmzero.execution.observers import ValidationObserver

    obs = ValidationObserver(_LooseTestParser(), validation_metric_pattern=r"val/time_per_step", reduction="final")
    for step, t in [(1, 5.0), (2, 3.0), (3, 3.4)]:
        obs.on_stdout_line(f"step:{step} - val/time_per_step:{t}")
    result = obs.get_result()
    assert result["reduction"] == "final"
    assert result["validation_score"] == pytest.approx(3.4)
    assert result["validation_step"] == 3


def test_build_observers_threads_score_direction_to_best_validation():
    """A minimize framework's build_observers must produce a minimize-aware best-val observer."""
    from autogluon.llmzero.training_frameworks.base import BaseTrainingFramework
    from autogluon.llmzero.training_frameworks.observer_context import ObserverContext

    class _MinFramework(BaseTrainingFramework):
        @classmethod
        def score_direction(cls):
            return "minimize"

        @classmethod
        def validation_metric_pattern(cls):
            return r"val/loss"

        @classmethod
        def stdout_parser(cls):
            return _LooseTestParser()

    ctx = ObserverContext(validation_score_mode="best_validation", validation_metric_pattern=r"val/loss")
    observers = _MinFramework.build_observers(ctx)
    best_val = next(o for o in observers if o.name == "validation")
    assert best_val.minimize is True


def test_observers_require_parser_positional():
    """Every observer requires a parser; constructing without one is a TypeError."""
    from autogluon.llmzero.execution.observers import (
        CheckpointPathObserver,
        StepMetricsObserver,
        ValidationObserver,
    )

    with pytest.raises(TypeError):
        StepMetricsObserver()
    with pytest.raises(TypeError):
        CheckpointPathObserver()
    with pytest.raises(TypeError):
        ValidationObserver(validation_metric_pattern="x")  # parser missing


def test_loose_parser_drives_step_and_validation_observers():
    """A minimal BaseStdoutParser subclass works end-to-end (Tier-1 contract)."""
    from autogluon.llmzero.execution.observers import StepMetricsObserver, ValidationObserver

    step_obs = StepMetricsObserver(_LooseTestParser())
    step_obs.on_stdout_line("step:5 - val-core/x__dev/reward:0.37 - actor/kl_loss:0.015")
    assert step_obs.get_result()["latest_step"] == 5

    val_obs = ValidationObserver(_LooseTestParser(), validation_metric_pattern=r"val-core.*reward")
    val_obs.on_stdout_line("step:5 - val-core/x__dev/reward:0.37 - actor/kl_loss:0.015")
    assert val_obs.get_result()["validation_score"] == pytest.approx(0.37)


# ---------------------------------------------------------------------------
# Log-stream routing (Task 1)
# ---------------------------------------------------------------------------


def test_step_observer_default_stdout_ignores_stderr():
    obs = StepMetricsObserver(_LooseTestParser())
    obs.on_stderr_line("step:5 - x:1.0")
    assert obs.latest_step is None  # stderr ignored by default
    obs.on_stdout_line("step:6 - x:2.0")
    assert obs.latest_step == 6


def test_step_observer_stderr_stream_ingests_stderr_only():
    obs = StepMetricsObserver(_LooseTestParser(), log_stream="stderr")
    obs.on_stdout_line("step:5 - x:1.0")
    assert obs.latest_step is None  # stdout ignored when log_stream=stderr
    obs.on_stderr_line("step:6 - x:2.0")
    assert obs.latest_step == 6


def test_step_observer_both_stream_ingests_both():
    obs = StepMetricsObserver(_LooseTestParser(), log_stream="both")
    obs.on_stdout_line("step:1 - x:1.0")
    obs.on_stderr_line("step:2 - x:2.0")
    assert obs.latest_step == 2
    assert len(obs.metrics_history) == 2


def test_validation_observer_stderr_stream():
    obs = ValidationObserver(_LooseTestParser(), validation_metric_pattern=r"val/score", log_stream="stderr")
    obs.on_stdout_line("step:1 - val/score:0.9")  # ignored
    assert obs.validation_count == 0
    obs.on_stderr_line("step:2 - val/score:0.8")  # ingested
    assert obs.validation_count == 1
    assert obs.best_validation_score == pytest.approx(0.8)
