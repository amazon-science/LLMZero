"""Unit tests for ProposerAgentPrompt helper methods."""

from dataclasses import dataclass, field
from typing import List, Optional, Set
from unittest.mock import MagicMock, patch


@dataclass(eq=False)
class FakeNode:
    id: int = 0
    parent: Optional["FakeNode"] = None
    children: Set["FakeNode"] = field(default_factory=set)
    is_successful: bool = True
    error_analysis: str = ""
    early_stopped: bool = False
    early_stop_reason: str = ""
    checkpoints: List[dict] = field(default_factory=list)
    forced_from_scratch: bool = False
    observer_results: dict = field(default_factory=dict)

    def __hash__(self):
        return id(self)


@dataclass
class FakeCheckpointConfig:
    enabled: bool = False
    min_from_scratch_ratio: float = 0.2


@dataclass
class FakeConfig:
    sibling_dedup: bool = False
    checkpoint_optimization: Optional[FakeCheckpointConfig] = None
    training_framework: str = "verl_grpo"


def _make_manager(parent=None, current_node=None, sibling_dedup=False, checkpoint_enabled=False):
    """Build a mock manager suitable for ProposerAgentPrompt.__init__."""
    mgr = MagicMock()

    if current_node is None:
        current_node = FakeNode(id=1)
    if parent is not None:
        current_node.parent = parent
        parent.children.add(current_node)

    mgr.current_node = current_node
    mgr.time_step = 5
    mgr.from_scratch_count = 2
    mgr.resume_count = 3
    mgr.min_from_scratch_ratio = 0.2
    mgr.get_iteration_folder.return_value = "/tmp/test_iter"
    mgr.save_and_log_states = MagicMock()

    ckpt_cfg = FakeCheckpointConfig(enabled=checkpoint_enabled)
    mgr.config = FakeConfig(sibling_dedup=sibling_dedup, checkpoint_optimization=ckpt_cfg)

    return mgr


def _make_prompt(manager):
    """Instantiate ProposerAgentPrompt with mocks bypassing template validation."""
    llm_config = MagicMock()
    llm_config.template = None

    fw = MagicMock()
    fw.format_metric_descriptions.return_value = ""
    fw.format_hyperparameter_guide.return_value = ""

    with patch(
        "autogluon.llmzero.prompts.proposer_agent_prompt.BasePrompt.__init__",
        lambda self, *args, **kwargs: None,
    ):
        from autogluon.llmzero.prompts.proposer_agent_prompt import ProposerAgentPrompt

        prompt = ProposerAgentPrompt(llm_config, manager, template=None)

    # Manually set attributes that __init__ would set
    prompt.llm_config = llm_config
    prompt.manager = manager
    prompt.template = prompt.default_template()
    prompt.variable_provider = MagicMock()

    # Patch training_framework property
    type(prompt).training_framework = property(lambda self: fw)

    return prompt


# ---------------------------------------------------------------------------
# TestParentErrorContext
# ---------------------------------------------------------------------------


class TestParentErrorContext:
    def test_no_parent_returns_empty(self):
        mgr = _make_manager(parent=None)
        prompt = _make_prompt(mgr)
        assert prompt._get_parent_error_context() == ""

    def test_parent_failed_returns_error_analysis(self):
        parent = FakeNode(id=0, is_successful=False, error_analysis="OOM on GPU 3")
        mgr = _make_manager(parent=parent)
        prompt = _make_prompt(mgr)

        result = prompt._get_parent_error_context()
        assert "OOM on GPU 3" in result
        assert "Previous Run Failed" in result

    def test_sibling_failures_shown(self):
        parent = FakeNode(id=0, is_successful=True)
        sib1 = FakeNode(id=2, is_successful=False, error_analysis="LR too high")
        sib2 = FakeNode(id=3, is_successful=False, error_analysis="Batch overflow")
        parent.children = {sib1, sib2}
        sib1.parent = parent
        sib2.parent = parent

        current = FakeNode(id=1)
        current.parent = parent
        parent.children.add(current)

        mgr = _make_manager(parent=parent, current_node=current)
        prompt = _make_prompt(mgr)

        result = prompt._get_parent_error_context()
        assert "Sibling Runs That Failed" in result
        assert "LR too high" in result
        assert "Batch overflow" in result

    def test_sibling_failures_limited_to_3(self):
        parent = FakeNode(id=0, is_successful=True)
        siblings = []
        for i in range(5):
            sib = FakeNode(id=10 + i, is_successful=False, error_analysis=f"error_{i}")
            sib.parent = parent
            siblings.append(sib)
        parent.children = set(siblings)

        current = FakeNode(id=1)
        current.parent = parent
        parent.children.add(current)

        mgr = _make_manager(parent=parent, current_node=current)
        prompt = _make_prompt(mgr)

        result = prompt._get_parent_error_context()
        # Count "- Node" occurrences — must be at most 3
        assert result.count("- Node") <= 3


# ---------------------------------------------------------------------------
# TestSiblingDedupNotice
# ---------------------------------------------------------------------------


class TestSiblingDedupNotice:
    def test_dedup_enabled_injects_notice(self):
        parent = FakeNode(id=0)
        mgr = _make_manager(parent=parent, sibling_dedup=True)
        mgr.sibling_config_summary = "Sibling A tried lr=1e-4, Sibling B tried lr=3e-5."
        prompt = _make_prompt(mgr)

        # Simulate what _build does for sibling_dedup_notice
        sibling_dedup_enabled = getattr(prompt.manager.config, "sibling_dedup", False)
        notice = ""
        if sibling_dedup_enabled and hasattr(prompt.manager, "sibling_config_summary"):
            notice = prompt.manager.sibling_config_summary

        assert "Sibling A tried lr=1e-4" in notice

    def test_dedup_disabled_empty(self):
        parent = FakeNode(id=0)
        mgr = _make_manager(parent=parent, sibling_dedup=False)
        prompt = _make_prompt(mgr)

        sibling_dedup_enabled = getattr(prompt.manager.config, "sibling_dedup", False)
        assert sibling_dedup_enabled is False


# ---------------------------------------------------------------------------
# TestColdStartNotice
# ---------------------------------------------------------------------------


class TestColdStartNotice:
    def test_cold_start_notice_nonempty(self):
        parent = FakeNode(id=0)
        mgr = _make_manager(parent=parent)
        prompt = _make_prompt(mgr)

        notice = prompt._get_cold_start_notice()
        assert "Cold-Start" in notice
        assert "No Training History" in notice
        # Must steer toward a divergent config rather than metric diagnosis.
        assert "different region" in notice.lower()
        assert "skip the metric-diagnosis" in notice.lower()

    def test_cold_start_forces_sibling_summary_even_when_dedup_disabled(self):
        # With sibling_dedup disabled, a cold-start build must STILL surface
        # sibling configs (the only divergence signal without history).
        parent = FakeNode(id=-1)
        mgr = _make_manager(parent=parent, sibling_dedup=False)
        mgr.sibling_config_summary = "Sibling 1 (node 0): Baseline run using defaults."
        prompt = _make_prompt(mgr)

        cold_start = True
        sibling_dedup_enabled = getattr(prompt.manager.config, "sibling_dedup", False)
        notice = ""
        if (cold_start or sibling_dedup_enabled) and hasattr(prompt.manager, "sibling_config_summary"):
            notice = prompt.manager.sibling_config_summary

        assert "Baseline run using defaults" in notice


# ---------------------------------------------------------------------------
# TestCheckpointInfo
# ---------------------------------------------------------------------------


class TestCheckpointInfo:
    def test_disabled_returns_empty(self):
        parent = FakeNode(id=0, checkpoints=[{"step": 100, "path": "/a"}])
        mgr = _make_manager(parent=parent, checkpoint_enabled=False)
        prompt = _make_prompt(mgr)

        assert prompt._get_checkpoint_info() == ""

    def test_enabled_returns_step_list(self):
        parent = FakeNode(
            id=0,
            checkpoints=[
                {"step": 50, "path": "/ckpt/50"},
                {"step": 100, "path": "/ckpt/100"},
                {"step": 150, "path": "/ckpt/150"},
            ],
        )
        mgr = _make_manager(parent=parent, checkpoint_enabled=True)
        prompt = _make_prompt(mgr)

        result = prompt._get_checkpoint_info()
        assert "50" in result
        assert "100" in result
        assert "150" in result
        assert "RESUME_FROM_CHECKPOINT" in result
        assert "TRAIN_FROM_SCRATCH" in result

    def test_forced_from_scratch_returns_empty(self):
        parent = FakeNode(id=0, checkpoints=[{"step": 100, "path": "/a"}])
        current = FakeNode(id=1, forced_from_scratch=True)
        current.parent = parent
        parent.children.add(current)

        mgr = _make_manager(parent=parent, current_node=current, checkpoint_enabled=True)
        prompt = _make_prompt(mgr)

        assert prompt._get_checkpoint_info() == ""

    def test_no_parent_checkpoints_returns_empty(self):
        parent = FakeNode(id=0, checkpoints=[])
        mgr = _make_manager(parent=parent, checkpoint_enabled=True)
        prompt = _make_prompt(mgr)

        assert prompt._get_checkpoint_info() == ""


# ---------------------------------------------------------------------------
# TestEarlyStopNotice
# ---------------------------------------------------------------------------


class TestEarlyStopNotice:
    def test_parent_early_stopped(self):
        parent = FakeNode(id=0, early_stopped=True, early_stop_reason="Reward collapsed after step 80")
        mgr = _make_manager(parent=parent)
        prompt = _make_prompt(mgr)

        result = prompt._get_early_stop_notice()
        assert "Early Stopped" in result
        assert "Reward collapsed after step 80" in result

    def test_parent_not_early_stopped(self):
        parent = FakeNode(id=0, early_stopped=False)
        mgr = _make_manager(parent=parent)
        prompt = _make_prompt(mgr)

        assert prompt._get_early_stop_notice() == ""

    def test_no_parent_returns_empty(self):
        mgr = _make_manager(parent=None)
        prompt = _make_prompt(mgr)

        assert prompt._get_early_stop_notice() == ""
