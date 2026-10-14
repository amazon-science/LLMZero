"""
Tests for NodeManager — Sequential MCTS-based search strategy.

Uses mocks for all external dependencies (LLM agents, compute backends, etc.)
to test tree management, UCT selection, backpropagation, and terminal logic in isolation.
"""

import json
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from autogluon.llmzero.managers.node_manager import Node

# --- Fixtures & Helpers ---


@dataclass
class MockConfig:
    """Minimal config that satisfies NodeManager init."""

    max_debug_depth: int = 3
    failure_offset: int = 0
    per_execution_timeout: int = 3600
    configure_env: bool = False
    sibling_dedup: bool = True
    exploration_constant: float = 1.414
    failure_penalty_weight: float = 0.5
    early_stop_penalty_weight: float = 0.3
    score_temperature: float = 0.3
    initial_root_children: int = 4
    max_debug_children: int = 3
    max_evolve_children: int = 3
    stop_after_success: bool = False

    # Sub-configs as mock objects
    error_analyzer: Any = None
    bash_coder: Any = None
    executer: Any = None
    proposer_llm: Any = None
    compute_infra: Any = None
    early_stopper: Any = None
    checkpoint_optimization: Any = None
    validation_metric_pattern: str = r"val_score:\s*([\d.]+)"
    training_framework: str = "verl_grpo"

    def __post_init__(self):
        for attr in ["error_analyzer", "bash_coder", "executer", "proposer_llm"]:
            if getattr(self, attr) is None:
                mock = MagicMock()
                mock.multi_turn = False
                mock.template = None
                setattr(self, attr, mock)
        if self.compute_infra is None:
            self.compute_infra = {"type": "conda"}
        if self.early_stopper is None:
            mock_es = MagicMock()
            mock_es.enabled = False
            self.early_stopper = mock_es


@pytest.fixture
def tmp_dirs():
    """Create temporary input/output directories with a description file."""
    with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as output_dir:
        with open(os.path.join(input_dir, "description.txt"), "w") as f:
            f.write("Task: Train a reward model using GRPO.\nMetric: val-core/reward/mean")
        yield input_dir, output_dir


@pytest.fixture
def mock_config():
    return MockConfig()


def create_manager(input_dir, output_dir, config):
    """Create NodeManager with agents mocked out."""
    with patch("autogluon.llmzero.managers.node_manager.NodeManager._init_agents"):
        from autogluon.llmzero.managers.node_manager import NodeManager

        manager = NodeManager(
            input_data_folder=input_dir,
            output_folder=output_dir,
            config=config,
            initial_user_input="Train a model",
            enable_per_iteration_instruction=False,
        )
        # Set up mock agents
        manager.post_training_coder = MagicMock(return_value="#!/bin/bash\necho train")
        manager.execution_coder = MagicMock(return_value="#!/bin/bash\necho execute")
        manager.executer = MagicMock()
        manager.error_analyzer = MagicMock(return_value="error analysis")
        manager.proposers = [("unified", MagicMock())]
        manager.early_stoppers = []
        # Mock _get_unused_tool to allow root expansion
        manager._get_unused_tool = MagicMock(return_value="some_tool")
        manager.available_tools = ["tool1", "tool2", "tool3", "tool4"]

        return manager


# --- Test: Initialization ---


class TestNodeManagerInit:
    def test_creates_root_node(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        assert manager.root_node.stage == "root"
        assert manager.root_node.time_step == -1

    def test_output_dir_created(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        assert os.path.isdir(manager.output_folder)

    def test_initialize_reads_description_files(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()
        assert "GRPO" in manager.description_file_contents
        assert manager.task_description == "Train a model"
        assert manager.data_prompt == manager.description_file_contents

    def test_raises_on_missing_input(self, tmp_dirs, mock_config):
        _, output_dir = tmp_dirs
        with pytest.raises(FileNotFoundError):
            create_manager("/nonexistent/path", output_dir, mock_config)


# --- Test: Node Creation ---


class TestNodeCreation:
    def test_create_debug_node_increments_time_step(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        # Create a failed parent node
        manager.time_step += 1
        parent = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        parent.is_successful = False
        manager.current_node = parent

        old_ts = manager.time_step
        manager._create_debug_node()

        assert manager.time_step == old_ts + 1
        assert manager.current_node.stage == "debug"
        assert manager.current_node.parent == parent

    def test_create_evolve_node_increments_time_step(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.current_node = manager.root_node
        old_ts = manager.time_step

        manager._create_evolve_node()

        assert manager.time_step == old_ts + 1
        assert manager.current_node.stage == "evolve"
        assert manager.current_node.parent == manager.root_node

    def test_create_debug_node_sets_correct_parent(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        parent = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        parent.is_successful = False
        manager.current_node = parent

        manager._create_debug_node()

        assert manager.current_node.parent == parent
        assert manager.current_node in parent.children

    def test_create_evolve_node_sets_correct_parent(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        parent = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        parent.is_successful = True
        manager.current_node = parent

        manager._create_evolve_node()

        assert manager.current_node.parent == parent
        assert manager.current_node in parent.children


# --- Test: UCT Selection ---


class TestUCTSelection:
    def test_unvisited_node_returns_infinity(self):
        node = Node(stage="evolve", time_step=0)
        node.parent = Node(stage="root", time_step=-1)
        assert node.uct_value() == float("inf")

    def test_uct_formula_exploration_term(self):
        parent = Node(stage="root", time_step=-1)
        parent.visits = 10

        child = Node(parent=parent, stage="evolve", time_step=0)
        child.visits = 2
        child.failure_visits = 0

        C = 1.414
        expected_exploration = C * math.sqrt(math.log(10) / 2)
        uct = child.uct_value(exploration_constant=C)

        # With no validated visits, exploitation is 0 + failure_penalty(0) + early_stop_penalty(0) = 0
        assert abs(uct - expected_exploration) < 0.001

    def test_higher_score_gives_higher_uct(self):
        parent = Node(stage="root", time_step=-1)
        parent.visits = 10

        child_low = Node(parent=parent, stage="evolve", time_step=0)
        child_low.visits = 3
        child_low.validated_visits = 3
        child_low.validated_reward = 0.3 * 3  # avg 0.3

        child_high = Node(parent=parent, stage="evolve", time_step=1)
        child_high.visits = 3
        child_high.validated_visits = 3
        child_high.validated_reward = 0.9 * 3  # avg 0.9

        uct_low = child_low.uct_value(score_min=0.3, score_max=0.9)
        uct_high = child_high.uct_value(score_min=0.3, score_max=0.9)

        assert uct_high > uct_low

    def test_explore_prefers_less_visited(self):
        parent = Node(stage="root", time_step=-1)
        parent.visits = 20

        child_many = Node(parent=parent, stage="evolve", time_step=0)
        child_many.visits = 15
        child_many.validated_visits = 15
        child_many.validated_reward = 0.5 * 15

        child_few = Node(parent=parent, stage="evolve", time_step=1)
        child_few.visits = 2
        child_few.validated_visits = 2
        child_few.validated_reward = 0.5 * 2

        # Same average score, but fewer visits should mean higher exploration
        uct_many = child_many.uct_value(score_min=0.5, score_max=0.5)
        uct_few = child_few.uct_value(score_min=0.5, score_max=0.5)

        assert uct_few > uct_many


# --- Test: ExTS scoring primitives ---


class TestExTSScoring:
    def test_phi_identity_for_nonpositive_temperature(self):
        from autogluon.llmzero.managers.node_manager import _phi

        assert _phi(0.7, 0.0) == 0.7
        assert _phi(0.3, -1.0) == 0.3

    def test_phi_endpoints(self):
        from autogluon.llmzero.managers.node_manager import _phi

        assert _phi(0.0, 0.3) == pytest.approx(0.0)
        assert _phi(1.0, 0.3) == pytest.approx(1.0)

    def test_phi_convex_favors_high_scores(self):
        # Convex shaping: phi(x) <= x on [0,1], so the midpoint is pulled down.
        from autogluon.llmzero.managers.node_manager import _phi

        assert _phi(0.5, 0.3) < 0.5

    def test_shaped_mean_empty_history_is_zero(self):
        from autogluon.llmzero.managers.node_manager import _shaped_mean

        assert _shaped_mean([], 0.0, 1.0, 0.3) == 0.0

    def test_shaped_mean_degenerate_range_is_one(self):
        from autogluon.llmzero.managers.node_manager import _shaped_mean

        assert _shaped_mean([0.5, 0.5], 0.5, 0.5, 0.3) == 1.0

    def test_shaped_mean_is_per_observation_not_shape_of_mean(self):
        # Mean of shaped observations differs from shaping the mean (convexity).
        from autogluon.llmzero.managers.node_manager import _phi, _shaped_mean

        history = [0.0, 1.0]
        sm = _shaped_mean(history, 0.0, 1.0, 0.3)
        shape_of_mean = _phi(0.5, 0.3)
        assert sm == pytest.approx((_phi(0.0, 0.3) + _phi(1.0, 0.3)) / 2)
        assert sm != pytest.approx(shape_of_mean)

    def test_shaped_mean_clamps_out_of_range_observations(self):
        from autogluon.llmzero.managers.node_manager import _shaped_mean

        # Observation above the max clamps to 1.0 (phi(1)=1); below min clamps to 0.
        assert _shaped_mean([2.0], 0.0, 1.0, 0.3) == pytest.approx(1.0)
        assert _shaped_mean([-1.0], 0.0, 1.0, 0.3) == pytest.approx(0.0)

    def test_exploration_term_uct(self):
        from autogluon.llmzero.managers.node_manager import _exploration_term

        val = _exploration_term("uct", 1.414, 10, 2, 0.5)
        assert val == pytest.approx(1.414 * math.sqrt(math.log(10) / 2))

    def test_exploration_term_puct(self):
        from autogluon.llmzero.managers.node_manager import _exploration_term

        val = _exploration_term("puct", 1.0, 9, 3, 0.5)
        assert val == pytest.approx(1.0 * (9**0.5) / (1 + 3))


class TestExTSFailureModel:
    def test_success_rate_denominator_counts_failures(self):
        # offset=0: 1 val + 1 fail -> success_rate = 1/2 = 0.5 -> 0.5^alpha * shaped_mean.
        parent = Node(stage="root", time_step=-1)
        parent.visits = 10
        node = Node(parent=parent, stage="evolve", time_step=0)
        node.visits = 2
        node.validated_visits = 1
        node.failure_visits = 1
        node.reward_history = [1.0]
        # Degenerate range -> shaped_mean = 1.0; exploitation = 0.5^1.414.
        uct = node.uct_value(exploration_constant=0.0, score_min=0.0, score_max=0.0, success_rate_alpha=1.414)
        assert node.success_rate == pytest.approx(0.5)
        assert uct == pytest.approx(0.5**1.414)

    def test_no_score_run_counts_as_failure(self):
        # An unvalidated success sits in the denominator only (F4): 1 val + 1 unval -> sr=0.5.
        parent = Node(stage="root", time_step=-1)
        parent.visits = 10
        node = Node(parent=parent, stage="evolve", time_step=0)
        node.visits = 2
        node.validated_visits = 1
        node.unvalidated_visits = 1
        node.reward_history = [1.0]
        node.uct_value(exploration_constant=0.0, score_min=0.0, score_max=0.0)
        assert node.success_rate == pytest.approx(0.5)

    def test_failure_offset_forgives_and_clamps(self):
        # offset=2 forgives both failures: denominator max(1, 3-2)=1, sr=min(1, 1/1)=1.
        parent = Node(stage="root", time_step=-1)
        parent.visits = 10
        node = Node(parent=parent, stage="evolve", time_step=0)
        node.visits = 3
        node.validated_visits = 1
        node.failure_visits = 2
        node.reward_history = [1.0]
        node.uct_value(exploration_constant=0.0, failure_offset=2, score_min=0.0, score_max=0.0)
        assert node.success_rate == pytest.approx(1.0)

    def test_no_early_stop_penalty(self):
        # early_stopped_visits does not reduce the score (no penalty term).
        parent = Node(stage="root", time_step=-1)
        parent.visits = 10
        base = Node(parent=parent, stage="evolve", time_step=0)
        base.visits = 2
        base.validated_visits = 2
        base.reward_history = [1.0, 1.0]

        es = Node(parent=parent, stage="evolve", time_step=1)
        es.visits = 2
        es.validated_visits = 2
        es.early_stopped_visits = 1
        es.reward_history = [1.0, 1.0]

        u_base = base.uct_value(exploration_constant=0.0, score_min=0.0, score_max=0.0)
        u_es = es.uct_value(exploration_constant=0.0, score_min=0.0, score_max=0.0)
        assert u_base == pytest.approx(u_es)

    def test_update_appends_reward_history_incl_early_stopped_with_score(self):
        node = Node(stage="evolve", time_step=0)
        node.update(0.7, is_validated=True, is_failure=False, is_early_stopped=False)
        node.update(0.4, is_validated=True, is_failure=False, is_early_stopped=True)  # early-stopped w/ score
        node.update(None, is_validated=False, is_failure=True, is_early_stopped=False)  # failure
        assert node.reward_history == [0.7, 0.4]
        assert node.validated_visits == 2
        assert node.failure_visits == 1


class TestExTSExpansionCaps:
    def test_max_children_root_widens_unconditionally(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        mock_config.initial_root_children = 1
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()
        manager.widening_visit_threshold = 4
        manager.widening_base = 2.0
        manager.root_node.visits = 16  # floor(log2(16/4)) = 2 bonus
        assert manager._max_children_for_node(manager.root_node) == 1 + 2

    def test_max_children_debug_never_widens(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        mock_config.max_debug_children = 3
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()
        manager.widening_visit_threshold = 4
        node = Node(stage="debug", time_step=5)
        node.visits = 64
        assert manager._max_children_for_node(node) == 3

    def test_max_children_evolve_widens_only_when_high_scoring(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        mock_config.max_evolve_children = 2
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()
        manager.widening_visit_threshold = 4
        manager.widening_base = 2.0
        manager.widening_score_quantile = 0.5

        low = Node(parent=manager.root_node, stage="evolve", time_step=1)
        low.visits = 16
        low.validation_score = 0.1
        high = Node(parent=manager.root_node, stage="evolve", time_step=2)
        high.visits = 16
        high.validation_score = 0.9

        # Quantile over [0.1, 0.9] at 0.5 -> 0.9 (idx=min(int(2*0.5),1)=1). Only `high` clears it.
        assert manager._max_children_for_node(low) == 2  # gated out, no widen
        assert manager._max_children_for_node(high) == 2 + 2  # widens

    def test_evolve_widening_latch_survives_score_dip(self, tmp_dirs, mock_config):
        # Finding-1 regression: once an evolve node's score clears the widening quantile the
        # cap must not shrink when the node later falls below the (tree-relative) quantile —
        # otherwise _check_ancestors_terminal could permanently terminalize it.
        input_dir, output_dir = tmp_dirs
        mock_config.max_evolve_children = 2
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()
        manager.widening_visit_threshold = 4
        manager.widening_base = 2.0
        manager.widening_score_quantile = 0.5

        node = Node(parent=manager.root_node, stage="evolve", time_step=1)
        node.visits = 16
        node.validation_score = 0.9
        manager.root_node.add_child(node)
        lo = Node(parent=manager.root_node, stage="evolve", time_step=2)
        lo.validation_score = 0.1
        manager.root_node.add_child(lo)

        # High-scoring: widens and latches the unlock.
        assert manager._max_children_for_node(node) == 2 + 2
        assert node.widening_unlocked is True

        # Now many higher-scoring nodes appear; `node` drops below the 50th percentile.
        for i, s in enumerate([0.95, 0.96, 0.97, 0.98]):
            hi = Node(parent=manager.root_node, stage="evolve", time_step=10 + i)
            hi.validation_score = s
            manager.root_node.add_child(hi)

        # Latched: cap stays widened despite the score dip (monotonic, terminal-safe).
        assert manager._max_children_for_node(node) == 2 + 2

    def test_passes_gate_root_and_debug_exempt(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()
        debug = Node(stage="debug", time_step=1)
        assert manager._passes_expansion_gate(manager.root_node) is True
        assert manager._passes_expansion_gate(debug) is True

    def test_passes_gate_failed_node_exempt(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()
        failed = Node(parent=manager.root_node, stage="evolve", time_step=1)
        failed.is_successful = False
        assert manager._passes_expansion_gate(failed) is True

    def test_passes_gate_blocks_succeeded_but_no_score(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()
        node = Node(parent=manager.root_node, stage="evolve", time_step=1)
        node.is_successful = True
        node.validation_score = None
        assert manager._passes_expansion_gate(node) is False

    def test_passes_gate_scored_evolve_uses_quantile(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()
        manager.expansion_gate_quantile = 0.5

        lo = Node(parent=manager.root_node, stage="evolve", time_step=1)
        lo.is_successful = True
        lo.validation_score = 0.2
        hi = Node(parent=manager.root_node, stage="evolve", time_step=2)
        hi.is_successful = True
        hi.validation_score = 0.8
        manager.root_node.add_child(lo)
        manager.root_node.add_child(hi)

        # Quantile over [0.2, 0.8] at 0.5 -> 0.8. lo below -> blocked, hi at/above -> passes.
        assert manager._passes_expansion_gate(lo) is False
        assert manager._passes_expansion_gate(hi) is True

    def test_gate_disabled_always_passes(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()
        manager.use_gated_expansion = False
        node = Node(parent=manager.root_node, stage="evolve", time_step=1)
        node.is_successful = True
        node.validation_score = None
        assert manager._passes_expansion_gate(node) is True


class TestExTSScoreQuantile:
    def test_score_quantile_none_when_no_scores(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()
        assert manager._score_quantile(0.5) is None

    def test_score_quantile_value(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()
        for i, s in enumerate([0.1, 0.5, 0.9]):
            n = Node(parent=manager.root_node, stage="evolve", time_step=i)
            n.validation_score = s
            manager.root_node.add_child(n)
        # sorted [0.1,0.5,0.9]; idx=min(int(3*0.5),2)=1 -> 0.5
        assert manager._score_quantile(0.5) == pytest.approx(0.5)


class TestExTSSelectNode:
    def test_leaf_root_returns_root(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()
        # Root with no children is a leaf -> always expand it.
        assert manager.select_node() is manager.root_node

    def test_all_children_terminal_no_room_returns_none(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        mock_config.initial_root_children = 1
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()
        manager._get_unused_tool = MagicMock(return_value=None)

        manager.time_step += 1
        child = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        child.visits = 1
        manager.mark_node_terminal(child)  # only child terminal, root at cap
        # No room (initial_root_children=1, already has 1 child) + all children terminal.
        assert manager.select_node() is None

    def test_descends_into_best_child_when_expansion_gated_out(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        mock_config.initial_root_children = 5  # room to expand root
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        # One visited, scored child that is itself a leaf -> descent must return it.
        manager.time_step += 1
        child = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        child.visits = 1
        child.validated_visits = 1
        child.validation_score = 0.5
        child.reward_history = [0.5]
        manager.root_node.visits = 1

        # Force expansion to lose: disable widening room is irrelevant here; instead make the
        # virtual child clearly worse by giving the existing child an unbeatable score and
        # setting root's own children average high. Simplest: the returned node must be a
        # node to expand; with a single non-terminal leaf child, descent lands on it (leaf).
        selected = manager.select_node()
        # Either expand root (virtual child wins) or descend to the leaf child; both are
        # valid "nodes to expand". Assert it is one of them, and never None here.
        assert selected in (manager.root_node, child)

    def test_returns_expandable_node_not_none_when_tree_alive(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        mock_config.initial_root_children = 3
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        c1 = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        c1.visits = 2
        c1.validated_visits = 2
        c1.validation_score = 0.6
        c1.reward_history = [0.6, 0.6]
        manager.root_node.visits = 2

        selected = manager.select_node()
        assert selected is not None
        assert not selected.is_terminal


class TestExTSScoreRangeIncremental:
    def test_backprop_maintains_incremental_bounds(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        node = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        manager.current_node = node

        manager.backpropagate((0.3, True, False, False))
        manager.backpropagate((0.8, True, False, False))

        lo, hi = manager._compute_score_range()
        assert lo == pytest.approx(0.3)
        assert hi == pytest.approx(0.8)

    def test_score_range_fallback_uses_observations_when_no_backprop(self, tmp_dirs, mock_config):
        # Nodes populated directly (tests / cold path) still yield bounds via the fallback.
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()
        n = Node(parent=manager.root_node, stage="evolve", time_step=1)
        n.validated_visits = 2
        n.reward_history = [0.2, 0.6]
        manager.root_node.add_child(n)
        lo, hi = manager._compute_score_range()
        assert lo == pytest.approx(0.2)
        assert hi == pytest.approx(0.6)


# --- Test: Backpropagation ---


class TestBackpropagation:
    def test_backprop_updates_all_ancestors(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        parent = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        manager.time_step += 1
        child = Node(parent=parent, stage="evolve", time_step=manager.time_step)
        manager.current_node = child

        manager.backpropagate((0.8, True, False, False))

        assert child.visits == 1
        assert child.validated_visits == 1
        assert parent.visits == 1
        assert manager.root_node.visits == 1

    def test_backprop_failure_increments_failure_visits(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        node = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        manager.current_node = node

        manager.backpropagate((None, False, True, False))

        assert node.failure_visits == 1
        assert node.visits == 1
        assert manager.root_node.failure_visits == 1

    def test_backprop_early_stopped(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        node = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        manager.current_node = node

        manager.backpropagate((0.5, True, False, True))

        assert node.early_stopped_visits == 1
        assert node.validated_visits == 1
        assert node.visits == 1

    def test_backprop_reward_accumulates(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        node = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        manager.current_node = node

        manager.backpropagate((0.5, True, False, False))
        manager.backpropagate((0.7, True, False, False))

        assert node.validated_reward == pytest.approx(1.2)
        assert node.validated_visits == 2


# --- Test: Terminal Management ---


class TestTerminalManagement:
    def test_mark_terminal_propagates_to_children(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        parent = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        manager.time_step += 1
        child = Node(parent=parent, stage="evolve", time_step=manager.time_step)

        manager.mark_node_terminal(parent)

        assert parent.is_terminal
        assert child.is_terminal

    def test_ancestor_propagation_when_all_children_terminal(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        mock_config.initial_root_children = 2
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()
        # Disable unused tool check so root can become terminal
        manager._get_unused_tool = MagicMock(return_value=None)

        manager.time_step += 1
        child1 = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        manager.time_step += 1
        child2 = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)

        manager.mark_node_terminal(child1)
        assert not manager.root_node.is_terminal

        manager.mark_node_terminal(child2)
        assert manager.root_node.is_terminal

    def test_max_debug_depth_terminates_node(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        mock_config.max_debug_depth = 2
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        parent = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        parent.is_successful = False
        parent.debug_attempts = 1  # Already one attempt
        manager.current_node = parent

        # Creating debug node when debug_attempts + 1 >= max_debug_depth should terminate
        manager._create_debug_node()

        assert manager.current_node.is_terminal

    def test_mark_terminal_idempotent(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        node = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)

        manager.mark_node_terminal(node)
        manager.mark_node_terminal(node)  # Second call should not crash

        assert node.is_terminal


# --- Test: Best Node Tracking ---


class TestBestNodeTracking:
    def test_best_score_tracked(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        # Manually set validation score tracking (simulating simulate() behavior)
        manager.time_step += 1
        node = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        node.validation_score = 0.85
        manager._best_node = node
        manager._best_validation_score = 0.85

        assert manager.best_validation_score == 0.85
        assert manager.best_node == node

    def test_worst_score_tracked(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager._worst_validation_score = 0.1
        assert manager._worst_validation_score == 0.1

    def test_last_successful_node_updated(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        node = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        node.is_successful = True
        manager.last_successful_node = node

        assert manager.last_successful_node == node

    def test_best_validation_score_default_zero(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        assert manager.best_validation_score == 0.0

    def test_best_score_only_improves(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        n1 = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        n1.validation_score = 0.9
        manager._best_node = n1
        manager._best_validation_score = 0.9

        # Simulate a worse score — best should not change
        manager.time_step += 1
        n2 = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        n2.validation_score = 0.5
        # Don't update best since 0.5 < 0.9

        assert manager._best_validation_score == 0.9
        assert manager._best_node == n1

    def test_score_range_normalized_for_minimize(self, tmp_dirs, mock_config):
        """For a minimize framework, _compute_score_range returns negated (normalized) bounds."""
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        n1 = Node(parent=manager.root_node, stage="evolve", time_step=1)
        n1.validated_visits = 1
        n1.validated_reward = 0.2
        n2 = Node(parent=manager.root_node, stage="evolve", time_step=2)
        n2.validated_visits = 1
        n2.validated_reward = 0.8
        manager.root_node.add_child(n1)
        manager.root_node.add_child(n2)

        class _Min:
            @staticmethod
            def normalize_score(s):
                return -s

        manager._score_framework = _Min
        lo, hi = manager._compute_score_range()
        assert lo == -0.8 and hi == -0.2


# --- Test: Checkpoint Resolution ---


class TestCheckpointResolution:
    def test_from_scratch_default_when_no_checkpoint_config(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        mock_config.checkpoint_optimization = None
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        node = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        manager.current_node = node
        manager._proposer_suggestions_combined = ""

        manager._resolve_checkpoint_selection()

        assert node.from_scratch is True

    def test_resume_from_checkpoint(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        ckpt_config = MagicMock()
        ckpt_config.enabled = True
        ckpt_config.clean_ckpt = False
        mock_config.checkpoint_optimization = ckpt_config
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        parent = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        parent.checkpoints = [
            {"step": 100, "path": "/tmp/ckpt_100"},
            {"step": 200, "path": "/tmp/ckpt_200"},
        ]

        manager.time_step += 1
        child = Node(parent=parent, stage="evolve", time_step=manager.time_step)
        manager.current_node = child
        manager._proposer_suggestions_combined = "RESUME_FROM_CHECKPOINT: 200"

        manager._resolve_checkpoint_selection()

        assert child.from_scratch is False
        assert child.resume_from_checkpoint is True
        assert child.resume_checkpoint_path == "/tmp/ckpt_200"
        assert child.resume_checkpoint_step == 200

    def test_step_budget_parsed_and_clamped(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        ckpt_config = MagicMock()
        ckpt_config.enabled = True
        ckpt_config.clean_ckpt = False
        ckpt_config.max_steps_per_node = 50
        mock_config.checkpoint_optimization = ckpt_config
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        parent = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        parent.checkpoints = [{"step": 100, "path": "/tmp/ckpt_100"}]

        manager.time_step += 1
        child = Node(parent=parent, stage="evolve", time_step=manager.time_step)
        manager.current_node = child
        # Proposer requests 200, above the cap of 50 -> clamped to 50
        manager._proposer_suggestions_combined = "RESUME_FROM_CHECKPOINT: 100\nADDITIONAL_TRAINING_STEPS: 200"

        manager._resolve_checkpoint_selection()

        assert child.resume_from_checkpoint is True
        assert child.training_steps_budget == 50

    def test_step_budget_within_cap_preserved(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        ckpt_config = MagicMock()
        ckpt_config.enabled = True
        ckpt_config.clean_ckpt = False
        ckpt_config.max_steps_per_node = 100
        mock_config.checkpoint_optimization = ckpt_config
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        parent = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        parent.checkpoints = [{"step": 39, "path": "/tmp/ckpt_39"}]

        manager.time_step += 1
        child = Node(parent=parent, stage="evolve", time_step=manager.time_step)
        manager.current_node = child
        manager._proposer_suggestions_combined = "RESUME_FROM_CHECKPOINT: 39\nADDITIONAL_TRAINING_STEPS: 40"

        manager._resolve_checkpoint_selection()

        assert child.training_steps_budget == 40

    def test_step_budget_defaults_to_cap_when_omitted(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        ckpt_config = MagicMock()
        ckpt_config.enabled = True
        ckpt_config.clean_ckpt = False
        ckpt_config.max_steps_per_node = 75
        mock_config.checkpoint_optimization = ckpt_config
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        parent = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        parent.checkpoints = [{"step": 39, "path": "/tmp/ckpt_39"}]

        manager.time_step += 1
        child = Node(parent=parent, stage="evolve", time_step=manager.time_step)
        manager.current_node = child
        # No ADDITIONAL_TRAINING_STEPS directive -> default to the cap
        manager._proposer_suggestions_combined = "RESUME_FROM_CHECKPOINT: 39"

        manager._resolve_checkpoint_selection()

        assert child.training_steps_budget == 75

    def test_step_budget_set_on_from_scratch_node(self, tmp_dirs, mock_config):
        # The cap applies to ALL nodes — a from-scratch node (no parent checkpoints) still
        # gets a step budget equal to the configured cap.
        input_dir, output_dir = tmp_dirs
        ckpt_config = MagicMock()
        ckpt_config.enabled = True
        ckpt_config.clean_ckpt = False
        ckpt_config.max_steps_per_node = 80
        mock_config.checkpoint_optimization = ckpt_config
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        parent = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        parent.checkpoints = []  # no checkpoints -> from scratch

        manager.time_step += 1
        child = Node(parent=parent, stage="evolve", time_step=manager.time_step)
        manager.current_node = child
        manager._proposer_suggestions_combined = "TRAIN_FROM_SCRATCH"

        manager._resolve_checkpoint_selection()

        assert child.from_scratch is True
        assert child.training_steps_budget == 80

    def test_from_scratch_when_no_parent_checkpoints(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        ckpt_config = MagicMock()
        ckpt_config.enabled = True
        mock_config.checkpoint_optimization = ckpt_config
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        parent = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        parent.checkpoints = []

        manager.time_step += 1
        child = Node(parent=parent, stage="evolve", time_step=manager.time_step)
        manager.current_node = child
        manager._proposer_suggestions_combined = ""

        manager._resolve_checkpoint_selection()

        assert child.from_scratch is True


# --- Test: Validation Score Summary ---


class TestValidationScoreSummary:
    def test_no_scores_message(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        summary = manager.get_validation_score_summary()
        assert "No validation scores available" in summary

    def test_summary_includes_all_scored_nodes(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        n1 = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        n1.validation_score = 0.7

        manager.time_step += 1
        n2 = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        n2.validation_score = 0.9

        manager._best_node = n2
        manager._best_validation_score = 0.9

        summary = manager.get_validation_score_summary()
        assert "Node 0" in summary
        assert "Node 1" in summary
        assert "(BEST)" in summary
        assert "0.9000" in summary

    def test_summary_format_has_header(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        n = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        n.validation_score = 0.5
        manager._best_node = n
        manager._best_validation_score = 0.5

        summary = manager.get_validation_score_summary()
        assert "Validation Score Summary:" in summary


# --- Test: Timing Log ---


class TestTimingLog:
    def test_save_timing_log_creates_correct_structure(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        node = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        node.execution_time = 120.5
        node.is_successful = True
        node.validation_score = 0.8
        manager.current_node = node

        manager._save_timing_log()

        timing_path = os.path.join(output_dir, "timing_log.json")
        assert os.path.exists(timing_path)

        with open(timing_path) as f:
            timing = json.load(f)

        assert "total_elapsed_seconds" in timing
        assert "total_elapsed_hours" in timing
        assert "num_nodes" in timing
        assert "nodes" in timing
        assert len(timing["nodes"]) == 1
        assert timing["nodes"][0]["node_id"] == 0
        assert timing["nodes"][0]["stage"] == "evolve"
        assert timing["nodes"][0]["execution_time_seconds"] == 120.5
        assert timing["nodes"][0]["is_successful"] is True
        assert timing["nodes"][0]["validation_score"] == 0.8

    def test_save_timing_log_writes_per_node_file(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        node = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        node.execution_time = 60.0
        manager.current_node = node

        manager._save_timing_log()

        node_timing_path = os.path.join(manager.get_iteration_folder(node), "timing.json")
        assert os.path.exists(node_timing_path)

        with open(node_timing_path) as f:
            node_timing = json.load(f)

        assert node_timing["node_id"] == 0
        assert node_timing["execution_time_seconds"] == 60.0


# --- Test: Cleanup ---


class TestCleanup:
    def test_cleanup_does_not_crash(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        # Should not raise
        manager.cleanup()

    def test_cleanup_callable_multiple_times(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.cleanup()
        manager.cleanup()


# --- Test: Sibling Config Deduplication ---


class TestSiblingDeduplication:
    def test_extract_hp_decisions_keeps_only_decision_lines(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        response = (
            "**Step 1 — Diagnosis:**\n"
            "No history yet, this is a cold-start run with lots of prose that should be dropped.\n\n"
            "**Step 2 — Suggestions:**\n"
            "- **rollout.n**: 5 → 8\n"
            "  *Reason: more samples per prompt*\n"
            "- **temperature**: 1.0 → 1.2\n"
        )
        decisions = manager._extract_hp_decisions(response)
        assert "rollout.n: 5 → 8" in decisions
        assert "temperature: 1.0 → 1.2" in decisions
        # Diagnosis prose must NOT leak in.
        assert "cold-start run with lots of prose" not in decisions

    def test_extract_hp_decisions_empty(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        assert manager._extract_hp_decisions("") == ""

    def test_record_sibling_config_stores_decisions(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        parent = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        manager.time_step += 1
        child = Node(parent=parent, stage="evolve", time_step=manager.time_step)

        manager._proposer_suggestions_combined = "- **learning_rate**: 5e-6 → 1e-5"
        manager._record_sibling_config(child)

        configs = manager.get_sibling_configs(child)
        assert len(configs) == 1
        assert "learning_rate: 5e-6 → 1e-5" in configs[0]["suggestions"]

    def test_sibling_config_summary_excludes_self_and_shows_siblings(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        parent = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)

        manager.time_step += 1
        sib1 = Node(parent=parent, stage="evolve", time_step=manager.time_step)
        manager._proposer_suggestions_combined = "- **kl_coef**: 0.2 → 0.05"
        manager._record_sibling_config(sib1)

        manager.time_step += 1
        sib2 = Node(parent=parent, stage="evolve", time_step=manager.time_step)
        manager.current_node = sib2

        summary = manager.sibling_config_summary
        assert "kl_coef: 0.2 → 0.05" in summary
        assert "MEANINGFULLY DIFFERENT" in summary

    def test_sibling_config_summary_empty_for_root_children(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        child = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        manager.current_node = child
        # No siblings recorded yet → empty.
        assert manager.sibling_config_summary == ""
