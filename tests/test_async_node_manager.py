"""
Tests for AsyncNodeManager — Top-K Weighted Deepening search strategy.

Uses mocks for all external dependencies (LLM agents, compute backends, etc.)
to test the scheduling logic, selection algorithm, and tree management in isolation.
"""

import asyncio
import os
import tempfile
import threading
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from autogluon.llmzero.managers.node_manager import Node

# --- Fixtures & Helpers ---


@dataclass
class MockConfig:
    """Minimal config that satisfies AsyncNodeManager init."""

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


def create_manager(input_dir, output_dir, config, max_concurrent=4):
    """Create AsyncNodeManager with agents mocked out."""
    with patch("autogluon.llmzero.managers.async_node_manager.AsyncNodeManager._init_agents"):
        from autogluon.llmzero.managers.async_node_manager import AsyncNodeManager

        manager = AsyncNodeManager(
            input_data_folder=input_dir,
            output_folder=output_dir,
            config=config,
            initial_user_input="Train a model",
            enable_per_iteration_instruction=False,
            max_concurrent=max_concurrent,
            selection_temperature=0.5,
        )
        # Set up mock agents
        manager.post_training_coder = MagicMock(return_value="#!/bin/bash\necho train")
        manager.execution_coder = MagicMock(return_value="#!/bin/bash\necho execute")
        manager.executer = MagicMock()
        manager.error_analyzer = MagicMock(return_value="error analysis")
        manager.proposers = [("unified", MagicMock())]
        manager.early_stoppers = []

        return manager


# --- Test: Initialization ---


class TestAsyncNodeManagerInit:
    def test_creates_virtual_root(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        assert manager.root_node.stage == "root"
        assert manager.root_node.time_step == -1

    def test_output_dir_created(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        assert os.path.isdir(manager.output_folder)

    def test_raises_on_missing_input(self, tmp_dirs, mock_config):
        _, output_dir = tmp_dirs
        with pytest.raises(FileNotFoundError):
            create_manager("/nonexistent/path", output_dir, mock_config)

    def test_max_concurrent_stored(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config, max_concurrent=8)
        assert manager.max_concurrent == 8

    def test_initialize_reads_description_files(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()
        assert "GRPO" in manager.description_file_contents
        assert manager.task_description == "Train a model"
        assert manager.data_prompt == manager.description_file_contents


# --- Test: Softmax ---


class TestSoftmax:
    def test_uniform_with_equal_scores(self):
        from autogluon.llmzero.managers.async_node_manager import AsyncNodeManager

        weights = AsyncNodeManager._softmax([1.0, 1.0, 1.0], temperature=1.0)
        assert all(abs(w - 1 / 3) < 0.001 for w in weights)

    def test_greedy_with_zero_temperature(self):
        from autogluon.llmzero.managers.async_node_manager import AsyncNodeManager

        weights = AsyncNodeManager._softmax([1.0, 2.0, 3.0], temperature=0.0)
        assert weights == [0.0, 0.0, 1.0]

    def test_higher_score_gets_more_weight(self):
        from autogluon.llmzero.managers.async_node_manager import AsyncNodeManager

        weights = AsyncNodeManager._softmax([1.0, 5.0], temperature=0.5)
        assert weights[1] > weights[0]

    def test_sums_to_one(self):
        from autogluon.llmzero.managers.async_node_manager import AsyncNodeManager

        weights = AsyncNodeManager._softmax([0.1, 0.5, 0.9, 1.2], temperature=0.3)
        assert abs(sum(weights) - 1.0) < 1e-6


# --- Test: Bootstrap ---


class TestBootstrap:
    def test_bootstrap_creates_n_children(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config, max_concurrent=4)
        manager.initialize()

        # Manually run bootstrap logic synchronously
        for i in range(manager.max_concurrent):
            manager.time_step += 1
            Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)

        assert len(manager.root_node.children) == 4

    def test_bootstrap_first_child_increments_time_step(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()
        manager.time_step += 1
        child = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        assert child.time_step == 0
        assert child.id == 0


# --- Test: Bootstrap Diversity (cold-start proposer) ---


class TestBootstrapDiversity:
    def test_first_child_is_baseline_no_proposer(self, tmp_dirs, mock_config):
        """Node 0 uses default config (no proposer call) and is recorded as a sibling."""
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config, max_concurrent=4)
        manager.initialize()

        proposer = MagicMock()
        manager.proposers = [("unified", proposer)]
        manager.post_training_coder = MagicMock(return_value="echo train")
        manager.execution_coder = MagicMock(return_value="echo exec")

        manager.time_step += 1
        node0 = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        manager._generate_code_for_node(node0, is_initial=True)

        # Baseline: proposer NOT called, suggestions empty.
        proposer.propose.assert_not_called()
        assert manager._proposer_suggestions_combined == ""
        # Baseline recorded under the root so later siblings can diverge.
        configs = manager._sibling_configs.get(manager.root_node.id, [])
        assert len(configs) == 1
        assert "aseline" in configs[0]["suggestions"]

    def test_later_bootstrap_children_call_cold_start_proposer(self, tmp_dirs, mock_config):
        """Bootstrap siblings 1..N-1 invoke the proposer with cold_start=True."""
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config, max_concurrent=4)
        manager.initialize()

        proposer = MagicMock()
        proposer.propose.return_value = "Try lr=2e-5, kl_coef=0.1"
        manager.proposers = [("unified", proposer)]
        manager.post_training_coder = MagicMock(return_value="echo train")
        manager.execution_coder = MagicMock(return_value="echo exec")

        manager.time_step += 1
        node0 = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        manager._generate_code_for_node(node0, is_initial=True)

        manager.time_step += 1
        node1 = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        manager._generate_code_for_node(node1, is_initial=False)

        proposer.propose.assert_called_once_with(cold_start=True)
        assert "lr=2e-5" in manager._proposer_suggestions_combined

    def test_bootstrap_children_get_distinct_suggestions(self, tmp_dirs, mock_config):
        """Each bootstrap sibling records its own distinct proposal for the next sibling."""
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config, max_concurrent=4)
        manager.initialize()

        # Proposer returns a different suggestion on each successive call.
        proposer = MagicMock()
        proposer.propose.side_effect = ["lr=1e-5", "lr=3e-5", "lr=5e-5"]
        manager.proposers = [("unified", proposer)]
        manager.post_training_coder = MagicMock(return_value="echo train")
        manager.execution_coder = MagicMock(return_value="echo exec")

        recorded = []
        for i in range(manager.max_concurrent):
            manager.time_step += 1
            node = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
            manager._generate_code_for_node(node, is_initial=(i == 0))
            recorded.append(manager._proposer_suggestions_combined)

        # Node 0 baseline empty; nodes 1..3 carry distinct proposals.
        assert recorded[0] == ""
        non_baseline = [r for r in recorded[1:]]
        assert all(r for r in non_baseline)  # all non-empty
        assert len(set(non_baseline)) == len(non_baseline)  # all distinct

        # Sibling list under root: 1 baseline + 3 proposed = 4 entries.
        configs = manager._sibling_configs.get(manager.root_node.id, [])
        assert len(configs) == manager.max_concurrent


# --- Test: Parent Selection ---


class TestParentSelection:
    def test_returns_none_when_no_completed_nodes(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()
        result = manager._select_parent_for_deepening()
        assert result is None

    def test_selects_from_top_k(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config, max_concurrent=4)
        manager.initialize()

        # Create nodes with scores
        nodes = []
        for i in range(5):
            manager.time_step += 1
            n = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
            n.validation_score = 0.1 * (i + 1)
            nodes.append(n)

        # K = ceil(sqrt(4)) = 2, so top-2 are nodes with scores 0.4, 0.5
        selected = [manager._select_parent_for_deepening() for _ in range(100)]
        # With epsilon=0.8, most selections should be from top-2
        top_2_count = sum(1 for s in selected if s.validation_score >= 0.4)
        assert top_2_count > 50  # At least half from top-2

    def test_skips_terminal_nodes(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        n1 = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        n1.validation_score = 0.9
        n1.is_terminal = True

        manager.time_step += 1
        n2 = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        n2.validation_score = 0.1

        result = manager._select_parent_for_deepening()
        assert result == n2

    def test_single_completed_node_always_selected(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        n = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        n.validation_score = 0.5

        assert manager._select_parent_for_deepening() == n

    def test_minimize_selects_lowest_raw_score_as_best(self, tmp_dirs, mock_config, monkeypatch):
        """For a minimize framework, the lowest raw validation score is the best node.

        max_concurrent=1 -> K=ceil(sqrt(1))=1, so top_k holds exactly the single
        best-normalized node. Force the exploit branch via random.random() == 0.0;
        softmax over a 1-element top_k deterministically returns that node.
        """
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config, max_concurrent=1)
        manager.initialize()

        # Lower raw score is better under minimize.
        raw_scores = [0.8, 0.2, 0.5]
        nodes = []
        for i, score in enumerate(raw_scores):
            manager.time_step += 1
            n = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
            n.validation_score = score
            nodes.append(n)
        lowest = nodes[1]  # validation_score == 0.2

        class _Min:
            @staticmethod
            def normalize_score(s):
                return -s

        manager._score_framework = _Min
        # Force the exploit branch (softmax over top-K) rather than random explore.
        monkeypatch.setattr("autogluon.llmzero.managers.async_node_manager.random.random", lambda: 0.0)

        selected = manager._select_parent_for_deepening()
        assert selected is lowest
        assert selected.validation_score == 0.2


# --- Test: Sibling Deduplication ---


class TestSiblingDeduplication:
    def test_records_sibling_config(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        parent = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)

        manager.time_step += 1
        child = Node(parent=parent, stage="evolve", time_step=manager.time_step)

        manager._proposer_suggestions_combined = "Try lr=0.001"
        manager._record_sibling_config(child)

        configs = manager.get_sibling_configs(child)
        assert len(configs) == 1
        assert "lr=0.001" in configs[0]["suggestions"]

    def test_sibling_config_summary_empty_for_root_children(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        child = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        manager.current_node = child

        assert manager.sibling_config_summary == ""

    def test_sibling_config_summary_shows_proposals(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        parent = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)

        manager.time_step += 1
        child1 = Node(parent=parent, stage="evolve", time_step=manager.time_step)
        manager._proposer_suggestions_combined = "Increase batch size"
        manager._record_sibling_config(child1)

        manager.time_step += 1
        child2 = Node(parent=parent, stage="evolve", time_step=manager.time_step)
        manager.current_node = child2

        summary = manager.sibling_config_summary
        assert "DIFFERENT" in summary
        assert "batch size" in summary


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

        manager._backpropagate(child, 0.5, True, False, False)

        assert child.visits == 1
        assert parent.visits == 1
        assert manager.root_node.visits == 1

    def test_backprop_failure(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        node = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)

        manager._backpropagate(node, None, False, True, False)

        assert node.failure_visits == 1


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

    def test_ancestor_terminal_when_all_children_terminal(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        child1 = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        manager.time_step += 1
        child2 = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)

        manager.mark_node_terminal(child1)
        assert not manager.root_node.is_terminal

        manager.mark_node_terminal(child2)
        assert manager.root_node.is_terminal


# --- Test: Completion Processing ---


class TestCompletionProcessing:
    def test_successful_completion_updates_best(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        node = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)

        result = (node, (0.85, "SUCCESS", "", "", "stdout", {}))
        manager._process_completion(node, result)

        assert manager._best_validation_score == 0.85
        assert manager._best_node == node
        assert node.is_successful
        assert node.validation_score == 0.85

    def test_failed_completion_records_error(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        node = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)

        result = (node, (None, "FAILURE", "OOM error", "error in stderr", "", {}))
        manager._process_completion(node, result)

        assert not node.is_successful
        assert "OOM error" in node.error_message
        assert len(manager._all_error_analyses) == 1

    def test_execution_time_transferred_from_result(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        node = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)

        # Simulate returned_node with execution_time set
        returned_node = Node(stage="evolve", time_step=0)
        returned_node.execution_time = 3600.5

        result = (returned_node, (0.5, "SUCCESS", "", "", "stdout", {}))
        manager._process_completion(node, result)

        assert node.execution_time == 3600.5

    def test_best_score_updates_only_on_improvement(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        n1 = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        result1 = (n1, (0.7, "SUCCESS", "", "", "", {}))
        manager._process_completion(n1, result1)

        manager.time_step += 1
        n2 = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        result2 = (n2, (0.5, "SUCCESS", "", "", "", {}))
        manager._process_completion(n2, result2)

        assert manager._best_validation_score == 0.7
        assert manager._best_node == n1


# --- Test: Root Relaunch on Failure ---


class TestRootRelaunch:
    """Async search has no debug path, so a failed direct child of the virtual root
    can never be deepened. Each such failure earns one credit to relaunch a fresh
    root child, keeping the root frontier alive instead of stalling the search."""

    def test_failed_root_child_increments_credit(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        node = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        result = (node, (None, "FAILURE", "OOM error", "stderr", "", {}))
        manager._process_completion(node, result)

        assert manager._failed_root_child_count == 1
        assert manager._root_relaunch_count == 0

    def test_failed_deep_child_does_not_earn_credit(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        # A successful root child, then a failing grandchild (depth 2, not a root child).
        manager.time_step += 1
        root_child = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        manager._process_completion(root_child, (root_child, (0.5, "SUCCESS", "", "", "", {})))

        manager.time_step += 1
        grandchild = Node(parent=root_child, stage="evolve", time_step=manager.time_step)
        manager._process_completion(grandchild, (grandchild, (None, "FAILURE", "boom", "err", "", {})))

        assert manager._failed_root_child_count == 0

    def test_fill_slots_relaunches_root_child_when_nothing_deepenable(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config, max_concurrent=2)
        manager.initialize()

        # Two bootstrap children both fail — no deepenable node remains.
        for _ in range(2):
            manager.time_step += 1
            node = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
            manager._process_completion(node, (node, (None, "FAILURE", "boom", "err", "", {})))

        assert manager._failed_root_child_count == 2
        assert manager._select_parent_for_deepening() is None

        # Stub _submit_job so fill_slots just registers the fresh node without a real job.
        submitted = []

        async def fake_submit(node, is_initial):
            submitted.append(node)
            manager._pending_tasks[object()] = node.id  # occupy a slot

        manager._submit_job = fake_submit

        asyncio.run(manager._fill_slots(max_iterations=100))

        # Both failure credits spent on fresh root children.
        assert manager._root_relaunch_count == 2
        assert len(submitted) == 2
        assert all(n.parent is manager.root_node for n in submitted)

    def test_relaunch_capped_by_failure_credits(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config, max_concurrent=4)
        manager.initialize()

        # One failed root child → exactly one relaunch credit.
        manager.time_step += 1
        node = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        manager._process_completion(node, (node, (None, "FAILURE", "boom", "err", "", {})))

        submitted = []

        async def fake_submit(node, is_initial):
            submitted.append(node)

        manager._submit_job = fake_submit

        asyncio.run(manager._fill_slots(max_iterations=100))

        # Only one relaunch despite open slots — capped by the single credit.
        assert manager._root_relaunch_count == 1
        assert len(submitted) == 1


# --- Test: Thread-Local Current Node ---


class TestThreadLocalCurrentNode:
    def test_event_loop_sees_main_current_node(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        node = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        manager.current_node = node

        assert manager.current_node == node

    def test_thread_sees_thread_local_node(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        main_node = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        manager.current_node = main_node

        manager.time_step += 1
        thread_node = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)

        seen_in_thread = [None]

        def worker():
            manager._thread_local.node = thread_node
            seen_in_thread[0] = manager.current_node
            manager._thread_local.node = None

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        assert seen_in_thread[0] == thread_node
        assert manager.current_node == main_node

    def test_thread_local_cleared_after_execution(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        node = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)

        seen_after = [None]

        def worker():
            manager._thread_local.node = node
            manager._thread_local.node = None
            seen_after[0] = manager.current_node

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        # After clearing thread-local, should fall back to main
        assert seen_after[0] == manager.root_node


# --- Test: Agent Compatibility Properties ---


class TestAgentCompatibility:
    def test_entry_point_script_property(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        node = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        node.entry_point_script = "#!/bin/bash\necho hello"
        manager.current_node = node

        assert manager.entry_point_script == "#!/bin/bash\necho hello"

    def test_previous_entry_point_script(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        parent = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        parent.entry_point_script = "parent script"

        manager.time_step += 1
        child = Node(parent=parent, stage="evolve", time_step=manager.time_step)
        manager.current_node = child

        assert manager.previous_entry_point_script == "parent script"

    def test_best_validation_score_default(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        assert manager.best_validation_score == 0.0

    def test_tutorial_prompt_returns_empty(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        assert manager.tutorial_prompt == ""

    def test_selected_tool_returns_empty(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        assert manager.selected_tool == ""

    def test_configure_env_delegates_to_config(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        assert manager.configure_env == mock_config.configure_env


# --- Test: Get All Nodes ---


class TestGetAllNodes:
    def test_returns_root_when_no_children(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        nodes = manager._get_all_nodes()
        assert len(nodes) == 1
        assert nodes[0] == manager.root_node

    def test_returns_all_nodes_in_tree(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)

        manager.time_step += 1
        n1 = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
        manager.time_step += 1
        n2 = Node(parent=n1, stage="evolve", time_step=manager.time_step)
        manager.time_step += 1
        n3 = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)

        all_nodes = manager._get_all_nodes()
        assert len(all_nodes) == 4  # root + 3


# --- Test: Checkpoint Resolution ---


class TestCheckpointResolution:
    def test_from_scratch_when_no_checkpoint_config(self, tmp_dirs, mock_config):
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


# --- Test: Async Search Loop ---


class TestAsyncSearchLoop:
    def test_search_stops_when_all_terminal(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config, max_concurrent=2)
        manager.initialize()

        # Simulate: bootstrap creates 2 nodes, both fail immediately
        async def mock_search():
            for i in range(2):
                manager.time_step += 1
                child = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)
                child.is_terminal = True

            # Now _pending_tasks is empty (we didn't submit real jobs)
            # The search should detect no in-flight and stop
            assert len(manager._pending_tasks) == 0

        asyncio.run(mock_search())


# --- Test: Error Result Handling ---


class TestErrorHandling:
    def test_make_error_result(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        node = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)

        result = manager._make_error_result(node, "Connection timeout")
        returned_node, sim_data = result
        assert returned_node == node
        assert sim_data[5] == "Connection timeout"  # stderr position
        assert sim_data[0] is None  # validation_score

    def test_process_completion_with_exception_result(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        manager = create_manager(input_dir, output_dir, mock_config)
        manager.initialize()

        manager.time_step += 1
        node = Node(parent=manager.root_node, stage="evolve", time_step=manager.time_step)

        # 7-tuple exception case
        result = (node, (None, False, True, False, "", "OOM killed", {}))
        manager._process_completion(node, result)

        assert not node.is_successful
        assert "OOM killed" in node.error_message


# --- Test: GlobalTokenTracker Thread Safety ---


class TestGlobalTokenTracker:
    def test_thread_safe_add_tokens(self):
        from autogluon.llmzero.llm.base_chat import GlobalTokenTracker

        tracker = GlobalTokenTracker()
        # Reset for test
        tracker.total_input_tokens = 0
        tracker.total_output_tokens = 0
        tracker.conversations = {}
        tracker.sessions = {}

        def add_many(prefix):
            for i in range(100):
                tracker.add_tokens(f"{prefix}_conv_{i}", f"{prefix}_session", 10, 5)

        threads = [threading.Thread(target=add_many, args=(f"t{i}",)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert tracker.total_input_tokens == 4000  # 4 threads * 100 * 10
        assert tracker.total_output_tokens == 2000  # 4 threads * 100 * 5

    def test_get_total_usage(self):
        from autogluon.llmzero.llm.base_chat import GlobalTokenTracker

        tracker = GlobalTokenTracker()
        tracker.total_input_tokens = 0
        tracker.total_output_tokens = 0
        tracker.conversations = {}
        tracker.sessions = {}

        tracker.add_tokens("conv1", "session1", 100, 50)
        usage = tracker.get_total_usage()

        assert usage["total"]["total_input_tokens"] == 100
        assert usage["total"]["total_output_tokens"] == 50
        assert usage["total"]["total_tokens"] == 150


# --- Test: NodeManager Simplification ---


class TestNodeManagerSimplified:
    def test_initialize_reads_files(self, tmp_dirs, mock_config):
        input_dir, output_dir = tmp_dirs
        with patch("autogluon.llmzero.managers.node_manager.NodeManager._init_agents"):
            from autogluon.llmzero.managers.node_manager import NodeManager

            manager = NodeManager(
                input_data_folder=input_dir,
                output_folder=output_dir,
                config=mock_config,
                initial_user_input="Train a model",
                enable_per_iteration_instruction=False,
            )
            manager.initialize()

            assert "GRPO" in manager.description_file_contents
            assert manager.task_description == "Train a model"

    def test_tutorial_prompt_empty(self, tmp_dirs, mock_config):
        with patch("autogluon.llmzero.managers.node_manager.NodeManager._init_agents"):
            from autogluon.llmzero.managers.node_manager import NodeManager

            input_dir, output_dir = tmp_dirs
            manager = NodeManager(
                input_data_folder=input_dir,
                output_folder=output_dir,
                config=mock_config,
                initial_user_input="Train",
                enable_per_iteration_instruction=False,
            )
            assert manager.tutorial_prompt == ""

    def test_selected_tool_empty(self, tmp_dirs, mock_config):
        with patch("autogluon.llmzero.managers.node_manager.NodeManager._init_agents"):
            from autogluon.llmzero.managers.node_manager import NodeManager

            input_dir, output_dir = tmp_dirs
            manager = NodeManager(
                input_data_folder=input_dir,
                output_folder=output_dir,
                config=mock_config,
                initial_user_input="Train",
                enable_per_iteration_instruction=False,
            )
            assert manager.selected_tool == ""
