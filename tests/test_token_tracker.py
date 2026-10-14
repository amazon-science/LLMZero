import json
import tempfile
import threading

import pytest

from autogluon.llmzero.llm.base_chat import GlobalTokenTracker


@pytest.fixture(autouse=True)
def reset_tracker():
    """Reset singleton state before each test."""
    tracker = GlobalTokenTracker()
    tracker.total_input_tokens = 0
    tracker.total_output_tokens = 0
    tracker.conversations = {}
    tracker.sessions = {}
    yield tracker


class TestSingletonBehavior:
    def test_same_instance_returned(self):
        a = GlobalTokenTracker()
        b = GlobalTokenTracker()
        assert a is b

    def test_double_checked_locking_from_threads(self):
        instances = []

        def create():
            instances.append(GlobalTokenTracker())

        threads = [threading.Thread(target=create) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(inst is instances[0] for inst in instances)


class TestAddTokens:
    def test_per_conversation_tracking(self, reset_tracker):
        tracker = reset_tracker
        tracker.add_tokens("conv1", "sess1", 10, 5)
        tracker.add_tokens("conv2", "sess1", 20, 15)

        assert tracker.conversations["conv1"] == {"input_tokens": 10, "output_tokens": 5}
        assert tracker.conversations["conv2"] == {"input_tokens": 20, "output_tokens": 15}

    def test_per_session_tracking(self, reset_tracker):
        tracker = reset_tracker
        tracker.add_tokens("conv1", "sessA", 10, 5)
        tracker.add_tokens("conv2", "sessA", 20, 15)
        tracker.add_tokens("conv3", "sessB", 7, 3)

        assert tracker.sessions["sessA"] == {"input_tokens": 30, "output_tokens": 20}
        assert tracker.sessions["sessB"] == {"input_tokens": 7, "output_tokens": 3}

    def test_total_counts(self, reset_tracker):
        tracker = reset_tracker
        tracker.add_tokens("c1", "s1", 100, 50)
        tracker.add_tokens("c2", "s2", 200, 80)

        assert tracker.total_input_tokens == 300
        assert tracker.total_output_tokens == 130

    def test_accumulates_same_conversation(self, reset_tracker):
        tracker = reset_tracker
        tracker.add_tokens("conv1", "s1", 10, 5)
        tracker.add_tokens("conv1", "s1", 10, 5)

        assert tracker.conversations["conv1"] == {"input_tokens": 20, "output_tokens": 10}


class TestThreadSafety:
    def test_concurrent_adds_no_lost_updates(self, reset_tracker):
        tracker = reset_tracker
        num_threads = 8
        adds_per_thread = 1000

        def worker(thread_id):
            for _ in range(adds_per_thread):
                tracker.add_tokens(f"conv_{thread_id}", "shared_session", 1, 2)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert tracker.total_input_tokens == num_threads * adds_per_thread
        assert tracker.total_output_tokens == num_threads * adds_per_thread * 2

        for i in range(num_threads):
            assert tracker.conversations[f"conv_{i}"]["input_tokens"] == adds_per_thread
            assert tracker.conversations[f"conv_{i}"]["output_tokens"] == adds_per_thread * 2

        assert tracker.sessions["shared_session"]["input_tokens"] == num_threads * adds_per_thread
        assert tracker.sessions["shared_session"]["output_tokens"] == num_threads * adds_per_thread * 2


class TestGetConversationUsage:
    def test_returns_correct_breakdown(self, reset_tracker):
        tracker = reset_tracker
        tracker.add_tokens("conv1", "s1", 50, 30)

        usage = tracker.get_conversation_usage("conv1")
        assert usage["input_tokens"] == 50
        assert usage["output_tokens"] == 30
        assert usage["total_conversation_tokens"] == 80

    def test_unknown_conversation_returns_zeros(self, reset_tracker):
        tracker = reset_tracker
        usage = tracker.get_conversation_usage("nonexistent")

        assert usage == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


class TestGetTotalUsage:
    def test_aggregates_all(self, reset_tracker):
        tracker = reset_tracker
        tracker.add_tokens("c1", "s1", 10, 5)
        tracker.add_tokens("c2", "s2", 20, 15)

        usage = tracker.get_total_usage()

        assert usage["total"]["total_input_tokens"] == 30
        assert usage["total"]["total_output_tokens"] == 20
        assert usage["total"]["total_tokens"] == 50
        assert "c1" in usage["conversations"]
        assert "c2" in usage["conversations"]
        assert usage["conversations"]["c1"]["total_tokens"] == 15
        assert "s1" in usage["sessions"]
        assert "s2" in usage["sessions"]

    def test_save_path_writes_valid_json(self, reset_tracker):
        tracker = reset_tracker
        tracker.add_tokens("c1", "s1", 100, 50)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/subdir/usage.json"
            usage = tracker.get_total_usage(save_path=path)

            with open(path) as f:
                loaded = json.load(f)

            assert loaded == usage


class TestEdgeCases:
    def test_zero_tokens(self, reset_tracker):
        tracker = reset_tracker
        tracker.add_tokens("c1", "s1", 0, 0)

        assert tracker.total_input_tokens == 0
        assert tracker.total_output_tokens == 0
        assert tracker.conversations["c1"] == {"input_tokens": 0, "output_tokens": 0}

    def test_very_large_token_counts(self, reset_tracker):
        tracker = reset_tracker
        large = 10**12
        tracker.add_tokens("c1", "s1", large, large)

        assert tracker.total_input_tokens == large
        assert tracker.total_output_tokens == large

    def test_many_conversations(self, reset_tracker):
        tracker = reset_tracker
        n = 500
        for i in range(n):
            tracker.add_tokens(f"conv_{i}", f"sess_{i % 10}", 1, 1)

        assert tracker.total_input_tokens == n
        assert tracker.total_output_tokens == n
        assert len(tracker.conversations) == n
        assert len(tracker.sessions) == 10
