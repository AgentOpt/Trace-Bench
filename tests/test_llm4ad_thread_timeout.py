"""Regression: LLM4AD evaluation must time out in worker threads too.

signal.alarm-based timeouts are silently skipped outside the main thread, which
is exactly where Trace trainers run rollouts (num_threads > 1). Before the
thread-timeout fallback, a hot-looping candidate stalled training forever.
"""
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "LLM4AD"))
from llm4ad_loader import _eval_with_thread_timeout  # noqa: E402

import pytest  # noqa: E402


def test_returns_value_under_timeout():
    assert _eval_with_thread_timeout(lambda: 42, timeout=2.0) == 42


def test_raises_timeout_instead_of_hanging():
    start = time.time()
    with pytest.raises(TimeoutError):
        _eval_with_thread_timeout(lambda: time.sleep(30), timeout=0.5)
    assert time.time() - start < 5.0  # returned promptly, no stall


def test_enforced_from_worker_thread():
    result = {}

    def worker():
        try:
            _eval_with_thread_timeout(lambda: time.sleep(30), timeout=0.5)
        except TimeoutError:
            result["timed_out"] = True

    t = threading.Thread(target=worker)
    t.start(); t.join(timeout=10)
    assert result.get("timed_out") is True
