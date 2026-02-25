"""Tests for HuggingFace dataset fallback in veribench adapter."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from trace_bench.veribench_adapter import (
    _clear_caches,
    _discover_from_dataset,
    _find_task_row,
    _task_name_from_row,
    build_bundle,
    discover_task_names,
)


def _clear_veribench_modules() -> None:
    for name in list(sys.modules.keys()):
        if name == "my_processing_agents" or name.startswith("my_processing_agents."):
            sys.modules.pop(name, None)
    sys.path[:] = [
        p
        for p in sys.path
        if not ((Path(p) / "my_processing_agents").is_dir() if p else False)
    ]
    _clear_caches()


# -- Fake dataset for mocking ------------------------------------------------

_FAKE_ROWS = [
    {"filename": "binary_search.py", "category": "cs_set",
     "code": "def binary_search(arr, target): pass",
     "user_query": "Translate binary_search to Lean 4",
     "system_prompt": "You are a Lean 4 expert."},
    {"filename": "bubble_sort.py", "category": "cs_set",
     "code": "def bubble_sort(arr): pass",
     "user_query": "Translate bubble_sort to Lean 4",
     "system_prompt": "You are a Lean 4 expert."},
    {"filename": "insertion_sort.py", "category": "cs_set",
     "code": "def insertion_sort(arr): pass",
     "user_query": "Translate insertion_sort to Lean 4",
     "system_prompt": "You are a Lean 4 expert."},
]


class _FakeDataset:
    """Minimal mock of a HuggingFace Dataset."""
    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    def __getitem__(self, idx):
        return self._rows[idx]


def _mock_load_dataset(*args, **kwargs):
    return _FakeDataset(_FAKE_ROWS)


# -- Tests -------------------------------------------------------------------


def test_task_name_from_row_uses_filename():
    row = {"filename": "binary_search.py"}
    assert _task_name_from_row(row, 0) == "binary_search"


def test_task_name_from_row_fallback_to_index():
    row = {"filename": None}
    assert _task_name_from_row(row, 7) == "task_7"


def test_discover_from_dataset_returns_names(monkeypatch):
    _clear_veribench_modules()
    monkeypatch.setattr("trace_bench.veribench_adapter.load_dataset", _mock_load_dataset, raising=False)
    # Inject the mock into the module namespace so _load_cached_dataset uses it
    import trace_bench.veribench_adapter as adapter
    monkeypatch.setattr(adapter, "_DATASET_CACHE", _FakeDataset(_FAKE_ROWS))
    monkeypatch.setattr(adapter, "_NAME_INDEX", {
        "binary_search": 0, "bubble_sort": 1, "insertion_sort": 2,
    })

    names = _discover_from_dataset()
    assert names == ["binary_search", "bubble_sort", "insertion_sort"]


def test_discover_task_names_falls_back_to_dataset(monkeypatch):
    """When entrypoint is unavailable, discovery falls back to HF dataset."""
    _clear_veribench_modules()
    monkeypatch.setenv("TRACE_BENCH_VERIBENCH_ROOT", "/path/does/not/exist")
    monkeypatch.delenv("TRACE_BENCH_VERIBENCH_ENTRYPOINT", raising=False)

    # Pre-populate the dataset cache with fake data
    import trace_bench.veribench_adapter as adapter
    monkeypatch.setattr(adapter, "_DATASET_CACHE", _FakeDataset(_FAKE_ROWS))
    monkeypatch.setattr(adapter, "_NAME_INDEX", {
        "binary_search": 0, "bubble_sort": 1, "insertion_sort": 2,
    })

    tasks = discover_task_names()
    assert "binary_search" in tasks
    assert "bubble_sort" in tasks
    assert len(tasks) == 3


def test_find_task_row_by_name(monkeypatch):
    _clear_veribench_modules()
    import trace_bench.veribench_adapter as adapter
    monkeypatch.setattr(adapter, "_DATASET_CACHE", _FakeDataset(_FAKE_ROWS))
    monkeypatch.setattr(adapter, "_NAME_INDEX", {
        "binary_search": 0, "bubble_sort": 1, "insertion_sort": 2,
    })

    row = _find_task_row("bubble_sort")
    assert row["filename"] == "bubble_sort.py"
    assert "code" in row


def test_find_task_row_by_index(monkeypatch):
    _clear_veribench_modules()
    import trace_bench.veribench_adapter as adapter
    monkeypatch.setattr(adapter, "_DATASET_CACHE", _FakeDataset(_FAKE_ROWS))
    monkeypatch.setattr(adapter, "_NAME_INDEX", {
        "binary_search": 0, "bubble_sort": 1, "insertion_sort": 2,
    })

    row = _find_task_row("task_1")
    assert row["filename"] == "bubble_sort.py"


def test_find_task_row_unknown_raises(monkeypatch):
    _clear_veribench_modules()
    import trace_bench.veribench_adapter as adapter
    monkeypatch.setattr(adapter, "_DATASET_CACHE", _FakeDataset(_FAKE_ROWS))
    monkeypatch.setattr(adapter, "_NAME_INDEX", {
        "binary_search": 0, "bubble_sort": 1, "insertion_sort": 2,
    })

    with pytest.raises(NotImplementedError, match="not found"):
        _find_task_row("nonexistent_task")


def test_build_bundle_from_dataset(monkeypatch):
    """build_bundle should construct a valid trace bundle from dataset when entrypoint is unavailable."""
    pytest.importorskip("opto", reason="opto (OpenTrace) required for bundle building")
    _clear_veribench_modules()
    monkeypatch.setenv("TRACE_BENCH_VERIBENCH_ROOT", "/path/does/not/exist")
    monkeypatch.delenv("TRACE_BENCH_VERIBENCH_ENTRYPOINT", raising=False)

    import trace_bench.veribench_adapter as adapter
    monkeypatch.setattr(adapter, "_DATASET_CACHE", _FakeDataset(_FAKE_ROWS))
    monkeypatch.setattr(adapter, "_NAME_INDEX", {
        "binary_search": 0, "bubble_sort": 1, "insertion_sort": 2,
    })

    bundle = build_bundle("binary_search")
    required = {"param", "guide", "train_dataset", "optimizer_kwargs", "metadata"}
    assert required.issubset(bundle.keys())
    assert bundle["metadata"]["benchmark"] == "veribench"
    assert bundle["metadata"]["task_name"] == "binary_search"
    assert "inputs" in bundle["train_dataset"]
    assert "infos" in bundle["train_dataset"]
    assert bundle["train_dataset"]["inputs"]
    assert bundle["train_dataset"]["infos"]
