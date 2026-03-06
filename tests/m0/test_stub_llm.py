import pytest

from trace_bench.registry import load_task_bundle


def _skip_if_missing_deps(exc: Exception):
    msg = str(exc).lower()
    if "graphviz" in msg or "opto" in msg:
        pytest.skip(f"Optional dependency missing: {exc}")


def test_example_tasks_load():
    try:
        bundle = load_task_bundle("trace_examples:greeting_stub", "LLM4AD/benchmark_tasks")
    except Exception as exc:
        _skip_if_missing_deps(exc)
        raise
    assert {"param", "guide", "train_dataset", "optimizer_kwargs", "metadata"}.issubset(bundle.keys())

    try:
        bundle2 = load_task_bundle("trace_examples:train_single_node_stub", "LLM4AD/benchmark_tasks")
    except Exception as exc:
        _skip_if_missing_deps(exc)
        raise
    assert {"param", "guide", "train_dataset", "optimizer_kwargs", "metadata"}.issubset(bundle2.keys())
