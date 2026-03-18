from __future__ import annotations

from pathlib import Path
import sys

import pytest

from trace_bench.registry import load_task_bundle
from trace_bench.veribench_adapter import _clear_caches


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


def _write_entrypoint(root: Path) -> None:
    pkg = root / "my_processing_agents"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "optimize_veribench_agent.py").write_text(
        (
            "def list_tasks():\n"
            "    return ['logic_task']\n"
            "def build_trace_problem(task_name=None, **kwargs):\n"
            "    return {\n"
            "        'param': object(),\n"
            "        'guide': object(),\n"
            "        'train_dataset': {'inputs': [None], 'infos': [task_name]},\n"
            "        'optimizer_kwargs': {'objective': 'solve task', 'memory_size': 5},\n"
            "        'metadata': {'task': task_name},\n"
            "    }\n"
        ),
        encoding="utf-8",
    )


def test_veribench_load_task_bundle_via_adapter(monkeypatch, tmp_path: Path):
    _clear_veribench_modules()
    root = tmp_path / "Veribench"
    _write_entrypoint(root)
    monkeypatch.setenv("TRACE_BENCH_VERIBENCH_ROOT", str(root))
    monkeypatch.delenv("TRACE_BENCH_VERIBENCH_ENTRYPOINT", raising=False)

    bundle = load_task_bundle("veribench:logic_task", "benchmarks/LLM4AD/benchmark_tasks")
    required = {"param", "guide", "train_dataset", "optimizer_kwargs", "metadata"}
    assert required.issubset(bundle.keys())


def test_veribench_placeholder_still_skips(monkeypatch):
    _clear_veribench_modules()
    monkeypatch.setenv("TRACE_BENCH_VERIBENCH_ROOT", "/path/does/not/exist")
    monkeypatch.delenv("TRACE_BENCH_VERIBENCH_ENTRYPOINT", raising=False)

    with pytest.raises(NotImplementedError):
        load_task_bundle("veribench:smoke_placeholder", "benchmarks/LLM4AD/benchmark_tasks")
