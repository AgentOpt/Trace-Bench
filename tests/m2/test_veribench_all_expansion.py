from __future__ import annotations

from pathlib import Path
import sys

from trace_bench.config import TaskConfig
from trace_bench.registry import expand_special_tasks
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
            "TASK_LIST = ['task_alpha', 'task_beta']\n"
            "def build_trace_problem(task_name=None, **kwargs):\n"
            "    return {\n"
            "        'param': object(),\n"
            "        'guide': object(),\n"
            "        'train_dataset': {'inputs': [None], 'infos': [task_name]},\n"
            "        'optimizer_kwargs': {'objective': 'x', 'memory_size': 5},\n"
            "        'metadata': {'task': task_name},\n"
            "    }\n"
        ),
        encoding="utf-8",
    )


def test_expand_veribench_all_into_discovered_tasks(monkeypatch, tmp_path: Path):
    _clear_veribench_modules()
    root = tmp_path / "Veribench"
    _write_entrypoint(root)
    monkeypatch.setenv("TRACE_BENCH_VERIBENCH_ROOT", str(root))
    monkeypatch.delenv("TRACE_BENCH_VERIBENCH_ENTRYPOINT", raising=False)

    tasks = [
        TaskConfig(id="veribench:all", eval_kwargs={"timeout_seconds": 7}),
        TaskConfig(id="internal:numeric_param", eval_kwargs={}),
    ]
    expanded = expand_special_tasks(tasks, "LLM4AD/benchmark_tasks")
    ids = [task.id for task in expanded]

    assert "veribench:task_alpha" in ids
    assert "veribench:task_beta" in ids
    assert "internal:numeric_param" in ids
    for task in expanded:
        if task.id.startswith("veribench:"):
            assert task.eval_kwargs == {"timeout_seconds": 7}


def test_expand_veribench_all_falls_back_to_placeholder(monkeypatch):
    _clear_veribench_modules()
    monkeypatch.setenv("TRACE_BENCH_VERIBENCH_ROOT", "/path/does/not/exist")
    monkeypatch.delenv("TRACE_BENCH_VERIBENCH_ENTRYPOINT", raising=False)
    # Also block the HF dataset fallback
    monkeypatch.setattr(
        "trace_bench.veribench_adapter._discover_from_dataset",
        lambda: (_ for _ in ()).throw(NotImplementedError("mocked: no dataset")),
    )

    expanded = expand_special_tasks([TaskConfig(id="veribench:all", eval_kwargs={})], "LLM4AD/benchmark_tasks")
    assert [task.id for task in expanded] == ["veribench:smoke_placeholder"]
