from __future__ import annotations

from pathlib import Path
import sys

from trace_bench.veribench_adapter import discover_task_names, _clear_caches


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
            "    return ['lean_intro', 'lean_structures', 'lean_intro']\n"
        ),
        encoding="utf-8",
    )


def test_veribench_adapter_discovers_tasks(monkeypatch, tmp_path: Path):
    _clear_veribench_modules()
    root = tmp_path / "Veribench"
    _write_entrypoint(root)
    monkeypatch.setenv("TRACE_BENCH_VERIBENCH_ROOT", str(root))
    monkeypatch.delenv("TRACE_BENCH_VERIBENCH_ENTRYPOINT", raising=False)

    tasks = discover_task_names()
    assert tasks == ["lean_intro", "lean_structures"]
