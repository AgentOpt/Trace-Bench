"""M2 blocker: task loader must support package-relative imports."""

from __future__ import annotations

import json
from pathlib import Path

from trace_bench.registry import load_task_module


def test_load_task_module_supports_relative_imports(tmp_path: Path):
    tasks_root = tmp_path / "benchmark_tasks"
    task_dir = tasks_root / "demo_rel_import"
    task_dir.mkdir(parents=True)

    (tasks_root / "index.json").write_text(
        json.dumps([{"key": "demo_rel_import", "module": "demo_rel_import"}]),
        encoding="utf-8",
    )
    (task_dir / "helper.py").write_text("VALUE = 7\n", encoding="utf-8")
    (task_dir / "__init__.py").write_text(
        "\n".join(
            [
                "from .helper import VALUE",
                "",
                "def build_trace_problem(**kwargs):",
                "    return {",
                "        'param': VALUE,",
                "        'guide': lambda x, y, z: (0.0, 'ok'),",
                "        'train_dataset': {'inputs': [0], 'infos': [0]},",
                "        'optimizer_kwargs': {},",
                "        'metadata': {'name': 'demo'},",
                "    }",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    mod = load_task_module("llm4ad:demo_rel_import", tasks_root)
    assert getattr(mod, "VALUE", None) == 7

