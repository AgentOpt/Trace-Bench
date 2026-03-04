from __future__ import annotations

import json
from pathlib import Path

from trace_bench.ui.discovery import discover_runs


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_discover_runs_recurses_into_children(tmp_path: Path):
    root = tmp_path / "runs-root"
    run_dir = root / "nested" / "20260227-000000-abcd1234"
    (run_dir / "meta").mkdir(parents=True, exist_ok=True)
    (run_dir / "meta" / "config.snapshot.yaml").write_text("mode: stub\n", encoding="utf-8")
    _write_json(run_dir / "summary.json", {"counts": {"ok": 1, "failed": 0, "skipped": 0}, "total_jobs": 1})
    _write_json(run_dir / "meta" / "manifest.json", {"run_id": run_dir.name, "generated_at": "x", "jobs": []})

    records = discover_runs(root)
    assert len(records) == 1
    assert records[0].run_id == run_dir.name
    assert records[0].label == str(Path("nested") / run_dir.name)
