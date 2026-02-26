from __future__ import annotations

import json
from pathlib import Path

from trace_bench.ui.discovery import discover_runs, load_job_details, load_run_summary


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_discover_runs_and_load_summary(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "20260219-000000-deadbeef"
    (run_dir / "meta").mkdir(parents=True, exist_ok=True)
    (run_dir / "jobs" / "job123").mkdir(parents=True, exist_ok=True)

    (run_dir / "meta" / "config.snapshot.yaml").write_text("mode: stub\n", encoding="utf-8")
    _write_json(run_dir / "meta" / "manifest.json", {"run_id": run_dir.name, "generated_at": "x", "jobs": []})
    _write_json(run_dir / "summary.json", {"counts": {"ok": 1, "failed": 0, "skipped": 0}, "total_jobs": 1})
    (run_dir / "results.csv").write_text(
        "run_id,job_id,task_id,suite,trainer_id,seed,status,score_initial,score_final,score_best,time_seconds,resolved_trainer_kwargs,resolved_optimizer_kwargs,eval_kwargs,feedback,tb_logdir\n"
        f"{run_dir.name},job123,internal:numeric_param,internal,PrioritySearch,123,ok,0,0,0,0.1,{{}},{{}},{{}},ok,jobs/job123/tb\n",
        encoding="utf-8",
    )
    _write_json(run_dir / "jobs" / "job123" / "job_meta.json", {"job_id": "job123", "status": "ok"})
    _write_json(run_dir / "jobs" / "job123" / "results.json", {"job_id": "job123", "status": "ok"})
    (run_dir / "jobs" / "job123" / "events.jsonl").write_text("{}", encoding="utf-8")

    runs = discover_runs(runs_dir)
    assert len(runs) == 1
    assert runs[0].run_id == run_dir.name

    summary = load_run_summary(run_dir)
    assert summary["summary"]["total_jobs"] == 1
    assert len(summary["results_rows"]) == 1

    job = load_job_details(run_dir, "job123")
    assert job["job_meta"]["status"] == "ok"


