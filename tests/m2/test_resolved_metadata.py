"""M2 blocker: persist resolved optimizer/guide/logger metadata."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from trace_bench.config import RunConfig
from trace_bench.runner import BenchRunner


def _run_single_job(tmp_path: Path, *, optimizer: str | None = None) -> tuple[Path, str]:
    cfg = RunConfig.from_dict(
        {
            "tasks": [{"id": "internal:numeric_param"}],
            "trainers": [
                {
                    "id": "PrioritySearch",
                    "params_variants": [{"ps_steps": 1, "ps_batches": 1, "ps_candidates": 2, "ps_proposals": 2}],
                    "optimizer": optimizer,
                    "optimizer_kwargs": {},
                }
            ],
            "seeds": [123],
            "mode": "stub",
        }
    )
    cfg.runs_dir = str(tmp_path / "runs")
    summary = BenchRunner(cfg).run()
    run_dir = Path(cfg.runs_dir) / summary.run_id
    return run_dir, summary.results[0]["job_id"]


def test_resolved_metadata_is_persisted(tmp_path: Path):
    os.chdir(Path(__file__).resolve().parents[2])
    run_dir, job_id = _run_single_job(tmp_path)

    job_meta = json.loads((run_dir / "jobs" / job_id / "job_meta.json").read_text(encoding="utf-8"))
    results_json = json.loads((run_dir / "jobs" / job_id / "results.json").read_text(encoding="utf-8"))

    assert job_meta["resolved_optimizer"] == "OPROv2"
    assert job_meta["resolved_guide"] is not None
    assert job_meta["resolved_logger"] == "ConsoleLogger"
    assert isinstance(job_meta["resolved_optimizer_kwargs"], dict)
    assert "objective" in job_meta["resolved_optimizer_kwargs"]

    assert results_json["resolved_optimizer"] == "OPROv2"
    assert results_json["resolved_guide"] is not None
    assert results_json["resolved_logger"] == "ConsoleLogger"
    assert isinstance(results_json["resolved_optimizer_kwargs"], dict)
    assert "objective" in results_json["resolved_optimizer_kwargs"]

    with (run_dir / "results.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["resolved_optimizer"] == "OPROv2"
    parsed_opt_kwargs = json.loads(rows[0]["resolved_optimizer_kwargs"])
    assert isinstance(parsed_opt_kwargs, dict)
    assert "objective" in parsed_opt_kwargs


def test_explicit_optimizer_is_recorded_as_resolved(tmp_path: Path):
    os.chdir(Path(__file__).resolve().parents[2])
    run_dir, job_id = _run_single_job(tmp_path, optimizer="OPROv2")
    job_meta = json.loads((run_dir / "jobs" / job_id / "job_meta.json").read_text(encoding="utf-8"))
    assert job_meta["resolved_optimizer"] == "OPROv2"

