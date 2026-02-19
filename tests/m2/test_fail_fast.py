"""M2: fail_fast stops execution after first failure."""
import json
import os
from pathlib import Path

import pytest

from trace_bench.config import RunConfig
from trace_bench.runner import BenchRunner


def _make_config(max_workers: int = 1, fail_fast: bool = True) -> RunConfig:
    return RunConfig.from_dict(
        {
            "tasks": [
                {"id": "internal:non_trainable"},  # This will fail (no trainable params)
                {"id": "internal:numeric_param"},
                {"id": "internal:code_param"},
            ],
            "trainers": [{"id": "PrioritySearch", "params_variants": [{}]}],
            "seeds": [123],
            "mode": "stub",
            "fail_fast": fail_fast,
            "max_workers": max_workers,
        }
    )


def test_fail_fast_sequential(tmp_path):
    """Sequential fail_fast: stops after first failure."""
    os.chdir(Path(__file__).resolve().parents[2])

    cfg = _make_config(max_workers=1, fail_fast=True)
    cfg.runs_dir = str(tmp_path / "runs")
    summary = BenchRunner(cfg).run()

    run_dir = Path(cfg.runs_dir) / summary.run_id
    manifest = json.loads((run_dir / "meta" / "manifest.json").read_text())

    # Should have some not_executed jobs
    executed = [j for j in manifest["jobs"] if j["status"] != "not_executed"]
    not_executed = [j for j in manifest["jobs"] if j["status"] == "not_executed"]

    # With fail_fast, non_trainable fails first -> remaining 2 should be not_executed
    assert len(not_executed) >= 1, "fail_fast should prevent some jobs from running"


def test_no_fail_fast_runs_all(tmp_path):
    """Without fail_fast, all jobs run regardless of failures."""
    os.chdir(Path(__file__).resolve().parents[2])

    cfg = _make_config(max_workers=1, fail_fast=False)
    cfg.runs_dir = str(tmp_path / "runs")
    summary = BenchRunner(cfg).run()

    # All 3 jobs should have run (no not_executed)
    assert len(summary.results) == 3

    run_dir = Path(cfg.runs_dir) / summary.run_id
    manifest = json.loads((run_dir / "meta" / "manifest.json").read_text())
    not_executed = [j for j in manifest["jobs"] if j["status"] == "not_executed"]
    assert len(not_executed) == 0, "Without fail_fast, all jobs should execute"


def test_fail_fast_parallel(tmp_path):
    """Parallel fail_fast: signals cancellation to remaining futures."""
    os.chdir(Path(__file__).resolve().parents[2])

    cfg = _make_config(max_workers=2, fail_fast=True)
    cfg.runs_dir = str(tmp_path / "runs")
    summary = BenchRunner(cfg).run()

    run_dir = Path(cfg.runs_dir) / summary.run_id
    manifest = json.loads((run_dir / "meta" / "manifest.json").read_text())

    # At least one job should have status "failed" (the non_trainable one)
    statuses = [j["status"] for j in manifest["jobs"]]
    assert "failed" in statuses, f"Expected at least one failure, got {statuses}"
