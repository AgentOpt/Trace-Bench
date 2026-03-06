"""M2: Parallel execution via ThreadPoolExecutor."""
import csv
import json
import os
from pathlib import Path

import pytest

from trace_bench.config import RunConfig
from trace_bench.runner import BenchRunner


def _make_config(max_workers: int, seeds=None) -> RunConfig:
    return RunConfig.from_dict(
        {
            "tasks": [
                {"id": "internal:numeric_param"},
                {"id": "internal:code_param"},
            ],
            "trainers": [
                {"id": "PrioritySearch", "params_variants": [{}]},
                {"id": "GEPA-Base", "params_variants": [{}]},
            ],
            "seeds": seeds or [123],
            "mode": "stub",
            "max_workers": max_workers,
        }
    )


def test_parallel_same_results_as_sequential(tmp_path):
    """Parallel execution (max_workers=2) produces the same set of jobs as sequential."""
    os.chdir(Path(__file__).resolve().parents[2])

    cfg_seq = _make_config(max_workers=1)
    cfg_seq.runs_dir = str(tmp_path / "seq")
    summary_seq = BenchRunner(cfg_seq).run()

    cfg_par = _make_config(max_workers=2)
    cfg_par.runs_dir = str(tmp_path / "par")
    summary_par = BenchRunner(cfg_par).run()

    # Same number of results
    assert len(summary_seq.results) == len(summary_par.results)

    # Same job IDs (order may differ)
    seq_ids = sorted(r["job_id"] for r in summary_seq.results)
    par_ids = sorted(r["job_id"] for r in summary_par.results)
    assert seq_ids == par_ids

    # Both produce results.csv with same number of rows
    seq_csv = Path(cfg_seq.runs_dir) / summary_seq.run_id / "results.csv"
    par_csv = Path(cfg_par.runs_dir) / summary_par.run_id / "results.csv"
    with open(seq_csv) as f:
        seq_rows = list(csv.DictReader(f))
    with open(par_csv) as f:
        par_rows = list(csv.DictReader(f))
    assert len(seq_rows) == len(par_rows) == 4


def test_parallel_artifacts_complete(tmp_path):
    """Each job has complete artifacts even when run in parallel."""
    os.chdir(Path(__file__).resolve().parents[2])

    cfg = _make_config(max_workers=4)
    cfg.runs_dir = str(tmp_path / "runs")
    summary = BenchRunner(cfg).run()

    run_dir = Path(cfg.runs_dir) / summary.run_id
    manifest = json.loads((run_dir / "meta" / "manifest.json").read_text())
    assert len(manifest["jobs"]) == 4

    for job_entry in manifest["jobs"]:
        job_dir = run_dir / "jobs" / job_entry["job_id"]
        assert job_dir.exists(), f"Missing job dir for {job_entry['job_id']}"
        assert (job_dir / "job_meta.json").exists()
        assert (job_dir / "results.json").exists()
        assert (job_dir / "events.jsonl").exists()


def test_parallel_max_workers_4(tmp_path):
    """max_workers=4 with 4 jobs works correctly."""
    os.chdir(Path(__file__).resolve().parents[2])

    cfg = _make_config(max_workers=4)
    cfg.runs_dir = str(tmp_path / "runs")
    summary = BenchRunner(cfg).run()
    assert len(summary.results) == 4

    statuses = {r["status"] for r in summary.results}
    # All should be ok or failed (not missing)
    assert statuses.issubset({"ok", "failed", "skipped"})
