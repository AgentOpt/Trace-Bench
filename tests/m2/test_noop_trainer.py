from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from trace_bench.config import RunConfig
from trace_bench.runner import BenchRunner


def test_noop_trainer_runs_through_bench_runner() -> None:
    """NoOpTrainer should work when instantiated by the Trace-Bench runner."""
    repo_root = Path(__file__).resolve().parents[2]
    run_root = repo_root / "pytest_tmp_global" / f"noop-{uuid4().hex}" / "runs"
    cfg = RunConfig.from_dict(
        {
            "runs_dir": str(run_root),
            "mode": "stub",
            "resume": "none",
            "tasks": [{"id": "trace_examples:greeting_stub"}],
            "trainers": [{"id": "NoOpTrainer", "params_variants": [{"num_steps": 1}]}],
            "seeds": [123],
        }
    )

    try:
        summary = BenchRunner(cfg).run(force=True)
        assert len(summary.results) == 1
        assert summary.results[0]["status"] == "ok"
    finally:
        shutil.rmtree(run_root.parent, ignore_errors=True)
