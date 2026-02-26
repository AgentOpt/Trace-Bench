from __future__ import annotations

import os

from trace_bench.integrations.mlflow_client import log_job_result, log_run_end, log_run_start


def test_mlflow_noop_when_not_configured(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    ctx = log_run_start(tmp_path / "runs" / "r1", {}, {}, {})
    assert ctx is None
    log_job_result(ctx, {"job_id": "j"}, {"job_id": "j", "status": "ok"})
    log_run_end(ctx, {"counts": {"ok": 1}, "total_jobs": 1})


def test_mlflow_noop_when_mlflow_missing(tmp_path, monkeypatch):
    # Even if user sets tracking URI, missing mlflow must not crash.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///tmp/mlruns")
    # Ensure import fails (if installed in environment, still make safe by shadowing).
    monkeypatch.setitem(__import__("sys").modules, "mlflow", None)

    ctx = log_run_start(tmp_path / "runs" / "r1", {"mode": "stub"}, {}, {})
    assert ctx is None
    log_job_result(ctx, {"job_id": "j"}, {"job_id": "j", "status": "ok"})
    log_run_end(ctx, {"counts": {"ok": 1}, "total_jobs": 1})

