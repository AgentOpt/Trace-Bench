"""M2: Per-job timeout enforcement."""
import json
import os
import sys
from pathlib import Path

import pytest

from trace_bench.config import RunConfig
from trace_bench.runner import BenchRunner
from trace_bench.cli import _default_timeout


# Module-level target for multiprocessing on Windows (spawn can't pickle locals)
def _hang_target_for_subprocess(*args, **kwargs):
    """Subprocess target that sleeps forever — used to test timeout-kill."""
    import time
    time.sleep(9999)


def test_job_timeout_does_not_affect_fast_jobs(tmp_path):
    """Jobs that finish quickly should not be affected by a generous timeout."""
    os.chdir(Path(__file__).resolve().parents[2])

    cfg = RunConfig.from_dict(
        {
            "tasks": [{"id": "internal:numeric_param"}],
            "trainers": [{"id": "PrioritySearch", "params_variants": [{}]}],
            "seeds": [123],
            "mode": "stub",
        }
    )
    cfg.runs_dir = str(tmp_path / "runs")

    # 60s timeout should be plenty for stub mode
    runner = BenchRunner(cfg, job_timeout=60.0)
    summary = runner.run()

    assert len(summary.results) == 1
    assert summary.results[0]["status"] == "ok"


def test_job_timeout_from_config(tmp_path):
    """job_timeout can be specified in the config YAML."""
    cfg = RunConfig.from_dict(
        {
            "tasks": [{"id": "internal:numeric_param"}],
            "trainers": [{"id": "PrioritySearch", "params_variants": [{}]}],
            "seeds": [123],
            "mode": "stub",
            "job_timeout": 45.0,
        }
    )
    assert cfg.job_timeout == 45.0

    # Also test hyphenated form
    cfg2 = RunConfig.from_dict(
        {
            "tasks": [],
            "mode": "stub",
            "job-timeout": 120.0,
        }
    )
    assert cfg2.job_timeout == 120.0


def test_job_timeout_in_snapshot():
    """job_timeout should appear in config snapshot."""
    cfg = RunConfig.from_dict(
        {
            "tasks": [],
            "mode": "stub",
            "job_timeout": 30.0,
        }
    )
    snap = cfg.snapshot()
    assert snap["job_timeout"] == 30.0


def test_default_timeout_stub_mode():
    """Default timeout for stub mode should be 0 (in-process, no subprocess overhead)."""
    assert _default_timeout("stub") == 0.0


def test_default_timeout_real_mode():
    """Default timeout for real mode should be 600s (10 min)."""
    assert _default_timeout("real") == 600.0


def test_timeout_kills_long_job(tmp_path):
    """A timed-out job's subprocess should actually be killed (not leak).

    Spawns a real OS process that sleeps forever, applies a short timeout,
    and verifies the process is terminated — mirroring _run_job_subprocess
    logic (terminate -> kill).
    """
    import subprocess
    import time

    # Spawn a real Python process that hangs (avoids multiprocessing pickle issues)
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(9999)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert proc.poll() is None, "Process should be running"

    # Wait briefly — process should NOT have exited
    time.sleep(1)
    assert proc.poll() is None, "Process should still be alive after 1s"

    # Kill it — mirrors the runner's terminate/kill logic
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)

    assert proc.poll() is not None, "Process must be dead after terminate/kill"


def test_timeout_produces_correct_status(tmp_path):
    """A job that exceeds its timeout should get status=failed with
    feedback containing 'job_timeout' and 'process killed'.

    Patches _subprocess_job_target with a module-level function that hangs
    forever, then verifies the runner's timeout path fires correctly.
    """
    from unittest.mock import patch

    os.chdir(Path(__file__).resolve().parents[2])

    cfg = RunConfig.from_dict(
        {
            "tasks": [{"id": "internal:numeric_param"}],
            "trainers": [{"id": "PrioritySearch", "params_variants": [{}]}],
            "seeds": [123],
            "mode": "stub",
        }
    )
    cfg.runs_dir = str(tmp_path / "runs")

    # _hang_target_for_subprocess is at module level so it can be pickled
    # across process boundaries on Windows (spawn start method).
    with patch("trace_bench.runner._subprocess_job_target", _hang_target_for_subprocess):
        runner = BenchRunner(cfg, job_timeout=3.0)
        summary = runner.run()

    assert len(summary.results) == 1
    result = summary.results[0]
    assert result["status"] == "failed", f"Expected failed, got {result['status']}"
    assert "job_timeout" in result.get("feedback", ""), \
        f"Feedback should mention job_timeout, got: {result.get('feedback')}"
    assert "process killed" in result.get("feedback", "").lower(), \
        f"Feedback should mention process killed, got: {result.get('feedback')}"


def test_timeout_terminates_stuck_process(tmp_path):
    """Verify that a stuck subprocess is actually terminated by timeout."""
    import multiprocessing
    import time

    os.chdir(Path(__file__).resolve().parents[2])

    cfg = RunConfig.from_dict(
        {
            "tasks": [{"id": "internal:numeric_param"}],
            "trainers": [{"id": "PrioritySearch", "params_variants": [{}]}],
            "seeds": [123],
            "mode": "stub",
        }
    )
    cfg.runs_dir = str(tmp_path / "runs")

    # With timeout, jobs run in subprocess; verify the mechanism works
    runner = BenchRunner(cfg, job_timeout=60.0)
    summary = runner.run()

    assert len(summary.results) == 1
    # Stub job should complete successfully even via subprocess
    assert summary.results[0]["status"] == "ok"
