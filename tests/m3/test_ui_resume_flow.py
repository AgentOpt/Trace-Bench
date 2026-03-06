"""Test the resume-from-UI flow: load snapshot config, preserve run_id, resume=auto."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _create_synthetic_run(runs_dir: Path, run_id: str) -> Path:
    """Create a minimal synthetic run directory with config snapshot."""
    run_dir = runs_dir / run_id
    meta_dir = run_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    config_yaml = (
        "mode: stub\n"
        "seeds: [123]\n"
        "max_workers: 1\n"
        "resume: auto\n"
        "tasks:\n"
        "  - id: internal:numeric_param\n"
        "trainers:\n"
        "  - id: PrioritySearch\n"
        "    params_variants:\n"
        "      - ps_steps: 1\n"
        "        ps_batches: 1\n"
    )
    (meta_dir / "config.snapshot.yaml").write_text(config_yaml, encoding="utf-8")

    summary = {"counts": {"ok": 1}, "total_jobs": 1}
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    return run_dir


def test_resume_preserves_run_id(tmp_path):
    """Resume with new_run_id=False should preserve the original run_id."""
    from trace_bench.ui.app import _resume_run

    runs_dir = tmp_path / "runs"
    _create_synthetic_run(runs_dir, "original-run-001")

    with patch("trace_bench.ui.app.BenchRunner") as MockRunner:
        mock_instance = MagicMock()
        mock_instance.run.return_value = MagicMock(run_id="original-run-001", results=[])
        MockRunner.return_value = mock_instance

        log, run_id = _resume_run(
            runs_dir_text=str(runs_dir),
            tasks_root=".",
            run_id="original-run-001",
            new_run_id=False,
        )

        # Verify BenchRunner was called with the right config
        call_args = MockRunner.call_args
        cfg = call_args[0][0]  # first positional arg
        assert cfg.run_id == "original-run-001"
        assert cfg.resume == "auto"
        assert "Resume complete" in log


def test_resume_with_new_run_id(tmp_path):
    """Resume with new_run_id=True generates a fresh run_id."""
    from trace_bench.ui.app import _resume_run

    runs_dir = tmp_path / "runs"
    _create_synthetic_run(runs_dir, "original-run-002")

    with patch("trace_bench.ui.app.BenchRunner") as MockRunner:
        mock_instance = MagicMock()
        mock_instance.run.return_value = MagicMock(run_id="new-run-xyz", results=[])
        MockRunner.return_value = mock_instance

        log, run_id = _resume_run(
            runs_dir_text=str(runs_dir),
            tasks_root=".",
            run_id="original-run-002",
            new_run_id=True,
        )

        call_args = MockRunner.call_args
        cfg = call_args[0][0]
        assert cfg.run_id is None  # None means BenchRunner will generate
        assert cfg.resume == "auto"


def test_resume_no_run_selected():
    """Resume with no run_id returns error."""
    from trace_bench.ui.app import _resume_run

    log, run_id = _resume_run(
        runs_dir_text="runs",
        tasks_root=".",
        run_id="",
        new_run_id=False,
    )
    assert "No run selected" in log
    assert run_id == ""


def test_resume_missing_config_snapshot(tmp_path):
    """Resume with missing config snapshot returns error."""
    from trace_bench.ui.app import _resume_run

    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "broken-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    # No config.snapshot.yaml created

    log, run_id = _resume_run(
        runs_dir_text=str(runs_dir),
        tasks_root=".",
        run_id="broken-run",
        new_run_id=False,
    )
    assert "No config snapshot" in log
    assert run_id == ""
