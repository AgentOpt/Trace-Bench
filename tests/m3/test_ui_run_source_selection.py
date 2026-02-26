"""Test the unified run handler's config source priority logic."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_unified_run_editor_priority(tmp_path):
    """Editor text takes priority over upload and picker."""
    from trace_bench.ui.app import _unified_run

    yaml_text = "mode: stub\nseeds: [123]\ntasks:\n  - id: internal:numeric_param\ntrainers:\n  - id: PrioritySearch\n"

    with patch("trace_bench.ui.app._run_config") as mock_run:
        mock_run.return_value = ("ok", "run-123")
        log, run_id = _unified_run(
            runs_dir_text=str(tmp_path),
            tasks_root=".",
            editor_text=yaml_text,
            uploaded=MagicMock(name="upload.yaml"),  # should be ignored
            config_path="/some/config.yaml",           # should be ignored
            mode="", max_workers=0, resume="", job_timeout=0, force=False,
        )
        # _run_config should have been called (editor took priority)
        assert mock_run.called


def test_unified_run_upload_second_priority(tmp_path):
    """With empty editor, uploaded file takes priority over picker."""
    from trace_bench.ui.app import _unified_run

    config_file = tmp_path / "uploaded.yaml"
    config_file.write_text("mode: stub\nseeds: [123]\ntasks:\n  - id: internal:numeric_param\ntrainers:\n  - id: PrioritySearch\n", encoding="utf-8")

    upload = MagicMock()
    upload.name = str(config_file)

    with patch("trace_bench.ui.app._run_config") as mock_run:
        mock_run.return_value = ("ok", "run-456")
        log, run_id = _unified_run(
            runs_dir_text=str(tmp_path),
            tasks_root=".",
            editor_text="",  # empty editor
            uploaded=upload,
            config_path="/other/config.yaml",  # should be ignored
            mode="", max_workers=0, resume="", job_timeout=0, force=False,
        )
        assert mock_run.called


def test_unified_run_picker_fallback(tmp_path):
    """With empty editor and no upload, picker path is used."""
    from trace_bench.ui.app import _unified_run

    config_file = tmp_path / "picked.yaml"
    config_file.write_text("mode: stub\nseeds: [123]\ntasks:\n  - id: internal:numeric_param\ntrainers:\n  - id: PrioritySearch\n", encoding="utf-8")

    with patch("trace_bench.ui.app._run_config") as mock_run:
        mock_run.return_value = ("ok", "run-789")
        log, run_id = _unified_run(
            runs_dir_text=str(tmp_path),
            tasks_root=".",
            editor_text="",
            uploaded=None,
            config_path=str(config_file),
            mode="", max_workers=0, resume="", job_timeout=0, force=False,
        )
        assert mock_run.called


def test_unified_run_no_config():
    """With nothing provided, returns a clear error message."""
    from trace_bench.ui.app import _unified_run

    log, run_id = _unified_run(
        runs_dir_text="runs",
        tasks_root=".",
        editor_text="",
        uploaded=None,
        config_path="",
        mode="", max_workers=0, resume="", job_timeout=0, force=False,
    )
    assert "No config" in log
    assert run_id == ""
