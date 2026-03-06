"""Guard M2 coverage config against accidental task regressions."""

from __future__ import annotations

from pathlib import Path

import yaml


def test_m2_coverage_config_includes_tsp_gls_2o():
    config_path = Path(__file__).resolve().parents[2] / "configs" / "m2_coverage.yaml"
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    task_ids = [task["id"] for task in data.get("tasks", [])]
    assert "llm4ad:optimization/tsp_gls_2O" in task_ids


def test_m2_coverage_config_task_count():
    config_path = Path(__file__).resolve().parents[2] / "configs" / "m2_coverage.yaml"
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tasks = data.get("tasks", [])
    assert len(tasks) == 29
