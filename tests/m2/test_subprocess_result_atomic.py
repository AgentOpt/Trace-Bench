"""M2 blocker: subprocess timeout path should be spawn-based and robust."""

from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path

from trace_bench.config import RunConfig, TrainerConfig
from trace_bench.runner import BenchRunner, _subprocess_job_target, _trainer_config_to_dict


def test_subprocess_target_writes_traceback_on_failure(tmp_path: Path):
    trainer_dict = _trainer_config_to_dict(TrainerConfig(id="PrioritySearch", params_variants=[{}]))
    result_file = tmp_path / "result.json"

    _subprocess_job_target(
        task_id="llm4ad:does_not_exist",
        tasks_root=str(tmp_path / "benchmark_tasks"),
        trainer_dict=trainer_dict,
        params={},
        mode="stub",
        eval_kwargs={},
        result_file=str(result_file),
    )

    payload = json.loads(result_file.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert "Traceback (most recent call last)" in payload.get("feedback", "")


def test_subprocess_uses_spawn_context(tmp_path: Path, monkeypatch):
    os.chdir(Path(__file__).resolve().parents[2])
    called: list[str] = []
    real_get_context = multiprocessing.get_context

    def _patched_get_context(method: str):
        called.append(method)
        return real_get_context(method)

    monkeypatch.setattr("trace_bench.runner.multiprocessing.get_context", _patched_get_context)

    cfg = RunConfig.from_dict(
        {
            "tasks": [{"id": "internal:numeric_param"}],
            "trainers": [{"id": "PrioritySearch", "params_variants": [{"ps_steps": 1, "ps_batches": 1}]}],
            "seeds": [123],
            "mode": "stub",
        }
    )
    cfg.runs_dir = str(tmp_path / "runs")

    summary = BenchRunner(cfg, job_timeout=30.0).run()
    assert summary.results
    assert "spawn" in called

