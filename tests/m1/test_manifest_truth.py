import json
from pathlib import Path

from trace_bench.config import RunConfig
from trace_bench.runner import BenchRunner


def test_manifest_matches_job_meta(tmp_path):
    cfg = RunConfig.from_dict(
        {
            "mode": "stub",
            "seeds": [123],
            "tasks": [
                {"id": "internal:numeric_param"},
                {"id": "trace_examples:greeting_stub"},
            ],
            "trainers": [
                {"id": "PrioritySearch", "params_variants": [{"threads": 2}]},
                {"id": "GEPA-Base", "params_variants": [{"gepa_iters": 1}]},
            ],
        }
    )
    cfg.runs_dir = str(tmp_path / "runs")

    summary = BenchRunner(cfg).run()
    run_dir = Path(cfg.runs_dir) / summary.run_id
    manifest = json.loads((run_dir / "meta" / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["jobs"], "expected manifest jobs"
    for entry in manifest["jobs"]:
        if entry.get("status") == "not_executed":
            continue
        job_meta_path = run_dir / "jobs" / entry["job_id"] / "job_meta.json"
        assert job_meta_path.exists()
        job_meta = json.loads(job_meta_path.read_text(encoding="utf-8"))
        assert entry["raw_params"] == job_meta["raw_params"]
        assert entry["resolved_trainer_kwargs"] == job_meta["resolved_trainer_kwargs"]
        assert entry["resolved_optimizer_kwargs"] == job_meta["resolved_optimizer_kwargs"]
        assert entry["resolved_guide_kwargs"] == job_meta["resolved_guide_kwargs"]
        assert entry["resolved_logger_kwargs"] == job_meta["resolved_logger_kwargs"]
        assert entry["eval_kwargs"] == job_meta["eval_kwargs"]

