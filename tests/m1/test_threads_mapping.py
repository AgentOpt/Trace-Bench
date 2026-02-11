import csv
import json
from pathlib import Path

from trace_bench.config import RunConfig
from trace_bench.runner import BenchRunner


def test_threads_maps_to_num_threads(tmp_path):
    cfg = RunConfig.from_dict(
        {
            "mode": "stub",
            "seeds": [123],
            "tasks": [{"id": "internal:numeric_param"}],
            "trainers": [{"id": "PrioritySearch", "params_variants": [{"threads": 3}]}],
        }
    )
    cfg.runs_dir = str(tmp_path / "runs")

    summary = BenchRunner(cfg).run()
    run_dir = Path(cfg.runs_dir) / summary.run_id

    job_dirs = [p for p in (run_dir / "jobs").iterdir() if p.is_dir()]
    assert job_dirs, "expected at least one job directory"
    meta = json.loads((job_dirs[0] / "job_meta.json").read_text(encoding="utf-8"))
    assert meta["resolved_trainer_kwargs"]["num_threads"] == 3

    with (run_dir / "results.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows, "expected at least one results row"
    resolved = json.loads(rows[0]["resolved_trainer_kwargs"])
    assert resolved["num_threads"] == 3
