import csv
import json
from pathlib import Path

from trace_bench.config import RunConfig
from trace_bench.runner import BenchRunner


def _run_stub(tmp_path: Path) -> Path:
    cfg = RunConfig.from_dict(
        {
            "mode": "stub",
            "seeds": [123],
            "tasks": [{"id": "internal:numeric_param"}],
            "trainers": [{"id": "PrioritySearch", "params_variants": [{"threads": 2}]}],
        }
    )
    cfg.runs_dir = str(tmp_path / "runs")
    summary = BenchRunner(cfg).run()
    return Path(cfg.runs_dir) / summary.run_id


def test_no_memory_addresses_in_artifacts(tmp_path):
    run_dir = _run_stub(tmp_path)
    for path in run_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in {".json", ".jsonl", ".csv"}:
            continue
        text = path.read_text(encoding="utf-8")
        assert "object at 0x" not in text


def test_structured_nested_fields_in_outputs(tmp_path):
    run_dir = _run_stub(tmp_path)
    job_dir = next((run_dir / "jobs").iterdir())

    meta = json.loads((job_dir / "job_meta.json").read_text(encoding="utf-8"))
    assert isinstance(meta["resolved_optimizer_kwargs"], dict)
    assert isinstance(meta["resolved_trainer_kwargs"], dict)

    results = json.loads((job_dir / "results.json").read_text(encoding="utf-8"))
    assert isinstance(results["resolved_optimizer_kwargs"], dict)
    assert isinstance(results["resolved_trainer_kwargs"], dict)

    event_lines = (job_dir / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert event_lines
    event = json.loads(event_lines[0])
    assert isinstance(event["resolved_optimizer_kwargs"], dict)
    assert isinstance(event["resolved_trainer_kwargs"], dict)

    with (run_dir / "results.csv").open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    parsed = json.loads(rows[0]["resolved_optimizer_kwargs"])
    assert isinstance(parsed, dict)

