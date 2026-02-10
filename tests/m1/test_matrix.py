import csv
import json
from pathlib import Path

from trace_bench.config import RunConfig, load_config
from trace_bench.matrix import compute_job_id, expand_matrix
from trace_bench.runner import BenchRunner


def test_expand_matrix_counts():
    cfg = RunConfig.from_dict(
        {
            "tasks": [{"id": "internal:numeric_param"}, {"id": "internal:code_param"}],
            "trainers": [
                {"id": "PrioritySearch", "params_variants": [{}]},
                {"id": "GEPA-Base", "params_variants": [{}]},
            ],
            "seeds": [123],
        }
    )
    jobs = expand_matrix(cfg)
    assert len(jobs) == 4


def test_job_id_stable():
    job_id_1 = compute_job_id("internal:numeric_param", "PrioritySearch", {"ps_steps": 1}, 123)
    job_id_2 = compute_job_id("internal:numeric_param", "PrioritySearch", {"ps_steps": 1}, 123)
    assert job_id_1 == job_id_2


def test_matrix_smoke_e2e(tmp_path):
    """Run 2 tasks x 2 trainers x 1 seed = 4 jobs end-to-end and verify results."""
    cfg = load_config("configs/m1_matrix_smoke.yaml")
    cfg.runs_dir = str(tmp_path / "runs")
    cfg.mode = "stub"

    summary = BenchRunner(cfg).run()
    run_dir = Path(cfg.runs_dir) / summary.run_id

    # results.csv must have exactly 4 data rows
    results_csv = run_dir / "results.csv"
    assert results_csv.exists()
    with open(results_csv) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4, f"Expected 4 rows in results.csv, got {len(rows)}"

    # summary.json must aggregate 4 jobs
    summary_json = run_dir / "summary.json"
    assert summary_json.exists()
    summary_data = json.loads(summary_json.read_text())
    assert summary_data["total_jobs"] == 4
