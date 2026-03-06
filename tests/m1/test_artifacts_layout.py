from pathlib import Path

from trace_bench.config import load_config
from trace_bench.runner import BenchRunner


def test_artifacts_layout(tmp_path):
    cfg = load_config("configs/smoke.yaml")
    cfg.runs_dir = str(tmp_path / "runs")

    summary = BenchRunner(cfg).run()
    run_dir = Path(cfg.runs_dir) / summary.run_id

    assert (run_dir / "meta" / "config.snapshot.yaml").exists()
    assert (run_dir / "meta" / "env.json").exists()
    assert (run_dir / "meta" / "git.json").exists()
    assert (run_dir / "meta" / "manifest.json").exists()
    assert (run_dir / "results.csv").exists()
    assert (run_dir / "summary.json").exists()

    jobs_dir = run_dir / "jobs"
    job_dirs = [p for p in jobs_dir.iterdir() if p.is_dir()]
    assert job_dirs, "expected at least one job directory"
    job_dir = job_dirs[0]
    assert (job_dir / "job_meta.json").exists()
    assert (job_dir / "results.json").exists()
    assert (job_dir / "events.jsonl").exists()
    assert (job_dir / "tb").exists()
