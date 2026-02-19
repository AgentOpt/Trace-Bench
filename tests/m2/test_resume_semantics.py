"""M2: Skip-existing / resume semantics."""
import json
import os
from pathlib import Path

import pytest

from trace_bench.config import RunConfig
from trace_bench.runner import BenchRunner


def _make_config(resume: str = "auto") -> RunConfig:
    return RunConfig.from_dict(
        {
            "tasks": [{"id": "internal:numeric_param"}, {"id": "internal:code_param"}],
            "trainers": [{"id": "PrioritySearch", "params_variants": [{}]}],
            "seeds": [123],
            "mode": "stub",
            "resume": resume,
        }
    )


def test_resume_auto_is_default():
    """Default resume mode should be 'auto'."""
    cfg = RunConfig.from_dict({"tasks": [], "mode": "stub"})
    assert cfg.resume == "auto"


def test_resume_skips_completed_jobs(tmp_path):
    """Second run with same config reuses completed jobs (resume=auto)."""
    os.chdir(Path(__file__).resolve().parents[2])

    cfg1 = _make_config()
    cfg1.runs_dir = str(tmp_path / "runs")
    summary1 = BenchRunner(cfg1).run()

    run_dir = Path(cfg1.runs_dir) / summary1.run_id
    assert run_dir.exists()

    # Run again with same run_id -> should reuse
    cfg2 = _make_config()
    cfg2.runs_dir = str(tmp_path / "runs")
    cfg2.run_id = summary1.run_id
    summary2 = BenchRunner(cfg2).run()

    assert summary2.run_id == summary1.run_id
    assert len(summary2.results) == len(summary1.results)

    # Manifest should show "reused" status for all jobs
    manifest = json.loads((run_dir / "meta" / "manifest.json").read_text())
    statuses = [j.get("status") for j in manifest["jobs"]]
    assert all(s == "reused" for s in statuses), f"Expected all reused, got {statuses}"


def test_resume_reruns_failed_jobs(tmp_path):
    """Failed jobs get re-run on resume=auto, not skipped."""
    os.chdir(Path(__file__).resolve().parents[2])

    cfg1 = _make_config()
    cfg1.runs_dir = str(tmp_path / "runs")
    summary1 = BenchRunner(cfg1).run()

    run_dir = Path(cfg1.runs_dir) / summary1.run_id

    # Corrupt one job's job_meta.json AND results.json to simulate failure
    jobs_dir = run_dir / "jobs"
    first_job_id = summary1.results[0]["job_id"]

    meta_path = jobs_dir / first_job_id / "job_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta["status"] = "failed"
        meta_path.write_text(json.dumps(meta))

    results_path = jobs_dir / first_job_id / "results.json"
    existing = json.loads(results_path.read_text())
    existing["status"] = "failed"
    results_path.write_text(json.dumps(existing))

    # Run again -> should re-run the failed job
    cfg2 = _make_config()
    cfg2.runs_dir = str(tmp_path / "runs")
    cfg2.run_id = summary1.run_id
    summary2 = BenchRunner(cfg2).run()

    assert len(summary2.results) == 2

    # At least one should NOT be "reused" (the failed job was re-run)
    manifest = json.loads((run_dir / "meta" / "manifest.json").read_text())
    statuses = [j.get("status") for j in manifest["jobs"]]
    assert any(s != "reused" for s in statuses), f"Expected at least one non-reused, got {statuses}"


def test_force_reruns_all(tmp_path):
    """force=True re-runs all jobs even if they completed."""
    os.chdir(Path(__file__).resolve().parents[2])

    cfg1 = _make_config()
    cfg1.runs_dir = str(tmp_path / "runs")
    summary1 = BenchRunner(cfg1).run()
    run_dir = Path(cfg1.runs_dir) / summary1.run_id

    cfg2 = _make_config()
    cfg2.runs_dir = str(tmp_path / "runs")
    cfg2.run_id = summary1.run_id
    summary2 = BenchRunner(cfg2).run(force=True)

    assert len(summary2.results) == 2

    # No jobs should be "reused" since force=True
    manifest = json.loads((run_dir / "meta" / "manifest.json").read_text())
    statuses = [j.get("status") for j in manifest["jobs"]]
    assert all(s != "reused" for s in statuses), f"Expected no reused, got {statuses}"


def test_resume_none_reruns_all(tmp_path):
    """resume='none' re-runs everything, same as force=True."""
    os.chdir(Path(__file__).resolve().parents[2])

    cfg1 = _make_config(resume="auto")
    cfg1.runs_dir = str(tmp_path / "runs")
    summary1 = BenchRunner(cfg1).run()
    run_dir = Path(cfg1.runs_dir) / summary1.run_id

    cfg2 = _make_config(resume="none")
    cfg2.runs_dir = str(tmp_path / "runs")
    cfg2.run_id = summary1.run_id
    summary2 = BenchRunner(cfg2).run()

    assert len(summary2.results) == 2

    manifest = json.loads((run_dir / "meta" / "manifest.json").read_text())
    statuses = [j.get("status") for j in manifest["jobs"]]
    assert all(s != "reused" for s in statuses), f"Expected no reused with resume=none, got {statuses}"


def test_resume_failed_only_reruns_failed(tmp_path):
    """resume='failed' only re-runs failed jobs, skips OK and never-run."""
    os.chdir(Path(__file__).resolve().parents[2])

    cfg1 = _make_config()
    cfg1.runs_dir = str(tmp_path / "runs")
    summary1 = BenchRunner(cfg1).run()
    run_dir = Path(cfg1.runs_dir) / summary1.run_id

    # Mark first job as failed in job_meta.json
    jobs_dir = run_dir / "jobs"
    first_job_id = summary1.results[0]["job_id"]

    meta_path = jobs_dir / first_job_id / "job_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta["status"] = "failed"
        meta_path.write_text(json.dumps(meta))

    results_path = jobs_dir / first_job_id / "results.json"
    existing = json.loads(results_path.read_text())
    existing["status"] = "failed"
    results_path.write_text(json.dumps(existing))

    # Run with resume="failed" -> only the failed job should re-run
    cfg2 = _make_config(resume="failed")
    cfg2.runs_dir = str(tmp_path / "runs")
    cfg2.run_id = summary1.run_id
    summary2 = BenchRunner(cfg2).run()

    manifest = json.loads((run_dir / "meta" / "manifest.json").read_text())
    statuses = {j["job_id"]: j["status"] for j in manifest["jobs"]}

    # First job (was failed) should have been re-run (ok or failed, not reused)
    assert statuses[first_job_id] != "reused", \
        f"Failed job should have been re-run, got status={statuses[first_job_id]}"

    # Second job (was OK) should be reused
    second_job_id = summary1.results[1]["job_id"]
    assert statuses[second_job_id] == "reused", \
        f"OK job should be reused with resume=failed, got status={statuses[second_job_id]}"


def test_resume_reads_job_meta_json(tmp_path):
    """Status check should read job_meta.json (primary), not just results.json."""
    os.chdir(Path(__file__).resolve().parents[2])

    cfg1 = _make_config()
    cfg1.runs_dir = str(tmp_path / "runs")
    summary1 = BenchRunner(cfg1).run()
    run_dir = Path(cfg1.runs_dir) / summary1.run_id

    # Verify job_meta.json exists for each job
    jobs_dir = run_dir / "jobs"
    for result in summary1.results:
        job_id = result["job_id"]
        meta_path = jobs_dir / job_id / "job_meta.json"
        assert meta_path.exists(), f"job_meta.json should exist for {job_id}"
        meta = json.loads(meta_path.read_text())
        assert "status" in meta, f"job_meta.json should have status field for {job_id}"

    # Now manipulate: set job_meta.json status=failed but leave results.json as ok
    first_job_id = summary1.results[0]["job_id"]
    meta_path = jobs_dir / first_job_id / "job_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["status"] = "failed"
    meta_path.write_text(json.dumps(meta))
    # results.json still says "ok"

    # Resume=auto should re-run this job because job_meta.json says "failed"
    cfg2 = _make_config(resume="auto")
    cfg2.runs_dir = str(tmp_path / "runs")
    cfg2.run_id = summary1.run_id
    summary2 = BenchRunner(cfg2).run()

    manifest = json.loads((run_dir / "meta" / "manifest.json").read_text())
    statuses = {j["job_id"]: j["status"] for j in manifest["jobs"]}
    assert statuses[first_job_id] != "reused", \
        "job_meta.json says failed -> should re-run, not reuse"


def test_resume_failed_skips_never_run(tmp_path):
    """resume='failed' should skip never-run jobs entirely (not execute them)."""
    os.chdir(Path(__file__).resolve().parents[2])

    # Run only one task first
    cfg1 = RunConfig.from_dict(
        {
            "tasks": [{"id": "internal:numeric_param"}],
            "trainers": [{"id": "PrioritySearch", "params_variants": [{}]}],
            "seeds": [123],
            "mode": "stub",
        }
    )
    cfg1.runs_dir = str(tmp_path / "runs")
    summary1 = BenchRunner(cfg1).run()
    run_dir = Path(cfg1.runs_dir) / summary1.run_id

    # Now configure with TWO tasks but same run_id + resume=failed.
    # The second task (code_param) was never run before.
    cfg2 = RunConfig.from_dict(
        {
            "tasks": [{"id": "internal:numeric_param"}, {"id": "internal:code_param"}],
            "trainers": [{"id": "PrioritySearch", "params_variants": [{}]}],
            "seeds": [123],
            "mode": "stub",
            "resume": "failed",
        }
    )
    cfg2.runs_dir = str(tmp_path / "runs")
    cfg2.run_id = summary1.run_id
    summary2 = BenchRunner(cfg2).run()

    manifest = json.loads((run_dir / "meta" / "manifest.json").read_text())
    manifest_statuses = {j["job_id"]: j["status"] for j in manifest["jobs"]}

    # The first task (OK) should be reused
    first_job_id = summary1.results[0]["job_id"]
    assert manifest_statuses[first_job_id] == "reused", \
        f"OK job should be reused with resume=failed, got {manifest_statuses[first_job_id]}"

    # The second task (code_param) was never run before.
    # With resume=failed it must NOT be executed — only previously-failed jobs
    # get re-run.  We verify two things:
    #   1. It must not appear in summary2.results (no execution happened).
    #   2. Its job directory must not contain a results.json produced by run 2.

    run1_job_ids = {r["job_id"] for r in summary1.results}
    run2_executed_ids = {r["job_id"] for r in summary2.results}
    never_run_ids = set(manifest_statuses.keys()) - run1_job_ids

    assert len(never_run_ids) > 0, \
        "Test setup error: expected at least one never-run job in the matrix"

    for job_id in never_run_ids:
        # Must NOT appear in the execution results
        assert job_id not in run2_executed_ids, \
            f"Never-run job {job_id} was executed (found in results) — " \
            f"resume=failed should skip never-run jobs"

        # Must have status "not_executed" in manifest (not "ok"/"failed"/"reused")
        status = manifest_statuses[job_id]
        assert status == "not_executed", \
            f"Never-run job {job_id} has status '{status}' in manifest — " \
            f"resume=failed should skip never-run jobs (expected 'not_executed')"
