"""Test TensorBoard logdir injection in the runner."""
from __future__ import annotations

from dataclasses import replace as dc_replace
from pathlib import Path

import pytest

from trace_bench.artifacts import JobArtifacts
from trace_bench.config import TaskConfig, TrainerConfig
from trace_bench.matrix import JobSpec
from trace_bench.runner import BenchRunner


def _make_job(logger: str = "TensorboardLogger", logger_kwargs: dict | None = None) -> JobSpec:
    """Create a minimal JobSpec for testing."""
    trainer = TrainerConfig(
        id="PrioritySearch",
        params_variants=[{"ps_steps": 1}],
        logger=logger,
        logger_kwargs=dict(logger_kwargs or {}),
    )
    task = TaskConfig(id="internal:numeric_param")
    return JobSpec(
        job_id="test-job-001",
        task=task,
        trainer=trainer,
        seed=123,
        params={"ps_steps": 1},
        resolved_kwargs={},
    )


def test_inject_tb_logdir_tensorboard_logger(tmp_path):
    """When logger is TensorboardLogger, logdir is injected."""
    job = _make_job(logger="TensorboardLogger")
    job_dir = tmp_path / "jobs" / "test-job-001"
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "tb").mkdir(exist_ok=True)
    artifacts = JobArtifacts(job_dir=job_dir)

    new_job = BenchRunner._inject_tb_logdir(job, artifacts)

    # Original job should NOT be mutated
    assert "logdir" not in (job.trainer.logger_kwargs or {})
    # New job should have logdir set
    assert new_job.trainer.logger_kwargs["logdir"] == str(job_dir / "tb")


def test_inject_tb_logdir_preserves_existing_kwargs(tmp_path):
    """Existing logger_kwargs are preserved alongside injected logdir."""
    job = _make_job(logger="TensorboardLogger", logger_kwargs={"flush_secs": 30})
    job_dir = tmp_path / "jobs" / "test-job-001"
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "tb").mkdir(exist_ok=True)
    artifacts = JobArtifacts(job_dir=job_dir)

    new_job = BenchRunner._inject_tb_logdir(job, artifacts)

    assert new_job.trainer.logger_kwargs["logdir"] == str(job_dir / "tb")
    assert new_job.trainer.logger_kwargs["flush_secs"] == 30


def test_inject_tb_logdir_skips_non_tensorboard(tmp_path):
    """When logger is not TensorBoard, job is returned unchanged."""
    job = _make_job(logger="ConsoleLogger")
    job_dir = tmp_path / "jobs" / "test-job-001"
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "tb").mkdir(exist_ok=True)
    artifacts = JobArtifacts(job_dir=job_dir)

    same_job = BenchRunner._inject_tb_logdir(job, artifacts)

    # Should be the exact same object (no copy needed)
    assert same_job is job


def test_inject_tb_logdir_none_logger(tmp_path):
    """When logger is None, job is returned unchanged."""
    job = _make_job(logger=None)
    job_dir = tmp_path / "jobs" / "test-job-001"
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "tb").mkdir(exist_ok=True)
    artifacts = JobArtifacts(job_dir=job_dir)

    same_job = BenchRunner._inject_tb_logdir(job, artifacts)
    assert same_job is job


def test_inject_tb_logdir_case_insensitive(tmp_path):
    """TensorBoard detection is case-insensitive."""
    job = _make_job(logger="tensorboardlogger")
    job_dir = tmp_path / "jobs" / "test-job-001"
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "tb").mkdir(exist_ok=True)
    artifacts = JobArtifacts(job_dir=job_dir)

    new_job = BenchRunner._inject_tb_logdir(job, artifacts)
    assert new_job.trainer.logger_kwargs["logdir"] == str(job_dir / "tb")
