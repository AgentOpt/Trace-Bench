from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
import hashlib
import json
import subprocess

from trace_bench.config import RunConfig, TaskConfig, TrainerConfig


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def _stable_hash(payload: Dict[str, Any], length: int = 8) -> str:
    data = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:length]


def compute_run_id(config_snapshot: Dict[str, Any], git_sha: Optional[str] = None) -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    payload = {"config": config_snapshot, "git": git_sha or _git_sha()}
    return f"{timestamp}-{_stable_hash(payload, 8)}"


def compute_job_id(task_id: str, trainer_id: str, resolved_kwargs: Dict[str, Any], seed: int) -> str:
    payload = {
        "task_id": task_id,
        "trainer_id": trainer_id,
        "resolved_kwargs": resolved_kwargs,
        "seed": seed,
    }
    return _stable_hash(payload, 12)


def task_suite(task_id: str) -> str:
    if ":" in task_id:
        return task_id.split(":", 1)[0]
    return "llm4ad"


def resolve_job_kwargs(task: TaskConfig, trainer: TrainerConfig, params: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "trainer_kwargs": dict(params),
        "optimizer": trainer.optimizer,
        "optimizer_kwargs": dict(trainer.optimizer_kwargs or {}),
        "guide": trainer.guide,
        "guide_kwargs": dict(trainer.guide_kwargs or {}),
        "logger": trainer.logger,
        "logger_kwargs": dict(trainer.logger_kwargs or {}),
        "eval_kwargs": dict(task.eval_kwargs or {}),
    }


@dataclass
class JobSpec:
    job_id: str
    task: TaskConfig
    trainer: TrainerConfig
    seed: int
    params: Dict[str, Any]
    resolved_kwargs: Dict[str, Any]

    @property
    def task_id(self) -> str:
        return self.task.id

    @property
    def trainer_id(self) -> str:
        return self.trainer.id

    @property
    def suite(self) -> str:
        return task_suite(self.task_id)


def expand_matrix(config: RunConfig) -> List[JobSpec]:
    jobs: List[JobSpec] = []
    for task in config.tasks:
        for trainer in config.trainers:
            variants = trainer.params_variants or [{}]
            for params in variants:
                for seed in config.seeds:
                    resolved = resolve_job_kwargs(task, trainer, params)
                    job_id = compute_job_id(task.id, trainer.id, resolved, seed)
                    jobs.append(
                        JobSpec(
                            job_id=job_id,
                            task=task,
                            trainer=trainer,
                            seed=seed,
                            params=params,
                            resolved_kwargs=resolved,
                        )
                    )
    return jobs
