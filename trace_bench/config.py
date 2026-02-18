from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import uuid


_LLM4AD_KNOBS = {
    "threads",
    "num_threads",
    "optimizer_kwargs",
    "eval_kwargs",
    "ps_steps",
    "ps_batches",
    "ps_candidates",
    "ps_proposals",
    "ps_mem_update",
    "gepa_iters",
    "gepa_train_bs",
    "gepa_merge_every",
    "gepa_pareto_subset",
}


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_yaml_or_json(path: Path) -> Dict[str, Any]:
    text = _load_text(path)
    # Prefer YAML if available
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(text)
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ValueError("Config must be a mapping at top-level")
        return data
    except Exception:
        # Fallback to JSON for environments without PyYAML
        try:
            data = json.loads(text)
            if not isinstance(data, dict):
                raise ValueError("Config must be a mapping at top-level")
            return data
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Failed to parse config {path}. Install PyYAML or use JSON syntax. Error: {exc}"
            )


def _as_dict(value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return dict(value or {})


def _normalize_key(key: str) -> str:
    return key.replace("-", "_")


def _extract_llm4ad_knobs(data: Dict[str, Any]) -> Dict[str, Any]:
    knobs: Dict[str, Any] = {}
    for raw_key, value in data.items():
        key = _normalize_key(raw_key)
        if key in _LLM4AD_KNOBS:
            knobs[key] = value
    return knobs


@dataclass
class TaskConfig:
    id: str
    eval_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainerConfig:
    id: str
    params_variants: List[Dict[str, Any]] = field(default_factory=list)
    optimizer: Optional[str] = None
    optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    guide: Optional[str] = None
    guide_kwargs: Dict[str, Any] = field(default_factory=dict)
    logger: Optional[str] = None
    logger_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunConfig:
    run_id: Optional[str] = None
    runs_dir: str = "runs"
    mode: str = "stub"
    seeds: List[int] = field(default_factory=lambda: [123])
    max_workers: int = 1
    fail_fast: bool = False
    resume: str = "auto"  # "auto" | "failed" | "none"
    job_timeout: Optional[float] = None  # per-job timeout in seconds
    tasks: List[TaskConfig] = field(default_factory=list)
    trainers: List[TrainerConfig] = field(default_factory=list)
    eval_kwargs: Dict[str, Any] = field(default_factory=dict)
    trainer_kwargs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunConfig":
        runs_dir = data.get("runs_dir", data.get("runs_root", "runs"))
        mode = data.get("mode", "stub")
        seeds = data.get("seeds")
        if seeds is None:
            seed = int(data.get("seed", 123))
            seeds = [seed]
        else:
            seeds = [int(x) for x in (seeds or [])] or [123]

        if "max_workers" in data:
            max_workers = data.get("max_workers")
        else:
            max_workers = data.get("n_concurrent", data.get("n-concurrent", 1))
        max_workers = int(max_workers)
        fail_fast = bool(data.get("fail_fast", False))

        resume = data.get("resume", "auto")
        if resume not in ("auto", "failed", "none"):
            raise ValueError(f"Invalid resume mode: {resume!r}. Must be auto|failed|none")

        job_timeout_raw = data.get("job_timeout", data.get("job-timeout"))
        job_timeout = float(job_timeout_raw) if job_timeout_raw is not None else None

        default_eval = _as_dict(data.get("eval_kwargs"))
        default_trainer_kwargs = _as_dict(data.get("trainer_kwargs"))
        default_trainer_kwargs.update(_extract_llm4ad_knobs(data))

        tasks: List[TaskConfig] = []
        for item in list(data.get("tasks", []) or []):
            if isinstance(item, str):
                tasks.append(TaskConfig(id=item, eval_kwargs=dict(default_eval)))
            elif isinstance(item, dict):
                task_id = item.get("id") or item.get("key") or item.get("task")
                if not task_id:
                    raise ValueError(f"Task entry missing id: {item}")
                eval_kwargs = dict(default_eval)
                eval_kwargs.update(_as_dict(item.get("eval_kwargs")))
                tasks.append(TaskConfig(id=str(task_id), eval_kwargs=eval_kwargs))
            else:
                raise ValueError(f"Unsupported task entry: {item}")

        trainers: List[TrainerConfig] = []
        for item in list(data.get("trainers", []) or []):
            if isinstance(item, str):
                params_variants = [dict(default_trainer_kwargs)]
                trainers.append(TrainerConfig(id=item, params_variants=params_variants))
                continue
            if not isinstance(item, dict):
                raise ValueError(f"Unsupported trainer entry: {item}")

            trainer_id = item.get("id") or item.get("name") or item.get("trainer") or item.get("key")
            if not trainer_id:
                raise ValueError(f"Trainer entry missing id: {item}")

            params_variants = item.get("params_variants")
            if params_variants is None:
                params = item.get("params") or item.get("trainer_kwargs") or {}
                params_variants = [params]
            normalized_variants: List[Dict[str, Any]] = []
            for variant in list(params_variants or [{}]):
                merged = dict(default_trainer_kwargs)
                merged.update(_extract_llm4ad_knobs(item))
                merged.update(dict(variant or {}))
                normalized_variants.append(merged)

            trainers.append(
                TrainerConfig(
                    id=str(trainer_id),
                    params_variants=normalized_variants,
                    optimizer=item.get("optimizer"),
                    optimizer_kwargs=_as_dict(item.get("optimizer_kwargs")),
                    guide=item.get("guide"),
                    guide_kwargs=_as_dict(item.get("guide_kwargs")),
                    logger=item.get("logger"),
                    logger_kwargs=_as_dict(item.get("logger_kwargs")),
                )
            )

        if not trainers:
            trainers = [TrainerConfig(id="PrioritySearch", params_variants=[dict(default_trainer_kwargs)])]

        return cls(
            run_id=data.get("run_id"),
            runs_dir=runs_dir,
            mode=mode,
            seeds=seeds,
            max_workers=max_workers,
            fail_fast=fail_fast,
            resume=resume,
            job_timeout=job_timeout,
            tasks=tasks,
            trainers=trainers,
            eval_kwargs=default_eval,
            trainer_kwargs=default_trainer_kwargs,
        )

    def ensure_run_id(self) -> str:
        if not self.run_id:
            self.run_id = str(uuid.uuid4())
        return self.run_id

    def snapshot(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "runs_dir": self.runs_dir,
            "mode": self.mode,
            "seeds": list(self.seeds),
            "max_workers": self.max_workers,
            "fail_fast": self.fail_fast,
            "resume": self.resume,
            "job_timeout": self.job_timeout,
            "tasks": [
                {"id": task.id, "eval_kwargs": dict(task.eval_kwargs)}
                for task in self.tasks
            ],
            "trainers": [
                {
                    "id": trainer.id,
                    "params_variants": [dict(p) for p in trainer.params_variants],
                    "optimizer": trainer.optimizer,
                    "optimizer_kwargs": dict(trainer.optimizer_kwargs),
                    "guide": trainer.guide,
                    "guide_kwargs": dict(trainer.guide_kwargs),
                    "logger": trainer.logger,
                    "logger_kwargs": dict(trainer.logger_kwargs),
                }
                for trainer in self.trainers
            ],
            "eval_kwargs": dict(self.eval_kwargs),
            "trainer_kwargs": dict(self.trainer_kwargs),
        }


def load_config(path: str | Path) -> RunConfig:
    config_path = Path(path)
    data = _load_yaml_or_json(config_path)
    return RunConfig.from_dict(data)


__all__ = ["RunConfig", "TaskConfig", "TrainerConfig", "load_config"]
