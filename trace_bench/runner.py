from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import random
import time

from trace_bench.artifacts import (
    RunArtifacts,
    append_event,
    append_results_csv,
    init_job_dir,
    init_run_dir,
    write_config_snapshot,
    write_env_json,
    write_git_json,
    write_manifest,
    write_job_meta,
    write_job_results,
    write_summary,
)
from trace_bench.config import RunConfig, TaskConfig, TrainerConfig
from trace_bench.matrix import JobSpec, compute_run_id, expand_matrix
from trace_bench.registry import load_task_bundle
from trace_bench.resolve import merge_kwargs, resolve_trainer_kwargs
from trace_bench.results import RESULT_COLUMNS, build_results_csv_row, build_results_row, summarize_results


try:
    from opto.trace.nodes import ParameterNode
except Exception:  # pragma: no cover - only when opto is not available
    ParameterNode = object  # type: ignore


@dataclass
class RunSummary:
    run_id: str
    results: List[Dict[str, Any]]


def _extract_response(model: Any, input_value: Any) -> Any:
    if isinstance(model, ParameterNode):
        return getattr(model, "data", model)
    if callable(model):
        output = model(input_value)
        return getattr(output, "data", output)
    return getattr(model, "data", model)


def _evaluate_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    dataset = bundle["train_dataset"]
    guide = bundle["guide"]
    inputs = dataset.get("inputs") or []
    infos = dataset.get("infos") or []
    if not inputs or not infos:
        return {"score": None, "feedback": "empty_dataset"}
    task_input = inputs[0]
    task_info = infos[0]
    response = _extract_response(bundle["param"], task_input)
    try:
        score, feedback = guide(task_input, response, task_info)
    except Exception as exc:
        return {"score": None, "feedback": f"eval_error: {exc}"}
    return {"score": score, "feedback": feedback}


def _resolve_algorithm(name: str):
    if name == "PrioritySearch":
        return "PrioritySearch"
    if name == "GEPA-Base":
        from opto.features.gepa.gepa_algorithms import GEPAAlgorithmBase
        return GEPAAlgorithmBase
    if name == "GEPA-UCB":
        from opto.features.gepa.gepa_algorithms import GEPAUCBSearch
        return GEPAUCBSearch
    if name == "GEPA-Beam":
        from opto.features.gepa.gepa_algorithms import GEPABeamPareto
        return GEPABeamPareto
    return name


def _train_bundle(bundle: Dict[str, Any], trainer_spec: TrainerConfig, params: Dict[str, Any], mode: str) -> Dict[str, Any]:
    from opto import trainer as opto_trainer

    algo_name = trainer_spec.id
    algo = _resolve_algorithm(algo_name)
    kwargs = resolve_trainer_kwargs(params, algo_name)

    optimizer = trainer_spec.optimizer
    guide = trainer_spec.guide or bundle["guide"]
    logger = trainer_spec.logger or "ConsoleLogger"
    guide_kwargs = merge_kwargs(bundle.get("guide_kwargs"), trainer_spec.guide_kwargs or {})
    logger_kwargs = merge_kwargs(bundle.get("logger_kwargs"), trainer_spec.logger_kwargs or {})

    optimizer_kwargs = merge_kwargs(bundle.get("optimizer_kwargs", {}), trainer_spec.optimizer_kwargs or {})

    if mode == "stub":
        try:
            from opto.utils.llm import DummyLLM

            def _dummy_response(*_args, **_kwargs):
                return '{"suggestion": {}}'

            dummy = DummyLLM(_dummy_response)
            if isinstance(optimizer_kwargs, list):
                for item in optimizer_kwargs:
                    item.setdefault("llm", dummy)
            elif isinstance(optimizer_kwargs, dict):
                optimizer_kwargs.setdefault("llm", dummy)
        except Exception:
            pass

    try:
        opto_trainer.train(
            model=bundle["param"],
            train_dataset=bundle["train_dataset"],
            algorithm=algo,
            guide=guide,
            optimizer=optimizer,
            logger=logger,
            optimizer_kwargs=optimizer_kwargs,
            guide_kwargs=guide_kwargs,
            logger_kwargs=logger_kwargs,
            **kwargs,
        )
        return {"status": "ok", "optimizer_kwargs": optimizer_kwargs, "trainer_kwargs": kwargs}
    except Exception as exc:
        return {"status": "failed", "error": str(exc), "optimizer_kwargs": optimizer_kwargs, "trainer_kwargs": kwargs}


def _has_trainables(model: Any) -> bool:
    if isinstance(model, ParameterNode):
        return bool(getattr(model, "trainable", True))
    if hasattr(model, "parameters"):
        try:
            params = model.parameters()
            return any(getattr(p, "trainable", False) for p in params)
        except Exception:
            return True
    return True


class BenchRunner:
    def __init__(self, config: RunConfig, tasks_root: str | Path = "LLM4AD/benchmark_tasks"):
        self.config = config
        self.tasks_root = Path(tasks_root)
        random.seed(self.config.seeds[0] if self.config.seeds else 123)
        self.artifacts: Optional[RunArtifacts] = None
        self._bundle_cache: Dict[str, Dict[str, Any]] = {}

    def _bundle_cache_key(self, task: TaskConfig) -> str:
        eval_sig = json.dumps(task.eval_kwargs or {}, sort_keys=True)
        return f"{task.id}|{eval_sig}"

    def _get_bundle(self, task: TaskConfig) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
        key = self._bundle_cache_key(task)
        if key in self._bundle_cache:
            cached = self._bundle_cache[key]
            return cached["status"], cached.get("bundle"), cached.get("error")
        try:
            bundle = load_task_bundle(task.id, self.tasks_root, eval_kwargs=task.eval_kwargs)
            entry = {"status": "ok", "bundle": bundle, "error": None}
        except NotImplementedError as exc:
            entry = {"status": "skipped", "bundle": None, "error": str(exc)}
        except Exception as exc:
            entry = {"status": "failed", "bundle": None, "error": f"task_load_error: {exc}"}
        self._bundle_cache[key] = entry
        return entry["status"], entry.get("bundle"), entry.get("error")

    def run(self) -> RunSummary:
        snapshot = self.config.snapshot()
        run_id = self.config.run_id or compute_run_id({k: v for k, v in snapshot.items() if k != "run_id"})
        self.config.run_id = run_id
        snapshot = self.config.snapshot()

        self.artifacts = init_run_dir(self.config.runs_dir, run_id)
        write_config_snapshot(self.artifacts.config_snapshot, snapshot)
        write_env_json(self.artifacts.env_json)
        write_git_json(self.artifacts.git_json)

        jobs = expand_matrix(self.config)

        results: List[Dict[str, Any]] = []
        for job in jobs:
            results.append(self._run_job(job))
            if self.config.fail_fast and results[-1].get("status") == "failed":
                break

        result_by_job = {row.get("job_id"): row for row in results}
        manifest_jobs: List[Dict[str, Any]] = []
        for job in jobs:
            row = result_by_job.get(job.job_id, {})
            resolved_trainer_kwargs = resolve_trainer_kwargs(job.params, job.trainer_id)
            status_hint, bundle, skip_reason = self._get_bundle(job.task)
            resolved_optimizer_kwargs = merge_kwargs(
                bundle.get("optimizer_kwargs", {}) if bundle else {},
                job.trainer.optimizer_kwargs or {},
            )
            resolved_guide_kwargs = merge_kwargs(
                bundle.get("guide_kwargs") if bundle else {},
                job.trainer.guide_kwargs or {},
            )
            resolved_logger_kwargs = merge_kwargs(
                bundle.get("logger_kwargs") if bundle else {},
                job.trainer.logger_kwargs or {},
            )
            eval_kwargs = row.get("eval_kwargs") or dict(job.task.eval_kwargs or {})
            manifest_jobs.append(
                {
                    "job_id": job.job_id,
                    "task_id": job.task_id,
                    "suite": job.suite,
                    "trainer_id": job.trainer_id,
                    "seed": job.seed,
                    "resolved_trainer_kwargs": resolved_trainer_kwargs,
                    "resolved_optimizer_kwargs": resolved_optimizer_kwargs,
                    "resolved_guide_kwargs": resolved_guide_kwargs,
                    "resolved_logger_kwargs": resolved_logger_kwargs,
                    "eval_kwargs": eval_kwargs,
                    "status_hint": status_hint,
                    "skip_reason": skip_reason or "",
                }
            )
        manifest = {
            "run_id": run_id,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "jobs": manifest_jobs,
        }
        write_manifest(self.artifacts.manifest_json, manifest)

        write_summary(self.artifacts.summary_json, summarize_results(results))
        return RunSummary(run_id=run_id, results=results)

    def _run_job(self, job: JobSpec) -> Dict[str, Any]:
        assert self.artifacts is not None
        job_artifacts = init_job_dir(self.artifacts, job.job_id)
        start_time = time.time()
        status = "ok"
        feedback: Optional[str] = None

        status_hint, bundle, bundle_error = self._get_bundle(job.task)
        if status_hint != "ok":
            status = status_hint
            feedback = bundle_error

        score_initial = None
        score_final = None
        score_best = None
        resolved_optimizer_kwargs: Dict[str, Any] = dict(job.trainer.optimizer_kwargs or {})
        resolved_trainer_kwargs: Dict[str, Any] = resolve_trainer_kwargs(job.params, job.trainer_id)

        if bundle is not None and status == "ok":
            resolved_optimizer_kwargs = merge_kwargs(
                bundle.get("optimizer_kwargs", {}), job.trainer.optimizer_kwargs or {}
            )
            if not _has_trainables(bundle["param"]):
                status = "failed"
                feedback = "no_trainable_parameters"
            else:
                initial = _evaluate_bundle(bundle)
                score_initial = initial.get("score")
                train_result = _train_bundle(bundle, job.trainer, job.params, self.config.mode)
                status = train_result.get("status", "ok")
                resolved_optimizer_kwargs = train_result.get("optimizer_kwargs") or resolved_optimizer_kwargs
                resolved_trainer_kwargs = train_result.get("trainer_kwargs") or resolved_trainer_kwargs
                if status == "failed":
                    feedback = f"training_error: {train_result.get('error', 'unknown')}"
                final = _evaluate_bundle(bundle)
                score_final = final.get("score")
                if status != "failed":
                    feedback = final.get("feedback") or feedback

                if isinstance(score_initial, (int, float)) and isinstance(score_final, (int, float)):
                    score_best = max(score_initial, score_final)
                else:
                    score_best = score_final if score_final is not None else score_initial

        elapsed = time.time() - start_time
        tb_rel = str(Path("jobs") / job.job_id / "tb")
        row = build_results_row(
            run_id=self.config.run_id or "",
            job_id=job.job_id,
            task_id=job.task_id,
            suite=job.suite,
            trainer_id=job.trainer_id,
            seed=job.seed,
            status=status,
            score_initial=score_initial,
            score_final=score_final,
            score_best=score_best,
            time_seconds=elapsed,
            resolved_trainer_kwargs=resolved_trainer_kwargs,
            resolved_optimizer_kwargs=resolved_optimizer_kwargs,
            eval_kwargs=job.task.eval_kwargs,
            feedback=feedback,
            tb_logdir=tb_rel,
        )
        resolved_guide_kwargs = merge_kwargs(
            bundle.get("guide_kwargs") if bundle else {},
            job.trainer.guide_kwargs,
        )
        resolved_logger_kwargs = merge_kwargs(
            bundle.get("logger_kwargs") if bundle else {},
            job.trainer.logger_kwargs,
        )
        job_meta = {
            "job_id": job.job_id,
            "task_id": job.task_id,
            "suite": job.suite,
            "trainer_id": job.trainer_id,
            "seed": job.seed,
            "status": status,
            "params": job.params,
            "resolved_trainer_kwargs": resolved_trainer_kwargs,
            "resolved_optimizer_kwargs": resolved_optimizer_kwargs,
            "resolved_guide_kwargs": resolved_guide_kwargs,
            "resolved_logger_kwargs": resolved_logger_kwargs,
            "optimizer": job.trainer.optimizer,
            "optimizer_kwargs": job.trainer.optimizer_kwargs,
            "guide": job.trainer.guide,
            "guide_kwargs": job.trainer.guide_kwargs,
            "logger": job.trainer.logger,
            "logger_kwargs": job.trainer.logger_kwargs,
            "eval_kwargs": job.task.eval_kwargs,
            "feedback": feedback or "",
            "tb_logdir": tb_rel,
        }
        write_job_meta(job_artifacts.job_meta, job_meta)
        append_results_csv(self.artifacts.results_csv, RESULT_COLUMNS, build_results_csv_row(row))
        append_event(job_artifacts.events_jsonl, row)
        write_job_results(job_artifacts.results_json, row)
        return row


__all__ = ["BenchRunner", "RunSummary"]
