from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import csv
import json
import logging
import multiprocessing
import os
import random
import tempfile
import threading
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr

from trace_bench.artifacts import (
    JobArtifacts,
    RunArtifacts,
    append_event,
    append_state_event,
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
    write_json,
    write_yaml,
    write_files_index,
)
from trace_bench.config import RunConfig, TaskConfig, TrainerConfig
from trace_bench.matrix import JobSpec, compute_run_id, expand_matrix
from trace_bench.registry import expand_special_tasks, load_task_bundle
from trace_bench.resolve import merge_kwargs, resolve_trainer_kwargs
from trace_bench.results import (
    RESULT_COLUMNS,
    build_leaderboard_rows,
    build_results_csv_row,
    build_results_row,
    summarize_results,
)

from trace_bench.integrations.mlflow_client import log_job_result, log_run_end, log_run_start


logger = logging.getLogger(__name__)

try:
    from opto.trace.nodes import ParameterNode
except Exception:  # pragma: no cover - only when opto is not available
    ParameterNode = object  # type: ignore


# ---------------------------------------------------------------------------
# Trainer / algorithm resolution
# ---------------------------------------------------------------------------

# Static map for well-known aliases. For anything not listed here,
# _resolve_algorithm falls back to dynamic import from the registry.
_ALGORITHM_MAP: Dict[str, Tuple[str, str]] = {
    "GEPA-Base": ("opto.features.gepa.gepa_algorithms", "GEPAAlgorithmBase"),
    "GEPA-UCB": ("opto.features.gepa.gepa_algorithms", "GEPAUCBSearch"),
    "GEPA-Beam": ("opto.features.gepa.gepa_algorithms", "GEPABeamPareto"),
}

# Reverse alias: class name -> trainer_id used in configs
_TRAINER_ALIASES_REV: Dict[str, str] = {
    "GEPAAlgorithmBase": "GEPA-Base",
    "GEPAUCBSearch": "GEPA-UCB",
    "GEPABeamPareto": "GEPA-Beam",
}


def _resolve_algorithm(name: str):
    """Resolve a trainer name to an algorithm class or string.

    Resolution order:
    1. "PrioritySearch" -> returned as string (opto handles it natively)
    2. Static _ALGORITHM_MAP entries (GEPA-*)
    3. Dynamic import: try to find the class in opto.trainer.algorithms
       and opto.features modules
    4. Fallback: return name as-is (let opto raise if unknown)
    """
    if name == "PrioritySearch":
        return "PrioritySearch"

    # Static map
    if name in _ALGORITHM_MAP:
        module_path, class_name = _ALGORITHM_MAP[name]
        import importlib
        try:
            mod = importlib.import_module(module_path)
            return getattr(mod, class_name)
        except Exception:
            return name

    # Dynamic resolution: try common locations
    import importlib

    # Check if name is a class name (not an alias)
    real_name = name
    for alias, tid in _TRAINER_ALIASES_REV.items():
        if tid == name:
            real_name = alias
            break

    # Search paths for dynamic resolution
    search_modules = [
        "opto.trainer.algorithms",
        "opto.trainer.algorithms.basic_algorithms",
        "opto.trainer.algorithms.beamsearch_algorithm",
        "opto.trainer.algorithms.UCBsearch",
        "opto.trainer.algorithms.aggregator",
        "opto.features.gepa.gepa_algorithms",
        "opto.features.priority_search.priority_search",
        "opto.features.priority_search.examples",
    ]

    for module_path in search_modules:
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, real_name, None) or getattr(mod, name, None)
            if cls is not None and isinstance(cls, type):
                return cls
        except Exception:
            continue

    return name


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RunSummary:
    run_id: str
    results: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _apply_llm_config(llm_cfg: Dict[str, Any]) -> None:
    llm_cfg = dict(llm_cfg or {})
    provider = str(llm_cfg.get("provider") or "").lower()
    base_url = str(llm_cfg.get("base_url") or llm_cfg.get("api_base") or "")
    model = str(llm_cfg.get("model") or "")
    api_key_env = str(llm_cfg.get("api_key_env") or "")
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
        os.environ["OPENAI_API_BASE"] = base_url
    if model:
        os.environ["TRACE_LITELLM_MODEL"] = model
    if api_key_env and os.environ.get(api_key_env):
        key = os.environ[api_key_env]
        if provider == "openrouter":
            os.environ["OPENROUTER_API_KEY"] = key
            os.environ.setdefault("OPENAI_API_KEY", key)
        elif provider == "openai":
            os.environ["OPENAI_API_KEY"] = key


def _current_llm_meta() -> Dict[str, str]:
    base = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE") or ""
    provider = "openrouter" if "openrouter.ai" in base else ("openai" if base else "")
    return {
        "llm_provider": provider,
        "llm_model": os.environ.get("TRACE_LITELLM_MODEL", ""),
        "llm_base_url": base,
    }


def _snapshot_model_state(model: Any) -> Dict[str, Any]:
    def _node_state(node: Any, idx: int | None = None) -> Dict[str, Any]:
        return {
            "index": idx,
            "name": getattr(node, "name", None),
            "trainable": getattr(node, "trainable", None),
            "value": getattr(node, "data", None),
        }

    if isinstance(model, ParameterNode):
        return {"kind": "ParameterNode", "parameter": _node_state(model)}
    if hasattr(model, "parameters"):
        try:
            params = list(model.parameters())
        except Exception:
            params = []
        return {
            "kind": type(model).__name__,
            "parameters": [_node_state(p, i) for i, p in enumerate(params)],
        }
    return {"kind": type(model).__name__, "repr": str(model)}


def _select_best_state(initial_state: Dict[str, Any], final_state: Dict[str, Any], score_initial: Any, score_final: Any) -> Dict[str, Any]:
    try:
        if score_final is not None and score_initial is not None and float(score_final) >= float(score_initial):
            return final_state
    except Exception:
        pass
    return initial_state or final_state


def _token_scope_note() -> str:
    return "trace_optimization_only"


def _state_rel_paths(job_id: str) -> Dict[str, str]:
    base = Path("jobs") / job_id / "artifacts"
    return {
        "initial_state_path": (base / "initial_state.yaml").as_posix(),
        "best_state_path": (base / "best_state.yaml").as_posix(),
        "final_state_path": (base / "final_state.yaml").as_posix(),
        "state_history_path": (base / "state_history.jsonl").as_posix(),
    }


def _extract_token_usage(value: Any) -> Dict[str, int]:
    prompt = completion = total = 0
    candidates = []
    if isinstance(value, dict):
        candidates.extend([value, value.get("usage"), value.get("last_usage"), value.get("token_usage")])
    else:
        for attr in ("usage", "last_usage", "token_usage", "stats"):
            try:
                candidates.append(getattr(value, attr))
            except Exception:
                pass
    for candidate in candidates:
        if isinstance(candidate, dict):
            try:
                prompt = max(prompt, int(candidate.get("prompt_tokens") or 0))
            except Exception:
                pass
            try:
                completion = max(completion, int(candidate.get("completion_tokens") or 0))
            except Exception:
                pass
            try:
                total = max(total, int(candidate.get("total_tokens") or 0))
            except Exception:
                pass
    if total == 0:
        total = prompt + completion
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


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


def _train_bundle(
    bundle: Dict[str, Any],
    trainer_spec: TrainerConfig,
    params: Dict[str, Any],
    mode: str,
) -> Dict[str, Any]:
    """Train a bundle synchronously. Timeout is handled at the job level."""
    from opto import trainer as opto_trainer

    algo_name = trainer_spec.id
    algo = _resolve_algorithm(algo_name)
    runtime = _resolve_runtime_from_bundle(bundle, trainer_spec, params)
    kwargs = runtime["resolved_trainer_kwargs"]
    optimizer_kwargs = runtime["resolved_optimizer_kwargs"]
    guide_kwargs = runtime["resolved_guide_kwargs"]
    logger_kwargs = runtime["resolved_logger_kwargs"]
    optimizer = trainer_spec.optimizer
    guide = runtime["guide_obj"]
    log = runtime["logger_obj"]

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

    # Pass through multi-objective config from bundle if present
    objective_config = bundle.get("objective_config")
    if objective_config is not None:
        kwargs["objective_config"] = objective_config

    none_aliases = {"none", "null", "off", "disable", "disabled"}

    def _call_train(pass_logger: bool) -> None:
        if pass_logger:
            opto_trainer.train(
                model=bundle["param"],
                train_dataset=bundle["train_dataset"],
                algorithm=algo,
                guide=guide,
                optimizer=optimizer,
                logger=log,
                optimizer_kwargs=optimizer_kwargs,
                guide_kwargs=guide_kwargs,
                logger_kwargs=logger_kwargs,
                **kwargs,
            )
            return
        opto_trainer.train(
            model=bundle["param"],
            train_dataset=bundle["train_dataset"],
            algorithm=algo,
            guide=guide,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            guide_kwargs=guide_kwargs,
            **kwargs,
        )

    pass_logger = not (isinstance(log, str) and log.strip().lower() in none_aliases)
    try:
        _call_train(pass_logger=pass_logger)
        return {
            "status": "ok",
            "resolved_optimizer": runtime["resolved_optimizer"],
            "resolved_guide": runtime["resolved_guide"],
            "resolved_logger": runtime["resolved_logger"],
            "resolved_trainer_kwargs": kwargs,
            "resolved_optimizer_kwargs": optimizer_kwargs,
            "resolved_guide_kwargs": guide_kwargs,
            "resolved_logger_kwargs": logger_kwargs,
            # Backward-compatible keys used by older tests/consumers.
            "trainer_kwargs": kwargs,
            "optimizer_kwargs": optimizer_kwargs,
        }
    except TypeError as exc:
        # Backward-compat: OpenTrace without logger-override patch may reject logger kwarg.
        msg = str(exc).lower()
        if pass_logger and "logger" in msg and "unexpected keyword" in msg:
            try:
                logger.warning("Trainer rejected logger kwarg; retrying without logger.")
                _call_train(pass_logger=False)
                return {
                    "status": "ok",
                    "resolved_optimizer": runtime["resolved_optimizer"],
                    "resolved_guide": runtime["resolved_guide"],
                    "resolved_logger": runtime["resolved_logger"],
                    "resolved_trainer_kwargs": kwargs,
                    "resolved_optimizer_kwargs": optimizer_kwargs,
                    "resolved_guide_kwargs": guide_kwargs,
                    "resolved_logger_kwargs": logger_kwargs,
                    "trainer_kwargs": kwargs,
                    "optimizer_kwargs": optimizer_kwargs,
                }
            except Exception:
                pass
        return {
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "resolved_optimizer": runtime["resolved_optimizer"],
            "resolved_guide": runtime["resolved_guide"],
            "resolved_logger": runtime["resolved_logger"],
            "resolved_trainer_kwargs": kwargs,
            "resolved_optimizer_kwargs": optimizer_kwargs,
            "resolved_guide_kwargs": guide_kwargs,
            "resolved_logger_kwargs": logger_kwargs,
            "trainer_kwargs": kwargs,
            "optimizer_kwargs": optimizer_kwargs,
        }
    except Exception as exc:
        # Backward-compat: if logger name/semantics are unsupported in upstream OpenTrace,
        # retry once without logger arguments.
        msg = str(exc).lower()
        if pass_logger and "logger" in msg:
            try:
                logger.warning("Training failed with logger override (%s); retrying without logger.", exc)
                _call_train(pass_logger=False)
                return {
                    "status": "ok",
                    "resolved_optimizer": runtime["resolved_optimizer"],
                    "resolved_guide": runtime["resolved_guide"],
                    "resolved_logger": runtime["resolved_logger"],
                    "resolved_trainer_kwargs": kwargs,
                    "resolved_optimizer_kwargs": optimizer_kwargs,
                    "resolved_guide_kwargs": guide_kwargs,
                    "resolved_logger_kwargs": logger_kwargs,
                    "trainer_kwargs": kwargs,
                    "optimizer_kwargs": optimizer_kwargs,
                }
            except Exception:
                pass
        return {
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "resolved_optimizer": runtime["resolved_optimizer"],
            "resolved_guide": runtime["resolved_guide"],
            "resolved_logger": runtime["resolved_logger"],
            "resolved_trainer_kwargs": kwargs,
            "resolved_optimizer_kwargs": optimizer_kwargs,
            "resolved_guide_kwargs": guide_kwargs,
            "resolved_logger_kwargs": logger_kwargs,
            "trainer_kwargs": kwargs,
            "optimizer_kwargs": optimizer_kwargs,
        }


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


def _component_identity(component: Any) -> Any:
    if component is None:
        return None
    if isinstance(component, str):
        return component
    if isinstance(component, type):
        return f"{component.__module__}.{component.__name__}"
    return f"{component.__class__.__module__}.{component.__class__.__name__}"


def _default_optimizer_name(model: Any) -> str:
    return "OPROv2" if isinstance(model, ParameterNode) else "OptoPrimeV2"


def _resolve_runtime_from_bundle(
    bundle: Dict[str, Any],
    trainer_spec: TrainerConfig,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    resolved_trainer_kwargs = resolve_trainer_kwargs(params, trainer_spec.id)
    resolved_optimizer_kwargs = merge_kwargs(bundle.get("optimizer_kwargs", {}), trainer_spec.optimizer_kwargs or {})
    resolved_guide_kwargs = merge_kwargs(bundle.get("guide_kwargs"), trainer_spec.guide_kwargs or {})
    resolved_logger_kwargs = merge_kwargs(bundle.get("logger_kwargs"), trainer_spec.logger_kwargs or {})

    guide_obj = trainer_spec.guide or bundle["guide"]
    logger_obj = trainer_spec.logger or "ConsoleLogger"
    optimizer_name = trainer_spec.optimizer or _default_optimizer_name(bundle["param"])

    return {
        "resolved_optimizer": _component_identity(optimizer_name),
        "resolved_guide": _component_identity(guide_obj),
        "resolved_logger": _component_identity(logger_obj),
        "resolved_trainer_kwargs": resolved_trainer_kwargs,
        "resolved_optimizer_kwargs": resolved_optimizer_kwargs,
        "resolved_guide_kwargs": resolved_guide_kwargs,
        "resolved_logger_kwargs": resolved_logger_kwargs,
        "guide_obj": guide_obj,
        "logger_obj": logger_obj,
    }


def _resolve_runtime_without_bundle(job: JobSpec) -> Dict[str, Any]:
    return {
        "resolved_optimizer": _component_identity(job.trainer.optimizer),
        "resolved_guide": _component_identity(job.trainer.guide),
        "resolved_logger": _component_identity(job.trainer.logger or "ConsoleLogger"),
        "resolved_trainer_kwargs": resolve_trainer_kwargs(job.params, job.trainer_id),
        "resolved_optimizer_kwargs": dict(job.trainer.optimizer_kwargs or {}),
        "resolved_guide_kwargs": merge_kwargs({}, job.trainer.guide_kwargs),
        "resolved_logger_kwargs": merge_kwargs({}, job.trainer.logger_kwargs),
    }


# ---------------------------------------------------------------------------
# Subprocess-based job execution (for hard timeout)
# ---------------------------------------------------------------------------

def _trainer_config_to_dict(tc: TrainerConfig) -> Dict[str, Any]:
    """Serialize TrainerConfig to a pickle/JSON-safe dict."""
    return {
        "id": tc.id,
        "params_variants": [dict(p) for p in tc.params_variants],
        "optimizer": tc.optimizer,
        "optimizer_kwargs": dict(tc.optimizer_kwargs or {}),
        "guide": tc.guide,
        "guide_kwargs": dict(tc.guide_kwargs or {}),
        "logger": tc.logger,
        "logger_kwargs": dict(tc.logger_kwargs or {}),
    }


def _trainer_config_from_dict(d: Dict[str, Any]) -> TrainerConfig:
    """Reconstruct TrainerConfig from dict in subprocess."""
    return TrainerConfig(
        id=d["id"],
        params_variants=d.get("params_variants", [{}]),
        optimizer=d.get("optimizer"),
        optimizer_kwargs=d.get("optimizer_kwargs", {}),
        guide=d.get("guide"),
        guide_kwargs=d.get("guide_kwargs", {}),
        logger=d.get("logger"),
        logger_kwargs=d.get("logger_kwargs", {}),
    )


def _subprocess_job_target(
    task_id: str,
    tasks_root: str,
    trainer_dict: Dict[str, Any],
    params: Dict[str, Any],
    mode: str,
    eval_kwargs: Dict[str, Any],
    result_file: str,
    stdout_log: str,
) -> None:
    """Run a full job in a child process: load bundle -> eval -> train -> eval.

    Writes a JSON payload to result_file atomically and appends process output
    to stdout_log so the UI can inspect progress after the fact.
    """
    import time as _time

    def _write_atomic(path: str, payload_value: Dict[str, Any]) -> None:
        from trace_bench.artifacts import sanitize_for_json

        target = Path(path)
        tmp = target.with_suffix(target.suffix + ".tmp")
        data = json.dumps(sanitize_for_json(payload_value), ensure_ascii=False)
        with tmp.open("w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)

    start = _time.time()
    payload: Dict[str, Any] = {
        "status": "failed",
        "score_initial": None,
        "score_final": None,
        "score_best": None,
        "feedback": None,
        "resolved_optimizer": None,
        "resolved_guide": None,
        "resolved_logger": None,
        "resolved_trainer_kwargs": {},
        "resolved_optimizer_kwargs": {},
        "resolved_guide_kwargs": {},
        "resolved_logger_kwargs": {},
        "initial_state": {},
        "final_state": {},
        "best_state": {},
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "elapsed": 0.0,
    }

    Path(stdout_log).parent.mkdir(parents=True, exist_ok=True)
    with open(stdout_log, "a", encoding="utf-8", errors="ignore") as logf, redirect_stdout(logf), redirect_stderr(logf):
        try:
            trainer_spec = _trainer_config_from_dict(trainer_dict)
            bundle = load_task_bundle(task_id, tasks_root, eval_kwargs=eval_kwargs)
            runtime = _resolve_runtime_from_bundle(bundle, trainer_spec, params)
            payload.update(
                {
                    "resolved_optimizer": runtime["resolved_optimizer"],
                    "resolved_guide": runtime["resolved_guide"],
                    "resolved_logger": runtime["resolved_logger"],
                    "resolved_trainer_kwargs": runtime["resolved_trainer_kwargs"],
                    "resolved_optimizer_kwargs": runtime["resolved_optimizer_kwargs"],
                    "resolved_guide_kwargs": runtime["resolved_guide_kwargs"],
                    "resolved_logger_kwargs": runtime["resolved_logger_kwargs"],
                }
            )

            if not _has_trainables(bundle["param"]):
                payload["status"] = "failed"
                payload["feedback"] = "no_trainable_parameters"
            else:
                payload["initial_state"] = _snapshot_model_state(bundle["param"])
                initial = _evaluate_bundle(bundle)
                payload["score_initial"] = initial.get("score")

                train_result = _train_bundle(bundle, trainer_spec, params, mode)
                payload["status"] = train_result.get("status", "ok")
                payload["resolved_optimizer"] = train_result.get("resolved_optimizer", payload["resolved_optimizer"])
                payload["resolved_guide"] = train_result.get("resolved_guide", payload["resolved_guide"])
                payload["resolved_logger"] = train_result.get("resolved_logger", payload["resolved_logger"])
                payload["resolved_guide_kwargs"] = (
                    train_result.get("resolved_guide_kwargs") or payload["resolved_guide_kwargs"]
                )
                payload["resolved_logger_kwargs"] = (
                    train_result.get("resolved_logger_kwargs") or payload["resolved_logger_kwargs"]
                )
                payload["resolved_optimizer_kwargs"] = (
                    train_result.get("resolved_optimizer_kwargs")
                    or train_result.get("optimizer_kwargs")
                    or payload["resolved_optimizer_kwargs"]
                )
                payload["resolved_trainer_kwargs"] = (
                    train_result.get("resolved_trainer_kwargs")
                    or train_result.get("trainer_kwargs")
                    or payload["resolved_trainer_kwargs"]
                )
                if payload["status"] == "failed":
                    trace = train_result.get("traceback")
                    suffix = f"\n{trace}" if trace else ""
                    payload["feedback"] = f"training_error: {train_result.get('error', 'unknown')}{suffix}"

                final = _evaluate_bundle(bundle)
                payload["score_final"] = final.get("score")
                if payload["status"] != "failed":
                    payload["feedback"] = final.get("feedback") or payload["feedback"]

                si, sf = payload["score_initial"], payload["score_final"]
                if isinstance(si, (int, float)) and isinstance(sf, (int, float)):
                    payload["score_best"] = max(si, sf)
                else:
                    payload["score_best"] = sf if sf is not None else si

                payload["final_state"] = _snapshot_model_state(bundle["param"])
                payload["best_state"] = _select_best_state(
                    payload["initial_state"], payload["final_state"], payload["score_initial"], payload["score_final"]
                )
                payload.update(_extract_token_usage(payload["resolved_optimizer_kwargs"]))

        except NotImplementedError as exc:
            payload["status"] = "skipped"
            payload["feedback"] = str(exc)
        except Exception as exc:
            payload["status"] = "failed"
            payload["feedback"] = f"subprocess_error: {exc}\n{traceback.format_exc()}"

        payload["elapsed"] = _time.time() - start
        _write_atomic(result_file, payload)


def _build_files_index(run_artifacts: RunArtifacts, manifest_jobs: List[Dict[str, Any]]) -> Dict[str, Any]:
    jobs_index: List[Dict[str, Any]] = []
    for job in manifest_jobs:
        job_id = str(job.get("job_id") or "")
        if not job_id:
            continue
        job_dir = run_artifacts.jobs_dir / job_id
        jobs_index.append({
            "job_id": job_id,
            "task_id": job.get("task_id"),
            "suite": job.get("suite"),
            "trainer_id": job.get("trainer_id"),
            "files": {
                "job_meta": str(job_dir / "job_meta.json"),
                "results_json": str(job_dir / "results.json"),
                "events_jsonl": str(job_dir / "events.jsonl"),
                "stdout_log": str(job_dir / "stdout.log"),
                "tb_dir": str(job_dir / "tb"),
                "initial_state_yaml": str(job_dir / "artifacts" / "initial_state.yaml"),
                "best_state_yaml": str(job_dir / "artifacts" / "best_state.yaml"),
                "final_state_yaml": str(job_dir / "artifacts" / "final_state.yaml"),
                "initial_state_json": str(job_dir / "artifacts" / "initial_state.json"),
                "best_state_json": str(job_dir / "artifacts" / "best_state.json"),
                "final_state_json": str(job_dir / "artifacts" / "final_state.json"),
                "state_history_jsonl": str(job_dir / "artifacts" / "state_history.jsonl"),
            },
        })
    return {
        "run_dir": str(run_artifacts.run_dir),
        "canonical_files": {
            "config_snapshot": str(run_artifacts.config_snapshot),
            "env_json": str(run_artifacts.env_json),
            "git_json": str(run_artifacts.git_json),
            "manifest_json": str(run_artifacts.manifest_json),
            "results_csv": str(run_artifacts.results_csv),
            "summary_json": str(run_artifacts.summary_json),
            "leaderboard_csv": str(run_artifacts.leaderboard_csv),
        },
        "notes": {
            "token_scope": _token_scope_note(),
            "execution_agent_tokens": "not captured unless the optimized code/agent explicitly logs them",
            "state_history_scope": "initial/best/final snapshots are persisted; full per-iteration state requires trainer/optimizer callbacks",
        },
        "jobs": jobs_index,
    }


# ---------------------------------------------------------------------------
# BenchRunner
# ---------------------------------------------------------------------------

class BenchRunner:
    def __init__(
        self,
        config: RunConfig,
        tasks_root: str | Path = "LLM4AD/benchmark_tasks",
        job_timeout: Optional[float] = None,
    ):
        self.config = config
        self.tasks_root = Path(tasks_root)
        self.config.tasks = expand_special_tasks(self.config.tasks, self.tasks_root)
        self.job_timeout = job_timeout
        random.seed(self.config.seeds[0] if self.config.seeds else 123)
        self.artifacts: Optional[RunArtifacts] = None
        self._bundle_cache: Dict[str, Dict[str, Any]] = {}
        self._csv_lock = threading.Lock()

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

    # ------------------------------------------------------------------
    # Resume: check existing job status and decide whether to skip
    # ------------------------------------------------------------------

    def _check_existing_status(self, job: JobSpec) -> Optional[str]:
        """Return the existing status of a previously-run job, or None.

        Checks job_meta.json first (canonical), falls back to results.json.
        """
        if self.artifacts is None:
            return None
        job_dir = self.artifacts.jobs_dir / job.job_id

        # Primary: job_meta.json (client requirement)
        meta_path = job_dir / "job_meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                return meta.get("status")
            except Exception:
                pass

        # Fallback: results.json
        results_path = job_dir / "results.json"
        if results_path.exists():
            try:
                existing = json.loads(results_path.read_text(encoding="utf-8"))
                return existing.get("status")
            except Exception:
                pass

        return None

    def _should_skip_job(self, job: JobSpec, resume_mode: str) -> bool:
        """Decide whether to skip a job based on resume mode.

        Resume modes:
          - "none"   : never skip, run everything fresh
          - "auto"   : skip OK/reused/skipped jobs, re-run failed, run new
          - "failed" : re-run ONLY failed jobs, skip OK, skip never-run
        """
        if resume_mode == "none":
            return False

        existing_status = self._check_existing_status(job)

        if resume_mode == "auto":
            # Skip if previously succeeded; re-run if failed or never ran
            if existing_status in ("ok", "reused", "skipped"):
                return True
            return False

        if resume_mode == "failed":
            # Only re-run jobs that previously failed
            if existing_status == "failed":
                return False  # don't skip — re-run it
            # Skip everything else (OK jobs AND never-run jobs)
            return True

        return False

    def _load_existing_results(self, job: JobSpec) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Load existing results for a skipped/reused job."""
        if self.artifacts is None:
            return None
        results_path = self.artifacts.jobs_dir / job.job_id / "results.json"
        if not results_path.exists():
            return None
        try:
            existing = json.loads(results_path.read_text(encoding="utf-8"))
            manifest_job = {
                "job_id": job.job_id,
                "task_id": existing.get("task_id", job.task_id),
                "suite": existing.get("suite", job.suite),
                "trainer_id": existing.get("trainer_id", job.trainer_id),
                "seed": existing.get("seed", job.seed),
                "raw_params": dict(job.params),
                "resolved_optimizer": existing.get("resolved_optimizer"),
                "resolved_guide": existing.get("resolved_guide"),
                "resolved_logger": existing.get("resolved_logger"),
                "resolved_trainer_kwargs": existing.get("resolved_trainer_kwargs", {}),
                "resolved_optimizer_kwargs": existing.get("resolved_optimizer_kwargs", {}),
                "resolved_guide_kwargs": existing.get("resolved_guide_kwargs", {}),
                "resolved_logger_kwargs": existing.get("resolved_logger_kwargs", {}),
                "eval_kwargs": existing.get("eval_kwargs", {}),
                "status": "reused",
                "feedback": existing.get("feedback", ""),
            }
            return existing, manifest_job
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Main run loop (supports parallel via max_workers)
    # ------------------------------------------------------------------

    def run(self, force: bool = False) -> RunSummary:
        snapshot = self.config.snapshot()
        run_id = self.config.run_id or compute_run_id({k: v for k, v in snapshot.items() if k != "run_id"})
        self.config.run_id = run_id
        snapshot = self.config.snapshot()

        self.artifacts = init_run_dir(self.config.runs_dir, run_id)
        _apply_llm_config(self.config.llm)
        write_config_snapshot(self.artifacts.config_snapshot, snapshot)
        write_env_json(self.artifacts.env_json)
        write_git_json(self.artifacts.git_json)

        # Optional MLflow mirroring (filesystem remains canonical).
        try:
            env_payload = json.loads(self.artifacts.env_json.read_text(encoding="utf-8"))
        except Exception:
            env_payload = {}
        try:
            git_payload = json.loads(self.artifacts.git_json.read_text(encoding="utf-8"))
        except Exception:
            git_payload = {}
        mlflow_ctx = log_run_start(self.artifacts.run_dir, snapshot, env_payload, git_payload)

        jobs = expand_matrix(self.config)
        max_workers = max(1, self.config.max_workers)

        # Resolve effective resume mode: --force overrides config
        resume_mode = "none" if force else self.config.resume

        # Resolve effective timeout: explicit > config > None
        effective_timeout = self.job_timeout if self.job_timeout is not None else self.config.job_timeout

        results: List[Dict[str, Any]] = []
        manifest_jobs: List[Dict[str, Any]] = []
        failed_flag = threading.Event()

        if max_workers == 1:
            # Sequential path (simple, no thread overhead)
            for job in jobs:
                if self.config.fail_fast and failed_flag.is_set():
                    break
                # Resume check
                if self._should_skip_job(job, resume_mode):
                    existing = self._load_existing_results(job)
                    if existing is not None:
                        row, manifest_job = existing
                        results.append(row)
                        manifest_jobs.append(manifest_job)
                        logger.info("Reused existing job %s (%s)", job.job_id, job.task_id)
                    else:
                        logger.info("Skipping job %s (%s) -- no existing results, resume=%s", job.job_id, job.task_id, resume_mode)
                    continue

                row, manifest_job = self._run_job(job, timeout=effective_timeout)
                results.append(row)
                manifest_jobs.append(manifest_job)
                # Mirror to MLflow if enabled.
                try:
                    job_meta_path = self.artifacts.jobs_dir / job.job_id / "job_meta.json"
                    job_meta_payload = json.loads(job_meta_path.read_text(encoding="utf-8")) if job_meta_path.exists() else {}
                except Exception:
                    job_meta_payload = {}
                log_job_result(mlflow_ctx, job_meta_payload, row)
                if row.get("status") == "failed":
                    failed_flag.set()
        else:
            # Parallel path using ThreadPoolExecutor.
            # Note: We use threads rather than processes because each job's
            # main cost is I/O-bound (LLM API calls). Threads share memory
            # (bundle cache, CSV lock) without serialization overhead.
            # Process-based parallelism is possible but not required.
            results_lock = threading.Lock()

            def _execute_job(j: JobSpec) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
                if self.config.fail_fast and failed_flag.is_set():
                    return None

                # Resume check
                if self._should_skip_job(j, resume_mode):
                    existing = self._load_existing_results(j)
                    if existing is not None:
                        logger.info("Reused existing job %s (%s)", j.job_id, j.task_id)
                        return existing
                    logger.info("Skipping job %s (%s) -- no existing results, resume=%s",
                                j.job_id, j.task_id, resume_mode)
                    return None

                row, manifest_job = self._run_job(j, timeout=effective_timeout)
                if row.get("status") == "failed":
                    failed_flag.set()
                return row, manifest_job

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_job: Dict[Future, JobSpec] = {}
                for job in jobs:
                    if self.config.fail_fast and failed_flag.is_set():
                        break
                    f = executor.submit(_execute_job, job)
                    future_to_job[f] = job

                for future in as_completed(future_to_job):
                    result = future.result()
                    if result is not None:
                        row, manifest_job = result
                        with results_lock:
                            results.append(row)
                            manifest_jobs.append(manifest_job)
                        # Mirror to MLflow if enabled.
                        try:
                            job_meta_path = self.artifacts.jobs_dir / row.get("job_id", "") / "job_meta.json"
                            job_meta_payload = json.loads(job_meta_path.read_text(encoding="utf-8")) if job_meta_path.exists() else {}
                        except Exception:
                            job_meta_payload = {}
                        log_job_result(mlflow_ctx, job_meta_payload, row)

        # Fill unexecuted jobs (fail_fast stopped, or other reasons)
        recorded_job_ids = {entry["job_id"] for entry in manifest_jobs}
        for job in jobs:
            if job.job_id in recorded_job_ids:
                continue
            status_hint, bundle, skip_reason = self._get_bundle(job.task)
            runtime = (
                _resolve_runtime_from_bundle(bundle, job.trainer, job.params)
                if bundle is not None
                else _resolve_runtime_without_bundle(job)
            )
            manifest_jobs.append(
                {
                    "job_id": job.job_id,
                    "task_id": job.task_id,
                    "suite": job.suite,
                    "trainer_id": job.trainer_id,
                    "seed": job.seed,
                    "raw_params": dict(job.params),
                    "resolved_optimizer": runtime.get("resolved_optimizer"),
                    "resolved_guide": runtime.get("resolved_guide"),
                    "resolved_logger": runtime.get("resolved_logger"),
                    "resolved_trainer_kwargs": runtime.get("resolved_trainer_kwargs", {}),
                    "resolved_optimizer_kwargs": runtime.get("resolved_optimizer_kwargs", {}),
                    "resolved_guide_kwargs": runtime.get("resolved_guide_kwargs", {}),
                    "resolved_logger_kwargs": runtime.get("resolved_logger_kwargs", {}),
                    "eval_kwargs": dict(job.task.eval_kwargs or {}),
                    "status": "not_executed",
                    "status_hint": status_hint,
                    "skip_reason": skip_reason or "fail_fast_stopped",
                }
            )

        manifest = {
            "run_id": run_id,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "jobs": manifest_jobs,
            "tags": list(self.config.tags),
        }
        write_manifest(self.artifacts.manifest_json, manifest)

        summary_payload = summarize_results(results)
        summary_payload["run_id"] = run_id
        summary_payload["tags"] = list(self.config.tags)
        summary_payload["token_scope"] = _token_scope_note()
        write_summary(self.artifacts.summary_json, summary_payload)

        leaderboard_rows = build_leaderboard_rows(results)
        if leaderboard_rows:
            with self.artifacts.leaderboard_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["rank", "task_id", "suite", "job_id", "trainer_id", "score_best", "time_seconds"],
                )
                writer.writeheader()
                writer.writerows(leaderboard_rows)

        write_files_index(self.artifacts.files_index_json, _build_files_index(self.artifacts, manifest_jobs))

        log_run_end(mlflow_ctx, summary_payload)
        return RunSummary(run_id=run_id, results=results)

    @staticmethod
    def _inject_tb_logdir(job: JobSpec, job_artifacts: JobArtifacts) -> JobSpec:
        """Inject TensorBoard logdir into trainer logger_kwargs.

        Returns a new JobSpec with a *copied* TrainerConfig so the original
        (potentially shared across jobs) is not mutated.
        """
        logger_name = str(job.trainer.logger or "").lower()
        if "tensorboard" not in logger_name:
            return job

        from dataclasses import replace as dc_replace
        new_logger_kwargs = dict(job.trainer.logger_kwargs or {})
        new_logger_kwargs["logdir"] = str(job_artifacts.tb_dir)
        new_trainer = dc_replace(job.trainer, logger_kwargs=new_logger_kwargs)
        return dc_replace(job, trainer=new_trainer)

    def _run_job(self, job: JobSpec, timeout: Optional[float] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        assert self.artifacts is not None
        job_artifacts = init_job_dir(self.artifacts, job.job_id)

        # Inject TensorBoard logdir into trainer logger_kwargs (copy to avoid
        # mutating shared TrainerConfig across jobs).
        job = self._inject_tb_logdir(job, job_artifacts)

        if timeout and timeout > 0:
            # ---- Subprocess path: hard-kill timeout ----
            payload = self._run_job_subprocess(job, timeout, str(job_artifacts.stdout_log))
        else:
            # ---- In-process path: no timeout overhead ----
            with job_artifacts.stdout_log.open("a", encoding="utf-8", errors="ignore") as _logf, redirect_stdout(_logf), redirect_stderr(_logf):
                payload = self._run_job_inprocess(job)

        status = payload.get("status", "failed")
        feedback = payload.get("feedback")
        score_initial = payload.get("score_initial")
        score_final = payload.get("score_final")
        score_best = payload.get("score_best")
        elapsed = payload.get("elapsed", 0.0)
        resolved_optimizer = payload.get("resolved_optimizer")
        resolved_guide = payload.get("resolved_guide")
        resolved_logger = payload.get("resolved_logger")
        resolved_trainer_kwargs = payload.get("resolved_trainer_kwargs", {})
        resolved_optimizer_kwargs = payload.get("resolved_optimizer_kwargs", {})
        resolved_guide_kwargs = payload.get("resolved_guide_kwargs", {})
        resolved_logger_kwargs = payload.get("resolved_logger_kwargs", {})

        tb_rel = str(Path("jobs") / job.job_id / "tb")
        llm_meta = _current_llm_meta()
        state_paths = _state_rel_paths(job.job_id)
        prompt_tokens = payload.get("prompt_tokens")
        completion_tokens = payload.get("completion_tokens")
        total_tokens = payload.get("total_tokens")
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
            resolved_optimizer=resolved_optimizer,
            resolved_guide=resolved_guide,
            resolved_logger=resolved_logger,
            resolved_trainer_kwargs=resolved_trainer_kwargs,
            resolved_optimizer_kwargs=resolved_optimizer_kwargs,
            resolved_guide_kwargs=resolved_guide_kwargs,
            resolved_logger_kwargs=resolved_logger_kwargs,
            eval_kwargs=job.task.eval_kwargs,
            feedback=feedback,
            llm_provider=llm_meta["llm_provider"],
            llm_model=llm_meta["llm_model"],
            llm_base_url=llm_meta["llm_base_url"],
            token_scope=_token_scope_note(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            initial_state_path=state_paths["initial_state_path"],
            best_state_path=state_paths["best_state_path"],
            final_state_path=state_paths["final_state_path"],
            state_history_path=state_paths["state_history_path"],
            tb_logdir=tb_rel,
        )
        job_meta = {
            "job_id": job.job_id,
            "task_id": job.task_id,
            "suite": job.suite,
            "trainer_id": job.trainer_id,
            "seed": job.seed,
            "status": status,
            "raw_params": dict(job.params),
            "params": job.params,
            "resolved_optimizer": resolved_optimizer,
            "resolved_guide": resolved_guide,
            "resolved_logger": resolved_logger,
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
            "llm": llm_meta,
            "token_scope": _token_scope_note(),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "state_files": state_paths,
            "tb_logdir": tb_rel,
        }
        write_job_meta(job_artifacts.job_meta, job_meta)
        initial_state = payload.get("initial_state") or {}
        final_state = payload.get("final_state") or {}
        best_state = payload.get("best_state") or {}
        if initial_state:
            write_json(job_artifacts.initial_state_json, initial_state)
            write_yaml(job_artifacts.initial_state_yaml, initial_state)
            append_state_event(job_artifacts.state_history_jsonl, {"kind": "initial", "step": 0, "state": initial_state, "score": score_initial})
        if final_state:
            write_json(job_artifacts.final_state_json, final_state)
            write_yaml(job_artifacts.final_state_yaml, final_state)
            append_state_event(job_artifacts.state_history_jsonl, {"kind": "final", "step": None, "state": final_state, "score": score_final})
        if best_state:
            write_json(job_artifacts.best_state_json, best_state)
            write_yaml(job_artifacts.best_state_yaml, best_state)
            append_state_event(job_artifacts.state_history_jsonl, {"kind": "best", "step": None, "state": best_state, "score": score_best})
        with self._csv_lock:
            append_results_csv(self.artifacts.results_csv, RESULT_COLUMNS, build_results_csv_row(row))
        append_event(job_artifacts.events_jsonl, row)
        write_job_results(job_artifacts.results_json, row)
        manifest_job = {
            "job_id": job.job_id,
            "task_id": job.task_id,
            "suite": job.suite,
            "trainer_id": job.trainer_id,
            "seed": job.seed,
            "raw_params": dict(job.params),
            "resolved_optimizer": resolved_optimizer,
            "resolved_guide": resolved_guide,
            "resolved_logger": resolved_logger,
            "resolved_trainer_kwargs": resolved_trainer_kwargs,
            "resolved_optimizer_kwargs": resolved_optimizer_kwargs,
            "resolved_guide_kwargs": resolved_guide_kwargs,
            "resolved_logger_kwargs": resolved_logger_kwargs,
            "eval_kwargs": dict(job.task.eval_kwargs or {}),
            "status": status,
            "feedback": feedback or "",
            "llm": llm_meta,
            "token_scope": _token_scope_note(),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "state_files": state_paths,
        }
        return row, manifest_job

    # ------------------------------------------------------------------
    # In-process job execution (no timeout)
    # ------------------------------------------------------------------

    def _run_job_inprocess(self, job: JobSpec) -> Dict[str, Any]:
        """Execute a job in the current process. Uses bundle cache."""
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
        initial_state: Dict[str, Any] = {}
        final_state: Dict[str, Any] = {}
        best_state: Dict[str, Any] = {}
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        runtime = _resolve_runtime_without_bundle(job)
        resolved_optimizer = runtime["resolved_optimizer"]
        resolved_guide = runtime["resolved_guide"]
        resolved_logger = runtime["resolved_logger"]
        resolved_trainer_kwargs: Dict[str, Any] = runtime["resolved_trainer_kwargs"]
        resolved_optimizer_kwargs: Dict[str, Any] = runtime["resolved_optimizer_kwargs"]
        resolved_guide_kwargs = runtime["resolved_guide_kwargs"]
        resolved_logger_kwargs = runtime["resolved_logger_kwargs"]

        if bundle is not None and status == "ok":
            runtime = _resolve_runtime_from_bundle(bundle, job.trainer, job.params)
            resolved_optimizer = runtime["resolved_optimizer"]
            resolved_guide = runtime["resolved_guide"]
            resolved_logger = runtime["resolved_logger"]
            resolved_trainer_kwargs = runtime["resolved_trainer_kwargs"]
            resolved_optimizer_kwargs = runtime["resolved_optimizer_kwargs"]
            resolved_guide_kwargs = runtime["resolved_guide_kwargs"]
            resolved_logger_kwargs = runtime["resolved_logger_kwargs"]
            if not _has_trainables(bundle["param"]):
                status = "failed"
                feedback = "no_trainable_parameters"
            else:
                initial_state = _snapshot_model_state(bundle["param"])
                initial = _evaluate_bundle(bundle)
                score_initial = initial.get("score")
                train_result = _train_bundle(
                    bundle, job.trainer, job.params, self.config.mode,
                )
                status = train_result.get("status", "ok")
                resolved_optimizer = train_result.get("resolved_optimizer", resolved_optimizer)
                resolved_guide = train_result.get("resolved_guide", resolved_guide)
                resolved_logger = train_result.get("resolved_logger", resolved_logger)
                resolved_optimizer_kwargs = (
                    train_result.get("resolved_optimizer_kwargs")
                    or train_result.get("optimizer_kwargs")
                    or resolved_optimizer_kwargs
                )
                resolved_trainer_kwargs = (
                    train_result.get("resolved_trainer_kwargs")
                    or train_result.get("trainer_kwargs")
                    or resolved_trainer_kwargs
                )
                resolved_guide_kwargs = train_result.get("resolved_guide_kwargs") or resolved_guide_kwargs
                resolved_logger_kwargs = train_result.get("resolved_logger_kwargs") or resolved_logger_kwargs
                if status == "failed":
                    trace = train_result.get("traceback")
                    suffix = f"\n{trace}" if trace else ""
                    feedback = f"training_error: {train_result.get('error', 'unknown')}{suffix}"
                final = _evaluate_bundle(bundle)
                score_final = final.get("score")
                if status != "failed":
                    feedback = final.get("feedback") or feedback

                if isinstance(score_initial, (int, float)) and isinstance(score_final, (int, float)):
                    score_best = max(score_initial, score_final)
                else:
                    score_best = score_final if score_final is not None else score_initial
                final_state = _snapshot_model_state(bundle["param"])
                best_state = _select_best_state(initial_state, final_state, score_initial, score_final)
                usage = _extract_token_usage(resolved_optimizer_kwargs)
                prompt_tokens = usage["prompt_tokens"]
                completion_tokens = usage["completion_tokens"]
                total_tokens = usage["total_tokens"]

        return {
            "status": status,
            "score_initial": score_initial,
            "score_final": score_final,
            "score_best": score_best,
            "feedback": feedback,
            "elapsed": time.time() - start_time,
            "resolved_optimizer": resolved_optimizer,
            "resolved_guide": resolved_guide,
            "resolved_logger": resolved_logger,
            "resolved_trainer_kwargs": resolved_trainer_kwargs,
            "resolved_optimizer_kwargs": resolved_optimizer_kwargs,
            "resolved_guide_kwargs": resolved_guide_kwargs,
            "resolved_logger_kwargs": resolved_logger_kwargs,
            "initial_state": initial_state,
            "final_state": final_state,
            "best_state": best_state,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    # ------------------------------------------------------------------
    # Subprocess job execution (with hard timeout)
    # ------------------------------------------------------------------

    def _run_job_subprocess(self, job: JobSpec, timeout: float, stdout_log: str) -> Dict[str, Any]:
        """Execute a job in a child process with hard-kill timeout.

        The subprocess reloads the bundle from scratch (opto objects are
        not pickle-able) and runs the full job: load -> eval -> train -> eval.
        """
        start_time = time.time()
        fd, result_file = tempfile.mkstemp(suffix=".json", prefix="tb_job_")
        os.close(fd)
        result_tmp = f"{result_file}.tmp"

        status_hint, bundle, bundle_error = self._get_bundle(job.task)
        fallback_runtime = (
            _resolve_runtime_from_bundle(bundle, job.trainer, job.params)
            if bundle is not None
            else _resolve_runtime_without_bundle(job)
        )

        trainer_dict = _trainer_config_to_dict(job.trainer)
        ctx = multiprocessing.get_context("spawn")
        proc = ctx.Process(
            target=_subprocess_job_target,
            args=(
                job.task_id,
                str(self.tasks_root),
                trainer_dict,
                dict(job.params),
                self.config.mode,
                dict(job.task.eval_kwargs or {}),
                result_file,
                stdout_log,
            ),
        )
        proc.start()
        proc.join(timeout=timeout)

        if proc.is_alive():
            # Hard kill: terminate first, then kill if stubborn
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join()
            payload = {
                "status": "failed",
                "score_initial": None,
                "score_final": None,
                "score_best": None,
                "feedback": f"job_timeout: exceeded {timeout}s (process killed)",
                "elapsed": time.time() - start_time,
                "resolved_optimizer": fallback_runtime.get("resolved_optimizer"),
                "resolved_guide": fallback_runtime.get("resolved_guide"),
                "resolved_logger": fallback_runtime.get("resolved_logger"),
                "resolved_trainer_kwargs": fallback_runtime.get("resolved_trainer_kwargs", {}),
                "resolved_optimizer_kwargs": fallback_runtime.get("resolved_optimizer_kwargs", {}),
                "resolved_guide_kwargs": fallback_runtime.get("resolved_guide_kwargs", {}),
                "resolved_logger_kwargs": fallback_runtime.get("resolved_logger_kwargs", {}),
                "initial_state": {},
                "final_state": {},
                "best_state": {},
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        else:
            # Process finished -- read result
            result_path = Path(result_file)
            if result_path.exists():
                try:
                    payload = json.loads(result_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    payload = {
                        "status": "failed",
                        "feedback": f"subprocess result unreadable: {exc}",
                        "elapsed": time.time() - start_time,
                        "resolved_optimizer": fallback_runtime.get("resolved_optimizer"),
                        "resolved_guide": fallback_runtime.get("resolved_guide"),
                        "resolved_logger": fallback_runtime.get("resolved_logger"),
                        "resolved_trainer_kwargs": fallback_runtime.get("resolved_trainer_kwargs", {}),
                        "resolved_optimizer_kwargs": fallback_runtime.get("resolved_optimizer_kwargs", {}),
                        "resolved_guide_kwargs": fallback_runtime.get("resolved_guide_kwargs", {}),
                        "resolved_logger_kwargs": fallback_runtime.get("resolved_logger_kwargs", {}),
                        "initial_state": {},
                        "final_state": {},
                        "best_state": {},
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    }
            else:
                payload = {
                    "status": "failed",
                    "feedback": f"subprocess crashed (no result file); status_hint={status_hint}; reason={bundle_error}",
                    "elapsed": time.time() - start_time,
                    "resolved_optimizer": fallback_runtime.get("resolved_optimizer"),
                    "resolved_guide": fallback_runtime.get("resolved_guide"),
                    "resolved_logger": fallback_runtime.get("resolved_logger"),
                    "resolved_trainer_kwargs": fallback_runtime.get("resolved_trainer_kwargs", {}),
                    "resolved_optimizer_kwargs": fallback_runtime.get("resolved_optimizer_kwargs", {}),
                    "resolved_guide_kwargs": fallback_runtime.get("resolved_guide_kwargs", {}),
                    "resolved_logger_kwargs": fallback_runtime.get("resolved_logger_kwargs", {}),
                    "initial_state": {},
                    "final_state": {},
                    "best_state": {},
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }

        payload.setdefault("resolved_optimizer", fallback_runtime.get("resolved_optimizer"))
        payload.setdefault("resolved_guide", fallback_runtime.get("resolved_guide"))
        payload.setdefault("resolved_logger", fallback_runtime.get("resolved_logger"))
        payload.setdefault("resolved_trainer_kwargs", fallback_runtime.get("resolved_trainer_kwargs", {}))
        payload.setdefault("resolved_optimizer_kwargs", fallback_runtime.get("resolved_optimizer_kwargs", {}))
        payload.setdefault("resolved_guide_kwargs", fallback_runtime.get("resolved_guide_kwargs", {}))
        payload.setdefault("resolved_logger_kwargs", fallback_runtime.get("resolved_logger_kwargs", {}))
        payload.setdefault("initial_state", {})
        payload.setdefault("final_state", {})
        payload.setdefault("best_state", {})
        payload.setdefault("prompt_tokens", 0)
        payload.setdefault("completion_tokens", 0)
        payload.setdefault("total_tokens", 0)

        # Clean up temp file
        try:
            Path(result_file).unlink(missing_ok=True)
        except Exception:
            pass
        try:
            Path(result_tmp).unlink(missing_ok=True)
        except Exception:
            pass

        return payload


__all__ = ["BenchRunner", "RunSummary"]
