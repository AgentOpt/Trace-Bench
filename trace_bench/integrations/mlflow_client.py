from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import json
import os

from trace_bench.artifacts import sanitize_for_json


def _has_mlflow() -> bool:
    try:
        import mlflow  # noqa: F401
        return True
    except Exception:
        return False


def _enabled() -> bool:
    # Only enable when explicitly configured.
    return bool(os.environ.get("MLFLOW_TRACKING_URI")) and _has_mlflow()


def _safe_import_mlflow():
    import mlflow  # type: ignore
    return mlflow


def _to_jsonable(value: Any) -> Any:
    return sanitize_for_json(value)


@dataclass
class MLflowRunContext:
    run_id: str
    active_run_id: str


@contextmanager
def bind_active_run(ctx: Optional[MLflowRunContext]):
    """Bind the active MLflow run to the current thread.

    Trace-Bench executes jobs in worker threads when ``max_workers > 1``.
    MLflow active-run state is thread-local, so any MLflow trace spans emitted
    inside those workers (for example by OpenTrace unified telemetry) must
    explicitly re-attach to the parent run in that thread.
    """
    if ctx is None or not _enabled():
        yield
        return

    mlflow = _safe_import_mlflow()
    if mlflow is None:
        yield
        return

    try:
        if os.environ.get("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

        active = mlflow.active_run()
        if active is not None and active.info.run_id == ctx.active_run_id:
            yield
            return

        with mlflow.start_run(run_id=ctx.active_run_id):
            yield
    except Exception:
        # Telemetry should never break the benchmark run.
        yield


def log_run_start(run_dir: str | Path, config_snapshot: Dict[str, Any], env_json: Dict[str, Any], git_json: Dict[str, Any]) -> Optional[MLflowRunContext]:
    """Create/activate an MLflow run and log run-level context.

    Filesystem remains source of truth; MLflow is a mirror.
    No-op if not enabled.
    """
    if not _enabled():
        return None

    mlflow = _safe_import_mlflow()
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    run_path = Path(run_dir)
    run_id = run_path.name

    active = mlflow.start_run(run_name=run_id)
    active_run_id = active.info.run_id

    mlflow.set_tag("trace_bench.run_id", run_id)
    mlflow.set_tag("trace_bench.run_dir", str(run_path))

    # Params: keep shallow + stable.
    for k, v in _to_jsonable(config_snapshot).items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            mlflow.log_param(f"cfg.{k}", v)

    # Artifacts: link canonical files.
    for rel in ["meta/config.snapshot.yaml", "meta/manifest.json", "results.csv", "summary.json", "meta/env.json", "meta/git.json"]:
        p = run_path / rel
        if p.exists():
            try:
                mlflow.log_artifact(str(p), artifact_path="trace_bench")
            except Exception:
                pass

    return MLflowRunContext(run_id=run_id, active_run_id=active_run_id)


def log_job_result(ctx: Optional[MLflowRunContext], job_meta: Dict[str, Any], results_row: Dict[str, Any]) -> None:
    """Log job-level params/metrics; no-op if not enabled or no context."""
    if ctx is None or not _enabled():
        return

    mlflow = _safe_import_mlflow()

    # Tags identify the job.
    job_id = str(job_meta.get("job_id") or results_row.get("job_id") or "")
    task_id = str(job_meta.get("task_id") or results_row.get("task_id") or "")
    trainer_id = str(job_meta.get("trainer_id") or results_row.get("trainer_id") or "")

    try:
        # Keep run-level metrics on the parent run, but isolate per-job tags/params/metrics
        # in nested runs so repeated keys do not collide.
        with mlflow.start_run(run_id=ctx.active_run_id):
            nested_name = f"job:{job_id}" if job_id else "job:unknown"
            with mlflow.start_run(run_name=nested_name, nested=True):
                mlflow.set_tag("trace_bench.parent_run_id", ctx.run_id)
                mlflow.set_tag("trace_bench.job_id", job_id)
                mlflow.set_tag("trace_bench.task_id", task_id)
                mlflow.set_tag("trace_bench.trainer_id", trainer_id)
                mlflow.set_tag("trace_bench.suite", str(job_meta.get("suite") or results_row.get("suite") or ""))

                for key in ("score_initial", "score_final", "score_best", "time_seconds"):
                    v = results_row.get(key)
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(key, float(v))

                mlflow.log_param("status", str(results_row.get("status") or job_meta.get("status") or ""))
                mlflow.log_param(
                    "resolved_trainer_kwargs",
                    json.dumps(_to_jsonable(job_meta.get("resolved_trainer_kwargs") or {}), sort_keys=True),
                )
                mlflow.log_param(
                    "resolved_optimizer_kwargs",
                    json.dumps(_to_jsonable(job_meta.get("resolved_optimizer_kwargs") or {}), sort_keys=True),
                )
    except Exception:
        # Telemetry should never break benchmark execution.
        return


def log_run_end(ctx: Optional[MLflowRunContext], summary_json: Dict[str, Any]) -> None:
    if ctx is None or not _enabled():
        return

    mlflow = _safe_import_mlflow()
    counts = summary_json.get("counts") or {}
    if isinstance(counts, dict):
        for k, v in counts.items():
            if isinstance(v, int):
                try:
                    mlflow.log_metric(f"run.count_{k}", float(v))
                except Exception:
                    pass
    try:
        mlflow.end_run()
    except Exception:
        pass


__all__ = ["MLflowRunContext", "bind_active_run", "log_run_start", "log_job_result", "log_run_end"]

