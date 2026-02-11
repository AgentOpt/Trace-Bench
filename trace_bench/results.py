from __future__ import annotations

from typing import Any, Dict, List
import json


RESULT_COLUMNS = [
    "run_id",
    "job_id",
    "task_id",
    "suite",
    "trainer_id",
    "seed",
    "status",
    "score_initial",
    "score_final",
    "score_best",
    "time_seconds",
    "resolved_trainer_kwargs",
    "resolved_optimizer_kwargs",
    "eval_kwargs",
    "feedback",
    "tb_logdir",
]


def _json_cell(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True)
    except Exception:
        return json.dumps(str(value))


def build_results_row(
    run_id: str,
    job_id: str,
    task_id: str,
    suite: str,
    trainer_id: str,
    seed: int,
    status: str,
    score_initial: Any,
    score_final: Any,
    score_best: Any,
    time_seconds: float,
    resolved_trainer_kwargs: Dict[str, Any],
    resolved_optimizer_kwargs: Dict[str, Any],
    eval_kwargs: Dict[str, Any],
    feedback: str | None,
    tb_logdir: str,
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "job_id": job_id,
        "task_id": task_id,
        "suite": suite,
        "trainer_id": trainer_id,
        "seed": seed,
        "status": status,
        "score_initial": score_initial,
        "score_final": score_final,
        "score_best": score_best,
        "time_seconds": round(time_seconds, 6),
        "resolved_trainer_kwargs": resolved_trainer_kwargs,
        "resolved_optimizer_kwargs": resolved_optimizer_kwargs,
        "eval_kwargs": eval_kwargs,
        "feedback": feedback or "",
        "tb_logdir": tb_logdir,
    }


def build_results_csv_row(row: Dict[str, Any]) -> Dict[str, Any]:
    csv_row = dict(row)
    csv_row["resolved_trainer_kwargs"] = _json_cell(row.get("resolved_trainer_kwargs"))
    csv_row["resolved_optimizer_kwargs"] = _json_cell(row.get("resolved_optimizer_kwargs"))
    csv_row["eval_kwargs"] = _json_cell(row.get("eval_kwargs"))
    return csv_row


def summarize_results(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, int] = {"ok": 0, "failed": 0, "skipped": 0}
    for row in rows:
        status = row.get("status") or "ok"
        if status not in counts:
            counts[status] = 0
        counts[status] += 1
    return {"counts": counts, "total_jobs": len(rows)}


__all__ = ["RESULT_COLUMNS", "build_results_row", "build_results_csv_row", "summarize_results"]
