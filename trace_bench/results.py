from __future__ import annotations

from typing import Any, Dict, List
import json

from trace_bench.artifacts import sanitize_for_json


RESULT_COLUMNS = [
    "run_id",
    "job_id",
    "task_id",
    "suite",
    "trainer_id",
    "seed",
    "status",
    "score_val_initial",
    "score_val_final",
    "score_val",
    "score_test",
    "time_seconds",
    "resolved_optimizer",
    "resolved_guide",
    "resolved_logger",
    "resolved_trainer_kwargs",
    "resolved_optimizer_kwargs",
    "resolved_guide_kwargs",
    "resolved_logger_kwargs",
    "eval_kwargs",
    "feedback",
    "llm_provider",
    "llm_model",
    "llm_base_url",
    "token_scope",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "initial_state_path",
    "best_state_path",
    "final_state_path",
    "state_history_path",
    "tb_logdir",
]


def _json_cell(value: Any) -> str:
    return json.dumps(sanitize_for_json(value), sort_keys=True, ensure_ascii=False)


def build_results_row(
    run_id: str,
    job_id: str,
    task_id: str,
    suite: str,
    trainer_id: str,
    seed: int,
    status: str,
    score_val_initial: Any,
    score_val_final: Any,
    score_val: Any,
    score_test: Any,
    time_seconds: float,
    resolved_optimizer: Any,
    resolved_guide: Any,
    resolved_logger: Any,
    resolved_trainer_kwargs: Dict[str, Any],
    resolved_optimizer_kwargs: Dict[str, Any],
    resolved_guide_kwargs: Dict[str, Any],
    resolved_logger_kwargs: Dict[str, Any],
    eval_kwargs: Dict[str, Any],
    feedback: str | None,
    llm_provider: str,
    llm_model: str,
    llm_base_url: str,
    token_scope: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
    initial_state_path: str,
    best_state_path: str,
    final_state_path: str,
    state_history_path: str,
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
        "score_val_initial": score_val_initial,
        "score_val_final": score_val_final,
        "score_val": score_val,
        "score_test": score_test,
        "time_seconds": round(time_seconds, 6),
        "resolved_optimizer": resolved_optimizer,
        "resolved_guide": resolved_guide,
        "resolved_logger": resolved_logger,
        "resolved_trainer_kwargs": resolved_trainer_kwargs,
        "resolved_optimizer_kwargs": resolved_optimizer_kwargs,
        "resolved_guide_kwargs": resolved_guide_kwargs,
        "resolved_logger_kwargs": resolved_logger_kwargs,
        "eval_kwargs": eval_kwargs,
        "feedback": feedback or "",
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "llm_base_url": llm_base_url,
        "token_scope": token_scope,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "initial_state_path": initial_state_path,
        "best_state_path": best_state_path,
        "final_state_path": final_state_path,
        "state_history_path": state_history_path,
        "tb_logdir": tb_logdir,
    }


def build_results_csv_row(row: Dict[str, Any]) -> Dict[str, Any]:
    csv_row = dict(row)
    csv_row["resolved_optimizer"] = str(row.get("resolved_optimizer"))
    csv_row["resolved_guide"] = str(row.get("resolved_guide"))
    csv_row["resolved_logger"] = str(row.get("resolved_logger"))
    csv_row["resolved_trainer_kwargs"] = _json_cell(row.get("resolved_trainer_kwargs"))
    csv_row["resolved_optimizer_kwargs"] = _json_cell(row.get("resolved_optimizer_kwargs"))
    csv_row["resolved_guide_kwargs"] = _json_cell(row.get("resolved_guide_kwargs"))
    csv_row["resolved_logger_kwargs"] = _json_cell(row.get("resolved_logger_kwargs"))
    csv_row["eval_kwargs"] = _json_cell(row.get("eval_kwargs"))
    csv_row["llm_provider"] = str(row.get("llm_provider") or "")
    csv_row["llm_model"] = str(row.get("llm_model") or "")
    csv_row["llm_base_url"] = str(row.get("llm_base_url") or "")
    csv_row["token_scope"] = str(row.get("token_scope") or "")
    csv_row["initial_state_path"] = str(row.get("initial_state_path") or "")
    csv_row["best_state_path"] = str(row.get("best_state_path") or "")
    csv_row["final_state_path"] = str(row.get("final_state_path") or "")
    csv_row["state_history_path"] = str(row.get("state_history_path") or "")
    return csv_row


def build_leaderboard_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def _score(v: Any) -> float:
        try:
            return float(v)
        except Exception:
            return float("-inf")

    # Keep the best run per (task_id, trainer_id, resolved_optimizer) so every
    # distinct trainer/optimizer combination appears once per task.
    # resolved_optimizer distinguishes e.g. dspy.MIPROv2 vs dspy.SIMBA which
    # both carry trainer_id="DSPyTrainer".
    best_by_key: Dict[str, Dict[str, Any]] = {}
    for row in rows or []:
        if str(row.get("status")) != "ok":
            continue
        task_id = str(row.get("task_id") or "")
        trainer_id = str(row.get("trainer_id") or "")
        resolved_optimizer = str(row.get("resolved_optimizer") or "")
        if not task_id:
            continue
        key = f"{task_id}\x00{trainer_id}\x00{resolved_optimizer}"
        current = best_by_key.get(key)
        if current is None or _score(row.get("score_val")) > _score(current.get("score_val")):
            best_by_key[key] = row

    # Sort by task_id then descending score_val so the winner is listed first
    # within each task group.
    sorted_rows = sorted(
        best_by_key.values(),
        key=lambda r: (str(r.get("task_id") or ""), -_score(r.get("score_val"))),
    )
    out: List[Dict[str, Any]] = []
    for rank, row in enumerate(sorted_rows, start=1):
        out.append({
            "rank": rank,
            "task_id": row.get("task_id"),
            "suite": row.get("suite"),
            "job_id": row.get("job_id"),
            "trainer_id": row.get("trainer_id"),
            "resolved_optimizer": row.get("resolved_optimizer"),
            "score_val": row.get("score_val"),
            "score_test": row.get("score_test"),
            "time_seconds": row.get("time_seconds"),
        })
    return out


def summarize_results(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, int] = {"ok": 0, "failed": 0, "skipped": 0}
    token_totals: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for row in rows:
        status = row.get("status") or "ok"
        if status not in counts:
            counts[status] = 0
        counts[status] += 1
        for k in token_totals.keys():
            try:
                token_totals[k] += int(row.get(k) or 0)
            except Exception:
                pass
    return {"counts": counts, "total_jobs": len(rows), "token_totals": token_totals}


__all__ = [
    "RESULT_COLUMNS",
    "build_results_row",
    "build_results_csv_row",
    "build_leaderboard_rows",
    "summarize_results",
]
