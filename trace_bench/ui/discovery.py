from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import csv
import json


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    run_dir: Path
    generated_at: str
    total_jobs: int
    counts: Dict[str, int]

    @property
    def failure_rate(self) -> float:
        total = int(self.total_jobs or 0)
        if total <= 0:
            return 0.0
        failed = int(self.counts.get("failed", 0) or 0)
        return failed / total


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(_read_text(path) or "{}")
    except Exception:
        return {}


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def discover_runs(runs_dir: str | Path) -> List[RunRecord]:
    runs_root = Path(runs_dir)
    if not runs_root.exists():
        return []

    records: List[RunRecord] = []
    for entry in runs_root.iterdir():
        if not entry.is_dir():
            continue
        meta_dir = entry / "meta"
        if not (meta_dir / "config.snapshot.yaml").exists() and not (entry / "config.snapshot.yaml").exists():
            continue

        summary = _read_json(entry / "summary.json")
        manifest = _read_json(meta_dir / "manifest.json")
        generated_at = str(manifest.get("generated_at") or "")
        total_jobs = int(summary.get("total_jobs") or 0)
        counts = summary.get("counts") or {}
        if not isinstance(counts, dict):
            counts = {}

        records.append(
            RunRecord(
                run_id=entry.name,
                run_dir=entry,
                generated_at=generated_at,
                total_jobs=total_jobs,
                counts={str(k): int(v) for k, v in counts.items() if isinstance(k, str)},
            )
        )

    records.sort(key=lambda r: r.run_dir.stat().st_mtime if r.run_dir.exists() else 0, reverse=True)
    return records


def load_run_summary(run_dir: str | Path) -> Dict[str, Any]:
    run_path = Path(run_dir)
    meta_dir = run_path / "meta"
    summary = _read_json(run_path / "summary.json")
    manifest = _read_json(meta_dir / "manifest.json")
    results_rows = _read_csv(run_path / "results.csv")
    config_text = _read_text(meta_dir / "config.snapshot.yaml") or _read_text(run_path / "config.snapshot.yaml")
    env_text = _read_text(meta_dir / "env.json")
    return {
        "run_dir": str(run_path),
        "summary": summary,
        "manifest": manifest,
        "results_rows": results_rows,
        "config_text": config_text,
        "env_text": env_text,
    }


def load_job_details(run_dir: str | Path, job_id: str) -> Dict[str, Any]:
    run_path = Path(run_dir)
    job_dir = run_path / "jobs" / job_id
    job_meta = _read_json(job_dir / "job_meta.json")
    job_results = _read_json(job_dir / "results.json")
    events_path = job_dir / "events.jsonl"
    events_head = ""
    try:
        if events_path.exists():
            with events_path.open("r", encoding="utf-8") as f:
                for _ in range(50):
                    line = f.readline()
                    if not line:
                        break
                    events_head += line
    except Exception:
        events_head = ""
    return {
        "job_dir": str(job_dir),
        "job_meta": job_meta,
        "job_results": job_results,
        "events_head": events_head,
        "tb_dir": str(job_dir / "tb"),
    }


def filter_results_rows(
    rows: List[Dict[str, Any]],
    suite: str = "",
    status: str = "",
    trainer_id: str = "",
    task_substring: str = "",
) -> List[Dict[str, Any]]:
    def _eq(value: str, want: str) -> bool:
        return (not want) or (value == want)

    def _contains(value: str, needle: str) -> bool:
        return (not needle) or (needle.lower() in value.lower())

    out: List[Dict[str, Any]] = []
    for r in rows or []:
        if not _eq(str(r.get("suite", "")), suite):
            continue
        if not _eq(str(r.get("status", "")), status):
            continue
        if not _eq(str(r.get("trainer_id", "")), trainer_id):
            continue
        if not _contains(str(r.get("task_id", "")), task_substring):
            continue
        out.append(r)
    return out


__all__ = [
    "RunRecord",
    "discover_runs",
    "load_run_summary",
    "load_job_details",
    "filter_results_rows",
]

