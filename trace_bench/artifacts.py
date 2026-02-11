from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import csv
import json
import os
import subprocess
from datetime import datetime
import platform
import sys


@dataclass
class RunArtifacts:
    run_dir: Path
    meta_dir: Path
    jobs_dir: Path

    @property
    def config_snapshot(self) -> Path:
        return self.meta_dir / "config.snapshot.yaml"

    @property
    def env_json(self) -> Path:
        return self.meta_dir / "env.json"

    @property
    def git_json(self) -> Path:
        return self.meta_dir / "git.json"

    @property
    def manifest_json(self) -> Path:
        return self.meta_dir / "manifest.json"

    @property
    def results_csv(self) -> Path:
        return self.run_dir / "results.csv"

    @property
    def summary_json(self) -> Path:
        return self.run_dir / "summary.json"


@dataclass
class JobArtifacts:
    job_dir: Path

    @property
    def job_meta(self) -> Path:
        return self.job_dir / "job_meta.json"

    @property
    def results_json(self) -> Path:
        return self.job_dir / "results.json"

    @property
    def events_jsonl(self) -> Path:
        return self.job_dir / "events.jsonl"

    @property
    def artifacts_dir(self) -> Path:
        return self.job_dir / "artifacts"

    @property
    def tb_dir(self) -> Path:
        return self.job_dir / "tb"


def init_run_dir(runs_dir: str, run_id: str) -> RunArtifacts:
    run_path = Path(runs_dir) / run_id
    meta_dir = run_path / "meta"
    jobs_dir = run_path / "jobs"
    meta_dir.mkdir(parents=True, exist_ok=True)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    return RunArtifacts(run_dir=run_path, meta_dir=meta_dir, jobs_dir=jobs_dir)


def init_job_dir(run_artifacts: RunArtifacts, job_id: str) -> JobArtifacts:
    job_dir = run_artifacts.jobs_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (job_dir / "tb").mkdir(parents=True, exist_ok=True)
    return JobArtifacts(job_dir=job_dir)


def _dump_yaml_or_json(data: Dict[str, Any]) -> str:
    try:
        import yaml  # type: ignore
        return yaml.safe_dump(data, sort_keys=False)
    except Exception:
        return json.dumps(data, indent=2, sort_keys=False)


def write_config_snapshot(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(_dump_yaml_or_json(data), encoding="utf-8")


def _git_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        root = Path(__file__).resolve().parents[1]
        info["repo_root"] = str(root)
        info["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip()
        info["branch"] = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root).decode().strip()
        return info
    except Exception:
        return info


_ENV_ALLOWLIST = {
    "TRACE_DEFAULT_LLM_BACKEND",
    "TRACE_LITELLM_MODEL",
    "TRACE_CUSTOMLLM_MODEL",
    "TRACE_CUSTOMLLM_URL",
    "CUDA_VISIBLE_DEVICES",
    "PYTHONPATH",
}

_ENV_PREFIX_ALLOWLIST = (
    "TRACE_",
    "OPENAI_",
    "ANTHROPIC_",
    "AZURE_",
    "HF_",
    "HUGGINGFACE_",
)

_SENSITIVE_TOKENS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


def _is_allowed_env_key(key: str) -> bool:
    if key in _ENV_ALLOWLIST:
        return True
    return any(key.startswith(prefix) for prefix in _ENV_PREFIX_ALLOWLIST)


def _redact_env_value(key: str, value: str) -> str:
    if any(token in key.upper() for token in _SENSITIVE_TOKENS):
        return "***REDACTED***"
    return value


def write_env_json(path: Path) -> None:
    env: Dict[str, str] = {}
    for key in sorted(os.environ.keys()):
        if _is_allowed_env_key(key):
            env[key] = _redact_env_value(key, os.environ.get(key, ""))
    payload = {
        "captured_at": datetime.utcnow().isoformat() + "Z",
        "env": env,
        "runtime": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_git_json(path: Path) -> None:
    path.write_text(json.dumps(_git_info(), indent=2), encoding="utf-8")

def _dump_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, default=str)


def write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    path.write_text(_dump_json(manifest), encoding="utf-8")


def write_job_meta(path: Path, job_meta: Dict[str, Any]) -> None:
    path.write_text(_dump_json(job_meta), encoding="utf-8")


def write_job_results(path: Path, results: Dict[str, Any]) -> None:
    path.write_text(_dump_json(results), encoding="utf-8")


def append_results_csv(path: Path, fieldnames: List[str], row: Dict[str, Any]) -> None:
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_event(path: Path, event: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


def write_summary(path: Path, summary: Dict[str, Any]) -> None:
    path.write_text(_dump_json(summary), encoding="utf-8")


__all__ = [
    "RunArtifacts",
    "JobArtifacts",
    "init_run_dir",
    "init_job_dir",
    "write_config_snapshot",
    "write_env_json",
    "write_git_json",
    "write_manifest",
    "write_job_meta",
    "write_job_results",
    "append_results_csv",
    "append_event",
    "write_summary",
]
