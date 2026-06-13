"""Harbor CLI wrapper for Terminal-Bench evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import os
import shutil
import subprocess
import time


@dataclass
class HarborEvalResult:
    score: float
    status: str
    job_dir: Path
    trial_dir: Optional[Path]
    job_result: Dict[str, Any]
    trial_result: Dict[str, Any]
    trajectory_path: Optional[Path]
    stdout: str
    stderr: str
    cmd: List[str]


def harbor_available() -> bool:
    return shutil.which("harbor") is not None


def default_env_type() -> str:
    """Prefer Daytona when an API key is present (useful for Colab)."""

    return "daytona" if os.environ.get("DAYTONA_API_KEY") else "docker"


def build_harbor_cmd(
    *,
    job_name: str,
    jobs_dir: Path,
    model_name: str,
    env_type: str,
    task_name: str,
    dataset: Optional[str] = None,
    task_path: Optional[Path] = None,
    agent_name: Optional[str] = None,
    agent_import_path: Optional[
        str
    ] = "trace_bench.terminal_bench.agent:TraceBenchTerminus2",
    agent_kwargs: Optional[Dict[str, Any]] = None,
    n_concurrent: int = 1,
    quiet: bool = True,
    extra_env: Optional[Dict[str, str]] = None,
) -> Tuple[List[str], Dict[str, str]]:
    """Return (cmd, env) for `harbor run`.

    Exactly one of (dataset, task_path) must be provided.
    """

    if (dataset is None) == (task_path is None):
        raise ValueError("Provide exactly one of dataset or task_path")

    cmd: List[str] = [
        "harbor",
        "run",
        "--job-name",
        job_name,
        "--jobs-dir",
        str(jobs_dir),
    ]

    if task_path is not None:
        cmd += ["--path", str(task_path)]
    else:
        cmd += [
            "--dataset",
            str(dataset),
            # Current Harbor docs expose this as --task-name / -t.
            # Older internal builds may have accepted --include-task-name, but
            # using the public CLI name makes this adapter less brittle.
            "--task-name",
            task_name,
            "--n-tasks",
            "1",
        ]

    if agent_name:
        cmd += ["--agent", agent_name]
    cmd += ["--model", model_name]
    if agent_import_path:
        cmd += ["--agent-import-path", agent_import_path]

    for k, v in (agent_kwargs or {}).items():
        cmd += ["--agent-kwarg", f"{k}={v}"]

    cmd += ["--env", env_type]
    cmd += ["--n-concurrent", str(int(n_concurrent))]

    if quiet:
        cmd += ["--quiet"]

    env = dict(os.environ)
    env.update(extra_env or {})
    return cmd, env


def _latest_subdir(path: Path) -> Optional[Path]:
    subs = [p for p in path.iterdir() if p.is_dir()]
    if not subs:
        return None
    subs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return subs[0]


def _find_job_dir(jobs_dir: Path, job_name: str) -> Path:
    direct = jobs_dir / job_name
    if direct.exists():
        return direct

    latest = _latest_subdir(jobs_dir)
    if latest is None:
        raise FileNotFoundError(f"No job directory created under {jobs_dir}")
    return latest


def _find_trial_dir(job_dir: Path) -> Optional[Path]:
    trials = [p for p in job_dir.iterdir() if p.is_dir() and p.name.startswith("trial")]
    if not trials:
        troot = job_dir / "trials"
        if troot.exists() and troot.is_dir():
            trials = [p for p in troot.iterdir() if p.is_dir()]
    if not trials:
        # Harbor 0.6 writes trial directories as <task-name>__<short-id>.
        trials = [
            p for p in job_dir.iterdir() if p.is_dir() and (p / "result.json").exists()
        ]
    if not trials:
        return None
    trials.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return trials[0]


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _run_harbor_subprocess(
    cmd: List[str],
    *,
    env: Dict[str, str],
    timeout_seconds: Optional[float],
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd, text=True, capture_output=True, env=env, timeout=timeout_seconds
        )
    except subprocess.TimeoutExpired as exc:
        stdout = (
            exc.stdout.decode("utf-8", errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        stderr = (
            exc.stderr.decode("utf-8", errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        if stderr:
            stderr += "\n"
        stderr += f"Harbor CLI timed out after {timeout_seconds} seconds"
        return subprocess.CompletedProcess(
            cmd, returncode=124, stdout=stdout, stderr=stderr
        )


def _extract_score(job_result: Dict[str, Any], trial_result: Dict[str, Any]) -> float:
    candidates: List[Any] = []
    for d in (trial_result, job_result):
        if not isinstance(d, dict):
            continue
        for key in (
            "score",
            "reward",
            "resolved",
            "success",
            "passed",
            "verified",
            "ok",
        ):
            if key in d:
                candidates.append(d[key])
        summary = d.get("summary")
        if isinstance(summary, dict):
            for key in ("score", "reward", "resolved", "success"):
                if key in summary:
                    candidates.append(summary[key])

    for val in candidates:
        if isinstance(val, bool):
            return 1.0 if val else 0.0
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            if val.lower() in {"true", "ok", "pass", "passed", "success"}:
                return 1.0
            if val.lower() in {"false", "fail", "failed"}:
                return 0.0
            try:
                return float(val)
            except Exception:
                continue

    return 0.0


def run_harbor_eval(
    *,
    job_name: str,
    jobs_dir: Path,
    model_name: str,
    env_type: str,
    task_name: str,
    dataset: Optional[str] = None,
    task_path: Optional[Path] = None,
    agent_name: Optional[str] = None,
    agent_import_path: Optional[
        str
    ] = "trace_bench.terminal_bench.agent:TraceBenchTerminus2",
    agent_kwargs: Optional[Dict[str, Any]] = None,
    n_concurrent: int = 1,
    quiet: bool = True,
    timeout_seconds: Optional[float] = None,
    extra_env: Optional[Dict[str, str]] = None,
) -> HarborEvalResult:
    if not harbor_available():
        raise RuntimeError(
            "Harbor CLI not found on PATH. Install Harbor or ensure the `harbor` "
            "binary is available before running Terminal-Bench tasks."
        )

    jobs_dir.mkdir(parents=True, exist_ok=True)

    cmd, env = build_harbor_cmd(
        job_name=job_name,
        jobs_dir=jobs_dir,
        model_name=model_name,
        env_type=env_type,
        task_name=task_name,
        dataset=dataset,
        task_path=task_path,
        agent_name=agent_name,
        agent_import_path=agent_import_path,
        agent_kwargs=agent_kwargs,
        n_concurrent=n_concurrent,
        quiet=quiet,
        extra_env=extra_env,
    )

    start = time.time()
    proc = _run_harbor_subprocess(cmd, env=env, timeout_seconds=timeout_seconds)

    # Compatibility fallback for Harbor 0.6.x builds that still expose dataset
    # filtering as --include-task-name instead of --task-name. Keep
    # build_harbor_cmd() on the documented/public flag while still allowing
    # older CLIs to run real smoke tests.
    if (
        proc.returncode != 0
        and dataset is not None
        and task_path is None
        and "--task-name" in cmd
    ):
        compat_cmd = list(cmd)
        compat_cmd[compat_cmd.index("--task-name")] = "--include-task-name"
        proc2 = _run_harbor_subprocess(
            compat_cmd, env=env, timeout_seconds=timeout_seconds
        )
        if proc2.returncode == 0:
            cmd, proc = compat_cmd, proc2

    if proc.returncode != 0 and agent_import_path and ":" in agent_import_path:
        mod, cls = agent_import_path.split(":", 1)
        fallback_agent_name = cls if agent_name is None else agent_name
        fallback_job_name = f"{job_name}-retry1"
        try:
            cmd2, env2 = build_harbor_cmd(
                job_name=fallback_job_name,
                jobs_dir=jobs_dir,
                model_name=model_name,
                env_type=env_type,
                task_name=task_name,
                dataset=dataset,
                task_path=task_path,
                agent_name=fallback_agent_name,
                agent_import_path=mod,
                agent_kwargs=agent_kwargs,
                n_concurrent=n_concurrent,
                quiet=quiet,
                extra_env=extra_env,
            )
            proc2 = _run_harbor_subprocess(
                cmd2, env=env2, timeout_seconds=timeout_seconds
            )
            if proc2.returncode == 0:
                cmd, env, proc, job_name = cmd2, env2, proc2, fallback_job_name
        except Exception:
            pass

    elapsed = time.time() - start

    try:
        job_dir = _find_job_dir(jobs_dir, job_name)
    except FileNotFoundError:
        job_dir = jobs_dir / job_name
    trial_dir = _find_trial_dir(job_dir) if job_dir.exists() else None

    job_result = _read_json(job_dir / "result.json")
    trial_result: Dict[str, Any] = {}
    trajectory_path: Optional[Path] = None
    if trial_dir is not None:
        trial_result = _read_json(trial_dir / "result.json")
        cand = list(trial_dir.glob("**/trajectory.json"))
        trajectory_path = cand[0] if cand else None

    score = _extract_score(job_result, trial_result)
    has_trial_exception = bool(
        isinstance(trial_result, dict) and trial_result.get("exception_info")
    )
    stats = job_result.get("stats") if isinstance(job_result, dict) else None
    has_job_errors = bool(
        isinstance(stats, dict) and int(stats.get("n_errored_trials") or 0) > 0
    )
    status = (
        "ok"
        if proc.returncode == 0 and not has_trial_exception and not has_job_errors
        else "failed"
    )

    if status == "failed" and not job_result:
        job_result = {
            "trace_bench": {
                "failure_reason": "harbor_failed_without_job_result",
                "returncode": proc.returncode,
            }
        }

    if isinstance(job_result, dict) and "trace_bench" not in job_result:
        job_result["trace_bench"] = {"wall_seconds": elapsed}

    return HarborEvalResult(
        score=score,
        status=status,
        job_dir=job_dir,
        trial_dir=trial_dir,
        job_result=job_result,
        trial_result=trial_result,
        trajectory_path=trajectory_path,
        stdout=proc.stdout,
        stderr=proc.stderr,
        cmd=cmd,
    )


__all__ = [
    "HarborEvalResult",
    "harbor_available",
    "default_env_type",
    "build_harbor_cmd",
    "run_harbor_eval",
]
