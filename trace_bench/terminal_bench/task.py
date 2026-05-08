"""Terminal-Bench task adapter for Trace-Bench."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import json
import os
import shutil

from trace_bench.terminal_bench import DEFAULT_HARBOR_DATASET
from trace_bench.terminal_bench.atif import (
    atif_feedback_snippet,
    atif_to_trace_like,
    load_atif,
)
from trace_bench.terminal_bench.harbor_runner import default_env_type, run_harbor_eval

try:
    from opto import trace
    from opto.trainer.guide import Guide
except Exception:  # pragma: no cover
    trace = None  # type: ignore
    Guide = object  # type: ignore


_DEFAULT_PROMPT_TEMPLATE = """You are an autonomous terminal agent solving command-line tasks in a sandboxed Linux environment.

Task Description:
{instruction}

Current terminal state:
{terminal_state}

Rules:
- Be precise and avoid unnecessary commands.
- Inspect files before editing.
- Run the task's tests or verifier when appropriate.
- Respond as JSON matching this schema:

{{
  "analysis": "What you observed and what remains to do.",
  "plan": "Your next steps.",
  "commands": [
    {{"keystrokes": "ls -la\n", "duration": 0.1}}
  ],
  "task_complete": false
}}

Every command must include a trailing newline in `keystrokes`. Use an empty
commands array if you only need to wait or if the task is complete.
"""


if trace is not None:

    @trace.model  # type: ignore[misc]
    class PromptTemplateAgent:
        """A trivial model whose output is a trainable prompt template string."""

        def __init__(self, initial_template: str = _DEFAULT_PROMPT_TEMPLATE):
            self.template = trace.node(initial_template, trainable=True)  # type: ignore[attr-defined]

        def __call__(self, _task_name: str) -> str:
            return self.emit(self.template)

        @trace.bundle(trainable=True)  # type: ignore[attr-defined]
        def emit(self, template: str) -> str:
            return template

else:  # pragma: no cover

    class PromptTemplateAgent:  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("opto.trace is required to build Terminal-Bench tasks")


class TerminalBenchGuide(Guide):  # type: ignore[misc]
    """Guide that evaluates prompt templates by running Harbor on a TB task."""

    def __init__(self, *, eval_kwargs: Optional[Dict[str, Any]] = None):
        self.eval_kwargs = eval_kwargs or {}
        self._context: Dict[str, Any] = {}
        self._eval_count = 0

    def set_trace_bench_context(self, **ctx: Any) -> None:
        self._context.update(ctx)
        self._eval_count = 0

    @staticmethod
    def _write_json(path: Path, payload: Dict[str, Any]) -> None:
        """Write stable machine-readable JSON.

        Avoid json.dumps(..., default=str): it can silently serialize Python
        objects with memory addresses and make artifacts non-reproducible.
        """
        from trace_bench.artifacts import sanitize_for_json

        path.write_text(
            json.dumps(sanitize_for_json(payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _mode(self) -> str:
        return (
            self._context.get("mode")
            or os.environ.get("TRACE_BENCH_MODE")
            or self.eval_kwargs.get("mode")
            or "real"
        ).lower()

    def _job_dir(self) -> Optional[Path]:
        jd = self._context.get("job_dir")
        if jd is None:
            return None
        return Path(jd)

    def _artifact_dir(self) -> Optional[Path]:
        ad = self._context.get("artifacts_dir")
        if ad is not None:
            return Path(ad)
        jd = self._job_dir()
        return (jd / "artifacts") if jd else None

    def get_feedback(self, query, response, reference, **_kwargs):  # type: ignore[override]
        task_name = str(query)
        prompt_template = str(response)

        if self._mode() == "stub":
            score = 0.0
            feedback = f"STUB: terminal_bench task={task_name} (harbor not invoked)"
            return score, feedback

        model_name = self.eval_kwargs.get("harbor_model") or os.environ.get(
            "HARBOR_MODEL"
        )
        if not model_name:
            raise RuntimeError(
                "Terminal-Bench real mode requires a Harbor model name. "
                "Set eval_kwargs.harbor_model or env var HARBOR_MODEL."
            )

        env_type = self.eval_kwargs.get("harbor_env") or default_env_type()
        dataset = self.eval_kwargs.get("harbor_dataset", DEFAULT_HARBOR_DATASET)
        task_path = self.eval_kwargs.get("harbor_task_path")

        artifact_dir = self._artifact_dir()
        if artifact_dir is None:
            artifact_dir = Path.cwd() / "artifacts"
        tb_art_dir = artifact_dir / "terminal_bench"
        tb_art_dir.mkdir(parents=True, exist_ok=True)

        template_path = tb_art_dir / "prompt_template.txt"
        template_path.write_text(prompt_template, encoding="utf-8")

        self._eval_count += 1
        job_name = (
            f"tracebench-{self._context.get('job_id','job')}-eval{self._eval_count:04d}"
        )
        jobs_dir = tb_art_dir / "harbor_jobs"

        agent_kwargs = dict(self.eval_kwargs.get("harbor_agent_kwargs") or {})
        agent_kwargs.setdefault("prompt_template_path", str(template_path))
        extra_env = dict(self.eval_kwargs.get("harbor_extra_env") or {})
        timeout_seconds = self.eval_kwargs.get("harbor_cli_timeout_seconds")

        result = run_harbor_eval(
            job_name=job_name,
            jobs_dir=jobs_dir,
            model_name=model_name,
            env_type=env_type,
            task_name=task_name,
            dataset=None if task_path else dataset,
            task_path=Path(task_path) if task_path else None,
            agent_kwargs=agent_kwargs,
            timeout_seconds=timeout_seconds,
            extra_env=extra_env,
        )

        (tb_art_dir / f"{job_name}.cmd.txt").write_text(
            " ".join(result.cmd), encoding="utf-8"
        )
        (tb_art_dir / f"{job_name}.stdout.txt").write_text(
            result.stdout or "", encoding="utf-8"
        )
        (tb_art_dir / f"{job_name}.stderr.txt").write_text(
            result.stderr or "", encoding="utf-8"
        )
        self._write_json(tb_art_dir / f"{job_name}.job_result.json", result.job_result)
        if result.trial_result:
            self._write_json(
                tb_art_dir / f"{job_name}.trial_result.json", result.trial_result
            )

        snippet = ""
        if result.trajectory_path and result.trajectory_path.exists():
            dst = tb_art_dir / f"{job_name}.trajectory.json"
            shutil.copyfile(result.trajectory_path, dst)
            atif_payload = load_atif(dst)
            trace_like = atif_to_trace_like(atif_payload)
            self._write_json(
                tb_art_dir / f"{job_name}.trajectory_trace.json", trace_like
            )
            snippet = atif_feedback_snippet(atif_payload)

        feedback_parts = [
            f"terminal_bench task={task_name}",
            f"score={result.score}",
            f"env={env_type}",
        ]
        if snippet:
            feedback_parts.append("\n" + snippet)
        feedback = " ".join(feedback_parts)

        return float(result.score), feedback


def build_trace_problem(task_slug: str, **override_eval_kwargs: Any) -> Dict[str, Any]:
    """Build a Trace problem bundle for a single Terminal-Bench task."""

    if trace is None:
        raise RuntimeError("opto.trace is required to build Terminal-Bench tasks")

    guide = TerminalBenchGuide(eval_kwargs=override_eval_kwargs)
    agent = PromptTemplateAgent()

    train_dataset = dict(inputs=[task_slug], infos=[None])
    optimizer_kwargs = dict(
        objective=(
            "Optimize the Terminus-2 prompt template to solve the given Terminal-Bench task "
            "with minimal errors and maximal success."
        ),
        memory_size=5,
    )

    return dict(
        param=agent,
        guide=guide,
        train_dataset=train_dataset,
        optimizer_kwargs=optimizer_kwargs,
        metadata=dict(benchmark="terminal_bench", entry=task_slug),
    )


def build_terminal_bench_bundle(
    task_id: str, eval_kwargs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    if not task_id.startswith("terminal_bench:"):
        raise ValueError(f"Not a terminal_bench task id: {task_id}")
    task_slug = task_id.split(":", 1)[1]
    return build_trace_problem(task_slug, **(eval_kwargs or {}))


__all__ = [
    "PromptTemplateAgent",
    "TerminalBenchGuide",
    "build_trace_problem",
    "build_terminal_bench_bundle",
]
