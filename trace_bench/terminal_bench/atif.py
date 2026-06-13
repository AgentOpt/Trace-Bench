"""Utilities for Harbor ATIF trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import json


@dataclass
class ATIFSummary:
    """Compact view of an ATIF trajectory."""

    n_steps: int
    last_steps: List[Dict[str, Any]]


def load_atif(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def _get_steps(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Best-effort extraction of step list across possible versions."""

    steps = payload.get("steps")
    if isinstance(steps, list):
        return [s for s in steps if isinstance(s, dict)]

    nested = payload.get("trajectory")
    if isinstance(nested, dict):
        steps = nested.get("steps")
        if isinstance(steps, list):
            return [s for s in steps if isinstance(s, dict)]

    return []


def summarize_atif(payload: Dict[str, Any], last_n: int = 6) -> ATIFSummary:
    steps = _get_steps(payload)
    n_steps = len(steps)
    if last_n <= 0:
        return ATIFSummary(n_steps=n_steps, last_steps=[])
    return ATIFSummary(n_steps=n_steps, last_steps=steps[-last_n:])


def atif_to_trace_like(payload: Dict[str, Any], last_n: int = 12) -> Dict[str, Any]:
    """Convert ATIF to a Trace-ish, stable JSON schema."""

    steps = _get_steps(payload)
    steps = steps[-last_n:] if last_n > 0 else steps

    def norm_tool_calls(step: Dict[str, Any]) -> List[Dict[str, Any]]:
        tool_calls = step.get("tool_calls") or step.get("toolCalls")
        if not isinstance(tool_calls, list):
            return []
        out: List[Dict[str, Any]] = []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            out.append(
                {
                    "name": tc.get("name") or tc.get("tool_name") or tc.get("tool"),
                    "args": tc.get("args") or tc.get("arguments"),
                    "id": tc.get("id") or tc.get("call_id"),
                }
            )
        return out

    events: List[Dict[str, Any]] = []
    for i, step in enumerate(steps):
        events.append(
            {
                "i": i,
                "role": step.get("role") or step.get("type") or step.get("speaker"),
                "content": step.get("content")
                or step.get("message")
                or step.get("text"),
                "tool_calls": norm_tool_calls(step),
                "observation": step.get("observation") or step.get("observation_text"),
                "timestamp": step.get("timestamp"),
            }
        )

    return {
        "format": "trace_bench.atif.v1",
        "n_steps": len(_get_steps(payload)),
        "events": events,
        "final_metrics": payload.get("final_metrics") or payload.get("finalMetrics"),
    }


def atif_feedback_snippet(
    payload: Dict[str, Any], max_chars: int = 4000, last_n: int = 6
) -> str:
    """Generate a compact, human/LLM-readable snippet."""

    summary = summarize_atif(payload, last_n=last_n)
    lines: List[str] = []
    lines.append(f"ATIF: steps={summary.n_steps}")
    start_idx = max(0, summary.n_steps - len(summary.last_steps))
    for idx, step in enumerate(summary.last_steps, start=start_idx):
        role = step.get("role") or step.get("type") or "?"
        content = step.get("content") or step.get("message") or step.get("text") or ""
        content = str(content).replace("\n", " ")
        if len(content) > 280:
            content = content[:280] + "…"
        lines.append(f"[{idx}] {role}: {content}")

        tool_calls = step.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            for tc in tool_calls[:3]:
                if not isinstance(tc, dict):
                    continue
                name = tc.get("name") or tc.get("tool_name") or tc.get("tool")
                lines.append(f"    tool_call: {name}")

    out = "\n".join(lines)
    if len(out) > max_chars:
        out = out[: max_chars - 1] + "…"
    return out


__all__ = [
    "ATIFSummary",
    "load_atif",
    "summarize_atif",
    "atif_to_trace_like",
    "atif_feedback_snippet",
]
