"""BBEH task module for the `hf:bbeh/*` tasks.

Provides task-specific data utilities and the guide.  Framework-specific agents
live in ``agents/trace_agent.py`` and ``agents/dspy_agent.py``.

Reference: ``Predict`` / ``BigBenchGuide`` classes in ``bbeh_trace.py``.
"""
from __future__ import annotations

import re
from typing import Any

try:
    from opto.trainer.guide import Guide
    _OPTO_AVAILABLE = True
except Exception:
    _OPTO_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def format_context(raw: Any) -> str:
    """No-op — BBEH tasks have no context field."""
    return ""


def make_dataset(tasks: list) -> dict:
    """Convert a list of HFQATask objects into a BBEH dataset dict.

    BBEHAgent.forward() takes plain question strings, so inputs and infos are
    unpacked from the task objects here rather than in the generic loader.
    """
    return {
        "inputs": [t.question for t in tasks],
        "infos": [t.answer for t in tasks],
    }


def eval_metric(true: str, prediction: str) -> bool:
    """Answer-checking logic translated from ``bbeh_trace.py::eval_metric``.

    - Multiple-choice ``(A)``/``(B)`` — finds the last ``([A-Z])`` token in
      the prediction and compares to gold.
    - Free-form — exact string match after stripping whitespace.
    """
    if re.findall(r"\([A-Z]\)", true):
        matches = re.findall(r"\([A-Z]\)", prediction)
        return (matches[-1] if matches else "") == true
    return prediction.strip() == true.strip()


# ---------------------------------------------------------------------------
# Guide
# ---------------------------------------------------------------------------

if _OPTO_AVAILABLE:
    class BBEHGuide(Guide):
        """Evaluation guide for BigBenchExtraHard tasks.

        Framework-agnostic: works with any agent that produces a string answer.
        Uses ``eval_metric`` which handles both multiple-choice and free-form answers.
        """

        def get_feedback(self, task: str, response: Any, info: str, **kwargs: Any):
            response_str = str(getattr(response, "data", response))
            if eval_metric(info, response_str):
                return 1.0, "The answer is correct."
            return 0.0, (
                f'The answer is wrong. Expected: "{info}", Got: "{response_str}". '
                "Improve the agent to produce the correct answer in the expected format: "
                "(A)/(B) for multiple-choice, or an exact string/number for free-form questions."
            )

else:
    class BBEHGuide:  # type: ignore[no-redef]
        pass


def make_guide() -> Any:
    """Return an instantiated BBEHGuide."""
    return BBEHGuide()
