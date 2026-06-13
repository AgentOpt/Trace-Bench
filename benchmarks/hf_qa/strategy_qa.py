"""StrategyQA — implicit multi-hop yes/no reasoning.

StrategyQA rows carry a boolean ``answer`` and no options field, so they need a
small shaping step (boolean -> Yes/No) rather than the MCQ path. Everything else
(prompt building, response extraction) reuses ``common`` helpers, keeping this
module DRY. This task gives the recursive optimizer a *reasoning-artifact
headroom* surface: the gold answer is one bit, but reaching it requires implicit
decomposition, so a better ``starting_artifact`` / prompt measurably helps.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

try:
    from .common import HFQATask, normalize_text, response_text
except ImportError:
    from common import HFQATask, normalize_text, response_text

try:
    from opto.trainer.guide import Guide
except Exception:
    Guide = object  # type: ignore[assignment,misc]


def _gold_bool(raw: Any) -> str:
    """Normalize StrategyQA's answer (bool / 'yes' / 'true' / 1) to 'yes'|'no'."""
    if isinstance(raw, bool):
        return "yes" if raw else "no"
    text = normalize_text(str(raw))
    if text in {"yes", "true", "1"}:
        return "yes"
    if text in {"no", "false", "0"}:
        return "no"
    return text  # leave unexpected values visible rather than guessing


def row_to_tasks(row: Dict[str, Any], cfg: Dict[str, Any]) -> List[HFQATask]:
    """Convert one StrategyQA row into a yes/no QA task."""
    question_field = cfg.get("question_field", "question")
    answer_field = cfg.get("answer_field", "answer")
    if question_field not in row:
        raise KeyError(f"StrategyQA row missing question field {question_field!r}")
    if answer_field not in row:
        raise KeyError(f"StrategyQA row missing answer field {answer_field!r}")

    gold = _gold_bool(row[answer_field])
    prompt = (
        f"Question: {row[question_field]}\n"
        "Answer with exactly 'Yes' or 'No'.\nAnswer:"
    )
    return [HFQATask(question=prompt, context="", answer=gold)]


def format_context(raw: Any) -> str:
    """StrategyQA is question-only."""
    return ""


def make_dataset(tasks: List[HFQATask]) -> Dict[str, List[Any]]:
    """Question-only dataset shape used by BBEH-style agents."""
    return {"inputs": [task.question for task in tasks], "infos": list(tasks)}


class StrategyQAGuide(Guide):  # type: ignore[misc]
    """Guide for exact yes/no StrategyQA scoring."""

    def get_feedback(
        self, task: Any, response: Any, info: HFQATask, **kwargs: Any
    ) -> Tuple[float, str]:
        """Score by the first yes/no token in the response."""
        text = normalize_text(response_text(response))
        # first explicit yes/no token wins (robust to trailing rationale)
        prediction = ""
        for token in text.replace(".", " ").split():
            if token in {"yes", "no"}:
                prediction = token
                break
        if prediction and prediction == info.answer:
            return 1.0, "The yes/no answer is correct."
        return 0.0, (
            f"The yes/no answer is wrong. Expected '{info.answer}', got "
            f"'{prediction or text[:40]}'. Decompose the implicit reasoning steps "
            "and state the final Yes/No explicitly."
        )


def make_guide() -> Any:
    """Return an instantiated StrategyQA guide."""
    return StrategyQAGuide()
