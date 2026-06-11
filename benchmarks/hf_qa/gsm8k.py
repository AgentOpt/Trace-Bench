from __future__ import annotations

from typing import Any, Dict, List, Tuple

try:
    from .common import HFQATask, extract_last_number, response_text
except ImportError:
    from common import HFQATask, extract_last_number, response_text

try:
    from opto.trainer.guide import Guide
except Exception:
    Guide = object  # type: ignore[assignment,misc]


def row_to_tasks(row: Dict[str, Any], cfg: Dict[str, Any]) -> List[HFQATask]:
    """Convert one GSM8K row into a numeric-answer task."""
    question_field = cfg.get("question_field", "question")
    answer_field = cfg.get("answer_field", "answer")
    if question_field not in row:
        raise KeyError(f"GSM8K row missing question field {question_field!r}")
    if answer_field not in row:
        raise KeyError(f"GSM8K row missing answer field {answer_field!r}")

    raw_answer = str(row[answer_field])
    gold = raw_answer.split("####")[-1].strip()
    return [
        HFQATask(
            question=str(row[question_field]),
            context="",
            answer=gold,
        )
    ]


def format_context(raw: Any) -> str:
    """Return an empty context because GSM8K is question-only."""
    return ""


def make_dataset(tasks: List[HFQATask]) -> Dict[str, List[Any]]:
    """Return the question-only dataset shape used by BBEH-style agents."""
    return {"inputs": [task.question for task in tasks], "infos": list(tasks)}


class GSM8KGuide(Guide):  # type: ignore[misc]
    """Guide for exact final-number GSM8K scoring."""

    def get_feedback(
        self, task: Any, response: Any, info: HFQATask, **kwargs: Any
    ) -> Tuple[float, str]:
        """Score a response by comparing the final extracted number."""
        response_str = response_text(response)
        prediction = extract_last_number(response_str)
        gold = extract_last_number(info.answer)
        if prediction and prediction == gold:
            return 1.0, "The numeric answer is correct."
        return 0.0, (
            f"The numeric answer is wrong. Expected final number: '{gold}', got '{prediction or response_str}'. "
            "Improve arithmetic reasoning and answer extraction so the final number is correct."
        )


def make_guide() -> Any:
    """Return an instantiated GSM8K guide."""
    return GSM8KGuide()
