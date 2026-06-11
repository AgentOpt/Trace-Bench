from __future__ import annotations

from typing import Any, Dict, List, Tuple

try:
    from .common import (
        HFQATask,
        answer_to_choice_label,
        choice_pairs,
        extract_last_choice,
        format_mcq_prompt,
        normalize_text,
        response_text,
    )
except ImportError:
    from common import (
        HFQATask,
        answer_to_choice_label,
        choice_pairs,
        extract_last_choice,
        format_mcq_prompt,
        normalize_text,
        response_text,
    )

try:
    from opto.trainer.guide import Guide
except Exception:
    Guide = object  # type: ignore[assignment,misc]


def row_to_tasks(row: Dict[str, Any], cfg: Dict[str, Any]) -> List[HFQATask]:
    """Convert one multiple-choice row into a prompt-based QA task."""
    question_field = cfg.get("question_field", "question")
    answer_field = cfg.get("answer_field", "answer")
    options_field = cfg.get("options_field", "options")
    context_field = cfg.get("context_field")
    if question_field not in row:
        raise KeyError(f"MCQ row missing question field {question_field!r}")
    if answer_field not in row:
        raise KeyError(f"MCQ row missing answer field {answer_field!r}")

    pairs = choice_pairs(row.get(options_field, row.get("choices")))
    if not pairs:
        raise ValueError(f"MCQ row has no options in field {options_field!r}")

    context = str(row.get(context_field, "")) if context_field else ""
    answer = row[answer_field]
    label = answer_to_choice_label(
        answer, pairs, index_base=int(cfg.get("answer_index_base", 0))
    )
    gold_text = next(
        (text for option_label, text in pairs if option_label == label), ""
    )
    prompt = format_mcq_prompt(str(row[question_field]), pairs, context=context)
    return [
        HFQATask(
            question=prompt,
            context=context,
            answer=label,
            meta={"gold_label": label, "gold_text": gold_text, "options": list(pairs)},
        )
    ]


def format_context(raw: Any) -> str:
    """Return plain-string context for MCQ tasks."""
    return str(raw or "")


def make_dataset(tasks: List[HFQATask]) -> Dict[str, List[Any]]:
    """Return the question-only dataset shape used by BBEH-style agents."""
    return {"inputs": [task.question for task in tasks], "infos": list(tasks)}


class MCQGuide(Guide):  # type: ignore[misc]
    """Guide for multiple-choice label or option-text scoring."""

    def get_feedback(
        self, task: Any, response: Any, info: HFQATask, **kwargs: Any
    ) -> Tuple[float, str]:
        """Score a response against the expected option label or text."""
        response_str = response_text(response)
        pred_label = extract_last_choice(response_str)
        gold_label = str(info.meta.get("gold_label", info.answer))
        gold_text = str(info.meta.get("gold_text", ""))

        if pred_label and pred_label.strip().lower() == gold_label.strip().lower():
            return 1.0, "The selected option is correct."
        if gold_text and normalize_text(gold_text) in normalize_text(response_str):
            return 1.0, "The selected option text is correct."
        return 0.0, (
            f'Incorrect. Expected option "{gold_label}" ({gold_text}), got "{response_str}". '
            "Improve option-sensitive reasoning and produce the final answer in the expected MCQ format."
        )


def make_guide() -> Any:
    """Return an instantiated MCQ guide."""
    return MCQGuide()
