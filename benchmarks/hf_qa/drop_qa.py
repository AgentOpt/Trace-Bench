from __future__ import annotations

from typing import Any, Dict, List, Tuple

try:
    from .common import HFQATask, best_alias_score, normalize_aliases, response_text
except ImportError:
    from common import HFQATask, best_alias_score, normalize_aliases, response_text

try:
    from opto.trainer.guide import Guide
except Exception:
    Guide = object  # type: ignore[assignment,misc]


def _flatten_answer_values(raw: Any) -> List[str]:
    """Extract non-empty answer strings from common DROP answer shapes."""
    values: List[str] = []
    if isinstance(raw, dict):
        for key in ("spans", "number", "numbers", "date", "dates"):
            value = raw.get(key)
            if isinstance(value, list):
                values.extend(str(item) for item in value if str(item).strip())
            elif value is not None and str(value).strip():
                values.append(str(value))
    elif isinstance(raw, list):
        values.extend(str(item) for item in raw if str(item).strip())
    elif raw is not None and str(raw).strip():
        values.append(str(raw))
    return values


def _aliases_from_drop_answer(raw: Any) -> List[str]:
    """Return normalized DROP answer aliases."""
    return normalize_aliases(_flatten_answer_values(raw))


def row_to_tasks(row: Dict[str, Any], cfg: Dict[str, Any]) -> List[HFQATask]:
    """Convert one DROP row into a passage QA task."""
    question_field = cfg.get("question_field", "question")
    answer_field = cfg.get("answer_field", "answers_spans")
    context_field = cfg.get("context_field", "passage")
    if question_field not in row:
        raise KeyError(f"DROP row missing question field {question_field!r}")

    aliases = _aliases_from_drop_answer(row.get(answer_field))
    return [
        HFQATask(
            question=str(row[question_field]),
            context=str(row.get(context_field, "")),
            answer=aliases[0] if aliases else "",
            meta={"aliases": aliases},
        )
    ]


def format_context(raw: Any) -> str:
    """Return plain-string context for DROP tasks."""
    return str(raw or "")


def make_dataset(tasks: List[HFQATask]) -> Dict[str, List[Any]]:
    """Return the default context-rich QA dataset shape."""
    return {"inputs": list(tasks), "infos": list(tasks)}


class DROPGuide(Guide):  # type: ignore[misc]
    """Guide for alias-tolerant DROP answer scoring."""

    def get_feedback(
        self, task: Any, response: Any, info: HFQATask, **kwargs: Any
    ) -> Tuple[float, str]:
        """Score a DROP response against the gold aliases."""
        response_str = response_text(response)
        aliases = info.meta.get("aliases", [info.answer])
        score = best_alias_score(response_str, aliases)
        if score >= 0.999:
            return 1.0, "Correct."
        return score, (
            f"Incorrect. Expected one of {aliases}, got '{response_str}'. "
            "Improve passage-based arithmetic/discrete reasoning and answer normalization."
        )


def make_guide() -> Any:
    """Return an instantiated DROP guide."""
    return DROPGuide()
