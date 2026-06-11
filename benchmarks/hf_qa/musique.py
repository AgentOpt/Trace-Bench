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


def format_context(raw: Any) -> str:
    """Format MuSiQue paragraph rows into numbered context blocks."""
    parts: List[str] = []
    for index, paragraph in enumerate(raw or []):
        title = paragraph.get("title", f"Paragraph {index + 1}")
        text = paragraph.get("paragraph_text", paragraph.get("paragraph", ""))
        parts.append(f"Paragraph {index + 1} ({title}):\n{text}")
    return "\n\n".join(parts)


def row_to_tasks(row: Dict[str, Any], cfg: Dict[str, Any]) -> List[HFQATask]:
    """Convert one MuSiQue row into a single multi-hop QA task."""
    question_field = cfg.get("question_field", "question")
    context_field = cfg.get("context_field", "paragraphs")
    if question_field not in row:
        raise KeyError(f"MuSiQue row missing question field {question_field!r}")

    aliases = normalize_aliases(
        [str(row.get("answer", ""))]
        + [str(alias) for alias in (row.get("answer_aliases", []) or [])]
    )
    return [
        HFQATask(
            question=str(row[question_field]),
            context=format_context(row.get(context_field, [])),
            answer=aliases[0] if aliases else "",
            meta={"aliases": aliases},
        )
    ]


def make_dataset(tasks: List[HFQATask]) -> Dict[str, List[Any]]:
    """Return the default context-rich QA dataset shape."""
    return {"inputs": list(tasks), "infos": list(tasks)}


class MuSiQueGuide(Guide):  # type: ignore[misc]
    """Guide for alias-tolerant MuSiQue answer scoring."""

    def get_feedback(
        self, task: Any, response: Any, info: HFQATask, **kwargs: Any
    ) -> Tuple[float, str]:
        """Score a MuSiQue response against the gold aliases."""
        response_str = response_text(response)
        aliases = info.meta.get("aliases", [info.answer])
        score = best_alias_score(response_str, aliases)
        if score >= 0.999:
            return 1.0, "Correct."
        return score, (
            f"Incorrect. Gold aliases: {aliases}. Got: '{response_str}'. "
            "Improve compositional multi-hop reasoning across the provided paragraphs."
        )


def make_guide() -> Any:
    """Return an instantiated MuSiQue guide."""
    return MuSiQueGuide()
