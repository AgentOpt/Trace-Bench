from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

try:
    from .common import HFQATask, best_alias_score, normalize_aliases, response_text
except ImportError:
    from common import HFQATask, best_alias_score, normalize_aliases, response_text

try:
    from opto.trainer.guide import Guide
except Exception:
    Guide = object  # type: ignore[assignment,misc]


def format_context(raw: Any) -> str:
    """Format Qasper full_text sections into a long-document context."""
    if not isinstance(raw, dict):
        return str(raw or "")

    section_names = raw.get("section_name", [])
    paragraphs = raw.get("paragraphs", [])
    parts: List[str] = []
    for index, section_paragraphs in enumerate(paragraphs):
        title = (
            section_names[index]
            if index < len(section_names)
            else f"Section {index + 1}"
        )
        joined = "\n".join(
            str(paragraph) for paragraph in section_paragraphs if str(paragraph).strip()
        )
        if joined:
            parts.append(f"{title}\n{joined}")
    return "\n\n".join(parts)


def _iter_qasper_answers(raw: Any) -> Iterable[Dict[str, Any]]:
    """Yield normalized answer dicts from Qasper's nested answer shape."""
    if isinstance(raw, list):
        for item in raw:
            yield from _iter_qasper_answers(item)
    elif isinstance(raw, dict):
        if "answer" in raw:
            yield from _iter_qasper_answers(raw["answer"])
        else:
            yield raw


def _aliases_and_evidence(raw_answers: Any) -> Tuple[List[str], List[str]]:
    """Extract answer aliases and evidence snippets from Qasper answers."""
    aliases: List[str] = []
    evidence: List[str] = []
    for answer in _iter_qasper_answers(raw_answers):
        if answer.get("unanswerable"):
            aliases.append("unanswerable")

        spans = answer.get("extractive_spans", []) or []
        aliases.extend(str(span) for span in spans if str(span).strip())

        free_form = str(answer.get("free_form_answer", "")).strip()
        if free_form:
            aliases.append(free_form)

        yes_no = answer.get("yes_no")
        if isinstance(yes_no, bool):
            aliases.append("yes" if yes_no else "no")

        evidence.extend(
            str(item)
            for item in (answer.get("evidence", []) or [])
            if str(item).strip()
        )

    return normalize_aliases(aliases), evidence


def _converted_qasper_task(row: Dict[str, Any], cfg: Dict[str, Any]) -> HFQATask:
    """Convert one row from script-free converted_qasper mirrors."""
    question_field = cfg.get("question_field", "input")
    answer_field = cfg.get("answer_field", "output")
    if question_field not in row:
        raise KeyError(f"Qasper row missing question field {question_field!r}")
    if answer_field not in row:
        raise KeyError(f"Qasper row missing answer field {answer_field!r}")

    raw_input = str(row[question_field])
    question = raw_input
    context = ""
    if raw_input.startswith("Q:") and "\nText:" in raw_input:
        question_part, context_part = raw_input.split("\nText:", 1)
        question = question_part.removeprefix("Q:").strip()
        context = context_part.strip()

    aliases = normalize_aliases([str(row[answer_field])])
    return HFQATask(
        question=question,
        context=context,
        answer=aliases[0] if aliases else "",
        meta={
            "aliases": aliases,
            "question_id": str(row.get("pid", row.get("id", ""))),
            "evidence": [],
        },
    )


def row_to_tasks(row: Dict[str, Any], cfg: Dict[str, Any]) -> List[HFQATask]:
    """Expand one Qasper paper row into one task per question."""
    if "qas" not in row:
        return [_converted_qasper_task(row, cfg)]

    context = format_context(row.get(cfg.get("context_field", "full_text"), {}))
    qas = row.get("qas", {})
    if not isinstance(qas, dict):
        raise TypeError(
            f"Qasper row field 'qas' must be a dict, got {type(qas).__name__}"
        )

    questions = qas.get("question", [])
    question_ids = qas.get("question_id", [""] * len(questions))
    answers = qas.get("answers", [])

    tasks: List[HFQATask] = []
    for index, question in enumerate(questions):
        raw_answers = answers[index] if index < len(answers) else []
        aliases, evidence = _aliases_and_evidence(raw_answers)
        tasks.append(
            HFQATask(
                question=str(question),
                context=context,
                answer=aliases[0] if aliases else "",
                meta={
                    "aliases": aliases,
                    "question_id": (
                        question_ids[index] if index < len(question_ids) else ""
                    ),
                    "evidence": evidence,
                },
            )
        )
    return tasks


def make_dataset(tasks: List[HFQATask]) -> Dict[str, List[Any]]:
    """Return the default context-rich QA dataset shape."""
    return {"inputs": list(tasks), "infos": list(tasks)}


class QasperGuide(Guide):  # type: ignore[misc]
    """Guide for alias-tolerant Qasper answer scoring."""

    def get_feedback(
        self, task: Any, response: Any, info: HFQATask, **kwargs: Any
    ) -> Tuple[float, str]:
        """Score a Qasper response against gold aliases and evidence."""
        response_str = response_text(response)
        aliases = info.meta.get("aliases", [info.answer])
        score = best_alias_score(response_str, aliases)
        if score >= 0.999:
            return 1.0, "Correct."
        evidence = info.meta.get("evidence", [])[:2]
        evidence_hint = f" Evidence hints: {evidence}." if evidence else ""
        return score, (
            f"Incorrect. Expected one of {aliases}, got '{response_str}'."
            f"{evidence_hint} Improve long-document scholarly QA and answer grounding."
        )


def make_guide() -> Any:
    """Return an instantiated Qasper guide."""
    return QasperGuide()
