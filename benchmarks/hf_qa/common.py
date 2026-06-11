from __future__ import annotations

from dataclasses import dataclass, field
import re
import string
from typing import Any, Dict, List, Sequence, Tuple


@dataclass
class HFQATask:
    """A single QA example with formatted context and optional task metadata."""

    question: str
    context: str
    answer: str
    meta: Dict[str, Any] = field(default_factory=dict)


def normalize_text(text: str) -> str:
    """Normalize answer text for exact-match and F1-style comparisons."""
    normalized = str(text).lower()
    normalized = re.sub(r"\b(a|an|the)\b", " ", normalized)
    normalized = normalized.translate(str.maketrans("", "", string.punctuation))
    return " ".join(normalized.split())


def token_f1(prediction: str, gold: str) -> float:
    """Return token-level F1 between normalized prediction and gold text."""
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    counts: Dict[str, int] = {}
    for token in pred_tokens:
        counts[token] = counts.get(token, 0) + 1

    overlap = 0
    for token in gold_tokens:
        if counts.get(token, 0) > 0:
            overlap += 1
            counts[token] -= 1
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def extract_last_number(text: str) -> str:
    """Extract the final integer or decimal number from a model response."""
    matches = re.findall(r"[-+]?\d[\d,]*\.?\d*", str(text))
    return matches[-1].replace(",", "") if matches else ""


def extract_last_choice(text: str) -> str:
    """Extract the final parenthesized or standalone choice label."""
    parenthesized = re.findall(r"\(([A-Za-z0-9]+)\)", str(text))
    if parenthesized:
        return parenthesized[-1]
    standalone = re.findall(r"\b([A-Z])\b", str(text))
    return standalone[-1] if standalone else ""


def response_text(response: Any) -> str:
    """Extract assistant text from common raw LLM response containers."""
    value = getattr(response, "data", response)

    choices = (
        value.get("choices")
        if isinstance(value, dict)
        else getattr(value, "choices", None)
    )
    if choices:
        first_choice = choices[0]
        message = (
            first_choice.get("message")
            if isinstance(first_choice, dict)
            else getattr(first_choice, "message", None)
        )
        if message is not None:
            content = (
                message.get("content")
                if isinstance(message, dict)
                else getattr(message, "content", None)
            )
            if content is not None:
                return str(content)
        text = (
            first_choice.get("text")
            if isinstance(first_choice, dict)
            else getattr(first_choice, "text", None)
        )
        if text is not None:
            return str(text)

    content = (
        value.get("content")
        if isinstance(value, dict)
        else getattr(value, "content", None)
    )
    if content is not None:
        return str(content)
    return str(value)


def choice_pairs(raw: Any) -> List[Tuple[str, str]]:
    """Return normalized (label, text) pairs from common HF MCQ schemas."""
    if isinstance(raw, dict):
        raw_texts = raw.get("text") or raw.get("texts") or raw.get("options") or []
        texts = [str(text) for text in raw_texts]
        raw_labels = raw.get("label") or raw.get("labels") or []
        labels = [str(label) for label in raw_labels]
        if not labels:
            labels = [chr(65 + index) for index in range(len(texts))]
        return list(zip(labels, texts))

    if isinstance(raw, list):
        if raw and isinstance(raw[0], dict):
            pairs: List[Tuple[str, str]] = []
            for index, item in enumerate(raw):
                label = str(item.get("label", chr(65 + index)))
                text = str(item.get("text", item.get("choice", "")))
                pairs.append((label, text))
            return pairs
        return [(chr(65 + index), str(choice)) for index, choice in enumerate(raw)]

    return []


def answer_to_choice_label(
    raw_answer: Any,
    pairs: Sequence[Tuple[str, str]],
    index_base: int = 0,
) -> str:
    """Map a raw HF answer value to the option label used by the prompt."""
    if not isinstance(index_base, int) or isinstance(index_base, bool):
        raise TypeError(f"index_base must be an integer, got {index_base!r}")

    if isinstance(raw_answer, int) and not isinstance(raw_answer, bool):
        index = raw_answer - index_base
        if 0 <= index < len(pairs):
            return pairs[index][0]
        return str(raw_answer)

    answer = str(raw_answer).strip()
    for label, text in pairs:
        if answer == label or normalize_text(answer) == normalize_text(text):
            return label
    return answer


def format_mcq_prompt(
    question: str, pairs: Sequence[Tuple[str, str]], context: str = ""
) -> str:
    """Format a multiple-choice prompt with optional context."""
    prefix = f"Context:\n{context}\n\n" if context else ""
    options = "\n".join(f"({label}) {text}" for label, text in pairs)
    return f"{prefix}Question: {question}\nOptions:\n{options}\nAnswer:"


def split_aliases(raw: str) -> List[str]:
    """Split a delimiter-separated alias string into non-empty aliases."""
    return [item.strip() for item in str(raw).split("||") if item.strip()]


def normalize_aliases(aliases: Sequence[str]) -> List[str]:
    """Deduplicate aliases by normalized text while preserving display text."""
    output: List[str] = []
    seen = set()
    for alias in aliases:
        key = normalize_text(alias)
        if key and key not in seen:
            seen.add(key)
            output.append(str(alias).strip())
    return output


def best_alias_score(prediction: str, aliases: Sequence[str]) -> float:
    """Score a prediction against aliases using exact, contains, then token F1."""
    if not aliases:
        return 0.0
    normalized_prediction = normalize_text(prediction)
    if any(normalized_prediction == normalize_text(alias) for alias in aliases):
        return 1.0
    if any(normalize_text(alias) in normalized_prediction for alias in aliases):
        return 1.0
    return max(token_f1(prediction, alias) for alias in aliases)
