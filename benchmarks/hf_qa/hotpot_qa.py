"""HotpotQA task module for the `hf:hotpot_qa` task.

Provides task-specific data utilities and the guide.  Framework-specific agents
live in ``agents/trace_agent.py`` and ``agents/dspy_agent.py``.

Reference:
  https://github.com/xuanfeiren/hotpotqa/blob/main/prompt_opt/trace_opt.py
"""
from __future__ import annotations

import re
import string
from typing import Any

try:
    from opto.trainer.guide import Guide
    _OPTO_AVAILABLE = True
except Exception:
    _OPTO_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def _normalize_answer(text: str) -> str:
    """Lowercase, remove articles and punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def make_dataset(tasks: list) -> dict:
    """Default dataset format: HFQATask objects passed through as-is."""
    return {"inputs": tasks, "infos": tasks}


def format_context(raw: Any) -> str:
    """Flatten HotpotQA's nested context into numbered paragraphs.

    Raw value is ``{"title": [str, ...], "sentences": [[str, ...], ...]}``.
    Each document becomes ``Paragraph N (title):\\n<sentences joined by space>``.
    """
    titles = raw.get("title", [])
    sentences = raw.get("sentences", [])
    parts = [
        f"Paragraph {j + 1} ({title}):\n{' '.join(sents)}"
        for j, (title, sents) in enumerate(zip(titles, sentences))
    ]
    return "\n\n".join(parts)


def check_answer(response: str, expected: str) -> bool:
    """Return True if the normalized response contains the normalized answer."""
    return _normalize_answer(expected) in _normalize_answer(response)


# ---------------------------------------------------------------------------
# Guide
# ---------------------------------------------------------------------------

if _OPTO_AVAILABLE:
    class HotpotQAGuide(Guide):
        """Evaluation guide for HotpotQA and other context-rich QA tasks.

        Framework-agnostic: works with any agent that produces a string answer.
        """

        def get_feedback(self, task: Any, response: Any, info: Any, **kwargs: Any):
            response_str = str(getattr(response, "data", response))
            if check_answer(response_str, info.answer):
                return 1.0, "Correct."
            return 0.0, (
                f"Incorrect. Expected: '{info.answer}', Got: '{response_str}'.\n"
                f"Question: {info.question}\n"
                f"Context: {info.context}\n"
                "Improve the agent's multi-hop reasoning over multiple context passages."
            )

else:
    class HotpotQAGuide:  # type: ignore[no-redef]
        pass


def make_guide() -> Any:
    """Return an instantiated HotpotQAGuide."""
    return HotpotQAGuide()
