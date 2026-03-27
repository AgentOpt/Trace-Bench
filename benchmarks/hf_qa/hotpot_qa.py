"""HotpotQA agent and guide for the `hf:hotpot_qa` task.

Translated from:
  https://github.com/xuanfeiren/hotpotqa/blob/main/prompt_opt/trace_opt.py
"""
from __future__ import annotations

import re
import string
from typing import Any

try:
    from opto.trace.modules import Module
    from opto.trace import node, bundle
    from opto.utils.llm import LLM
    from opto.trainer.guide import Guide
    _OPTO_AVAILABLE = True
except Exception:
    _OPTO_AVAILABLE = False


# ---------------------------------------------------------------------------
# Answer checking
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
    """Flatten HotpotQA's nested context into a plain string.

    Raw value is ``{"title": [str, ...], "sentences": [[str, ...], ...]}``.
    Each document becomes ``<title>\\n<sentences joined>``.
    """
    titles = raw.get("title", [])
    sentences = raw.get("sentences", [])
    parts = [f"{t}\n{''.join(s)}" for t, s in zip(titles, sentences)]
    return "\n\n".join(parts)


def check_answer(response: str, expected: str) -> bool:
    """Return True if the normalized response contains the normalized answer."""
    return _normalize_answer(expected) in _normalize_answer(response)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

if _OPTO_AVAILABLE:
    class HFQAAgent(Module):
        """Prompt-optimization agent for context-rich QA (HotpotQA, SQuAD, …).

        Holds a single trainable ``meta_instructions`` node. The LLM is called
        inside a non-trainable ``@bundle`` so Trace can attribute gradients to
        the instruction parameter.
        """

        def __init__(
            self,
            initial_instructions: str = "Answer the question based on the context.",
            node_description: str = "Meta-instructions guiding the agent's reasoning and answer format.",
        ) -> None:
            super().__init__()
            self.instructions = node(
                initial_instructions,
                trainable=True,
                name="meta_instructions",
                description=node_description,
            )
            self.llm = LLM()

        @bundle(trainable=False)
        def format_and_call(self, instructions: Any, task: Any) -> str:
            """Assemble the prompt and call the LLM."""
            prompt = (
                f"{instructions}\n\n"
                f"Context:\n{task.context}\n\n"
                f"Question: {task.question}\n\n"
                "Answer:"
            )
            return self.llm([{"role": "user", "content": prompt}])

        def forward(self, task: Any) -> Any:
            return self.format_and_call(self.instructions, task)

else:
    class HFQAAgent:  # type: ignore[no-redef]
        pass


# ---------------------------------------------------------------------------
# Guide
# ---------------------------------------------------------------------------

if _OPTO_AVAILABLE:
    class HFQAGuide(Guide):
        """Exact-match guide for context-rich QA tasks."""

        def get_feedback(self, task: Any, response: Any, info: Any, **kwargs: Any):
            response_str = str(getattr(response, "data", response))
            if check_answer(response_str, info.answer):
                return 1.0, "Correct."
            return 0.0, (
                f"Incorrect. Expected: '{info.answer}', Got: '{response_str}'.\n"
                f"Context: {info.context}\n"
                f"Question: {info.question}\n"
                "Identify the failure mode and update meta_instructions with "
                "improved reasoning and formatting guidance."
            )

else:
    class HFQAGuide:  # type: ignore[no-redef]
        pass
