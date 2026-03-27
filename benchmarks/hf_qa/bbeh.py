"""BBEH agent and guide for the `hf:bbeh/*` tasks.

Translated from the reference ``Predict`` / ``BigBenchGuide`` classes in
``bbeh_trace.py``.
"""
from __future__ import annotations

import re
from textwrap import dedent
from typing import Any

try:
    from opto.trace.modules import Module
    from opto.trace import bundle
    from opto.trace.nodes import ParameterNode
    import opto.trace.operators as trace_ops
    from opto.utils.llm import LLM
    from opto.trainer.guide import Guide
    _OPTO_AVAILABLE = True
except Exception:
    _OPTO_AVAILABLE = False


# ---------------------------------------------------------------------------
# Answer checking
# ---------------------------------------------------------------------------

def format_context(raw: Any) -> str:
    """No-op — BBEH tasks have no context field."""
    return ""


def make_dataset(tasks: list) -> dict:
    """Convert a list of HFQATask objects into a BBEH train/val/test dataset dict.

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
# Agent
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = dedent("""\
    Given the fields `question`, produce the fields `answer`.

    ---

    Follow the following format.

    Question:
    Answer:

    ---
    Question:
    """)

_PROMPT_DESCRIPTION = (
    "[ParameterNode] This is the Prompt Template to the LLM. "
    "Need to include information about what the format of answers LLM should output. "
    "They can be (A)/(B), a number like 8, or a string, or Yes/No/Ambiguous."
)

if _OPTO_AVAILABLE:
    class BBEHAgent(Module):
        """Agent for BigBenchExtraHard tasks.

        Translated from the reference ``Predict`` class:

        - ``prompt_template`` is a trainable ``ParameterNode``; the optimizer
          can rewrite the entire template.
        - ``extract_answer`` is a trainable bundle; the optimizer can rewrite
          the answer-parsing code.
        - The LLM is called via ``trace_ops.call_llm`` so the call is tracked
          as a Trace node.
        """

        def __init__(self) -> None:
            super().__init__()
            self.llm = LLM()
            self.prompt_template = ParameterNode(
                _PROMPT_TEMPLATE,
                trainable=True,
                description=_PROMPT_DESCRIPTION,
            )

        @bundle(trainable=True, allow_external_dependencies=True)
        def extract_answer(self, prompt_template: Any, question: str, response: Any) -> str:
            """Parse the LLM response and return the final answer.

            Need to read in the response, which can contain additional thought,
            deliberation, and an answer. Process the response and find where
            the answer is. Can call the LLM again to refine if necessary.

            Args:
                prompt_template: The prompt used to query the LLM.
                question: The question text including any "Options" section.
                response: Raw LLM response string. Extract the answer in the
                          exact format the evaluator expects: (A)/(B), a number,
                          a string, or Yes/No.
            """
            answer = str(response).split("Answer:")[-1].strip()
            return answer

        def forward(self, question: str) -> Any:
            # prompt_template is the system prompt (trainable); question is the user turn
            response = trace_ops.call_llm(self.llm, self.prompt_template, question)
            return self.extract_answer(self.prompt_template, question, response)

else:
    class BBEHAgent:  # type: ignore[no-redef]
        pass


# ---------------------------------------------------------------------------
# Guide
# ---------------------------------------------------------------------------

if _OPTO_AVAILABLE:
    class BBEHGuide(Guide):
        """Guide for BigBenchExtraHard tasks.

        Translated from ``BigBenchGuide`` in ``bbeh_trace.py``.
        Uses ``eval_metric`` which handles both multiple-choice and free-form answers.
        """

        def get_feedback(self, task: str, response: Any, info: str, **kwargs: Any):
            response_str = str(getattr(response, "data", response))
            if eval_metric(info, response_str):
                return 1.0, "The answer is correct! No need to change anything."
            return 0.0, (
                f'The answer is wrong. We expect the output of your answer to be "{info}". '
                "Please modify the prompt and relevant parts of the program to help LLM "
                "produce the right answer."
            )

else:
    class BBEHGuide:  # type: ignore[no-redef]
        pass
