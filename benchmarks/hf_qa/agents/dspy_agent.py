"""DSPy agents for the hf: task suite.

Agents
------
DSPyHotpotQAAgent
    HotpotQA agent.  Uses ``dspy.Predict`` with a (context, question → answer)
    signature.  Expects ``HFQATask`` objects with ``.question`` and ``.context``
    attributes.  The signature instruction is set from ``initial_instructions``
    at init time and is the parameter that DSPy optimisers will update.

DSPyBBEHAgent
    Question-only agent for BigBenchExtraHard tasks.  Uses
    ``dspy.Predict("question -> answer")``.  Expects plain question strings.

Both agents return plain strings so the task guides (``HotpotQAGuide``,
``BBEHGuide``) can score them without any framework-specific unwrapping.

Factory
-------
make_agent(agent_class, **kwargs) -> agent instance
"""
from __future__ import annotations

from typing import Any

try:
    import dspy
    _DSPY_AVAILABLE = True
except ImportError:
    _DSPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# DSPyHotpotQAAgent
# ---------------------------------------------------------------------------

if _DSPY_AVAILABLE:
    class HotpotQAAnswer(dspy.Signature):
        """Answer the question based on the context."""
        context: str = dspy.InputField(desc="Context paragraphs for the question")
        question: str = dspy.InputField(desc="The question to answer")
        answer: str = dspy.OutputField(desc="A concise, factual answer")

    class DSPyHotpotQAAgent(dspy.Module):
        """DSPy Module that uses a Predict to answer HotpotQA questions."""
        def __init__(self):
            super().__init__()
            self.predict = dspy.Predict(HotpotQAAnswer)

        def forward(self, task: Any) -> str:
            result = self.predict(context=task.context, question=task.question)
            return result.answer

else:
    class DSPyHotpotQAAgent:  # type: ignore[no-redef]
        pass


# ---------------------------------------------------------------------------
# DSPyBBEHAgent
# ---------------------------------------------------------------------------

if _DSPY_AVAILABLE:
    class DSPyBBEHAgent(dspy.Module):
        """DSPy agent for BigBenchExtraHard tasks.

        Uses a minimal ``question -> answer`` signature; the instruction is
        the trainable component that DSPy optimisers can update.

        Compatible with ``BBEHGuide``: returns a plain string so the guide
        can call ``eval_metric`` without any DSPy-specific unwrapping.
        """

        def __init__(self) -> None:
            super().__init__()
            self.prog = dspy.Predict("question -> answer")

        def forward(self, question: str) -> str:
            result = self.prog(question=question)
            return result.answer

else:
    class DSPyBBEHAgent:  # type: ignore[no-redef]
        pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_agent(agent_class: str, **kwargs: Any) -> Any:
    """Return an instantiated DSPy agent for the given task class.

    Parameters
    ----------
    agent_class:
        Task-type key from ``hf_tasks.yaml`` (e.g. ``"hfqa"``, ``"bbeh"``).
    **kwargs:
        Forwarded to ``DSPyHotpotQAAgent.__init__`` for non-BBEH tasks.
        Accepted keys: ``initial_instructions``, ``node_description``.
    """
    if not _DSPY_AVAILABLE:
        raise ImportError(
            "DSPy agents require the `dspy` package. "
            "Install it with: pip install dspy-ai"
        )
    if agent_class == "bbeh":
        return DSPyBBEHAgent()
    return DSPyHotpotQAAgent(**kwargs)
