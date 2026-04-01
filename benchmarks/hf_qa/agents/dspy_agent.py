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

Each agent class also provides a ``to_dspy_examples(inputs, infos)`` classmethod
that converts the Trace-Bench dataset format into DSPy ``Example`` objects.
The DSPy trainer calls this method generically — it does not need to know
anything about the benchmark's data format.

Factory
-------
make_agent(agent_class, **kwargs) -> agent instance
"""
from __future__ import annotations

from typing import Any, List

try:
    import dspy
    _DSPY_AVAILABLE = True
except ImportError:
    _DSPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# DSPyHotpotQAAgent
# ---------------------------------------------------------------------------

if _DSPY_AVAILABLE:
    class _HotpotQASignature(dspy.Signature):
        """Answer the question based on the context."""
        context: str = dspy.InputField(desc="Context paragraphs for the question")
        question: str = dspy.InputField(desc="The question to answer")
        answer: str = dspy.OutputField(desc="A concise, factual answer")

    class DSPyHotpotQAAgent(dspy.Module):
        """DSPy agent for HotpotQA.

        The trainable component is the instruction on the ``dspy.Predict``
        signature, which DSPy optimisers (COPRO, GEPA, …) can rewrite.

        Compatible with ``HotpotQAGuide``: returns a plain string so the
        guide can call ``check_answer`` without any DSPy-specific unwrapping.
        """

        def __init__(
            self,
            initial_instructions: str = "Answer the question based on the context.",
            node_description: str = "",
        ) -> None:
            super().__init__()
            self.predict = dspy.Predict(
                _HotpotQASignature.with_instructions(initial_instructions)
            )

        def forward(self, task: Any) -> str:
            result = self.predict(context=task.context, question=task.question)
            return result.answer

        @classmethod
        def to_examples(cls, inputs: List[Any], infos: List[Any]) -> List[Any]:
            """Convert HotpotQA dataset lists to DSPy Examples.

            Inputs are ``HFQATask`` objects; infos are also ``HFQATask`` objects
            (same object since ``make_dataset`` passes tasks through as-is).

            ``_task`` and ``_info`` store the original objects so the trainer's
            metric can call ``guide.get_feedback(task, response, info)`` without
            any task-format-specific logic in the trainer.
            """
            return [
                dspy.Example(
                    context=x.context,
                    question=x.question,
                    answer=info.answer if hasattr(info, "answer") else str(info),
                    _task=x,
                    _info=info,
                ).with_inputs("context", "question")
                for x, info in zip(inputs, infos)
            ]

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

        @classmethod
        def to_examples(cls, inputs: List[Any], infos: List[Any]) -> List[Any]:
            """Convert BBEH dataset lists to DSPy Examples.

            Inputs are plain question strings; infos are plain answer strings.

            ``_task`` and ``_info`` store the original objects so the trainer's
            metric can call ``guide.get_feedback(task, response, info)`` without
            any task-format-specific logic in the trainer.
            """
            return [
                dspy.Example(question=x, answer=str(info), _task=x, _info=info).with_inputs("question")
                for x, info in zip(inputs, infos)
            ]

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
