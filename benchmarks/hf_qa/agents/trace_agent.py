"""OpenTrace (opto) agents for the hf: task suite.

Agents
------
TraceHotpotQAAgent
    HotpotQA agent.  Holds a single trainable ``meta_instructions`` node.
    Expects inputs to be ``HFQATask`` objects with ``.question`` and
    ``.context`` attributes.

TraceBBEHAgent
    Question-only agent for BigBenchExtraHard tasks.  Holds a trainable
    ``prompt_template`` parameter node and a trainable ``extract_answer`` bundle.
    Expects plain question strings as inputs.

Factory
-------
make_agent(agent_class, **kwargs) -> agent instance
"""
from __future__ import annotations

from textwrap import dedent
from typing import Any

try:
    from opto.trace.modules import Module
    from opto.trace import node, bundle
    from opto.trace.nodes import ParameterNode
    import opto.trace.operators as trace_ops
    from opto.utils.llm import LLM
    _OPTO_AVAILABLE = True
except Exception:
    _OPTO_AVAILABLE = False


# ---------------------------------------------------------------------------
# TraceHotpotQAAgent
# ---------------------------------------------------------------------------

if _OPTO_AVAILABLE:
    class TraceHotpotQAAgent(Module):
        """HotpotQA agent using OpenTrace.

        Holds a single trainable ``meta_instructions`` node.  The LLM is called
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
    class TraceHotpotQAAgent:  # type: ignore[no-redef]
        pass


# ---------------------------------------------------------------------------
# TraceBBEHAgent — question-only agent (was BBEHAgent in bbeh.py)
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
    class TraceBBEHAgent(Module):
        """Agent for BigBenchExtraHard tasks.

        - ``prompt_template`` is a trainable ``ParameterNode``; the optimiser
          can rewrite the entire template.
        - ``extract_answer`` is a trainable bundle; the optimiser can rewrite
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
            response = trace_ops.call_llm(self.llm, self.prompt_template, question)
            return self.extract_answer(self.prompt_template, question, response)

else:
    class TraceBBEHAgent:  # type: ignore[no-redef]
        pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_agent(agent_class: str, **kwargs: Any) -> Any:
    """Return an instantiated Trace agent for the given task class.

    Parameters
    ----------
    agent_class:
        Task-type key from ``hf_tasks.yaml`` (e.g. ``"hfqa"``, ``"bbeh"``).
    **kwargs:
        Forwarded to ``TraceHotpotQAAgent.__init__`` for non-BBEH tasks.
        Accepted keys: ``initial_instructions``, ``node_description``.
    """
    if not _OPTO_AVAILABLE:
        raise ImportError(
            "TraceHotpotQAAgent / TraceBBEHAgent require the `opto` package. "
            "Install OpenTrace: pip install trace-opt"
        )
    if agent_class == "bbeh":
        return TraceBBEHAgent()
    return TraceHotpotQAAgent(**kwargs)
