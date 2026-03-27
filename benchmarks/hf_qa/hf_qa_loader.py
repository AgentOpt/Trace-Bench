"""HuggingFace QA loader for the `hf:` task suite.

This is the Trace-Bench adapter for HuggingFace QA datasets.  It is a direct
translation of the standalone HotpotQA training script
(https://github.com/xuanfeiren/hotpotqa/blob/main/prompt_opt/trace_opt.py)
into the Trace-Bench ``build_trace_problem`` pattern.

Supported tasks are declared in ``hf_tasks.yaml`` in this directory.
The generic loader handles any entry in that file; adding a new HF QA dataset
requires only a new YAML entry — no Python changes.

Usage in a YAML config::

    tasks:
      - id: hf:hotpot_qa
        eval_kwargs:
          num_train: 10
"""
from __future__ import annotations

import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ---------------------------------------------------------------------------
# Optional dependency guard
# ---------------------------------------------------------------------------
try:
    from datasets import load_dataset as _hf_load_dataset
    _DATASETS_AVAILABLE = True
except ImportError:
    _DATASETS_AVAILABLE = False

try:
    from opto.trace.modules import Module
    from opto.trace import node, bundle
    from opto.utils.llm import LLM
    from opto.trainer.guide import Guide
    _OPTO_AVAILABLE = True
except Exception:
    _OPTO_AVAILABLE = False

# ---------------------------------------------------------------------------
# Task config index
# ---------------------------------------------------------------------------

_YAML_PATH = Path(__file__).parent / "hf_tasks.yaml"
_TASK_INDEX: Optional[Dict[str, Dict[str, Any]]] = None


def _load_task_index() -> Dict[str, Dict[str, Any]]:
    global _TASK_INDEX
    if _TASK_INDEX is None:
        raw = yaml.safe_load(_YAML_PATH.read_text(encoding="utf-8")) or []
        _TASK_INDEX = {entry["id"]: entry for entry in raw}
    return _TASK_INDEX


def _load_task_config(task_id: str) -> Dict[str, Any]:
    index = _load_task_index()
    if task_id not in index:
        known = sorted(index.keys())
        raise KeyError(
            f"Unknown hf task '{task_id}'. Known tasks: {known}. "
            f"To add a new task, append an entry to {_YAML_PATH}."
        )
    return index[task_id]


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------

@dataclass
class HFQATask:
    """A single QA example with pre-formatted context."""
    question: str
    context: str
    answer: str


def _normalize_answer(text: str) -> str:
    """Lowercase, remove articles and punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _check_answer(response: str, expected: str) -> bool:
    """Return True if the normalized response contains the normalized answer."""
    return _normalize_answer(expected) in _normalize_answer(response)


def _format_context(raw: Any, context_format: str) -> str:
    """Flatten a raw HF context field into a plain string.

    Supported formats
    -----------------
    ``hotpot_qa``
        Raw value is ``{"title": [str, ...], "sentences": [[str, ...], ...]}``.
        Each document becomes ``<title>\\n<sentences joined>``.
    ``plain``
        Raw value is already a plain string.
    ``squad``
        Raw value is a plain string (SQuAD stores context directly).
    """
    if context_format == "hotpot_qa":
        titles = raw.get("title", [])
        sentences = raw.get("sentences", [])
        parts = []
        for title, sents in zip(titles, sentences):
            parts.append(f"{title}\n{''.join(sents)}")
        return "\n\n".join(parts)
    if context_format in ("plain", "squad"):
        return str(raw)
    return str(raw)


def _load_hf_data(cfg: Dict[str, Any], n: int) -> List[HFQATask]:
    """Load up to *n* examples from a HuggingFace dataset."""
    if not _DATASETS_AVAILABLE:
        raise ImportError(
            "The `datasets` package is required for hf: tasks. "
            "Install it with: pip install trace-bench[hf]"
        )
    ds = _hf_load_dataset(
        cfg["dataset_id"],
        cfg.get("dataset_config"),
        split=cfg.get("split", "validation"),
    )
    tasks: List[HFQATask] = []
    q_field = cfg.get("question_field", "question")
    a_field = cfg.get("answer_field", "answer")
    ctx_fmt = cfg.get("context_format", "plain")

    for row in ds:
        if len(tasks) >= n:
            break
        question = str(row[q_field])
        raw_answer = row[a_field]
        # Some datasets wrap the answer (e.g. SQuAD: {"text": [...], ...})
        if isinstance(raw_answer, dict):
            texts = raw_answer.get("text", [])
            answer = texts[0] if texts else ""
        else:
            answer = str(raw_answer)
        raw_context = row.get("context", row.get("passage", ""))
        context = _format_context(raw_context, ctx_fmt)
        tasks.append(HFQATask(question=question, context=context, answer=answer))

    return tasks


# ---------------------------------------------------------------------------
# Agent  (translated from HotpotQAAgent in trace_opt.py)
# ---------------------------------------------------------------------------

if _OPTO_AVAILABLE:
    class HFQAAgent(Module):
        """Prompt-optimization agent for HF QA tasks.

        Holds a single trainable ``meta_instructions`` node.  The LLM is called
        inside a ``@bundle`` so Trace can attribute gradients to the instruction
        parameter (same structure as the reference HotpotQAAgent).
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
        def format_and_call(self, instructions: Any, task: HFQATask) -> str:
            """Format the prompt and call the LLM.

            Replaces ``hotpotqa_eval.evaluate_single`` from the reference script:
            assembles the full prompt, calls ``self.llm``, and returns the raw
            answer string.
            """
            prompt = (
                f"{instructions}\n\n"
                f"Context:\n{task.context}\n\n"
                f"Question: {task.question}\n\n"
                "Answer:"
            )
            return self.llm([{"role": "user", "content": prompt}])

        def forward(self, task: HFQATask) -> Any:
            return self.format_and_call(self.instructions, task)

else:
    # Stub so the module can be imported even without opto (e.g. for discovery)
    class HFQAAgent:  # type: ignore[no-redef]
        pass


# ---------------------------------------------------------------------------
# Guide  (translated from HotpotQAGuide in trace_opt.py)
# ---------------------------------------------------------------------------

if _OPTO_AVAILABLE:
    class HFQAGuide(Guide):
        """Exact-match guide for HF QA tasks.

        Normalizes both the model response and the gold answer before comparing,
        so minor capitalization / article differences don't penalize correct answers.
        """

        def get_feedback(
            self,
            task: HFQATask,
            response: Any,
            info: HFQATask,
            **kwargs: Any,
        ):
            response_str = str(getattr(response, "data", response))
            is_correct = _check_answer(response_str, info.answer)
            if is_correct:
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_DEFAULT_INSTRUCTIONS = "Answer the question based on the context."


def build_trace_problem(
    task_id: str = "hotpot_qa",
    num_train: int = 10,
    initial_instructions: str = _DEFAULT_INSTRUCTIONS,
    **_kwargs: Any,
) -> Dict[str, Any]:
    """Build a Trace-Bench task bundle for an HF QA task.

    Parameters
    ----------
    task_id:
        Key from ``hf_tasks.yaml`` (e.g. ``"hotpot_qa"``).  This is injected
        automatically by the registry when the task id is ``hf:<task_id>``.
    num_train:
        Number of training examples to load from the dataset.
    initial_instructions:
        Starting value for the trainable ``meta_instructions`` parameter.

    Returns
    -------
    Trace-Bench bundle dict with keys:
        ``param``, ``guide``, ``train_dataset``, ``optimizer_kwargs``, ``metadata``.
    """
    cfg = _load_task_config(task_id)
    tasks = _load_hf_data(cfg, num_train)

    agent = HFQAAgent(
        initial_instructions=initial_instructions,
        node_description=cfg.get(
            "description",
            "Meta-instructions guiding the agent's reasoning and answer format.",
        ),
    )
    guide = HFQAGuide()
    objective = cfg.get("objective", "Optimize the agent's instructions for this QA task.")

    return dict(
        param=agent,
        guide=guide,
        train_dataset={"inputs": tasks, "infos": tasks},
        optimizer_kwargs=dict(objective=objective, memory_size=10),
        metadata=dict(
            benchmark="hf",
            task_id=task_id,
            dataset_id=cfg["dataset_id"],
            dataset_config=cfg.get("dataset_config"),
            num_train=num_train,
        ),
    )
