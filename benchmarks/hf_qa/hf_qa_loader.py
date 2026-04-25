"""HuggingFace QA loader for the `hf:` task suite.

Shared data layer: task config index, HFQATask dataclass, data loading, and
``build_trace_problem``.  Task-specific logic (data utilities + guide) lives in
the task module; framework-specific agent logic lives in the agent module:

  hotpot_qa.py        — make_dataset, format_context, check_answer, HotpotQAGuide
  bbeh.py             — make_dataset, format_context, eval_metric, BBEHGuide
  agents/trace_agent.py — TraceQAAgent, TraceBBEHAgent
  agents/dspy_agent.py  — DSPyQAAgent, DSPyBBEHAgent

Each task module must implement:
  - ``make_dataset(tasks: list) -> dict``   — convert HFQATask list to {inputs, infos}
  - ``format_context(raw: Any) -> str``     — flatten the raw HF context field
  - ``make_guide() -> Guide``               — return an instantiated guide

Each agent module must implement:
  - ``make_agent(agent_class: str, **kwargs) -> agent``

Supported tasks are declared in ``hf_tasks.yaml``.  Adding a new task that
fits the existing agent patterns only requires a YAML entry.  Adding a new
agent pattern requires a new task module + ``agent_class`` wiring below.
Adding a new framework requires a new agent module + ``framework`` wiring below.

Usage in a YAML config::

    tasks:
      - id: hf:hotpot_qa           # single task, default Trace framework
      - id: hf:bbeh/boolean_expressions  # single BBEH subtask
      - id: hf:bbeh                # expands to all 23 BBEH subtasks
      - id: hf:bbeh_eight          # expands to the curated 8 subtasks

    tasks:
      - id: hf:hotpot_qa
        eval_kwargs:
          framework: dspy          # use the DSPy agent instead
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

import hotpot_qa as _hotpot_qa_mod
import bbeh as _bbeh_mod

# Registry mapping agent_class (from hf_tasks.yaml) → task module.
# Task modules provide: make_dataset(), format_context(), make_guide().
# To add a new task type: create a module implementing the interface above,
# then add it here and add a YAML entry with the matching agent_class.
_TASK_MODULES: Dict[str, Any] = {
    "hfqa": _hotpot_qa_mod,
    "bbeh": _bbeh_mod,
}

# Registry mapping framework name → agent module (lazily populated).
# Agent modules provide: make_agent(agent_class, **kwargs).
# To add a new framework: create agents/<framework>_agent.py implementing
# the interface above, then add an entry here.
_FRAMEWORK_MODULES: Dict[str, Any] = {}

_SUPPORTED_FRAMEWORKS = ("trace", "dspy")


def _get_framework_module(framework: str) -> Any:
    """Return (and cache) the agent module for *framework*."""
    if framework not in _FRAMEWORK_MODULES:
        if framework == "trace":
            from agents import trace_agent
            _FRAMEWORK_MODULES["trace"] = trace_agent
        elif framework == "dspy":
            from agents import dspy_agent
            _FRAMEWORK_MODULES["dspy"] = dspy_agent
        else:
            raise ValueError(
                f"Unknown framework: {framework!r}. "
                f"Supported: {', '.join(_SUPPORTED_FRAMEWORKS)}"
            )
    return _FRAMEWORK_MODULES[framework]


# ---------------------------------------------------------------------------
# Optional dependency guard
# ---------------------------------------------------------------------------
try:
    import datasets as _datasets_lib
    from datasets import load_dataset as _hf_load_dataset
    _datasets_lib.logging.set_verbosity_error()
    _DATASETS_AVAILABLE = True
except ImportError:
    _DATASETS_AVAILABLE = False


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
# Shared data layer
# ---------------------------------------------------------------------------

@dataclass
class HFQATask:
    """A single QA example with pre-formatted context."""
    question: str
    context: str
    answer: str


def _format_context(raw: Any, context_format: str, agent_class: str) -> str:
    """Flatten a raw HF context field into a plain string.

    ``none``  — no context field, returns ``""``
    ``plain`` — already a string, returned as-is
    Anything else — delegated to the task module's ``format_context(raw)``.
    """
    if context_format == "none":
        return ""
    if context_format == "plain":
        return str(raw)
    mod = _TASK_MODULES.get(agent_class)
    if mod is not None:
        return mod.format_context(raw)
    return str(raw)


def _load_hf_data(cfg: Dict[str, Any], split: str, n: int) -> List[HFQATask]:
    """Load up to *n* examples from *split* of the dataset described by *cfg*."""
    if not _DATASETS_AVAILABLE:
        raise ImportError(
            "The `datasets` package is required for hf: tasks. "
            "Install it with: pip install trace-bench[hf]"
        )
    ds = _hf_load_dataset(
        cfg["dataset_id"],
        cfg.get("dataset_config"),
        split=split,
    )
    q_field = cfg.get("question_field", "question")
    a_field = cfg.get("answer_field", "answer")
    ctx_fmt = cfg.get("context_format", "plain")
    agent_class = cfg.get("agent_class", "hfqa")

    tasks: List[HFQATask] = []
    for row in ds:
        if len(tasks) >= n:
            break
        question = str(row[q_field])
        raw_answer = row[a_field]
        if isinstance(raw_answer, dict):
            texts = raw_answer.get("text", [])
            answer = texts[0] if texts else ""
        else:
            answer = str(raw_answer)
        raw_context = row.get("context", row.get("passage", ""))
        context = _format_context(raw_context, ctx_fmt, agent_class)
        tasks.append(HFQATask(question=question, context=context, answer=answer))

    return tasks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_dataset(tasks: List[HFQATask], agent_class: str) -> Dict[str, List[Any]]:
    return _TASK_MODULES[agent_class].make_dataset(tasks)


def _make_guide(agent_class: str) -> Any:
    """Return an instantiated guide for *agent_class* (task-specific, framework-agnostic)."""
    return _TASK_MODULES[agent_class].make_guide()


def _make_agent(agent_class: str, framework: str, **kwargs: Any) -> Any:
    """Return an instantiated agent for *agent_class* using *framework*."""
    mod = _get_framework_module(framework)
    return mod.make_agent(agent_class, **kwargs)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_DEFAULT_INSTRUCTIONS = "Answer the question based on the context."


def build_trace_problem(
    task_id: str = "hotpot_qa",
    subtask: Optional[str] = None,
    num_train: Optional[int] = None,
    num_validate: Optional[int] = None,
    num_test: Optional[int] = None,
    initial_instructions: str = _DEFAULT_INSTRUCTIONS,
    framework: str = "trace",
    **_kwargs: Any,
) -> Dict[str, Any]:
    """Build a Trace-Bench task bundle for an HF QA task.

    Parameters
    ----------
    task_id:
        Key from ``hf_tasks.yaml`` (e.g. ``"hotpot_qa"`` or ``"bbeh"``).
        Injected automatically by the registry from the ``hf:<task_id>`` prefix.
    subtask:
        For multi-split datasets (e.g. BBEH), the specific split to load
        (e.g. ``"boolean_expressions"``). Injected by the registry from the
        ``hf:bbeh/<subtask>`` part of the task id.
    num_train:
        Number of training examples.  Falls back to the task's ``num_train``
        field in ``hf_tasks.yaml``, then to 10 if neither is set.
    num_validate:
        Number of validation examples.  Falls back to the task's ``num_validate``
        field in ``hf_tasks.yaml``, then to 0.
    num_test:
        Number of test examples.  Falls back to the task's ``num_test``
        field in ``hf_tasks.yaml``, then to 0.
    initial_instructions:
        Starting value for the ``meta_instructions`` parameter (QA agents only).
    framework:
        Agent framework to use. One of ``"trace"`` (default) or ``"dspy"``.
        Can be overridden per-task via ``eval_kwargs.framework`` in the run config.
    """
    cfg = _load_task_config(task_id)

    # Resolve dataset sizes: eval_kwargs > hf_tasks.yaml > hardcoded fallback.
    num_train    = num_train    if num_train    is not None else cfg.get("num_train",    10)
    num_validate = num_validate if num_validate is not None else cfg.get("num_validate",  0)
    num_test     = num_test     if num_test     is not None else cfg.get("num_test",      0)

    if subtask:
        # BBEH-style multi-split tasks: every subset (train/val/test) comes from
        # the same split, which is the subtask name (e.g. "boolean_expressions").
        split = subtask
        total = num_train + num_validate + num_test
        all_tasks = _load_hf_data(cfg, split, total)
        train_tasks = all_tasks[:num_train]
        val_tasks   = all_tasks[num_train:num_train + num_validate]
        test_tasks  = all_tasks[num_train + num_validate:]
    else:
        # Tasks with explicit train/test HF splits (e.g. HotpotQA):
        #   train + val  → loaded from `train_split`  (default: "train")
        #   test         → loaded from `test_split`   (default: same as train_split)
        # Falls back to the legacy `split` field if neither is set.
        fallback = cfg.get("split", "train")
        train_split = cfg.get("train_split", fallback)
        test_split  = cfg.get("test_split",  train_split)

        train_val_tasks = _load_hf_data(cfg, train_split, num_train + num_validate)
        train_tasks = train_val_tasks[:num_train]
        val_tasks   = train_val_tasks[num_train:]
        test_tasks  = _load_hf_data(cfg, test_split, num_test) if num_test > 0 else []

    objective = cfg.get("objective", "Optimize the agent's instructions for this QA task.")
    agent_class = cfg.get("agent_class", "hfqa")

    agent_kwargs: Dict[str, Any] = dict(
        initial_instructions=initial_instructions,
        node_description=cfg.get(
            "description",
            "Meta-instructions guiding the agent's reasoning and answer format.",
        ),
    )

    agent = _make_agent(agent_class, framework, **agent_kwargs)
    guide = _make_guide(agent_class)

    bundle: Dict[str, Any] = dict(
        param=agent,
        guide=guide,
        train_dataset=_make_dataset(train_tasks, agent_class),
        optimizer_kwargs=dict(objective=objective, memory_size=10),
        metadata=dict(
            benchmark="hf",
            task_id=task_id,
            subtask=subtask,
            dataset_id=cfg["dataset_id"],
            dataset_config=cfg.get("dataset_config"),
            num_train=num_train,
            num_validate=num_validate,
            num_test=num_test,
            framework=framework,
        ),
    )
    if val_tasks:
        bundle["validate_dataset"] = _make_dataset(val_tasks, agent_class)
    if test_tasks:
        bundle["test_dataset"] = _make_dataset(test_tasks, agent_class)
    return bundle
