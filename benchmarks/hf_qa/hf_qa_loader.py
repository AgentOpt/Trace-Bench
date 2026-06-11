"""HuggingFace QA loader for the `hf:` task suite.

Shared data layer: task config index, HFQATask dataclass, data loading, and
``build_trace_problem``.  Task-specific logic (data utilities + guide) lives in
the task module; framework-specific agent logic lives in the agent module:

  hotpot_qa.py        — make_dataset, format_context, check_answer, HotpotQAGuide
  bbeh.py             — make_dataset, format_context, eval_metric, BBEHGuide
  musique.py          — row_to_tasks, format_context, MuSiQueGuide
  gsm8k.py            — row_to_tasks, GSM8KGuide
  mcqa.py             — row_to_tasks, MCQGuide
  drop_qa.py          — row_to_tasks, DROPGuide
  qasper.py           — row_to_tasks, format_context, QasperGuide
  agents/trace_agent.py — TraceQAAgent, TraceBBEHAgent
  agents/dspy_agent.py  — DSPyQAAgent, DSPyBBEHAgent

Each task module must implement:
  - ``make_dataset(tasks: list) -> dict``   — convert HFQATask list to {inputs, infos}
  - ``format_context(raw: Any) -> str``     — flatten the raw HF context field
  - ``make_guide() -> Guide``               — return an instantiated guide
Optional task hooks:
  - ``row_to_tasks(row: dict, cfg: dict) -> list[HFQATask]``

Each agent module must implement:
  - ``make_agent(agent_class: str, **kwargs) -> agent``

Supported tasks are declared in ``hf_tasks.yaml``.  Adding a new task that
fits existing task semantics + agent patterns mostly requires a YAML entry.
Adding new task semantics requires a task module; adding a new framework
requires a new agent module + ``framework`` wiring below.

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

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

try:
    from . import bbeh as _bbeh_mod
    from . import drop_qa as _drop_qa_mod
    from . import gsm8k as _gsm8k_mod
    from . import hotpot_qa as _hotpot_qa_mod
    from . import mcqa as _mcqa_mod
    from . import musique as _musique_mod
    from . import qasper as _qasper_mod
    from .common import HFQATask
except ImportError:
    import bbeh as _bbeh_mod
    import drop_qa as _drop_qa_mod
    import gsm8k as _gsm8k_mod
    import hotpot_qa as _hotpot_qa_mod
    import mcqa as _mcqa_mod
    import musique as _musique_mod
    import qasper as _qasper_mod
    from common import HFQATask

# Registry mapping task_type (from hf_tasks.yaml) → task module.
# Task modules provide: make_dataset(), format_context(), make_guide().
# `task_type` defaults to `agent_class` for backward compatibility.
_TASK_MODULES: Dict[str, Any] = {
    "hfqa": _hotpot_qa_mod,
    "bbeh": _bbeh_mod,
    "drop": _drop_qa_mod,
    "gsm8k": _gsm8k_mod,
    "mcqa": _mcqa_mod,
    "musique": _musique_mod,
    "qasper": _qasper_mod,
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


def _task_type_key(cfg: Dict[str, Any]) -> str:
    """Return the task semantics key for a YAML task config."""
    task_type = cfg.get("task_type", cfg.get("agent_class", "hfqa"))
    if not isinstance(task_type, str) or not task_type:
        raise ValueError(f"task_type must be a non-empty string, got {task_type!r}")
    return task_type


def _get_task_module(task_type: str) -> Any:
    """Return the task module registered for a task_type key."""
    try:
        return _TASK_MODULES[task_type]
    except KeyError as exc:
        known = ", ".join(sorted(_TASK_MODULES))
        raise ValueError(
            f"Unknown HF task_type {task_type!r}. Known task types: {known}"
        ) from exc


def _format_context(raw: Any, context_format: str, task_type: str) -> str:
    """Flatten a raw HF context field into a plain string.

    ``none``  — no context field, returns ``""``
    ``plain`` — already a string, returned as-is
    Anything else — delegated to the task module's ``format_context(raw)``.
    """
    if context_format == "none":
        return ""
    if context_format == "plain":
        return str(raw)
    mod = _TASK_MODULES.get(task_type)
    if mod is not None:
        return mod.format_context(raw)
    return str(raw)


def _resolve_count(
    name: str,
    override: Optional[int],
    cfg: Dict[str, Any],
    default: int,
) -> int:
    """Resolve a dataset partition size and validate it."""
    value = override if override is not None else cfg.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be a non-negative integer, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def _load_hf_partitions(
    cfg: Dict[str, Any],
    specs: List[Tuple[str, str, int]],
) -> Dict[str, List[HFQATask]]:
    """Load named HF partitions without overlap when splits are reused."""
    partitions: Dict[str, List[HFQATask]] = {name: [] for name, _, _ in specs}
    by_split: Dict[str, List[Tuple[str, int]]] = {}

    for name, split, count in specs:
        if not isinstance(count, int) or isinstance(count, bool):
            raise TypeError(
                f"Partition {name!r} count must be an integer, got {count!r}"
            )
        if count < 0:
            raise ValueError(
                f"Partition {name!r} count must be non-negative, got {count}"
            )
        by_split.setdefault(split, []).append((name, count))

    for split, split_specs in by_split.items():
        total = sum(count for _, count in split_specs)
        loaded = _load_hf_data(cfg, split, total) if total > 0 else []
        start = 0
        for name, count in split_specs:
            partitions[name] = loaded[start : start + count]
            start += count

    return partitions


def _default_row_to_tasks(row: Dict[str, Any], cfg: Dict[str, Any]) -> List[HFQATask]:
    """Convert a generic scalar QA row into one HFQATask."""
    q_field = cfg.get("question_field", "question")
    a_field = cfg.get("answer_field", "answer")
    ctx_fmt = cfg.get("context_format", "plain")
    task_type = _task_type_key(cfg)
    context_field = cfg.get("context_field")

    if q_field not in row:
        raise KeyError(f"Dataset row missing question field {q_field!r}")
    if a_field not in row:
        raise KeyError(f"Dataset row missing answer field {a_field!r}")

    question = str(row[q_field])
    raw_answer = row[a_field]
    if isinstance(raw_answer, dict):
        texts = raw_answer.get("text", [])
        answer = str(texts[0]) if texts else ""
    else:
        answer = str(raw_answer)
    raw_context = (
        row.get(context_field, "")
        if context_field
        else row.get("context", row.get("passage", ""))
    )
    context = _format_context(raw_context, ctx_fmt, task_type)
    return [HFQATask(question=question, context=context, answer=answer)]


def _load_hf_data(cfg: Dict[str, Any], split: str, n: int) -> List[HFQATask]:
    """Load up to *n* examples from *split* of the dataset described by *cfg*."""
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError(f"n must be a non-negative integer, got {n!r}")
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if n == 0:
        return []
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
    task_type = _task_type_key(cfg)
    task_mod = _get_task_module(task_type)
    row_to_tasks = getattr(task_mod, "row_to_tasks", None)

    tasks: List[HFQATask] = []
    for row in ds:
        if len(tasks) >= n:
            break
        expanded = (
            row_to_tasks(row, cfg)
            if callable(row_to_tasks)
            else _default_row_to_tasks(row, cfg)
        )
        for task in expanded:
            tasks.append(task)
            if len(tasks) >= n:
                break

    return tasks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_dataset_from_cfg(
    tasks: List[HFQATask], cfg: Dict[str, Any]
) -> Dict[str, List[Any]]:
    """Convert HFQATask objects using the configured task semantics."""
    return _get_task_module(_task_type_key(cfg)).make_dataset(tasks)


def _make_guide(cfg: Dict[str, Any]) -> Any:
    """Return an instantiated guide for the configured task semantics."""
    return _get_task_module(_task_type_key(cfg)).make_guide()


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
    num_eval_repeats: int = 1,
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
    num_train = _resolve_count("num_train", num_train, cfg, 10)
    num_validate = _resolve_count("num_validate", num_validate, cfg, 0)
    num_test = _resolve_count("num_test", num_test, cfg, 0)

    if subtask:
        # BBEH-style multi-split tasks: every subset (train/val/test) comes from
        # the same split, which is the subtask name (e.g. "boolean_expressions").
        split = subtask
        train_split_name = split
        validate_split_name = split
        test_split_name = split
        partitions = _load_hf_partitions(
            cfg,
            [
                ("train", split, num_train),
                ("validate", split, num_validate),
                ("test", split, num_test),
            ],
        )
    else:
        # Tasks with explicit train/test HF splits (e.g. HotpotQA):
        #   train + val  → loaded from `train_split`  (default: "train")
        #   test         → loaded from `test_split`   (default: same as train_split)
        # Falls back to the legacy `split` field if neither is set.
        fallback = cfg.get("split", "train")
        train_split = cfg.get("train_split", fallback)
        test_split = cfg.get("test_split", train_split)
        train_split_name = train_split
        validate_split_name = train_split
        test_split_name = test_split
        partitions = _load_hf_partitions(
            cfg,
            [
                ("train", train_split, num_train),
                ("validate", train_split, num_validate),
                ("test", test_split, num_test),
            ],
        )

    train_tasks = partitions["train"]
    val_tasks = partitions["validate"]
    test_tasks = partitions["test"]

    objective = cfg.get(
        "objective", "Optimize the agent's instructions for this QA task."
    )
    agent_class = cfg.get("agent_class", "hfqa")

    if initial_instructions == _DEFAULT_INSTRUCTIONS:
        initial_instructions = cfg.get("initial_instructions", _DEFAULT_INSTRUCTIONS)
    agent_kwargs: Dict[str, Any] = dict(
        initial_instructions=initial_instructions,
        node_description=cfg.get(
            "description",
            "Meta-instructions guiding the agent's reasoning and answer format.",
        ),
    )

    agent = _make_agent(agent_class, framework, **agent_kwargs)
    guide = _make_guide(cfg)

    bundle: Dict[str, Any] = dict(
        param=agent,
        guide=guide,
        train_dataset=_make_dataset_from_cfg(train_tasks, cfg),
        optimizer_kwargs=dict(objective=objective, memory_size=10),
        metadata=dict(
            benchmark="hf",
            task_id=task_id,
            subtask=subtask,
            dataset_id=cfg["dataset_id"],
            task_type=_task_type_key(cfg),
            dataset_config=cfg.get("dataset_config"),
            num_train=num_train,
            num_validate=num_validate,
            num_test=num_test,
            train_split=train_split_name,
            validate_split=validate_split_name,
            test_split=test_split_name,
            framework=framework,
        ),
    )
    if val_tasks:
        bundle["validate_dataset"] = _make_dataset_from_cfg(val_tasks, cfg)
    if test_tasks:
        bundle["test_dataset"] = _make_dataset_from_cfg(test_tasks, cfg)
    bundle["num_eval_repeats"] = max(1, int(num_eval_repeats))
    return bundle
