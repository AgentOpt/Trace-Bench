"""Offline tests for the recursive_opt diagnostic task families.

All tests are row-level: they feed synthetic rows to ``row_to_tasks`` and
synthetic responses to the guides, so they run without any HuggingFace network
access (mirroring the existing tests/m1/test_hf_qa_ext.py convention).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

import benchmarks.hf_qa.hf_qa_loader as loader
from benchmarks.hf_qa import bbeh, code_exec, strategy_qa


# --------------------------------------------------------------------------- #
# YAML wiring
# --------------------------------------------------------------------------- #
def test_new_yaml_entries_resolve_to_task_types() -> None:
    """New task ids map to the expected task semantics."""
    expected = {
        "bbeh_horizon": "bbeh",      # reuses bbeh semantics (curated subtasks)
        "aqua_rat": "mcqa",          # reuses mcqa semantics
        "strategy_qa": "strategy_qa",
        "humaneval": "code_exec",
        "mbpp": "code_exec",
    }
    for task_id, task_type in expected.items():
        assert loader._load_task_config(task_id)["task_type"] == task_type


def test_bbeh_horizon_expands_subtasks() -> None:
    """bbeh_horizon carries the long-chain subtask list for credit_horizon."""
    cfg = loader._load_task_config("bbeh_horizon")
    assert "web_of_lies" in cfg["subtasks"]
    assert "multistep_arithmetic" in cfg["subtasks"]


def test_bbeh_failure_feedback_names_task_context() -> None:
    """Long-horizon feedback should identify the concrete failed task."""
    guide = bbeh.make_guide()
    score, feedback = guide.get_feedback(
        "Question: What is 2 + 2?\nAnswer:",
        "5",
        "4",
    )

    assert score == 0.0
    assert "What is 2 + 2" in feedback
    assert 'Expected: "4"' in feedback


# --------------------------------------------------------------------------- #
# StrategyQA boolean shaping
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw,expected", [
    (True, "yes"), (False, "no"), ("yes", "yes"), ("False", "no"), (1, "yes"),
])
def test_strategy_qa_gold_normalization(raw, expected) -> None:
    task = strategy_qa.row_to_tasks(
        {"question": "Is water wet?", "answer": raw},
        {"question_field": "question", "answer_field": "answer"})[0]
    assert task.answer == expected
    assert "Yes" in task.question  # the Yes/No instruction is injected


def test_strategy_qa_guide_scores_first_boolean_token() -> None:
    guide = strategy_qa.make_guide()
    task = strategy_qa.row_to_tasks(
        {"question": "Q?", "answer": True},
        {"question_field": "question", "answer_field": "answer"})[0]
    assert guide.get_feedback(task.question, "Yes, because ...", task)[0] == 1.0
    assert guide.get_feedback(task.question, "No.", task)[0] == 0.0


def test_strategy_qa_missing_field_raises() -> None:
    with pytest.raises(KeyError, match="question field"):
        strategy_qa.row_to_tasks({"answer": True}, {})


# --------------------------------------------------------------------------- #
# Code execution (HumanEval / MBPP semantics)
# --------------------------------------------------------------------------- #
def _humaneval_row(correct: bool) -> dict:
    body = "    return a + b\n" if correct else "    return a - b\n"
    return {
        "prompt": "def add(a, b):\n    '''add'''\n",
        "entry_point": "add",
        "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n",
        "_body": body,
    }


def test_code_exec_passes_correct_humaneval_solution() -> None:
    row = _humaneval_row(correct=True)
    task = code_exec.row_to_tasks(row, {"question_field": "prompt",
        "test_field": "test", "entry_field": "entry_point"})[0]
    guide = code_exec.make_guide(timeout=10)
    response = "def add(a, b):\n    return a + b\n"
    score, feedback = guide.get_feedback(task.question, response, task)
    assert score == 1.0 and "passes" in feedback


def test_code_exec_fails_wrong_solution_without_raising() -> None:
    row = _humaneval_row(correct=True)
    task = code_exec.row_to_tasks(row, {"question_field": "prompt",
        "test_field": "test", "entry_field": "entry_point"})[0]
    guide = code_exec.make_guide(timeout=10)
    score, feedback = guide.get_feedback(task.question, "def add(a, b):\n    return a - b\n", task)
    assert score == 0.0 and "fails" in feedback


def test_code_exec_handles_fenced_code_blocks() -> None:
    row = _humaneval_row(correct=True)
    task = code_exec.row_to_tasks(row, {"question_field": "prompt",
        "test_field": "test", "entry_field": "entry_point"})[0]
    guide = code_exec.make_guide(timeout=10)
    fenced = "Here is the solution:\n```python\ndef add(a, b):\n    return a + b\n```"
    assert guide.get_feedback(task.question, fenced, task)[0] == 1.0


def test_code_exec_joins_mbpp_test_list() -> None:
    """MBPP-style list-valued tests are joined into a runnable block."""
    row = {"prompt": "Write add.", "entry_point": "add",
           "test_list": ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]}
    task = code_exec.row_to_tasks(row, {"question_field": "prompt",
        "test_field": "test_list", "entry_field": "entry_point"})[0]
    assert "assert add(1, 2) == 3" in task.meta["test"]
    guide = code_exec.make_guide(timeout=10)
    # MBPP asserts run at module import; no check() needed
    assert guide.get_feedback(task.question, "def add(a, b):\n    return a + b\n", task)[0] == 1.0


def test_code_exec_times_out_safely() -> None:
    row = {"prompt": "loop", "entry_point": "f",
           "test": "def check(candidate):\n    candidate()\n"}
    task = code_exec.row_to_tasks(row, {"question_field": "prompt",
        "test_field": "test", "entry_field": "entry_point"})[0]
    guide = code_exec.make_guide(timeout=1)
    score, feedback = guide.get_feedback(task.question, "def f():\n    while True:\n        pass\n", task)
    assert score == 0.0 and "timed out" in feedback


def test_code_exec_empty_response_scores_zero() -> None:
    row = _humaneval_row(correct=True)
    task = code_exec.row_to_tasks(row, {"question_field": "prompt",
        "test_field": "test", "entry_field": "entry_point"})[0]
    assert code_exec.make_guide().get_feedback(task.question, "", task)[0] == 0.0


def test_make_guide_passes_timeout_from_cfg() -> None:
    """Loader introspection wires cfg['timeout'] into code_exec.make_guide."""
    guide = loader._make_guide({"task_type": "code_exec", "timeout": 3})
    assert guide._timeout == 3
    # modules without a timeout param are unaffected
    assert loader._make_guide({"task_type": "gsm8k"}) is not None


# --------------------------------------------------------------------------- #
# Guide efficiency: directional feedback (the optimizer's learning signal)
# --------------------------------------------------------------------------- #
def test_code_exec_feedback_names_the_failing_assertion() -> None:
    """A wrong solution must yield a DIRECTIONAL diagnostic, not 'tests failed'.

    The feedback string is what optimizer.backward() consumes, so it must point
    at the specific failure (exception + failing line) for efficient repair.
    """
    row = {"prompt": "def add(a,b):\n    '''add'''\n", "entry_point": "add",
           "test": "def check(candidate):\n    assert candidate(2, 3) == 5, 'add must sum'\n"}
    task = code_exec.row_to_tasks(row, {"question_field": "prompt",
        "test_field": "test", "entry_field": "entry_point"})[0]
    guide = code_exec.make_guide(timeout=10)
    score, feedback = guide.get_feedback(task.question, "def add(a, b):\n    return a - b\n", task)
    assert score == 0.0
    # names the exception type and surfaces the assertion (not just "failed")
    assert "AssertionError" in feedback
    assert "add must sum" in feedback or "assert candidate(2, 3) == 5" in feedback


def test_code_exec_timeout_feedback_is_actionable() -> None:
    row = {"prompt": "loop", "entry_point": "f",
           "test": "def check(candidate):\n    candidate()\n"}
    task = code_exec.row_to_tasks(row, {"question_field": "prompt",
        "test_field": "test", "entry_field": "entry_point"})[0]
    score, feedback = code_exec.make_guide(timeout=1).get_feedback(
        task.question, "def f():\n    while True:\n        pass\n", task)
    assert score == 0.0 and "terminate" in feedback


# --------------------------------------------------------------------------- #
# Trace surface: code tasks bind to the code agent (traces prompt, passes code)
# --------------------------------------------------------------------------- #
def test_code_tasks_use_code_agent_class() -> None:
    """humaneval/mbpp must use agent_class: code (not bbeh) so the response
    code is passed through to the unit-test guide unmangled."""
    for tid in ("humaneval", "mbpp"):
        assert loader._load_task_config(tid)["agent_class"] == "code"


def test_code_agent_traces_prompt_and_passes_response_through() -> None:
    pytest.importorskip("opto", reason="OpenTrace required for agent wiring")
    from benchmarks.hf_qa.agents import trace_agent

    agent = trace_agent.make_agent("code")
    # trainable code-generation prompt is exposed for the optimizer
    assert agent.prompt_template.trainable
    # there is no extract_answer step that would split on "Answer:" and mangle code
    assert not hasattr(agent, "extract_answer")
