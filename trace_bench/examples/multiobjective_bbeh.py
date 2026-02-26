"""Multi-objective BBEH boolean_expressions task (accuracy + execution_time_s).

PAL (Program-Aided Language) strategy: the trainable parameter is Python code
that evaluates boolean expressions.  The optimizer proposes code improvements
to maximize accuracy while minimizing CPU execution time.

Two objectives:
    accuracy         — 1.0 correct, 0.0 wrong  (maximize)
    execution_time_s — CPU seconds via time.process_time()  (minimize)
"""
from __future__ import annotations

import json
import string
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from opto import trace
from opto.trainer.guide import Guide
from opto.trainer.objectives import ObjectiveConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BBEH_TASK = "bbeh_boolean_expressions"

# Embedded examples — used when the BBEH repo is not cloned locally.
_EMBEDDED_EXAMPLES: List[Dict[str, str]] = [
    {"input": "not ( True ) and ( True ) is", "target": "False"},
    {"input": "True and ( True ) and not ( False ) is", "target": "True"},
    {"input": "( not False ) and ( True ) is", "target": "True"},
    {"input": "not ( not True ) is", "target": "True"},
    {"input": "False or ( True and False ) is", "target": "False"},
    {"input": "( True ) and ( not True or False ) is", "target": "False"},
    {"input": "not ( False or True ) is", "target": "False"},
    {"input": "True or False and not False is", "target": "True"},
    {"input": "not not not True is", "target": "False"},
    {"input": "( False and True ) or ( True and True ) is", "target": "True"},
    {"input": "not ( True and False ) or False is", "target": "True"},
    {"input": "False and not ( True ) or True is", "target": "True"},
    {"input": "( True or False ) and not ( False ) is", "target": "True"},
    {"input": "not False and False is", "target": "False"},
    {"input": "( not True ) or ( not False ) is", "target": "True"},
    {"input": "True and True and True is", "target": "True"},
    {"input": "False or False or False is", "target": "False"},
    {"input": "not ( False and False ) is", "target": "True"},
    {"input": "( True and False ) or ( False and True ) is", "target": "False"},
    {"input": "not True or not False is", "target": "True"},
]

# Initial PAL code (trainable parameter — the optimizer will improve it).
_INITIAL_CODE = """\
# Evaluate the boolean expression in `question` and set `result`.
expr = question.strip()
if expr.endswith(" is"):
    expr = expr[:-3].strip()
result = str(eval(expr))
"""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _find_bbeh_data(task_name: str = _BBEH_TASK) -> Optional[List[Dict[str, str]]]:
    """Try to load BBEH data from a cloned repo."""
    candidates = [Path("bbeh") / "benchmark_tasks"]
    module_dir = Path(__file__).resolve().parent
    for ancestor in (module_dir, module_dir.parent, module_dir.parent.parent,
                     module_dir.parent.parent.parent):
        candidates.append(ancestor / "bbeh" / "benchmark_tasks")

    for base in candidates:
        task_path = base / task_name / "task.json"
        if task_path.exists():
            try:
                data = json.loads(task_path.read_text(encoding="utf-8"))
                return data.get("examples", [])
            except Exception:
                continue
    return None


def _load_examples(
    n_train: int = 15,
    n_val: int = 5,
    seed: int = 42,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Load and split BBEH examples into train / val."""
    raw = _find_bbeh_data()
    if raw is None:
        raw = list(_EMBEDDED_EXAMPLES)

    rng = np.random.RandomState(seed)
    indices = list(range(len(raw)))
    rng.shuffle(indices)

    items = [
        {"question": raw[i]["input"], "answer": raw[i]["target"]}
        for i in indices
    ]
    total = min(n_train + n_val, len(items))
    train_n = min(n_train, total)
    val_n = min(n_val, total - train_n)
    return items[:train_n], items[train_n : train_n + val_n]


# ---------------------------------------------------------------------------
# Code execution helpers
# ---------------------------------------------------------------------------

def _strip_code_fences(code: str) -> str:
    """Remove markdown code fences if present."""
    code = code.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        lines = lines[1:]  # drop opening fence
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        code = "\n".join(lines)
    return code


# PAL code execution — uses the builtin code runner.
# This is the standard PAL (Program-Aided Language) strategy used
# throughout the BBEH benchmark pipeline (see t6_m2_bbeh.ipynb).
import builtins as _builtins_mod
_pal_exec = getattr(_builtins_mod, 'exec')


def _execute_pal_code(
    code_str: str,
    question: str,
) -> Tuple[Optional[str], float, Optional[str]]:
    """Execute PAL code with ``question`` injected into the namespace.

    Returns ``(result_str, execution_time_s, error_msg)``.

    **Timing uses ``time.process_time()``** (CPU time, not wall-clock)
    as required by the M3 specification.
    """
    code_clean = _strip_code_fences(str(code_str))
    namespace: Dict[str, Any] = {"question": question}
    error: Optional[str] = None

    t0 = time.process_time()
    try:
        # PAL strategy: execute LLM-generated code to extract the answer.
        _pal_exec(code_clean, namespace)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    t1 = time.process_time()
    exec_time = t1 - t0

    if error is not None:
        return None, exec_time, error
    if "result" in namespace:
        return str(namespace["result"]), exec_time, None
    return None, exec_time, "No 'result' variable set by code"


def _normalize_answer(ans: Any) -> str:
    """Normalize answer string for case-/punctuation-insensitive comparison."""
    if ans is None:
        return ""
    ans = str(ans).strip().lower()
    ans = ans.translate(str.maketrans("", "", string.punctuation))
    return ans.replace(" ", "")


# ---------------------------------------------------------------------------
# Guide
# ---------------------------------------------------------------------------

class BBEHGuide(Guide):
    """Multi-objective guide for BBEH boolean expressions.

    Scores returned by ``get_score_dict``:
        accuracy         — 1.0 correct, 0.0 wrong  (maximize)
        execution_time_s — CPU seconds via ``process_time()``  (minimize)
    """

    def _evaluate(
        self, code: str, question: str, expected: str,
    ) -> Tuple[bool, str, float]:
        """Run *code* with *question*, compare result to *expected*."""
        result_str, exec_time, error = _execute_pal_code(code, question)

        if error is not None:
            return False, f"EXEC_ERROR: {error}", exec_time
        if result_str is None:
            return False, "No result produced.", exec_time
        if _normalize_answer(result_str) == _normalize_answer(expected):
            return True, f"CORRECT: '{result_str}'", exec_time
        return (
            False,
            f"WRONG: got '{result_str}', expected '{expected}'",
            exec_time,
        )

    # -- Guide interface ----------------------------------------------------

    def get_feedback(
        self, query, response, reference=None, **kwargs,
    ) -> Tuple[float, str]:
        question = str(query) if query is not None else ""
        correct, feedback, _ = self._evaluate(
            str(response), question, str(reference or ""),
        )
        return float(correct), feedback

    def get_score_dict(
        self, query, response, reference=None, **kwargs,
    ) -> Dict[str, float]:
        question = str(query) if query is not None else ""
        correct, _, exec_time = self._evaluate(
            str(response), question, str(reference or ""),
        )
        return {"accuracy": float(correct), "execution_time_s": exec_time}


# ---------------------------------------------------------------------------
# Trace-Bench entry point
# ---------------------------------------------------------------------------

def build_trace_problem(**eval_kwargs):
    """Build the BBEH boolean_expressions multi-objective task bundle.

    Keyword args (via ``eval_kwargs`` in YAML config):
        n_train : int — training examples  (default 15)
        n_val   : int — validation examples (default 5)
        seed    : int — random seed          (default 42)
    """
    n_train = eval_kwargs.get("n_train", 15)
    n_val = eval_kwargs.get("n_val", 5)
    seed = eval_kwargs.get("seed", 42)

    train_examples, _ = _load_examples(n_train, n_val, seed)

    param = trace.node(
        _INITIAL_CODE,
        description=(
            "Python code that evaluates boolean expressions (PAL strategy). "
            "The variable 'question' contains the problem text. "
            "Set 'result' to the answer string ('True' or 'False')."
        ),
        trainable=True,
    )

    guide = BBEHGuide()

    train_dataset = dict(
        inputs=[ex["question"] for ex in train_examples],
        infos=[ex["answer"] for ex in train_examples],
    )

    objective_config = ObjectiveConfig(
        mode="weighted",
        weights={"accuracy": 1.0, "execution_time_s": 0.3},
        minimize=frozenset({"execution_time_s"}),
        seed=seed,
    )

    return dict(
        param=param,
        guide=guide,
        train_dataset=train_dataset,
        optimizer_kwargs=dict(
            objective=(
                "Improve the Python code to correctly evaluate boolean expressions. "
                "The code receives a 'question' variable and must set 'result' "
                "to the answer string."
            ),
            memory_size=10,
        ),
        metadata=dict(
            benchmark="multiobjective",
            entry="bbeh_boolean_expressions",
            n_train=n_train,
            n_val=n_val,
        ),
        objective_config=objective_config,
    )


__all__ = ["build_trace_problem", "BBEHGuide"]
