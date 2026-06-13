"""Code-generation tasks scored by executing unit tests (HumanEval / MBPP).

This is the one genuinely-new task semantics in this batch: the response is a
Python function body, and the guide SCORES IT BY RUNNING the dataset's own unit
tests in an isolated subprocess. That gives the recursive optimizer a
``CodeArtifactLevel`` surface with a real, non-LLM, deterministic pass/fail
signal (like the validator-style component in recursive_opt example B) — no
judge model required.

Safety: candidate code is executed, so each evaluation runs in a separate
``python -c`` subprocess with a wall-clock timeout and is never ``exec``'d in the
host process. The timeout/inability-to-run cases score 0.0 with feedback rather
than raising, so a single bad candidate cannot abort a sweep.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from .common import HFQATask, response_text
except ImportError:
    from common import HFQATask, response_text

try:
    from opto.trainer.guide import Guide
except Exception:
    Guide = object  # type: ignore[assignment,misc]

# Default per-candidate execution budget (seconds). Overridable via cfg.
_DEFAULT_TIMEOUT = 10


def _extract_code(response: Any) -> str:
    """Pull a code body out of an LLM response, tolerating ``` fences."""
    text = response_text(response)
    if "```" in text:
        # take the content of the first fenced block
        parts = text.split("```")
        if len(parts) >= 2:
            block = parts[1]
            # drop an optional language tag on the first line
            if "\n" in block:
                first, rest = block.split("\n", 1)
                if first.strip().lower() in {"python", "py", ""}:
                    return rest
            return block
    return text


def row_to_tasks(row: Dict[str, Any], cfg: Dict[str, Any]) -> List[HFQATask]:
    """Convert one code row into a code task.

    Supports both schemas via cfg fields (defaults target HumanEval):
      - prompt_field : the function signature + docstring (the question)
      - test_field   : unit-test source string
      - entry_field  : name of the function under test (for the test harness)
      - setup_field  : optional import/setup preamble (MBPP: 'test_setup_code')
    """
    prompt_field = cfg.get("question_field", cfg.get("prompt_field", "prompt"))
    test_field = cfg.get("test_field", "test")
    entry_field = cfg.get("entry_field", "entry_point")
    setup_field = cfg.get("setup_field")
    if prompt_field not in row:
        raise KeyError(f"Code row missing prompt field {prompt_field!r}")
    if test_field not in row:
        raise KeyError(f"Code row missing test field {test_field!r}")

    # Test source may be a single string (HumanEval) or a list of assert
    # strings (MBPP 'test_list'); join lists into a runnable test block.
    raw_test = row[test_field]
    test_src = "\n".join(str(t) for t in raw_test) if isinstance(raw_test, (list, tuple)) else str(raw_test)

    meta = {
        "test": test_src,
        "entry_point": str(row.get(entry_field, "")),
        "setup": str(row.get(setup_field, "")) if setup_field else "",
    }
    return [HFQATask(question=str(row[prompt_field]), context="", answer=meta["entry_point"], meta=meta)]


def format_context(raw: Any) -> str:
    """Code tasks carry their spec in the prompt; no separate context."""
    return ""


def make_dataset(tasks: List[HFQATask]) -> Dict[str, List[Any]]:
    """Question-only dataset shape."""
    return {"inputs": [task.question for task in tasks], "infos": list(tasks)}


def _run_unit_tests(candidate: str, info: HFQATask, timeout: int) -> Tuple[bool, str]:
    """Execute candidate + dataset tests in an isolated subprocess.

    On failure, return a DIRECTIONAL diagnostic — the exception type, message,
    and the specific failing assertion/line where available — because that text
    becomes the optimizer's entire learning signal (``optimizer.backward(target,
    feedback)``). A bare "tests failed" gives the optimizer nothing to act on.
    """
    entry = info.meta.get("entry_point", "")
    setup = info.meta.get("setup", "")
    test_src = info.meta.get("test", "")
    # HumanEval tests define check(candidate); MBPP tests are bare asserts.
    harness = "\n".join([
        "import traceback",
        setup,
        candidate,
        test_src,
        # Call check(entry) when the test file defines it (HumanEval); MBPP's
        # module-level asserts have already run on import of test_src above.
        f"\nif 'check' in dir():\n    check({entry})\n" if entry else "",
        "print('PASS')",
    ])
    # Wrap execution so we can surface the precise failing line, not just the
    # last stderr line (which often loses the assertion text).
    runner = (
        "import traceback, sys\n"
        "try:\n"
        + "\n".join("    " + line for line in harness.splitlines())
        + "\nexcept Exception as exc:\n"
        "    tb = traceback.extract_tb(sys.exc_info()[2])\n"
        "    frame = tb[-1] if tb else None\n"
        "    loc = f' at: {frame.line}' if frame and frame.line else ''\n"
        "    print(f'FAIL: {type(exc).__name__}: {exc}{loc}', file=sys.stderr)\n"
        "    sys.exit(1)\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "candidate_test.py"
        script.write_text(runner)
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False, (f"timed out after {timeout}s — the function likely has an "
                           "infinite loop or is far too slow; ensure it terminates")
        if proc.returncode == 0 and "PASS" in proc.stdout:
            return True, "all unit tests passed"
        # Prefer our structured FAIL line; fall back to raw stderr/stdout.
        for line in (proc.stderr or "").splitlines():
            if line.startswith("FAIL:"):
                return False, line[len("FAIL:"):].strip()
        tail = (proc.stderr or proc.stdout or "no output").strip().splitlines()
        return False, (tail[-1][:200] if tail else "unit tests failed")


class CodeExecGuide(Guide):  # type: ignore[misc]
    """Guide that scores generated code by unit-test pass/fail (1.0 / 0.0)."""

    def __init__(self, timeout: int = _DEFAULT_TIMEOUT) -> None:
        self._timeout = int(timeout)

    def get_feedback(
        self, task: Any, response: Any, info: HFQATask, **kwargs: Any
    ) -> Tuple[float, str]:
        """Run the dataset's tests against the extracted candidate code."""
        candidate = _extract_code(response)
        if not candidate.strip():
            return 0.0, "Empty code response; produce a complete function definition."
        passed, detail = _run_unit_tests(candidate, info, self._timeout)
        if passed:
            return 1.0, "The implementation passes all unit tests."
        return 0.0, (
            f"The implementation fails: {detail}. Revise the function so this "
            "specific failure is resolved while keeping all other cases correct."
        )


def make_guide(timeout: int = _DEFAULT_TIMEOUT) -> Any:
    """Return an instantiated code-execution guide."""
    return CodeExecGuide(timeout=timeout)
