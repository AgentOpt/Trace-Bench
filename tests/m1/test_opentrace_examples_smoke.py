import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


EXAMPLE_ALLOWLIST = {
    "autogen",
    "datasets",
    "dotenv",
    "dspy",
    "graphviz",
    "textgrad",
}


def _open_trace_root() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root.parent / "OpenTrace"


def _example_files() -> list[Path]:
    root = _open_trace_root() / "examples"
    if not root.exists():
        pytest.skip("OpenTrace examples directory not found")
    return sorted([p for p in root.rglob("*.py") if p.is_file()])


def _is_argparse_script(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return False
    return "argparse" in text or "ArgumentParser(" in text


def _extract_missing_module(output: str) -> str | None:
    match = re.search(r"No module named ['\"]([^'\"]+)['\"]", output)
    if match:
        return match.group(1)
    return None


def _run_smoke(path: Path):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_open_trace_root())

    env["TRACE_BENCH_SMOKE"] = "1"

    if _is_argparse_script(path):
        cmd = [sys.executable, str(path), "--help"]
    else:
        cmd = [
            sys.executable,
            "-c",
            f"import runpy; runpy.run_path(r'{path.as_posix()}', run_name='__not_main__')",
        ]

    try:
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            cwd=str(path.parent),
            timeout=30,
        )
        return proc
    except subprocess.TimeoutExpired:
        raise AssertionError(f"Smoke timed out for {path}")


@pytest.mark.parametrize("path", _example_files())
def test_opentrace_examples_smoke(path: Path):
    strict = os.environ.get("TRACE_BENCH_STRICT_EXAMPLES") == "1"
    proc = _run_smoke(path)
    if proc.returncode == 0:
        return

    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    missing = _extract_missing_module(output)
    if missing and missing in EXAMPLE_ALLOWLIST and not strict:
        pytest.skip(f"Optional dependency missing for {path.name}: {missing}")

    raise AssertionError(f"Smoke failed for {path}:\n{output}")
