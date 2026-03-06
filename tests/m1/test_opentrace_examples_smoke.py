from __future__ import annotations

import py_compile
import tempfile
from pathlib import Path

import pytest


def _open_trace_root() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root.parent / "OpenTrace"


def _example_files() -> list[Path]:
    root = _open_trace_root() / "examples"
    if not root.exists():
        pytest.skip("OpenTrace examples directory not found")
    return sorted(p for p in root.rglob("*.py") if p.is_file())


@pytest.mark.parametrize("path", _example_files())
def test_opentrace_examples_smoke(path: Path):
    with tempfile.TemporaryDirectory() as tmp_dir:
        py_compile.compile(str(path), cfile=str(Path(tmp_dir) / f"{path.stem}.pyc"), doraise=True)
