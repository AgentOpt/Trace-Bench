"""Canonical repo-root path, computed once."""
from pathlib import Path

# src/trace_bench/_paths.py  →  parents[0]=trace_bench  →  parents[1]=src  →  parents[2]=repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
