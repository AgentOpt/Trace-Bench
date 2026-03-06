from __future__ import annotations

import os
from pathlib import Path

from trace_bench.config import load_config
from trace_bench.runner import BenchRunner
from trace_bench.veribench_adapter import _clear_caches


def test_m2_veribench_smoke_config_runs(tmp_path: Path):
    _clear_caches()
    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root)

    cfg = load_config(repo_root / "configs" / "m2_veribench_smoke.yaml")
    cfg.runs_dir = str(tmp_path / "runs")

    summary = BenchRunner(cfg).run()
    suites = {row.get("suite") for row in summary.results}
    # veribench tasks will either execute (if dataset available) or skip
    assert {"internal", "trace_examples", "veribench"}.issubset(suites)
