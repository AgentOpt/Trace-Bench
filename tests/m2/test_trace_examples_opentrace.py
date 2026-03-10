from __future__ import annotations

import os
from pathlib import Path
import shutil
from uuid import uuid4

from trace_bench.config import RunConfig
from trace_bench.registry import discover_trace_examples, load_task_bundle
from trace_bench.runner import BenchRunner


_NEW_TASK_IDS = {
    "trace_examples:opentrace_greeting",
    "trace_examples:opentrace_train_single_node",
}


def test_discover_trace_examples_includes_opentrace_wrappers():
    ids = {spec.id for spec in discover_trace_examples()}
    assert _NEW_TASK_IDS.issubset(ids)


def test_opentrace_trace_example_bundle_contract():
    os.chdir(Path(__file__).resolve().parents[2])
    required = {"param", "guide", "train_dataset", "optimizer_kwargs", "metadata"}
    for task_id in sorted(_NEW_TASK_IDS):
        bundle = load_task_bundle(task_id, "benchmarks/LLM4AD/benchmark_tasks")
        missing = required - set(bundle.keys())
        assert not missing, f"{task_id} missing keys: {sorted(missing)}"


def test_opentrace_trace_examples_run_in_stub_mode():
    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root)
    cfg = RunConfig.from_dict(
        {
            "tasks": [{"id": task_id} for task_id in sorted(_NEW_TASK_IDS)],
            "trainers": [
                {
                    "id": "GEPA-Base",
                    "params_variants": [
                        {
                            "gepa_iters": 1,
                            "gepa_train_bs": 1,
                            "gepa_merge_every": 1,
                            "gepa_pareto_subset": 2,
                        }
                    ],
                }
            ],
            "seeds": [123],
            "mode": "stub",
        }
    )
    run_root = repo_root / "pytest_tmp_global" / f"opentrace-{uuid4().hex}" / "runs"
    run_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        cfg.runs_dir = str(run_root)
        summary = BenchRunner(cfg).run()
        assert len(summary.results) == len(_NEW_TASK_IDS)
        assert all(row.get("status") == "ok" for row in summary.results)
    finally:
        shutil.rmtree(run_root.parent, ignore_errors=True)
