#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""trainers_benchmark.py — backward-compatible wrapper.

DEPRECATED: Use 'trace-bench run --config <yaml>' directly.

This script preserves the old CLI interface (--tasks, --task, --algos, etc.)
but delegates to the trace-bench runner internally.

Examples (old style, still works):
    python trainers_benchmark.py --tasks ./benchmark_tasks --task circle_packing
    python trainers_benchmark.py --tasks ./benchmark_tasks --task circle_packing --algos PrioritySearch --ps-steps 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


def _build_config_dict(args: argparse.Namespace) -> dict:
    """Map old CLI args to a RunConfig-compatible dict."""
    task_keys = [key.strip() for key in args.task.split(",") if key.strip()]
    tasks = []
    for key in task_keys:
        task_id = f"llm4ad:{key}" if ":" not in key else key
        eval_kwargs = json.loads(args.eval_kwargs) if args.eval_kwargs else {}
        tasks.append({"id": task_id, "eval_kwargs": eval_kwargs})

    algo_names = [s.strip() for s in args.algos.split(",") if s.strip()]
    trainers = []
    for name in algo_names:
        trainer_entry: dict = {"id": name}
        params: dict = {}
        if name == "PrioritySearch":
            params.update({
                "ps_steps": args.ps_steps,
                "ps_batches": args.ps_batches,
                "ps_candidates": args.ps_candidates,
                "ps_proposals": args.ps_proposals,
                "ps_mem_update": args.ps_mem_update,
                "threads": args.threads,
            })
        elif name.startswith("GEPA"):
            params.update({
                "gepa_iters": args.gepa_iters,
                "gepa_train_bs": args.gepa_train_bs,
                "gepa_merge_every": args.gepa_merge_every,
                "gepa_pareto_subset": args.gepa_pareto_subset,
                "threads": args.threads,
            })
        else:
            params["threads"] = args.threads

        if args.optimizer_kwargs:
            trainer_entry["optimizer_kwargs"] = json.loads(args.optimizer_kwargs)

        trainer_entry["params"] = params
        trainers.append(trainer_entry)

    return {
        "tasks": tasks,
        "trainers": trainers,
        "seeds": [123],
        "mode": "real",
        "max_workers": 1,
        "resume": "auto",
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="DEPRECATED: Use 'trace-bench run --config <yaml>' directly."
    )
    ap.add_argument("--tasks", type=str, required=True,
                     help="Folder with benchmark task directories")
    ap.add_argument("--task", type=str, required=True,
                     help='Task key(s) (comma-separated)')
    ap.add_argument("--algos", type=str, default="PrioritySearch",
                     help="Comma-separated algorithms")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--optimizer-kwargs", type=str, default="")
    ap.add_argument("--eval-kwargs", type=str, default="")
    ap.add_argument("--gepa-iters", type=int, default=3)
    ap.add_argument("--gepa-train-bs", type=int, default=2)
    ap.add_argument("--gepa-merge-every", type=int, default=2)
    ap.add_argument("--gepa-pareto-subset", type=int, default=3)
    ap.add_argument("--ps-steps", type=int, default=2)
    ap.add_argument("--ps-batches", type=int, default=2)
    ap.add_argument("--ps-candidates", type=int, default=3)
    ap.add_argument("--ps-proposals", type=int, default=3)
    ap.add_argument("--ps-mem-update", type=int, default=2)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    print(
        "WARNING: trainers_benchmark.py is deprecated. "
        "Use 'trace-bench run --config <yaml>' directly.",
        file=sys.stderr,
    )

    args = _parse_args(argv)
    config_dict = _build_config_dict(args)

    # Write temp config and delegate to trace-bench runner
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    try:
        json.dump(config_dict, tmp, indent=2)
        tmp.close()

        # Import here to avoid circular imports at module level
        from trace_bench.cli import cmd_run

        return cmd_run(
            config_path=tmp.name,
            root=str(Path(args.tasks).resolve()),
        )
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
