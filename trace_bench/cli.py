from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

from trace_bench.config import load_config
from trace_bench.matrix import compute_run_id, expand_matrix
from trace_bench.registry import (
    discover_tasks,
    discover_trainers,
    expand_special_tasks,
    load_task_bundle,
)
from trace_bench.resolve import merge_kwargs, resolve_trainer_kwargs
from trace_bench.runner import BenchRunner, _has_trainables
from trace_bench.artifacts import init_run_dir, write_manifest
from trace_bench.ui import launch_ui


def cmd_list_tasks(root: str, bench: str | None = None) -> int:
    specs = discover_tasks(root, bench=bench)
    for spec in specs:
        print(spec.id)
    return 0


def cmd_list_trainers(include_all: bool = False) -> int:
    specs = discover_trainers()
    for spec in specs:
        if not include_all and not spec.available:
            continue
        status = "available" if spec.available else "unavailable"
        print(f"{spec.id}\t{status}")
    return 0


def _task_in_bench(task_key: str, bench: str | None) -> bool:
    if not bench:
        return True
    if ":" not in task_key:
        task_key = f"llm4ad:{task_key}"
    if "veribench" in bench and task_key.startswith("veribench:"):
        return True
    if "trace_examples" in bench and task_key.startswith("trace_examples:"):
        return True
    if "internal" in bench and task_key.startswith("internal:"):
        return True
    if "llm4ad" in bench and task_key.startswith("llm4ad:"):
        return True
    return False


_ALLOWED_TRAINER_KWARGS = {
    "threads",
    "num_threads",
    "num_epochs",
    "num_steps",
    "num_batches",
    "num_candidates",
    "num_proposals",
    "num_iters",
    "num_search_iterations",
    "train_batch_size",
    "merge_every",
    "pareto_subset_size",
    "ps_steps",
    "ps_batches",
    "ps_candidates",
    "ps_proposals",
    "ps_mem_update",
    "gepa_iters",
    "gepa_train_bs",
    "gepa_merge_every",
    "gepa_pareto_subset",
    # LLM4AD pass-through knobs (merged into params_variants by config parser)
    "optimizer_kwargs",
    "eval_kwargs",
}


def _resolve_symbol(module_name: str, symbol: str) -> bool:
    try:
        module = __import__(module_name, fromlist=[symbol])
        return hasattr(module, symbol)
    except Exception:
        return False


def _default_timeout(mode: str) -> float:
    """Mode-based default timeout: stub=0 (in-process), real=600s (10min).

    Stub mode uses DummyLLM which returns instantly, so subprocess
    isolation adds pure overhead (each child re-imports the full
    dependency tree).  Real mode talks to an LLM API that can hang,
    so a 10-minute safety net is applied by default.
    """
    return 0.0 if mode == "stub" else 600.0


def _validate_trainer_params(trainer, errors: list[str]) -> None:
    for params in trainer.params_variants or [{}]:
        for key in params.keys():
            if key not in _ALLOWED_TRAINER_KWARGS:
                errors.append(f"unknown trainer kwarg '{key}' for {trainer.id}")

    if trainer.optimizer and not _resolve_symbol("opto.optimizers", trainer.optimizer):
        errors.append(f"optimizer not found: {trainer.optimizer}")
    if trainer.guide and not _resolve_symbol("opto.trainer.guide", trainer.guide):
        errors.append(f"guide not found: {trainer.guide}")
    if trainer.logger and not _resolve_symbol("opto.trainer.loggers", trainer.logger):
        errors.append(f"logger not found: {trainer.logger}")


def cmd_validate(
    config_path: str,
    root: str,
    bench: str | None = None,
    strict: bool = False,
    runs_dir: str | None = None,
) -> int:
    cfg = load_config(config_path)
    if runs_dir:
        cfg.runs_dir = runs_dir
    tasks_root = Path(root)
    cfg.tasks = expand_special_tasks(cfg.tasks, tasks_root)
    errors = 0
    if bench:
        discover_tasks(tasks_root, bench=bench)
    trainers = discover_trainers()
    trainer_ids = {t.id for t in trainers if t.available}
    strict_errors: list[str] = []
    for trainer in cfg.trainers:
        if trainer.id not in trainer_ids:
            errors += 1
            print(f"[FAIL] trainer {trainer.id}: not available")
        if strict:
            _validate_trainer_params(trainer, strict_errors)
    if strict_errors:
        for msg in strict_errors:
            print(f"[FAIL] {msg}")
        errors += len(strict_errors)

    bundle_cache: dict[str, dict | None] = {}

    def _bundle_cache_key(task) -> str:
        eval_sig = json.dumps(task.eval_kwargs or {}, sort_keys=True)
        return f"{task.id}|{eval_sig}"

    def _cache_bundle(task, bundle):
        bundle_cache[_bundle_cache_key(task)] = bundle

    def _get_cached_bundle(task):
        key = _bundle_cache_key(task)
        if key in bundle_cache:
            return bundle_cache[key]
        try:
            bundle = load_task_bundle(task.id, tasks_root, eval_kwargs=task.eval_kwargs)
            _cache_bundle(task, bundle)
        except Exception:
            bundle_cache[key] = None
        return bundle_cache.get(key)

    for task in cfg.tasks:
        task_id = task.id
        if not _task_in_bench(task_id, bench):
            continue
        try:
            bundle = load_task_bundle(task_id, tasks_root, eval_kwargs=task.eval_kwargs)
            _cache_bundle(task, bundle)
            print(f"[OK] {task_id}")
            if strict:
                if not _has_trainables(bundle["param"]):
                    if task_id == "internal:non_trainable":
                        print(f"[EXPECTED] {task_id}: no_trainable_parameters")
                    else:
                        errors += 1
                        print(f"[FAIL] {task_id}: no_trainable_parameters")
        except NotImplementedError as exc:
            print(f"[SKIP] {task_id}: {exc}")
        except Exception as exc:
            errors += 1
            print(f"[FAIL] {task_id}: {exc}")

    if strict:
        jobs = expand_matrix(cfg)
        if not jobs:
            errors += 1
            print("[FAIL] matrix: no jobs expanded")
        else:
            print(f"\n[OK] matrix: {len(jobs)} jobs expanded deterministically")
            seen_trainers: set[str] = set()
            seen_tasks: set[str] = set()
            for job in jobs:
                seen_trainers.add(job.trainer_id)
                seen_tasks.add(job.task_id)
                print(f"  job {job.job_id}: {job.task_id} x {job.trainer_id} (seed={job.seed})")
            print(f"\n  tasks:    {sorted(seen_tasks)}")
            print(f"  trainers: {sorted(seen_trainers)}")
            run_id = compute_run_id(cfg.snapshot())
            artifacts = init_run_dir(cfg.runs_dir, run_id)
            manifest_jobs = []
            for job in jobs:
                bundle = _get_cached_bundle(job.task)
                status_hint = "ok"
                skip_reason = ""
                if bundle is None:
                    try:
                        bundle = load_task_bundle(job.task_id, tasks_root, eval_kwargs=job.task.eval_kwargs)
                        _cache_bundle(job.task, bundle)
                    except NotImplementedError as exc:
                        status_hint = "skipped"
                        skip_reason = str(exc)
                    except Exception as exc:
                        status_hint = "failed"
                        skip_reason = f"task_load_error: {exc}"

                manifest_jobs.append(
                    {
                        "job_id": job.job_id,
                        "task_id": job.task_id,
                        "suite": job.suite,
                        "trainer_id": job.trainer_id,
                        "seed": job.seed,
                        "raw_params": dict(job.params),
                        "resolved_trainer_kwargs": resolve_trainer_kwargs(job.params, job.trainer_id),
                        "resolved_optimizer_kwargs": merge_kwargs(
                            (bundle or {}).get("optimizer_kwargs", {}),
                            job.trainer.optimizer_kwargs or {},
                        ),
                        "resolved_guide_kwargs": merge_kwargs(
                            (bundle or {}).get("guide_kwargs"),
                            job.trainer.guide_kwargs or {},
                        ),
                        "resolved_logger_kwargs": merge_kwargs(
                            (bundle or {}).get("logger_kwargs"),
                            job.trainer.logger_kwargs or {},
                        ),
                        "eval_kwargs": dict(job.task.eval_kwargs or {}),
                        "status_hint": status_hint,
                        "skip_reason": skip_reason,
                    }
                )
            manifest = {
                "run_id": run_id,
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "jobs": manifest_jobs,
            }
            write_manifest(artifacts.manifest_json, manifest)
            print(f"[OK] manifest written: {artifacts.manifest_json}")
    return 1 if errors else 0


def cmd_run(
    config_path: str,
    root: str,
    runs_dir: str | None = None,
    max_workers: int | None = None,
    force: bool = False,
    job_timeout: float | None = None,
    resume: str | None = None,
) -> int:
    cfg = load_config(config_path)
    if runs_dir:
        cfg.runs_dir = runs_dir
    if max_workers is not None:
        cfg.max_workers = max_workers
    if resume is not None:
        cfg.resume = resume
    cfg.tasks = expand_special_tasks(cfg.tasks, root)

    # Resolve timeout: CLI flag > config YAML > mode-based default
    effective_timeout = job_timeout if job_timeout is not None else None
    if effective_timeout is None and cfg.job_timeout is None:
        effective_timeout = _default_timeout(cfg.mode)

    runner = BenchRunner(cfg, tasks_root=root, job_timeout=effective_timeout)
    runner.run(force=force)
    return 0


def cmd_ui(runs_dir: str, tasks_root: str | None = None, share: bool = False, port: int | None = None) -> int:
    return launch_ui(runs_dir, tasks_root=tasks_root, share=share, port=port)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trace-bench")
    sub = parser.add_subparsers(dest="cmd", required=True)

    list_p = sub.add_parser("list-tasks", help="List discoverable tasks")
    list_p.add_argument("--root", default="LLM4AD/benchmark_tasks")
    list_p.add_argument(
        "--bench",
        "--dataset-name",
        dest="bench",
        default=None,
        help="Bench selection: llm4ad,trace_examples,internal,veribench",
    )

    list_t = sub.add_parser("list-trainers", help="List discoverable trainers")
    list_t.add_argument("--all", action="store_true", help="Include unavailable trainers")

    val_p = sub.add_parser("validate", help="Validate tasks in config")
    val_p.add_argument("--config", required=True)
    val_p.add_argument("--root", default="LLM4AD/benchmark_tasks")
    val_p.add_argument(
        "--bench",
        "--dataset-name",
        dest="bench",
        default=None,
        help="Bench selection: llm4ad,trace_examples,internal,veribench",
    )
    val_p.add_argument("--strict", action="store_true")
    val_p.add_argument("--runs-dir", "--output-dir", dest="runs_dir", default=None)

    run_p = sub.add_parser("run", help="Run a benchmark config")
    run_p.add_argument("--config", required=True)
    run_p.add_argument("--root", default="LLM4AD/benchmark_tasks")
    run_p.add_argument("--runs-dir", "--output-dir", dest="runs_dir", default=None)
    run_p.add_argument("--max-workers", "--n-concurrent", dest="max_workers", type=int, default=None)
    run_p.add_argument("--force", action="store_true", help="Re-run all jobs even if results exist")
    run_p.add_argument("--resume", choices=["auto", "failed", "none"], default=None,
                        help="Resume mode: auto (default), failed (only re-run failed), none (fresh run)")
    run_p.add_argument("--job-timeout", dest="job_timeout", type=float, default=None, help="Per-job timeout in seconds")

    ui_p = sub.add_parser("ui", help="Launch Gradio UI")
    ui_p.add_argument("--runs-dir", default="runs")
    ui_p.add_argument("--tasks-root", default=None, help="Root directory for task discovery")
    ui_p.add_argument("--share", action="store_true", help="Enable Gradio public share link (auto-detected on Colab)")
    ui_p.add_argument("--port", type=int, default=None, help="Server port for Gradio UI")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "list-tasks":
        return cmd_list_tasks(args.root, args.bench)
    if args.cmd == "list-trainers":
        return cmd_list_trainers(args.all)
    if args.cmd == "validate":
        return cmd_validate(args.config, args.root, args.bench, args.strict, args.runs_dir)
    if args.cmd == "run":
        return cmd_run(args.config, args.root, args.runs_dir, args.max_workers, args.force, args.job_timeout, args.resume)
    if args.cmd == "ui":
        return cmd_ui(args.runs_dir, tasks_root=args.tasks_root, share=args.share, port=args.port)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
