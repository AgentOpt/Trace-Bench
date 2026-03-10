"""Backward-compatible task helpers. Use trace_bench.registry instead."""

from .registry import discover_tasks, load_task_bundle, load_task_module, TaskSpec

__all__ = ["discover_tasks", "load_task_bundle", "load_task_module", "TaskSpec"]
