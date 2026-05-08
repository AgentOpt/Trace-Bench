"""Terminal-Bench integration helpers.

This module intentionally keeps hard dependencies optional:
- The Harbor CLI ("harbor") is required only when running Terminal-Bench tasks.
- The Harbor Python package is required only when using the custom Terminus-2 wrapper.

Trace-Bench exposes Terminal-Bench tasks under the task-id namespace:
  terminal_bench:<task_slug>

Where <task_slug> is the Terminal-Bench 2.x task slug (e.g. "regex-log").
"""

from __future__ import annotations

from typing import List

DEFAULT_HARBOR_DATASET = "terminal-bench@2.0"

SAMPLE_TASK_SLUGS: List[str] = [
    "regex-log",
    "log-summary-date-ranges",
    "multi-source-data-merger",
    "financial-document-processor",
]


def sample_task_ids() -> List[str]:
    return [f"terminal_bench:{slug}" for slug in SAMPLE_TASK_SLUGS]


__all__ = [
    "DEFAULT_HARBOR_DATASET",
    "SAMPLE_TASK_SLUGS",
    "sample_task_ids",
]
