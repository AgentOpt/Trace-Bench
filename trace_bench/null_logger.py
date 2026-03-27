from __future__ import annotations

from typing import Any

try:
    from opto.trainer.loggers import BaseLogger as _BaseLogger
except Exception:  # pragma: no cover - only when opto is unavailable
    class _BaseLogger:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass


class NullLogger(_BaseLogger):
    """Trace-Bench local no-op logger used for logger='none' overrides."""

    def log(self, name: str, data: Any, step: Any, **kwargs: Any) -> None:  # noqa: ARG002
        return None


__all__ = ["NullLogger"]
