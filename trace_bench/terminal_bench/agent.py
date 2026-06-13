"""Harbor agent wrapper(s) used by Trace-Bench.

The main purpose is to make Terminus-2 prompt templates overridable from Trace-Bench.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

try:  # Harbor is optional; only required for real Terminal-Bench runs.
    from harbor.agents.terminus_2.terminus_2 import Terminus2
except Exception as _exc:  # pragma: no cover
    Terminus2 = None  # type: ignore
    _HARBOR_IMPORT_ERROR: Exception = _exc


if Terminus2 is None:  # pragma: no cover

    class TraceBenchTerminus2:  # type: ignore
        """Placeholder when Harbor is not installed."""

        def __init__(self, *_: Any, **__: Any) -> None:
            raise ImportError(
                "Harbor is required to use TraceBenchTerminus2. "
                "Install harbor and retry. Original import error: "
                f"{_HARBOR_IMPORT_ERROR}"
            )

else:

    class TraceBenchTerminus2(Terminus2):
        """Terminus2 with an overridable prompt template."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            prompt_template_path: Optional[str] = kwargs.pop(
                "prompt_template_path", None
            )
            prompt_template: Optional[str] = kwargs.pop("prompt_template", None)
            super().__init__(*args, **kwargs)

            if prompt_template_path:
                self._prompt_template = Path(prompt_template_path).read_text(
                    encoding="utf-8"
                )
            elif prompt_template is not None:
                self._prompt_template = str(prompt_template)


__all__ = ["TraceBenchTerminus2"]
