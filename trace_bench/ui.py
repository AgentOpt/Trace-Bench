"""
Compatibility shim.

This repo uses a `trace_bench/ui/` package for the Gradio UI implementation.
On some environments / import orders, a legacy `trace_bench/ui.py` module can
take precedence over the package directory. To avoid `ModuleNotFoundError` for
`trace_bench.ui.app`, we explicitly expose `trace_bench/ui/` as this module's
submodule search path.
"""

from __future__ import annotations

import os

# Treat this module as a package that can resolve submodules from `trace_bench/ui/`.
__path__ = [os.path.join(os.path.dirname(__file__), "ui")]

from trace_bench.ui.app import launch_ui  # noqa: E402

__all__ = ["launch_ui"]
