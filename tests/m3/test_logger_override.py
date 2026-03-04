from __future__ import annotations

import types

from trace_bench.cli import _normalize_logger_override as _normalize_cli_logger_override
from trace_bench.ui.app import _normalize_logger_override as _normalize_ui_logger_override


def test_cli_logger_override_normalization():
    assert _normalize_cli_logger_override(None) is None
    assert _normalize_cli_logger_override("default") is None
    assert _normalize_cli_logger_override("config") is None
    assert _normalize_cli_logger_override("none") == "none"
    assert _normalize_cli_logger_override("off") == "none"


def test_ui_logger_override_normalization(monkeypatch):
    fake_mod = types.SimpleNamespace(ConsoleLogger=object())

    def _fake_import(_name: str):
        return fake_mod

    monkeypatch.setattr("importlib.import_module", _fake_import)

    assert _normalize_ui_logger_override(None) is None
    assert _normalize_ui_logger_override("default") is None
    assert _normalize_ui_logger_override("none") == "none"
    assert _normalize_ui_logger_override("ConsoleLogger") == "ConsoleLogger"
    assert _normalize_ui_logger_override("UnknownLogger") is None
