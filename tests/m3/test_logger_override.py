from __future__ import annotations

import types

from trace_bench.cli import _normalize_logger_override as _normalize_cli_logger_override
from trace_bench.null_logger import NullLogger
from trace_bench.runner import _build_logger, _train_with_local_logger_fallback
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


def test_build_logger_uses_local_null_logger():
    logger_obj, resolved = _build_logger("none", {})
    assert isinstance(logger_obj, NullLogger)
    assert resolved == "trace_bench.null_logger.NullLogger"


def test_train_with_local_logger_fallback_sets_logger(monkeypatch):
    class _FakeTrainer:
        def __init__(self, model, optimizer):
            self.model = model
            self.optimizer = optimizer
            self.logger = None

        def train(self, *, guide, train_dataset, **kwargs):
            return {
                "guide": guide,
                "train_dataset": train_dataset,
                "logger_type": type(self.logger).__name__,
                "kwargs": kwargs,
            }

    fake_train_mod = types.SimpleNamespace(
        load_trainer_class=lambda _algorithm: _FakeTrainer,
        load_optimizer=lambda _optimizer, _model, **_kwargs: "OPTIMIZER",
        load_guide=lambda guide, **_kwargs: guide,
    )

    real_import = __import__("importlib").import_module

    def _fake_import(name: str):
        if name == "opto.trainer.train":
            return fake_train_mod
        return real_import(name)

    monkeypatch.setattr("importlib.import_module", _fake_import)

    result = _train_with_local_logger_fallback(
        bundle={"param": "MODEL", "train_dataset": {"inputs": [1], "infos": [1]}},
        algorithm="AnyTrainer",
        optimizer="AnyOptimizer",
        guide="GUIDE",
        logger_obj=NullLogger(),
        optimizer_kwargs={},
        guide_kwargs={},
        trainer_kwargs={"batch_size": 1},
    )

    assert result["guide"] == "GUIDE"
    assert result["logger_type"] == "NullLogger"
    assert result["kwargs"] == {"batch_size": 1}
