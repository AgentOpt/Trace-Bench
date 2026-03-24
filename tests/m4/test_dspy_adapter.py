import types
import sys

from tests.m4._helpers import make_bundle
from trace_bench.config import TrainerConfig
from trace_bench.integrations.external_optimizers import run_dspy_trainer


class _FakeMIPROv2:
    def __init__(self, metric=None, **kwargs):
        self.metric = metric
        self.kwargs = kwargs

    def compile(self, student, trainset=None, **kwargs):
        student.greeting = "Hello"
        return student


class _FakeBootstrapFewShot(_FakeMIPROv2):
    pass


class _FakeBootstrapStrict:
    def __init__(self, metric=None, **kwargs):
        self.metric = metric
        self.kwargs = kwargs

    def compile(self, student, trainset=None):
        student.greeting = "Hello"
        return student


def test_dspy_mipro_adapter(monkeypatch):
    fake = types.SimpleNamespace(MIPROv2=_FakeMIPROv2, BootstrapFewShot=_FakeBootstrapFewShot)
    monkeypatch.setitem(sys.modules, "dspy", fake)
    bundle = make_bundle()
    trainer = TrainerConfig(id="DSPy-MIPROv2", optimizer_kwargs={"auto": "light"})
    result = run_dspy_trainer(bundle, trainer, {"num_trials": 1}, mode="stub")
    assert result["status"] == "ok"
    assert result["resolved_optimizer"] == "DSPy.DSPy-MIPROv2"
    assert bundle["param"].data == "Hello"


def test_dspy_fallback_to_teleprompt(monkeypatch):
    teleprompt = types.SimpleNamespace(MIPROv2=_FakeMIPROv2, BootstrapFewShot=_FakeBootstrapFewShot)
    fake = types.SimpleNamespace(teleprompt=teleprompt)
    monkeypatch.setitem(sys.modules, "dspy", fake)
    bundle = make_bundle()
    trainer = TrainerConfig(id="DSPy-MIPROv2", optimizer_kwargs={"auto": "light"})
    result = run_dspy_trainer(bundle, trainer, {"num_trials": 1}, mode="stub")
    assert result["status"] == "ok"
    assert bundle["param"].data == "Hello"


def test_dspy_fallback_to_mipro_alias(monkeypatch):
    teleprompt = types.SimpleNamespace(MIPRO=_FakeMIPROv2, BootstrapFewShot=_FakeBootstrapFewShot)
    fake = types.SimpleNamespace(teleprompt=teleprompt)
    monkeypatch.setitem(sys.modules, "dspy", fake)
    bundle = make_bundle()
    trainer = TrainerConfig(id="DSPy-MIPROv2", optimizer_kwargs={"auto": "light"})
    result = run_dspy_trainer(bundle, trainer, {"num_trials": 1}, mode="stub")
    assert result["status"] == "ok"
    assert bundle["param"].data == "Hello"


def test_dspy_compile_filters_optimizer_kwargs(monkeypatch):
    fake = types.SimpleNamespace(BootstrapFewShot=_FakeBootstrapStrict)
    monkeypatch.setitem(sys.modules, "dspy", fake)
    bundle = make_bundle()
    trainer = TrainerConfig(id="DSPy-BootstrapFewShot", optimizer_kwargs={"max_rounds": 1})
    result = run_dspy_trainer(bundle, trainer, {"optimizer_kwargs": {"ignored": True}}, mode="stub")
    assert result["status"] == "ok"
    assert bundle["param"].data == "Hello"
