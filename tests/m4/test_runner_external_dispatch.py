import types
import sys

from tests.m4._helpers import make_bundle
from trace_bench.config import TrainerConfig
from trace_bench.runner import _train_bundle


class _FakeMIPROv2:
    def __init__(self, metric=None, **kwargs):
        self.metric = metric

    def compile(self, student, trainset=None, **kwargs):
        student.greeting = "Hello"
        return student


class _FakeBootstrapFewShot(_FakeMIPROv2):
    pass


def test_runner_dispatches_external_trainer(monkeypatch):
    fake = types.SimpleNamespace(MIPROv2=_FakeMIPROv2, BootstrapFewShot=_FakeBootstrapFewShot)
    monkeypatch.setitem(sys.modules, "dspy", fake)
    bundle = make_bundle()
    trainer = TrainerConfig(id="DSPy-MIPROv2")
    result = _train_bundle(bundle, trainer, {"num_trials": 1}, mode="stub")
    assert result["status"] == "ok"
    assert result["resolved_optimizer"] == "DSPy.DSPy-MIPROv2"
