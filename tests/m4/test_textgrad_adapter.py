import types
import sys

from tests.m4._helpers import make_bundle
from trace_bench.config import TrainerConfig
from trace_bench.integrations.external_optimizers import run_textgrad_trainer


class _Variable:
    def __init__(self, value, role_description=None, requires_grad=True):
        self.value = value


class _TextLoss:
    def __init__(self, objective, engine=None):
        self.objective = objective
        self.engine = engine

    def __call__(self, variable):
        class _Loss:
            def __init__(self, variable):
                self.variable = variable

            def backward(self):
                return None
        return _Loss(variable)


class _TGD:
    def __init__(self, parameters=None, **kwargs):
        self.parameters = parameters or []
        self.kwargs = kwargs

    def step(self):
        for p in self.parameters:
            p.value = "Hello"


def test_textgrad_adapter(monkeypatch):
    calls = {}
    def _set_backward_engine(engine, override=False):
        calls["engine"] = engine
        calls["override"] = override
    fake = types.SimpleNamespace(Variable=_Variable, TextLoss=_TextLoss, TGD=_TGD, set_backward_engine=_set_backward_engine)
    monkeypatch.setitem(sys.modules, "textgrad", fake)
    bundle = make_bundle()
    trainer = TrainerConfig(id="TextGrad-TGD", optimizer_kwargs={"engine": "test-engine"})
    result = run_textgrad_trainer(bundle, trainer, {"num_steps": 1}, mode="stub")
    assert result["status"] == "ok"
    assert result["resolved_optimizer"] == "TextGrad.TGD"
    assert bundle["param"].data == "Hello"
    assert calls["engine"] == "test-engine"
