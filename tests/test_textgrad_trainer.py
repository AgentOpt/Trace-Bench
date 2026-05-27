import importlib
import sys
import types

import pytest


class _DummyParam:
    def __init__(self, name: str, value: str) -> None:
        self.name = name
        self.py_name = name
        self.data = value
        self.trainable = True


class _DummyAgent:
    def __init__(self, greeting: str = "Hi") -> None:
        self.greeting = _DummyParam("greeting", greeting)

    def parameters(self):
        return [self.greeting]

    def __call__(self, query: str) -> str:
        name = query.split()[-1].strip("!.?")
        return f"{self.greeting.data}, {name}!"


class _DummyGuide:
    def __call__(self, task_input: str, response: str, task_info: str):
        del task_input
        return (1.0 if response == task_info else 0.0), f"expected {task_info}"


def _import_textgrad_trainer(monkeypatch, proposal: str):
    fake_module = types.ModuleType("opto.optimizers.textgrad")

    class _FakeTextGrad:
        def __init__(self, parameters, **_kwargs) -> None:
            self.parameters = list(parameters)

        def zero_feedback(self) -> None:
            return None

        def backward(self, target, feedback) -> None:
            del target, feedback
            return None

        def step(self, bypassing=False, verbose=False):
            del bypassing, verbose
            return {self.parameters[0]: proposal}

    fake_module.TextGrad = _FakeTextGrad
    monkeypatch.setitem(sys.modules, "opto.optimizers.textgrad", fake_module)
    sys.modules.pop("trace_bench.trainers.textgrad_trainer", None)
    return importlib.import_module("trace_bench.trainers.textgrad_trainer")


def test_textgrad_trainer_updates_parameter(monkeypatch) -> None:
    trainer_module = _import_textgrad_trainer(monkeypatch, proposal="Hello")
    trainer = trainer_module.TextGradTrainer(_DummyAgent("Hi"))
    result = trainer.train(
        guide=_DummyGuide(),
        train_dataset={"inputs": ["Hello Sam"], "infos": ["Hello, Sam!"]},
        mode="real",
        num_epochs=1,
        batch_size=1,
        ensure_improvement=False,
    )
    assert result["status"] == "ok"
    assert result["resolved_optimizer"] == "opto.optimizers.textgrad.TextGrad"
    assert trainer.param.greeting.data == "Hello"


def test_textgrad_trainer_rejects_worse_candidate(monkeypatch) -> None:
    trainer_module = _import_textgrad_trainer(monkeypatch, proposal="Bad")
    trainer = trainer_module.TextGradTrainer(_DummyAgent("Hello"))
    trainer.train(
        guide=_DummyGuide(),
        train_dataset={"inputs": ["Hello Sam"], "infos": ["Hello, Sam!"]},
        mode="real",
        num_epochs=1,
        batch_size=1,
        ensure_improvement=True,
    )
    assert trainer.param.greeting.data == "Hello"


def test_textgrad_trainer_requires_trainable_parameters(monkeypatch) -> None:
    trainer_module = _import_textgrad_trainer(monkeypatch, proposal="Hello")

    class _NoTrainables:
        def parameters(self):
            return []

    trainer = trainer_module.TextGradTrainer(_NoTrainables())
    with pytest.raises(ValueError, match="no trainable parameters"):
        trainer.train(
            guide=_DummyGuide(),
            train_dataset={"inputs": ["Hello Sam"], "infos": ["Hello, Sam!"]},
            mode="real",
        )
