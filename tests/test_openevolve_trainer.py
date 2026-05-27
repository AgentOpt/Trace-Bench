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


def _import_openevolve_trainer(monkeypatch, best_code: str):
    fake_module = types.ModuleType("openevolve")

    def _run_evolution(*, initial_program, evaluator, iterations, **_kwargs):
        del initial_program, evaluator, iterations
        return types.SimpleNamespace(best_code=best_code)

    fake_module.run_evolution = _run_evolution
    monkeypatch.setitem(sys.modules, "openevolve", fake_module)
    sys.modules.pop("trace_bench.trainers.openevolve_trainer", None)
    return importlib.import_module("trace_bench.trainers.openevolve_trainer")


def test_openevolve_trainer_updates_parameter(monkeypatch) -> None:
    trainer_module = _import_openevolve_trainer(
        monkeypatch,
        best_code='candidate = {"greeting": "Hello"}\n',
    )
    trainer = trainer_module.OpenEvolveTrainer(_DummyAgent("Hi"))
    result = trainer.train(
        guide=_DummyGuide(),
        train_dataset={"inputs": ["Hello Sam"], "infos": ["Hello, Sam!"]},
        mode="real",
        iterations=1,
        ensure_improvement=False,
    )
    assert result["status"] == "ok"
    assert result["resolved_optimizer"] == "openevolve.run_evolution"
    assert trainer.param.greeting.data == "Hello"


def test_openevolve_trainer_rejects_worse_candidate(monkeypatch) -> None:
    trainer_module = _import_openevolve_trainer(
        monkeypatch,
        best_code='candidate = {"greeting": "Bad"}\n',
    )
    trainer = trainer_module.OpenEvolveTrainer(_DummyAgent("Hello"))
    trainer.train(
        guide=_DummyGuide(),
        train_dataset={"inputs": ["Hello Sam"], "infos": ["Hello, Sam!"]},
        mode="real",
        iterations=1,
        ensure_improvement=True,
    )
    assert trainer.param.greeting.data == "Hello"


def test_openevolve_trainer_rejects_invalid_candidate_program(monkeypatch) -> None:
    trainer_module = _import_openevolve_trainer(
        monkeypatch,
        best_code='print("bad candidate")\n',
    )
    trainer = trainer_module.OpenEvolveTrainer(_DummyAgent("Hello"))
    with pytest.raises(ValueError, match="Candidate program"):
        trainer.train(
            guide=_DummyGuide(),
            train_dataset={"inputs": ["Hello Sam"], "infos": ["Hello, Sam!"]},
            mode="real",
            iterations=1,
            ensure_improvement=False,
        )


def test_openevolve_trainer_requires_trainable_parameters(monkeypatch) -> None:
    trainer_module = _import_openevolve_trainer(
        monkeypatch,
        best_code='candidate = {"greeting": "Hello"}\n',
    )

    class _NoTrainables:
        def parameters(self):
            return []

    trainer = trainer_module.OpenEvolveTrainer(_NoTrainables())
    with pytest.raises(ValueError, match="no trainable parameters"):
        trainer.train(
            guide=_DummyGuide(),
            train_dataset={"inputs": ["Hello Sam"], "infos": ["Hello, Sam!"]},
            mode="real",
            iterations=1,
        )
