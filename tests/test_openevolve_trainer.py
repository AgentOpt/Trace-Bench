import asyncio
import importlib
import os
import sys
import tempfile
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


def _install_fake_openevolve_config(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_config_module = types.ModuleType("openevolve.config")

    class _FakeDatabaseConfig:
        def __init__(self) -> None:
            self.population_size = 1000
            self.num_islands = 5

    class _FakeLLMConfig:
        def __init__(self) -> None:
            self.api_base = "https://api.openai.com/v1"
            self.api_key = None
            self.max_tokens = 4096
            self.temperature = 0.7
            self.timeout = 60
            self.retries = 3
            self.retry_delay = 5
            self.models = []
            self.evaluator_models = []

    class _FakeConfig:
        def __init__(self, max_iterations: int, random_seed: int | None) -> None:
            self.max_iterations = max_iterations
            self.random_seed = random_seed
            self.database = _FakeDatabaseConfig()
            self.llm = _FakeLLMConfig()

    class _FakeLLMModelConfig:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    fake_config_module.Config = _FakeConfig
    fake_config_module.LLMModelConfig = _FakeLLMModelConfig
    monkeypatch.setitem(sys.modules, "openevolve.config", fake_config_module)


def _import_openevolve_trainer(monkeypatch: pytest.MonkeyPatch, best_code: str, capture: dict[str, object] | None = None) -> types.ModuleType:
    fake_module = types.ModuleType("openevolve")

    def _run_evolution(*, initial_program, evaluator, iterations, config=None, **_kwargs):
        if capture is not None:
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
                handle.write(best_code)
                candidate_path = handle.name
            try:
                capture["evaluation"] = evaluator(candidate_path)
            finally:
                os.unlink(candidate_path)
            capture["config"] = config
            capture["iterations"] = iterations
        del initial_program
        return types.SimpleNamespace(best_code=best_code)

    fake_module.run_evolution = _run_evolution
    monkeypatch.setitem(sys.modules, "openevolve", fake_module)
    _install_fake_openevolve_config(monkeypatch)
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


def test_openevolve_trainer_runs_inside_active_event_loop(monkeypatch) -> None:
    trainer_module = _import_openevolve_trainer(
        monkeypatch,
        best_code='candidate = {"greeting": "Hello"}\n',
    )

    async def _run_training() -> dict[str, object]:
        trainer = trainer_module.OpenEvolveTrainer(_DummyAgent("Hi"))
        return trainer.train(
            guide=_DummyGuide(),
            train_dataset={"inputs": ["Hello Sam"], "infos": ["Hello, Sam!"]},
            mode="real",
            iterations=1,
            ensure_improvement=False,
        )

    result = asyncio.run(_run_training())
    assert result["status"] == "ok"


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


def test_openevolve_trainer_returns_combined_score_and_configures_population(monkeypatch) -> None:
    capture: dict = {}
    trainer_module = _import_openevolve_trainer(
        monkeypatch,
        best_code='candidate = {"greeting": "Hello"}\n',
        capture=capture,
    )
    trainer = trainer_module.OpenEvolveTrainer(_DummyAgent("Hi"))
    trainer.train(
        guide=_DummyGuide(),
        train_dataset={"inputs": ["Hello Sam"], "infos": ["Hello, Sam!"]},
        mode="real",
        iterations=2,
        population_size=12,
        num_islands=3,
        model="openrouter/openai/gpt-4o-mini",
        api_key="test-key",
        ensure_improvement=False,
    )
    assert capture["evaluation"]["combined_score"] == capture["evaluation"]["score"]
    assert capture["config"].database.population_size == 12
    assert capture["config"].database.num_islands == 3
    assert capture["config"].llm.models[0].kwargs["name"] == "openai/gpt-4o-mini"
