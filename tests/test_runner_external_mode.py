from trace_bench.config import TrainerConfig
from trace_bench.runner import _train_bundle


class _DummyAgent:
    def parameters(self):
        return []

    def __call__(self, query):
        return query


class _DummyGuide:
    def __call__(self, task_input, response, task_info):
        return 1.0, "ok"


class _FakeExternalTrainer:
    USES_TRACE_OPTIMIZER = False

    def __init__(self, agent, logger=None):
        del logger
        self.param = agent

    def train(self, guide, train_dataset, mode="real", **_kwargs):
        del guide, train_dataset
        return {"status": "ok", "resolved_optimizer": mode}


def test_runner_passes_mode_to_external_trainers(monkeypatch) -> None:
    monkeypatch.setattr("trace_bench.runner._resolve_algorithm", lambda _name: _FakeExternalTrainer)
    bundle = {
        "param": _DummyAgent(),
        "guide": _DummyGuide(),
        "train_dataset": {"inputs": ["x"], "infos": ["y"]},
        "optimizer_kwargs": {},
        "metadata": {},
    }
    trainer = TrainerConfig(id="FakeExternalTrainer", logger="none")
    result = _train_bundle(bundle=bundle, trainer_spec=trainer, params={}, mode="stub")
    assert result["status"] == "ok"
    assert result["resolved_optimizer"] == "stub"


class _FakeNonDSPyExternalTrainer(_FakeExternalTrainer):
    FRAMEWORK = "trace"

    def train(self, guide, train_dataset, mode="real", **kwargs):
        del guide, train_dataset, mode
        return {"status": "ok", "resolved_optimizer": kwargs.get("dspy_lm", "absent")}


class _FakeDSPyExternalTrainer(_FakeExternalTrainer):
    FRAMEWORK = "dspy"

    def train(self, guide, train_dataset, mode="real", **kwargs):
        del guide, train_dataset, mode
        return {"status": "ok", "resolved_optimizer": kwargs.get("dspy_lm", "absent")}


def test_runner_does_not_inject_dspy_stub_into_non_dspy_external_trainers(monkeypatch) -> None:
    monkeypatch.setattr("trace_bench.runner._resolve_algorithm", lambda _name: _FakeNonDSPyExternalTrainer)
    bundle = {"param": _DummyAgent(), "guide": _DummyGuide(), "train_dataset": {"inputs": ["x"], "infos": ["y"]}, "optimizer_kwargs": {}, "metadata": {}}
    trainer = TrainerConfig(id="FakeNonDSPyExternalTrainer", logger="none")
    result = _train_bundle(bundle=bundle, trainer_spec=trainer, params={}, mode="stub")
    assert result["status"] == "ok"
    assert result["resolved_optimizer"] == "absent"


def test_runner_injects_dspy_stub_only_for_dspy_external_trainers(monkeypatch) -> None:
    monkeypatch.setattr("trace_bench.runner._resolve_algorithm", lambda _name: _FakeDSPyExternalTrainer)
    bundle = {"param": _DummyAgent(), "guide": _DummyGuide(), "train_dataset": {"inputs": ["x"], "infos": ["y"]}, "optimizer_kwargs": {}, "metadata": {}}
    trainer = TrainerConfig(id="FakeDSPyExternalTrainer", logger="none")
    result = _train_bundle(bundle=bundle, trainer_spec=trainer, params={}, mode="stub")
    assert result["status"] == "ok"
    assert result["resolved_optimizer"] == "stub"
