from trace_bench.cli import _validate_trainer_params
from trace_bench.config import TrainerConfig


class _FakeExternalTrainer:
    USES_TRACE_OPTIMIZER = False

    def train(
        self,
        guide,
        train_dataset,
        *,
        iterations: int = 1,
        ensure_improvement: bool = True,
        verbose: bool = False,
        **_kwargs,
    ):
        return {}


def test_validate_trainer_params_uses_train_signature(monkeypatch) -> None:
    monkeypatch.setattr("trace_bench.cli._resolve_algorithm", lambda _trainer_id: _FakeExternalTrainer)
    trainer = TrainerConfig(
        id="OpenEvolveTrainer",
        params_variants=[{"iterations": 2, "ensure_improvement": False}],
    )
    errors = []
    _validate_trainer_params(trainer, errors)
    assert errors == []


def test_validate_trainer_params_rejects_unknown_kwarg(monkeypatch) -> None:
    monkeypatch.setattr("trace_bench.cli._resolve_algorithm", lambda _trainer_id: _FakeExternalTrainer)
    trainer = TrainerConfig(id="OpenEvolveTrainer", params_variants=[{"unknown": 1}])
    errors = []
    _validate_trainer_params(trainer, errors)
    assert errors == ["unknown trainer kwarg 'unknown' for OpenEvolveTrainer"]
