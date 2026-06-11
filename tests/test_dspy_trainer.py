import itertools
from typing import Any

import pytest


def test_dspy_trainer_restores_empty_global_lm(monkeypatch: pytest.MonkeyPatch) -> None:
    dspy = pytest.importorskip("dspy")
    trainer_module = pytest.importorskip("trace_bench.trainers.dspy_trainer")
    from dspy.utils import DummyLM

    previous_lm = getattr(dspy.settings, "lm", None)
    dspy.configure(lm=None)
    trainer = trainer_module.DSPyTrainer(object())

    def _train_inner(**_kwargs: Any) -> dict[str, str]:
        return {"status": "ok"}

    monkeypatch.setattr(trainer, "_train_inner", _train_inner)
    try:
        trainer.train(
            guide=object(),
            train_dataset={"inputs": [], "infos": []},
            dspy_lm=DummyLM(itertools.cycle([{"answer": "ok"}])),
        )
        assert getattr(dspy.settings, "lm", None) is None
    finally:
        dspy.configure(lm=previous_lm)
