from __future__ import annotations

from trace_bench.config import TrainerConfig
from trace_bench.registry import discover_trainers
from trace_bench.runner import _train_bundle


def test_discover_trainers_marks_priority_search_examples_unavailable(monkeypatch):
    monkeypatch.setattr("trace_bench.registry.priority_search_example_trainers_supported", lambda: False)

    specs = {spec.id: spec for spec in discover_trainers()}

    assert specs["SequentialUpdate"].available is False
    assert specs["SequentialSearch"].available is False
    assert specs["BeamSearch"].available is False


def test_train_bundle_fails_fast_for_unfixed_priority_search_examples(monkeypatch):
    monkeypatch.setattr("trace_bench.runner.priority_search_example_trainers_supported", lambda: False)

    result = _train_bundle(
        bundle={
            "param": object(),
            "guide": object(),
            "train_dataset": {"inputs": [1], "infos": [1]},
            "optimizer_kwargs": {},
            "guide_kwargs": {},
            "logger_kwargs": {},
            "metadata": {},
        },
        trainer_spec=TrainerConfig(id="SequentialUpdate", params_variants=[{}]),
        params={},
        mode="stub",
    )

    assert result["status"] == "failed"
    assert "SequentialUpdate" in result["error"]
    assert "search_template.py" in result["error"]
