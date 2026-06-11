from trace_bench.resolve import resolve_trainer_kwargs


def test_resolve_trainer_kwargs_does_not_inject_gepa_defaults_for_external_trainers() -> None:
    assert resolve_trainer_kwargs({}, "TextGradTrainer") == {}
    assert resolve_trainer_kwargs({"iterations": 3}, "OpenEvolveTrainer") == {
        "iterations": 3
    }


def test_resolve_trainer_kwargs_preserves_gepa_defaults() -> None:
    resolved = resolve_trainer_kwargs({}, "GEPA-UCB")
    assert resolved["num_search_iterations"] == 1
    assert resolved["train_batch_size"] == 2
    assert resolved["merge_every"] == 2
    assert resolved["pareto_subset_size"] == 2
