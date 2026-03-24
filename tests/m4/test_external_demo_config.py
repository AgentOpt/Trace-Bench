from trace_bench.config import load_config


def test_external_demo_config_parses():
    cfg = load_config("configs/m4_external_optimizers_demo.yaml")
    trainer_ids = [t.id for t in cfg.trainers]
    assert trainer_ids == [
        "DSPy-MIPROv2",
        "DSPy-BootstrapFewShot",
        "TextGrad-TGD",
        "OpenEvolve",
        "GEPA-Base",
    ]
    assert cfg.tasks[0].id == "trace_examples:greeting_stub"


def test_trainer_comparison_rows_shape():
    from trace_bench.results import build_trainer_comparison_rows
    rows = [
        {"task_id": "t1", "suite": "s", "job_id": "j1", "trainer_id": "A", "status": "ok", "score_best": 0.5, "time_seconds": 1.0},
        {"task_id": "t1", "suite": "s", "job_id": "j2", "trainer_id": "B", "status": "ok", "score_best": 0.9, "time_seconds": 2.0},
        {"task_id": "t1", "suite": "s", "job_id": "j3", "trainer_id": "C", "status": "failed", "score_best": None, "time_seconds": 3.0},
    ]
    out = build_trainer_comparison_rows(rows)
    assert [r["trainer_id"] for r in out] == ["B", "A"]
