from trace_bench.config import load_config


def test_load_config_smoke():
    cfg = load_config("configs/smoke.yaml")
    assert cfg.mode == "stub"
    assert cfg.tasks[0].id == "internal:numeric_param"
    assert cfg.runs_dir == "runs"
