import types
import sys

from trace_bench.cli import _validate_trainer_params
from trace_bench.config import TrainerConfig
from trace_bench.registry import discover_trainers


def test_validate_allows_external_kwargs(monkeypatch):
    monkeypatch.setitem(sys.modules, "dspy", types.SimpleNamespace(MIPROv2=object, BootstrapFewShot=object))
    monkeypatch.setitem(sys.modules, "textgrad", types.SimpleNamespace(TGD=object))
    monkeypatch.setitem(sys.modules, "openevolve", types.SimpleNamespace(run_evolution=lambda **k: None))

    ids = {spec.id for spec in discover_trainers() if spec.available}
    assert "DSPy-MIPROv2" in ids
    assert "TextGrad-TGD" in ids
    assert "OpenEvolve" in ids

    errors = []
    _validate_trainer_params(TrainerConfig(id="DSPy-MIPROv2", params_variants=[{"num_trials": 1}]), errors)
    _validate_trainer_params(TrainerConfig(id="TextGrad-TGD", params_variants=[{"num_steps": 1}]), errors)
    _validate_trainer_params(TrainerConfig(id="OpenEvolve", params_variants=[{"iterations": 1}]), errors)
    assert errors == []

def test_validate_external_optimizer_kwargs_are_not_flagged(tmp_path, capsys):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        """
mode: stub
tasks:
  - id: trace_examples:greeting_stub
trainers:
  - id: DSPy-MIPROv2
    optimizer_kwargs:
      auto: light
      max_bootstrapped_demos: 1
      max_labeled_demos: 1
    params_variants:
      - num_trials: 1
""",
        encoding="utf-8",
    )
    from trace_bench.cli import cmd_validate
    rc = cmd_validate(str(cfg), "LLM4AD/benchmark_tasks", strict=True)
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "unknown trainer kwarg 'optimizer_kwargs'" not in out
