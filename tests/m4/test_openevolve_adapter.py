import types
import sys
import tempfile
from pathlib import Path

from tests.m4._helpers import make_bundle
from trace_bench.config import TrainerConfig
from trace_bench.integrations.external_optimizers import run_openevolve_trainer


class _Result:
    def __init__(self, best_code):
        self.best_code = best_code


def _run_evolution(initial_program, evaluator, iterations=1):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "candidate.py"
        p.write_text("Hello", encoding="utf-8")
        evaluator(str(p))
    return _Result(best_code="Hello")


def test_openevolve_adapter(monkeypatch):
    fake = types.SimpleNamespace(run_evolution=_run_evolution)
    monkeypatch.setitem(sys.modules, "openevolve", fake)
    bundle = make_bundle()
    trainer = TrainerConfig(id="OpenEvolve", optimizer_kwargs={"include_artifacts": True})
    result = run_openevolve_trainer(bundle, trainer, {"iterations": 1}, mode="stub")
    assert result["status"] == "ok"
    assert result["resolved_optimizer"] == "OpenEvolve.run_evolution"
    assert bundle["param"].data == "Hello"


def test_openevolve_filters_unsupported_kwargs(monkeypatch):
    fake = types.SimpleNamespace(run_evolution=_run_evolution)
    monkeypatch.setitem(sys.modules, "openevolve", fake)
    bundle = make_bundle()
    trainer = TrainerConfig(id="OpenEvolve", optimizer_kwargs={"include_artifacts": True})
    result = run_openevolve_trainer(bundle, trainer, {"iterations": 1, "max_artifact_bytes": 123}, mode="stub")
    assert result["status"] == "ok"
