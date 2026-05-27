import importlib
import sys
import types

from trace_bench.registry import discover_trainers


def _install_fake_external_dependencies(monkeypatch) -> None:
    fake_textgrad_module = types.ModuleType("opto.optimizers.textgrad")

    class _FakeTextGrad:
        def __init__(self, parameters, **_kwargs) -> None:
            self.parameters = list(parameters)

    fake_textgrad_module.TextGrad = _FakeTextGrad
    monkeypatch.setitem(sys.modules, "opto.optimizers.textgrad", fake_textgrad_module)

    fake_openevolve_module = types.ModuleType("openevolve")
    fake_openevolve_module.run_evolution = lambda **_kwargs: {"best_code": 'candidate = {}'}
    monkeypatch.setitem(sys.modules, "openevolve", fake_openevolve_module)


def test_discover_trainers_lists_new_external_trainers_when_dependencies_are_available(monkeypatch) -> None:
    _install_fake_external_dependencies(monkeypatch)

    import trace_bench.trainers.textgrad_trainer as textgrad_trainer
    import trace_bench.trainers.openevolve_trainer as openevolve_trainer

    importlib.reload(textgrad_trainer)
    importlib.reload(openevolve_trainer)

    specs = {spec.id: spec for spec in discover_trainers()}
    assert specs["TextGradTrainer"].available is True
    assert specs["OpenEvolveTrainer"].available is True
