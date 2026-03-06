from __future__ import annotations

import types


def test_resolve_trainer_base_prefers_trainer(monkeypatch):
    from trace_bench import registry

    class Trainer:
        pass

    module = types.SimpleNamespace(Trainer=Trainer)

    monkeypatch.setattr(registry, "ensure_opto_importable", lambda: None)
    monkeypatch.setattr(
        registry.importlib,
        "import_module",
        lambda name: module if name == "opto.trainer.algorithms.algorithm" else (_ for _ in ()).throw(ImportError()),
    )

    resolved = registry._resolve_trainer_base_class()
    assert resolved is Trainer


def test_resolve_trainer_base_falls_back_to_abstract(monkeypatch):
    from trace_bench import registry

    class AbstractAlgorithm:
        pass

    module = types.SimpleNamespace(AbstractAlgorithm=AbstractAlgorithm)

    monkeypatch.setattr(registry, "ensure_opto_importable", lambda: None)
    monkeypatch.setattr(
        registry.importlib,
        "import_module",
        lambda name: module if name == "opto.trainer.algorithms.algorithm" else (_ for _ in ()).throw(ImportError()),
    )

    resolved = registry._resolve_trainer_base_class()
    assert resolved is AbstractAlgorithm

