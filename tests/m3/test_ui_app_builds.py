"""Test that the Gradio UI app can be constructed without launching a server."""
from __future__ import annotations

import pytest


def test_launch_ui_function_importable():
    """launch_ui is importable from trace_bench.ui."""
    from trace_bench.ui import launch_ui
    assert callable(launch_ui)


def test_launch_ui_returns_1_without_gradio(monkeypatch):
    """Without gradio installed, launch_ui returns 1 (not crash)."""
    import trace_bench.ui.app as app_mod

    # Force gradio import to fail
    monkeypatch.setattr(app_mod, "_import_gradio", lambda: None)
    result = app_mod.launch_ui("runs")
    assert result == 1


def test_list_configs_empty(tmp_path):
    """_list_configs on non-existent dir returns empty list."""
    from trace_bench.ui.app import _list_configs
    assert _list_configs(str(tmp_path / "no_such_dir")) == []


def test_list_configs_finds_yaml(tmp_path):
    """_list_configs finds YAML files in a directory."""
    from trace_bench.ui.app import _list_configs
    (tmp_path / "a.yaml").write_text("mode: stub\n", encoding="utf-8")
    (tmp_path / "b.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "c.txt").write_text("ignore\n", encoding="utf-8")
    found = _list_configs(str(tmp_path))
    assert len(found) == 2  # a.yaml + b.json
    assert any("a.yaml" in f for f in found)
    assert any("b.json" in f for f in found)


def test_discover_tasks_safe_handles_error():
    """_discover_tasks_safe returns error string on failure."""
    from trace_bench.ui.app import _discover_tasks_safe
    result = _discover_tasks_safe("/nonexistent/path", "llm4ad")
    # Should return a string (either task list or error), not crash
    assert isinstance(result, str)


def test_discover_trainers_safe_returns_string():
    """_discover_trainers_safe returns a string."""
    from trace_bench.ui.app import _discover_trainers_safe
    result = _discover_trainers_safe()
    assert isinstance(result, str)


def test_discover_tasks_for_ui_no_preselection(monkeypatch):
    """Task discovery should not preselect any values."""
    import trace_bench.ui.app as app_mod
    from trace_bench.registry import TaskSpec

    class _FakeGr:
        @staticmethod
        def CheckboxGroup(**kwargs):
            return kwargs

    monkeypatch.setattr(app_mod, "_import_gradio", lambda: _FakeGr)
    monkeypatch.setattr(
        "trace_bench.registry.discover_tasks",
        lambda _root, _bench=None: [
            TaskSpec(id="internal:numeric_param", suite="internal", module="internal_numeric_param"),
            TaskSpec(id="trace_examples:greeting_stub", suite="trace_examples", module="greeting_stub"),
        ],
    )
    _text, update = app_mod._discover_tasks_for_ui(".", "")
    assert update["choices"] == ["internal:numeric_param", "trace_examples:greeting_stub"]
    assert update["value"] == []


def test_discover_trainers_for_ui_no_preselection(monkeypatch):
    """Trainer discovery should not preselect any values."""
    import trace_bench.ui.app as app_mod
    from trace_bench.registry import TrainerSpec

    class _FakeGr:
        @staticmethod
        def CheckboxGroup(**kwargs):
            return kwargs

    monkeypatch.setattr(app_mod, "_import_gradio", lambda: _FakeGr)
    monkeypatch.setattr(
        "trace_bench.registry.discover_trainers",
        lambda: [
            TrainerSpec(id="PrioritySearch", source="x", available=True),
            TrainerSpec(id="Nope", source="x", available=False),
        ],
    )
    _text, update = app_mod._discover_trainers_for_ui()
    assert update["choices"] == ["PrioritySearch"]
    assert update["value"] == []
