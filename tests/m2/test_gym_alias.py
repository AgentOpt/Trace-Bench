"""M2 blocker: gymnasium should be aliased to gym for legacy tasks."""

from __future__ import annotations

import sys
import types

from trace_bench.registry import _ensure_gym_alias


def test_gym_alias_prefers_gymnasium(monkeypatch):
    fake_gymnasium = types.ModuleType("gymnasium")
    monkeypatch.setitem(sys.modules, "gymnasium", fake_gymnasium)
    monkeypatch.delitem(sys.modules, "gym", raising=False)

    _ensure_gym_alias()

    assert sys.modules.get("gym") is fake_gymnasium

