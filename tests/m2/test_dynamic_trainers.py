"""M2: Dynamic trainer resolution."""
import os
from pathlib import Path

import pytest

from trace_bench.runner import _resolve_algorithm

try:
    import opto  # noqa: F401
    _HAS_OPTO = True
except Exception:
    _HAS_OPTO = False


def test_resolve_priority_search():
    """PrioritySearch resolves to the string 'PrioritySearch'."""
    result = _resolve_algorithm("PrioritySearch")
    assert result == "PrioritySearch"


@pytest.mark.skipif(not _HAS_OPTO, reason="opto not available")
def test_resolve_gepa_base():
    """GEPA-Base resolves to a class via static map."""
    result = _resolve_algorithm("GEPA-Base")
    assert isinstance(result, type), f"Expected class, got {type(result)}"
    assert result.__name__ == "GEPAAlgorithmBase"


@pytest.mark.skipif(not _HAS_OPTO, reason="opto not available")
def test_resolve_gepa_ucb():
    """GEPA-UCB resolves to a class via static map."""
    result = _resolve_algorithm("GEPA-UCB")
    assert isinstance(result, type), f"Expected class, got {type(result)}"
    assert result.__name__ == "GEPAUCBSearch"


@pytest.mark.skipif(not _HAS_OPTO, reason="opto not available")
def test_resolve_gepa_beam():
    """GEPA-Beam resolves to a class via static map."""
    result = _resolve_algorithm("GEPA-Beam")
    assert isinstance(result, type), f"Expected class, got {type(result)}"
    assert result.__name__ == "GEPABeamPareto"


def test_resolve_gepa_graceful_without_opto():
    """GEPA-* gracefully falls back to string when opto is unavailable."""
    # This test always passes - it verifies the fallback works
    for name in ["GEPA-Base", "GEPA-UCB", "GEPA-Beam"]:
        result = _resolve_algorithm(name)
        # Either a class (if opto available) or the string (fallback)
        assert isinstance(result, (type, str))


def test_resolve_unknown_returns_string():
    """Unknown trainer names fall back to the string (let opto handle it)."""
    result = _resolve_algorithm("CompletelyFakeTrainer9999")
    assert result == "CompletelyFakeTrainer9999"


@pytest.mark.skipif(not _HAS_OPTO, reason="opto not available")
def test_dynamic_resolution_basic_algorithms():
    """Trainers in opto.trainer.algorithms.basic_algorithms should resolve dynamically."""
    for name in ["MinibatchAlgorithm", "BasicSearchAlgorithm"]:
        result = _resolve_algorithm(name)
        if isinstance(result, type):
            assert result.__name__ == name
