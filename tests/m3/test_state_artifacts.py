from __future__ import annotations

from trace_bench.runner import _select_best_state, _state_rel_paths, _token_scope_note


def test_state_rel_paths_shape():
    paths = _state_rel_paths("job123")
    assert paths["initial_state_path"].endswith("jobs/job123/artifacts/initial_state.yaml")
    assert paths["best_state_path"].endswith("jobs/job123/artifacts/best_state.yaml")
    assert paths["final_state_path"].endswith("jobs/job123/artifacts/final_state.yaml")
    assert paths["state_history_path"].endswith("jobs/job123/artifacts/state_history.jsonl")


def test_select_best_state_prefers_improved_final():
    initial = {"v": 1}
    final = {"v": 2}
    assert _select_best_state(initial, final, 1.0, 2.0) == final
    assert _select_best_state(initial, final, 2.0, 1.0) == initial


def test_token_scope_note_is_explicit():
    assert _token_scope_note() == "trace_optimization_only"
