from pathlib import Path

import pytest

from trace_bench.registry import _parse_bench, discover_tasks, load_task_bundle


def test_parse_bench_allows_terminal_bench():
    assert _parse_bench("terminal_bench") == {"terminal_bench"}


def test_terminal_bench_is_explicit_not_default():
    assert "terminal_bench" not in _parse_bench(None)


def test_discover_terminal_bench_subset_smoke():
    repo_root = Path(__file__).resolve().parents[1]
    tasks = discover_tasks(repo_root, bench="terminal_bench")
    ids = [t.id for t in tasks]
    assert any(tid.startswith("terminal_bench:") for tid in ids)


def test_load_terminal_bench_bundle_stub_feedback(monkeypatch):
    monkeypatch.setenv("TRACE_BENCH_MODE", "stub")
    bundle = load_task_bundle(
        "terminal_bench:regex-log", Path(__file__).resolve().parents[1]
    )
    assert bundle["metadata"]["benchmark"] == "terminal_bench"
    score, feedback = bundle["guide"].get_feedback("regex-log", "prompt", None)
    assert score == 0.0
    assert "harbor not invoked" in feedback


def test_terminal_bench_real_mode_requires_model(monkeypatch):
    from trace_bench.terminal_bench.task import TerminalBenchGuide

    monkeypatch.setenv("TRACE_BENCH_MODE", "real")
    monkeypatch.delenv("HARBOR_MODEL", raising=False)
    guide = TerminalBenchGuide(eval_kwargs={})
    with pytest.raises(RuntimeError) as exc:
        guide.get_feedback("regex-log", "prompt", None)
    assert "requires a Harbor model" in str(exc.value)


def test_terminal_bench_default_prompt_uses_harbor_placeholders(monkeypatch):
    monkeypatch.setenv("TRACE_BENCH_MODE", "stub")
    bundle = load_task_bundle(
        "terminal_bench:regex-log", Path(__file__).resolve().parents[1]
    )
    rendered = str(bundle["param"]("regex-log"))
    assert "{instruction}" in rendered
    assert "{terminal_state}" in rendered
    assert "{{task_instruction}}" not in rendered
