from __future__ import annotations

from trace_bench.runner import _current_llm_meta, _extract_token_usage


def test_current_llm_meta_detects_openrouter(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("TRACE_LITELLM_MODEL", "openrouter/openai/gpt-4o-mini")
    meta = _current_llm_meta()
    assert meta["llm_provider"] == "openrouter"
    assert "openrouter" in meta["llm_base_url"]


def test_extract_token_usage_from_dict():
    usage = _extract_token_usage({"usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}})
    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == 4
    assert usage["total_tokens"] == 14
