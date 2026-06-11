from __future__ import annotations

import copy
from typing import Any, Dict

from trace_bench.runner import (
    _MaxCompletionTokensLLM,
    _current_llm_meta,
    _extract_token_usage,
    _make_optimizer_litellm,
    _uses_max_completion_tokens,
)


def test_current_llm_meta_detects_openrouter(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("TRACE_LITELLM_MODEL", "openrouter/openai/gpt-4o-mini")
    meta = _current_llm_meta()
    assert meta["llm_provider"] == "openrouter"
    assert "openrouter" in meta["llm_base_url"]


def test_extract_token_usage_from_dict():
    usage = _extract_token_usage(
        {"usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}}
    )
    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == 4
    assert usage["total_tokens"] == 14


def test_uses_max_completion_tokens_for_gpt5_and_reasoning_models():
    assert _uses_max_completion_tokens("gpt-5.4-nano")
    assert _uses_max_completion_tokens("openai/o3-mini")
    assert not _uses_max_completion_tokens("gpt-4o-mini")


def test_max_completion_tokens_llm_translates_keyword():
    calls = []

    def fake_llm(**kwargs: Any) -> str:
        calls.append(kwargs)
        return "ok"

    wrapped = _MaxCompletionTokensLLM(fake_llm)

    assert wrapped(max_tokens=7, response_format={"type": "json_object"}) == "ok"
    assert calls == [
        {"max_completion_tokens": 7, "response_format": {"type": "json_object"}}
    ]


def test_max_completion_tokens_llm_deepcopy_keeps_delegate() -> None:
    class CallableLLM:
        def __call__(self, **kwargs: Any) -> Dict[str, Any]:
            return kwargs

    copied = copy.deepcopy(_MaxCompletionTokensLLM(CallableLLM()))

    assert copied(max_tokens=5) == {"max_completion_tokens": 5}


def test_make_optimizer_litellm_falls_back_without_mm_beta() -> None:
    class OldLiteLLM:
        def __init__(self, model: str) -> None:
            self.model = model

        def __call__(self, **kwargs: Any) -> Dict[str, Any]:
            return kwargs

    llm = _make_optimizer_litellm(OldLiteLLM, "gpt-5.4-nano")

    assert isinstance(llm, _MaxCompletionTokensLLM)
    assert llm(max_tokens=3) == {"max_completion_tokens": 3}


def test_make_optimizer_litellm_uses_mm_beta_when_supported() -> None:
    class NewLiteLLM:
        def __init__(self, model: str, mm_beta: bool = True) -> None:
            self.model = model
            self.mm_beta = mm_beta

    llm = _make_optimizer_litellm(NewLiteLLM, "gpt-4o-mini")

    assert llm.model == "gpt-4o-mini"
    assert llm.mm_beta is False
