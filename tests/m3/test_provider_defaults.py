from __future__ import annotations

import os

from trace_bench.ui.app import _provider_defaults


def test_provider_defaults_openai_uses_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("TRACE_LITELLM_MODEL", "gpt-4o-mini")

    base, key, model = _provider_defaults("openai", "", "")
    assert base == "https://api.openai.com/v1"
    assert key == "sk-openai"
    assert model == "gpt-4o-mini"


def test_provider_defaults_openrouter_prefers_openrouter_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-openrouter")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")

    base, key, model = _provider_defaults("openrouter", "", "")
    assert base == "https://openrouter.ai/api/v1"
    assert key == "sk-openrouter"
    assert model


def test_provider_defaults_custom_keeps_manual_values(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("TRACE_LITELLM_MODEL", raising=False)

    base, key, model = _provider_defaults("custom", "https://custom/v1", "my-model")
    assert base == "https://custom/v1"
    assert key == ""
    assert model == "my-model"
