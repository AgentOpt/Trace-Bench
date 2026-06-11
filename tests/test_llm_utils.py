import pytest

from trace_bench.llm import openai_compatible_model_name


def test_openai_compatible_model_name_strips_openrouter_prefix() -> None:
    assert (
        openai_compatible_model_name("openrouter/openai/gpt-4o-mini")
        == "openai/gpt-4o-mini"
    )


def test_openai_compatible_model_name_keeps_other_model_names() -> None:
    assert openai_compatible_model_name("gpt-4o-mini") == "gpt-4o-mini"


def test_openai_compatible_model_name_requires_string() -> None:
    with pytest.raises(TypeError, match="model must be a string"):
        openai_compatible_model_name(None)  # type: ignore[arg-type]
