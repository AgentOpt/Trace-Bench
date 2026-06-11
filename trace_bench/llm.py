from __future__ import annotations


def openai_compatible_model_name(model: str) -> str:
    """Return the model identifier expected by OpenAI-compatible clients."""
    if not isinstance(model, str):
        raise TypeError("model must be a string.")
    if model.startswith("openrouter/"):
        return model.split("/", 1)[1]
    return model
