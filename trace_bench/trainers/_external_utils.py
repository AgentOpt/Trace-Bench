from __future__ import annotations

from copy import deepcopy
import importlib
from typing import Any, Dict, List, Mapping, Sequence, Tuple


def collect_trainable_parameters(model: Any) -> List[Any]:
    """Return trainable parameter-like objects from a model or standalone parameter."""
    if hasattr(model, "parameters") and callable(model.parameters):
        parameters = [parameter for parameter in model.parameters() if getattr(parameter, "trainable", False)]
        if parameters:
            return list(parameters)
        raise ValueError("Model.parameters() returned no trainable parameters.")
    if getattr(model, "trainable", False) and hasattr(model, "data"):
        return [model]
    raise TypeError("Expected a model with parameters() or a standalone trainable parameter-like object.")


def coerce_like(example_value: Any, candidate_value: Any) -> Any:
    """Coerce a candidate value to the same literal-like type as the current parameter value."""
    if isinstance(example_value, bool):
        if not isinstance(candidate_value, bool):
            raise TypeError("Expected a boolean candidate value.")
        return candidate_value
    if isinstance(example_value, int) and not isinstance(example_value, bool):
        if isinstance(candidate_value, bool):
            raise TypeError("Expected an integer candidate value.")
        if isinstance(candidate_value, int):
            return candidate_value
        if isinstance(candidate_value, float) and candidate_value.is_integer():
            return int(candidate_value)
        raise TypeError("Expected an integer candidate value.")
    if isinstance(example_value, float):
        if isinstance(candidate_value, bool) or not isinstance(candidate_value, (int, float)):
            raise TypeError("Expected a numeric candidate value.")
        return float(candidate_value)
    if isinstance(example_value, str):
        if not isinstance(candidate_value, str):
            raise TypeError("Expected a string candidate value.")
        return candidate_value
    if isinstance(example_value, list):
        if not isinstance(candidate_value, list):
            raise TypeError("Expected a list candidate value.")
        return candidate_value
    if isinstance(example_value, tuple):
        if not isinstance(candidate_value, (list, tuple)):
            raise TypeError("Expected a sequence candidate value.")
        return tuple(candidate_value)
    if isinstance(example_value, dict):
        if not isinstance(candidate_value, dict):
            raise TypeError("Expected a mapping candidate value.")
        return candidate_value
    raise TypeError(f"Unsupported trainable parameter value type: {type(example_value).__name__}.")


def snapshot_parameter_values(parameters: Sequence[Any]) -> Dict[Any, Any]:
    """Deep-copy the current values of the provided parameters."""
    return {parameter: deepcopy(getattr(parameter, "data")) for parameter in parameters}


def _set_parameter_value(parameter: Any, value: Any) -> None:
    """Set a parameter-like object's value in a way that works across Trace variants."""
    try:
        setattr(parameter, "data", deepcopy(value))
        return
    except Exception:
        pass
    if hasattr(parameter, "_data"):
        setattr(parameter, "_data", deepcopy(value))
        return
    raise TypeError("Parameter object does not expose a writable data field.")


def restore_parameter_values(snapshot: Mapping[Any, Any]) -> None:
    """Restore a parameter snapshot created by snapshot_parameter_values()."""
    for parameter, value in snapshot.items():
        _set_parameter_value(parameter, value)


def apply_parameter_updates(update_dict: Mapping[Any, Any]) -> None:
    """Apply candidate parameter updates in place."""
    for parameter, value in update_dict.items():
        _set_parameter_value(parameter, value)


def score_model_on_dataset(agent: Any, guide: Any, dataset: Dict[str, Any], *, suppress_exceptions: bool = False) -> Tuple[float, List[str]]:
    """Evaluate an agent on a Trace-Bench dataset and return mean score plus feedback strings."""
    inputs = dataset.get("inputs") or []
    infos = dataset.get("infos") or dataset.get("info") or []
    if len(inputs) != len(infos):
        raise ValueError("Dataset 'inputs' and 'infos' must have the same length.")
    if not inputs:
        raise ValueError("Dataset must contain at least one example.")

    scores: List[float] = []
    feedbacks: List[str] = []
    for index, (task_input, task_info) in enumerate(zip(inputs, infos)):
        try:
            output = agent(task_input)
            response = getattr(output, "data", output)
            score, feedback = guide(task_input, response, task_info)
            scores.append(float(score))
            feedbacks.append(str(feedback))
        except Exception as exc:
            if not suppress_exceptions:
                raise
            scores.append(float("-inf"))
            feedbacks.append(f"evaluation_error[{index}]: {type(exc).__name__}")

    return sum(scores) / len(scores), feedbacks


def summarize_feedback(feedbacks: Sequence[str], *, max_items: int = 3) -> str:
    """Return a compact textual summary of the first few feedback strings."""
    items = [str(item) for item in feedbacks[:max_items]]
    return " | ".join(items)


def resolve_external_trainer_base() -> type:
    """Resolve the most compatible trainer base across OpenTrace variants."""
    try:
        module = importlib.import_module("opto.trainer.algorithms.algorithm")
    except Exception:
        return object

    for class_name in ("Trainer", "AbstractAlgorithm", "Algorithm", "AlgorithmBase"):
        trainer_base = getattr(module, class_name, None)
        if isinstance(trainer_base, type):
            return trainer_base

    return object
