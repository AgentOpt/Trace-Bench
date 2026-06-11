from __future__ import annotations

import ast
import asyncio
import inspect
import os
from functools import partial
from pathlib import Path
from pprint import pformat
from threading import RLock
from typing import Any, Dict, List, Optional, Union

try:
    from openevolve import run_evolution as _run_evolution
except Exception as exc:
    raise ImportError("OpenEvolveTrainer requires the optional 'openevolve' package.") from exc

from trace_bench.llm import openai_compatible_model_name
from trace_bench.trainers._external_utils import apply_parameter_updates, collect_trainable_parameters, coerce_like, resolve_external_trainer_base, restore_parameter_values, score_model_on_dataset, snapshot_parameter_values, summarize_feedback

_TrainerBase = resolve_external_trainer_base()
_EVALUATION_LOCK = RLock()

def _validate_literal_value(value: Any) -> None:
    """Ensure a parameter value round-trips through repr() and ast.literal_eval()."""
    try:
        ast.literal_eval(repr(value))
    except Exception as exc:
        raise TypeError(f"OpenEvolveTrainer supports only literal-like parameter values; got {type(value).__name__}.") from exc

def _serialize_candidate_program(parameters: List[Any]) -> str:
    """Serialize the current trainable parameter values to a safe Python literal program."""
    payload: Dict[str, Any] = {}
    for parameter in parameters:
        value = getattr(parameter, "data")
        _validate_literal_value(value)
        payload[parameter.py_name] = value
    return "candidate = " + pformat(payload, sort_dicts=True) + "\n"

def _parse_candidate_program(program_text: str, parameters: List[Any]) -> Dict[Any, Any]:
    """Parse a candidate program and coerce it back into parameter values."""
    try:
        syntax_tree = ast.parse(program_text, mode="exec")
    except SyntaxError as exc:
        raise ValueError("Candidate program must be valid Python.") from exc
    if len(syntax_tree.body) != 1 or not isinstance(syntax_tree.body[0], ast.Assign):
        raise ValueError("Candidate program must contain exactly one assignment to 'candidate'.")
    assignment = syntax_tree.body[0]
    if len(assignment.targets) != 1 or not isinstance(assignment.targets[0], ast.Name) or assignment.targets[0].id != "candidate":
        raise ValueError("Candidate program must assign a literal mapping to 'candidate'.")
    try:
        candidate_mapping = ast.literal_eval(assignment.value)
    except Exception as exc:
        raise ValueError("Candidate mapping must be parseable via ast.literal_eval().") from exc
    if not isinstance(candidate_mapping, dict):
        raise ValueError("Candidate mapping must be a dict.")
    expected_names = {parameter.py_name for parameter in parameters}
    if set(candidate_mapping.keys()) != expected_names:
        raise ValueError("Candidate mapping keys must exactly match the trainable parameter names.")
    update_dict: Dict[Any, Any] = {}
    for parameter in parameters:
        update_dict[parameter] = coerce_like(getattr(parameter, "data"), candidate_mapping[parameter.py_name])
    return update_dict

def _extract_best_code(result: Any) -> str:
    """Extract the best candidate program text from an OpenEvolve result object."""
    if isinstance(result, dict):
        for key in ("best_code", "code", "best_program"):
            value = result.get(key)
            if isinstance(value, str):
                return value
    for attribute in ("best_code", "code", "best_program"):
        value = getattr(result, attribute, None)
        if isinstance(value, str):
            return value
    raise ValueError("run_evolution did not return a best_code-like string.")

def _filter_supported_kwargs(function: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop kwargs that are not accepted by the target callable."""
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in signature.parameters}

def _run_evolution_compatible(kwargs: Dict[str, Any]) -> Any:
    """Run OpenEvolve even when the caller already owns an asyncio event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _run_evolution(**kwargs)

    try:
        import nest_asyncio
    except ImportError as exc:
        raise RuntimeError(
            "OpenEvolveTrainer requires nest_asyncio when called from an active asyncio event loop."
        ) from exc
    nest_asyncio.apply()
    return _run_evolution(**kwargs)

def _build_openevolve_config(*, model: Optional[str], api_base: Optional[str], api_key: Optional[str], api_key_env: str, max_tokens: int, temperature: Optional[float], iterations: int, seed: Optional[int], population_size: Optional[int], num_islands: Optional[int]) -> Any:
    """Build an OpenEvolve config for OpenAI-compatible providers when requested."""
    if api_key_env and not isinstance(api_key_env, str):
        raise TypeError("api_key_env must be a string.")
    if population_size is not None and population_size < 1:
        raise ValueError("population_size must be at least 1.")
    if num_islands is not None and num_islands < 1:
        raise ValueError("num_islands must be at least 1.")
    resolved_api_key = api_key or (os.environ.get(api_key_env) if api_key_env else None)
    resolved_api_base = api_base or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    resolved_model = model or os.environ.get("TRACE_LITELLM_MODEL")
    if not any((resolved_api_key, resolved_api_base, resolved_model, population_size, num_islands)):
        return None
    if not resolved_model:
        resolved_model = "gpt-4o-mini"
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1.")

    from openevolve.config import Config, LLMModelConfig

    config = Config(max_iterations=iterations, random_seed=seed)
    if population_size is not None:
        config.database.population_size = population_size
    if num_islands is not None:
        config.database.num_islands = num_islands
    if resolved_api_base:
        config.llm.api_base = resolved_api_base
    if resolved_api_key:
        config.llm.api_key = resolved_api_key
    config.llm.max_tokens = max_tokens
    config.llm.temperature = temperature
    model_config = LLMModelConfig(
        name=openai_compatible_model_name(resolved_model),
        api_base=config.llm.api_base,
        api_key=config.llm.api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=config.llm.timeout,
        retries=config.llm.retries,
        retry_delay=config.llm.retry_delay,
        random_seed=seed,
    )
    config.llm.models = [model_config]
    config.llm.evaluator_models = [model_config]
    return config

class OpenEvolveTrainer(_TrainerBase):
    """Trace-Bench wrapper around OpenEvolve using safe literal parameter serialization."""

    USES_TRACE_OPTIMIZER = False

    def __init__(self, agent: Any, optimizer: Any = None, logger: Any = None, **_kwargs: Any) -> None:
        del optimizer
        self.param = agent
        self.logger = logger

    def train(self, guide: Any, train_dataset: Dict[str, Any], *, mode: str = "real", validate_dataset: Optional[Dict[str, Any]] = None, iterations: int = 10, population_size: Optional[int] = None, num_islands: Optional[int] = None, seed: Optional[int] = None, ensure_improvement: bool = True, improvement_threshold: float = 0.0, verbose: Union[bool, str] = False, model: Optional[str] = None, api_base: Optional[str] = None, api_key: Optional[str] = None, api_key_env: str = "OPENAI_API_KEY", max_tokens: int = 4096, temperature: Optional[float] = 0.7, output_dir: Optional[str] = None, cleanup: bool = True, **_kwargs: Any) -> Dict[str, Any]:
        """Optimize Trace parameters with OpenEvolve via a literal candidate mapping."""
        if mode not in {"real", "stub"}:
            raise ValueError("mode must be either 'real' or 'stub'.")
        if iterations < 1:
            raise ValueError("iterations must be at least 1.")
        if mode == "stub":
            return {"status": "ok", "resolved_optimizer": "openevolve.run_evolution"}

        parameters = collect_trainable_parameters(self.param)
        evaluation_dataset = validate_dataset or train_dataset
        baseline_snapshot = snapshot_parameter_values(parameters)
        baseline_score, _ = score_model_on_dataset(agent=self.param, guide=guide, dataset=evaluation_dataset, suppress_exceptions=True)

        def evaluator(candidate_path: str) -> Dict[str, Any]:
            program_text = Path(candidate_path).read_text(encoding="utf-8")
            try:
                update_dict = _parse_candidate_program(program_text, parameters)
            except (TypeError, ValueError) as exc:
                return {"score": float("-inf"), "combined_score": float("-inf"), "feedback": str(exc)}
            with _EVALUATION_LOCK:
                snapshot = snapshot_parameter_values(parameters)
                try:
                    apply_parameter_updates(update_dict)
                    score, feedbacks = score_model_on_dataset(agent=self.param, guide=guide, dataset=evaluation_dataset, suppress_exceptions=True)
                finally:
                    restore_parameter_values(snapshot)
            return {"score": score, "combined_score": score, "feedback": summarize_feedback(feedbacks), "artifacts": {"candidate": {parameter.py_name: value for parameter, value in update_dict.items()}}}

        initial_program = _serialize_candidate_program(parameters)
        config = _build_openevolve_config(model=model, api_base=api_base, api_key=api_key, api_key_env=api_key_env, max_tokens=max_tokens, temperature=temperature, iterations=iterations, seed=seed, population_size=population_size, num_islands=num_islands)
        run_kwargs = {"iterations": iterations, "population_size": population_size, "num_islands": num_islands, "seed": seed, "verbose": verbose if isinstance(verbose, bool) else False, "config": config, "output_dir": output_dir, "cleanup": cleanup}
        filtered_kwargs = _filter_supported_kwargs(_run_evolution, {key: value for key, value in run_kwargs.items() if value is not None})
        result = _run_evolution_compatible(
            {"initial_program": initial_program, "evaluator": partial(evaluator), **filtered_kwargs}
        )

        best_code = _extract_best_code(result)
        best_update = _parse_candidate_program(best_code, parameters)
        apply_parameter_updates(best_update)
        if ensure_improvement:
            candidate_score, _ = score_model_on_dataset(agent=self.param, guide=guide, dataset=evaluation_dataset, suppress_exceptions=True)
            if candidate_score < baseline_score + improvement_threshold:
                restore_parameter_values(baseline_snapshot)

        return {"status": "ok", "resolved_optimizer": "openevolve.run_evolution"}
