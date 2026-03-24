from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import importlib
import inspect
import os
import tempfile


def _package_available(package_name: str) -> bool:
    try:
        importlib.import_module(package_name)
        return True
    except Exception:
        return False


_EXTERNAL_TRAINERS: Dict[str, Dict[str, str]] = {
    "DSPy-MIPROv2": {"package": "dspy", "source": "external:dspy"},
    "DSPy-BootstrapFewShot": {"package": "dspy", "source": "external:dspy"},
    "TextGrad-TGD": {"package": "textgrad", "source": "external:textgrad"},
    "OpenEvolve": {"package": "openevolve", "source": "external:openevolve"},
}

_EXTERNAL_ALLOWED_PARAMS: Dict[str, set[str]] = {
    "DSPy-MIPROv2": {
        "threads", "num_threads", "num_trials", "teacher", "valset", "seed", "verbose",
        "minibatch", "minibatch_size", "minibatch_full_eval_steps", "max_errors", "valset_ratio",
    },
    "DSPy-BootstrapFewShot": {
        "threads", "num_threads", "teacher", "seed", "verbose", "max_errors",
    },
    "TextGrad-TGD": {
        "num_steps", "steps", "verbose", "learning_rate", "lr",
    },
    "OpenEvolve": {
        "iterations", "num_iterations", "num_steps", "seed", "verbose",
        "population_size", "num_islands", "include_artifacts", "max_artifact_bytes",
    },
}


def discover_external_trainers() -> List[Dict[str, Any]]:
    return [
        {
            "id": trainer_id,
            "source": meta["source"],
            "available": _package_available(meta["package"]),
        }
        for trainer_id, meta in _EXTERNAL_TRAINERS.items()
    ]


def is_external_trainer(trainer_id: str) -> bool:
    return trainer_id in _EXTERNAL_TRAINERS


def allowed_external_trainer_kwargs(trainer_id: str) -> Optional[set[str]]:
    return _EXTERNAL_ALLOWED_PARAMS.get(trainer_id)


def _safe_score(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except Exception:
        return None


def _state_serializer_fallback(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    return {"kind": type(obj).__name__, "repr": repr(obj)}


def _extract_bundle_score(bundle: Dict[str, Any], candidate_value: Any) -> tuple[Any, Any]:
    dataset = bundle["train_dataset"]
    guide = bundle["guide"]
    inputs = dataset.get("inputs") or []
    infos = dataset.get("infos") or []
    if not inputs or not infos:
        return None, "empty_dataset"
    task_input = inputs[0]
    task_info = infos[0]
    try:
        score, feedback = guide(task_input, candidate_value, task_info)
        return score, feedback
    except Exception as exc:
        return None, f"guide_error: {exc}"


def _score_from_eval_result(result: Any) -> tuple[Any, Any, Dict[str, Any]]:
    feedback = None
    artifacts: Dict[str, Any] = {}
    score = None
    if isinstance(result, tuple):
        if len(result) >= 2:
            score, feedback = result[0], result[1]
        else:
            score = result[0]
    elif isinstance(result, dict):
        if "score" in result:
            score = result.get("score")
        elif isinstance(result.get("metrics"), dict):
            metrics = result["metrics"]
            score = metrics.get("score")
            if score is None and metrics:
                score = next(iter(metrics.values()))
        feedback = result.get("feedback")
        artifacts = dict(result.get("artifacts") or {})
    else:
        score = result
    return score, feedback, artifacts


def _apply_text_candidate(bundle: Dict[str, Any], candidate_text: str) -> None:
    param = bundle.get("param")
    if callable(getattr(param, "_set", None)):
        param._set(candidate_text)
        return
    if hasattr(param, "data"):
        setattr(param, "data", candidate_text)
        return
    raise ValueError("Cannot apply text candidate back to bundle param")


def _select_best_state(initial_state: Dict[str, Any], final_state: Dict[str, Any], score_initial: Any, score_final: Any) -> Dict[str, Any]:
    si = _safe_score(score_initial)
    sf = _safe_score(score_final)
    if sf is not None and (si is None or sf >= si):
        return final_state
    return initial_state or final_state


def _build_dspy_runtime(bundle: Dict[str, Any]) -> Dict[str, Any]:
    spec = (bundle.get("frameworks") or {}).get("dspy")
    if not isinstance(spec, dict):
        raise ValueError("Task does not expose a DSPy adapter payload under bundle['frameworks']['dspy']")
    required = ["program_factory", "metric", "evaluate"]
    missing = [k for k in required if not callable(spec.get(k))]
    if missing:
        raise ValueError(f"DSPy adapter missing callables: {missing}")
    return spec


def _build_textgrad_runtime(bundle: Dict[str, Any]) -> Dict[str, Any]:
    spec = (bundle.get("frameworks") or {}).get("textgrad")
    if isinstance(spec, dict):
        return spec
    param = bundle.get("param")
    value = getattr(param, "data", None)
    if isinstance(value, str):
        objective = (bundle.get("optimizer_kwargs") or {}).get("objective", "Improve this text")
        return {
            "initial_text": value,
            "role_description": "trainable text",
            "evaluate": lambda candidate: _extract_bundle_score(bundle, candidate)[0],
            "feedback": lambda candidate: _extract_bundle_score(bundle, candidate)[1],
            "objective": objective,
            "apply_update": lambda candidate_value, b=bundle: _apply_text_candidate(b, candidate_value),
        }
    raise ValueError("Task does not expose a TextGrad adapter and generic text fallback is not possible")


def _build_openevolve_runtime(bundle: Dict[str, Any]) -> Dict[str, Any]:
    spec = (bundle.get("frameworks") or {}).get("openevolve")
    if not isinstance(spec, dict):
        raise ValueError("Task does not expose an OpenEvolve adapter payload under bundle['frameworks']['openevolve']")
    if not (spec.get("initial_program") is not None or callable(spec.get("initial_program_factory"))):
        raise ValueError("OpenEvolve adapter missing initial_program or initial_program_factory")
    if not (callable(spec.get("evaluate_path")) or callable(spec.get("evaluate_program"))):
        raise ValueError("OpenEvolve adapter missing evaluate_path or evaluate_program")
    return spec


def _dspy_runner_class(trainer_id: str):
    dspy = importlib.import_module("dspy")
    mapping = {
        "DSPy-MIPROv2": ["MIPROv2", "MIPRO"],
        "DSPy-BootstrapFewShot": ["BootstrapFewShot"],
    }
    class_candidates = mapping.get(trainer_id)
    if not class_candidates:
        raise ValueError(f"Unknown DSPy trainer: {trainer_id}")

    namespaces = [dspy]
    teleprompt = getattr(dspy, "teleprompt", None)
    if teleprompt is not None:
        namespaces.append(teleprompt)
    try:
        namespaces.append(importlib.import_module("dspy.teleprompt"))
    except Exception:
        pass

    for cls_name in class_candidates:
        for ns in namespaces:
            if hasattr(ns, cls_name):
                return getattr(ns, cls_name)

    raise AttributeError(f"DSPy optimizer class not found: {class_candidates[0]}")


def run_dspy_trainer(bundle: Dict[str, Any], trainer_spec: Any, params: Dict[str, Any], mode: str) -> Dict[str, Any]:
    spec = _build_dspy_runtime(bundle)
    if mode == "stub" and callable(spec.get("stub_setup")):
        spec["stub_setup"]()
    elif callable(spec.get("setup")):
        spec["setup"]()

    program = spec["program_factory"]()
    metric = spec["metric"]
    to_trainset = spec.get("to_trainset")
    trainset = spec.get("trainset")
    if callable(to_trainset):
        trainset = to_trainset(bundle.get("train_dataset", {}))
    elif callable(trainset):
        trainset = trainset(bundle.get("train_dataset", {}))
    if trainset is None:
        trainset = bundle.get("train_dataset")

    optimizer_init = dict(spec.get("optimizer_init") or {})
    optimizer_init.update(dict(trainer_spec.optimizer_kwargs or {}))
    compile_kwargs = dict(spec.get("compile_kwargs") or {})
    compile_kwargs.update(dict(params or {}))
    # optimizer kwargs belong to optimizer ctor, not compile().
    compile_kwargs.pop("optimizer_kwargs", None)
    if "threads" in compile_kwargs and "num_threads" not in compile_kwargs:
        compile_kwargs["num_threads"] = compile_kwargs.pop("threads")

    runner_cls = _dspy_runner_class(trainer_spec.id)
    ctor_kwargs = {"metric": metric, **optimizer_init}
    ctor_kwargs = _filter_supported_kwargs(runner_cls, ctor_kwargs)
    optimizer = runner_cls(**ctor_kwargs)

    # Some DSPy versions expect these on compile() instead of constructor.
    for k in ("max_bootstrapped_demos", "max_labeled_demos", "max_rounds", "num_trials", "auto", "eval_kwargs"):
        if k in optimizer_init and k not in compile_kwargs:
            compile_kwargs[k] = optimizer_init[k]
    compile_kwargs.setdefault("eval_kwargs", {})
    # Keep benchmark runs non-interactive for MIPRO variants.
    compile_kwargs.setdefault("requires_permission_to_run", False)
    compile_kwargs.setdefault("view_data", False)
    compile_kwargs.setdefault("view_examples", False)

    compile_kwargs = _filter_supported_kwargs(optimizer.compile, compile_kwargs)
    evaluate = spec["evaluate"]
    score_initial = evaluate(program, trainset)
    compiled = optimizer.compile(student=program, trainset=trainset, **compile_kwargs)
    score_final = evaluate(compiled, trainset)

    if callable(spec.get("sync_to_bundle")):
        spec["sync_to_bundle"](compiled, bundle)

    serializer = spec.get("state_serializer") if callable(spec.get("state_serializer")) else _state_serializer_fallback
    initial_state = serializer(program)
    final_state = serializer(compiled)
    best_state = _select_best_state(initial_state, final_state, score_initial, score_final)
    feedback = f"compiled with {trainer_spec.id}"
    return {
        "status": "ok",
        "score_initial": score_initial,
        "score_final": score_final,
        "score_best": max(filter(lambda x: x is not None, [_safe_score(score_initial), _safe_score(score_final)]), default=None),
        "feedback": feedback,
        "resolved_optimizer": f"DSPy.{trainer_spec.id}",
        "resolved_trainer_kwargs": compile_kwargs,
        "resolved_optimizer_kwargs": {k: v for k, v in ctor_kwargs.items() if k != "metric"},
        "resolved_guide": "dspy metric",
        "resolved_logger": None,
        "initial_state": initial_state,
        "final_state": final_state,
        "best_state": best_state,
    }


def _resolve_textgrad_engine(tg: Any, optimizer_init: Dict[str, Any]) -> Any:
    engine = (
        optimizer_init.get("engine")
        or os.environ.get("TRACE_TEXTGRAD_ENGINE")
        or os.environ.get("TRACE_LITELLM_MODEL")
        or os.environ.get("OPENAI_MODEL")
    )
    if not engine:
        raise ValueError(
            "TextGrad-TGD requires an engine. Set trainer.optimizer_kwargs.engine "
            "or TRACE_TEXTGRAD_ENGINE / TRACE_LITELLM_MODEL."
        )
    try:
        tg.set_backward_engine(engine, override=True)
    except TypeError:
        tg.set_backward_engine(engine)
    return engine


def _filter_supported_kwargs(fn: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(fn)
    except Exception:
        return dict(kwargs)
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


def run_textgrad_trainer(bundle: Dict[str, Any], trainer_spec: Any, params: Dict[str, Any], mode: str) -> Dict[str, Any]:
    del mode
    tg = importlib.import_module("textgrad")
    spec = _build_textgrad_runtime(bundle)
    if callable(spec.get("setup")):
        spec["setup"]()

    initial_text = spec.get("initial_text")
    if initial_text is None and callable(spec.get("variable_factory")):
        initial = spec["variable_factory"](bundle)
        # variable_factory may directly return a Variable or text
        if hasattr(initial, "value"):
            variable = initial
            initial_text = getattr(variable, "value", None)
        else:
            initial_text = initial
            variable = None
    else:
        variable = None

    if variable is None:
        variable = tg.Variable(
            initial_text,
            role_description=spec.get("role_description", "trainable text"),
            requires_grad=True,
        )

    optimizer_init = dict(spec.get("optimizer_init") or {})
    optimizer_init.update(dict(trainer_spec.optimizer_kwargs or {}))
    engine = _resolve_textgrad_engine(tg, optimizer_init)
    optimizer_init.setdefault("engine", engine)

    loss_fn = spec.get("loss_fn")
    if loss_fn is None:
        if callable(spec.get("loss_fn_factory")):
            loss_fn = spec["loss_fn_factory"](bundle)
        else:
            objective = spec.get("objective") or (bundle.get("optimizer_kwargs") or {}).get("objective", "Improve this text")
            loss_fn = tg.TextLoss(objective, engine=engine)

    optimizer = tg.TGD(parameters=[variable], **optimizer_init)
    num_steps = int(params.get("num_steps", params.get("steps", 1)))

    evaluate = spec.get("evaluate") or (lambda value: _extract_bundle_score(bundle, value)[0])
    feedback_fn = spec.get("feedback") or (lambda value: _extract_bundle_score(bundle, value)[1])
    score_initial = evaluate(getattr(variable, "value", initial_text))
    for _ in range(num_steps):
        loss = loss_fn(variable)
        loss.backward()
        optimizer.step()

    candidate = getattr(variable, "value", initial_text)
    if callable(spec.get("apply_update")):
        spec["apply_update"](candidate)
    else:
        _apply_text_candidate(bundle, candidate)

    score_final = evaluate(candidate)
    serializer = spec.get("state_serializer") if callable(spec.get("state_serializer")) else (lambda obj: {"kind": "TextGradVariable", "value": getattr(obj, "value", obj)})
    initial_state = serializer(initial_text)
    final_state = serializer(candidate)
    best_state = _select_best_state(initial_state, final_state, score_initial, score_final)
    return {
        "status": "ok",
        "score_initial": score_initial,
        "score_final": score_final,
        "score_best": max(filter(lambda x: x is not None, [_safe_score(score_initial), _safe_score(score_final)]), default=None),
        "feedback": feedback_fn(candidate),
        "resolved_optimizer": "TextGrad.TGD",
        "resolved_trainer_kwargs": dict(params or {}),
        "resolved_optimizer_kwargs": optimizer_init,
        "resolved_guide": "TextLoss / adapter evaluator",
        "resolved_logger": None,
        "initial_state": initial_state,
        "final_state": final_state,
        "best_state": best_state,
    }


def run_openevolve_trainer(bundle: Dict[str, Any], trainer_spec: Any, params: Dict[str, Any], mode: str) -> Dict[str, Any]:
    del mode
    spec = _build_openevolve_runtime(bundle)
    if callable(spec.get("setup")):
        spec["setup"]()
    openevolve = importlib.import_module("openevolve")
    run_evolution = getattr(openevolve, "run_evolution")

    initial_program = spec.get("initial_program")
    if initial_program is None and callable(spec.get("initial_program_factory")):
        initial_program = spec["initial_program_factory"](bundle)
    if not isinstance(initial_program, str):
        raise ValueError("OpenEvolve initial program must be a string")

    def evaluator(candidate_path: str):
        if callable(spec.get("evaluate_path")):
            return spec["evaluate_path"](candidate_path)
        code = Path(candidate_path).read_text(encoding="utf-8")
        return spec["evaluate_program"](code)

    run_kwargs = dict(spec.get("run_kwargs") or {})
    run_kwargs.update(dict(trainer_spec.optimizer_kwargs or {}))
    iterations = int(params.get("iterations", params.get("num_iterations", params.get("num_steps", 1))))
    other_kwargs = {k: v for k, v in (params or {}).items() if k not in {"iterations", "num_iterations", "num_steps"}}
    safe_kwargs = _filter_supported_kwargs(run_evolution, {**run_kwargs, **other_kwargs})

    # Baseline evaluation from the initial program text
    if callable(spec.get("evaluate_program")):
        score_initial, feedback_initial, _ = _score_from_eval_result(spec["evaluate_program"](initial_program))
    else:
        with tempfile.TemporaryDirectory(prefix="tb_openevolve_init_") as td:
            p = Path(td) / "candidate.py"
            p.write_text(initial_program, encoding="utf-8")
            score_initial, feedback_initial, _ = _score_from_eval_result(spec["evaluate_path"](str(p)))

    result = run_evolution(initial_program=initial_program, evaluator=evaluator, iterations=iterations, **safe_kwargs)

    best_code = None
    if isinstance(result, dict):
        best_code = result.get("best_code") or result.get("code") or result.get("best_program")
    else:
        for attr in ("best_code", "code", "best_program"):
            if hasattr(result, attr):
                best_code = getattr(result, attr)
                break
    if not isinstance(best_code, str):
        raise ValueError("OpenEvolve result does not expose best_code")

    if callable(spec.get("apply_candidate")):
        spec["apply_candidate"](best_code, bundle)
    else:
        _apply_text_candidate(bundle, best_code)

    if callable(spec.get("evaluate_program")):
        score_final, feedback_final, artifacts = _score_from_eval_result(spec["evaluate_program"](best_code))
    else:
        with tempfile.TemporaryDirectory(prefix="tb_openevolve_best_") as td:
            p = Path(td) / "candidate.py"
            p.write_text(best_code, encoding="utf-8")
            score_final, feedback_final, artifacts = _score_from_eval_result(spec["evaluate_path"](str(p)))

    serializer = spec.get("state_serializer") if callable(spec.get("state_serializer")) else (lambda code: {"kind": "OpenEvolveProgram", "program": code})
    initial_state = serializer(initial_program)
    final_state = serializer(best_code)
    best_state = _select_best_state(initial_state, final_state, score_initial, score_final)
    feedback = feedback_final or feedback_initial or "optimized with OpenEvolve"
    if artifacts:
        feedback = f"{feedback} | artifacts: {artifacts}"
    return {
        "status": "ok",
        "score_initial": score_initial,
        "score_final": score_final,
        "score_best": max(filter(lambda x: x is not None, [_safe_score(score_initial), _safe_score(score_final)]), default=None),
        "feedback": feedback,
        "resolved_optimizer": "OpenEvolve.run_evolution",
        "resolved_trainer_kwargs": {"iterations": iterations, **other_kwargs},
        "resolved_optimizer_kwargs": safe_kwargs,
        "resolved_guide": "OpenEvolve evaluator",
        "resolved_logger": None,
        "initial_state": initial_state,
        "final_state": final_state,
        "best_state": best_state,
    }


def run_external_trainer(bundle: Dict[str, Any], trainer_spec: Any, params: Dict[str, Any], mode: str) -> Dict[str, Any]:
    if trainer_spec.id.startswith("DSPy-"):
        return run_dspy_trainer(bundle, trainer_spec, params, mode)
    if trainer_spec.id == "TextGrad-TGD":
        return run_textgrad_trainer(bundle, trainer_spec, params, mode)
    if trainer_spec.id == "OpenEvolve":
        return run_openevolve_trainer(bundle, trainer_spec, params, mode)
    raise ValueError(f"Unknown external trainer: {trainer_spec.id}")
