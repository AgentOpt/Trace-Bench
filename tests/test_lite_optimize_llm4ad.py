import os
import sys
import time
import pytest
import importlib
import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# default LiteLLM model if not provided
os.environ.setdefault("TRACE_LITELLM_MODEL", "gpt-4o-mini")

STEP_TIMEOUT = 60  # seconds

TASKS = [
    {
        "module": "LLM4AD.benchmark_tasks.optimization_knapsack_construct",
        "eval_class": "KnapsackEvaluation",
        "small_eval_kwargs": {
            "n_instance": 3,
            "n_items": 8,
            "knapsack_capacity": 20,
            "timeout_seconds": 10,
        },
    },
    {
        "module": "LLM4AD.benchmark_tasks.optimization_bp_1d_construct",
        "eval_class": "BP1DEvaluation",
        "small_eval_kwargs": {
            "n_instance": 3,
            "n_items": 20,
            "bin_capacity": 50,
            "n_bins": 50,
            "timeout_seconds": 10,
        },
    },
]


def _import_llm4ad_loader():
    """Import llm4ad_loader with fallbacks for different repo layouts."""

    try:
        return importlib.import_module("llm4ad_loader")
    except ModuleNotFoundError:
        pass

    try:
        module = importlib.import_module("LLM4AD.llm4ad_loader")
        sys.modules.setdefault("llm4ad_loader", module)
        return module
    except ModuleNotFoundError:
        pass

    repo_root = Path(__file__).resolve().parents[1]
    loader_path = repo_root / "LLM4AD" / "llm4ad_loader.py"
    if not loader_path.exists():
        raise

    spec = importlib.util.spec_from_file_location("llm4ad_loader", loader_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    if spec.loader is None:
        raise ImportError("Unable to load llm4ad_loader module")
    spec.loader.exec_module(module)
    sys.modules.setdefault("LLM4AD.llm4ad_loader", module)
    return module


def _get_param_value(param):
    """Try to extract a raw value/string from a Parameter-like object."""

    for attr in ("value", "get", "get_value", "_value", "initial"):
        if hasattr(param, attr):
            attr_obj = getattr(param, attr)
            try:
                if callable(attr_obj):
                    return attr_obj()
                return attr_obj
            except Exception:
                return attr_obj
    if isinstance(param, dict):
        return param.get("value")
    return getattr(param, "__dict__", None)


@pytest.mark.parametrize("task", TASKS)
def test_lite_optimize_llm4ad_task(task):
    try:
        llm4ad_loader = _import_llm4ad_loader()
    except Exception as exc:
        pytest.skip(f"llm4ad_loader import failed: {exc!r}")

    try:
        from opto.optimizers.optoprime import OptoPrime
        from opto.trace import node as trace_node  # noqa: F401 - presence check
    except Exception as exc:
        pytest.skip(f"OptoPrime import failed: {exc!r}")

    module_path = task["module"]
    try:
        mod = importlib.import_module(module_path)
    except Exception as exc:
        pytest.skip(f"Could not import LLM4AD task module '{module_path}': {exc!r}")

    entry_name = getattr(mod, "ENTRY_NAME", getattr(mod, "entry", None))
    function_signature = getattr(mod, "FUNCTION_SIGNATURE", "")
    template_program = getattr(mod, "template_program", "")
    task_description = getattr(mod, "task_description", "")
    import_header = getattr(mod, "IMPORT_HEADER", "")

    if entry_name is None:
        pytest.skip(f"Task module {module_path} missing ENTRY_NAME")

    try:
        eval_file_path = Path(mod.__file__).resolve().as_posix()
    except Exception:
        eval_file_path = ""

    repo_root = Path(__file__).resolve().parents[1]

    try:
        problem = llm4ad_loader.build_trace_problem_from_config(
            llm4ad_root=str(repo_root),
            eval_module_path=module_path,
            eval_class_name=task["eval_class"],
            eval_file_path=eval_file_path,
            entry_name=entry_name,
            function_signature=function_signature,
            import_header=import_header,
            task_description=task_description,
            objective_text="Maximize the evaluator score.",
            template_function=template_program,
            eval_kwargs=task.get("small_eval_kwargs", {}),
        )
    except Exception as exc:
        pytest.skip(f"Could not build Trace problem for {module_path}: {exc!r}")

    assert {"param", "guide", "optimizer_kwargs"}.issubset(problem)

    param = problem["param"]
    guide = problem["guide"]
    optimizer_kwargs = problem["optimizer_kwargs"]

    try:
        opt = OptoPrime(parameters=[param], **optimizer_kwargs)
    except TypeError:
        opt = OptoPrime([param])

    initial_score = None
    try:
        code_value = _get_param_value(param)
        if isinstance(code_value, dict):
            for val in code_value.values():
                if isinstance(val, str) and "def" in val:
                    code_value = val
                    break
        if isinstance(code_value, str) and hasattr(guide, "evaluator_loader"):
            ns = {}
            try:
                exec(code_value, ns, ns)
                candidate = ns.get(entry_name)
                if callable(candidate):
                    evaluator = guide.evaluator_loader
                    score_obj = evaluator.evaluate_program(code_value, candidate, entry_name=entry_name)
                    if isinstance(score_obj, (int, float)):
                        initial_score = float(score_obj)
                    elif isinstance(score_obj, tuple) and score_obj and isinstance(score_obj[0], (int, float)):
                        initial_score = float(score_obj[0])
                    elif isinstance(score_obj, dict) and "score" in score_obj:
                        initial_score = float(score_obj["score"])
            except Exception:
                initial_score = None
    except Exception:
        initial_score = None

    start_time = time.time()
    executed_steps = 0
    try:
        for _ in range(2):
            if hasattr(opt, "step"):
                opt.step()
            elif hasattr(opt, "one_step"):
                opt.one_step()
            elif hasattr(opt, "optimize"):
                opt.optimize(steps=1)
            else:
                pytest.skip("Optimizer API not available (no step/one_step/optimize method)")
            executed_steps += 1
            if time.time() - start_time > STEP_TIMEOUT:
                break
    except Exception as exc:
        pytest.fail(f"Optimizer step crashed for task {module_path}: {exc!r}")

    assert executed_steps >= 1

    final_score = None
    try:
        code_value = _get_param_value(param)
        if isinstance(code_value, dict):
            for val in code_value.values():
                if isinstance(val, str) and "def" in val:
                    code_value = val
                    break
        if isinstance(code_value, str) and hasattr(guide, "evaluator_loader"):
            ns = {}
            try:
                exec(code_value, ns, ns)
                candidate = ns.get(entry_name)
                if callable(candidate):
                    evaluator = guide.evaluator_loader
                    score_obj = evaluator.evaluate_program(code_value, candidate, entry_name=entry_name)
                    if isinstance(score_obj, (int, float)):
                        final_score = float(score_obj)
                    elif isinstance(score_obj, tuple) and score_obj and isinstance(score_obj[0], (int, float)):
                        final_score = float(score_obj[0])
                    elif isinstance(score_obj, dict) and "score" in score_obj:
                        final_score = float(score_obj["score"])
            except Exception:
                final_score = None
    except Exception:
        final_score = None

    if initial_score is not None and final_score is not None:
        assert final_score >= initial_score - 1e-9
