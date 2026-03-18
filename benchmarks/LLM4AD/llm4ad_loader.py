#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""llm4ad_loader.py
Autonomous LLM4AD task runner for Trace optimization.

This module provides a complete, self-contained implementation of LLM4AD evaluators
that doesn't depend on the original LLM4AD codebase. All necessary components are
either reimplemented here or copied from the original tasks.
"""

import sys, os, types, traceback, inspect, importlib, importlib.util, textwrap, json, time, multiprocessing as mp, signal
from typing import Any, Dict, Literal, Callable
from abc import ABC, abstractmethod
import ast
import numpy as np
from pathlib import Path

# You must have Trace installed and importable as `opto`.
from opto import trace
from opto.trainer.guide import Guide
from opto.trace.nodes import ParameterNode


# --- BEGIN: Bench-only augmentation of trace.node for code parameters ---
_ORIG_TRACE_NODE = trace.node


def _node_with_constraints(value, *args, **kwargs):
    """Append standard constraints to code parameters, leaving everything else unchanged."""

    try:
        name = kwargs.get("name", "")
        desc = kwargs.get("description", "") or ""
        trainable = kwargs.get("trainable", False)
        is_code_param = trainable and isinstance(value, str) and (
            name == "__code" or "The code should start with" in desc
        )
        if is_code_param:
            extra_constraints = (
                "\n"
                "Constraints (bench-added):\n"
                "- Do NOT change the function name or its signature.\n"
                "- Use only the imports provided by the harness; avoid top-level I/O or non-determinism.\n"
                "- The return value MUST strictly follow the schema described in the docstring/task."
            )
            if "Constraints (bench-added)" not in desc:
                kwargs["description"] = (desc + extra_constraints).strip()
    except Exception:
        # Fail open: fall back to the original call if anything goes wrong.
        pass
    return _ORIG_TRACE_NODE(value, *args, **kwargs)


trace.node = _node_with_constraints
# --- END: Bench-only augmentation ---


# ============================================================================
# LLM4AD Base Classes (reimplemented for autonomy)
# ============================================================================

class Evaluation(ABC):
    """Base evaluation class reimplemented from LLM4AD for benchmark tasks."""
    
    def __init__(
        self,
        template_program: str = '',
        task_description: str = '',
        timeout_seconds: int | float = 30,
        random_seed: int | None = None,
        exec_code: bool = True,
        safe_evaluate: bool = False,  # Simplified - no multiprocessing by default
        **kwargs
    ):
        """Simplified Evaluation base class.
        
        Args:
            template_program: The template program string (not used in our implementation)
            task_description: Description of the task (not used in our implementation)
            timeout_seconds: Time limit for evaluation
            random_seed: Random seed to set (not implemented)
            exec_code: Whether to exec the code (always True in our case)
            safe_evaluate: Whether to use safe evaluation (simplified, always False)
            **kwargs: Additional arguments (stored but not used)
        """
        self.template_program = template_program
        self.task_description = task_description
        self.timeout_seconds = timeout_seconds
        self.random_seed = random_seed
        self.exec_code = exec_code
        self.safe_evaluate = safe_evaluate
        self.kwargs = kwargs
    
    @abstractmethod
    def evaluate_program(self, program_str: str, callable_func: Callable, **kwargs) -> Any | None:
        """Evaluate a program. Must be implemented by subclasses.
        
        Args:
            program_str: The program as a string
            callable_func: The compiled callable function
            **kwargs: Additional evaluation arguments
            
        Returns:
            Evaluation score/result
        """
        pass


class LLM4ADEvaluatorLoader:
    """Dynamically load and instantiate LLM4AD evaluators from their original modules."""
    
    def __init__(self, llm4ad_root: str, eval_module_path: str, eval_class_name: str, eval_file_path: str = None, **eval_kwargs):
        self.llm4ad_root = Path(llm4ad_root)
        self.eval_module_path = eval_module_path
        self.eval_class_name = eval_class_name
        self.eval_file_path = eval_file_path
        self.eval_kwargs = eval_kwargs
        self._evaluator = None
    
    def _load_evaluator(self):
        """Load the evaluator class from LLM4AD and instantiate it."""
        if self._evaluator is not None:
            return self._evaluator
            
        try:
            # Add LLM4AD root and evaluation file directory to Python path temporarily
            original_path = sys.path.copy()
            if str(self.llm4ad_root) not in sys.path:
                sys.path.insert(0, str(self.llm4ad_root))
            # Also add the evaluation file's directory for local imports
            if self.eval_file_path:
                eval_dir = str(Path(self.eval_file_path).parent)
                if eval_dir not in sys.path:
                    sys.path.insert(0, eval_dir)
            
            try:
                # Try importing the module normally first
                try:
                    eval_module = importlib.import_module(self.eval_module_path)
                except (ImportError, ModuleNotFoundError):
                    # Fallback: direct file execution for problematic paths
                    eval_file_path = getattr(self, 'eval_file_path', None)
                    if eval_file_path and Path(eval_file_path).exists():
                        spec = importlib.util.spec_from_file_location(
                            f"eval_module_{hash(eval_file_path)}", 
                            eval_file_path
                        )
                        eval_module = importlib.util.module_from_spec(spec)
                        sys.modules[spec.name] = eval_module
                        spec.loader.exec_module(eval_module)
                    else:
                        raise
                
                # Get the evaluator class
                evaluator_class = getattr(eval_module, self.eval_class_name)
                
                # Instantiate with provided kwargs
                self._evaluator = evaluator_class(**self.eval_kwargs)
                
                return self._evaluator
                
            finally:
                # Restore original Python path
                sys.path = original_path
                
        except Exception as e:
            raise RuntimeError(f"Failed to load LLM4AD evaluator {self.eval_class_name} from {self.eval_module_path}: {e}")
    
    def evaluate_program(self, program_str: str, callable_func, **kwargs):
        """Evaluate using the LLM4AD evaluator's evaluate_program method."""
        evaluator = self._load_evaluator()
        return evaluator.evaluate_program(program_str, callable_func, **kwargs)


# -------- Minimal AST transforms inspired by LLM4AD.ModifyCode --------
def _add_numpy_import(program: str) -> str:
    tree = ast.parse(program)
    for node in tree.body:
        if isinstance(node, ast.Import) and any(alias.name == 'numpy' and alias.asname == 'np' for alias in node.names):
            return program
    tree.body.insert(0, ast.Import(names=[ast.alias(name='numpy', asname='np')]))
    return ast.unparse(tree)

def _inject_np_seed_in_func(program: str, func_name: str, seed: int) -> str:
    tree = ast.parse(program)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            node.body = [ast.parse(f'np.random.seed({seed})').body[0]] + node.body
            break
    return ast.unparse(tree)

class _DivToProtected(ast.NodeTransformer):
    def __init__(self, name: str): self.name = name
    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Div):
            return ast.Call(func=ast.Name(id=self.name, ctx=ast.Load()), args=[node.left, node.right], keywords=[])
        return node

def _replace_div_with_protected(program: str, delta: float, add_numba: bool=False) -> str:
    prot = f"""\n\ndef _protected_div(x, y, delta={delta}):\n    return x / (y + delta)\n"""
    tree = ast.parse(program)
    tree = _DivToProtected('_protected_div').visit(tree)
    code = ast.unparse(tree) + prot
    if add_numba:
        try:
            import numba  # noqa
            tree = ast.parse(code)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == '_protected_div':
                    node.decorator_list.append(ast.Call(func=ast.Attribute(value=ast.Name(id='numba', ctx=ast.Load()), attr='jit', ctx=ast.Load()),
                                                        args=[], keywords=[ast.keyword(arg='nopython', value=ast.Constant(value=True))]))
            code = ast.unparse(tree)
        except Exception:
            pass
    return code

def _apply_llm4ad_transforms(code: str, entry: str, cfg: dict) -> str:
    use_prot = bool(cfg.get('use_protected_div', False))
    delta = float(cfg.get('protected_div_delta', 1e-5))
    seed = cfg.get('random_seed', None)
    use_numba = bool(cfg.get('use_numba_accelerate', False))
    # seed injection needs numpy
    if seed is not None:
        code = _add_numpy_import(code)
        code = _inject_np_seed_in_func(code, entry, int(seed))
    if use_prot:
        code = _replace_div_with_protected(code, delta, add_numba=use_numba)
    return code


class LLM4ADEvaluatorGuide(Guide):
    """Trace Guide that uses LLM4AD evaluators for feedback."""
    
    def __init__(self, evaluator_loader: LLM4ADEvaluatorLoader, entry_name: str, import_header: str = '', timeout: float | None = None):
        self.evaluator_loader = evaluator_loader
        self._entry = entry_name
        self._import_header = import_header
        self._timeout = timeout
    
    def get_feedback(self, task: str, response: str, info: Any, **kwargs):
        # response is a code string (candidate). Compile it and evaluate using LLM4AD.
        start = time.time()
        feedback_lines = []

        def timeout_handler(signum, frame):
            raise TimeoutError("Evaluation timed out")

        try:
            # Set timeout (default 30 seconds for LLM4AD evaluations)
            timeout = self._timeout or 30.0
            use_signal = True
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(timeout))
            except ValueError as e:
                # signal only works in main thread - skip timeout when in thread
                if "main thread" in str(e):
                    use_signal = False
                else:
                    raise

            # Build namespace and exec the code
            ns: Dict[str, Any] = {}
            header = info.get('imports', '') if isinstance(info, dict) else self._import_header
            full_code = header + "\n" + response if header else response
            # Apply LLM4AD-like transforms if requested by evaluator kwargs
            cfg = getattr(self.evaluator_loader._evaluator, 'kwargs', {}) if getattr(self.evaluator_loader, '_evaluator', None) else {}
            if cfg:
                full_code = _apply_llm4ad_transforms(full_code, self._entry, cfg)
            exec(full_code, ns, ns)

            # robust entry detection
            func = ns.get(self._entry, None)
            if not (callable(func)):
                # auto-detect: functions defined in this exec (filename '<string>')
                cand = [ (k,v) for k,v in ns.items()
                         if callable(v) and hasattr(v, '__code__') and getattr(v.__code__, 'co_filename', '') == '<string>' ]
                if len(cand) == 1:
                    self._entry, func = cand[0][0], cand[0][1]
                else:
                    msg = f"Entry '{self._entry}' not found; candidates: {[k for k,_ in cand]}"
                    if use_signal: signal.alarm(0)
                    env = {"status": "error", "phase": "compile", "error": "entry_not_found", "candidates": [k for k,_ in cand]}
                    return -1000000.0, "TRACE_FEEDBACK_JSON=" + json.dumps(env) + "\n" + msg

            # Use LLM4AD's evaluate_program method
            try:
                score = self.evaluator_loader.evaluate_program(full_code, func, entry_name=self._entry)
                if use_signal:
                    signal.alarm(0)
                elapsed = time.time() - start
                # allow tuple/dict returns for richer feedback
                fb_detail = None
                if isinstance(score, tuple) and len(score) >= 1: score, fb_detail = score[0], score[1:]
                if isinstance(score, dict) and 'score' in score: fb_detail, score = score, score['score']
                if score is None or score == float('-inf') or score == float('inf'):
                    # Try to give a more informative error for infinite scores
                    if score == float('-inf'):
                        env = {"status": "error", "phase": "evaluate", "error": "score_negative_infinity"}
                        feedback_lines.append("TRACE_FEEDBACK_JSON=" + json.dumps(env))
                        feedback_lines.append('LLM4AD eval returned -inf (possible constraint violation or error)')
                        return -1000000.0, '\n'.join(feedback_lines)
                    elif score == float('inf'):
                        env = {"status": "error", "phase": "evaluate", "error": "score_positive_infinity"}
                        feedback_lines.append("TRACE_FEEDBACK_JSON=" + json.dumps(env))
                        feedback_lines.append('LLM4AD eval returned +inf (possible error in evaluation)')
                        return -1000000.0, '\n'.join(feedback_lines)
                    else:
                        env = {"status": "error", "phase": "evaluate", "error": "score_none"}
                        feedback_lines.append("TRACE_FEEDBACK_JSON=" + json.dumps(env))
                        feedback_lines.append('LLM4AD eval returned None')
                        return -1000000.0, '\n'.join(feedback_lines)

                env = {"status": "ok", "phase": "evaluate", "score": float(score) if score is not None else None}
                if fb_detail is not None:
                    env["details"] = fb_detail
                feedback_lines.append("TRACE_FEEDBACK_JSON=" + json.dumps(env, ensure_ascii=False))
                feedback_lines.append(f'LLM4AD eval OK in {elapsed:.2f}s; score={score}')
                return float(score), '\n'.join(feedback_lines)

            except (ValueError, RuntimeError, AssertionError) as eval_err:
                # Handle evaluation-specific errors more gracefully
                if use_signal:
                    signal.alarm(0)
                elapsed = time.time() - start
                env = {"status": "error", "phase": "evaluate", "error": str(eval_err)}
                feedback_lines.append("TRACE_FEEDBACK_JSON=" + json.dumps(env, ensure_ascii=False))
                feedback_lines.append(f'LLM4AD eval constraint violation in {elapsed:.2f}s: {eval_err}')
                # Return a large negative score instead of -inf to allow optimization
                return -1000000.0, '\n'.join(feedback_lines)

        except TimeoutError:
            if use_signal:
                signal.alarm(0)
            env = {"status": "timeout", "phase": "evaluate", "timeout_seconds": timeout}
            return -1000000.0, "TRACE_FEEDBACK_JSON=" + json.dumps(env) + f"\nEvaluation timed out after {timeout}s"
        except Exception as e:
            if use_signal:
                signal.alarm(0)
            tb = traceback.format_exc(limit=3)
            env = {"status": "error", "phase": "evaluate", "error": str(e)}
            return -1000000.0, "TRACE_FEEDBACK_JSON=" + json.dumps(env) + f"\nLLM4AD eval failed: {e}\n{tb}"

    def __call__(self, task: str, response: str, info: Any, **kwargs):
        return self.get_feedback(task, response, info, **kwargs)


def build_trace_problem_from_config(
    llm4ad_root: str,
    eval_module_path: str, 
    eval_class_name: str,
    eval_file_path: str,
    entry_name: str,
    function_signature: str,
    import_header: str,
    task_description: str,
    objective_text: str,
    template_function: str,
    eval_kwargs: dict,
    **override_eval_kwargs
) -> dict:
    """
    Build a Trace problem from LLM4AD task configuration.
    
    This is a common implementation that replaces the build_trace_problem function
    that was duplicated in every converted task file.
    
    Returns:
        dict with keys: param, guide, train_dataset, optimizer_kwargs, metadata
    """
    
    # 1) make the trainable code parameter
    initial_code = template_function.strip()
    param = trace.node(initial_code, name='__code', description=f'The code should start with: {function_signature}', trainable=True)

    # 2) Create dynamic LLM4AD evaluator loader
    eval_kwargs_final = eval_kwargs.copy()
    eval_kwargs_final.update(override_eval_kwargs)
    
    evaluator_loader = LLM4ADEvaluatorLoader(
        llm4ad_root=llm4ad_root,
        eval_module_path=eval_module_path,
        eval_class_name=eval_class_name,
        eval_file_path=eval_file_path,
        **eval_kwargs_final
    )
    
    # 3) Create guide that uses the LLM4AD evaluator
    timeout = eval_kwargs_final.get('timeout_seconds', 30)
    guide = LLM4ADEvaluatorGuide(evaluator_loader, entry_name, import_header, timeout=timeout)

    # 4) dataset: minimal 1-sample dataset
    train_dataset = dict(
        inputs=[task_description],
        infos=[{'imports': import_header, 'entry': entry_name}]
    )

    # 5) optimizer hints (objective)
    optimizer_kwargs = dict(
        objective=objective_text,
        memory_size=10
    )

    return dict(
        param=param,
        guide=guide,
        train_dataset=train_dataset,
        optimizer_kwargs=optimizer_kwargs,
        metadata=dict(
            entry=entry_name,
            function_signature=function_signature,
            llm4ad_eval=eval_class_name,
            eval_module=eval_module_path,
            llm4ad_root=llm4ad_root,
        )
    )

# ---------- helpers: robust evaluation in threads (no signal) ----------
def _eval_in_subprocess(evaluator, entry_name: str, full_code: str, timeout: float):
    """
    Run compile+evaluate in a child process so we can enforce a wall-time timeout
    even when we're in a non-main thread (where signal.alarm is unavailable).
    """
    ctx = mp.get_context("fork") if hasattr(mp, "get_context") else mp
    q = ctx.Queue()

    def _worker(q_):
        try:
            ns = {}
            exec(full_code, ns, ns)
            func = ns.get(entry_name, None)
            if not callable(func):
                q_.put(dict(status="error", phase="compile", error="entry_not_found"))
                return
            score = evaluator.evaluate_program(full_code, func)
            q_.put(dict(status="ok", phase="evaluate", score=score))
        except Exception as e:
            tb = traceback.format_exc(limit=2)
            q_.put(dict(status="error", phase="evaluate", error=str(e), traceback=tb))

    p = ctx.Process(target=_worker, args=(q,))
    p.daemon = True
    p.start()
    p.join(timeout)
    if p.is_alive():
        try:
            p.terminate()
        finally:
            p.join()
        return -1000000.0, {"status": "timeout", "phase": "evaluate", "timeout_seconds": timeout}

    try:
        res = q.get_nowait()
    except Exception:
        return -1000000.0, {"status": "error", "phase": "evaluate", "error": "no_result_from_child"}

    if res.get("status") != "ok":
        # compile/runtime error path
        err = {k: res.get(k) for k in ("status", "phase", "error", "traceback") if k in res}
        return -1000000.0, err

    score = res.get("score", None)
    try:
        import numpy as _np
        finite = (score is not None) and _np.isfinite(score)
    except Exception:
        finite = isinstance(score, (int, float))
    if not finite:
        return -1000000.0, {"status": "error", "phase": "evaluate", "error": "non_finite_score", "raw": str(score)}
    return float(score), {"status": "ok", "phase": "evaluate", "score": float(score)}

class AutonomousEvaluatorGuide(Guide):
    """Trace Guide that uses benchmark (embedded) LLM4AD evaluators."""
    
    def __init__(self, evaluator: Evaluation, entry_name: str, import_header: str = '', timeout: float | None = None):
        self.evaluator = evaluator
        self._entry = entry_name
        self._import_header = import_header
        self._timeout = timeout

    def get_feedback(self, task: str, response: str, info: Any, **kwargs):
        # response is a code string (candidate). Compile it and evaluate using embedded evaluator.
        import signal
        start = time.time()
        feedback_lines = []
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Evaluation timed out")
        
        try:
            # Set timeout (default 30 seconds for LLM4AD evaluations)
            timeout = self._timeout or 30.0
            use_signal = True
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(timeout))
            except ValueError as e:
                # signal only works in main thread - skip timeout when in thread
                if "main thread" in str(e):
                    use_signal = False
                else:
                    raise
            
            header = info.get('imports', '') if isinstance(info, dict) else self._import_header
            full_code = header + "\n" + response if header else response
            # Apply LLM4AD-like transforms if requested by evaluator kwargs
            cfg = getattr(self.evaluator, 'kwargs', {}) if hasattr(self.evaluator, 'kwargs') else {}
            if cfg:
                full_code = _apply_llm4ad_transforms(full_code, self._entry, cfg)

            # Path A: we have signal — evaluate in-process with alarm
            if use_signal:
                ns: Dict[str, Any] = {}
                exec(full_code, ns, ns)
                func = ns.get(self._entry, None)
                if not callable(func):
                    cand = [k for k,v in ns.items() if callable(v) and hasattr(v,'__code__') and getattr(v.__code__,'co_filename','')=='<string>']
                    signal.alarm(0)
                    env = {"status":"error","phase":"compile","error":"entry_not_found","candidates":cand}
                    return -1000000.0, "TRACE_FEEDBACK_JSON=" + json.dumps(env) + f"\nEntry '{self._entry}' not found; candidates: {cand}"
                score = self.evaluator.evaluate_program(full_code, func)
                signal.alarm(0)
            # Path B: no signal — evaluate in a child process with timeout
            else:
                score, env = _eval_in_subprocess(self.evaluator, self._entry, full_code, timeout)
                elapsed = time.time() - start
                feedback_lines.append("TRACE_FEEDBACK_JSON=" + json.dumps(env, ensure_ascii=False))
                if env.get("status") == "ok":
                    feedback_lines.append(f'Autonomous eval OK in {elapsed:.2f}s; score={score}')
                elif env.get("status") == "timeout":
                    return -1000000.0, '\n'.join(feedback_lines)
                else:
                    return -1000000.0, '\n'.join(feedback_lines)

            # Normalize score and structure feedback
            elapsed = time.time() - start
            fb_detail = None
            if isinstance(score, tuple) and len(score) >= 1: score, fb_detail = score[0], score[1:]
            if isinstance(score, dict) and 'score' in score: fb_detail, score = score, score['score']
            try:
                import numpy as _np
                finite = (score is not None) and _np.isfinite(score)
            except Exception:
                finite = isinstance(score, (int, float))
            if not finite:
                env = {"status":"error","phase":"evaluate","error":"non_finite_score","raw": str(score)}
                feedback_lines.append("TRACE_FEEDBACK_JSON=" + json.dumps(env))
                return -1000000.0, '\n'.join(feedback_lines)
            env = {"status":"ok","phase":"evaluate","score": float(score)}
            if fb_detail is not None:
                env["details"] = fb_detail
            feedback_lines.append("TRACE_FEEDBACK_JSON=" + json.dumps(env, ensure_ascii=False))
            feedback_lines.append(f'Autonomous eval OK in {elapsed:.2f}s; score={score}')
            return float(score), '\n'.join(feedback_lines)

        except TimeoutError:
            if use_signal:
                signal.alarm(0)
            env = {"status": "timeout", "phase": "evaluate", "timeout_seconds": timeout}
            return -1000000.0, "TRACE_FEEDBACK_JSON=" + json.dumps(env) + f"\nEvaluation timed out after {timeout}s"
        except Exception as e:
            if use_signal:
                signal.alarm(0)
            tb = traceback.format_exc(limit=3)
            env = {"status": "error", "phase": "evaluate", "error": str(e)}
            return -1000000.0, "TRACE_FEEDBACK_JSON=" + json.dumps(env) + f"\nAutonomous eval failed: {e}\n{tb}"

    def __call__(self, task: str, response: str, info: Any, **kwargs):
        return self.get_feedback(task, response, info, **kwargs)
    
def load_subdir_as_text(repo_id: str, subdir: str, *, skip_ext: tuple[str, ...] = (".py",), streaming: bool = False):
    """
    Load files from a subdirectory in a Hugging Face dataset as text format.
    
    Args:
        repo_id: The repository ID on Hugging Face (e.g., "CO-Bench/CO-Bench")
        subdir: The subdirectory path within the dataset
        skip_ext: File extensions to skip (default: (".py",))
        streaming: Whether to use streaming mode
        
    Returns:
        A dict where keys are original filenames and values are loaded datasets
        
    Example:
        ds = load_subdir_as_text("CO-Bench/CO-Bench", "Aircraft landing")
        # Returns: {"airland1.txt": Dataset(...), "airland2.txt": Dataset(...), ...}
    """
    from huggingface_hub import list_repo_files
    from datasets import load_dataset
    from pathlib import PurePosixPath
    prefix = subdir.rstrip("/") + "/"
    files = [
        f for f in list_repo_files(repo_id, repo_type="dataset")
        if f.startswith(prefix) and not f.endswith(skip_ext)
    ]
    if not files:
        raise FileNotFoundError(f"No matching files inside '{subdir}' on {repo_id}")
    
    # Create a mapping from sanitized split names to original filenames
    def sanitize_split_name(filename):
        """Convert filename to valid split name (only alphanumeric, dots, underscores)"""
        import re
        # Replace hyphens and other special chars with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9._]', '_', filename)
        return sanitized
    
    # Build data_files dict with sanitized split names
    data_files = {}
    filename_mapping = {}  # Maps sanitized names back to original names
    
    for f in files:
        original_filename = PurePosixPath(f).name
        sanitized_name = sanitize_split_name(original_filename)
        data_files[sanitized_name] = f
        filename_mapping[sanitized_name] = original_filename
    
    # Load the dataset
    dataset = load_dataset(
        repo_id,
        data_files=data_files,
        streaming=streaming,
    )
    
    # Return a dict with original filenames as keys
    result = {}
    for sanitized_name, original_filename in filename_mapping.items():
        result[original_filename] = dataset[sanitized_name]
    
    return result


def load_subdir_as_pickle(repo_id: str, subdir: str, *, include_subdirs: tuple[str, ...] = (), streaming: bool = False):
    """
    Load pickle files from a subdirectory in a Hugging Face dataset.
    
    Args:
        repo_id: The repository ID on Hugging Face (e.g., "CO-Bench/CO-Bench")
        subdir: The subdirectory path within the dataset
        include_subdirs: Tuple of subdirectory names to include (if empty, includes all)
        streaming: Whether to use streaming mode
        
    Returns:
        A dict where keys are subdirectory names and values are dicts of 
        {filename: loaded_pickle_content}
        
    Example:
        result = load_subdir_as_pickle("CO-Bench/CO-Bench", "Maximal independent set", 
                                     include_subdirs=("er_test", "er_large_test"))
        # Returns: {"er_test": {"file1.gpickle": graph1, ...}, "er_large_test": {...}}
    """
    import pickle
    from huggingface_hub import hf_hub_download, list_repo_files
    
    prefix = subdir.rstrip("/") + "/"
    files = [
        f for f in list_repo_files(repo_id, repo_type="dataset")
        if f.startswith(prefix) and f.endswith(('.pickle', '.gpickle', '.pkl'))
    ]
    
    if not files:
        raise FileNotFoundError(f"No pickle files found inside '{subdir}' on {repo_id}")
    
    # Organize files by subdirectory
    subdirs = {}
    for file_path in files:
        parts = file_path.split('/')
        if len(parts) >= 3:  # "subdir/subsubdir/filename"
            subsubdir = parts[1]  # The subdirectory under main subdir
            filename = parts[2]   # The actual filename
            
            # Filter by include_subdirs if specified
            if include_subdirs and subsubdir not in include_subdirs:
                continue
                
            if subsubdir not in subdirs:
                subdirs[subsubdir] = {}
            
            # Download and load the pickle file
            try:
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=file_path,
                    repo_type="dataset"
                )
                
                with open(local_path, "rb") as f:
                    pickle_content = pickle.load(f)
                
                subdirs[subsubdir][filename] = pickle_content
                
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
                continue
    
    return subdirs 