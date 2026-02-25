from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import importlib
import os
import re
import sys

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_ENTRYPOINT = "my_processing_agents.optimize_veribench_agent"
_HF_DATASET_ID = "allenanie/veribench_with_prompts"

# Module-level caches (populated lazily)
_DATASET_CACHE: Any = None
_NAME_INDEX: Optional[Dict[str, int]] = None

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_veribench_root() -> Path:
    env_root = os.environ.get("TRACE_BENCH_VERIBENCH_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return (_repo_root() / "Veribench").resolve()


def _entrypoint_name() -> str:
    return os.environ.get("TRACE_BENCH_VERIBENCH_ENTRYPOINT", _DEFAULT_ENTRYPOINT)


def _ensure_import_path(root: Path) -> None:
    root_str = str(root)
    if root.exists() and root_str not in sys.path:
        sys.path.insert(0, root_str)


# ---------------------------------------------------------------------------
# Normalisation helpers (for entrypoint-based discovery)
# ---------------------------------------------------------------------------


def _to_task_name(item: Any) -> Optional[str]:
    if item is None:
        return None
    if isinstance(item, str):
        name = item.strip()
        return name or None
    if isinstance(item, dict):
        for key in ("task", "task_name", "task_id", "id", "name", "slug"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None
    return str(item)


def _normalize_tasks(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw.strip()] if raw.strip() else []
    if isinstance(raw, dict):
        candidates = raw.keys()
    elif isinstance(raw, Iterable):
        candidates = raw
    else:
        return []

    out: List[str] = []
    seen: set = set()
    for item in candidates:
        name = _to_task_name(item)
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


# ---------------------------------------------------------------------------
# HuggingFace dataset helpers
# ---------------------------------------------------------------------------


def _load_cached_dataset() -> Any:
    """Load (and cache) the VeriBench HuggingFace dataset."""
    global _DATASET_CACHE, _NAME_INDEX
    if _DATASET_CACHE is not None:
        return _DATASET_CACHE
    try:
        from datasets import load_dataset
        ds = load_dataset(_HF_DATASET_ID, split="train")
    except Exception as exc:
        raise NotImplementedError(
            f"veribench_unavailable: cannot load HF dataset {_HF_DATASET_ID}: {exc}"
        ) from exc
    _DATASET_CACHE = ds
    # Build name → row-index lookup
    index: Dict[str, int] = {}
    for i, row in enumerate(ds):
        name = _task_name_from_row(row, i)
        if name not in index:
            index[name] = i
    _NAME_INDEX = index
    return ds


def _task_name_from_row(row: Any, idx: int) -> str:
    """Derive a stable task name from a dataset row."""
    filename = row.get("filename") if hasattr(row, "get") else None
    if filename and isinstance(filename, str):
        # Strip .py extension if present
        name = filename.strip()
        if name.endswith(".py"):
            name = name[:-3]
        return name
    return f"task_{idx}"


def _find_task_row(task_name: str) -> Dict[str, Any]:
    """Resolve a task name to its dataset row."""
    ds = _load_cached_dataset()
    assert _NAME_INDEX is not None

    # Direct name match
    if task_name in _NAME_INDEX:
        return dict(ds[_NAME_INDEX[task_name]])

    # Try as integer index (e.g. "task_42" → 42, or just "42")
    idx = None
    m = re.match(r"^task_(\d+)$", task_name)
    if m:
        idx = int(m.group(1))
    elif task_name.isdigit():
        idx = int(task_name)
    if idx is not None and 0 <= idx < len(ds):
        return dict(ds[idx])

    raise NotImplementedError(
        f"veribench_unavailable: task '{task_name}' not found in dataset "
        f"({len(ds)} tasks available)"
    )


def _discover_from_dataset() -> List[str]:
    """Discover VeriBench task names from the HuggingFace dataset."""
    ds = _load_cached_dataset()
    names: List[str] = []
    seen: set = set()
    for i, row in enumerate(ds):
        name = _task_name_from_row(row, i)
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


# ---------------------------------------------------------------------------
# Entrypoint-based discovery (original path)
# ---------------------------------------------------------------------------


def load_entrypoint() -> Any:
    root = resolve_veribench_root()
    module_name = _entrypoint_name()
    _ensure_import_path(root)
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        raise NotImplementedError(
            f"veribench_unavailable: cannot import {module_name} from {root}: {exc}"
        ) from exc


def _discover_from_entrypoint() -> List[str]:
    """Try entrypoint-based discovery (list_tasks / TASKS attrs)."""
    module = load_entrypoint()

    for accessor in ("list_tasks", "get_task_list", "discover_tasks"):
        fn = getattr(module, accessor, None)
        if not callable(fn):
            continue
        try:
            names = _normalize_tasks(fn())
            if names:
                return names
        except Exception:
            continue

    for attr in ("TASKS", "TASK_LIST", "AVAILABLE_TASKS"):
        if not hasattr(module, attr):
            continue
        names = _normalize_tasks(getattr(module, attr))
        if names:
            return names

    raise NotImplementedError(
        "veribench_unavailable: entrypoint loaded but task list API not found"
    )


# ---------------------------------------------------------------------------
# Public discovery API (entrypoint → HF dataset → error)
# ---------------------------------------------------------------------------


def discover_task_names() -> List[str]:
    """Discover VeriBench task names.

    Tries the entrypoint module first, then falls back to the HuggingFace
    dataset.  Raises ``NotImplementedError`` only if both paths fail.
    """
    # Path 1: entrypoint-based
    try:
        return _discover_from_entrypoint()
    except NotImplementedError:
        pass

    # Path 2: HuggingFace dataset
    return _discover_from_dataset()


# ---------------------------------------------------------------------------
# Entrypoint-based bundle building (original path)
# ---------------------------------------------------------------------------


def _call_builder(builder: Any, task_name: str, eval_kwargs: Dict[str, Any]) -> Any:
    for key in ("task_name", "task", "task_id", "name"):
        kwargs = dict(eval_kwargs)
        kwargs[key] = task_name
        try:
            return builder(**kwargs)
        except TypeError:
            continue
    try:
        return builder(**dict(eval_kwargs))
    except Exception as exc:
        raise NotImplementedError(
            f"veribench_unavailable: builder call failed for task '{task_name}': {exc}"
        ) from exc


def _build_from_entrypoint(task_name: str, eval_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Try building a bundle via the entrypoint module."""
    module = load_entrypoint()
    for builder_name in ("build_trace_problem", "build_task_bundle", "build_problem"):
        builder = getattr(module, builder_name, None)
        if not callable(builder):
            continue
        bundle = _call_builder(builder, task_name, eval_kwargs)
        if not isinstance(bundle, dict):
            raise NotImplementedError(
                f"veribench_unavailable: {builder_name} returned non-dict for '{task_name}'"
            )
        return bundle
    raise NotImplementedError(
        "veribench_unavailable: entrypoint has no compatible bundle builder"
    )


# ---------------------------------------------------------------------------
# Dataset-based bundle building (fallback)
# ---------------------------------------------------------------------------


def _build_from_dataset(task_name: str, eval_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Build a trace-compatible bundle directly from the HF dataset."""
    row = _find_task_row(task_name)

    python_code = row.get("code", "")
    user_query = row.get("user_query", "")
    category = row.get("category", "unknown")

    # Lazy import opto (only needed when actually building bundles)
    try:
        import opto.trace as trace
        from opto.trainer.guide import Guide
    except ImportError as exc:
        raise NotImplementedError(
            f"veribench_unavailable: opto not installed: {exc}"
        ) from exc

    # --- Trainable agent ---------------------------------------------------
    @trace.model
    class _VBAgent:
        def __init__(self) -> None:
            self.lean_code = trace.node(
                "-- placeholder: Lean 4 translation pending",
                trainable=True,
            )

        def __call__(self, task_input: Any) -> Any:
            return self.translate(self.lean_code, task_input)

        @trace.bundle(trainable=True)  # type: ignore[misc]
        def translate(self, lean_code: str, task_input: Any) -> str:
            return lean_code

    # --- Guide (heuristic Lean-4 quality) ----------------------------------
    class _VBGuide(Guide):
        """Evaluate Lean 4 output quality without the Lean compiler."""

        _LEAN_KEYWORDS = {"theorem", "lemma", "def", "inductive", "structure",
                          "namespace", "open", "import", "where", "by"}

        def get_feedback(
            self,
            query: Any,
            response: Any,
            reference: Any,
            **kwargs: Any,
        ) -> tuple:
            text = str(response)
            score = 0.0
            parts: List[str] = []

            # Keyword presence (each adds 0.1, max 0.5)
            kw_hits = sum(1 for kw in self._LEAN_KEYWORDS if kw in text)
            kw_score = min(kw_hits * 0.1, 0.5)
            score += kw_score

            # Penalty for 'sorry' (each removes 0.1)
            sorry_count = text.lower().count("sorry")
            sorry_penalty = min(sorry_count * 0.1, 0.3)
            score -= sorry_penalty

            # Length bonus (non-trivial output)
            if len(text) > 50:
                score += 0.2

            score = max(0.0, min(score, 1.0))
            parts.append(f"lean_keywords={kw_hits}")
            parts.append(f"sorry_count={sorry_count}")
            parts.append(f"len={len(text)}")
            feedback = "; ".join(parts)
            return score, feedback

    agent = _VBAgent()
    guide = _VBGuide()

    # Truncate objective to avoid huge optimizer prompts
    task_desc = user_query[:800] if user_query else python_code[:800]

    return {
        "param": agent,
        "guide": guide,
        "train_dataset": {
            "inputs": [user_query or python_code],
            "infos": [{"python_code": python_code, "category": category}],
        },
        "optimizer_kwargs": {
            "objective": (
                "Translate the following Python code into correct Lean 4 code. "
                "Produce valid Lean 4 syntax with theorems and proofs.\n\n"
                + task_desc
            ),
            "memory_size": 5,
        },
        "metadata": {
            "benchmark": "veribench",
            "entry": "VeriBenchAgent",
            "task_name": task_name,
            "category": category,
        },
    }


# ---------------------------------------------------------------------------
# Public bundle API (entrypoint → dataset → error)
# ---------------------------------------------------------------------------


def build_bundle(task_name: str, eval_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build a trace bundle for *task_name*.

    Tries the entrypoint module first, then falls back to constructing a
    bundle from the HuggingFace dataset.
    """
    eval_kwargs = dict(eval_kwargs or {})

    # Path 1: entrypoint-based
    try:
        return _build_from_entrypoint(task_name, eval_kwargs)
    except NotImplementedError:
        pass

    # Path 2: dataset-based
    return _build_from_dataset(task_name, eval_kwargs)


def _clear_caches() -> None:
    """Reset module-level caches (for testing)."""
    global _DATASET_CACHE, _NAME_INDEX
    _DATASET_CACHE = None
    _NAME_INDEX = None


__all__ = [
    "build_bundle",
    "discover_task_names",
    "load_entrypoint",
    "resolve_veribench_root",
    "_clear_caches",
]
