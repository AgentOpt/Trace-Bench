from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import importlib
import os
import sys


_DEFAULT_ENTRYPOINT = "my_processing_agents.optimize_veribench_agent"


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
    seen = set()
    for item in candidates:
        name = _to_task_name(item)
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


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


def discover_task_names() -> List[str]:
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
        "veribench_unavailable: entrypoint loaded but task list API not found "
        "(expected list_tasks/get_task_list/TASKS)"
    )


def _call_builder(builder: Any, task_name: str, eval_kwargs: Dict[str, Any]) -> Any:
    # Try common task-name keyword variants.
    for key in ("task_name", "task", "task_id", "name"):
        kwargs = dict(eval_kwargs)
        kwargs[key] = task_name
        try:
            return builder(**kwargs)
        except TypeError:
            continue

    # Some builders may infer task from kwargs or internal defaults.
    try:
        return builder(**dict(eval_kwargs))
    except Exception as exc:
        raise NotImplementedError(
            f"veribench_unavailable: builder call failed for task '{task_name}': {exc}"
        ) from exc


def build_bundle(task_name: str, eval_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    module = load_entrypoint()
    eval_kwargs = dict(eval_kwargs or {})

    for builder_name in ("build_trace_problem", "build_task_bundle", "build_problem"):
        builder = getattr(module, builder_name, None)
        if not callable(builder):
            continue
        bundle = _call_builder(builder, task_name, eval_kwargs)
        if not isinstance(bundle, dict):
            raise NotImplementedError(
                f"veribench_unavailable: {builder_name} returned non-dict bundle for '{task_name}'"
            )
        return bundle

    raise NotImplementedError(
        "veribench_unavailable: entrypoint does not expose a compatible bundle builder "
        "(expected build_trace_problem/build_task_bundle/build_problem)"
    )


__all__ = [
    "build_bundle",
    "discover_task_names",
    "load_entrypoint",
    "resolve_veribench_root",
]
