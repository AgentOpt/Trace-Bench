from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set
import importlib
import importlib.util
import json
import sys


@dataclass
class TaskSpec:
    id: str
    suite: str
    module: str


@dataclass
class TrainerSpec:
    id: str
    source: str
    available: bool


_INTERNAL_TASKS = {
    "internal:code_param": "internal_code_param",
    "internal:numeric_param": "internal_numeric_param",
    "internal:multi_param": "internal_multi_param",
    "internal:non_trainable": "internal_non_trainable",
}

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_sys_path(path: Path) -> None:
    if path.exists():
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def ensure_opto_importable() -> None:
    try:
        import opto  # noqa: F401
        return
    except Exception:
        pass
    repo_root = _repo_root()
    _ensure_sys_path(repo_root.parent / "OpenTrace")


def ensure_llm4ad_importable(tasks_root: Path) -> None:
    _ensure_sys_path(_repo_root())
    _ensure_sys_path(tasks_root.parent)
    # Provide llm4ad_loader alias for task imports
    try:
        module = importlib.import_module("LLM4AD.llm4ad_loader")
        sys.modules.setdefault("llm4ad_loader", module)
    except Exception:
        pass


def _load_index(tasks_root: Path) -> List[Dict[str, Any]]:
    index_path = tasks_root / "index.json"
    if not index_path.exists():
        return []
    return json.loads(index_path.read_text(encoding="utf-8"))


def discover_llm4ad(tasks_root: Path) -> List[TaskSpec]:
    specs: List[TaskSpec] = []
    index = _load_index(tasks_root)
    if index:
        for entry in index:
            key = entry.get("key")
            module = entry.get("module") or entry.get("wrapper")
            if key and module:
                specs.append(TaskSpec(id=f"llm4ad:{key}", suite="llm4ad", module=module))
        return specs
    # fallback: directories
    for path in tasks_root.iterdir():
        if path.is_dir():
            specs.append(TaskSpec(id=f"llm4ad:{path.name}", suite="llm4ad", module=path.name))
    return specs


def discover_trace_examples() -> List[TaskSpec]:
    return [
        TaskSpec(id="trace_examples:greeting_stub", suite="trace_examples", module="greeting_stub"),
        TaskSpec(id="trace_examples:train_single_node_stub", suite="trace_examples", module="train_single_node_stub"),
    ]


def discover_internal() -> List[TaskSpec]:
    return [
        TaskSpec(id=task_id, suite="internal", module=module)
        for task_id, module in _INTERNAL_TASKS.items()
    ]

def discover_veribench() -> List[TaskSpec]:
    raise NotImplementedError("VeriBench tasks not yet wired: awaiting Trace team entrypoint/task list.")


def discover_trainers() -> List[TrainerSpec]:
    ensure_opto_importable()
    candidates = [
        ("PrioritySearch", "opto.features.priority_search", "PrioritySearch"),
        ("GEPA-Base", "opto.features.gepa.gepa_algorithms", "GEPAAlgorithmBase"),
        ("GEPA-UCB", "opto.features.gepa.gepa_algorithms", "GEPAUCBSearch"),
        ("GEPA-Beam", "opto.features.gepa.gepa_algorithms", "GEPABeamPareto"),
    ]
    specs: List[TrainerSpec] = []
    for trainer_id, module, symbol in candidates:
        available = True
        try:
            mod = importlib.import_module(module)
            getattr(mod, symbol)
        except Exception:
            available = False
        specs.append(TrainerSpec(id=trainer_id, source=module, available=available))
    return specs


def _parse_bench(bench: Optional[str]) -> Set[str]:
    if not bench:
        return {"llm4ad", "trace_examples", "internal"}
    normalized = bench.replace("+", ",")
    parts = [p.strip() for p in normalized.split(",") if p.strip()]
    if not parts:
        return {"llm4ad", "trace_examples", "internal"}
    allowed = {"llm4ad", "trace_examples", "internal", "veribench"}
    unknown = [p for p in parts if p not in allowed]
    if unknown:
        raise ValueError(f"Unknown bench selector(s): {unknown}. Allowed: {sorted(allowed)}")
    return set(parts)


def discover_tasks(tasks_root: str | Path, bench: Optional[str] = None) -> List[TaskSpec]:
    root = Path(tasks_root)
    selected = _parse_bench(bench)
    specs: List[TaskSpec] = []
    if "llm4ad" in selected:
        specs.extend(discover_llm4ad(root))
    if "trace_examples" in selected:
        specs.extend(discover_trace_examples())
    if "internal" in selected:
        specs.extend(discover_internal())
    if "veribench" in selected:
        specs.extend(discover_veribench())
    return specs


def _normalize_task_id(task_id: str) -> str:
    if task_id.startswith("example:"):
        return task_id.replace("example:", "trace_examples:", 1)
    if ":" in task_id:
        return task_id
    return f"llm4ad:{task_id}"


def load_task_module(task_id: str, tasks_root: str | Path):
    ensure_opto_importable()
    root = Path(tasks_root)
    task_id = _normalize_task_id(task_id)
    if task_id.startswith("trace_examples:"):
        module_name = task_id.split(":", 1)[1]
        return importlib.import_module(f"trace_bench.examples.{module_name}")
    if task_id.startswith("internal:"):
        module_name = _INTERNAL_TASKS.get(task_id, task_id.split(":", 1)[1])
        return importlib.import_module(f"trace_bench.examples.{module_name}")
    if task_id.startswith("veribench:"):
        raise NotImplementedError("VeriBench tasks not yet wired: awaiting Trace team entrypoint/task list.")

    ensure_llm4ad_importable(root)
    mapping = {spec.id.split(":", 1)[1]: spec.module for spec in discover_llm4ad(root)}
    task_key = task_id.split(":", 1)[1]
    module_dir = mapping.get(task_key, task_key)
    module_path = root / module_dir / "__init__.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Task module not found: {module_path}")

    module_name = f"trace_bench_task_{module_dir}_{abs(hash(str(module_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_task_bundle(task_id: str, tasks_root: str | Path, eval_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    task_id = _normalize_task_id(task_id)
    if task_id.startswith("veribench:"):
        raise NotImplementedError("VeriBench tasks not yet wired: awaiting Trace team entrypoint/task list.")
    mod = load_task_module(task_id, tasks_root)
    if not hasattr(mod, "build_trace_problem"):
        raise AttributeError(f"Task module {task_id} missing build_trace_problem")
    bundle = mod.build_trace_problem(**(eval_kwargs or {}))
    required = {"param", "guide", "train_dataset", "optimizer_kwargs", "metadata"}
    missing = required - set(bundle.keys())
    if missing:
        raise KeyError(f"Task bundle missing keys: {sorted(missing)}")
    return bundle


__all__ = [
    "TaskSpec",
    "TrainerSpec",
    "discover_tasks",
    "discover_trainers",
    "discover_veribench",
    "load_task_bundle",
    "load_task_module",
]
