from __future__ import annotations

from typing import Any, Dict, List


_FILTERED_KWARGS = {"eval_kwargs", "optimizer_kwargs"}


def _default_trainer_kwargs(algo_name: str) -> Dict[str, Any]:
    if algo_name == "PrioritySearch":
        return dict(num_epochs=1, num_steps=1, num_batches=1, num_candidates=2, num_proposals=2)
    if algo_name == "GEPA-Base":
        return dict(num_iters=1, train_batch_size=2, merge_every=2, pareto_subset_size=2)
    # GEPA-UCB and GEPA-Beam use num_search_iterations
    return dict(num_search_iterations=1, train_batch_size=2, merge_every=2, pareto_subset_size=2)


def _param_alias_map(algo_name: str) -> Dict[str, str]:
    base = {
        "threads": "num_threads",
        "ps_steps": "num_steps",
        "ps_batches": "num_batches",
        "ps_candidates": "num_candidates",
        "ps_proposals": "num_proposals",
        "ps_mem_update": "memory_update_frequency",
        "gepa_train_bs": "train_batch_size",
        "gepa_merge_every": "merge_every",
        "gepa_pareto_subset": "pareto_subset_size",
    }
    if algo_name == "GEPA-Base":
        base["gepa_iters"] = "num_iters"
    else:
        base["gepa_iters"] = "num_search_iterations"
    return base


def resolve_trainer_kwargs(params: Dict[str, Any], algo_name: str) -> Dict[str, Any]:
    try:
        from trace_bench.integrations.external_optimizers import is_external_trainer
        if is_external_trainer(algo_name):
            return {k: v for k, v in (params or {}).items() if k not in _FILTERED_KWARGS}
    except Exception:
        pass
    kwargs = _default_trainer_kwargs(algo_name)
    alias_map = _param_alias_map(algo_name)
    for key, value in (params or {}).items():
        if key in _FILTERED_KWARGS:
            continue
        mapped_key = alias_map.get(key, key)
        kwargs[mapped_key] = value
    return kwargs


def _clone(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _clone(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone(v) for v in value]
    return value


def merge_kwargs(base: Any, override: Any) -> Any:
    if override is None:
        return _clone(base)
    if base is None:
        return _clone(override)
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        merged.update(override)
        return merged
    if isinstance(base, list) and isinstance(override, dict):
        if not base:
            return [_clone(override)]
        return [
            merge_kwargs(item, override) if isinstance(item, (dict, list)) else _clone(item)
            for item in base
        ]
    if isinstance(base, dict) and isinstance(override, list):
        if not override:
            return _clone(base)
        return [
            merge_kwargs(base, item) if isinstance(item, (dict, list)) else _clone(item)
            for item in override
        ]
    if isinstance(base, list) and isinstance(override, list):
        merged: List[Any] = []
        max_len = max(len(base), len(override))
        for idx in range(max_len):
            left = base[idx] if idx < len(base) else None
            right = override[idx] if idx < len(override) else None
            if left is None:
                merged.append(_clone(right))
            elif right is None:
                merged.append(_clone(left))
            else:
                merged.append(merge_kwargs(left, right))
        return merged
    return _clone(override)


__all__ = ["resolve_trainer_kwargs", "merge_kwargs"]
