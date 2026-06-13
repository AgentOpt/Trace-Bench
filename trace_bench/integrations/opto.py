"""Bundle evaluation helper for external OpenTrace / opto ``recursive_opt`` consumers.

WHY THIS EXISTS
---------------
NewTrace's ``opto.features.recursive_opt`` adapter (``TraceBenchTaskAdapter``)
needs to evaluate a loaded Trace-Bench bundle on its own dataset and read back
BOTH the scalar reward and the optional multi-objective ``score_dict`` for its
recursive scoring transforms (relative deltas, clipping, objective
scalarization). The benchmark repo is the right home for "load a bundle and run
it on its data"; the consumer keeps only the recursion-specific transforms.

This module is therefore a thin, stable, PUBLIC API surface for that external
consumer. It is intentionally not wired into ``runner.py``'s job pipeline (the
runner has its own batched/staged evaluation with stub-LLM handling). To avoid
the semantic drift the reviewer rightly flagged, the per-example evaluation here
REUSES the runner's own ``_extract_response`` and ``_evaluation_dataset`` rather
than re-deriving them — so "what counts as the response / which split is used"
has exactly one definition in this repo.

The only deliberate behavioral difference from ``runner._score_dataset`` is
shape, by design for the consumer: this returns per-example ``BundleEval``
records (reward + feedback + optional score_dict) instead of a single
``{score, feedback}`` summary, because the recursive optimizer scores each
example's objective vector. Aggregate reward matches the runner's mean.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from trace_bench.registry import load_task_bundle, normalize_task_id

__all__ = ["BundleEval", "evaluate_bundle", "load_and_evaluate"]


@dataclass
class BundleEval:
    """One example's evaluation: official reward, feedback, optional score dict."""

    reward: float
    feedback: str
    score_dict: Optional[Dict[str, float]] = None


def evaluate_bundle(
    bundle: Dict[str, Any],
    *,
    max_examples: int = 1,
    strict_score_dict: bool = False,
) -> Tuple[float, List[BundleEval]]:
    """Evaluate a loaded bundle on up to ``max_examples`` of its own dataset.

    Returns ``(mean_reward, per_example_evals)``. Split selection and response
    extraction are delegated to the runner's helpers (single source of truth).

    - ``strict_score_dict=False`` (default): a guide whose ``get_score_dict``
      raises yields ``score_dict=None`` for that example (the scalar reward is
      still recorded). This matches "score_dict is an optional enrichment".
    - ``strict_score_dict=True``: re-raise instead of swallowing, for consumers
      whose multi-objective scoring MUST have the dict (so a silent None can't
      masquerade as a degenerate objective vector).

    Raises ``ValueError`` when the chosen dataset is empty — a silent
    zero-example "success" would hide a real loading/dataset failure.
    """
    # Reuse the runner's canonical helpers so this integration cannot drift from
    # the repo's own response-extraction / split-selection logic. Imported lazily
    # to avoid a circular import (runner -> integrations package -> this module).
    from trace_bench.runner import _evaluation_dataset, _extract_response

    dataset_name, dataset = _evaluation_dataset(bundle)
    inputs = list(dataset.get("inputs") or [])
    infos = list(dataset.get("infos") or dataset.get("info") or [])
    limit = min(len(inputs), len(infos) or len(inputs),
                max_examples if max_examples and max_examples > 0 else len(inputs))
    if limit <= 0:
        raise ValueError(f"bundle dataset {dataset_name!r} has no evaluable examples")

    guide = bundle["guide"]
    param = bundle["param"]
    out: List[BundleEval] = []
    for i in range(limit):
        info = infos[i] if i < len(infos) else None
        response = _extract_response(param, inputs[i])
        reward, feedback = guide(inputs[i], response, info)
        score_dict = None
        if hasattr(guide, "get_score_dict"):
            try:
                score_dict = guide.get_score_dict(inputs[i], response, info)
            except Exception:
                if strict_score_dict:
                    raise
                score_dict = None
        out.append(BundleEval(reward=float(reward), feedback=str(feedback),
                              score_dict=score_dict))
    return sum(ev.reward for ev in out) / len(out), out


def load_and_evaluate(
    task_id: str,
    tasks_root: str,
    *,
    eval_kwargs: Optional[Dict[str, Any]] = None,
    max_examples: int = 1,
    strict_score_dict: bool = False,
) -> Tuple[Dict[str, Any], float, List[BundleEval]]:
    """One-call helper: load the bundle for ``task_id``, then evaluate it."""
    bundle = load_task_bundle(normalize_task_id(task_id), tasks_root,
                              eval_kwargs=eval_kwargs)
    mean_reward, evals = evaluate_bundle(
        bundle, max_examples=max_examples, strict_score_dict=strict_score_dict)
    return bundle, mean_reward, evals
