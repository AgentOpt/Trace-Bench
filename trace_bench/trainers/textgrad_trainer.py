from __future__ import annotations

from typing import Any, Dict, Optional, Union

from opto import trace

try:
    from opto.optimizers.textgrad import TextGrad as _TraceTextGrad
except Exception as exc:
    raise ImportError("TextGradTrainer requires opto.optimizers.textgrad from the NewTrace fork.") from exc

from trace_bench.trainers._external_utils import (
    apply_parameter_updates,
    collect_trainable_parameters,
    coerce_like,
    resolve_external_trainer_base,
    restore_parameter_values,
    score_model_on_dataset,
    snapshot_parameter_values,
)


_TrainerBase = resolve_external_trainer_base()


class TextGradTrainer(_TrainerBase):
    """Trace-Bench wrapper around the Trace-native TextGrad optimizer from NewTrace."""

    USES_TRACE_OPTIMIZER = False

    def __init__(self, agent: Any, optimizer: Any = None, logger: Any = None, **_kwargs: Any) -> None:
        del optimizer
        self.param = agent
        self.logger = logger

    def _normalize_updates(self, update_dict: Dict[Any, Any]) -> Dict[Any, Any]:
        """Coerce proposed values back to the current parameter types."""
        normalized: Dict[Any, Any] = {}
        for parameter, candidate_value in update_dict.items():
            normalized[parameter] = coerce_like(getattr(parameter, "data"), candidate_value)
        return normalized

    def _standard_optimization_step(self, guide: Any, task_input: Any, task_info: Any, min_score: float) -> tuple[Any, float, Any]:
        """Run one forward/feedback step, preserving Trace execution errors as feedback."""
        try:
            target = self.param(task_input)
            response = getattr(target, "data", target)
            score, feedback = guide(task_input, response, task_info)
            return target, float(score), feedback
        except trace.ExecutionError as exc:
            target = exc.exception_node
            return target, float(min_score), target.create_feedback("full")

    def train(self, guide: Any, train_dataset: Dict[str, Any], *, mode: str = "real", num_epochs: int = 1, batch_size: int = 1, min_score: float = 0.0, validate_dataset: Optional[Dict[str, Any]] = None, ensure_improvement: bool = True, improvement_threshold: float = 0.0, max_tokens: int = 4096, llm: Any = None, verbose: Union[bool, str] = False, **_kwargs: Any) -> Dict[str, Any]:
        """Optimize Trace parameters with the TextGrad optimizer provided by NewTrace."""
        if mode not in {"real", "stub"}:
            raise ValueError("mode must be either 'real' or 'stub'.")
        if num_epochs < 1:
            raise ValueError("num_epochs must be at least 1.")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if mode == "stub":
            return {"status": "ok", "resolved_optimizer": "opto.optimizers.textgrad.TextGrad"}

        parameters = collect_trainable_parameters(self.param)
        inputs = train_dataset.get("inputs") or []
        infos = train_dataset.get("infos") or train_dataset.get("info") or []
        if len(inputs) != len(infos):
            raise ValueError("train_dataset 'inputs' and 'infos' must have the same length.")
        if not inputs:
            raise ValueError("train_dataset must contain at least one example.")

        optimizer = _TraceTextGrad(parameters=parameters, max_tokens=max_tokens, llm=llm)
        for _ in range(num_epochs):
            for start in range(0, len(inputs), batch_size):
                batch_inputs = inputs[start : start + batch_size]
                batch_infos = infos[start : start + batch_size]
                evaluation_dataset = validate_dataset or {"inputs": batch_inputs, "infos": batch_infos}
                optimizer.zero_feedback()
                for task_input, task_info in zip(batch_inputs, batch_infos):
                    target, _score, feedback = self._standard_optimization_step(guide=guide, task_input=task_input, task_info=task_info, min_score=min_score)
                    optimizer.backward(target, feedback)

                proposal = optimizer.step(bypassing=True, verbose=verbose)
                normalized = self._normalize_updates(proposal)
                if not normalized:
                    continue

                snapshot = snapshot_parameter_values(parameters)
                baseline_score: Optional[float] = None
                if ensure_improvement:
                    baseline_score, _ = score_model_on_dataset(agent=self.param, guide=guide, dataset=evaluation_dataset, suppress_exceptions=True)

                apply_parameter_updates(normalized)
                if ensure_improvement and baseline_score is not None:
                    candidate_score, _ = score_model_on_dataset(agent=self.param, guide=guide, dataset=evaluation_dataset, suppress_exceptions=True)
                    if candidate_score < baseline_score + improvement_threshold:
                        restore_parameter_values(snapshot)

        return {"status": "ok", "resolved_optimizer": "opto.optimizers.textgrad.TextGrad"}
