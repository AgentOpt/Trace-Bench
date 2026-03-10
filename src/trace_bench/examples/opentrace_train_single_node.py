from __future__ import annotations

from opto import trace
from opto.trainer.guide import Guide


class SingleNodeRegressionGuide(Guide):
    def get_feedback(self, _query, response, reference, **_kwargs):
        response_value = getattr(response, "data", response)
        reference_value = str(reference)
        if ":" in reference_value:
            reference_value = reference_value.split(":", 1)[1].strip()
        try:
            score = -abs(float(response_value) - float(reference_value))
        except Exception:
            score = -1.0
        feedback = f"target={reference_value}"
        return score, feedback


@trace.model
class OpenTraceSingleNodeAgent:
    def __init__(self):
        self.value = trace.node(0, description="An integer to guess", trainable=True)

    def __call__(self, _input):
        return self.emit(self.value)

    @trace.bundle(trainable=True)
    def emit(self, value):
        return value


def build_trace_problem(**_override_eval_kwargs):
    agent = OpenTraceSingleNodeAgent()
    guide = SingleNodeRegressionGuide()
    train_dataset = dict(inputs=[None], infos=["Correct answer is: 3"])
    optimizer_kwargs = dict(
        objective="Guess the scalar value hidden in the training info.",
        memory_size=5,
    )
    return dict(
        param=agent,
        guide=guide,
        train_dataset=train_dataset,
        optimizer_kwargs=optimizer_kwargs,
        metadata=dict(
            benchmark="trace_examples",
            entry="OpenTraceSingleNodeAgent",
            source_example="OpenTrace/examples/train_single_node.py",
        ),
    )


__all__ = ["build_trace_problem", "OpenTraceSingleNodeAgent"]
