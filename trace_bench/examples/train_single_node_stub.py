from __future__ import annotations

from opto import trace
from opto.trainer.guide import Guide


class RegressionGuide(Guide):
    def get_feedback(self, query, response, reference, **kwargs):
        try:
            score = -abs(float(response) - float(reference))
        except Exception:
            score = -1.0
        feedback = f"target={reference}"
        return score, feedback


@trace.model
class SingleNodeAgent:
    def __init__(self):
        self.guess = trace.node(0.0, trainable=True)

    def __call__(self, _input):
        return self.output(self.guess)

    @trace.bundle(trainable=True)
    def output(self, guess):
        return guess


def build_trace_problem(**override_eval_kwargs):
    agent = SingleNodeAgent()
    guide = RegressionGuide()
    train_dataset = dict(
        inputs=[None],
        infos=[3.0],
    )
    optimizer_kwargs = dict(
        objective="Match the target scalar value.",
        memory_size=5,
    )
    return dict(
        param=agent,
        guide=guide,
        train_dataset=train_dataset,
        optimizer_kwargs=optimizer_kwargs,
        metadata=dict(benchmark="example", entry="SingleNodeAgent"),
    )


__all__ = ["build_trace_problem", "SingleNodeAgent"]
