from __future__ import annotations

from opto import trace
from opto.trainer.guide import Guide


class SumGuide(Guide):
    def get_feedback(self, _query, response, reference, **_kwargs):
        try:
            score = -abs(float(response) - float(reference))
        except Exception:
            score = -1.0
        feedback = f"target={reference}"
        return score, feedback


@trace.model
class MultiParamAgent:
    def __init__(self):
        self.a = trace.node(1.0, trainable=True)
        self.b = trace.node(1.0, trainable=True)

    def __call__(self, _input):
        return self.combine(self.a, self.b)

    @trace.bundle(trainable=True)
    def combine(self, a, b):
        return float(getattr(a, "data", a)) + float(getattr(b, "data", b))


def build_trace_problem(**_override_eval_kwargs):
    agent = MultiParamAgent()
    guide = SumGuide()
    train_dataset = dict(inputs=[None], infos=[3.0])
    optimizer_kwargs = dict(objective="Make a+b match the target value.", memory_size=5)
    return dict(
        param=agent,
        guide=guide,
        train_dataset=train_dataset,
        optimizer_kwargs=optimizer_kwargs,
        metadata=dict(benchmark="internal", entry="MultiParamAgent"),
    )


__all__ = ["build_trace_problem", "MultiParamAgent"]
