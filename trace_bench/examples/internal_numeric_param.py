from __future__ import annotations

from opto import trace
from opto.trainer.guide import Guide


class NumericGuide(Guide):
    def get_feedback(self, _query, response, reference, **_kwargs):
        try:
            score = -abs(float(response) - float(reference))
        except Exception:
            score = -1.0
        feedback = f"target={reference}"
        return score, feedback


@trace.model
class NumericParamAgent:
    def __init__(self):
        self.value = trace.node(0.0, trainable=True)

    def __call__(self, _input):
        return self.emit(self.value)

    @trace.bundle(trainable=True)
    def emit(self, value):
        return value


def build_trace_problem(**_override_eval_kwargs):
    agent = NumericParamAgent()
    guide = NumericGuide()
    train_dataset = dict(inputs=[None], infos=[3.0])
    optimizer_kwargs = dict(objective="Match the numeric target value.", memory_size=5)
    return dict(
        param=agent,
        guide=guide,
        train_dataset=train_dataset,
        optimizer_kwargs=optimizer_kwargs,
        metadata=dict(benchmark="internal", entry="NumericParamAgent"),
    )


__all__ = ["build_trace_problem", "NumericParamAgent"]
