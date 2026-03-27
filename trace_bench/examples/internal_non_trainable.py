from __future__ import annotations

from opto import trace
from opto.trainer.guide import Guide


class NoTrainGuide(Guide):
    def get_feedback(self, _query, response, reference, **_kwargs):
        score = 1.0 if response == reference else 0.0
        feedback = "Correct" if score == 1.0 else "Mismatch"
        return score, feedback


@trace.model
class NonTrainableAgent:
    def __init__(self):
        self.value = trace.node("fixed", trainable=False)

    def __call__(self, _input):
        return self.emit(self.value)

    @trace.bundle(trainable=False)
    def emit(self, value):
        return value


def build_trace_problem(**_override_eval_kwargs):
    agent = NonTrainableAgent()
    guide = NoTrainGuide()
    train_dataset = dict(inputs=[None], infos=["fixed"])
    optimizer_kwargs = dict(objective="This should fail due to no trainables.", memory_size=1)
    return dict(
        param=agent,
        guide=guide,
        train_dataset=train_dataset,
        optimizer_kwargs=optimizer_kwargs,
        metadata=dict(benchmark="internal", entry="NonTrainableAgent"),
    )


__all__ = ["build_trace_problem", "NonTrainableAgent"]
