from __future__ import annotations

from opto import trace
from opto.trainer.guide import Guide


class CodeExactGuide(Guide):
    def get_feedback(self, _query, response, reference, **_kwargs):
        score = 1.0 if response == reference else 0.0
        feedback = "Correct" if score == 1.0 else "Mismatch"
        return score, feedback


@trace.model
class CodeParamAgent:
    def __init__(self):
        self.code = trace.node("def f(x): return x", trainable=True)

    def __call__(self, _input):
        return self.emit(self.code)

    @trace.bundle(trainable=True)
    def emit(self, code):
        return code


def build_trace_problem(**_override_eval_kwargs):
    agent = CodeParamAgent()
    guide = CodeExactGuide()
    train_dataset = dict(inputs=[None], infos=["def f(x): return x"])
    optimizer_kwargs = dict(objective="Match the target code exactly.", memory_size=5)
    return dict(
        param=agent,
        guide=guide,
        train_dataset=train_dataset,
        optimizer_kwargs=optimizer_kwargs,
        metadata=dict(benchmark="internal", entry="CodeParamAgent"),
    )


__all__ = ["build_trace_problem", "CodeParamAgent"]
