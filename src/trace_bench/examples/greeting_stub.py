from __future__ import annotations

from opto import trace
from opto.trainer.guide import Guide


class ExactMatchGuide(Guide):
    def get_feedback(self, query: str, response: str, reference: str, **kwargs):
        score = 1.0 if response == reference else 0.0
        feedback = "Correct" if score == 1.0 else f"Expected: {reference}"
        return score, feedback


@trace.model
class GreetingAgent:
    def __init__(self):
        self.greeting = trace.node("Hello", trainable=True)

    def __call__(self, user_query: str):
        name = user_query.split()[-1].strip("!.?")
        return self.compose(self.greeting, name)

    @trace.bundle(trainable=True)
    def compose(self, greeting, name: str):
        greeting_value = getattr(greeting, "data", greeting)
        return f"{greeting_value}, {name}!"


def build_trace_problem(**override_eval_kwargs):
    agent = GreetingAgent()
    guide = ExactMatchGuide()
    train_dataset = dict(
        inputs=["Hello I am Sam"],
        infos=["Hello, Sam!"],
    )
    optimizer_kwargs = dict(
        objective="Generate a correct greeting using the name from the query.",
        memory_size=5,
    )
    return dict(
        param=agent,
        guide=guide,
        train_dataset=train_dataset,
        optimizer_kwargs=optimizer_kwargs,
        metadata=dict(benchmark="example", entry="GreetingAgent"),
    )


__all__ = ["build_trace_problem", "GreetingAgent"]
