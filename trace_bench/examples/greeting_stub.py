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


class _GreetingDSPyProgram:
    def __init__(self, greeting: str = "Hello"):
        self.greeting = greeting

    def __call__(self, user_query: str) -> str:
        name = user_query.split()[-1].strip("!.?")
        return f"{self.greeting}, {name}!"

    # DSPy teleprompters expect a resettable/copyable student program.
    def deepcopy(self):
        return _GreetingDSPyProgram(self.greeting)

    def reset_copy(self):
        return self.deepcopy()

    def predictors(self):
        return []

    def named_predictors(self):
        return []


def _greeting_score(candidate_greeting: str) -> tuple[float, str]:
    guide = ExactMatchGuide()
    query = "Hello I am Sam"
    response = f"{candidate_greeting}, Sam!"
    return guide.get_feedback(query, response, "Hello, Sam!")


def _apply_greeting(candidate_greeting: str, bundle: dict) -> None:
    bundle["param"].greeting._set(candidate_greeting)


def _serialize_greeting_state(value) -> dict:
    if isinstance(value, _GreetingDSPyProgram):
        greeting = value.greeting
    else:
        greeting = value
    return {"kind": "GreetingState", "greeting": greeting}


def _build_dspy_program():
    return _GreetingDSPyProgram("Hello")


def _evaluate_dspy(program, trainset):
    query = trainset[0]["input"]
    expected = trainset[0]["label"]
    score, _ = ExactMatchGuide().get_feedback(query, program(query), expected)
    return score


def _sync_dspy_to_bundle(program, bundle):
    _apply_greeting(program.greeting, bundle)


def _loss_factory(_bundle):
    class _Loss:
        def __call__(self, variable):
            class _Value:
                def __init__(self, target):
                    self.target = target
                def backward(self):
                    return None
            return _Value(variable)
    return _Loss()


def _evaluate_text(candidate_text):
    return _greeting_score(candidate_text)[0]


def _feedback_text(candidate_text):
    return _greeting_score(candidate_text)[1]


def _initial_program(_bundle):
    return "Hello"


def _evaluate_program(candidate_program: str):
    score, feedback = _greeting_score(candidate_program)
    return {"score": score, "feedback": feedback, "artifacts": {"candidate": candidate_program}}


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
        frameworks={
            "dspy": {
                "program_factory": _build_dspy_program,
                "metric": lambda *args, **kwargs: None,
                "to_trainset": lambda dataset: [{"input": dataset["inputs"][0], "label": dataset["infos"][0]}],
                "evaluate": _evaluate_dspy,
                "sync_to_bundle": _sync_dspy_to_bundle,
                "state_serializer": _serialize_greeting_state,
            },
            "textgrad": {
                "initial_text": "Hello",
                "role_description": "greeting prefix",
                "loss_fn_factory": _loss_factory,
                "evaluate": _evaluate_text,
                "feedback": _feedback_text,
                "apply_update": lambda candidate, b=None: _apply_greeting(candidate, b or {"param": agent}),
                "state_serializer": _serialize_greeting_state,
            },
            "openevolve": {
                "initial_program_factory": _initial_program,
                "evaluate_program": _evaluate_program,
                "apply_candidate": lambda candidate, b: _apply_greeting(candidate, b),
                "state_serializer": _serialize_greeting_state,
            },
        },
    )


__all__ = ["build_trace_problem", "GreetingAgent"]
