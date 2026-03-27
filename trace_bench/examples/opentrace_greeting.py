from __future__ import annotations

from opto import trace
from opto.trainer.guide import Guide


class GreetingFeedbackGuide(Guide):
    def get_feedback(self, query: str, response: str, reference: str, **_kwargs):
        value = str(getattr(response, "data", response))
        expected = "Hola" if reference == "es" else "Hello"
        score = 1.0 if expected in value else 0.0
        feedback = "Correct" if score == 1.0 else f"Expected language={reference}"
        return score, feedback


@trace.model
class OpenTraceGreetingAgent:
    def __init__(self):
        self.instruct1 = trace.node("Decide the language", trainable=True)
        self.instruct2 = trace.node("Extract name if present", trainable=True)

    def __call__(self, user_query: str):
        lang = self.decide_lang(self.instruct1, user_query)
        user_name = self.extract_name(self.instruct2, user_query)
        return self.greet(lang, user_name)

    @trace.bundle(trainable=True)
    def decide_lang(self, _instruction, user_query: str):
        text = str(user_query).lower()
        return "es" if any(token in text for token in ("hola", "soy", "juan")) else "en"

    @trace.bundle(trainable=True)
    def extract_name(self, _instruction, user_query: str):
        text = str(user_query).strip().strip(".!?")
        lowered = text.lower()
        if "soy " in lowered:
            return text[lowered.rfind("soy ") + 4 :].strip()
        if "i am " in lowered:
            return text[lowered.rfind("i am ") + 5 :].strip()
        return text.split()[-1] if text else "Friend"

    @trace.bundle(trainable=True)
    def greet(self, lang: str, user_name: str):
        greeting = "Hola" if str(lang).lower() == "es" else "Hello"
        return f"{greeting}, {user_name}!"


def build_trace_problem(**_override_eval_kwargs):
    agent = OpenTraceGreetingAgent()
    guide = GreetingFeedbackGuide()
    train_dataset = dict(
        inputs=["Hola, soy Juan.", "Hello, I am Sam."],
        infos=["es", "en"],
    )
    optimizer_kwargs = dict(
        objective="Produce a correctly localized greeting from the user query.",
        memory_size=5,
    )
    return dict(
        param=agent,
        guide=guide,
        train_dataset=train_dataset,
        optimizer_kwargs=optimizer_kwargs,
        metadata=dict(
            benchmark="trace_examples",
            entry="OpenTraceGreetingAgent",
            source_example="OpenTrace/examples/greeting.py",
        ),
    )


__all__ = ["build_trace_problem", "OpenTraceGreetingAgent"]
