class DummyParam:
    def __init__(self, value="Hi"):
        self.data = value
        self.trainable = True
    def _set(self, value):
        self.data = value

class DummyGuide:
    def __call__(self, query, response, reference):
        score = 1.0 if response == reference else 0.0
        feedback = "Correct" if score == 1.0 else f"Expected: {reference}"
        return score, feedback

def make_bundle():
    param = DummyParam("Hi")
    guide = DummyGuide()
    train_dataset = {"inputs": ["Hello I am Sam"], "infos": ["Hello, Sam!"]}

    def score_for_greeting(candidate_greeting: str):
        response = f"{candidate_greeting}, Sam!"
        return guide("Hello I am Sam", response, "Hello, Sam!")

    class Program:
        def __init__(self, greeting="Hi"):
            self.greeting = greeting
        def __call__(self, query):
            name = query.split()[-1].strip("!.?")
            return f"{self.greeting}, {name}!"

    bundle = {
        "param": param,
        "guide": guide,
        "train_dataset": train_dataset,
        "optimizer_kwargs": {"objective": "Improve greeting", "memory_size": 5},
        "metadata": {"benchmark": "test", "entry": "dummy"},
        "frameworks": {
            "dspy": {
                "program_factory": lambda: Program("Hi"),
                "metric": lambda *args, **kwargs: None,
                "to_trainset": lambda dataset: [{"input": dataset["inputs"][0], "label": dataset["infos"][0]}],
                "evaluate": lambda program, trainset: score_for_greeting(program.greeting)[0],
                "sync_to_bundle": lambda program, bundle: bundle["param"]._set(program.greeting),
                "state_serializer": lambda value: {"kind": "Program", "greeting": getattr(value, "greeting", value)},
            },
            "textgrad": {
                "initial_text": "Hi",
                "role_description": "greeting prefix",
                "evaluate": lambda candidate: score_for_greeting(candidate)[0],
                "feedback": lambda candidate: score_for_greeting(candidate)[1],
                "apply_update": lambda candidate: param._set(candidate),
                "state_serializer": lambda value: {"kind": "TextState", "greeting": value},
            },
            "openevolve": {
                "initial_program": "Hi",
                "evaluate_program": lambda candidate: {
                    "score": score_for_greeting(candidate)[0],
                    "feedback": score_for_greeting(candidate)[1],
                    "artifacts": {"candidate": candidate},
                },
                "apply_candidate": lambda candidate, bundle: bundle["param"]._set(candidate),
                "state_serializer": lambda value: {"kind": "ProgramText", "greeting": value},
            },
        },
    }
    return bundle
