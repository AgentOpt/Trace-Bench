# Adding an Agent

This guide walks through creating a new optimizable agent and packaging it as a Trace-Bench task.

## Prerequisites

- OpenTrace (`opto`) installed (`pip install trace-opt` or `pip install -e ../OpenTrace`)
- Trace-Bench installed: `pip install -e .`

## Step 1: Define the Agent

An agent is a `@trace.model` class. Mark the parts you want the optimizer to improve as trainable.

```python
from opto import trace

@trace.model
class GreetingAgent:
    def __init__(self):
        # A trainable parameter node -- the optimizer can change this string
        self.greeting = trace.node("Hello", trainable=True)

    def __call__(self, user_query: str):
        name = user_query.split()[-1].strip("!.?")
        return self.compose(self.greeting, name)

    @trace.bundle(trainable=True)
    def compose(self, greeting, name: str):
        """Trainable bundle -- the optimizer can rewrite this function's logic."""
        greeting_value = getattr(greeting, "data", greeting)
        return f"{greeting_value}, {name}!"
```

There are two kinds of trainable components:
- **`trace.node(..., trainable=True)`** -- a parameter node holding a value (string, number, code).
- **`@trace.bundle(trainable=True)`** -- a method whose implementation can be rewritten by the optimizer.

## Step 2: Define a Guide

The guide evaluates agent outputs and returns `(score, feedback)`:

```python
from opto.trainer.guide import Guide

class ExactMatchGuide(Guide):
    def get_feedback(self, query: str, response: str, reference: str, **kwargs):
        score = 1.0 if response == reference else 0.0
        feedback = "Correct" if score == 1.0 else f"Expected: {reference}"
        return score, feedback
```

The guide receives:
- `query` -- the input passed to the agent
- `response` -- the agent's output
- `reference` -- the expected answer (from `train_dataset["infos"]`)

## Step 3: Package as a Bundle

Create a `build_trace_problem()` function that returns the required dict:

```python
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
```

### Required Bundle Keys

| Key | Type | Description |
|-----|------|-------------|
| `param` | object | The agent/model with trainable parameters |
| `guide` | `Guide` | Evaluator that returns `(score, feedback)` |
| `train_dataset` | dict | `{"inputs": [...], "infos": [...]}` |
| `optimizer_kwargs` | dict | Passed to the optimizer (e.g. `objective`, `memory_size`) |
| `metadata` | dict | Descriptive info (`benchmark`, `entry`, etc.) |

### Optional Bundle Keys

| Key | Type | Description |
|-----|------|-------------|
| `guide_kwargs` | dict | Extra kwargs passed when constructing the guide |
| `logger_kwargs` | dict | Extra kwargs for the logger |
| `optimizer` | str | Override the default optimizer class name |

## Step 4: Place the File

For a quick example or internal task, place it in `trace_bench/examples/`:

```
trace_bench/examples/my_agent.py
```

Then register it in the `discover_trace_examples()` function in `trace_bench/registry.py`, or use the `internal:` prefix for test tasks.

For a proper benchmark task, see [Adding a Task](adding-task.md).

## Step 5: Validate

```bash
trace-bench validate --config configs/smoke.yaml --root benchmarks/LLM4AD/benchmark_tasks --strict
```

The validator checks that:
- The bundle loads without errors
- All required keys are present
- The `param` has at least one trainable parameter (in strict mode)

## Step 6: Run

Create a config that references your task:

```yaml
mode: stub
seeds: [123]
tasks:
  - id: trace_examples:greeting_stub
trainers:
  - id: PrioritySearch
    params_variants:
      - ps_steps: 1
        ps_batches: 1
```

```bash
trace-bench run --config my_config.yaml --root benchmarks/LLM4AD/benchmark_tasks
```

## Working Example

The complete working example is at `trace_bench/examples/greeting_stub.py`. It demonstrates all the patterns described above in under 50 lines.

## Checklist

- Agent uses `@trace.model`
- At least one trainable component (`trace.node(..., trainable=True)` or `@trace.bundle(trainable=True)`)
- Guide returns `(score, feedback)` with numeric score
- `build_trace_problem()` returns required keys

## Common Errors

- **Guide returns wrong shape**: must return `(score, feedback)` or `(score_dict, feedback)`.
- **No trainables**: strict validation fails if `param` has no trainable nodes.
- **Dataset mismatch**: ensure `inputs` and `infos` lists are aligned by index.

## Further Reading

- [Agents and Tasks](agents-and-tasks.md) -- conceptual overview
- [Adding a Task](adding-task.md) -- packaging for an external benchmark suite
