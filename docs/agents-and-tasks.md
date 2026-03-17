# Agents and Tasks

This page explains the distinction between **agents/models** and **tasks** in Trace-Bench, and how they map to OpenTrace concepts. Understanding this distinction is essential for working with the framework.

---

## OpenTrace Concept Map

In OpenTrace (`opto`), optimization works with four building blocks. Each has a
specific role:

| OpenTrace Concept | Role | Trace-Bench Equivalent |
|-------------------|------|----------------------|
| **Agent / Model** (`param`) | The object being optimized. Decorated with `@trace.model`, it contains trainable parameters and/or methods. | The `param` key in the task bundle. |
| **Guide** | The evaluator. Scores agent outputs and returns `(score, feedback)`. Subclass of `opto.trainer.guide.Guide`. | The `guide` key in the task bundle. |
| **Optimizer** | Updates trainable parameters based on feedback (e.g. `OptoPrime`). | Configured in the run config under `trainers[].optimizer`. |
| **Trainer** | The optimization loop. Loads data, calls the agent, gets feedback from the guide, and updates via the optimizer. | Configured in the run config under `trainers[].id`. |

In short:

- The **agent/model** is what gets improved.
- The **guide** judges how good the agent's output is.
- The **trainer** drives the loop that improves the agent over time.
- The **optimizer** is the algorithm the trainer uses to propose parameter changes.

---

## What Is a Task Bundle?

In Trace-Bench, the runtime/execution unit is a **task bundle** -- a dict
returned by `build_trace_problem()`:

```python
{
    "param":             agent,           # the model/agent with trainable parts
    "guide":             guide,           # evaluator
    "train_dataset":     dataset,         # {"inputs": [...], "infos": [...]}
    "optimizer_kwargs":  optimizer_kwargs, # {"objective": "...", "memory_size": 5}
    "metadata":          metadata,        # {"benchmark": "...", "entry": "..."}
}
```

The trainer and optimizer are **not** part of the bundle. They are configured
separately in the run config, so the same task can be evaluated with different
optimization strategies.

### Required Bundle Keys

| Key | Type | Description |
|-----|------|-------------|
| `param` | object | The agent/model with trainable parameters |
| `guide` | `Guide` | Evaluator that returns `(score, feedback)` |
| `train_dataset` | dict | `{"inputs": [...], "infos": [...]}` aligned by index |
| `optimizer_kwargs` | dict | Passed to the optimizer (e.g. `objective`, `memory_size`) |
| `metadata` | dict | Descriptive info (`benchmark`, `entry`, etc.) |

### Optional Bundle Keys

| Key | Type | Description |
|-----|------|-------------|
| `guide_kwargs` | dict | Extra kwargs passed when constructing the guide |
| `logger_kwargs` | dict | Extra kwargs for the logger |
| `optimizer` | str | Override the default optimizer class name |
| `objective_config` | dict | For multi-objective tasks |

Trace-Bench does not inspect the agent logic; it only requires a valid bundle dict.

---

## Users Select Tasks, Not Agents

> **Important:** Trace-Bench does **not** have a separate "agent selector"
> concept. There is no top-level agent configuration in YAML, CLI, or UI.

Users select a **task ID**, and that task internally wraps:
- A reusable `@trace.model` agent, **or**
- A single trainable parameter / code node.

In a YAML config, you write:

```yaml
tasks:
  - id: trace_examples:greeting_stub    # selects the task (which includes the agent)

trainers:
  - id: PrioritySearch                  # optimization algorithm (independent of task)
    optimizer: OptoPrime                 # inner optimizer
```

The task ID (`trace_examples:greeting_stub`) determines which agent is used.
The trainer/optimizer is configured separately. You never configure the agent
directly in the run config -- it comes from the task bundle.

In the CLI:

```bash
trace-bench run --config my_config.yaml
```

In the UI: you pick tasks from the **Discover Tasks** checkbox list, not from
an agent list. The UI builds the config for you.

---

## What Is an Agent?

An agent is a `@trace.model` class with trainable parameters. The simplest agent has a single trainable string node:

```python
from opto import trace

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
```

The trainable parts are:
- `self.greeting` -- a `ParameterNode` (the greeting word)
- `self.compose` -- a `@trace.bundle` (the composition logic)

During optimization, the trainer/optimizer will propose changes to these trainable components.

---

## What Is a Task?

A **task** is a complete benchmark problem packaged as a bundle. It bundles an agent, a guide, training data, and optimizer hints into one dict. Each task:

- Has a unique ID in `suite:name` format (e.g. `llm4ad:admissible_set`, `trace_examples:greeting_stub`)
- Lives in a module that exports `build_trace_problem(**eval_kwargs) -> dict`
- May wrap a reusable agent class or define a single trainable parameter

A task may be simple (one trainable string) or complex (a multi-parameter agent with code generation). The benchmarking framework does not care -- it only needs the bundle dict.

---

## Example: Same Agent Reused Across Multiple Tasks

A single agent class can appear in multiple tasks, each with different data,
guides, or objectives. Here, `GreetingAgent` is reused in two tasks:

**Task A -- English greetings** (`greeting_english.py`):

```python
from my_agents import GreetingAgent

def build_trace_problem(**kwargs):
    return dict(
        param=GreetingAgent(),
        guide=ExactMatchGuide(),
        train_dataset=dict(
            inputs=["Hello I am Sam", "Hi I am Alex"],
            infos=["Hello, Sam!", "Hello, Alex!"],
        ),
        optimizer_kwargs=dict(objective="Produce a correct English greeting."),
        metadata=dict(benchmark="custom", entry="GreetingAgent-EN"),
    )
```

**Task B -- Multilingual greetings** (`greeting_multilingual.py`):

```python
from my_agents import GreetingAgent

def build_trace_problem(**kwargs):
    return dict(
        param=GreetingAgent(),
        guide=MultilingualGuide(),          # different guide
        train_dataset=dict(
            inputs=["Hola soy Juan", "Hello I am Sam"],
            infos=["Hola, Juan!", "Hello, Sam!"],
        ),
        optimizer_kwargs=dict(objective="Produce a localized greeting."),
        metadata=dict(benchmark="custom", entry="GreetingAgent-ML"),
    )
```

Both tasks use the same `GreetingAgent` class, but each is a separate task
with its own ID, guide, dataset, and evaluation criteria. In the config:

```yaml
tasks:
  - id: custom:greeting_english
  - id: custom:greeting_multilingual
```

This is a real pattern in the codebase: `GreetingAgent` appears in both
`trace_examples:greeting_stub` (exact match, single example) and
`trace_examples:opentrace_greeting` (multilingual, two examples).

---

## Decision Guide: "Adding an Agent" vs "Adding a Task"

Use this to decide which guide to follow:

| I want to... | Follow... |
|--------------|-----------|
| Create a new optimizable object (new `@trace.model` class) | [Adding an Agent](adding-agent.md) |
| Wrap an existing agent into a benchmark problem with data and scoring | [Adding a Task](adding-task.md) |
| Integrate an external benchmark suite (e.g. a paper's benchmark set) | [Adding a Benchmark](adding-benchmark.md) |
| Both create a new agent AND wrap it as a task | [Adding an Agent](adding-agent.md) first, then [Adding a Task](adding-task.md) |

**Rule of thumb:**
- If you already have a `@trace.model` class and just need to benchmark it, go to **Adding a Task**.
- If you need to build the optimizable object from scratch, start with **Adding an Agent**.
- If you have a collection of many tasks from one source, see **Adding a Benchmark**.

---

## Agent vs Task: Summary

| Aspect | Agent / Model | Task |
|--------|--------------|------|
| What it is | A `@trace.model` with trainable parts | A complete benchmark problem |
| Configured by | Part of the task bundle (`param` key) | Config file (`tasks[]`) |
| Selected by user? | No -- comes from the task bundle | Yes -- user picks task IDs in config/CLI/UI |
| Reusable across tasks? | Yes, same agent class in multiple tasks | No, each task is a unique problem |
| Contains evaluation logic? | No | Yes (guide + dataset + metadata) |

---

## Common Pitfalls

- **No trainables**: if `param` has no trainable nodes, strict validation fails.
- **Mismatched dataset**: `inputs` and `infos` must be same length.
- **Missing metadata**: include at least `benchmark` and `entry` to make results traceable.
- **Expecting an "agent selector"**: there is no agent selector in Trace-Bench. Users select tasks; the task provides the agent.

---

## Further Reading

- [Adding an Agent](adding-agent.md) -- step-by-step guide to creating a new optimizable agent
- [Adding a Task](adding-task.md) -- how to package an agent into a benchmark task
- [Adding a Benchmark](adding-benchmark.md) -- how to integrate an external benchmark suite
- [Task Inventory](task-inventory.md) -- list of all available tasks by suite
- [Config Reference](config-reference.md) -- YAML schema for specifying tasks and trainers
