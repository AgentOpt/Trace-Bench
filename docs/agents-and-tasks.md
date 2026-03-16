# Agents and Tasks

This page explains the distinction between **agents/models** and **tasks** in Trace-Bench, and how they map to OpenTrace concepts.

## OpenTrace Concepts

In OpenTrace (`opto`), optimization works with these building blocks:

- **Model / Agent** (`param`): The object being optimized. Decorated with `@trace.model`, it contains one or more `trace.node(trainable=True)` parameters and/or `@trace.bundle(trainable=True)` methods.
- **Guide**: Evaluates the agent's output and returns `(score, feedback)`. Subclass of `opto.trainer.guide.Guide`.
- **Optimizer**: Updates trainable parameters based on feedback (e.g. `OptoPrime`).
- **Trainer**: Drives the optimization loop -- loads data, calls the agent, gets feedback from the guide, and updates via the optimizer.

## Trace-Bench Mapping

Trace-Bench wraps these into a **task bundle** -- a dict returned by `build_trace_problem()`:

```python
{
    "param":             agent,           # the model/agent with trainable parts
    "guide":             guide,           # evaluator
    "train_dataset":     dataset,         # {"inputs": [...], "infos": [...]}
    "optimizer_kwargs":  optimizer_kwargs, # {"objective": "...", "memory_size": 5}
    "metadata":          metadata,        # {"benchmark": "...", "entry": "..."}
}
```

The trainer and optimizer are **not** part of the bundle. They are configured separately in the run config, so the same task can be evaluated with different optimization strategies.

## Bundle Schema Notes

Required keys:

- `param` -- agent/model with trainable components
- `guide` -- evaluator (returns score + feedback)
- `train_dataset` -- `{"inputs": [...], "infos": [...]}` aligned by index
- `optimizer_kwargs` -- hints (objective, memory_size, etc.)
- `metadata` -- benchmark and task identity

Optional keys you may include:

- `guide_kwargs` / `logger_kwargs`
- `optimizer` (override optimizer class name)
- `objective_config` (for multi-objective tasks)

Trace-Bench does not inspect the agent logic; it only requires a valid bundle dict.

## What Is an Agent?

An agent is a `@trace.model` class with trainable parameters. The simplest agent has a single trainable string node:

```python
@trace.model
class GreetingAgent:
    def __init__(self):
        self.greeting = trace.node("Hello", trainable=True)

    def __call__(self, user_query: str):
        name = user_query.split()[-1].strip("!.?")
        return self.compose(self.greeting, name)

    @trace.bundle(trainable=True)
    def compose(self, greeting, name: str):
        return f"{greeting}, {name}!"
```

The trainable parts are:
- `self.greeting` -- a `ParameterNode` (the greeting word)
- `self.compose` -- a `@trace.bundle` (the composition logic)

During optimization, the trainer/optimizer will propose changes to these trainable components.

## Common Pitfalls

- **No trainables**: if `param` has no trainable nodes, strict validation fails.
- **Mismatched dataset**: `inputs` and `infos` must be same length.
- **Missing metadata**: include at least `benchmark` and `entry` to make results traceable.

## What Is a Task?

A **task** is a complete benchmark problem packaged as a bundle. It bundles an agent, a guide, training data, and optimizer hints into one dict. Each task:

- Has a unique ID in `suite:name` format (e.g. `llm4ad:admissible_set`, `trace_examples:greeting_stub`)
- Lives in a module that exports `build_trace_problem(**eval_kwargs) -> dict`
- May wrap a reusable agent class or define a single trainable parameter

A task may be simple (one trainable string) or complex (a multi-parameter agent with code generation). The benchmarking framework does not care -- it only needs the bundle dict.

## Agent vs Task: Summary

| Aspect | Agent / Model | Task |
|--------|--------------|------|
| What it is | A `@trace.model` with trainable parts | A complete benchmark problem |
| Configured by | Part of the task bundle (`param` key) | Config file (`tasks[]`) |
| Reusable across tasks? | Yes, same agent class in multiple tasks | No, each task is a unique problem |
| Contains evaluation logic? | No | Yes (guide + dataset + metadata) |

## Config Mapping

In a YAML config:

```yaml
tasks:
  - id: trace_examples:greeting_stub    # selects the task (which includes the agent)

trainers:
  - id: PrioritySearch                  # optimization algorithm (independent of task)
    optimizer: OptoPrime                 # inner optimizer
    guide: null                         # override guide (null = use task's guide)
```

The `trainers[].guide` field can override the task's built-in guide, but in most cases you leave it unset and let the task provide its own evaluator.

## Further Reading

- [Adding an Agent](adding-agent.md) -- step-by-step guide to creating a new optimizable agent
- [Adding a Task](adding-task.md) -- how to package an agent into a benchmark task
