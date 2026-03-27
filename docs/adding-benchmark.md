# Adding an External Benchmark

Trace-Bench supports integrating external benchmark suites through an **adapter pattern**. VeriBench is the reference implementation.

## The Adapter Pattern

An adapter bridges an external benchmark into Trace-Bench by implementing two functions:

1. **`discover_task_names() -> List[str]`** -- returns a list of task identifiers from the external benchmark.
2. **`build_bundle(task_name, eval_kwargs=None) -> dict`** -- constructs a Trace-compatible bundle for a given task.

The bundle must contain the standard keys: `param`, `guide`, `train_dataset`, `optimizer_kwargs`, `metadata`.

## VeriBench: Reference Implementation

The VeriBench adapter lives at `trace_bench/veribench_adapter.py`. It demonstrates the full pattern.

### Discovery

`discover_task_names()` tries two paths in order:

1. **Entrypoint-based**: imports a configurable entrypoint module and looks for `list_tasks()`, `get_task_list()`, or `TASKS` attributes.
2. **HuggingFace dataset**: falls back to loading the dataset `allenanie/veribench_with_prompts` and extracting task names from filenames.

```python
def discover_task_names() -> List[str]:
    # Path 1: entrypoint module
    try:
        return _discover_from_entrypoint()
    except NotImplementedError:
        pass
    # Path 2: HuggingFace dataset
    return _discover_from_dataset()
```

### Bundle Building

`build_bundle(task_name, eval_kwargs)` also tries two paths:

1. **Entrypoint-based**: calls `build_trace_problem()` or `build_task_bundle()` on the entrypoint module.
2. **Dataset-based**: constructs an agent and guide directly from the HF dataset row.

The dataset-based fallback creates a `@trace.model` agent with a trainable `lean_code` node and a heuristic guide that scores Lean 4 output quality.

### Configuration

The adapter is configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `TRACE_BENCH_VERIBENCH_ROOT` | `benchmarks/Veribench` | Path to VeriBench installation |
| `TRACE_BENCH_VERIBENCH_ENTRYPOINT` | `my_processing_agents.optimize_veribench_agent` | Python module to import |

## Writing a New Adapter

### Step 1: Create the Adapter Module

Create `trace_bench/<name>_adapter.py`:

```python
from typing import Any, Dict, List, Optional

def discover_task_names() -> List[str]:
    """Return all available task names from the external benchmark."""
    # Load your benchmark's task list
    return ["task_a", "task_b", "task_c"]

def build_bundle(task_name: str, eval_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build a Trace-compatible bundle for the given task."""
    eval_kwargs = dict(eval_kwargs or {})

    # Import opto only when building bundles
    import opto.trace as trace
    from opto.trainer.guide import Guide

    # Define agent
    @trace.model
    class MyAgent:
        def __init__(self):
            self.solution = trace.node("initial", trainable=True)
        def __call__(self, task_input):
            return self.solve(self.solution, task_input)
        @trace.bundle(trainable=True)
        def solve(self, solution, task_input):
            return solution

    # Define guide
    class MyGuide(Guide):
        def get_feedback(self, query, response, reference, **kwargs):
            score = ...  # evaluate response
            feedback = ...
            return score, feedback

    return {
        "param": MyAgent(),
        "guide": MyGuide(),
        "train_dataset": {"inputs": [...], "infos": [...]},
        "optimizer_kwargs": {"objective": "...", "memory_size": 5},
        "metadata": {"benchmark": "my_bench", "entry": "MyAgent", "task_name": task_name},
    }
```

### Step 2: Register in the Registry

In `trace_bench/registry.py`, add imports and a discovery function:

```python
from trace_bench.my_bench_adapter import build_bundle as build_mybench_bundle
from trace_bench.my_bench_adapter import discover_task_names as discover_mybench_task_names

def discover_mybench() -> List[TaskSpec]:
    try:
        names = discover_mybench_task_names()
    except NotImplementedError:
        names = []
    return [TaskSpec(id=f"mybench:{name}", suite="mybench", module="mybench_adapter")
            for name in names]
```

### Step 3: Add to Bench Selector

Update `_parse_bench()` in `registry.py` to include your suite:

```python
allowed = {"llm4ad", "trace_examples", "internal", "veribench", "mybench"}
```

And add the discovery call in `discover_tasks()`:

```python
if "mybench" in selected:
    specs.extend(discover_mybench())
```

### Step 4: Wire Up Bundle Loading

In `load_task_bundle()`, add handling for your prefix:

```python
if task_id.startswith("mybench:"):
    task_name = task_id.split(":", 1)[1]
    return build_mybench_bundle(task_name, eval_kwargs=eval_kwargs)
```

### Step 5: Validate and Run

```bash
trace-bench list-tasks --bench mybench --root benchmarks/LLM4AD/benchmark_tasks
trace-bench validate --config my_config.yaml --root benchmarks/LLM4AD/benchmark_tasks --strict
trace-bench run --config my_config.yaml --root benchmarks/LLM4AD/benchmark_tasks
```

## Design Guidelines

- **Lazy imports**: only import heavy dependencies (datasets, torch, etc.) inside `build_bundle`, not at module level. This keeps `list-tasks` fast.
- **Graceful degradation**: if the external benchmark is not installed, `discover_task_names()` should raise `NotImplementedError` with a descriptive message.
- **Stable task names**: task names should be deterministic and stable across runs. Avoid random suffixes or timestamps.
- **Bundle validation**: the registry validates that all required keys (`param`, `guide`, `train_dataset`, `optimizer_kwargs`, `metadata`) are present. Missing keys raise immediately.

## Checklist

- Adapter exposes `discover_task_names()` and `build_bundle()`
- `discover_task_names()` raises `NotImplementedError` when dependencies missing
- `build_bundle()` returns required keys and stable `metadata`
- Registry wired in `discover_tasks()` and `_parse_bench()`

## Further Reading

- [Agents and Tasks](agents-and-tasks.md) -- conceptual overview of the bundle structure
- [Adding a Task](adding-task.md) -- simpler path for single tasks
- Source: `trace_bench/veribench_adapter.py`
