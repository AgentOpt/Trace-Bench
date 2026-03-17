# Adding a Task

This guide covers how to add a new benchmark task to Trace-Bench.

## Task ID Convention

Every task has an ID in `suite:name` format:

| Suite | Pattern | Location |
|-------|---------|----------|
| `llm4ad` | `llm4ad:task_name` | `benchmarks/LLM4AD/benchmark_tasks/<task_name>/` |
| `trace_examples` | `trace_examples:name` | `src/trace_bench/examples/<name>.py` |
| `internal` | `internal:name` | `src/trace_bench/examples/<name>.py` |
| `veribench` | `veribench:name` | Via adapter (see [Adding a Benchmark](adding-benchmark.md)) |

If a task ID has no colon prefix, it defaults to `llm4ad:`.

## Adding an LLM4AD Task

### 1. Create the Directory

```
benchmarks/LLM4AD/benchmark_tasks/<task_name>/
    __init__.py      # must export build_trace_problem
    get_instance.py  # (optional) data loading helpers
```

### 2. Implement build_trace_problem

The `__init__.py` must export a `build_trace_problem(**eval_kwargs)` function:

```python
from opto import trace
from opto.trainer.guide import Guide

class MyGuide(Guide):
    def get_feedback(self, query, response, reference, **kwargs):
        # score in [0, 1], feedback as string
        score = ...
        feedback = ...
        return score, feedback

@trace.model
class MyAgent:
    def __init__(self):
        self.code = trace.node("# initial code", trainable=True)

    def __call__(self, problem_input):
        return self.solve(self.code, problem_input)

    @trace.bundle(trainable=True)
    def solve(self, code, problem_input):
        # agent logic
        return code

def build_trace_problem(**eval_kwargs):
    return dict(
        param=MyAgent(),
        guide=MyGuide(),
        train_dataset=dict(
            inputs=["problem 1", "problem 2"],
            infos=["expected 1", "expected 2"],
        ),
        optimizer_kwargs=dict(
            objective="Solve the optimization problem.",
            memory_size=5,
        ),
        metadata=dict(
            benchmark="llm4ad",
            entry="MyAgent",
        ),
    )
```

### 3. Register in the Index

Add an entry to `benchmarks/LLM4AD/benchmark_tasks/index.json`:

```json
{
    "key": "my_task",
    "module": "my_task"
}
```

If `index.json` does not exist, discovery falls back to scanning directories -- any subdirectory with an `__init__.py` is treated as a task.

### 4. Validate

```bash
trace-bench validate --config configs/smoke.yaml --root benchmarks/LLM4AD/benchmark_tasks --strict
```

Check for:
- `[OK] llm4ad:my_task` -- bundle loads and has trainable parameters
- `[FAIL]` -- error message will indicate missing keys or import failures

### 5. Run

Reference the task in a config:

```yaml
mode: stub
seeds: [123]
tasks:
  - id: llm4ad:my_task
trainers:
  - id: PrioritySearch
    params_variants:
      - ps_steps: 2
        ps_batches: 1
```

```bash
trace-bench run --config my_config.yaml --root benchmarks/LLM4AD/benchmark_tasks
```

## Adding an Example/Internal Task

For lightweight tasks (testing, demos):

1. Create `src/trace_bench/examples/<name>.py` with `build_trace_problem()`.
2. Register it in `registry.py`:
   - For `trace_examples:` suite: add to `discover_trace_examples()`.
   - For `internal:` suite: add to the `_INTERNAL_TASKS` dict.

## Bundle Requirements

The dict returned by `build_trace_problem()` must contain:

| Key | Required | Description |
|-----|----------|-------------|
| `param` | yes | Agent/model with trainable parameters |
| `guide` | yes | Guide that returns `(score, feedback)` |
| `train_dataset` | yes | Dict with `inputs` and `infos` lists |
| `optimizer_kwargs` | yes | Dict with at least `objective` |
| `metadata` | yes | Dict with `benchmark` and `entry` |
| `guide_kwargs` | no | Extra guide configuration |
| `logger_kwargs` | no | Extra logger configuration |

## eval_kwargs

Tasks can accept `eval_kwargs` from the config to customize evaluation behavior:

```yaml
tasks:
  - id: llm4ad:my_task
    eval_kwargs:
      difficulty: hard
      max_samples: 100
```

These are passed as keyword arguments to `build_trace_problem(**eval_kwargs)`.

## Checklist

- `build_trace_problem()` returns all required keys
- `train_dataset["inputs"]` and `train_dataset["infos"]` are same length
- At least one trainable parameter exists in `param`
- Task appears in `trace-bench list-tasks --bench llm4ad`

## Troubleshooting

- **ImportError**: ensure your module is under `benchmarks/LLM4AD/benchmark_tasks/` and exports `build_trace_problem`.
- **No trainables**: add a `trace.node(..., trainable=True)` or `@trace.bundle(trainable=True)`.
- **Missing objective**: ensure `optimizer_kwargs` contains a descriptive `objective` string.

## Notebooks

- `notebooks/03_task_coverage.ipynb` -- explore task properties and coverage across suites

## Further Reading

- [Agents and Tasks](agents-and-tasks.md) -- conceptual overview and decision guide
- [Adding an Agent](adding-agent.md) -- creating the optimizable agent itself
- [Adding a Benchmark](adding-benchmark.md) -- integrating a full external benchmark suite
- [Task Inventory](task-inventory.md) -- browse all available tasks by suite
