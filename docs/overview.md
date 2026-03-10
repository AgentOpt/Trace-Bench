# Overview

## What Trace-Bench Is

Trace-Bench is a reproducible benchmarking harness for LLM-based optimization. It takes a **config** that declares which tasks to run, which trainers (optimization algorithms) to use, and which seeds to repeat, then executes every combination and writes structured results.

Trace-Bench builds on [OpenTrace](https://github.com/microsoft/Trace) (`opto`). If you know how to make a parameter trainable in Trace, you already know 90% of what you need.

## What Trace-Bench Is Not

- It is **not** an optimization library. Optimization lives in `opto`. Trace-Bench only orchestrates and records.
- It is **not** a model serving framework. LLM calls go through the provider configured in your environment or config.
- It does **not** train neural networks. "Training" here means iterative prompt/parameter optimization via Trace.

## Concept Glossary

| Term | Definition |
|------|------------|
| **Benchmark / Suite** | A collection of related tasks from one source. Examples: `llm4ad`, `veribench`, `trace_examples`, `internal`. |
| **Task** | A single optimization problem wrapped as a bundle. Identified by `suite:name` (e.g. `llm4ad:admissible_set`). |
| **Agent / Model** | The object being optimized. In Trace terms, this is the `param` -- a `@trace.model` class or `ParameterNode` with trainable parts. |
| **Trainer** | An optimization algorithm from `opto.trainer.algorithms` or `opto.features`. It drives the optimization loop by calling `train()`. |
| **Optimizer** | The inner optimizer used by a trainer (e.g. `OptoPrime`). Configured via `optimizer` and `optimizer_kwargs` in the trainer config. |
| **Guide** | An evaluator that scores agent outputs and returns `(score, feedback)`. Subclass of `opto.trainer.guide.Guide`. |
| **Logger** | Records training events (e.g. `ConsoleLogger`, `TensorboardLogger`, `WandbLogger`). Configurable per-trainer or via CLI `--logger`. |
| **Run** | One execution of a config. Has a deterministic `run_id` derived from the config snapshot. Contains multiple jobs. |
| **Job** | One (task, trainer, params_variant, seed) combination within a run. Has a unique `job_id`. |
| **Artifacts** | Files produced by a run: `results.csv`, `summary.json`, `leaderboard.csv`, per-job state snapshots, event logs. |

## How a Run Works

1. The config is loaded and the **matrix** is expanded: every task x trainer x params_variant x seed = one job.
2. A deterministic `run_id` is computed from the config snapshot (same config always produces the same run_id).
3. Jobs execute (sequentially or with `max_workers` threads). Each job loads the task bundle, instantiates the trainer, and calls `trainer.train()`.
4. Results are written incrementally. After all jobs finish, a summary and leaderboard are generated.

## Available Suites

| Suite | Description | Task count |
|-------|-------------|------------|
| `llm4ad` | Algorithm design tasks from the LLM4AD benchmark | ~30+ tasks |
| `veribench` | Python-to-Lean4 translation tasks from VeriBench | ~100+ tasks |
| `trace_examples` | Small example tasks shipped with Trace-Bench | 4 tasks |
| `internal` | Synthetic tasks for testing and validation | 7 tasks |

Use `trace-bench list-tasks --bench <suite>` to see what is available in your installation.

## Config at a Glance

A minimal config file looks like this:

```yaml
mode: stub           # "stub" uses DummyLLM; "real" calls your LLM
seeds: [123]
tasks:
  - id: internal:numeric_param
trainers:
  - id: PrioritySearch
    params_variants:
      - ps_steps: 1
        ps_batches: 1
```

See [Running Experiments](running-experiments.md) for the full CLI reference and config options.

## Getting Started

Install the package:

```bash
pip install -e .
```

Verify tasks are discoverable:

```bash
trace-bench list-tasks --root benchmarks/LLM4AD/benchmark_tasks
```

Run the smoke test:

```bash
trace-bench run --config configs/smoke.yaml --root benchmarks/LLM4AD/benchmark_tasks
```

## Notebooks

| Notebook | Use Case |
|----------|----------|
| `notebooks/01_quick_start.ipynb` | End-to-end first run |
| `notebooks/02_api_walkthrough.ipynb` | Python API, `RunConfig`, `BenchRunner`, and result inspection |
| `notebooks/03_task_coverage.ipynb` | Exploring available tasks across suites |
| `notebooks/04_gradio_ui.ipynb` | Launching and using the Gradio UI |
| `notebooks/05_full_benchmark.ipynb` | Running a full 65-task benchmark matrix |
| `notebooks/06_multiobjective_convex.ipynb` | Multi-objective optimization (convex function) |
| `notebooks/07_multiobjective_bbeh.ipynb` | Multi-objective optimization (BIG-Bench Hard) |
| `notebooks/08_multiobjective_gsm8k.ipynb` | Multi-objective optimization (GSM8K math) |

## Next Steps

- [Running Experiments](running-experiments.md) — CLI reference and fair comparison workflow
- [UI Guide](ui-guide.md) — Launch and use the Gradio dashboard
- [Config Reference](config-reference.md) — YAML schema, matrix expansion, output artifacts
- [Result Analysis](result-analysis.md) — Reading and comparing experiment results
