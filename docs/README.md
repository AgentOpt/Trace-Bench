# Trace-Bench Documentation

Trace-Bench is a benchmarking framework for evaluating LLM-based optimization algorithms built on [OpenTrace](https://github.com/AgentOpt/Trace). It provides a reproducible harness that pairs **tasks** (benchmark problems) with **trainers** (optimization algorithms), runs them across seeds, and produces structured artifacts for comparison.

## How to Use This Documentation

Start with **Overview** to learn the core concepts and how a run is structured. If you want to execute experiments, read **Running Experiments** and **Config Reference** next. For UI workflows, jump to **UI Guide**. For extension points, follow **Adding a Task**, **Adding an Agent**, **Adding a Trainer**, and **Adding a Benchmark**.

## Validation Evidence

See [Validation Evidence](validation.md) for the exact commands and transcripts used to validate the repository after the layout changes.

## Quick Start

```bash
pip install -e .
trace-bench list-tasks
trace-bench run --config configs/smoke.yaml --runs-dir runs
```

## Table of Contents

| Page | Description |
|------|-------------|
| [Overview](overview.md) | What Trace-Bench is, concept glossary, and pointers to intro notebooks |
| [Running Experiments](running-experiments.md) | CLI reference, fair comparisons, reading results |
| [Agents and Tasks](agents-and-tasks.md) | Technical distinction between agents/models and tasks |
| [Adding an Agent](adding-agent.md) | How to optimize a new agent with Trace-Bench |
| [Adding a Task](adding-task.md) | How to contribute a new benchmark task |
| [Adding a Trainer](adding-trainer.md) | How trainers are discovered and registered |
| [Adding a Benchmark](adding-benchmark.md) | How to integrate an external benchmark suite |
| [UI Guide](ui-guide.md) | Gradio UI tabs, workflows, and screenshots |
| [MLflow Integration](mlflow.md) | Enabling MLflow, what is logged |
| [Config Reference](config-reference.md) | YAML schema, matrix expansion, resume modes, output artifacts |
| [Result Analysis](result-analysis.md) | Reading results via CLI, UI, and Python |

## Notebooks

| Notebook | Topic |
|----------|-------|
| `notebooks/01_quick_start.ipynb` | First run in under 5 minutes |
| `notebooks/02_api_walkthrough.ipynb` | Python API and config objects |
| `notebooks/03_task_coverage.ipynb` | Exploring available tasks |
| `notebooks/04_gradio_ui.ipynb` | Interactive results dashboard |
| `notebooks/05_full_benchmark.ipynb` | Running a full benchmark matrix |
| `notebooks/06_multiobjective_convex.ipynb` | Multi-objective optimization (convex) |
| `notebooks/07_multiobjective_bbeh.ipynb` | Multi-objective optimization (BBEH) |
| `notebooks/08_multiobjective_gsm8k.ipynb` | Multi-objective optimization (GSM8K) |

## Project Layout

```
src/trace_bench/       Python package (CLI, runner, registry, config)
benchmarks/            Benchmark suites (LLM4AD, KernelBench, Veribench)
configs/               YAML run configurations
notebooks/             Jupyter notebooks with worked examples
runs/                  Default output directory (created on first run)
```
