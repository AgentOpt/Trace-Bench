# Trace-Bench
Benchmark to evaluate LLM as an optimizer.

Currently, we are adding problems/domains one folder at a time.

The instructions to run each task are located inside the task folder.

## Quick Start (Runner/CLI)

```bash
# 1) List tasks (LLM4AD + example stubs)
trace-bench list-tasks --root LLM4AD/benchmark_tasks

# 2) Validate a config
trace-bench validate --config configs/smoke.yaml

# 3) Run Stub smoke (deterministic, no keys)
trace-bench run --config configs/smoke.yaml --runs-dir runs

# 4) Run Real smoke (requires OPENROUTER_API_KEY -- see below)
trace-bench run --config configs/smoke_real.yaml --runs-dir runs

# 5) Run M2 coverage (stub mode, 65 tasks x 2 trainers)
trace-bench run --config configs/m2_coverage.yaml --runs-dir runs

# 6) Run tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q

# Launch UI (stub)
trace-bench ui --runs-dir runs
```

## Real-Mode Setup (OpenRouter)

To run benchmarks in **real** mode with actual LLM calls, set up OpenRouter:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
export OPENAI_API_KEY="$OPENROUTER_API_KEY"
export OPENAI_API_BASE="https://openrouter.ai/api/v1"
export TRACE_DEFAULT_LLM_BACKEND="LiteLLM"
export TRACE_LITELLM_MODEL="openrouter/x-ai/grok-4.1-fast"
```

Then run with a real-mode config:
```bash
trace-bench run --config configs/m2_optimizing_subset.yaml --runs-dir runs
```

## M2 CLI Flags

```
trace-bench run --config <yaml>
    --runs-dir DIR          Output directory (default: runs)
    --max-workers N         Parallel jobs (default: from config or 1)
    --resume auto|failed|none
                            Resume mode (default: auto)
                              auto   = skip OK jobs, re-run failed + new
                              failed = re-run only failed, skip OK + never-run
                              none   = fresh run, re-run everything
    --force                 Shorthand for --resume none
    --job-timeout SECONDS   Per-job timeout (default: 30s stub, 600s real)
```

## Run Artifacts

```
runs/<run_id>/
  meta/
    config.snapshot.yaml    # Frozen config used for this run
    env.json                # Captured environment variables
    git.json                # Git commit/branch info
    manifest.json           # All jobs with status + resolved kwargs
  summary.json              # Aggregate counts (ok/failed/skipped)
  results.csv               # One row per job
  jobs/
    <job_id>/
      job_meta.json         # Per-job metadata (canonical status source)
      results.json          # Per-job results
      events.jsonl          # Event log
      tb/                   # TensorBoard logs
      artifacts/            # Task-specific outputs
```

## Dependencies

### Core (M1+M2)

System:
- Graphviz (system package)

Python:
- `graphviz`, `pyyaml`, `pytest`, `numpy`, `matplotlib`
- `litellm==1.75.0`, `aiohttp>=3.9,<3.13`
- `scipy`, `networkx`, `gymnasium`

### Full Coverage (M2 notebook)
- `pandas`, `datasets`, `sympy`, `pymoo`, `gym`

### OpenTrace Examples (100% pass)
- `datasets`, `textgrad`, `dspy`, `autogen`, `python-dotenv`

## OpenTrace Examples Smoke (100% Pass Mode)

To enforce 100% example smoke in CI, run:
```bash
TRACE_BENCH_STRICT_EXAMPLES=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```
Without strict mode, the smoke test skips only when optional deps are missing.

## VeriBench Integration

VeriBench is fully integrated via `trace_bench/veribench_adapter.py` with two discovery paths:

1. **Entrypoint module** (preferred): If `Veribench/my_processing_agents/optimize_veribench_agent.py` is installed locally, the adapter imports it and uses its task list and bundle builder directly.
2. **HuggingFace dataset fallback**: When the entrypoint is not available, the adapter loads tasks from the [`allenanie/veribench_with_prompts`](https://huggingface.co/datasets/allenanie/veribench_with_prompts) dataset (~140 tasks) and builds trace-compatible bundles automatically.

### Usage

```bash
# Run all VeriBench tasks (uses whichever discovery path is available)
trace-bench run --config configs/veribench.yaml

# List discovered VeriBench tasks
trace-bench list-tasks --bench veribench
```

### Installing the entrypoint module (optional)

To use the entrypoint path instead of the HF fallback, place the VeriBench agent code under `Veribench/my_processing_agents/`:

```
Veribench/
  my_processing_agents/
    __init__.py
    optimize_veribench_agent.py   # must expose build_trace_problem() or list_tasks()
  install.sh
  pyproject.toml
```

See `Veribench/README.md` for platform-specific installation instructions.

## Problem Sets

### General Problem Sets
- Simple QA Problem
- A problem set that uses a ReAct agent
- A problem set that uses a tool-calling agent
- Code writing/generation
- Math proof generation
- A **reasoning** problem set that uses multi-agent (Learning to reason)

### LLM4AD problems set
A comprehensive collection of **65 benchmark tasks** derived from the [LLM4AD (Large Language Models for Algorithm Design)](https://github.com/Optima-CityU/LLM4AD).
Current implementation of graph is a single node.

- **Optimization - Basic** (18 tasks): `circle_packing`, `online_bin_packing_local`, etc.
- **Optimization - Constructive** (15 tasks): `optimization_tsp_construct`, `optimization_knapsack_construct`, `optimization_set_cover_construct`, etc.
- **Optimization - CO-Bench** (36 tasks): `optimization_travelling_salesman_problem`, `optimization_job_shop_scheduling`, `optimization_container_loading`, etc.
- **Machine Learning** (5 tasks): `machine_learning_acrobot`, `machine_learning_pendulum`, `machine_learning_moon_lander`, etc.
- **Scientific Discovery** (6 tasks): `science_discovery_ode_1d`, `science_discovery_oscillator1`, etc.

**Supported Algorithms:** PrioritySearch, GEPA-Base, GEPA-UCB, GEPA-Beam

**See detailed usage guide:** `LLM4AD/readme.md`

## Agent Architecture
- ReAct agent

All the libraries from other repos are stored and managed in the `external` folder -- this folder will be created if one of the `install.sh` script is run inside the task folder.
