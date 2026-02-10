# Trace-Bench
Benchmark to evaluate LLM as an optimizer.

Currently, we are adding problems/domains one folder at a time. 

The instructions to run each task are located inside the task folder.

## Quick Start (Runner/CLI)

```bash
# M1 review checklist (recommended order)
# 1) List tasks (LLM4AD + example stubs)
trace-bench list-tasks --root LLM4AD/benchmark_tasks

# 2) Validate a config
trace-bench validate --config configs/smoke.yaml

# 3) Run Stub smoke (deterministic, no keys)
trace-bench run --config configs/smoke.yaml --runs-dir runs

# 4) Run Real smoke (requires OPENAI_API_KEY)
trace-bench run --config configs/smoke_real.yaml --runs-dir runs

# 5) Run tests (disable external plugin autoload)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q

# List tasks (LLM4AD + example stubs)
trace-bench list-tasks --root LLM4AD/benchmark_tasks

# Validate a config
trace-bench validate --config configs/smoke.yaml

# Run a smoke benchmark
trace-bench run --config configs/smoke.yaml

# Launch UI (stub)
trace-bench ui --runs-dir runs
```

Expected run artifacts:
- `runs/<run_id>/config.snapshot.yaml`
- `runs/<run_id>/env.json`
- `runs/<run_id>/results.csv`
- `runs/<run_id>/events.jsonl`
- `runs/<run_id>/summary.json`
- `runs/<run_id>/tb/`

## M1 Dependencies (Required for Full Pass)

System:
- Graphviz (system package)

Python:
- `graphviz`, `pyyaml`, `pytest`, `numpy`, `matplotlib`, `litellm==1.75.0`

OpenTrace examples strict smoke (for 100% pass):
- `datasets`, `textgrad`, `dspy`, `autogen`, `python-dotenv`

## OpenTrace Examples Smoke (100% Pass Mode)

To enforce 100% example smoke in CI, run:
```bash
TRACE_BENCH_STRICT_EXAMPLES=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```
Without strict mode, the smoke test skips only when optional deps are missing.

## VeriBench Status (In Scope, Pending Input)

VeriBench is in scope but requires the Trace team to provide the task entrypoint/task list.
CLI flags are ready (`--bench veribench`), and will raise a clear `NotImplementedError` until the entrypoint is provided.

## Problem Sets

### General Problem Sets
- Simple QA Problem
- A problem set that uses a ReAct agent
- A problem set that uses a tool-calling agent
- Code writing/generation
- Math proof generation
- A **reasoning** problem set that uses multi-agent (Learning to reason)

### LLM4AD problems set
A comprehensive collection of **60 benchmark tasks** derived from the [LLM4AD (Large Language Models for Algorithm Design)](https://github.com/Optima-CityU/LLM4AD).
Current implementation of graph is a single node.

- **Optimization - Basic** (18 tasks): `circle_packing`, `online_bin_packing_local`, etc.
- **Optimization - Constructive** (15 tasks): `optimization_tsp_construct`, `optimization_knapsack_construct`, `optimization_set_cover_construct`, etc.
- **Optimization - CO-Bench** (21 tasks): `optimization_travelling_salesman_problem`, `optimization_job_shop_scheduling`, `optimization_container_loading`, etc.
- **Machine Learning** (5 tasks): `machine_learning_acrobot`, `machine_learning_pendulum`, `machine_learning_moon_lander`, etc.
- **Scientific Discovery** (1 task): `science_discovery_ode_1d`

**Supported Algorithms:** PrioritySearch, GEPA-Base, GEPA-UCB, GEPA-Beam

**See detailed usage guide:** `LM4AD/readme.md`

## Agent Architecture
- ReAct agent

All the libraries from other repos are stored and managed in the `external` folder -- this folder will be created if one of the `install.sh` script is run inside the task folder.
