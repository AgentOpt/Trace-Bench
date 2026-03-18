# VeriBench

VeriBench tasks are integrated into Trace-Bench via `trace_bench/veribench_adapter.py`.

## How task discovery works

The adapter uses two discovery paths (tried in order):

1. **Entrypoint module** — imports `my_processing_agents.optimize_veribench_agent` from this directory and calls its task list / bundle builder API.
2. **HuggingFace dataset fallback** — loads tasks directly from [`allenanie/veribench_with_prompts`](https://huggingface.co/datasets/allenanie/veribench_with_prompts) (~140 tasks) and builds trace-compatible bundles automatically.

If neither path succeeds, VeriBench tasks are skipped with a structured reason rather than raising.

## Installing the entrypoint module

Place the VeriBench agent code under this directory:

```
Veribench/
  my_processing_agents/
    __init__.py
    optimize_veribench_agent.py   # entry point
  install.sh
  pyproject.toml
  README.md
```

The entrypoint module should expose at least one of:
- `list_tasks()` / `get_task_list()` / `discover_tasks()` — returns a list of task names
- `build_trace_problem(task_name=...)` / `build_task_bundle(task_name=...)` — returns a trace-compatible bundle dict

## Environment variables

| Variable | Purpose |
|----------|---------|
| `TRACE_BENCH_VERIBENCH_ROOT` | Override the path to this directory (default: `<repo>/Veribench`) |
| `TRACE_BENCH_VERIBENCH_ENTRYPOINT` | Override the module name (default: `my_processing_agents.optimize_veribench_agent`) |

## Platform dependencies

The installation is managed by `uv`. All agent scripts that interface with `PyPantograph` need to be run under this environment.

```bash
bash install.sh
source .venv/bin/activate
```

Deactivate by running `deactivate`.

### Verified Platforms

- Ubuntu 24.04 LTS

**Note**: This script has been verified on Ubuntu 24.04 LTS. For other OS, particularly MacOS, it has not been verified.
The lean environment might have issues with Apple M-series chips. Consult installation guides of `PyPantograph` and `uv` for more details.
