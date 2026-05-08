# Trace-Bench

A benchmarking framework for evaluating LLM-as-optimizer algorithms, built on [OpenTrace](https://github.com/AgentOpt/Trace).

Trace-Bench provides a **CLI**, **Gradio UI**, and **notebook workflows** to run reproducible experiments across LLM4AD, VeriBench, KernelBench, HuggingFace QA, and Terminal-Bench via Harbor with fair comparisons between trainers, optimizers, and LLM backends.

**Full documentation:** [docs/](docs/README.md)

## Install

```bash
git clone https://github.com/AgentOpt/Trace-Bench.git
cd Trace-Bench

# Trace-Bench depends on Trace/OpenTrace (`opto`).
# Install one of the following before running commands:

# Option A (preferred if published in your environment):
pip install trace-opt

# Option B (editable sibling checkout):
git clone https://github.com/AgentOpt/Trace.git ../OpenTrace
pip install -e ../OpenTrace

# Install Trace-Bench (core)
pip install -e .

# Optional extras can be combined:
pip install -e ".[hf]"          # HuggingFace QA benchmarks
pip install -e ".[dspy]"        # DSPy trainer adapter
pip install -e ".[terminal-bench]"  # Terminal-Bench 2.0 via Harbor
pip install -e ".[all-external]"  # everything
```

Note: the `terminal-bench` extra installs Harbor, which currently requires Python >=3.12.

## Quick Start

```bash
# List available tasks
trace-bench list-tasks

# Run a smoke test (stub mode, no API keys needed)
trace-bench run --config configs/smoke.yaml --runs-dir runs

# Launch the Gradio UI
trace-bench ui --runs-dir runs
```

## Real-Mode Setup

To run benchmarks with actual LLM calls, configure a provider through environment variables or a config `llm` block.

Direct OpenAI:

```bash
export OPENAI_API_KEY="sk-..."
export TRACE_DEFAULT_LLM_BACKEND="LiteLLM"
export TRACE_LITELLM_MODEL="gpt-5.4-nano"
```

OpenRouter-compatible endpoint:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
export OPENAI_API_KEY="$OPENROUTER_API_KEY"
export OPENAI_API_BASE="https://openrouter.ai/api/v1"
export TRACE_DEFAULT_LLM_BACKEND="LiteLLM"
export TRACE_LITELLM_MODEL="openrouter/x-ai/grok-4.1-fast"
```

Then run with a real-mode config:
```bash
trace-bench run --config configs/smoke_real.yaml --runs-dir runs
```


## Terminal-Bench / Harbor Support

Trace-Bench can optimize Harbor's Terminus-2 prompt template against Terminal-Bench 2.0 tasks. Task IDs use the `terminal_bench:<task_slug>` namespace, for example `terminal_bench:regex-log`.

### Install and configure

```bash
# Install Trace-Bench plus Harbor support.
pip install -e ".[terminal-bench]"

# For local Docker execution:
export HARBOR_MODEL="openai/gpt-5-nano"
export OPENAI_API_KEY="sk-..."

# For hosted Daytona execution, recommended in Colab or environments without Docker:
export DAYTONA_API_KEY="dtn_..."
export HARBOR_MODEL="openai/gpt-5-nano"
export OPENAI_API_KEY="sk-..."
```

Do not put API keys in config files or commit them to the repository. Trace-Bench reads `HARBOR_MODEL`, `OPENAI_API_KEY`, and `DAYTONA_API_KEY` from the process environment at runtime. If you install Harbor manually instead of using the extra, use `pip install "harbor[daytona]>=0.6.0"` for Daytona-backed runs or `pip install "harbor>=0.6.0"` for Docker-only runs.

### Offline smoke run

The demo config defaults to `mode: stub`, so Harbor is not invoked and no API keys are required:

```bash
trace-bench validate --config configs/terminal_bench_demo.yaml --runs-dir runs --strict
trace-bench run --config configs/terminal_bench_demo.yaml --runs-dir runs
```

### Real Terminal-Bench run

Switch the config to `mode: real`, or create a small real-mode config with a Terminal-Bench task. In real mode, each guide evaluation shells out to `harbor run` and stores Harbor command/stdout/stderr/results/trajectory artifacts under the Trace-Bench job artifacts directory (`runs/<run_id>/jobs/<job_id>/artifacts/terminal_bench/`).

```yaml
mode: real
tasks:
  - id: terminal_bench:regex-log
    eval_kwargs:
      harbor_dataset: terminal-bench@2.0
      harbor_env: daytona        # or docker
      harbor_model: null         # read HARBOR_MODEL from the environment
      harbor_cli_timeout_seconds: 1800
```

Then run:

```bash
trace-bench run --config path/to/terminal_bench_real.yaml --runs-dir runs
```

### Limits and operational notes

- Terminal-Bench execution is external: real mode requires the Harbor CLI, a supported environment backend (`docker`, `daytona`, etc.), and model credentials accepted by Harbor/LiteLLM. The model must be compatible with Harbor/LiteLLM's chat-completion path; if a model reports a Responses API `messages`/`input` mismatch, use a compatible model alias or upgrade Harbor/LiteLLM.
- The built-in task discovery intentionally lists only a curated smoke/demo subset. You can still run any Harbor Terminal-Bench 2.0 slug directly as `terminal_bench:<slug>`.
- Real runs can be slow and cost-bearing because optimizers may evaluate the trainable prompt multiple times; keep `max_workers`, `ps_steps`, `ps_candidates`, and `ps_proposals` small while testing.
- Daytona quotas/API limits are outside Trace-Bench's control. Use `harbor_cli_timeout_seconds` and low concurrency when keys are limited.
- Stub mode verifies Trace-Bench wiring only; it does not prove Harbor, Daytona, Docker, credentials, or a specific Terminal-Bench task works.
- Trace-Bench currently optimizes the Terminus-2 prompt template, not Harbor environment definitions or benchmark tests.

See `configs/terminal_bench_demo.yaml` and `notebooks/05_terminal_bench_demo.ipynb` for an end-to-end walkthrough.

## Repository Layout

```
trace_bench/           # Python package
  trainers/            # External / user-customized trainers only.
                       # Built-in trainers (PrioritySearch, GEPA-*) live in
                       # OpenTrace and are discovered automatically.
benchmarks/
  LLM4AD/                  # 65 algorithm design tasks
  hf_qa/                   # HuggingFace QA tasks
  KernelBench/             # CUDA kernel optimization
  Veribench/               # Lean 4 formal verification
configs/                   # YAML experiment configs
notebooks/                 # Jupyter notebooks (Colab-ready)
docs/                      # Full documentation
tests/                     # Test suite (m0/ m1/ m2/ m3/)
runs/                      # Output directory (gitignored)
```

## Benchmarks

### LLM4AD (65 tasks)
Algorithm design tasks from [LLM4AD](https://github.com/Optima-CityU/LLM4AD): optimization (basic, constructive, CO-Bench), machine learning, and scientific discovery.

### HuggingFace QA
Config-driven QA tasks under `hf:*`: HotpotQA, BBEH, MuSiQue, GSM8K, ARC-Challenge, QASC, DROP, QuALITY, and Qasper. Install with `pip install -e ".[hf]"`.

### VeriBench (~140 tasks)
Formal verification tasks translating Python to Lean 4. Integrated via `trace_bench/veribench_adapter.py` with dual discovery: local entrypoint module or [HuggingFace dataset](https://huggingface.co/datasets/allenanie/veribench_with_prompts) fallback.

### KernelBench
CUDA kernel optimization with remote evaluation server. See `benchmarks/KernelBench/README.md`.

## CLI Reference

```
trace-bench list-tasks      List discoverable tasks
trace-bench list-trainers    List available trainers
trace-bench validate         Validate a config file
trace-bench run              Run a benchmark experiment
trace-bench ui               Launch the Gradio UI
```

See [docs/running-experiments.md](docs/running-experiments.md) for full CLI usage.

## Dependencies

**Core:**
- Python >= 3.9, Graphviz (system package)
- `graphviz`, `pyyaml`, `pytest`, `litellm`, `aiohttp`, `tensorboard`, `tensorboardX`, `scikit-learn`

**Full coverage (all benchmarks):**
- `pandas`, `datasets`, `sympy`, `pymoo`, `gymnasium`, `scipy`, `networkx`

## Optional Extras

| Extra | Installs | Enables |
|-------|----------|---------|
| `hf` | `datasets>=2.0` | `hf:*` QA tasks |
| `dspy` | `dspy-ai>=2.0` | `DSPyTrainer` in `trace_bench/trainers/` |
| `terminal-bench` | `harbor[daytona]>=0.6.0` | `terminal_bench:*` tasks through Harbor, including Daytona env support |
| `all-external` | all of the above | everything |

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

## License

MIT

## External Trainers

- `DSPyTrainer` (`trace_bench/trainers/dspy_trainer.py`)
- `TextGradTrainer` (`trace_bench/trainers/textgrad_trainer.py`)
- `OpenEvolveTrainer` (`trace_bench/trainers/openevolve_trainer.py`)
