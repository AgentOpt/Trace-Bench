# Trace-Bench

A benchmarking framework for evaluating LLM-as-optimizer algorithms, built on [OpenTrace](https://github.com/AgentOpt/Trace).

Trace-Bench provides a **CLI**, **Gradio UI**, and **notebook workflows** to run reproducible experiments across multiple benchmarks (LLM4AD, VeriBench, KernelBench) with fair comparisons between trainers, optimizers, and LLM backends.

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

# Install with optional extras (can be combined):
pip install -e ".[hf]"          # HuggingFace QA benchmarks (HotpotQA, BBEH, …)
pip install -e ".[dspy]"        # DSPy trainer adapter
pip install -e ".[all-external]"  # everything
```

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

To run benchmarks with actual LLM calls, configure an API provider:

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

## Repository Layout

```
trace_bench/           # Python package
  trainers/            # External / user-customized trainers only.
                       # Built-in trainers (PrioritySearch, GEPA-*) live in
                       # OpenTrace and are discovered automatically.
benchmarks/
  LLM4AD/                  # 65 algorithm design tasks
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
| `hf` | `datasets>=2.0` | `hf:hotpot_qa`, `hf:bbeh/*` tasks |
| `dspy` | `dspy-ai>=2.0` | `DSPyTrainer` in `trace_bench/trainers/` |
| `all-external` | all of the above | everything |

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

## License

MIT
