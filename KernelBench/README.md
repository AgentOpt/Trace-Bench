# Installation

KernelBench requires evaluations on GPUs. We support two different ways of interfacing with GPUs:
1. Modal AI approach
2. Self-hosted server approach

First, run the command to install KernelBench dependencies:

```bash
bash install.sh
```

Activate the uv environment by running:
```bash
source .venv/bin/activate
```

Deactivate by running `deactivate`.

## Modal AI Approach

First, create an account with [Modal AI](http://modal.com/login), then install the Modal package and run the setup command:

(If `install.sh` is run, modal package should have been installed)
```bash
# pip install modal
python -m modal setup
```

## Verified Platforms

- Ubuntu 24.04 LTS
- MacOS Ventura with M1 Pro chip

## Agent

We also need to run a RAG agent to help with the benchmarking.

