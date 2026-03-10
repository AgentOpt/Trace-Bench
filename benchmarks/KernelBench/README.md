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

## Modal Approach

First, create an account with [Modal AI](http://modal.com/login), then install the Modal package and run the setup command:

(If `install.sh` is run, modal package should have been installed)
```bash
# pip install modal
python -m modal setup
```

Once the setup is done, you can run the evaluation script `cuda_eval_modal.py` to test your setup:

```bash
modal run evaluate_with_modal.py
```

## CUDA Eval Server Approach

If you have your own server to run / use, and don't want to pay for Modal, we provide two scripts:
- `cuda_eval_server.py`
- `cuda_eval_client.py`

You can start the server by running:
```bash
python cuda_eval_server.py --cuda-devices cuda:0 cuda:1 cuda:2 cuda:3 --port 6000
```

Client can be used as code:

```python
from cuda_eval_client import CUDAEvalClient

client = CUDAEvalClient("http://localhost:6000")
problem_id = 1 
sample_id = 0

job_id = client.submit_evaluation(
            problem_id=problem_id, sample_id=sample_id,
            custom_cuda=custom_cuda,
            ref_arch_src=ref_arch_src,
            level=1
)

result = client.wait_for_job(job_id, timeout=300)
kernel_exec_result = result['result']
```

The server and client are both implemented in an async style and once an evaluation job is submitted, we use a blocking event `wait_for_job` to query
the evaluation result from the server.

## Verified Platforms

- Ubuntu 24.04 LTS
- MacOS Ventura with M1 Pro chip

## Agent

We also need to run a RAG agent to help with the benchmarking.

