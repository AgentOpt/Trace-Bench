"""
This file modifies https://github.com/ScalingIntelligence/KernelBench/blob/main/scripts/generate_and_eval_single_sample_modal.py
Original authors: William Hu, Anne Ouyang, Simon Guo

Note: Triton evaluation is not supported due to KernelBench's original code
"""

import os, sys
import json
import modal

from datasets import load_dataset

app = modal.App("trace_bench_kernel_triton_eval")

# NOTE: this assumes you have run `install.sh` inside this repo so that KernelBench is stored and built under `./external/KernelBench`
REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

REPO_EXTERNAL_LIB_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../external/",
    )
)
KERNEL_BENCH_PATH = os.path.join(REPO_EXTERNAL_LIB_PATH, "KernelBench")

gpu_arch_mapping = {"L40S": ["Ada"], "H100": ["Hopper"], "A100": ["Ampere"], "L4": ["Ada"], "T4": ["Turing"],
                    "A10G": ["Ampere"]}
cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  # includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install("git",
                 "gcc-10",
                 "g++-10",
                 "clang"  # note i skip a step
                 )
    .pip_install(  # required to build flash-attn
        "anthropic",
        "numpy",
        "openai",
        "packaging",
        "pydra_config",
        "torch==2.5.0",
        "tqdm",
        "datasets",
        "transformers",
        "google-generativeai",
        "together",
        "pytest",
        "ninja",
        "utils",
        "dotenv",
        "triton"
    ).env({
        "PYTHONPATH": "/root/KernelBench"  # add KernelBench to python path
    }).add_local_dir(
        KERNEL_BENCH_PATH,
        remote_path="/root/KernelBench"
    ).add_local_file(
        os.path.join(REPO_TOP_PATH, "KernelBench/level1_prob1_cuda_custom_cuda_gpt5_example.txt"),
        remote_path="/root"
    )
)


@app.function(image=image, timeout=3600, gpu="L40S")
def eval_single_sample_modal(ref_arch_src, custom_cuda, verbose, gpu_arch):
    # 3. Evaluate Kernel
    # NOTE: no need to wrap around process here as only a single sample
    # see batch eval for examples of process isolation
    from src.eval import eval_kernel_against_ref
    from src.utils import set_gpu_arch

    set_gpu_arch(gpu_arch)

    return eval_kernel_against_ref(
        ref_arch_src, custom_cuda, verbose=verbose, measure_performance=True, num_correct_trials=5,
        num_perf_trials=100
    )


def quick_test_on_setup():
    """
    This allows you to test if your modal AI account setup is fully correct.
    Should take about ~1m to finish.
    :return:
    """
    ds = load_dataset("allenanie/kernelbench_with_prompts")
    examples = [ex for ex in ds['train'] if ex['backend'] == 'cuda']
    ex1 = examples[0]
    ref_arch_src = ex1['ref_arch_src']
    custom_cuda = open("level1_prob1_cuda_custom_cuda_gpt5_example.txt").read()
    gpu = "L40S"
    verbose = True

    kernel_exec_result = eval_single_sample_modal.remote(ref_arch_src, custom_cuda, verbose, gpu_arch_mapping[gpu])
    print(f"Evaluation result:")
    print(kernel_exec_result)


@app.local_entrypoint()
def main():
    quick_test_on_setup()


if __name__ == '__main__':
    pass
