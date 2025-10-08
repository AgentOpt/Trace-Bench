"""
This file modifies https://github.com/ScalingIntelligence/KernelBench/blob/main/scripts/generate_and_eval_single_sample_modal.py
Original authors: William Hu, Anne Ouyang, Simon Guo
"""

import os, sys
import json
import modal

from datasets import load_dataset

app = modal.App("trace_bench_kernel_triton_eval")

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
                "clang" # note i skip a step
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
    )
)

@app.cls(image=image)
class EvalFunc:

    @modal.method()
    def eval_single_sample_modal(self, ref_arch_src, custom_cuda, verbose, gpu_arch):
        # 3. Evaluate Kernel
        # NOTE: no need to wrap around process here as only a single sample
        # see batch eval for examples of process isolation
        from src.eval import eval_kernel_against_ref
        from src.utils import set_gpu_arch
        set_gpu_arch(gpu_arch)
        return eval_kernel_against_ref(
            ref_arch_src, custom_cuda, verbose=verbose, measure_performance=True, num_correct_trials=5, num_perf_trials=100
        )

@app.local_entrypoint()
def main(foo: int, bar: str):
    pass
    # some_modal_function.remote(foo, bar)

if __name__ == '__main__':
    pass