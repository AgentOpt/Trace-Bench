# this is to be able to release cuda memory

# now we need to be careful on handling eval
import argparse
import json
import multiprocessing as mp
import os
import sys
import torch
from pydra import REQUIRED, Config
from dataclasses import dataclass
from datasets import load_dataset

from src.dataset import construct_kernelbench_dataset
from src.eval import build_compile_cache, eval_kernel_against_ref, KernelExecResult, check_metadata_serializable_all_types
from src.utils import set_gpu_arch, read_file

torch.set_printoptions(precision=4, threshold=10)

@dataclass
class WorkArgs:
    problem_id: int # logically indexed
    sample_id: int
    device: torch.device
    level: int = None  # level for the problem (needed when using preset_problems)

class EvalConfig(Config):

    def __init__(self):
        self.run_name = "priority_search_trial"  # name of the run to evaluate

        self.dataset_src = REQUIRED  # either huggingface or local

        # name of dataset name on Hugging Face
        self.dataset_name = "ScalingIntelligence/KernelBench"

        # Problem Specification
        self.level = None  # Required only when not using preset_problems

        # subset of problems to evaluate
        self.subset = (None, None)  # (start_id, end_id), these are the logical index

        # preset list of specific problems to run (overrides level and subset if provided)
        # format: ["level1_problem1", "level1_problem103", "level2_problem14", ...]
        self.preset_problems = ["level1_problem1", "level1_problem23", "level1_problem33", "level1_problem40",
                                "level1_problem6", "level2_problem14", "level2_problem2", "level2_problem97"]

        # Evaluation Mode: local (requires GPU), see modal (cloud GPU) in the modal file
        self.eval_mode = "local"

        # Construct this from mapping from architecture name to torch cuda arch list in the future
        # you can either specify SM version or just use the name
        self.gpu_arch = ["Ada"]

        # Logging
        # Top Directory to Store Runs
        self.runs_dir = os.path.join("/home/ubuntu/KernelBench", "runs")

        self.verbose = False

        # Eval settings
        self.num_correct_trials = 5
        self.num_perf_trials = 100
        self.timeout = 180  # in seconds
        self.measure_performance = True

        # Eval Flow setting
        # To speedup evaluation, you can start building the kernel on CPU on disk as cache
        self.build_cache = False
        self.num_cpu_workers = 20  # number of parallel process to to parallelize the build on CPUs

        # Directory to build kernels for evaluation
        self.kernel_eval_build_dir = os.path.join("/home/ubuntu/KernelBench", "cache")

        # number of GPUs to do batch evaluation
        self.num_gpu_devices = 1


def evaluate_single_sample(work_args: WorkArgs, configs: EvalConfig, custom_cuda: str,
                           ref_arch_src: str):  # KernelExecResult | None
    """
    Evaluate a single sample on a single GPU
    """
    problem_id, sample_id, device, level = (
        work_args.problem_id,
        work_args.sample_id,
        work_args.device,
        work_args.level,
    )

    # Determine which level to use - either from work_args.level (for preset problems) or configs.level
    level_to_use = level if level is not None else configs.level

    # For preset problems, dataset is a dict of {level: dataset}
    # if isinstance(dataset, dict):
    #     curr_dataset = dataset[level_to_use]
    # else:
    #     curr_dataset = dataset

    # fetch reference architecture from problem directory
    # ref_arch_src = fetch_ref_arch_from_problem_id(curr_dataset, problem_id, configs.dataset_src, level_to_use)

    # fetch kernel from disk
    # Add database support in the future
    kernel_src = custom_cuda  # fetch_kernel_from_disk(run_dir, level_to_use, problem_id, sample_id)

    assert kernel_src is not None, f"Kernel not found for problem {problem_id} sample {sample_id}"

    build_dir = os.path.join(configs.kernel_eval_build_dir, configs.run_name, f"{problem_id}", f"{sample_id}")

    try:
        eval_result = eval_kernel_against_ref(
            original_model_src=ref_arch_src,
            custom_model_src=kernel_src,
            measure_performance=configs.measure_performance,
            verbose=configs.verbose,
            num_correct_trials=configs.num_correct_trials,
            num_perf_trials=configs.num_perf_trials,
            build_dir=build_dir,
            device=device,
        )
        return eval_result
    except Exception as e:
        print(
            f"[WARNING] Last level catch on {sample_id}: Some issue evaluating for kernel: {e} "
        )
        if "CUDA error" in str(e):
            # NOTE: count this as compilation failure as it is not runnable code
            metadata = {
                "cuda_error": f"CUDA Error: {str(e)}",
                "hardware": torch.cuda.get_device_name(device=device),
                "device": str(device),
            }  # log this for debugging as this usually signifies illegal memory access
            eval_result = KernelExecResult(
                compiled=False, correctness=False, metadata=metadata
            )
            return eval_result
        else:
            metadata = {"other_error": f"error: {str(e)}",
                        "hardware": torch.cuda.get_device_name(device=device),
                        "device": str(device)
                        }  # for debugging
            eval_result = KernelExecResult(compiled=False, correctness=False,
                                           metadata=metadata)
            return eval_result


def cuda_single_eval_wrapper(curr_work: WorkArgs, configs: dict, custom_cuda: str, ref_arch_src: str):
    """
    Wrapper to handle timeout and keyboard interrupt (without multiprocessing)
    """
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Evaluation timed out after {configs.timeout} seconds")
    
    try:
        # Set up timeout using signal (Unix systems)
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(configs.timeout))
        
        # Run evaluation directly
        try:
            result = evaluate_single_sample(curr_work, configs, custom_cuda, ref_arch_src)
        except KeyboardInterrupt:
            print("\n [Terminate] Caught KeyboardInterrupt...")
            raise
        finally:
            # Cancel the alarm
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
        
        print(f"[Eval Result] Problem ID: {curr_work.problem_id}, Sample ID: {curr_work.sample_id}: {result}")
        return result
        
    except TimeoutError:
        print(f"[WARNING] Evaluation TIMED OUT for Problem ID: {curr_work.problem_id}, Sample ID: {curr_work.sample_id}")
        # Return a timeout result
        return KernelExecResult(
            compiled=False,
            correctness=False,
            metadata={
                "timeout_error": True,
                "timeout_seconds": configs.timeout,
                "problem_id": curr_work.problem_id,
                "sample_id": curr_work.sample_id
            }
        )


def kernel_exec_result_to_dict(result: KernelExecResult) -> dict:
    """
    Convert KernelExecResult to a JSON-serializable dictionary
    """
    if result is None:
        return None
    
    return {
        "compiled": result.compiled,
        "correctness": result.correctness,
        "metadata": result.metadata,
        "runtime": result.runtime,
        "runtime_stats": result.runtime_stats
    }


def create_argparser():
    """Create argument parser for cuda_single_eval_wrapper"""
    parser = argparse.ArgumentParser(
        description="Evaluate a single CUDA kernel against reference implementation"
    )
    
    # WorkArgs parameters
    parser.add_argument(
        "--problem-id", 
        type=int, 
        required=True,
        help="Problem ID (logical index)"
    )
    parser.add_argument(
        "--sample-id", 
        type=int, 
        required=True,
        help="Sample ID for the kernel to evaluate"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda:0",
        help="CUDA device to use (e.g., 'cuda:0', 'cuda:1')"
    )
    parser.add_argument(
        "--level", 
        type=int, 
        default=None,
        help="Level for the problem (needed when using preset_problems)"
    )
    
    # CUDA source parameters
    parser.add_argument(
        "--custom-cuda", 
        type=str, 
        required=True,
        help="Custom CUDA kernel source code (either file path or direct string)"
    )
    parser.add_argument(
        "--custom-cuda-file", 
        action="store_true",
        help="Treat --custom-cuda as file path instead of direct source code"
    )
    parser.add_argument(
        "--ref-arch-src", 
        type=str, 
        required=True,
        help="Reference architecture source code (either file path or direct string)"
    )
    parser.add_argument(
        "--ref-arch-file", 
        action="store_true",
        help="Treat --ref-arch-src as file path instead of direct source code"
    )
    
    return parser


def main():
    """Main function that parses arguments and calls cuda_single_eval_wrapper"""
    parser = create_argparser()
    args = parser.parse_args()
    
    # Create WorkArgs from parsed arguments
    device = torch.device(args.device)
    curr_work = WorkArgs(
        problem_id=args.problem_id,
        sample_id=args.sample_id,
        device=device,
        level=args.level
    )
    
    # Handle custom_cuda source
    if args.custom_cuda_file:
        try:
            custom_cuda = read_file(args.custom_cuda)
        except Exception as e:
            print(f"Error reading custom CUDA file '{args.custom_cuda}': {e}")
            sys.exit(1)
    else:
        custom_cuda = args.custom_cuda
    
    # Handle ref_arch_src source
    if args.ref_arch_file:
        try:
            ref_arch_src = read_file(args.ref_arch_src)
        except Exception as e:
            print(f"Error reading reference architecture file '{args.ref_arch_src}': {e}")
            sys.exit(1)
    else:
        ref_arch_src = args.ref_arch_src
    
    # Create fixed EvalConfig
    configs = EvalConfig()
    
    # Call the wrapper function (now without multiprocessing)
    try:
        result = cuda_single_eval_wrapper(curr_work, configs, custom_cuda, ref_arch_src)
        
        # Convert result to JSON-serializable format
        result_dict = kernel_exec_result_to_dict(result)
        
        # Print as JSON for easy parsing by calling scripts
        print("=== KERNEL_EXEC_RESULT_JSON ===")
        print(json.dumps(result_dict, indent=2))
        print("=== END_KERNEL_EXEC_RESULT_JSON ===")
        
        # Also print the traditional format for debugging
        print(f"Final evaluation result: {result}")
        return result
    except KeyboardInterrupt:
        print("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        # Create an error result and output it as JSON too
        error_result = KernelExecResult(
            compiled=False,
            correctness=False,
            metadata={
                "error": str(e),
                "error_type": "main_exception",
                "problem_id": args.problem_id,
                "sample_id": args.sample_id,
                "device": args.device
            }
        )
        error_dict = kernel_exec_result_to_dict(error_result)
        print("=== KERNEL_EXEC_RESULT_JSON ===")
        print(json.dumps(error_dict, indent=2))
        print("=== END_KERNEL_EXEC_RESULT_JSON ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
