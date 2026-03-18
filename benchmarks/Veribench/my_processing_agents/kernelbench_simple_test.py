#!/usr/bin/env python3
"""
Simple KernelBench test script:
1. Load dataset (level 1, cuda backend)
2. Generate CUDA code with LiteLLM (gemini-2.0-flash-exp-1114)
3. Evaluate with Modal AI
"""

import sys
import os

# Add paths for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
kernelbench_root = os.path.join(os.path.dirname(project_root), "KernelBench")
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if kernelbench_root not in sys.path:
    sys.path.insert(0, kernelbench_root)

from datasets import load_dataset
import litellm
import re

# Load environment variables
try:
    from my_processing_agents import secrets_local
except ImportError:
    print("Warning: secrets_local.py not found. Make sure API keys are set in environment.")


def load_kernelbench_dataset():
    """
    Load KernelBench dataset and filter for level 1, cuda backend.

    Returns:
        List of problems matching criteria
    """
    print("Loading KernelBench dataset from HuggingFace...")

    
   
    ds = load_dataset("allenanie/kernelbench_with_prompts")
    print(f"✓ Loaded preprocessed dataset: {len(ds['train'])} examples")

    # Filter for level 1 and cuda backend
    problems = [
        ex for ex in ds['train']
        if ex.get('backend') == 'cuda' and ex.get('level') == 1
    ]

    print(f"✓ Filtered to {len(problems)} level 1 CUDA problems")
    return problems

    

def generate_cuda_with_llm(prompt: str, model: str = "gemini/gemini-2.0-flash-exp"):
    """
    Generate CUDA code using LiteLLM.

    Args:
        prompt: The prompt to send to the LLM
        model: Model name for LiteLLM

    Returns:
        Generated CUDA code (string)
    """
    print(f"\n📝 Generating CUDA code with {model}...")

    # Configure LiteLLM
    litellm.drop_params = True
    litellm.suppress_debug_info = True

    try:
        response = litellm.completion(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert CUDA programmer. Generate optimized CUDA kernels."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=8192  # CUDA code can be long
        )

        generated_code = response.choices[0].message.content
        print(f"✓ Generated {len(generated_code)} characters of code")

        return generated_code

    except Exception as e:
        print(f"❌ Error generating code: {e}")
        raise


def extract_code_from_markdown(text: str) -> str:
    """
    Extract Python/CUDA code from markdown code blocks.

    Args:
        text: LLM response that may contain markdown

    Returns:
        Extracted code
    """
    # Try to find code in markdown blocks
    patterns = [
        r'```python\n(.*?)```',
        r'```cuda\n(.*?)```',
        r'```\n(.*?)```'
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

    # No markdown found, return as-is
    return text.strip()


def evaluate_with_modal(ref_arch_src: str, custom_cuda: str, verbose: bool = True):
    """
    Evaluate CUDA code using Modal AI.

    Args:
        ref_arch_src: Reference PyTorch code
        custom_cuda: Generated CUDA code
        verbose: Whether to print verbose output

    Returns:
        Evaluation result
    """
    print("\n🚀 Evaluating with Modal AI...")

    # Import Modal evaluation function
    from evaluate_with_modal import app, eval_single_sample_modal, gpu_arch_mapping

    # Run evaluation on Modal
    with app.run():
        result = eval_single_sample_modal.remote(
            ref_arch_src=ref_arch_src,
            custom_cuda=custom_cuda,
            verbose=verbose,
            gpu_arch=gpu_arch_mapping["L40S"]  # Use L40S GPU
        )

    return result


def print_result(result):
    """
    Pretty print evaluation result.

    Args:
        result: KernelExecResult from Modal
    """
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print(f"\n✓ Compilation: {'✅ SUCCESS' if result.compiled else '❌ FAILED'}")
    print(f"✓ Correctness: {'✅ PASSED' if result.correctness else '❌ FAILED'}")

    if result.correctness and result.runtime > 0:
        print(f"⚡ Runtime: {result.runtime:.6f} seconds")

        if result.runtime_stats:
            stats = result.runtime_stats
            print(f"\n📊 Performance Statistics:")
            print(f"   Mean:   {stats['mean']:.6f}s")
            print(f"   Median: {stats['median']:.6f}s")
            print(f"   Std:    {stats['std']:.6f}s")
            print(f"   Min:    {stats['min']:.6f}s")
            print(f"   Max:    {stats['max']:.6f}s")

        if hasattr(result, 'metadata') and 'speedup' in result.metadata:
            speedup = result.metadata['speedup']
            print(f"\n🚀 Speedup: {speedup:.2f}x vs PyTorch")

    if not result.compiled or not result.correctness:
        if hasattr(result, 'metadata') and 'error' in result.metadata:
            print(f"\n❌ Error: {result.metadata['error']}")

    print("=" * 70)


def main():
    """
    Main execution flow:
    1. Load dataset
    2. Generate CUDA code
    3. Evaluate with Modal
    """
    print("=" * 70)
    print("KERNELBENCH SIMPLE TEST")
    print("=" * 70)

    # Step 1: Load dataset
    problems = load_kernelbench_dataset()

    if not problems:
        print("❌ No problems found! Check dataset loading.")
        return

    # Get first problem
    problem = problems[0]

    print(f"\n📋 Problem Info:")
    print(f"   Problem ID: {problem.get('problem_id', 'N/A')}")
    print(f"   Level: {problem.get('level', 'N/A')}")
    print(f"   Backend: {problem.get('backend', 'N/A')}")

    # Show reference code snippet
    ref_code = problem['ref_arch_src']
    print(f"\n📄 Reference PyTorch Code (first 300 chars):")
    print("-" * 70)
    print(ref_code[:300] + "..." if len(ref_code) > 300 else ref_code)
    print("-" * 70)

    # Step 2: Generate CUDA code with LLM
    prompt = problem['input']

    try:
        generated_code = generate_cuda_with_llm(
            prompt=prompt,
            model="gemini/gemini-2.5-flash-lite"  # Using Gemini 2.0 Flash
        )

        # Extract code from markdown if needed
        cuda_code = extract_code_from_markdown(generated_code)

        print(f"\n📝 Generated CUDA Code (first 500 chars):")
        print("-" * 70)
        print(cuda_code[:500] + "..." if len(cuda_code) > 500 else cuda_code)
        print("-" * 70)

        # Step 3: Evaluate with Modal
        result = evaluate_with_modal(
            ref_arch_src=ref_code,
            custom_cuda=cuda_code,
            verbose=True
        )

        # Step 4: Print results
        print_result(result)

        # Save results to file
        import json
        result_dict = {
            'problem_id': problem.get('problem_id'),
            'level': problem.get('level'),
            'compiled': result.compiled,
            'correctness': result.correctness,
            'runtime': result.runtime if result.correctness else -1.0,
            'speedup': result.metadata.get('speedup') if hasattr(result, 'metadata') else None
        }

        output_file = os.path.join(project_root, 'my_processing_agents', 'kernelbench_test_result.json')
        with open(output_file, 'w') as f:
            json.dump(result_dict, f, indent=2)

        print(f"\n💾 Results saved to: {output_file}")

        # Final summary
        if result.compiled and result.correctness:
            print("\n🎉 SUCCESS! CUDA code compiled and passed correctness test!")
        elif result.compiled:
            print("\n⚠️  Code compiled but failed correctness test.")
        else:
            print("\n❌ Code failed to compile.")

    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
