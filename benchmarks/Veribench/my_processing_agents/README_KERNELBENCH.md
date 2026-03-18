# KernelBench Simple Test Script

## Overview

`kernelbench_simple_test.py` is a simple script that demonstrates the complete KernelBench workflow:

1. **Load Dataset** - Level 1 CUDA problems from HuggingFace
2. **Generate Code** - Use LiteLLM with Gemini 2.0 Flash to generate optimized CUDA kernels
3. **Evaluate** - Use Modal AI cloud GPU to test compilation, correctness, and performance

## Prerequisites

### 1. Install KernelBench

```bash
cd ../../KernelBench/
./install.sh
cd ../Veribench/
```

This installs the external KernelBench repository needed for evaluation.

### 2. Setup Modal AI

```bash
pip install modal
python -m modal setup  # Login to Modal account
```

### 3. Set API Keys

Create or update `secrets_local.py` in this directory:

```python
# my_processing_agents/secrets_local.py
import os

# Google AI (for Gemini)
os.environ["GEMINI_API_KEY"] = "your-gemini-api-key-here"

# Or use OpenAI
# os.environ["OPENAI_API_KEY"] = "your-openai-key-here"
```

## Usage

### Run the Test

```bash
cd /Users/xuanfeiren/Documents/Trace-Bench/Veribench/

# Activate environment if needed
source .venv/bin/activate

# Run the script
python my_processing_agents/kernelbench_simple_test.py
```

### Expected Output

```
======================================================================
KERNELBENCH SIMPLE TEST
======================================================================
Loading KernelBench dataset from HuggingFace...
✓ Loaded preprocessed dataset: 250 examples
✓ Filtered to 100 level 1 CUDA problems

📋 Problem Info:
   Problem ID: 100
   Level: 1
   Backend: cuda

📄 Reference PyTorch Code (first 300 chars):
----------------------------------------------------------------------
import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, predictions, targets):
        return torch.mean(torch.clamp(1 - predictions * targets, min=0))
...
----------------------------------------------------------------------

📝 Generating CUDA code with gemini/gemini-2.0-flash-exp...
✓ Generated 3842 characters of code

🚀 Evaluating with Modal AI...

======================================================================
EVALUATION RESULTS
======================================================================

✓ Compilation: ✅ SUCCESS
✓ Correctness: ✅ PASSED
⚡ Runtime: 0.000245 seconds

📊 Performance Statistics:
   Mean:   0.000245s
   Median: 0.000243s
   Std:    0.000001s
   Min:    0.000238s
   Max:    0.000289s

🚀 Speedup: 3.2x vs PyTorch
======================================================================

💾 Results saved to: .../my_processing_agents/kernelbench_test_result.json

🎉 SUCCESS! CUDA code compiled and passed correctness test!
```

## Script Details

### What It Does

1. **Dataset Loading**
   - Tries `allenanie/kernelbench_with_prompts` first (preprocessed with prompts)
   - Falls back to `ScalingIntelligence/KernelBench` raw dataset
   - Filters for `level=1` and `backend='cuda'`
   - Gets first problem

2. **LLM Generation**
   - Uses LiteLLM with `gemini/gemini-2.0-flash-exp` model
   - Sends prompt with reference PyTorch code
   - Extracts generated code from markdown blocks
   - Configurable model (can change to GPT-4, Claude, etc.)

3. **Modal Evaluation**
   - Uploads code to Modal AI cloud GPU (L40S)
   - Compiles CUDA with `torch.utils.cpp_extension.load_inline()`
   - Tests correctness by comparing outputs
   - Benchmarks performance (100 trials)
   - Returns detailed results

4. **Result Saving**
   - Saves JSON with compilation, correctness, runtime, speedup
   - File: `kernelbench_test_result.json`

### Key Functions

```python
load_kernelbench_dataset()
# Returns: List of level 1 CUDA problems

generate_cuda_with_llm(prompt, model="gemini/gemini-2.0-flash-exp")
# Returns: Generated CUDA code string

extract_code_from_markdown(text)
# Returns: Code extracted from ```python...``` blocks

evaluate_with_modal(ref_arch_src, custom_cuda, verbose=True)
# Returns: KernelExecResult with compiled, correctness, runtime, etc.

print_result(result)
# Prints: Formatted evaluation results
```

## Customization

### Change LLM Model

Edit line in `main()`:

```python
generated_code = generate_cuda_with_llm(
    prompt=prompt,
    model="gpt-4"  # or "claude-3-5-sonnet-20241022", etc.
)
```

### Change GPU Type

Edit in `evaluate_with_modal()`:

```python
result = eval_single_sample_modal.remote(
    ref_arch_src=ref_arch_src,
    custom_cuda=custom_cuda,
    verbose=verbose,
    gpu_arch=gpu_arch_mapping["H100"]  # or "A100", "L40S", etc.
)
```

### Test Different Problems

Modify in `main()`:

```python
# Get first problem
problem = problems[0]  # Change to problems[5], problems[10], etc.
```

### Batch Processing

To process multiple problems:

```python
for i, problem in enumerate(problems[:10]):  # First 10 problems
    print(f"\n{'='*70}")
    print(f"Processing problem {i+1}/10...")
    # ... rest of code
```

## Output Files

### `kernelbench_test_result.json`

```json
{
  "problem_id": 100,
  "level": 1,
  "compiled": true,
  "correctness": true,
  "runtime": 0.000245,
  "speedup": 3.2
}
```

## Troubleshooting

### Error: "No module named 'src'"

**Solution:** Run KernelBench install script:
```bash
cd ../../KernelBench/
./install.sh
```

### Error: "Modal not authenticated"

**Solution:** Login to Modal:
```bash
python -m modal setup
```

### Error: "API key not found"

**Solution:** Set API key in `secrets_local.py`:
```python
import os
os.environ["GEMINI_API_KEY"] = "your-key-here"
```

### Error: "Dataset not found"

**Solution:** Check internet connection. The script will:
1. Try preprocessed dataset first
2. Fall back to raw dataset
3. Create prompts on-the-fly

## Next Steps

### Batch Evaluation

Create a batch script to test all Level 1 problems:

```python
results = []
for problem in problems:
    # Generate and evaluate
    result = process_problem(problem)
    results.append(result)

# Calculate metrics
success_rate = sum(r['correctness'] for r in results) / len(results)
avg_speedup = mean([r['speedup'] for r in results if r['correctness']])
```

### Optimization Loop

Integrate with optimization frameworks like the one in `solution_PS.py`:

```python
from opto.optimizers import OptoPrimeV2

# Create agent that generates CUDA code
# Use optimizer to improve over multiple iterations
# Use Modal evaluation as the feedback signal
```

### Custom Prompts

Experiment with different prompting strategies:

```python
prompt = f"""You are a CUDA optimization expert.

Task: Optimize this PyTorch operation for maximum GPU performance.

Reference PyTorch code:
{ref_arch_src}

Requirements:
1. Use shared memory for reductions
2. Vectorize memory access (float4)
3. Optimize thread/block configuration
4. Minimize global memory traffic

Generate optimized CUDA kernel:"""
```

## Architecture

```
kernelbench_simple_test.py
    │
    ├─> load_kernelbench_dataset()
    │   └─> HuggingFace datasets API
    │
    ├─> generate_cuda_with_llm()
    │   └─> LiteLLM API (Gemini/GPT/Claude)
    │
    └─> evaluate_with_modal()
        └─> Modal AI (cloud GPU)
            ├─> Compile with load_inline()
            ├─> Test correctness
            └─> Benchmark performance
```

This script provides a complete, working example of the KernelBench evaluation pipeline!
