# DSPy-Based Lean 4 Code Generation

## Overview

`solution_dspy.py` is an automated agent that translates Python programs to Lean 4 code using **DSPy** (Declarative Self-improving Language Programs). It uses a sequential refinement approach with compilation feedback to iteratively improve the generated Lean code until it compiles successfully.

## What Does It Do?

The script automates the process of:

1. **Initial Generation**: Takes a Python program and generates an initial Lean 4 translation
2. **Compilation Testing**: Tests the generated Lean code using the VeribenchGuide verifier
3. **Iterative Refinement**: If compilation fails, uses the error feedback to generate an improved version
4. **Success Detection**: Continues refining until the code compiles successfully (score = 1.0) or reaches the maximum number of attempts

## Architecture

### Core Components

#### 1. **DSPyLeanAgent** (Main Agent Class)

A DSPy Module that manages the code generation and refinement process:

- **Generator**: `dspy.ChainOfThought` module for initial code generation
  - Uses `LeanCodeGenerator` signature
  - Generates complete Lean 4 code from Python input
  
- **Refiner**: `dspy.ChainOfThought` module for fixing compilation errors
  - Uses `LeanCodeRefiner` signature
  - Takes current code + compilation feedback → produces improved code

#### 2. **DSPy Signatures**

```python
class LeanCodeGenerator(dspy.Signature):
    python_program = dspy.InputField(desc="Python program to translate")
    lean_code = dspy.OutputField(desc="Complete, compilable Lean 4 code")

class LeanCodeRefiner(dspy.Signature):
    python_program = dspy.InputField(desc="Original Python program")
    current_lean_code = dspy.InputField(desc="Current Lean 4 code that failed to compile")
    compilation_feedback = dspy.InputField(desc="Compilation errors and feedback")
    lean_code = dspy.OutputField(desc="Improved Lean 4 code that fixes the errors")
```

#### 3. **Sequential Optimization Loop**

The `sequential_optimization()` function implements the main refinement loop:

```
For each attempt (1 to max_attempts):
  1. Generate/refine Lean code using DSPy agent
  2. Evaluate with VeribenchGuide (get score + feedback)
  3. Track history (attempt, score, feedback)
  4. Update best solution if score improved
  5. If score == 1.0 → SUCCESS, return
  6. Otherwise → use feedback for next iteration
```

## How It Works

### Step-by-Step Process

1. **Initialization**
   - Configure DSPy with the specified LLM (Claude, GPT, Gemini, etc.)
   - Load the Python program from Veribench dataset
   - Initialize DSPyLeanAgent and VeribenchGuide

2. **First Attempt (Generation)**
   - Agent's generator module creates initial Lean 4 code
   - Code is evaluated by VeribenchGuide
   - Returns score (0.0-1.0) and compilation feedback

3. **Subsequent Attempts (Refinement)**
   - If score < 1.0, agent's refiner module receives:
     - Original Python program
     - Current Lean code that failed
     - Compilation error messages
   - Refiner generates improved Lean code
   - Process repeats

4. **Termination**
   - **Success**: Score reaches 1.0 (code compiles correctly)
   - **Failure**: Maximum attempts reached without success

5. **Results Saving**
   - Always saves a summary with key metrics
   - Optionally saves full results with history (if `--save_results` flag is used)

## Usage

### Basic Command

```bash
uv run python my_processing_agents/solution_dspy.py --task_idx 14
```

### With Verbose Output

```bash
uv run python my_processing_agents/solution_dspy.py \
    --task_idx 14 \
    --verbose \
    --save_results
```

### With Custom Model and Max Attempts

```bash
uv run python my_processing_agents/solution_dspy.py \
    --task_idx 14 \
    --model gpt-4o \
    --max_attempts 100 \
    --verbose \
    --save_results
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--task_idx` | int | 14 | Task index from Veribench dataset (0-100+) |
| `--model` | str | `claude-3.5-sonnet` | LLM model name (auto-prefixes provider) |
| `--max_attempts` | int | 50 | Maximum number of refinement attempts |
| `--verbose` | flag | False | Print detailed logs during optimization |
| `--save_results` | flag | False | Save full results with history to JSON |

### Supported Models

The script automatically adds the correct provider prefix:

- **Anthropic**: `claude-3.5-sonnet`, `claude-3-opus-20240229`, etc.
- **OpenAI**: `gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo`, etc.
- **Google**: `gemini-1.5-pro`, `gemini-1.5-flash`, etc.

Or specify with explicit provider: `anthropic/claude-3.5-sonnet`

## Output

### Console Output

```
Configuring DSPy with anthropic/claude-3.5-sonnet...

Loading task 14 from Veribench dataset...
Task loaded successfully. Task ID: 14

Python code to translate (first 500 chars):
======================================================================
[Python code snippet]
======================================================================

Initializing DSPy agent and VeribenchGuide...

Starting DSPy sequential optimization:
  Model: anthropic/claude-3.5-sonnet
  Max attempts: 50
  Target: Achieve score 1.0 (successful Lean compilation)

[If --verbose is enabled, shows each attempt]

Saved result to results/dspy_task_14_result.json

======================================================================
Task 14 | Duration: 0.5s | Model: claude-3.5-sonnet
======================================================================
✓ SUCCESS: Reached score 1.0 at attempt 4/50
======================================================================
```

### Saved Files

#### Summary File (Always Created)
**Location**: `results/dspy_task_{task_idx}_summary.json`

```json
{
  "task_idx": 14,
  "task_id": "14",
  "success": true,
  "final_score": 1.0,
  "num_metric_calls": 4,
  "duration_seconds": 0.52,
  "model": "anthropic/claude-3.5-sonnet"
}
```

#### Full Results File (Only with --save_results)
**Location**: `results/dspy_task_{task_idx}_result.json`

```json
{
  "task_idx": 14,
  "task_id": "14",
  "success": true,
  "attempts": 4,
  "best_score": 1.0,
  "best_lean_code": "[Complete Lean 4 code]",
  "duration_seconds": 0.52,
  "history": [
    {
      "attempt": 1,
      "score": 0.0,
      "feedback": "Lean compilation FAILED with 2 error(s)..."
    },
    {
      "attempt": 2,
      "score": 0.0,
      "feedback": "Lean compilation FAILED with 1 error(s)..."
    },
    {
      "attempt": 3,
      "score": 0.0,
      "feedback": "Lean compilation FAILED with 1 error(s)..."
    },
    {
      "attempt": 4,
      "score": 1.0,
      "feedback": "The answer is correct!"
    }
  ],
  "settings": {
    "model": "anthropic/claude-3.5-sonnet",
    "max_attempts": 50
  }
}
```

## Key Features

### 1. **Automatic Retry Logic**
- Uses `litellm.num_retries = 3` for handling rate limits
- 300-second request timeout for long-running LLM calls

### 2. **Reproducibility**
- Sets random seeds for NumPy and PyTorch
- Ensures consistent results across runs

### 3. **System Prompts with Examples**
- Uses `SYSTEM_PROMPT_WITH_EXAMPLES` for better code generation
- Provides instructions for complete, compilable Lean 4 code

### 4. **Error Handling**
- Gracefully handles generation errors
- Continues optimization even if individual attempts fail

### 5. **Best Solution Tracking**
- Always keeps track of the best score achieved
- Returns best solution even if perfect score not reached

## Comparison with Other Solutions

### vs. `solution_PS.py` (PrompterSketch)
- **DSPy**: Uses structured DSPy modules with signatures
- **PS**: Uses Trace optimization framework with text-based prompting
- **DSPy**: Two separate modules (generator + refiner)
- **PS**: Single prompt optimized through gradient descent

### vs. `solution_GEPA.py` (GEPA)
- **DSPy**: Sequential refinement based on feedback
- **GEPA**: Ensemble of multiple agents with voting
- **DSPy**: Focuses on single-agent improvement
- **GEPA**: Leverages diversity across multiple agents

### vs. `solution_opt.py` (Basic Optimization)
- **DSPy**: Framework-based structured approach
- **Opt**: Direct LLM calls with manual prompt engineering
- **DSPy**: Automatic prompt management
- **Opt**: Manual prompt construction

## Requirements

- Python 3.8+
- DSPy library
- litellm for LLM integration
- VeribenchGuide for Lean compilation testing
- Valid API keys for chosen LLM provider (set in `secrets_local.py`)

## Environment Variables

Set in `my_processing_agents/secrets_local.py`:

```python
ANTHROPIC_API_KEY = "your-key-here"  # For Claude models
OPENAI_API_KEY = "your-key-here"     # For GPT models
GOOGLE_API_KEY = "your-key-here"     # For Gemini models
```

## Example Workflow

```bash
# Test on a single task with verbose output
uv run python my_processing_agents/solution_dspy.py \
    --task_idx 14 \
    --verbose \
    --save_results

# Run on multiple tasks (use a loop)
for i in {0..20}; do
    uv run python my_processing_agents/solution_dspy.py \
        --task_idx $i \
        --save_results
done

# Try different models
uv run python my_processing_agents/solution_dspy.py \
    --task_idx 14 \
    --model gpt-4o \
    --save_results
```

## Performance Metrics

The script tracks:
- **Success rate**: Whether score 1.0 was achieved
- **Number of attempts**: How many refinement iterations needed
- **Duration**: Total time from start to finish
- **Best score**: Highest score achieved across all attempts
- **History**: Complete trace of all attempts with scores and feedback

## Troubleshooting

### Error: "unrecognized arguments: --save_results"
- Ensure the file is saved with the correct argument definition
- Check that line 229-230 contains the `--save_results` argument

### Error: API rate limits
- Script automatically retries up to 3 times
- Consider adding delays between tasks if running batch jobs

### Error: Lean compilation timeout
- Increase `litellm.request_timeout` if needed
- Some complex programs may require longer compilation time

## References

- **DSPy Documentation**: https://dspy-docs.vercel.app/
- **Veribench**: Verification benchmark for Python-to-Lean translation
- **VeribenchGuide**: Feedback mechanism for Lean code evaluation

## License

Same as parent project.

