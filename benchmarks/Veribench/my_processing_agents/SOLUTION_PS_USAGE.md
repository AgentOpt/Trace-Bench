# Solution PS - Single Task Optimization with W&B Logging

## Overview

`solution_PS.py` optimizes a single Lean solution using the PrioritySearch algorithm. It now supports Weights & Biases (W&B) logging for experiment tracking.

## Basic Usage

### Without W&B Logging (Default)

```bash
python -m my_processing_agents.solution_PS --task_idx 0 --num_steps 50
```

This will:
- Optimize task 0 from the Veribench dataset
- Use DefaultLogger (print to console only)
- Run for maximum 50 optimization steps

### With W&B Logging

```bash
python -m my_processing_agents.solution_PS \
    --task_idx 0 \
    --num_steps 50 \
    --use_wandb \
    --project_name veribench-single-task \
    --run_name task_0_experiment_1
```

This will:
- Log all metrics to Weights & Biases
- Create/use project "veribench-single-task"
- Name the run "task_0_experiment_1"
- Track all optimization progress in W&B dashboard

## Arguments

### Task Configuration
- `--task_idx`: Task index from Veribench dataset (default: 0)
- `--num_steps`: Maximum optimization steps (default: 50)
- `--num_candidates`: Number of candidates for exploration (default: 1)
- `--num_proposals`: Number of proposals per candidate (default: 1)

### Execution Configuration
- `--num_threads`: Number of threads for parallel processing (default: 20)
- `--log_frequency`: How often to log results (default: 1)
- `--test_frequency`: How often to run evaluation (default: None)

### W&B Logging Configuration
- `--use_wandb`: Enable Weights & Biases logging (flag, default: False)
- `--project_name`: W&B project name (default: 'veribench-single-task')
- `--run_name`: W&B run name (default: 'task_{task_idx}')

## Examples

### Example 1: Quick Test Without Logging

```bash
python -m my_processing_agents.solution_PS \
    --task_idx 0 \
    --num_steps 10
```

Output:
```
Loading task 0 from Veribench dataset...
Task loaded successfully. Task ID: ...

Creating PrioritySearch algorithm...
Using DefaultLogger (no W&B logging)

Starting PrioritySearch optimization (max 10 steps)...
Target: Achieve score 1.0 (successful compilation)
...
```

### Example 2: Full Run with W&B Logging

```bash
python -m my_processing_agents.solution_PS \
    --task_idx 5 \
    --num_steps 100 \
    --num_candidates 2 \
    --num_proposals 3 \
    --use_wandb \
    --project_name my-veribench-experiments \
    --run_name task5_2candidates_3proposals
```

Output:
```
Loading task 5 from Veribench dataset...
Task loaded successfully. Task ID: ...

Creating PrioritySearch algorithm...
Using Weights & Biases logging: project='my-veribench-experiments', run='task5_2candidates_3proposals'

Starting PrioritySearch optimization (max 100 steps)...
Target: Achieve score 1.0 (successful compilation)
...
```

View results at: https://wandb.ai/YOUR_USERNAME/my-veribench-experiments/runs/...

### Example 3: Batch Experiments

Run multiple tasks with W&B logging:

```bash
# Task 0
python -m my_processing_agents.solution_PS \
    --task_idx 0 --num_steps 50 --use_wandb \
    --project_name veribench-batch-1 --run_name task_0

# Task 1
python -m my_processing_agents.solution_PS \
    --task_idx 1 --num_steps 50 --use_wandb \
    --project_name veribench-batch-1 --run_name task_1

# Task 2
python -m my_processing_agents.solution_PS \
    --task_idx 2 --num_steps 50 --use_wandb \
    --project_name veribench-batch-1 --run_name task_2
```

All runs will be tracked under the same project "veribench-batch-1" for easy comparison.

## W&B Configuration Logged

When using `--use_wandb`, the following configuration is automatically logged:

```python
{
    'task_idx': <int>,           # Task index
    'task_id': <str>,            # Task ID from dataset
    'num_steps': <int>,          # Max optimization steps
    'num_candidates': <int>,     # Number of candidates
    'num_threads': <int>,        # Number of threads
    'num_proposals': <int>,      # Number of proposals
    'log_frequency': <int>,      # Log frequency
    'test_frequency': <int>,     # Test frequency
}
```

This allows you to:
- Compare different hyperparameter configurations
- Track which task is being optimized
- Reproduce experiments with exact settings

## Output

### Console Output

```
==================================================================
FINAL RESULT - Score: 1.0
==================================================================
SUCCESS! Lean code compiles correctly!

Final Lean code:
--------------------------------------------------
theorem test : 1 + 1 = 2 := rfl
--------------------------------------------------
```

### W&B Dashboard

When using W&B, you'll see:
- **Metrics**: Score progression over steps
- **Config**: All hyperparameters used
- **Logs**: Console output
- **System**: CPU/GPU usage, runtime
- **Custom**: Any additional logged data from PrioritySearch

## Comparison with optimize_veribench_agent.py

| Feature | solution_PS.py | optimize_veribench_agent.py |
|---------|----------------|----------------------------|
| Scope | Single task | Multiple tasks |
| Dataset | One task from Veribench | Full Veribench dataset |
| W&B Logging | ✅ Yes | ✅ Yes |
| Use Case | Deep optimization of one task | Broad optimization across tasks |
| Run Time | Shorter (one task) | Longer (many tasks) |

## Tips

### 1. Naming Conventions

Use descriptive run names to easily identify experiments:

```bash
--run_name task_{idx}_candidates_{n}_steps_{s}
```

Example:
```bash
--run_name task_0_candidates_5_steps_100
```

### 2. Project Organization

Group related experiments under one project:

```bash
# Experiment 1: Different candidate counts
--project_name veribench-candidate-study

# Experiment 2: Different step counts  
--project_name veribench-step-study
```

### 3. Local Testing

Test without W&B first:

```bash
# Quick test without logging
python -m my_processing_agents.solution_PS --task_idx 0 --num_steps 5

# If it works, add W&B
python -m my_processing_agents.solution_PS --task_idx 0 --num_steps 50 --use_wandb
```

### 4. Parallel Experiments

Since each run is independent, you can run multiple in parallel:

```bash
# Terminal 1
python -m my_processing_agents.solution_PS --task_idx 0 --use_wandb --run_name task_0 &

# Terminal 2
python -m my_processing_agents.solution_PS --task_idx 1 --use_wandb --run_name task_1 &

# Terminal 3
python -m my_processing_agents.solution_PS --task_idx 2 --use_wandb --run_name task_2 &
```

All will be logged to W&B separately.

## Troubleshooting

### W&B Login Required

If you haven't logged into W&B:

```bash
wandb login
```

Or set your API key:

```bash
export WANDB_API_KEY=your_api_key_here
```

### Offline Mode

To log locally without uploading to W&B:

```bash
export WANDB_MODE=offline
python -m my_processing_agents.solution_PS --task_idx 0 --use_wandb
```

### Disable W&B

Simply omit the `--use_wandb` flag:

```bash
python -m my_processing_agents.solution_PS --task_idx 0  # No W&B
```

## Summary

✅ **Added W&B logging support**  
✅ **Same API as optimize_veribench_agent.py**  
✅ **Easy to enable/disable with --use_wandb flag**  
✅ **Automatic config tracking**  
✅ **Backward compatible (default is no W&B)**

Now you can track your single-task optimizations just like multi-task runs! 🚀

