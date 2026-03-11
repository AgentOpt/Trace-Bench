#!/bin/bash

# Run GEPA direct code evolution for tasks 10-50, with 3 runs each
# Results saved to gepa_1, gepa_2, gepa_3 directories

# Configuration
MODEL="claude-3.5-sonnet"
MAX_ITERATIONS=20  # or use --max_metric_calls 50

# Define task range
START_TASK=10
END_TASK=50

echo "=================================================="
echo "GEPA Direct Code Evolution - Tasks ${START_TASK} to ${END_TASK}"
echo "Model: ${MODEL}"
echo "Max iterations: ${MAX_ITERATIONS}"
echo "Runs: 3 (gepa_1, gepa_2, gepa_3)"
echo "=================================================="
echo ""

# Run 1: gepa_1
echo "=== Starting Run 1 (gepa_1) ==="
for task_idx in $(seq ${START_TASK} ${END_TASK}); do
    echo "--- Task ${task_idx} (Run 1/3) ---"
    uv run python my_processing_agents/solution_GEPA_direct_with_LLMjudge.py \
        --task_idx ${task_idx} \
        --model ${MODEL} \
        --max_metric_calls 50 \
        --save_results \
        --save_name gepa_1 \
        --log_dir results_llm_judge/gepa_1_logs/task_${task_idx}
    
    # Check exit code
    if [ $? -ne 0 ]; then
        echo "ERROR: Task ${task_idx} (Run 1) failed!"
    else
        echo "✓ Task ${task_idx} (Run 1) completed"
    fi
    echo ""
done

echo "=== Run 1 (gepa_1) Complete ==="
echo ""

# Run 2: gepa_2
echo "=== Starting Run 2 (gepa_2) ==="
for task_idx in $(seq ${START_TASK} ${END_TASK}); do
    echo "--- Task ${task_idx} (Run 2/3) ---"
    uv run python my_processing_agents/solution_GEPA_direct_with_LLMjudge.py \
        --task_idx ${task_idx} \
        --model ${MODEL} \
        --max_iterations ${MAX_ITERATIONS} \
        --save_results \
        --save_name gepa_2 \
        --log_dir results_llm_judge/gepa_2_logs/task_${task_idx}
    
    # Check exit code
    if [ $? -ne 0 ]; then
        echo "ERROR: Task ${task_idx} (Run 2) failed!"
    else
        echo "✓ Task ${task_idx} (Run 2) completed"
    fi
    echo ""
done

echo "=== Run 2 (gepa_2) Complete ==="
echo ""

# Run 3: gepa_3
echo "=== Starting Run 3 (gepa_3) ==="
for task_idx in $(seq ${START_TASK} ${END_TASK}); do
    echo "--- Task ${task_idx} (Run 3/3) ---"
    uv run python my_processing_agents/solution_GEPA_direct_with_LLMjudge.py \
        --task_idx ${task_idx} \
        --model ${MODEL} \
        --max_iterations ${MAX_ITERATIONS} \
        --save_results \
        --save_name gepa_3 \
        --log_dir results_llm_judge/gepa_3_logs/task_${task_idx}
    
    # Check exit code
    if [ $? -ne 0 ]; then
        echo "ERROR: Task ${task_idx} (Run 3) failed!"
    else
        echo "✓ Task ${task_idx} (Run 3) completed"
    fi
    echo ""
done

echo "=== Run 3 (gepa_3) Complete ==="
echo ""

echo "=================================================="
echo "All runs complete!"
echo "Results saved to:"
echo "  - results_llm_judge/gepa_1/"
echo "  - results_llm_judge/gepa_2/"
echo "  - results_llm_judge/gepa_3/"
echo "=================================================="
