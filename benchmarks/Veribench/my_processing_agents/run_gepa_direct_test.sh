#!/bin/bash

# Test script: Run GEPA direct for task 10 only (single run)
# Use this to verify everything works before running the full batch

MODEL="claude-3.5-sonnet"
MAX_ITERATIONS=20
TASK_IDX=10

echo "=================================================="
echo "GEPA Direct Code Evolution - Test Run"
echo "Task: ${TASK_IDX}"
echo "Model: ${MODEL}"
echo "Max iterations: ${MAX_ITERATIONS}"
echo "=================================================="
echo ""

uv run python my_processing_agents/solution_GEPA_direct_with_LLMjudge.py \
    --task_idx ${TASK_IDX} \
    --model ${MODEL} \
    --max_iterations ${MAX_ITERATIONS} \
    --save_results \
    --save_name gepa_direct \
    --log_dir results_llm_judge/gepa_direct_logs/task_${TASK_IDX}

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Test run completed successfully!"
    echo "Check results at: results_llm_judge/gepa_direct/task_${TASK_IDX}_result.json"
else
    echo ""
    echo "✗ Test run failed!"
fi
