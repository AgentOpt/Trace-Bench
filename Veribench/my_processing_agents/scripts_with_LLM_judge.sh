#!/bin/bash

# Easy_set category task IDs (41 tasks total)
# Run all easy_set tasks (10-50)
cd Trace-Bench/Veribench

uv run python my_processing_agents/solution_PS_withLLMjudge.py \
        --task_idx 10 \
        --num_steps 2 \
        --num_threads 30 \
        --log_frequency 1 \
        --test_frequency 1 \
        --num_candidates 5 \
        --algorithm PS_epsNet_Summarizer \
        --epsilon 0.02 \
        --epsilon_for_summarizer 0.02 \
        --with_llm_judge \
        --use_wandb \
        --project_name "Veribench-POLCA" 

uv run python my_processing_agents/solution_PS_withLLMjudge.py \
        --task_idx 10 \
        --num_steps 20 \
        --num_threads 30 \
        --log_frequency 1 \
        --test_frequency 1 \
        --num_candidates 5 \
        --algorithm PS_epsNet_Summarizer \
        --epsilon 0.02 \
        --epsilon_for_summarizer 0.02 \
        --with_llm_judge \
        --use_wandb \
        --project_name "Veribench-POLCA" 



## OpenEvolve
uv run python my_processing_agents/solution_openevolve_with_LLMjudge.py \
                --task_idx 10 \
                --save_results 
for task_idx in {10..50}; do
    uv run python my_processing_agents/solution_openevolve_with_LLMjudge.py \
                --task_idx $task_idx \
                --save_results
done

# GEPA
uv run python my_processing_agents/solution_GEPA_direct_with_LLMjudge.py \
        --task_idx 10 \
        --max_metric_calls 50 \
        --save_results \
        --run 1 \
        --log_dir results_llm_judge/gepa \
        --save_name gepa_1

for task_idx in {10..50}; do
    uv run python my_processing_agents/solution_GEPA_direct_with_LLMjudge.py \
        --task_idx $task_idx \
        --max_metric_calls 50 \
        --save_results \
        --run 1 \
        --log_dir results_llm_judge/gepa \
        --save_name gepa_1
done