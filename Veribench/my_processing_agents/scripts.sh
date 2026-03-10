# Basic training
python optimize_veribench_agent.py --num_train_samples 10 --num_steps 5

# With WandB logging
python optimize_veribench_agent.py --num_train_samples 20 --use_wandb --project_name veribench-experiment

python my_processing_agents/optimize_veribench_agent.py \
        --num_train_samples 10 \
        --num_validate_samples 10 \
        --num_test_samples 10 \
        --num_candidates 5 \
        --batch_size 2 \
        --num_batches 1 \
        --num_steps 101 \
        --num_threads 20 \
        --memory_update_frequency 0 \
        --num_eval_samples 1 \
        --test_frequency 1 \
        --log_frequency 1 \
        --num_proposals 1 \
        --project_name "veribench-debug" \
        --run_name "debug-num_candidates-5" \
        --optoprime_version v2 \
        --use_wandb
# PS
python my_processing_agents/optimize_veribench_agent.py \
        --num_train_samples 10 \
        --num_validate_samples 10 \
        --num_test_samples 10 \
        --num_candidates 5 \
        --batch_size 2 \
        --num_batches 1 \
        --num_steps 101 \
        --num_threads 20 \
        --memory_update_frequency 0 \
        --num_eval_samples 1 \
        --test_frequency 1 \
        --log_frequency 1 \
        --num_proposals 1 \
        --project_name "veribench" \
        --run_name "PS" \
        --optoprime_version v2 \
        --use_wandb



# PS + EpsnetPS
python my_processing_agents/optimize_veribench_agent.py \
        --num_train_samples 10 \
        --num_validate_samples 10 \
        --num_test_samples 10 \
        --num_candidates 5 \
        --batch_size 2 \
        --num_batches 1 \
        --num_steps 101 \
        --num_threads 20 \
        --memory_update_frequency 0 \
        --num_eval_samples 1 \
        --test_frequency 1 \
        --log_frequency 1 \
        --num_proposals 1 \
        --project_name "veribench-debug" \
        --run_name "PS+EpsnetPS" \
        --optoprime_version v2 \
        --use_wandb \
        --ablation \
        --epsnetPS 

# PS + EpsnetPS + Summarizer
python my_processing_agents/optimize_veribench_agent.py \
        --num_train_samples 10 \
        --num_validate_samples 10 \
        --num_test_samples 10 \
        --num_candidates 5 \
        --batch_size 2 \
        --num_batches 1 \
        --num_steps 101 \
        --num_threads 20 \
        --memory_update_frequency 0 \
        --num_eval_samples 1 \
        --test_frequency 1 \
        --log_frequency 1 \
        --num_proposals 1 \
        --project_name "veribench-debug" \
        --run_name "PS+EpsnetPS+Summarizer" \
        --optoprime_version v2 \
        --use_wandb \
        --ablation \
        --epsnetPS \
        --use_summarizer

python my_processing_agents/optimize_veribench_agent.py \
        --num_train_samples 10 \
        --num_validate_samples 10 \
        --num_test_samples 10 \
        --num_candidates 5 \
        --batch_size 2 \
        --num_batches 1 \
        --num_steps 101 \
        --num_threads 50 \
        --memory_update_frequency 0 \
        --num_eval_samples 1 \
        --test_frequency 1 \
        --log_frequency 1 \
        --num_proposals 1 \
        --project_name "veribench-debug" \
        --run_name "PS+EPSNet+Summarizer" \
        --optoprime_version v2 \
        --use_wandb \
        --ablation \
        --epsnetPS \
        --use_summarizer


python my_processing_agents/optimize_veribench_agent.py \
        --num_train_samples 10 \
        --num_validate_samples 10 \
        --num_test_samples 10 \
        --num_candidates 1 \
        --batch_size 1 \
        --num_batches 10 \
        --num_steps 21 \
        --num_threads 50 \
        --memory_update_frequency 0 \
        --num_eval_samples 1 \
        --test_frequency 1 \
        --log_frequency 1 \
        --num_proposals 1 \
        --project_name "veribench-10-tasks" \
        --run_name "PS-candidates-1-batch_size-1-num_batches-10-claude-3-5-sonnet" \
        --optoprime_version v2 \
        --use_wandb \
        --ablation

for task_idx in {0..119}; do
        python my_processing_agents/solution_PS.py \
                --task_idx $task_idx \
                --num_steps 10 \
                --num_threads 30 \
                --log_frequency 1 \
                --test_frequency 1 \
                --num_candidates 5 \
                --use_wandb \
                --algorithm PS \
                --project_name "veribench-per-task"
done

for task_idx in {0..139}; do
        uv run python my_processing_agents/solution_PS.py \
                --task_idx $task_idx \
                --num_steps 10 \
                --num_threads 30 \
                --log_frequency 1 \
                --test_frequency 1 \
                --num_candidates 5 \
                --use_wandb \
                --algorithm PS_Summarizer \
                --project_name "veribench-per-task_PS_Summarizer"
done

uv run python my_processing_agents/solution_PS.py \
                --task_idx 2 \
                --num_steps 10 \
                --num_threads 30 \
                --log_frequency 1 \
                --test_frequency 1 \
                --num_candidates 5 \
                --algorithm PS \
                --with_unit_tests

for task_idx in {0..139}; do
        uv run python my_processing_agents/solution_PS.py \
                --task_idx $task_idx \
                --num_steps 20 \
                --num_threads 30 \
                --log_frequency 1 \
                --test_frequency 1 \
                --num_candidates 5 \
                --use_wandb \
                --algorithm PS_epsNet_Summarizer \
                --project_name "new_veribench-per-task_PS_epsNet_Summarizer"
done
for task_idx in {0..139}; do
        uv run python my_processing_agents/solution_PS.py \
                --task_idx $task_idx \
                --num_steps 11 \
                --num_threads 30 \
                --log_frequency 1 \
                --test_frequency 1 \
                --num_candidates 5 \
                --use_wandb \
                --algorithm PS \
                --project_name "veribench-per-task-PS2" 
done    
for task_idx in {0..139}; do
        uv run python my_processing_agents/solution_PS.py \
                --task_idx $task_idx \
                --num_steps 11 \
                --num_threads 30 \
                --log_frequency 1 \
                --test_frequency 1 \
                --num_candidates 5 \
                --use_wandb \
                --algorithm PS \
                --project_name "veribench-per-task-PS3" 
done    


uv run python my_processing_agents/solution_PS.py \
                --task_idx 2 \
                --num_steps 11 \
                --num_threads 30 \
                --log_frequency 1 \
                --test_frequency 1 \
                --num_candidates 5 
                # --use_wandb \
                # --algorithm PS \
                # --project_name "veribench-per-task-PS2" \

uv run python my_processing_agents/solution_PS.py \
                --task_idx 131 \
                --num_steps 50 \
                --num_threads 30 \
                --log_frequency 1 \
                --test_frequency 1 \
                --num_candidates 5 \
                --use_wandb \
                --algorithm PS_epsNet \
                --epsilon 0.02 \
                --project_name "veribench-debug" 
# Dec 25 noon
for task_idx in {0..139}; do
        uv run python my_processing_agents/solution_PS.py \
                --task_idx $task_idx \
                --num_steps 20 \
                --num_threads 30 \
                --log_frequency 1 \
                --test_frequency 1 \
                --num_candidates 5 \
                --use_wandb \
                --algorithm PS_epsNet \
                --epsilon 0.02 \
                --project_name "veribench-PS_epsNet-3"
done
for task_idx in {0..139}; do
        uv run python my_processing_agents/solution_PS.py \
                --task_idx $task_idx \
                --num_steps 20 \
                --num_threads 30 \
                --log_frequency 1 \
                --test_frequency 1 \
                --num_candidates 5 \
                --use_wandb \
                --algorithm PS_Summarizer \
                --epsilon 0.0 \
                --project_name "veribench-PS_Summarizer-3"
done

for task_idx in {0..139}; do
        uv run python my_processing_agents/solution_dspy3.py \
                --task_idx $task_idx \
                --save_results
done

for task_idx in {113..139}; do
        uv run python my_processing_agents/solution_openevolve.py \
                --task_idx $task_idx \
                --save_results
done
for task_idx in {85..139}; do
        uv run python my_processing_agents/solution_openevolve3.py \
                --task_idx $task_idx \
                --save_results
done

# uv run python my_processing_agents/solution_PS.py \
#                 --task_idx 3 \
#                 --num_steps 10 \
#                 --num_threads 30 \
#                 --log_frequency 1 \
#                 --test_frequency 1 \
#                 --num_candidates 5 \
#                 --algorithm PS \
#                 --with_unit_tests
uv run python my_processing_agents/solution_dspy.py \
                --task_idx 10 \
                --save_results

# gepa 
# debug
for task_idx in {0..139}; do
    uv run python my_processing_agents/solution_GEPA.py --task_idx $task_idx --save_results --log_dir results/running_log/gepa_1 --max_metric_calls 50 --save_name gepa_1
done



for task_idx in {0..139}; do
    uv run python my_processing_agents/solution_GEPA.py --task_idx $task_idx --save_results --log_dir gepa_running_log/results/gepa_debug --max_metric_calls 50 --save_name gepa_1
done

for task_idx in {0..139}; do
    uv run python my_processing_agents/solution_GEPA.py --task_idx $task_idx --save_results --log_dir gepa_running_log/results/gepa_debug --max_metric_calls 150 --save_name gepa_2
done

for task_idx in {0..139}; do
    uv run python my_processing_agents/solution_GEPA.py --task_idx $task_idx --save_results --log_dir gepa_running_log/results/gepa_debug --max_metric_calls 150 --save_name gepa_3
done