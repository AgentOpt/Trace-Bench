# Optimize a single Lean solution using PrioritySearch algorithm
# The agent is a container for Lean code which gets optimized until it compiles correctly

# Optimization process (for one run, only one task used)
# 1. Initialize the agent with the initial Lean code
# 2. Pick candidates. The sample process is very quick, if all scores are 0, the sample/evaluation is just call the guide for one time, the score should still be 0. 
# 3. Optimizer propose new parameters. According to the picked candidates, with its feedback.

import sys
import os
# Add project root to sys.path so we can import from my_processing_agents and guide
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
np.random.seed(10)
torch.manual_seed(10)

import time
import argparse

from opto import trace
from opto.trainer.loggers import DefaultLogger, WandbLogger
from opto.optimizers import OptoPrimeV2

from my_processing_agents.solution_opt import extract_python_code, load_single_task
from guide.guide import VeribenchGuide , VeribenchGuidewithUnitTests , VeribenchGuidewithLLMJudge
import litellm
litellm.drop_params = True
litellm.suppress_debug_info = True

from my_processing_agents import secrets_local  # Load environment variables from gitignored file
from my_processing_agents.system_prompts import SYSTEM_PROMPT, EXAMPLES, SYSTEM_PROMPT_WITH_EXAMPLES

@trace.model
class LeanCodeAgent:
    """
    An agent that is a container for Lean code. 
    The Lean code is a trainable parameter that gets optimized.
    """

    def __init__(self):
        """
        Initialize the agent with initial Lean code.
        """
        initial_lean_code = "-- Lean 4 translation of the Python program"
        self.lean_code = trace.node(initial_lean_code, trainable=True)

    @trace.bundle()
    def get_lean_code(self, python_program: str, lean_code: str) -> str:
        """
        Get the Lean code. This function serves as the traced computation.
        
        Args:
            python_program: The original Python program extracted from dataset (for context in optimization)
            lean_code: The current Lean code (trainable parameter)
            
        Returns:
            The Lean code that should compile correctly as a translation of the Python program
        """
        return lean_code

    def forward(self, task: str) -> str:
        """
        Forward pass that returns the current Lean code.
        
        Args:
            task: The Python program (already extracted from user_query)
        """
        return self.get_lean_code(task, self.lean_code)


def create_single_task_dataset(task_idx: int):
    """
    Create a dataset with a single task for PrioritySearch training.
    
    Args:
        task_idx: Index of the task to load
        
    Returns:
        Dictionary with 'inputs' (extracted Python code) and 'infos' (task ids)
    """
    task = load_single_task(task_idx)
    python_code = extract_python_code(task['user_query'])
    return {
        'inputs': [python_code],
        'infos': [task_idx]
    }
# from veribench_dataset_utils.create_datasets import create_single_task_dataset


def main():
    parser = argparse.ArgumentParser(description='Optimize a single Lean solution using PrioritySearch')
    parser.add_argument('--task_idx', type=int, default=0, help='Task index from the Veribench dataset')
    parser.add_argument('--num_steps', type=int, default=50, help='Maximum number of optimization steps')
    parser.add_argument('--num_candidates', type=int, default=1, help='Number of candidates for exploration')
    parser.add_argument('--num_threads', type=int, default=20, help='Number of threads for parallel processing')
    parser.add_argument('--num_proposals', type=int, default=1, help='Number of proposals for each candidate')
    parser.add_argument('--log_frequency', type=int, default=1, help='How often to log results')
    parser.add_argument('--test_frequency', type=int, default=None, help='How often to run evaluation')
    parser.add_argument('--with_llm_judge', action='store_true', default=False,
                       help='Whether to run with LLM judge')
    parser.add_argument('--algorithm', type=str, default='PS',choices=['PS','PS_Summarizer','PS_epsNet_Summarizer','PS_epsNet'], help='Algorithm to use')

    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon value for EpsilonNetPS')

    parser.add_argument('--with_unit_tests', action='store_true', default=False,
                       help='Whether to run with unit tests')

    # Logger parameters
    parser.add_argument('--use_wandb', action='store_true', default=False,
                       help='Whether to use Weights & Biases for logging')
    parser.add_argument('--project_name', type=str, default='veribench-single-task',
                       help='Name of the W&B project')
    parser.add_argument('--run_name', type=str, default=None,
                       help='Name of the W&B run (default: task_{task_idx})')
    
    args = parser.parse_args()

    task_idx = args.task_idx
    num_steps = args.num_steps
    num_threads = args.num_threads
    log_frequency = args.log_frequency
    test_frequency = args.test_frequency
    num_proposals = args.num_proposals
    # Step 1: Load the task
    print(f"Loading task {task_idx} from Veribench dataset...")
    task = load_single_task(task_idx)
    user_query = task['user_query']
    # python_program = extract_python_code(user_query)
    print(f"Task loaded successfully. Task ID: {task['task_id']}")
    
    # Step 2: Initialize the agent with a dummy Lean code string
    
    agent = LeanCodeAgent()
    
    # Step 4: Initialize the optimizer with a clear objective
    optimizer = OptoPrimeV2(agent.parameters(), max_tokens=8192, initial_var_char_limit=10000)
    optimizer.objective = f"""Your task is to produce valid Lean 4 code that correctly translates a Python program.

You will see:
- The original Python program (in # Inputs)
- The current Lean 4 code attempt (in # Variables)
- Compilation feedback: either "success" or error messages from the Lean 4 compiler (in # Feedback)

Goal: Produce Lean 4 code that compiles successfully (score = 1).
- If the current Lean code is a dummy placeholder (e.g., just a comment), generate a complete Lean 4 translation from the Python program.
- Otherwise, fix the compilation errors in the current Lean code while preserving correct parts.

Key rules:
- Preserve the algorithm logic from the Python program
- Use correct Lean 4 syntax and type annotations
- Output only valid Lean 4 code

Original System and User Prompts which may be helpful:

{SYSTEM_PROMPT_WITH_EXAMPLES}

The translated Lean 4 code should be a faithful representation of the Python code.
It should be correct and compiles.
If a theorem keeps giving error, you can use := sorry to skip it.
"""

    # Step 5: Initialize guide and logger
    if args.with_LLM_judge:
        guide = VeribenchGuidewithLLMJudge()
    elif args.with_unit_tests:
        guide = VeribenchGuidewithUnitTests()
    else:
        guide = VeribenchGuide()
    
    # Create config dictionary for logging
    config_dict = {
        'task_idx': task_idx,
        'task_id': task['task_id'],
        'num_steps': num_steps,
        'num_candidates': args.num_candidates,
        'num_threads': num_threads,
        'num_proposals': num_proposals,
        'log_frequency': log_frequency,
        'test_frequency': test_frequency,
    }
    
    # Set run name if not provided
    run_name = args.run_name if args.run_name else f"task_{task_idx}"
    
    # Initialize logger based on wandb flag
    if args.use_wandb:
        logger = WandbLogger(project=args.project_name, verbose=True, name=run_name, config=config_dict)
    else:
        logger = DefaultLogger(verbose=True)
    
    # Step 6: Create single-task dataset
    train_dataset = create_single_task_dataset(task_idx)
    
    # Step 7: Create PrioritySearch algorithm
    print("\nCreating PrioritySearch algorithm...")
    if args.use_wandb:
        print(f"Using Weights & Biases logging: project='{args.project_name}', run='{run_name}'")
    else:
        print("Using DefaultLogger (no W&B logging)")

    # Algorithm selection
    if args.algorithm == 'PS':
        from opto.features.priority_search.priority_search_ablation import PS_veribench 
        algorithm = PS_veribench(
            epsilon=0,
            use_summarizer=False,
            agent=agent,
            optimizer=optimizer,
            logger=logger,
            num_threads=num_threads,
        )
    elif args.algorithm == 'PS_Summarizer':
        from opto.features.priority_search.priority_search_ablation import PS_veribench
        algorithm = PS_veribench(
            epsilon=args.epsilon,
            use_summarizer=True,
            summarizer_model_name="claude-3.5-sonnet",
            agent=agent,
            optimizer=optimizer,
            logger=logger,
            num_threads=num_threads
        )
    elif args.algorithm == 'PS_epsNet_Summarizer':
        from opto.features.priority_search.priority_search_ablation import PS_veribench
        algorithm = PS_veribench(
            epsilon=args.epsilon,
            use_summarizer=True,
            summarizer_model_name="claude-3.5-sonnet",
            agent=agent,
            optimizer=optimizer,
            logger=logger)
    elif args.algorithm == 'PS_epsNet':
        from opto.features.priority_search.priority_search_ablation import PS_veribench
        algorithm = PS_veribench(
            agent=agent,
            optimizer=optimizer,
            logger=logger,
            epsilon=args.epsilon)
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")
    
    # Step 8: Run PrioritySearch training
    print(f"\nStarting PrioritySearch optimization (max {num_steps} steps)...")
    print(f"Target: Achieve score 1.0 (successful compilation)")
    
    start_time = time.time()
    
    algorithm.train(
        guide=guide,
        train_dataset=train_dataset,
        validate_dataset=train_dataset,  # Same task for validation
        test_dataset=train_dataset,       # Same task for test
        batch_size=1,
        num_batches=1,
        num_steps=num_steps,
        num_threads=num_threads,
        num_eval_samples=1,
        validate_exploration_candidates=False,
        use_best_candidate_to_explore=False,
        num_candidates=args.num_candidates,
        num_proposals=num_proposals,
        score_function='mean',
        log_frequency=log_frequency,
        test_frequency=test_frequency,
    )
    
    duration = time.time() - start_time
    print(f"\nOptimization completed in {duration:.2f} seconds")
    
    # Step 9: Print final result
    final_lean_code = agent.lean_code.data
    final_score, _ = guide.get_feedback(task=user_query, response=final_lean_code)
    
    print(f"\n{'='*70}")
    print(f"FINAL RESULT - Score: {final_score}")
    print(f"{'='*70}")
    if final_score == 1.0:
        print("SUCCESS! Lean code compiles correctly!")
    else:
        print("Did not reach target score 1.0")
    print(f"\nFinal Lean code:")
    print("-" * 50)
    print(final_lean_code)
    print("-" * 50)


if __name__ == "__main__":
    main()
