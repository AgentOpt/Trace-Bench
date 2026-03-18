# Train agent using PrioritySearch algorithm on Veribench

# set np and torch seeds
import numpy as np
import torch
np.random.seed(10)
torch.manual_seed(10)

import os
import sys
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
import json
import time
import argparse
from typing import Union, Optional, Dict, Any, List

import opto
from opto import trace
from opto.trace.nodes import GRAPH
from opto.trace.modules import Module
from opto.utils.llm import LLM
from opto.optimizers.utils import print_color
from opto.trainer.guide import Guide
from opto.trainer.loggers import WandbLogger, DefaultLogger
from opto.trainer.utils import async_run
from guide.guide import VeribenchGuide
from my_processing_agents.lean_interpretor import lean_interpreter, remove_import_error
from my_processing_agents.system_prompts import SYSTEM_PROMPT, EXAMPLES

import litellm
litellm.drop_params = True
litellm.suppress_debug_info = True

# import nest_asyncio
# nest_asyncio.apply()

# provider = "vertex_ai"
import secrets_local

OBJECTIVE = """You are optimizing `additional_instructions` - a text parameter that guides an LLM to translate Python code into valid Lean 4 code.

CONTEXT:
- The agent receives Python code and generates Lean 4 code
- The generated code is compiled by a Lean 4 compiler
- #Feedback contains compilation results: either success or error messages

YOUR TASK:
Analyze the compilation errors in #Feedback and update `additional_instructions` to help the LLM avoid similar errors in the future.

STRATEGY:
1. If feedback says "correct": Keep the current instructions, optionally refine
2. If feedback contains errors:
   - Identify the ROOT CAUSE of each error from the error message
   - Determine what Lean 4 rule or syntax was violated
   - Add a GENERAL rule to `additional_instructions` that prevents this class of errors
   - Use concrete Lean 4 syntax examples when helpful

GUIDELINES FOR `additional_instructions`:
- Keep rules concise and actionable
- Generalize from specific errors to broader patterns
- Accumulate successful rules (don't remove what works)
- Prioritize rules that prevent the most frequent error types
- Include correct Lean 4 syntax when the error is about wrong syntax

OUTPUT: Return the complete updated `additional_instructions` text.
"""


@trace.model
class VeribenchAgent:
    """
    A simple traced agent for solving Veribench tasks.
    Uses trainable system_prompt to optimize LLM prompting.
    The system prompt is a trainable node that is optimized to produce correct Lean code.
    The task is user_query in the dataset.
    """

    def __init__(self, model: str = secrets_local.MODEL):
        self.model = model
        self.llm = LLM(model=model)
        self.system_prompt = SYSTEM_PROMPT
        self.examples = EXAMPLES
        self.additional_instructions = trace.node("Here are the additional instructions to help the agent solve the task: ", trainable=True)

    @trace.bundle()
    def solve(self, system_prompt: str, additional_instructions: str, examples: str, task_input: dict) -> str:
        """
        Generate Lean 4 code for a given task using a single LLM call.
        
        This function constructs a prompt by combining the system prompt, additional instructions,
        and examples, then sends it to the LLM to generate Lean 4 code that solves the task.
        
        Args:
            system_prompt (str): Base system prompt containing core instructions for Lean 4 code generation. This is fixed and provides the foundational guidance.
            additional_instructions (str): **TRAINABLE PARAMETER** - Extra instructions and tips that guide the LLM to produce better Lean 4 code. This parameter is optimized based on feedback from compilation errors. Modify this to:
                - Add patterns that fix common syntax errors
                - Include best practices for theorem proving
                - Provide hints for type annotations and imports
                - Add warnings about common pitfalls
            examples (str): Few-shot examples demonstrating correct Python to Lean 4 translations. These are fixed reference examples.
            task_input (dict): The user query containing the Python code and requirements to translate into Lean 4.
        
        Returns:
            str: The extracted Lean 4 code from the LLM response, or None if extraction fails. The code is extracted from ```lean ... ``` code blocks in the response. If the response is not a valid Lean 4 code, return None.
        """
        # system prompt = system prompt + additional instructions + examples  
        system_prompt = self.system_prompt + "\n\n" + additional_instructions + "\n\n" + self.examples
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_input}
        ]

        response = self.llm(messages=messages, max_tokens=8192)
        response_text = response.choices[0].message.content
        # extract the lean code from the response
        lean_pattern = r'```lean\s*\n(.*?)\n```'
        try:
            matches = re.findall(lean_pattern, response_text, re.DOTALL)
            return matches[0].strip() if matches else None
        except Exception as e:
            print_color(f"Error extracting lean code: {e}", "red")
            return None

    def forward(self, task: dict) -> str:
        """Forward pass that calls solve with trainable parameters."""
        return self.solve(self.system_prompt, self.additional_instructions, self.examples, task)




def create_dataset(num_tasks: int, offset: int = 0):
    """
    Create dataset from Veribench tasks.
    
    Args:
        num_tasks: Number of tasks to load
        offset: Starting offset in the dataset
        
    Returns:
        Dictionary with 'inputs' (user queries) and 'infos' (task ids)
    """
    from datasets import load_dataset
    
    dataset = load_dataset("allenanie/veribench_with_prompts")
    
    # Calculate the actual range
    start_idx = offset
    end_idx = offset + num_tasks
    
    tasks = dataset['train'][start_idx:end_idx]
    
    # HuggingFace slice returns dict of lists: {'user_query': [...], ...}
    inputs = tasks['user_query']
    infos = list(range(start_idx, end_idx))
    
    return {'inputs': inputs, 'infos': infos}


def main():
    """Main function for PrioritySearch training."""
    parser = argparse.ArgumentParser(description='Train agent using PrioritySearch algorithm on Veribench')
    
    # Dataset parameters
    parser.add_argument('--num_train_samples', type=int, default=10,
                       help='Number of training samples')
    parser.add_argument('--num_validate_samples', type=int, default=10,
                       help='Number of validation samples')
    parser.add_argument('--num_test_samples', type=int, default=1,
                       help='Number of test samples')
    parser.add_argument('--train_offset', type=int, default=0,
                       help='Starting offset for training samples')
    parser.add_argument('--validate_offset', type=int, default=0,
                       help='Starting offset for validation samples')
    parser.add_argument('--test_offset', type=int, default=0,
                       help='Starting offset for test samples')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Training batch size')
    parser.add_argument('--num_batches', type=int, default=1,
                       help='Number of batches to use from the dataset in each iteration')
    parser.add_argument('--num_epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--num_steps', type=int, default=5,
                       help='Number of training steps')
    parser.add_argument('--num_threads', type=int, default=1,
                       help='Number of threads for parallel processing')
    parser.add_argument('--test_frequency', type=int, default=None,
                       help='How often to run evaluation (test_frequency)')
    parser.add_argument('--log_frequency', type=int, default=1,
                       help='How often to log results')
    parser.add_argument('--save_frequency', type=int, default=None,
                       help='How often to save the agent')
    parser.add_argument('--save_path', type=str, default='checkpoints/veribench_priority_search_agent.pkl',
                       help='Path to save the agent')
    parser.add_argument('--num_eval_samples', type=int, default=1,
                       help='Number of times to evaluate each input')
    
    # PrioritySearch-specific parameters
    parser.add_argument('--num_candidates', type=int, default=2,
                       help='Number of candidates to propose for exploration')
    parser.add_argument('--num_proposals', type=int, default=1,
                       help='Number of proposals to generate per optimizer')
    parser.add_argument('--validate_exploration_candidates', action='store_true', default=False,
                       help='Whether to validate the proposed parameters for exploration')
    parser.add_argument('--use_best_candidate_to_explore', action='store_true', default=False,
                       help='Whether to use the best candidate as part of the exploration candidates')
    parser.add_argument('--memory_size', type=int, default=None,
                       help='Size of the heap memory to store the candidates; if None, no limit is set')
    parser.add_argument('--score_function', type=str, default='mean',
                       choices=['mean', 'ucb', 'time'],
                       help='Function to compute the score for the candidates')
    parser.add_argument('--long_term_memory_size', type=int, default=None,
                       help='Size of the long-term memory to store the candidates; if None, no limit is set')
    parser.add_argument('--ucb_exploration_constant', type=float, default=1.0,
                       help='Exploration constant for UCB score function')
    parser.add_argument('--score_range_min', type=float, default=0.0,
                       help='Minimum score for score range (used with UCB)')
    parser.add_argument('--score_range_max', type=float, default=1.0,
                       help='Maximum score for score range (used with UCB)')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='claude-3.5-sonnet',
                       help='Model to use for the agent')
    parser.add_argument('--project_name', type=str, default='veribench-priority-search',
                       help='Name of the project')
    parser.add_argument('--run_name', type=str, default='debug',
                       help='Name of the run')
    parser.add_argument('--verbose', action='store_true', default=False,
                       help='Whether to print verbose output')
    parser.add_argument('--memory_update_frequency', type=int, default=2,
                       help='Duration of the short-term memory')
    parser.add_argument('--use_regressor', action='store_true', default=False,
                       help='Whether to use the regressor')
    parser.add_argument('--regressor_type', type=str, default='logistic', 
                       choices=['logistic', 'linear', 'linear_ucb', 'llm'],
                       help='Type of the regressor')
    parser.add_argument('--regressor_model_name', type=str, default='claude-3.5-sonnet',
                       help='Model name for the regressor')
    parser.add_argument('--regressor_alpha', type=float, default=0.1,
                       help='UCB exploration parameter for the regressor')
    parser.add_argument('--regressor_regularization_strength', type=float, default=0.0001,
                       help='Regularization strength for the regressor')
    parser.add_argument('--regressor_transformation_exploration_factor', type=float, default=0.0,
                       help='Transformation exploration factor for the regressor')
    parser.add_argument('--regressor_projection_dim', type=int, default=None,
                       help='Projection dimension for the regressor')
    parser.add_argument('--regressor_rich_text', action='store_true', default=False,
                       help='Whether to use rich text with problem definition for embeddings')
    parser.add_argument('--optoprime_version', type=str, default='v2', choices=['v1', 'v2'],
                       help='Optimizer to use')
    parser.add_argument('--use_validation', action='store_true', default=False,
                       help='Whether to use validation, only matters in use_regressor version')
    
    # Generator-specific parameters
    parser.add_argument('--use_generator', action='store_true', default=False,
                       help='Whether to use the LLM generator for candidate generation')
    parser.add_argument('--generator_frequency', type=int, default=5,
                       help='Frequency of generating new candidates using LLM generator')
    parser.add_argument('--generator_attempts', type=int, default=50,
                       help='Number of attempts to generate new candidates using LLM generator')
    parser.add_argument('--generator_patience', type=int, default=3,
                       help='Number of attempts to generate new candidates using LLM generator')
    parser.add_argument('--num_generator_candidates', type=int, default=5,
                       help='Number of candidates to generate using LLM generator')
    parser.add_argument('--generator_model_name', type=str, default='claude-3.5-sonnet',
                       help='Model name for the LLM generator')
    parser.add_argument('--generator_temperature', type=float, default=0.6,
                       help='Temperature for the LLM generator')
    parser.add_argument('--generator_verbose', action='store_true', default=False,
                       help='Whether to enable verbose output for the generator')
    parser.add_argument('--ablation', action='store_true', default=False,
                       help='Whether to run ablation study')
    parser.add_argument('--ucb_exploration', action='store_true', default=False,
                       help='UCB exploration')
    parser.add_argument('--epsnetPS', action='store_true', default=False,
                       help='Whether to run epsnetPS')
    parser.add_argument('--epsNet_epsilon', type=float, default=0.1,
                       help='Epsilon value for EpsilonNetPS to filter candidates based on distance')
    parser.add_argument('--use_summarizer', action='store_true', default=False,
                       help='Whether to use the summarizer')
    parser.add_argument('--pareto', action='store_true', default=False,
                       help='Whether to use the pareto frontier')
    
    # Logger parameters
    parser.add_argument('--use_wandb', action='store_true', default=False,
                       help='Whether to use Weights & Biases for logging')
    
    args = parser.parse_args()

    if args.ablation:
        # ucb_exploration or epsnetPS, cannot be used together
        assert not (args.ucb_exploration and args.epsnetPS), "ucb_exploration and epsnetPS cannot be used together"
        if args.ucb_exploration:
            from opto.features.priority_search.priority_search_ablation import PrioritySearchUCBExploration as PrioritySearch
        elif args.epsnetPS:
            from opto.features.priority_search.priority_search_ablation import EpsilonNetPS as PrioritySearch
        else:
            from opto.features.priority_search.priority_search_ablation import PrioritySearch as PrioritySearch
    else:
        from opto.features.priority_search.priority_search import PrioritySearch
    
    # Import regressor variants
    # from opto.features.priority_search.exhausted_priority_search import ExhaustedPrioritySearch_v2 as PrioritySearch_with_Regressor
    # from opto.features.priority_search.priority_search_with_regressor import PrioritySearch_with_Regressor_and_Generator
    from opto.features.priority_search.priority_search_ablation import ParetobasedPS
    # assert num_eval_samples == 1
    # assert args.num_eval_samples == 1, "num_eval_samples must be 1"
    try:
        # Create datasets
        print("Creating datasets...")
        train_dataset = create_dataset(num_tasks=args.num_train_samples, offset=args.train_offset)
        validate_dataset = create_dataset(num_tasks=args.num_validate_samples, offset=args.validate_offset)
        test_dataset = create_dataset(num_tasks=args.num_test_samples, offset=args.test_offset)
        
        print(f"Training samples: {len(train_dataset['inputs'])}")
        print(f"Validation samples: {len(validate_dataset['inputs'])}")
        print(f"Test samples: {len(test_dataset['inputs'])}")
        
        # Initialize agent
        print(f"Initializing agent with model: {args.model}")
        agent = VeribenchAgent(model=args.model)
        
        # Initialize guide, optimizer, and logger
        guide = VeribenchGuide()
        
        if args.optoprime_version == 'v1':
            from opto.optimizers import OptoPrime
            optimizer = OptoPrime(agent.parameters(), max_tokens=8000)
        else:
            from opto.optimizers import OptoPrimeV2
            optimizer = OptoPrimeV2(agent.parameters(), max_tokens=8192, initial_var_char_limit=10000)
        optimizer.objective = OBJECTIVE
        
        # Prepare configuration for logging
        config_dict = {
            'num_train_samples': args.num_train_samples,
            'num_validate_samples': args.num_validate_samples,
            'num_test_samples': args.num_test_samples,
            'train_offset': args.train_offset,
            'validate_offset': args.validate_offset,
            'test_offset': args.test_offset,
            'batch_size': args.batch_size,
            'num_batches': args.num_batches,
            'num_epochs': args.num_epochs,
            'num_steps': args.num_steps,
            'memory_update_frequency': args.memory_update_frequency,
            'num_threads': args.num_threads,
            'test_frequency': args.test_frequency,
            'log_frequency': args.log_frequency,
            'save_frequency': args.save_frequency,
            'save_path': args.save_path,
            'num_eval_samples': args.num_eval_samples,
            'num_candidates': args.num_candidates,
            'num_proposals': args.num_proposals,
            'validate_exploration_candidates': args.validate_exploration_candidates,
            'use_best_candidate_to_explore': args.use_best_candidate_to_explore,
            'memory_size': args.memory_size,
            'score_function': args.score_function,
            'ucb_exploration_constant': args.ucb_exploration_constant,
            'score_range_min': args.score_range_min,
            'score_range_max': args.score_range_max,
            'model': args.model,
            'verbose': args.verbose,
            'use_validation': args.use_validation,
            'regressor_rich_text': args.regressor_rich_text,
            'use_generator': args.use_generator,
            'generator_frequency': args.generator_frequency,
            'generator_attempts': args.generator_attempts,
            'generator_patience': args.generator_patience,
            'num_generator_candidates': args.num_generator_candidates,
            'generator_model_name': args.generator_model_name,
            'generator_temperature': args.generator_temperature,
            'generator_verbose': args.generator_verbose,
            'epsNet_epsilon': args.epsNet_epsilon,
        }
        
        if args.use_wandb:
            logger = WandbLogger(project=args.project_name, verbose=True, name=args.run_name, config=config_dict)
        else:
            logger = DefaultLogger(verbose=True)
        
        # Create PrioritySearch algorithm
        print("Creating PrioritySearch algorithm...")
        
        print("Using basic PrioritySearch")
        algorithm_kwargs = {
            "agent": agent,
            "optimizer": optimizer,
            "logger": logger,
            "num_threads": args.num_threads,
        }
        # Add epsilon parameter for EpsilonNetPS
        if args.epsnetPS:
            algorithm_kwargs["epsilon"] = args.epsNet_epsilon
        
        algorithm = PrioritySearch(**algorithm_kwargs)
        
        if args.use_summarizer:
            assert args.epsnetPS, "use_summarizer can only be used with epsnetPS"
        algorithm.use_summarizer = args.use_summarizer
        
        # Set score range for UCB
        score_range = (args.score_range_min, args.score_range_max) if args.score_function == 'ucb' else None
        
        # Training parameters for PrioritySearch
        train_params = {
            "guide": guide,
            "train_dataset": train_dataset,
            "validate_dataset": validate_dataset,
            "test_dataset": test_dataset,
            "batch_size": args.batch_size,
            "num_batches": args.num_batches,
            "score_range": score_range,
            "num_epochs": args.num_epochs,
            "num_steps": args.num_steps,
            "long_term_memory_size": args.long_term_memory_size,
            "memory_update_frequency": args.memory_update_frequency,
            "num_threads": args.num_threads,
            "verbose": args.verbose,
            "test_frequency": args.test_frequency,
            "num_eval_samples": args.num_eval_samples,
            "num_test_samples": args.num_eval_samples,
            "log_frequency": args.log_frequency,
            "save_frequency": args.save_frequency,
            "save_path": args.save_path,
            # PrioritySearch specific parameters
            "num_candidates": args.num_candidates,
            "num_proposals": args.num_proposals,
            "validate_exploration_candidates": args.validate_exploration_candidates,
            "use_best_candidate_to_explore": args.use_best_candidate_to_explore,
            "memory_size": args.memory_size,
            "score_function": args.score_function,
            "ucb_exploration_constant": args.ucb_exploration_constant,
            "use_validation": args.use_validation,
            "regressor_type": args.regressor_type,
            "regressor_alpha": args.regressor_alpha,
            "regressor_transformation_exploration_factor": args.regressor_transformation_exploration_factor,
            "regressor_projection_dim": args.regressor_projection_dim,
            "regressor_regularization_strength": args.regressor_regularization_strength,
            "regressor_rich_text": args.regressor_rich_text,
            # Generator-specific parameters
            "generator_frequency": args.generator_frequency,
            "generator_attempts": args.generator_attempts,
            "generator_patience": args.generator_patience,
            "num_generator_candidates": args.num_generator_candidates,
        }
        
        # Start training
        print("Starting training with PrioritySearch...")
        print(f"Model: {args.model}")
        print(f"Batch size: {args.batch_size}")
        print(f"Number of batches: {args.num_batches}")
        print(f"Number of epochs: {args.num_epochs}")
        print(f"Number of steps: {args.num_steps}")
        print(f"Number of threads: {args.num_threads}")
        print(f"Number of candidates: {args.num_candidates}")
        print(f"Number of proposals: {args.num_proposals}")
        print(f"Score function: {args.score_function}")
        print(f"UCB exploration constant: {args.ucb_exploration_constant}")
        print(f"Memory size: {args.memory_size}")
        print(f"Validate exploration candidates: {args.validate_exploration_candidates}")
        print(f"Use best candidate to explore: {args.use_best_candidate_to_explore}")
        print(f"Use validation: {args.use_validation}")
        print(f"Regressor type: {args.regressor_type}")
        print(f"Regressor alpha: {args.regressor_alpha}")
        print(f"Regressor regularization strength: {args.regressor_regularization_strength}")
        print(f"Use generator: {args.use_generator}")
        if args.use_generator:
            print(f"Generator frequency: {args.generator_frequency}")
            print(f"Generator attempts: {args.generator_attempts}")
            print(f"Generator patience: {args.generator_patience}")
            print(f"Number of generator candidates: {args.num_generator_candidates}")
            print(f"Generator model: {args.generator_model_name}")
            print(f"Generator temperature: {args.generator_temperature}")
            print(f"Generator verbose: {args.generator_verbose}")
        if args.epsnetPS:
            print(f"EpsilonNetPS epsilon: {args.epsNet_epsilon}")
        
        start_time = time.time()
        algorithm.train(**train_params)
        duration = time.time() - start_time
        
        print(f"Training completed in {duration:.2f} seconds")
           
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
