# Optimize a single Lean solution using DSPy sequential refinement
# Uses DSPy modules to iteratively improve Lean code based on compilation feedback

import sys
import os
# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
np.random.seed(10)
torch.manual_seed(10)

import time
import argparse
import json

import dspy

from my_processing_agents.solution_opt import extract_python_code, load_single_task
from guide.guide import VeribenchGuide

import litellm
litellm.drop_params = True
litellm.suppress_debug_info = True

# Add retry logic for rate limiting errors
litellm.num_retries = 3
litellm.request_timeout = 300

from my_processing_agents import secrets_local  # Load environment variables
from my_processing_agents.system_prompts import SYSTEM_PROMPT_WITH_EXAMPLES


# Build signature instructions with system prompts
GENERATOR_INSTRUCTIONS = f"""Generate complete, compilable Lean 4 code that translates the given Python program.

{SYSTEM_PROMPT_WITH_EXAMPLES}

Requirements:
- Generate COMPLETE Lean 4 code (not partial or commented code)
- Use correct Lean 4 syntax and type annotations
- Preserve the algorithm logic from Python
- Include all necessary imports, functions, and test cases
- Use ':= sorry' for complex proofs if needed
"""

REFINER_INSTRUCTIONS = f"""Fix compilation errors in Lean 4 code based on feedback.

{SYSTEM_PROMPT_WITH_EXAMPLES}

Requirements:
- Fix ALL compilation errors shown in the feedback
- Preserve correct parts of the current code
- Use correct Lean 4 syntax
- Generate COMPLETE, working Lean 4 code
- Use ':= sorry' for complex proofs if needed
"""


class LeanCodeGenerator(dspy.Signature):
    python_program = dspy.InputField(desc="Python program to translate")
    lean_code = dspy.OutputField(desc="Complete, compilable Lean 4 code")


class LeanCodeRefiner(dspy.Signature):
    python_program = dspy.InputField(desc="Original Python program")
    current_lean_code = dspy.InputField(desc="Current Lean 4 code that failed to compile")
    compilation_feedback = dspy.InputField(desc="Compilation errors and feedback")
    lean_code = dspy.OutputField(desc="Improved Lean 4 code that fixes the errors")


class DSPyLeanAgent(dspy.Module):
    """
    DSPy agent that sequentially refines Lean code based on compilation feedback.
    """
    
    def __init__(self):
        super().__init__()
        # Create signatures with custom instructions
        generator_sig = LeanCodeGenerator.with_instructions(GENERATOR_INSTRUCTIONS)
        refiner_sig = LeanCodeRefiner.with_instructions(REFINER_INSTRUCTIONS)
        
        # Initial generator for first attempt
        self.generator = dspy.ChainOfThought(generator_sig)
        # Refiner for subsequent attempts based on feedback
        self.refiner = dspy.ChainOfThought(refiner_sig)
    
    def forward(self, python_program: str, feedback: str = None, current_code: str = None):
        """
        Generate or refine Lean code.
        
        Args:
            python_program: Python code to translate
            feedback: Compilation feedback (None for first attempt)
            current_code: Current Lean code (None for first attempt)
            
        Returns:
            dspy.Prediction with lean_code
        """
        if feedback is None or current_code is None:
            # First attempt - generate from scratch
            result = self.generator(python_program=python_program)
        else:
            # Refinement - fix based on feedback
            result = self.refiner(
                python_program=python_program,
                current_lean_code=current_code,
                compilation_feedback=feedback
            )
        
        return result


def sequential_optimization(
    agent: DSPyLeanAgent,
    guide: VeribenchGuide,
    python_program: str,
    max_attempts: int = 50,
    verbose: bool = False
) -> dict:
    """
    Sequentially optimize Lean code using DSPy agent and VeribenchGuide feedback.
    
    Args:
        agent: DSPy agent for generating/refining Lean code
        guide: VeribenchGuide for evaluating Lean code
        python_program: Python program to translate
        max_attempts: Maximum number of refinement attempts
        verbose: Whether to print detailed logs
        
    Returns:
        Dictionary with results
    """
    current_lean_code = None
    current_feedback = None
    best_score = 0.0
    best_lean_code = None
    
    history = []
    
    for attempt in range(1, max_attempts + 1):
        if verbose:
            print(f"\n--- Attempt {attempt}/{max_attempts} ---")
        
        # Generate or refine Lean code
        try:
            result = agent(
                python_program=python_program,
                feedback=current_feedback,
                current_code=current_lean_code
            )
            lean_code = result.lean_code
        except Exception as e:
            if verbose:
                print(f"Error generating code: {e}")
            lean_code = "-- Error generating Lean code"
        
        # Evaluate with VeribenchGuide
        try:
            score, feedback = guide.get_feedback(
                task=python_program,
                response=lean_code
            )
        except Exception as e:
            score = 0.0
            feedback = f"Error during evaluation: {str(e)}"
        
        # Track history
        history.append({
            'attempt': attempt,
            'score': score,
            'feedback': feedback[:500],  # Truncate for storage
        })
        
        if verbose:
            print(f"Score: {score}")
            print(f"Feedback: {feedback[:200]}...")
        
        # Update best
        if score > best_score:
            best_score = score
            best_lean_code = lean_code
        
        # Check for success
        if score >= 1.0:
            if verbose:
                print(f"\n✓ SUCCESS at attempt {attempt}!")
            return {
                'success': True,
                'attempts': attempt,
                'best_score': best_score,
                'best_lean_code': best_lean_code,
                'final_feedback': feedback,
                'history': history
            }
        
        # Prepare for next iteration
        current_lean_code = lean_code
        current_feedback = feedback
    
    # Failed to reach score 1.0
    if verbose:
        print(f"\n✗ FAILED after {max_attempts} attempts")
    
    return {
        'success': False,
        'attempts': max_attempts,
        'best_score': best_score,
        'best_lean_code': best_lean_code,
        'final_feedback': current_feedback,
        'history': history
    }


def main():
    parser = argparse.ArgumentParser(description='Optimize Lean solution using DSPy sequential refinement')
    parser.add_argument('--task_idx', type=int, default=14, 
                       help='Task index from Veribench dataset')
    parser.add_argument('--model', type=str, default='claude-3.5-sonnet', 
                       help='LLM model name (e.g., claude-3.5-sonnet, gpt-4o)')
    parser.add_argument('--max_attempts', type=int, default=50, 
                       help='Maximum number of refinement attempts')
    parser.add_argument('--verbose', action='store_true', default=False,
                       help='Print detailed logs during optimization')
    # Logging parameters
    parser.add_argument('--save_results', action='store_true', default=False,
                       help='Save results to JSON file (default: False)')
    
    args = parser.parse_args()
    
    # Prepare model name with provider prefix
    model_name = args.model
    if "/" not in model_name:
        if "claude" in model_name.lower():
            model_name = f"anthropic/{model_name}"
        elif "gpt" in model_name.lower():
            model_name = f"openai/{model_name}"
        elif "gemini" in model_name.lower():
            model_name = f"gemini/{model_name}"
    
    # Configure DSPy
    print(f"Configuring DSPy with {model_name}...")
    lm = dspy.LM(model=model_name, max_tokens=8192,cache=False)
    dspy.configure(lm=lm)
    
    
    # Load task
    print(f"\nLoading task {args.task_idx} from Veribench dataset...")
    task = load_single_task(args.task_idx)
    python_program = extract_python_code(task['user_query'])
    print(f"Task loaded successfully. Task ID: {task['task_id']}")
    print(f"\nPython code to translate (first 500 chars):")
    print("=" * 70)
    print(python_program[:500] + ("..." if len(python_program) > 500 else ""))
    print("=" * 70)
    
    # Initialize components
    print("\nInitializing DSPy agent and VeribenchGuide...")
    agent = DSPyLeanAgent()
    guide = VeribenchGuide()
    
    # Run sequential optimization
    print(f"\nStarting DSPy sequential optimization:")
    print(f"  Model: {model_name}")
    print(f"  Max attempts: {args.max_attempts}")
    print(f"  Target: Achieve score 1.0 (successful Lean compilation)")
    print()
    
    start_time = time.time()
    
    result = sequential_optimization(
        agent=agent,
        guide=guide,
        python_program=python_program,
        max_attempts=args.max_attempts,
        verbose=args.verbose
    )
    
    duration = time.time() - start_time
    
    # Save result if requested
    if args.save_results:
        save_path = f"results/dspy_2/dspy_task_{args.task_idx}_result.json"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        result_data = {
            'task_idx': args.task_idx,
            'task_id': task['task_id'],
            'success': result['success'],
            'attempts': result['attempts'],
            'best_score': result['best_score'],
            'best_lean_code': result['best_lean_code'],
            'duration_seconds': duration,
            'history': result['history'],
            'settings': {
                'model': model_name,
                'max_attempts': args.max_attempts,
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        print(f"\nSaved result to {save_path}")
    
    # Clean one-line summary
    print("\n" + "=" * 70)
    print(f"Task {args.task_idx} | Duration: {duration:.1f}s | Model: {model_name.split('/')[-1]}")
    print("=" * 70)
    if result['success']:
        print(f"✓ SUCCESS: Reached score 1.0 at attempt {result['attempts']}/{args.max_attempts}")
    else:
        print(f"✗ FAILED: Max attempts ({result['attempts']}) reached without achieving score 1.0")
        print(f"  Best score achieved: {result['best_score']}")
    print("=" * 70)


if __name__ == "__main__":
    main()

