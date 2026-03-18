# Optimize a single Lean solution using OpenEvolve
# Uses evolutionary algorithm to discover compilable Lean code

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
import tempfile
import shutil
from pathlib import Path

from openevolve.api import run_evolution, EvolutionResult
from openevolve.config import Config, LLMConfig, DatabaseConfig, EvaluatorConfig, PromptConfig

from my_processing_agents.solution_opt import extract_python_code, load_single_task
from my_processing_agents.system_prompts import SYSTEM_PROMPT_WITH_EXAMPLES

import litellm
litellm.drop_params = True
litellm.suppress_debug_info = True
try:
    from my_processing_agents import secrets_local  # Load environment variables
except ImportError:
    print("secrets_local not found, using local secrets")
    pass

def write_initial_lean_program(temp_dir: str, python_program: str, task_idx: int) -> str:
    """
    Write initial Lean program with evolution markers.
    
    Args:
        temp_dir: Temporary directory
        python_program: Python code being translated (for context)
        task_idx: Task index
        
    Returns:
        Path to the initial program file
    """
    # Include Python code as comment for context
    initial_content = f"""# Python program being translated (for context):
# Task: Translate the following Python code to Lean 4
#
# {python_program[:500].replace(chr(10), chr(10) + '# ')}...

# EVOLVE-BLOCK-START
-- Lean 4 translation of the Python program
# EVOLVE-BLOCK-END
"""
    
    program_path = os.path.join(temp_dir, f"initial_task_{task_idx}.lean")
    with open(program_path, 'w') as f:
        f.write(initial_content)
    
    return program_path


def write_evaluator_wrapper(temp_dir: str, python_program: str, task_idx: int) -> str:
    """
    Write evaluator that wraps VeribenchGuide.
    
    Args:
        temp_dir: Temporary directory
        python_program: Python code to translate
        task_idx: Task index
        
    Returns:
        Path to evaluator file
    """
    # Use repr() to properly escape the Python code for embedding
    python_repr = repr(python_program)
    
    evaluator_code = f'''"""
Evaluator for Lean code translation using VeribenchGuide
"""
import sys
sys.path.insert(0, '{project_root}')

from guide.guide import VeribenchGuide
from openevolve.evaluation_result import EvaluationResult

# Python program to translate (passed from solution_openevolve.py)
PYTHON_PROGRAM = {python_repr}

# Initialize guide once (reuse across evaluations)
_guide_instance = None

def get_guide():
    global _guide_instance
    if _guide_instance is None:
        _guide_instance = VeribenchGuide()
    return _guide_instance


def evaluate(program_path):
    """
    Evaluate evolved Lean code using VeribenchGuide.
    
    Args:
        program_path: Path to the evolved Lean program
        
    Returns:
        EvaluationResult with score and compilation feedback
    """
    guide = get_guide()
    
    try:
        # Read the evolved Lean code
        with open(program_path, 'r') as f:
            lean_code_full = f.read()
        
        # Extract only the evolved block (between markers)
        start_marker = "# EVOLVE-BLOCK-START"
        end_marker = "# EVOLVE-BLOCK-END"
        
        if start_marker in lean_code_full and end_marker in lean_code_full:
            lean_code = lean_code_full.split(start_marker)[1].split(end_marker)[0].strip()
        else:
            # Fallback: use entire file if markers not found
            lean_code = lean_code_full.strip()
        
        # Handle empty code
        if not lean_code or lean_code == "":
            return EvaluationResult(
                metrics={{"combined_score": 0.0}},
                artifacts={{
                    "stderr": "Generated empty Lean code. Please generate complete Lean 4 code, not empty output."
                }}
            )
        
        # Evaluate with VeribenchGuide
        score, feedback = guide.get_feedback(
            task=PYTHON_PROGRAM,
            response=lean_code
        )
        
        # Pass through guide's feedback completely unmodified
        return EvaluationResult(
            metrics={{
                "combined_score": float(score),
            }},
            artifacts={{
                "stderr": feedback,  # Use stderr for OpenEvolve compatibility
            }}
        )
        
    except Exception as e:
        # Handle evaluation errors
        import traceback
        return EvaluationResult(
            metrics={{"combined_score": 0.0}},
            artifacts={{
                "stderr": f"Evaluation error: {{str(e)}}\\n\\nTraceback:\\n{{traceback.format_exc()}}"
            }}
        )
'''
    
    evaluator_path = os.path.join(temp_dir, f"evaluator_task_{task_idx}.py")
    with open(evaluator_path, 'w') as f:
        f.write(evaluator_code)
    
    return evaluator_path


def create_openevolve_config(model_name: str, max_iterations: int, num_workers: int = 4) -> Config:
    """
    Create OpenEvolve configuration.
    
    Args:
        model_name: LLM model name
        max_iterations: Maximum evolution iterations
        
    Returns:
        Config object for OpenEvolve
    """
    config = Config()
    
    # Set max iterations
    config.max_iterations = max_iterations
    
    # LLM configuration - determine API base and key based on model
    api_base = None
    api_key = None
    
    if "/" not in model_name:
        if "claude" in model_name.lower():
            api_base = os.environ.get('ANTHROPIC_API_BASE', 'https://api.anthropic.com/v1')
            # Ensure API base ends with /v1 for OpenAI-compatible endpoints
            if api_base and not api_base.endswith('/v1'):
                api_base = api_base.rstrip('/') + '/v1'
            api_key = os.environ.get('ANTHROPIC_API_KEY')
        elif "gpt" in model_name.lower():
            api_base = os.environ.get('OPENAI_API_BASE', 'https://api.openai.com/v1')
            api_key = os.environ.get('OPENAI_API_KEY')
        elif "gemini" in model_name.lower():
            api_base = 'https://generativelanguage.googleapis.com/v1beta/openai/'
            api_key = os.environ.get('OPENAI_API_KEY')  # Gemini uses OPENAI_API_KEY
        else:
            api_base = os.environ.get('OPENAI_API_BASE', 'https://api.openai.com/v1')
            api_key = os.environ.get('OPENAI_API_KEY')
    else:
        # Has provider prefix, extract model name
        provider, model_name = model_name.split('/', 1)
        if provider == 'anthropic':
            api_base = os.environ.get('ANTHROPIC_API_BASE', 'https://api.anthropic.com/v1')
            api_key = os.environ.get('ANTHROPIC_API_KEY')
        elif provider == 'openai':
            api_base = os.environ.get('OPENAI_API_BASE', 'https://api.openai.com/v1')
            api_key = os.environ.get('OPENAI_API_KEY')
        elif provider == 'gemini':
            api_base = 'https://generativelanguage.googleapis.com/v1beta/openai/'
            api_key = os.environ.get('OPENAI_API_KEY')
    
    # Create LLMConfig with primary_model set - this will auto-populate models list
    config.llm = LLMConfig(
        primary_model=model_name,
        api_base=api_base,
        api_key=api_key,
        temperature=0.7,
        max_tokens=8192,
        timeout=300,
        retries=10
        # Note: random_seed is not set here to prevent propagation to models
    )
    
    # System message for Lean translation
    config.prompt = PromptConfig()
    config.prompt.system_message = f"""You are an expert in translating Python programs to Lean 4 code.

Your task: Generate COMPLETE, COMPILABLE Lean 4 code that correctly translates the Python program.

{SYSTEM_PROMPT_WITH_EXAMPLES}

Requirements:
- Generate COMPLETE Lean 4 code (not partial or commented placeholders)
- Fix ALL compilation errors shown in the artifacts/feedback
- Preserve the algorithm logic from the Python program
- Use correct Lean 4 syntax and type annotations
- Include necessary imports, functions, and test cases
- Use := sorry for complex proofs if needed

The artifacts will show compilation feedback from previous attempts - use it to fix errors!
Output ONLY valid Lean 4 code, not meta-instructions."""
    
    # Database configuration - optimized for single objective
    # Note: Even with single objective, we need feature_dimensions for diversity
    config.database = DatabaseConfig()
    config.database.population_size = 50  # Small population for single task
    config.database.num_islands = 1  # No islands needed
    config.database.elite_selection_ratio = 0.2
    config.database.exploitation_ratio = 0.8  # High exploitation for single objective
    # Use built-in features for diversity even with single objective
    # "complexity" = code length, "diversity" = structural difference
    config.database.feature_dimensions = ["complexity", "diversity"]
    
    # Evaluator configuration
    config.evaluator = EvaluatorConfig()
    config.evaluator.enable_artifacts = True  # CRITICAL for feedback loop!
    config.evaluator.timeout = 60
    config.evaluator.parallel_evaluations = num_workers  # Number of parallel workers
    config.evaluator.cascade_evaluation = False  # Disable cascade (not using multi-stage)
    
    # Evolution settings
    config.diff_based_evolution = False  # Full rewrites (better for Lean syntax)
    config.file_suffix = ".lean"  # Use Lean extension
    config.language = "lean"
    config.random_seed = None  # Don't set master seed (prevents LLM seed propagation)
    config.database.random_seed = 10  # Reproducibility for sampling/database only
    config.checkpoint_interval = 10  # Save checkpoints
    
    # Compromise early stopping: Balance between speed and exploration
    # - If success is reached early, stops relatively soon (saves time)
    # - If stuck at 0.0, gives reasonable attempts before stopping
    # - If success never reached, doesn't waste all 50 iterations
    config.early_stopping_patience = None  # Stop after 20 iterations without improvement
    config.convergence_threshold = 0.001  # Any improvement resets counter
    config.early_stopping_metric = "combined_score"
    
    return config


def extract_metric_calls(output_dir: str) -> int:
    """
    Extract number of metric calls from OpenEvolve output.
    
    Args:
        output_dir: OpenEvolve output directory
        
    Returns:
        Number of metric calls (programs evaluated)
    """
    # Try to load database and count programs
    try:
        db_path = os.path.join(output_dir, "database.json")
        if os.path.exists(db_path):
            with open(db_path, 'r') as f:
                db_data = json.load(f)
                return len(db_data.get('programs', []))
    except:
        pass
    
    # Fallback: count checkpoint directories (approximation)
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    if os.path.exists(checkpoints_dir):
        checkpoints = [d for d in os.listdir(checkpoints_dir) if d.startswith('checkpoint_')]
        if checkpoints:
            # Get last checkpoint number
            last_checkpoint = max([int(d.split('_')[1]) for d in checkpoints if d.split('_')[1].isdigit()])
            return last_checkpoint
    
    return 0


def main():
    parser = argparse.ArgumentParser(description='Optimize Lean solution using OpenEvolve')
    parser.add_argument('--task_idx', type=int, default=70, 
                       help='Task index from Veribench dataset')
    parser.add_argument('--model', type=str, default='claude-3.5-sonnet', 
                       help='LLM model name (e.g., claude-3.5-sonnet, gpt-4o)')
    parser.add_argument('--max_iterations', type=int, default=50, 
                       help='Maximum number of evolution iterations')
    parser.add_argument('--num_workers', type=int, default=5,
                       help='Number of parallel workers for evaluation (1=sequential, 4+=parallel)')
    parser.add_argument('--verbose', action='store_true', default=False,
                       help='Print detailed logs during evolution')
    
    # Logging parameters
    parser.add_argument('--save_results', action='store_true', default=False,
                       help='Save full results (default: only summary)')
    
    args = parser.parse_args()
    
    # Load task
    print(f"\nLoading task {args.task_idx} from Veribench dataset...")
    task = load_single_task(args.task_idx)
    python_program = extract_python_code(task['user_query'])
    print(f"Task loaded successfully. Task ID: {task['task_id']}")
    print(f"\nPython code to translate (first 500 chars):")
    print("=" * 70)
    print(python_program[:500] + ("..." if len(python_program) > 500 else ""))
    print("=" * 70)
    
    # Create temporary directory for OpenEvolve files
    temp_dir = tempfile.mkdtemp(prefix=f"openevolve_task_{args.task_idx}_")
    print(f"\nCreated temporary directory: {temp_dir}")
    
    try:
        # Write initial Lean program
        print("Creating initial Lean program...")
        initial_program_path = write_initial_lean_program(temp_dir, python_program, args.task_idx)
        
        # Write evaluator wrapper
        print("Creating evaluator wrapper...")
        evaluator_path = write_evaluator_wrapper(temp_dir, python_program, args.task_idx)
        
        # Create config
        print("Creating OpenEvolve configuration...")
        config = create_openevolve_config(args.model, args.max_iterations, args.num_workers)
        
        # Set output directory
        output_dir = os.path.join(temp_dir, "openevolve_output")
        
        # Run OpenEvolve
        print(f"\nStarting OpenEvolve evolution:")
        print(f"  Model: {args.model}")
        print(f"  Max iterations: {args.max_iterations}")
        print(f"  Parallel workers: {args.num_workers} (evaluations run in parallel)")
        print(f"  Single objective: combined_score (0.0 = failed, 1.0 = success)")
        print(f"  Early stopping: After 20 iterations without improvement")
        print(f"  Evolution strategy: Full rewrites (not diff-based)")
        print(f"  Artifact feedback: ENABLED (VeribenchGuide feedback passed to LLM)")
        print()
        
        start_time = time.time()
        
        result: EvolutionResult = run_evolution(
            initial_program=initial_program_path,
            evaluator=evaluator_path,
            config=config,
            iterations=args.max_iterations,
            output_dir=output_dir,
            cleanup=False  # Don't cleanup yet, need to extract data
        )
        
        duration = time.time() - start_time
        
        # Extract results
        best_score = result.best_score if result.best_program else 0.0
        best_lean_code = result.best_code if result.best_program else "-- No successful code generated"
        
        # Count metric calls (number of programs evaluated)
        # OpenEvolve evaluates multiple programs, but we approximate by iteration count
        # Each iteration typically evaluates 1-2 programs (parent + child)
        num_metric_calls = extract_metric_calls(output_dir)
        if num_metric_calls == 0:
            num_metric_calls = args.max_iterations  # Fallback approximation
        
        # Find when success was achieved
        success_at = None
        if result.best_program and best_score >= 1.0:
            success_at = result.best_program.iteration_found
            if success_at is None:
                success_at = num_metric_calls  # Fallback
        
        # Always save summary to OpenEvolve folder
        os.makedirs('results/OpenEvolve_2', exist_ok=True)
        summary_path = f"results/OpenEvolve_2/openevolve_task_{args.task_idx}_summary.json"
        
        summary_data = {
            'task_idx': args.task_idx,
            'task_id': task['task_id'],
            'success': best_score >= 1.0,
            'final_score': best_score,
            'num_metric_calls': num_metric_calls,
            'success_at_metric_call': success_at,
            'duration_seconds': duration,
            'model': args.model,
            'method': 'openevolve',
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"\nSaved summary to {summary_path}")
        
        # Save full result if requested
        if args.save_results:
            full_result_path = f"results/OpenEvolve_2/openevolve_task_{args.task_idx}_result.json"
            
            result_data = {
                'task_idx': args.task_idx,
                'task_id': task['task_id'],
                'success': best_score >= 1.0,
                'final_score': best_score,
                'num_metric_calls': num_metric_calls,
                'success_at_metric_call': success_at,
                'best_lean_code': best_lean_code,
                'duration_seconds': duration,
                'metrics': result.metrics,
                'settings': {
                    'model': args.model,
                    'max_iterations': args.max_iterations,
                    'evolution_strategy': 'full_rewrite',
                    'population_size': config.database.population_size,
                }
            }
            
            with open(full_result_path, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            print(f"Saved full result to {full_result_path}")
        
        # Clean one-line summary
        print("\n" + "=" * 70)
        print(f"Task {args.task_idx} | Duration: {duration:.1f}s | Model: {args.model} | Method: OpenEvolve")
        print("=" * 70)
        if best_score >= 1.0:
            if success_at:
                print(f"✓ SUCCESS: Reached score 1.0 at iteration {success_at}/{num_metric_calls}")
            else:
                print(f"✓ SUCCESS: Final score = {best_score}")
        else:
            print(f"✗ FAILED: Max iterations ({num_metric_calls}) reached without achieving score 1.0")
            print(f"  Best score achieved: {best_score}")
        print("=" * 70)
        
    finally:
        # Cleanup temporary directory
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"\nCleaned up temporary directory: {temp_dir}")
            except Exception as e:
                print(f"\nWarning: Could not clean up temp directory: {e}")


if __name__ == "__main__":
    main()

