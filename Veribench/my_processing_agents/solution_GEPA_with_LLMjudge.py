# Optimize a single Lean solution using DSPy GEPA with LLM Judge
# The Lean code is optimized as a text string based on compilation feedback + LLM judge

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
from datetime import datetime

import dspy
from dspy.teleprompt import GEPA

from my_processing_agents.solution_opt import extract_python_code, load_single_task
from guide.guide import VeribenchGuidewithLLMJudge as VeribenchGuide

from gepa.utils.stop_condition import StopperProtocol

import litellm
litellm.drop_params = True
litellm.suppress_debug_info = True

# Add retry logic for rate limiting errors
litellm.num_retries = 3  # Retry up to 3 times
litellm.request_timeout = 300  # 5 minute timeout per request
try:
    from my_processing_agents import secrets_local  # Load environment variables
except ImportError:
    print("secrets_local not found, using local secrets")
    pass
from my_processing_agents.system_prompts import SYSTEM_PROMPT_WITH_EXAMPLES

# Custom instructions for DSPy signature
GENERATOR_INSTRUCTIONS = f"""Your task is to produce valid Lean 4 code that correctly translates a Python program.

The feedback contains:
- Compilation result (30% of score)
- Unit test result (30% of score)
- LLM judge semantic equivalence score (40% of score)

Goal: Maximize the score to 1.0.
- Generate a complete Lean 4 translation from the Python program.

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

# Global guide for metric
_GLOBAL_GUIDE = None


class MaxIterationsStopper(StopperProtocol):
    """Custom stopper that stops after a maximum number of iterations (candidates discovered)."""

    def __init__(self, max_iterations: int):
        self.max_iterations = max_iterations

    def __call__(self, state) -> bool:
        """Return True if we should stop (reached max iterations)."""
        num_candidates = len(state.program_candidates)
        return num_candidates >= self.max_iterations


class PerfectScoreStopper(StopperProtocol):
    """Custom stopper that stops immediately when any candidate achieves perfect score (1.0)."""

    def __init__(self, perfect_score: float = 1.0):
        self.perfect_score = perfect_score

    def __call__(self, state) -> bool:
        """Return True if we should stop (any candidate achieved perfect score on any example)."""
        # Check validation aggregate scores - stop if ANY candidate has perfect aggregate score
        if hasattr(state, 'program_full_scores_val_set') and state.program_full_scores_val_set:
            max_score = max(state.program_full_scores_val_set)
            if max_score >= self.perfect_score:
                print(f"\n🎯 Perfect score {self.perfect_score} achieved on validation set! Stopping optimization.")
                return True

        # Check individual validation subscores - stop if ANY candidate has ANY perfect score
        # (For single-task optimization, this catches if any candidate solved the task)
        if hasattr(state, 'prog_candidate_val_subscores'):
            for candidate_idx, subscores in enumerate(state.prog_candidate_val_subscores):
                if subscores:  # Non-empty dict of {val_id: score}
                    # For single-task: stop if the task has perfect score
                    if any(score >= self.perfect_score for score in subscores.values()):
                        print(f"\n🎯 Perfect score {self.perfect_score} achieved by candidate {candidate_idx}! Stopping optimization.")
                        return True

        # CRITICAL: Check subsample scores in iteration trace
        # When skip_perfect_score=True, perfect subsample scores don't trigger validation,
        # but for single-task optimization, ANY perfect subsample score = success
        if hasattr(state, 'full_program_trace') and state.full_program_trace:
            latest_iter = state.full_program_trace[-1]
            # Get 1-indexed iteration number
            iteration_num = latest_iter.get('i', -1) + 1
            # Check both parent subsample scores AND new candidate subsample scores
            subsample_scores = latest_iter.get('subsample_scores', [])
            new_subsample_scores = latest_iter.get('new_subsample_scores', [])
            all_subsample_scores = list(subsample_scores) + list(new_subsample_scores)
            # Stop if ANY subsample score is perfect (for single-task, this means we solved it)
            if all_subsample_scores and any(score >= self.perfect_score for score in all_subsample_scores):
                print(f"\n🎯 Perfect subsample score {self.perfect_score} achieved in iteration {iteration_num}! Stopping optimization.")
                return True

        return False


class LeanCodeGenerator(dspy.Signature):
    """Generate Lean 4 code from Python program."""
    python_program = dspy.InputField(desc="Python program to translate to Lean 4")
    lean_code = dspy.OutputField(desc="Complete, compilable Lean 4 code")


class LeanTranslator(dspy.Module):
    """DSPy module for translating Python to Lean 4."""

    def __init__(self):
        super().__init__()
        # Create signature with custom instructions
        sig = LeanCodeGenerator.with_instructions(GENERATOR_INSTRUCTIONS)
        self.generator = dspy.ChainOfThought(sig)

    def forward(self, python_program: str):
        """
        Generate Lean 4 code from Python program.

        Args:
            python_program: Python code to translate

        Returns:
            dspy.Prediction with lean_code
        """
        result = self.generator(python_program=python_program)
        return dspy.Prediction(lean_code=result.lean_code)


def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """
    DSPy metric that evaluates Lean code with LLM judge.

    Args:
        gold: Example with python_program input
        pred: Prediction with lean_code output
        trace: Optional trace information
        pred_name: Optional predictor name
        pred_trace: Optional predictor trace

    Returns:
        dspy.Prediction with score and feedback
    """
    if _GLOBAL_GUIDE is None:
        return dspy.Prediction(score=0.0, feedback="System Error: Guide not initialized")

    # Extract inputs and outputs
    python_program = gold.python_program
    lean_code = pred.lean_code
    task_idx = gold.task_id if hasattr(gold, 'task_id') else None

    # Evaluate the Lean code using VeribenchGuide with LLM judge
    try:
        score, feedback = _GLOBAL_GUIDE.get_feedback(
            task=python_program,
            response=lean_code,
            info=task_idx
        )
    except Exception as e:
        score = 0.0
        feedback = f"Error during evaluation: {str(e)}"

    return dspy.Prediction(score=score, feedback=feedback)


def create_single_task_dataset(task_idx: int):
    """
    Create a DSPy dataset with a single task for GEPA optimization.

    Args:
        task_idx: Index of the task to load

    Returns:
        List with one dspy.Example containing Python program
    """
    task = load_single_task(task_idx)
    python_code = extract_python_code(task['user_query'])

    return [
        dspy.Example(
            python_program=python_code,
            task_id=task_idx,
            user_query=task['user_query']
        ).with_inputs("python_program")
    ]


def process_iteration_trace(iteration_trace, gepa_result, initial_seed_score, python_program):
    """
    Process the full iteration trace to extract simplified per-iteration statistics.

    For single-task optimization, tracks the best score achieved and cumulative metric calls
    at each iteration.
    
    Note: GEPA optimizes the prompt/instructions that guide the LLM to generate Lean code,
    not the Lean code itself. The candidates are DSPy programs with different instructions.
    We run each candidate program to generate the actual Lean code.

    Args:
        iteration_trace: List of iteration dictionaries from gepa_state.full_program_trace
        gepa_result: DspyGEPAResult with candidate discovery information
        initial_seed_score: Score of the initial seed program on the task
        python_program: Python program input (to generate Lean code from candidates)

    Returns:
        List of dictionaries, each containing:
        - iteration: Iteration number (starts from 1)
        - best_score_so_far: Best score achieved on the task up to this iteration
        - num_metric_calls_so_far: Total metric calls (evaluations) used so far
        - calls_this_iter: Number of metric calls in this iteration only
        - accepted_new_candidate: Whether a new candidate was accepted this iteration
        - new_candidate_score: Validation score of new candidate if accepted, None otherwise
        - best_lean_program: Best Lean code so far at this iteration (generated by running the best program)
    """
    detailed_history = []

    # Generate Lean code for all candidates by running each optimized program
    # Note: GEPA optimizes the prompts/instructions, not the Lean code directly
    # We need to run each program to get the actual Lean code it generates
    candidate_lean_codes = []
    try:
        # gepa_result.candidates is a list of DSPy Module objects (programs with optimized instructions)
        if hasattr(gepa_result, 'candidates') and gepa_result.candidates:
            print(f"Generating Lean code for {len(gepa_result.candidates)} candidate programs...")
            for idx, candidate_program in enumerate(gepa_result.candidates):
                try:
                    # Run the candidate program to generate Lean code
                    pred = candidate_program(python_program=python_program)
                    lean_code = pred.lean_code if hasattr(pred, 'lean_code') else str(pred)
                    candidate_lean_codes.append(lean_code)
                except Exception as e:
                    candidate_lean_codes.append(f"-- Error generating code from candidate {idx}: {str(e)}")
            print(f"Successfully generated Lean code for {len(candidate_lean_codes)} candidates")
        else:
            print("Warning: gepa_result.candidates not found or empty")
    except Exception as e:
        print(f"Warning: Could not generate Lean codes for candidates: {e}")
        # If we can't access programs, we'll use placeholders

    # Initialize with seed evaluation
    # The seed is evaluated on the full validation set before iteration 0
    num_val_tasks = len(gepa_result.val_subscores[0]) if gepa_result.val_subscores else 1
    cumulative_metric_calls = num_val_tasks  # Initial seed evaluation
    best_score_so_far = initial_seed_score  # Start with seed score
    best_candidate_idx = 0  # Seed is initially the best

    # If no iterations were run (e.g., seed was perfect), create a synthetic iteration 1 for the seed
    if not iteration_trace:
        seed_lean_code = candidate_lean_codes[0] if candidate_lean_codes else "-- Lean code not available (could not run seed program)"
        detailed_history.append({
            "iteration": 1,
            "best_score_so_far": initial_seed_score,
            "num_metric_calls_so_far": num_val_tasks,
            "calls_this_iter": num_val_tasks,
            "accepted_new_candidate": True,  # Seed is always "accepted"
            "new_candidate_score": initial_seed_score,
            "best_lean_program": seed_lean_code,
        })
        return detailed_history

    for iter_dict in iteration_trace:
        iteration_num = iter_dict.get("i", -1) + 1  # Convert to 1-indexed

        # Count metric calls in THIS iteration
        calls_this_iter = 0

        # 1. Parent evaluation on subsample
        subsample_ids = iter_dict.get("subsample_ids", [])
        if subsample_ids:
            calls_this_iter += len(subsample_ids)

        # 2. New candidate evaluation on subsample (if proposal generated)
        new_subsample_scores = iter_dict.get("new_subsample_scores", [])
        if new_subsample_scores:
            calls_this_iter += len(subsample_ids)

        # 3. Merge evaluation (if merge was invoked)
        if iter_dict.get("invoked_merge", False):
            # Merge requires evaluation of merged candidate
            calls_this_iter += len(subsample_ids)

        # 4. Full validation evaluation (if candidate was accepted)
        evaluated_val_indices = iter_dict.get("evaluated_val_indices", [])
        if evaluated_val_indices:
            calls_this_iter += len(evaluated_val_indices)

        # Update cumulative count
        cumulative_metric_calls += calls_this_iter

        # Track best score - check both subsample and validation scores
        # For single-task optimization, we want the maximum score seen on the task

        # Check subsample scores from this iteration
        subsample_scores = iter_dict.get("subsample_scores", [])
        all_scores_this_iter = list(subsample_scores) + list(new_subsample_scores)
        if all_scores_this_iter:
            iter_max_score = max(all_scores_this_iter)
            if iter_max_score > best_score_so_far:
                best_score_so_far = iter_max_score

        # Check if new candidate was accepted (has validation score)
        new_program_idx = iter_dict.get("new_program_idx")
        new_candidate_score = None
        accepted_new_candidate = False

        if new_program_idx is not None:
            accepted_new_candidate = True
            # Get validation score for accepted candidate
            if new_program_idx < len(gepa_result.val_aggregate_scores):
                new_candidate_score = gepa_result.val_aggregate_scores[new_program_idx]
                # Update best candidate if this one is better
                if new_candidate_score > best_score_so_far:
                    best_score_so_far = new_candidate_score
                    best_candidate_idx = new_program_idx
                elif new_candidate_score == best_score_so_far and new_program_idx < len(candidate_lean_codes):
                    # If tied, update to the newer candidate
                    best_candidate_idx = new_program_idx

        # Get best Lean program at this iteration
        # This is the Lean code generated by running the best candidate program
        if candidate_lean_codes and best_candidate_idx < len(candidate_lean_codes):
            best_lean_program = candidate_lean_codes[best_candidate_idx]
        else:
            best_lean_program = "-- Lean code not available (could not run candidate programs)"

        # Build iteration info
        iter_info = {
            "iteration": iteration_num,
            "best_score_so_far": best_score_so_far,
            "num_metric_calls_so_far": cumulative_metric_calls,
            "calls_this_iter": calls_this_iter,
            "accepted_new_candidate": accepted_new_candidate,
            "new_candidate_score": new_candidate_score,
            "best_lean_program": best_lean_program,
        }

        detailed_history.append(iter_info)

    return detailed_history


def main():
    global _GLOBAL_GUIDE

    parser = argparse.ArgumentParser(description='Optimize Lean solution using DSPy GEPA with LLM Judge')
    parser.add_argument('--task_idx', type=int, default=2,
                       help='Task index from Veribench dataset')
    parser.add_argument('--model', type=str, default='claude-3.5-sonnet',
                       help='LLM model name (e.g., claude-3.5-sonnet, gpt-4o)')

    # Budget control - use either max_iterations OR max_metric_calls
    budget_group = parser.add_mutually_exclusive_group()
    budget_group.add_argument('--max_iterations', type=int, default=None,
                       help='Maximum number of optimization iterations (automatically calculates max_metric_calls)')
    budget_group.add_argument('--max_metric_calls', type=int, default=None,
                       help='Budget for GEPA (number of evaluations). Default: 50 if --max_iterations not specified')

    parser.add_argument('--reflection_minibatch_size', type=int, default=1,
                       help='Minibatch size for reflection (default: 1 for single task)')
    parser.add_argument('--num_threads', type=int, default=5,
                       help='Number of threads for parallel evaluation (default: 5)')

    # Logging parameters
    parser.add_argument('--use_wandb', action='store_true', default=False,
                       help='Use Weights & Biases for logging')
    parser.add_argument('--project', type=str, default='veribench-gepa-llmjudge',
                       help='WandB project name')
    parser.add_argument('--run_name', type=str, default=None,
                       help='WandB run name (default: gepa_llmjudge_task_{task_idx})')
    parser.add_argument('--log_dir', type=str, default='results_llm_judge/gepa',
                       help='Directory for GEPA logs and snapshots')
    parser.add_argument('--log_frequency', type=int, default=1,
                       help='Save snapshots every N iterations')
    parser.add_argument('--save_results', action='store_true', default=True,
                       help='Save results to JSON file (default: True)')
    parser.add_argument('--save_name', type=str, default='gepa',
                       help='Name of the result directory (default: gepa)')

    args = parser.parse_args()

    # Calculate stopping condition from max_iterations or max_metric_calls
    gepa_kwargs = {}

    # Always add perfect score stopper to stop immediately when score=1.0 is achieved
    stoppers = [PerfectScoreStopper(perfect_score=1.0)]

    if args.max_iterations is not None:
        # Use custom iteration-based stopper for exact iteration count
        print(f"Using max_iterations={args.max_iterations} (exact iteration-based stopping)")
        iteration_stopper = MaxIterationsStopper(max_iterations=args.max_iterations)
        stoppers.append(iteration_stopper)
        max_metric_calls = None  # Don't set max_metric_calls when using iteration stopper
    elif args.max_metric_calls is not None:
        max_metric_calls = args.max_metric_calls
        print(f"Using max_metric_calls={max_metric_calls}")
    else:
        # Default budget
        max_metric_calls = 50
        print(f"Using default max_metric_calls={max_metric_calls}")

    # Set stop_callbacks with all stoppers (perfect score + optional max_iterations)
    gepa_kwargs['stop_callbacks'] = stoppers
    print(f"Stop conditions: Perfect score (1.0) OR {len(stoppers)-1} additional condition(s)")

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
    lm = dspy.LM(model=model_name, max_tokens=8192, cache=False)
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

    # Initialize Guide with LLM Judge
    print("\nInitializing VeribenchGuide with LLM Judge...")
    _GLOBAL_GUIDE = VeribenchGuide()

    # Create dataset
    print("Creating dataset...")
    dataset = create_single_task_dataset(args.task_idx)

    # Initialize DSPy program
    print("Initializing LeanTranslator DSPy module...")
    program = LeanTranslator()

    # Set run name and create unique log directory for each run (no resume)
    # Use microseconds to ensure uniqueness even with rapid consecutive runs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:21]  # Include microseconds, trim to reasonable length
    run_name = args.run_name if args.run_name else f"gepa_task_{args.task_idx}_{timestamp}"

    # Create unique log directory to prevent resuming from previous runs
    # Each run gets a fresh directory -> GEPA will NOT find gepa_state.bin -> NO RESUME
    log_dir = None
    if args.save_results and args.log_dir:
        log_dir = f"{args.log_dir}_{timestamp}"
        os.makedirs(log_dir, exist_ok=True)  # Create directory to ensure it's fresh
        print(f"  Fresh log directory (no resume): {log_dir}")

    # Run GEPA optimization
    print(f"\nStarting DSPy GEPA optimization:")
    print(f"  Model: {model_name}")
    if args.max_iterations is not None:
        print(f"  Max iterations: {args.max_iterations} (exact)")
    else:
        print(f"  Max metric calls: {max_metric_calls}")
    print(f"  Number of threads: {args.num_threads}")
    print(f"  Reflection minibatch size: {args.reflection_minibatch_size}")
    print(f"  Target: Achieve score 1.0 (successful Lean compilation)")
    print(f"  Early stopping: ENABLED (will stop immediately when score=1.0 is achieved)")
    if args.use_wandb:
        print(f"  WandB logging: project='{args.project}', run='{run_name}'")
    print()

    start_time = time.time()

    gepa = GEPA(
        metric=gepa_metric,
        reflection_lm=lm,
        candidate_selection_strategy='pareto',
        max_metric_calls=max_metric_calls,
        reflection_minibatch_size=args.reflection_minibatch_size,
        track_stats=True,
        num_threads=args.num_threads,
        use_wandb=args.use_wandb,
        wandb_init_kwargs={'project': args.project, 'name': run_name} if args.use_wandb else None,
        log_dir=log_dir,  # Unique timestamped directory, no resume
        log_frequency=args.log_frequency,
        # Early stopping when perfect score is reached
        perfect_score=1.0,
        skip_perfect_score=True,
        # Pass custom stoppers (like max_iterations) via gepa_kwargs
        gepa_kwargs=gepa_kwargs,
    )

    optimized_program = gepa.compile(
        student=program,
        trainset=dataset,
        valset=dataset
    )

    duration = time.time() - start_time

    # Extract results from GEPA (stored in optimized_program.detailed_results when track_stats=True)
    gepa_result = optimized_program.detailed_results

    best_score = gepa_result.val_aggregate_scores[gepa_result.best_idx]
    # Get initial seed score (first candidate, index 0)
    initial_seed_score = gepa_result.val_aggregate_scores[0] if gepa_result.val_aggregate_scores else 0.0

    # Load detailed iteration trace from saved state
    iteration_trace = None
    if log_dir and os.path.exists(os.path.join(log_dir, "gepa_state.bin")):
        try:
            import pickle
            with open(os.path.join(log_dir, "gepa_state.bin"), "rb") as f:
                state_data = pickle.load(f)
                iteration_trace = state_data.get("full_program_trace", [])
                print(f"\nLoaded detailed iteration trace with {len(iteration_trace)} iterations")
        except Exception as e:
            print(f"Warning: Could not load iteration trace: {e}")
            iteration_trace = None

    # Get optimized instruction text - try multiple access patterns
    best_instruction = None

    # Method 1: Try generator.signature.instructions
    try:
        best_instruction = optimized_program.generator.signature.instructions
    except (AttributeError, KeyError):
        pass

    # Method 2: Try extended_signature if regular signature didn't work
    if not best_instruction:
        try:
            best_instruction = optimized_program.generator.extended_signature.instructions
        except (AttributeError, KeyError):
            pass

    # Method 3: Try accessing from predictors list
    if not best_instruction:
        try:
            predictors = list(optimized_program.predictors())
            if predictors:
                pred = predictors[0]
                if hasattr(pred, 'signature') and hasattr(pred.signature, 'instructions'):
                    best_instruction = pred.signature.instructions
                elif hasattr(pred, 'extended_signature') and hasattr(pred.extended_signature, 'instructions'):
                    best_instruction = pred.extended_signature.instructions
        except (AttributeError, KeyError, IndexError):
            pass

    # Fallback to original instruction if extraction failed
    if not best_instruction:
        best_instruction = GENERATOR_INSTRUCTIONS

    # Always save summary
    os.makedirs('results_llm_judge', exist_ok=True)

    # Get success metric call if successful
    success_at = None
    if best_score >= 1.0:
        first_success_idx = next((i for i, s in enumerate(gepa_result.val_aggregate_scores) if s >= 1.0), None)
        if first_success_idx is not None and gepa_result.discovery_eval_counts:
            success_at = gepa_result.discovery_eval_counts[first_success_idx]

    name = args.save_name
    # Minimal summary saved always (even with early stopping)
    os.makedirs(f"results_llm_judge/{name}", exist_ok=True)
    summary_path = f"results_llm_judge/{name}/gepa_task_{args.task_idx}_summary.json"

    # Determine if early stopping occurred
    early_stopped = best_score >= 1.0

    summary_data = {
        'task_idx': args.task_idx,
        'task_id': task['task_id'],
        'success': best_score >= 1.0,
        'final_score': best_score,
        'num_metric_calls': gepa_result.total_metric_calls,
        'success_at_metric_call': success_at,
        'duration_seconds': duration,
        'model': model_name,
        'early_stopped': early_stopped,
    }

    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nSaved summary to {summary_path}")

    # Full result with iteration history
    if args.save_results:
        save_path = f"results_llm_judge/{name}/task_{args.task_idx}_result.json"

        # Build basic iteration history from discoveries (backwards compatible)
        history = []
        for i, (score, eval_count) in enumerate(zip(gepa_result.val_aggregate_scores, gepa_result.discovery_eval_counts)):
            history.append({
                'iteration': i + 1,
                'score': score,
                'metric_calls': eval_count,
                'is_best': i == gepa_result.best_idx
            })

        # Build detailed iteration history from full trace
        # Always process iteration trace (handles empty/None cases by creating synthetic iteration 1)
        detailed_iteration_history = process_iteration_trace(
            iteration_trace if iteration_trace else [],
            gepa_result,
            initial_seed_score,
            python_program
        )

        # Calculate statistics
        num_iterations_total = len(detailed_iteration_history)
        final_best_score = detailed_iteration_history[-1]['best_score_so_far'] if detailed_iteration_history else 0.0

        # Update best_score to be the maximum from iteration history (this is the true best score achieved)
        best_score = final_best_score

        print(f"\nIteration Statistics:")
        print(f"  Total iterations: {num_iterations_total}")
        print(f"  Successful candidates: {len(gepa_result.val_aggregate_scores)}")
        print(f"  Total metric calls used: {gepa_result.total_metric_calls}")
        print(f"  Final best score: {final_best_score}")
        print(f"\nPer-iteration breakdown:")
        for iter_info in detailed_iteration_history:
            print(f"  Iter {iter_info['iteration']}: "
                  f"best_score={iter_info['best_score_so_far']:.3f}, "
                  f"calls_so_far={iter_info['num_metric_calls_so_far']}, "
                  f"accepted={iter_info['accepted_new_candidate']}")

        result_data = {
            'task_idx': args.task_idx,
            'task_id': task['task_id'],
            'success': best_score >= 1.0,
            'early_stopped': early_stopped,
            'final_score': best_score,
            'num_metric_calls': gepa_result.total_metric_calls,
            'num_iterations_total': len(detailed_iteration_history) if detailed_iteration_history else 0,
            'num_candidates_discovered': len(gepa_result.val_aggregate_scores),
            'success_at_metric_call': success_at,
            'duration_seconds': duration,
            'initial_seed_score': initial_seed_score,
            'iteration_history': detailed_iteration_history,  # Per-iteration: iteration, best_score_so_far, num_metric_calls_so_far, calls_this_iter, accepted_new_candidate, new_candidate_score, best_lean_program
            'settings': {
                'model': model_name,
                'max_iterations': args.max_iterations,
                'max_metric_calls_budget': max_metric_calls if max_metric_calls else 'N/A (using max_iterations)',
                'reflection_minibatch_size': args.reflection_minibatch_size,
                'num_threads': args.num_threads,
                'perfect_score_stopping_enabled': True,
            }
        }

        with open(save_path, 'w') as f:
            json.dump(result_data, f, indent=2)

        print(f"\nSaved result with iteration history to {save_path}")

    # Clean one-line summary
    print("\n" + "=" * 70)
    print(f"Task {args.task_idx} | Duration: {duration:.1f}s | Model: {model_name.split('/')[-1]}")
    print("=" * 70)
    if best_score >= 1.0:
        if success_at is not None:
            print(f"✓ SUCCESS: Reached score 1.0 at metric call {success_at}/{gepa_result.total_metric_calls} (early stopped)")
        else:
            print(f"✓ SUCCESS: Final score = {best_score}")
    else:
        print(f"✗ FAILED: Budget exhausted ({gepa_result.total_metric_calls} metric calls)")
        print(f"  Best score achieved: {best_score}")
    print("=" * 70)

    # Print optimized instruction
    # print("\nOptimized Lean Code Generator Instruction:")
    # print("=" * 70)
    # print(best_instruction)
    # print("=" * 70)


if __name__ == "__main__":
    main()
