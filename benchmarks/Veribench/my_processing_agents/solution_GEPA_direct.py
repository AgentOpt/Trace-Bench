# Optimize Lean 4 code directly using GEPA (Generalized Evolutionary Prompting Algorithm)
# GEPA iteratively evolves Lean code DIRECTLY through reflection on compilation feedback
# Unlike solution_GEPA_with_LLMjudge.py, this does NOT use DSPy adapter (no prompt optimization)
# Instead, candidates ARE the Lean code itself, just like the kernel optimization example

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
import re
import shutil

import gepa  # GEPA optimization library
from gepa.core.state import GEPAState

from my_processing_agents.solution_opt import extract_python_code, load_single_task
from guide.guide import VeribenchGuide
from my_processing_agents.system_prompts import SYSTEM_PROMPT_WITH_EXAMPLES

import litellm
litellm.drop_params = True
litellm.suppress_debug_info = True
litellm.num_retries = 3
litellm.request_timeout = 300
try:
    from my_processing_agents import secrets_local  # Load environment variables
except ImportError:
    print("secrets_local not found, using local secrets")
    pass


# Custom stopper to limit GEPA iterations
class MaxIterationsStopper:
    """Stop GEPA after a maximum number of proposal iterations."""

    def __init__(self, max_iterations: int):
        self.max_iterations = max_iterations

    def __call__(self, gepa_state: GEPAState) -> bool:
        """Return True if max iterations reached."""
        return gepa_state.i >= self.max_iterations


class PerfectScoreStopper:
    """Stop GEPA when perfect score (1.0) is achieved."""

    def __init__(self, perfect_score: float = 1.0):
        self.perfect_score = perfect_score

    def __call__(self, gepa_state: GEPAState) -> bool:
        """Return True if any candidate achieved perfect score."""
        # Check if any validation score is perfect
        if hasattr(gepa_state, 'program_full_scores_val_set') and gepa_state.program_full_scores_val_set:
            max_score = max(gepa_state.program_full_scores_val_set)
            if max_score >= self.perfect_score:
                print(f"\n🎯 Perfect score {self.perfect_score} achieved! Stopping optimization.")
                return True
        return False


# Reflection prompt template for Lean code optimization
# GEPA requires exactly these placeholders: <curr_instructions> and <inputs_outputs_feedback>
# Using the same instructions as solution_GEPA_with_LLMjudge.py
REFLECTION_PROMPT_TEMPLATE = f"""Generate complete, compilable Lean 4 code that translates the given Python program.

{SYSTEM_PROMPT_WITH_EXAMPLES}

Requirements:
- Generate COMPLETE Lean 4 code (not partial or commented code)
- Use correct Lean 4 syntax and type annotations
- Preserve the algorithm logic from Python
- Include all necessary imports, functions, and test cases
- Use ':= sorry' for complex proofs if needed
"""


class LeanCodeAdapter(gepa.GEPAAdapter):
    """
    Adapter for optimizing Lean 4 code directly.
    
    Unlike DSPy adapter which optimizes prompts, this adapter treats Lean code
    as the candidate itself (like kernel optimization example).
    
    The candidate is a dict: {"lean_code": "<lean 4 code string>"}
    The objective is to maximize the combined score (compilation + tests + LLM judge).
    """

    def __init__(self, task_idx: int, python_program: str, guide: VeribenchGuide):
        """
        Initialize the Lean code adapter.

        Args:
            task_idx: Task index for evaluation
            python_program: Python program to translate
            guide: VeribenchGuide for evaluation
        """
        self.task_idx = task_idx
        self.python_program = python_program
        self.guide = guide

    def evaluate(self, batch, candidate, capture_traces=False):
        """
        Evaluate the candidate Lean code.

        For Lean optimization, we have one task per run, so batch size is always 1.

        Args:
            batch: List with one example (the task)
            candidate: {"lean_code": "<lean 4 code string>"}
            capture_traces: Whether to capture trajectories for reflection

        Returns:
            EvaluationBatch with scores and feedback
        """
        lean_code = candidate["lean_code"]

        # Strip markdown code fences if LLM generated them
        lean_code = self._strip_markdown_fences(lean_code)

        # Evaluate the Lean code using VeribenchGuide
        try:
            score, feedback = self.guide.get_feedback(
                task=self.python_program,
                response=lean_code,
                info=self.task_idx
            )

            # Debug output
            print(f"\n{'='*70}")
            print(f"LEAN CODE (first 500 chars):")
            print(lean_code[:500] + ("..." if len(lean_code) > 500 else ""))
            print(f"\nScore: {score:.4f}")
            print(f"Feedback (first 300 chars): {feedback[:300]}{('...' if len(feedback) > 300 else '')}")
            print(f"{'='*70}\n")

        except Exception as e:
            score = 0.0
            feedback = f"Error during evaluation: {str(e)}"
            print(f"\nEvaluation error: {e}\n")

        # Prepare outputs
        outputs = [{
            "lean_code": lean_code,
            "score": score,
            "feedback": feedback
        }]

        scores = [score]

        # Capture trajectory if requested (needed for reflection)
        trajectories = None
        if capture_traces:
            trajectories = [{
                "python_program": self.python_program,
                "lean_code": lean_code,
                "score": score,
                "feedback": feedback
            }]

        return gepa.EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories
        )

    def _strip_markdown_fences(self, code: str) -> str:
        """Strip markdown code fences and language identifiers from LLM output."""
        # Remove opening fence with optional language identifier
        code = re.sub(r'^```[a-zA-Z0-9+]*\n?', '', code.strip())
        # Remove standalone language identifier at the beginning
        code = re.sub(r'^(lean|lean4)\s*\n', '', code, flags=re.IGNORECASE)
        # Remove closing fence
        code = re.sub(r'\n?```\s*$', '', code)
        return code.strip()

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        """
        Build reflective dataset for GEPA to propose improved Lean code.

        This extracts the evaluation feedback so GEPA can reflect on compilation errors
        and propose improvements.

        Args:
            candidate: {"lean_code": "<current lean code>"}
            eval_batch: Results from evaluate() with trajectories
            components_to_update: ["lean_code"]

        Returns:
            {"lean_code": [examples with inputs, outputs, and feedback]}
        """
        reflective_data = {}

        if "lean_code" in components_to_update:
            examples = []

            # GEPA expects examples with Inputs, Generated Outputs, and Feedback
            for traj in eval_batch.trajectories:
                example = {
                    "Inputs": {
                        "Python Program": traj["python_program"]
                    },
                    "Generated Outputs": {
                        "Current Lean Code": traj["lean_code"]
                    },
                    "Feedback": traj["feedback"]
                }
                examples.append(example)

            reflective_data["lean_code"] = examples

        return reflective_data


def extract_iteration_history_from_result(result: gepa.GEPAResult, log_dir: str = None, initial_seed_score: float = 0.0) -> list:
    """
    Extract detailed iteration history from GEPA result, including ALL iterations
    (both accepted and rejected proposals).
    
    This creates an iteration history similar to solution_GEPA_with_LLMjudge.py but includes
    best_lean_program field for each iteration. Unlike the previous version, this processes
    the full iteration trace to include all iterations, not just ones with better programs.
    
    Args:
        result: GEPA result object
        log_dir: Optional directory containing gepa_state.bin with full iteration trace
        initial_seed_score: Score of the initial seed candidate
        
    Returns:
        List of iteration dictionaries with fields matching DSPy GEPA format + best_lean_program
    """
    history = []
    
    # Try to load full iteration trace from saved state
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
    
    # Get initial seed Lean code
    seed_lean_code = '-- No code yet'
    if result.candidates and len(result.candidates) > 0:
        seed_candidate = result.candidates[0]
        seed_lean_code = seed_candidate.get('lean_code', '-- Code not available')
    
    # Create a mapping from candidate index to Lean code
    candidate_lean_codes = []
    for candidate in result.candidates:
        candidate_lean_codes.append(candidate.get('lean_code', '-- Code not available'))
    
    # If we have the full trace, process all iterations
    if iteration_trace is not None:
        # If no iterations were run (e.g., seed was perfect), create a synthetic iteration 1 for the seed
        if len(iteration_trace) == 0:
            history.append({
                "iteration": 1,
                "score": initial_seed_score,
                "best_score_so_far": initial_seed_score,
                "num_metric_calls_so_far": 1,  # At least one evaluation for seed
                "calls_this_iter": 1,
                "accepted_new_candidate": True,  # Seed is always "accepted"
                "new_candidate_score": initial_seed_score,
                "best_lean_program": seed_lean_code,
            })
            print(f"Extracted iteration history with 1 entry (seed only, no iterations)")
            return history
        
        # Initialize tracking variables
        # Seed is evaluated before iteration 0, so start with 1 evaluation
        cumulative_metric_calls = 1
        best_score_so_far = initial_seed_score
        best_candidate_idx = 0  # Seed is initially the best
        
        # Process each iteration in the trace
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
                calls_this_iter += len(subsample_ids) if subsample_ids else 0
            
            # 3. Merge evaluation (if merge was invoked)
            if iter_dict.get("invoked_merge", False):
                # Merge requires evaluation of merged candidate
                calls_this_iter += len(subsample_ids) if subsample_ids else 0
            
            # 4. Full validation evaluation (if candidate was accepted)
            evaluated_val_indices = iter_dict.get("evaluated_val_indices", [])
            if evaluated_val_indices:
                calls_this_iter += len(evaluated_val_indices)
            
            # Update cumulative count
            cumulative_metric_calls += calls_this_iter
            
            # Track best score - check both subsample and validation scores
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
                if new_program_idx < len(result.val_aggregate_scores):
                    new_candidate_score = result.val_aggregate_scores[new_program_idx]
                    # Update best candidate if this one is better
                    if new_candidate_score > best_score_so_far:
                        best_score_so_far = new_candidate_score
                        best_candidate_idx = new_program_idx
                    elif new_candidate_score == best_score_so_far and new_program_idx < len(candidate_lean_codes):
                        # If tied, update to the newer candidate
                        best_candidate_idx = new_program_idx
            
            # Get best Lean program at this iteration
            if candidate_lean_codes and best_candidate_idx < len(candidate_lean_codes):
                best_lean_program = candidate_lean_codes[best_candidate_idx]
            else:
                best_lean_program = seed_lean_code if iteration_num == 1 else "-- Lean code not available"
            
            # Determine the score for this iteration
            # Use new_candidate_score if accepted, otherwise use best subsample score from this iteration
            iteration_score = new_candidate_score
            if iteration_score is None and all_scores_this_iter:
                iteration_score = max(all_scores_this_iter)
            
            # Build iteration info
            iter_info = {
                "iteration": iteration_num,
                "score": iteration_score,  # Score of candidate evaluated this iteration (or None)
                "best_score_so_far": best_score_so_far,
                "num_metric_calls_so_far": cumulative_metric_calls,
                "calls_this_iter": calls_this_iter,
                "accepted_new_candidate": accepted_new_candidate,
                "new_candidate_score": new_candidate_score,
                "best_lean_program": best_lean_program,
            }
            
            history.append(iter_info)
        
        print(f"Extracted full iteration history with {len(history)} entries (including rejected proposals)")
    else:
        # Fallback: Build history from candidates discovered only
        # This is the old behavior - only iterations with accepted candidates
        print("Warning: Full iteration trace not available, falling back to candidate-only history")
        best_score_so_far = initial_seed_score
    best_lean_program = '-- No code yet'
    
    for i, (score, eval_count) in enumerate(zip(result.val_aggregate_scores, result.discovery_eval_counts)):
        is_new_best = score > best_score_so_far
        
        # Get the Lean code for this candidate
        candidate = result.candidates[i]
        lean_code = candidate.get('lean_code', '-- Code not available')
        
        # Update best if this is better
        if is_new_best:
            best_score_so_far = score
            best_lean_program = lean_code
        
        history.append({
            'iteration': i + 1,
            'score': score,
            'best_score_so_far': best_score_so_far,
            'num_metric_calls_so_far': eval_count,
            'is_new_best': is_new_best,
            'accepted_new_candidate': True,  # All in result were accepted
            'new_candidate_score': score,
            'best_lean_program': best_lean_program,
        })
    
        print(f"Extracted iteration history with {len(history)} entries (candidates only)")
    
    return history


def main():
    parser = argparse.ArgumentParser(description='Optimize Lean solution using GEPA direct code evolution')
    parser.add_argument('--task_idx', type=int, default=2,
                       help='Task index from Veribench dataset')
    parser.add_argument('--model', type=str, default='claude-3.5-sonnet',
                       help='LLM model name (e.g., claude-3.5-sonnet, gpt-4o)')

    # Budget control - use either max_iterations OR max_metric_calls
    budget_group = parser.add_mutually_exclusive_group()
    budget_group.add_argument('--max_iterations', type=int, default=None,
                       help='Maximum number of optimization iterations')
    budget_group.add_argument('--max_metric_calls', type=int, default=None,
                       help='Budget for GEPA (number of evaluations). Default: 50')

    parser.add_argument('--reflection_minibatch_size', type=int, default=1,
                       help='Minibatch size for reflection (default: 1 for single task)')

    # Logging parameters
    parser.add_argument('--project', type=str, default='veribench-gepa-direct',
                       help='Project name for organization')
    parser.add_argument('--log_dir', type=str, default='results_llm_judge/gepa_direct',
                       help='Base directory for GEPA logs. Final format: {log_dir}_{run}_logs/task_{task_idx} (e.g., results_llm_judge/gepa_1_logs/task_10)')
    parser.add_argument('--run', type=int, default=1,
                       help='Run number (used to create unique log directory per run)')
    parser.add_argument('--log_frequency', type=int, default=1,
                       help='Save snapshots every N iterations')
    parser.add_argument('--save_results', action='store_true', default=True,
                       help='Save results to JSON file (default: True)')
    parser.add_argument('--save_name', type=str, default='gepa_direct',
                       help='Name of the result directory (default: gepa_direct)')
    parser.add_argument('--no-resume', action='store_true', default=False,
                       help='Disable resume (start fresh even if gepa_state.bin exists)')

    args = parser.parse_args()

    # Calculate stopping condition
    stoppers = [PerfectScoreStopper(perfect_score=1.0)]

    if args.max_iterations is not None:
        print(f"Using max_iterations={args.max_iterations}")
        iteration_stopper = MaxIterationsStopper(max_iterations=args.max_iterations)
        stoppers.append(iteration_stopper)
        max_metric_calls = None
    elif args.max_metric_calls is not None:
        max_metric_calls = args.max_metric_calls
        print(f"Using max_metric_calls={max_metric_calls}")
    else:
        max_metric_calls = 50
        print(f"Using default max_metric_calls={max_metric_calls}")

    print(f"Stop conditions: Perfect score (1.0) OR budget exhausted")

    # Prepare model name with provider prefix
    model_name = args.model
    if "/" not in model_name:
        if "claude" in model_name.lower():
            model_name = f"anthropic/{model_name}"
        elif "gpt" in model_name.lower():
            model_name = f"openai/{model_name}"
        elif "gemini" in model_name.lower():
            model_name = f"gemini/{model_name}"

    print(f"\nConfiguring with {model_name}...")

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
    print("\nInitializing VeribenchGuide...")
    guide = VeribenchGuide()

    # Create adapter
    print("Initializing LeanCodeAdapter...")
    adapter = LeanCodeAdapter(
        task_idx=args.task_idx,
        python_program=python_program,
        guide=guide
    )

    # Create dataset for GEPA (single task)
    trainset = [{"task_id": args.task_idx}]
    valset = trainset  # Same task for validation

    # Initial seed candidate - dummy Lean code
    seed_candidate = {
        "lean_code": "-- This is initial placeholder. Generate complete Lean 4 code based on the Python program."
    }

    print(f"\nInitial seed: '{seed_candidate['lean_code']}'")

    # Prepare reflection prompt with task description
    reflection_prompt = REFLECTION_PROMPT_TEMPLATE.replace(
        "<task_description>", 
        python_program
    ).replace(
        "{system_prompt}",
        SYSTEM_PROMPT_WITH_EXAMPLES
    )

    # Create log directory using run number and task ID
    # Format: {base_log_dir}_{run}_logs/task_{task_idx}
    # Example: results_llm_judge/gepa_1_logs/task_10
    log_dir = None
    if args.save_results and args.log_dir:
        base_log_dir = args.log_dir.rstrip('/')
        # Remove trailing _logs if present (to avoid gepa_direct_logs_1_logs)
        if base_log_dir.endswith('_logs'):
            base_log_dir = base_log_dir[:-5]
        log_dir = f"{base_log_dir}_{args.run}_logs/task_{args.task_idx}"
        
        # Always create the directory (GEPA will resume automatically if gepa_state.bin exists)
        os.makedirs(log_dir, exist_ok=True)
        
        # Check if resuming (default: True, unless --no-resume is set)
        should_resume = not args.no_resume
        state_file = os.path.join(log_dir, "gepa_state.bin")
        
        if should_resume and os.path.exists(state_file):
            print(f"✓ Resuming from existing log directory: {log_dir}")
            print(f"  Found gepa_state.bin - GEPA will automatically resume from saved state")
            print(f"  Run: {args.run}, Task: {args.task_idx}")
        elif args.no_resume:
            # User explicitly requested fresh start
            if os.path.exists(state_file):
                print(f"Removing existing state file (--no-resume specified)")
                os.remove(state_file)
            print(f"  Log directory: {log_dir} (fresh start, --no-resume)")
        else:
            print(f"  Log directory: {log_dir} (fresh start)")

    # Run GEPA optimization
    print(f"\nStarting GEPA direct code evolution:")
    print(f"  Model: {model_name}")
    if args.max_iterations is not None:
        print(f"  Max iterations: {args.max_iterations}")
    else:
        print(f"  Max metric calls: {max_metric_calls}")
    print(f"  Reflection minibatch size: {args.reflection_minibatch_size}")
    print(f"  Target: Achieve score 1.0 (successful Lean compilation)")
    print(f"  Early stopping: ENABLED (stops when score=1.0)")
    print(f"  Mode: DIRECT CODE EVOLUTION (candidates are Lean code, not prompts)")
    print()

    start_time = time.time()

    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        adapter=adapter,

        # Reflection configuration
        reflection_lm=model_name,
        reflection_prompt_template=reflection_prompt,
        candidate_selection_strategy='pareto',
        reflection_minibatch_size=args.reflection_minibatch_size,
        skip_perfect_score=True,  # Skip validation for perfect subsample scores

        # Component selection
        module_selector='all',

        # Budget
        max_metric_calls=max_metric_calls,
        stop_callbacks=stoppers,

        # Logging
        run_dir=log_dir,
        log_frequency=args.log_frequency,
        track_best_outputs=True,
        display_progress_bar=True,

        # No wandb
        use_wandb=False,

        # Reproducibility
        seed=10,
    )

    duration = time.time() - start_time

    # Extract results
    best_candidate = result.best_candidate
    best_score = result.val_aggregate_scores[result.best_idx]
    best_lean_code = best_candidate['lean_code']
    
    # Get initial seed score
    initial_seed_score = result.val_aggregate_scores[0] if result.val_aggregate_scores else 0.0

    # Find when success was achieved
    success_at = None
    if best_score >= 1.0:
        first_success_idx = next((i for i, s in enumerate(result.val_aggregate_scores) if s >= 1.0), None)
        if first_success_idx is not None and result.discovery_eval_counts:
            success_at = result.discovery_eval_counts[first_success_idx]

    # Extract iteration history from GEPA result (includes best_lean_program for each iteration)
    # This now processes ALL iterations including rejected proposals
    iteration_history = extract_iteration_history_from_result(result, log_dir, initial_seed_score)

    # Always save summary
    name = args.save_name
    os.makedirs(f"results/{name}", exist_ok=True)
    summary_path = f"results/{name}/gepa_task_{args.task_idx}_summary.json"

    early_stopped = best_score >= 1.0

    summary_data = {
        'task_idx': args.task_idx,
        'task_id': task['task_id'],
        'success': best_score >= 1.0,
        'final_score': best_score,
        'num_metric_calls': result.total_metric_calls,
        'success_at_metric_call': success_at,
        'duration_seconds': duration,
        'model': model_name,
        'early_stopped': early_stopped,
        'method': 'gepa_direct',
    }

    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nSaved summary to {summary_path}")

    # Full result with iteration history
    if args.save_results:
        save_path = f"results/{name}/task_{args.task_idx}_result.json"

        # Print iteration statistics
        print(f"\nIteration Statistics:")
        print(f"  Total iterations: {len(iteration_history)}")
        print(f"  Successful candidates: {len(result.val_aggregate_scores)}")
        print(f"  Total metric calls used: {result.total_metric_calls}")
        print(f"  Final best score: {best_score}")
        print(f"\nPer-iteration breakdown:")
        for iter_info in iteration_history[:10]:  # Show first 10
            score_str = f"{iter_info['score']:.3f}" if iter_info.get('score') is not None else "N/A"
            print(f"  Iter {iter_info['iteration']}: "
                  f"score={score_str}, "
                  f"best_score={iter_info['best_score_so_far']:.3f}, "
                  f"calls_so_far={iter_info['num_metric_calls_so_far']}")
        if len(iteration_history) > 10:
            print(f"  ... ({len(iteration_history) - 10} more iterations)")

        result_data = {
            'task_idx': args.task_idx,
            'task_id': task['task_id'],
            'success': best_score >= 1.0,
            'early_stopped': early_stopped,
            'final_score': best_score,
            'num_metric_calls': result.total_metric_calls,
            'num_iterations_total': len(iteration_history),
            'num_candidates_discovered': len(result.val_aggregate_scores),
            'success_at_metric_call': success_at,
            'best_lean_code': best_lean_code,
            'duration_seconds': duration,
            'initial_seed_score': initial_seed_score,
            'iteration_history': iteration_history,  # Per-iteration: iteration, score, best_score_so_far, num_metric_calls_so_far, is_new_best, accepted_new_candidate, new_candidate_score, best_lean_program
            'val_aggregate_scores': result.val_aggregate_scores,
            'settings': {
                'model': model_name,
                'max_iterations': args.max_iterations,
                'max_metric_calls_budget': max_metric_calls if max_metric_calls else 'N/A (using max_iterations)',
                'reflection_minibatch_size': args.reflection_minibatch_size,
                'perfect_score_stopping_enabled': True,
                'mode': 'direct_code_evolution',
            }
        }

        with open(save_path, 'w') as f:
            json.dump(result_data, f, indent=2)

        print(f"\nSaved result with iteration history to {save_path}")

    # Clean one-line summary
    print("\n" + "=" * 70)
    print(f"Task {args.task_idx} | Duration: {duration:.1f}s | Model: {model_name.split('/')[-1]} | Mode: Direct")
    print("=" * 70)
    if best_score >= 1.0:
        if success_at is not None:
            print(f"✓ SUCCESS: Reached score 1.0 at metric call {success_at}/{result.total_metric_calls}")
        else:
            print(f"✓ SUCCESS: Final score = {best_score}")
    else:
        print(f"✗ FAILED: Budget exhausted ({result.total_metric_calls} metric calls)")
        print(f"  Best score achieved: {best_score}")
    print("=" * 70)


if __name__ == "__main__":
    main()
