"""
Simple test to verify async_run works with lean evaluation.
"""
from opto.trainer.utils import async_run
from opto.optimizers.utils import print_color
from my_processing_agents.lean_interpretor import lean_interpreter

# Test with a simple valid Lean code
VALID_LEAN_CODE = 'def main : IO Unit := IO.println "Hello, world!"'

# Test with actually invalid Lean code (type mismatch - String cannot be Nat)
INVALID_LEAN_CODE = 'def foo : Nat := "hello"'  # type error: String is not Nat

# Test with code that triggers coupling error (metavariables depending on each other)
COUPLING_ERROR_CODE = '''
example (p q: Prop): p → q → p ∧ q := by
  intro hp hq
  have h1 : p := sorry
  have h2 : p ∧ q := ⟨h1, sorry⟩
  exact h2
'''

def run_single_eval(lean_code, task_id):
    """Run a single evaluation using lean_interpreter."""
    print_color(f"Starting eval for task {task_id}...", "cyan")
    result = lean_interpreter(lean_code)
    score = 1.0 if result["valid"] else 0.0
    feedback = result["summary"]
    print_color(f"Finished eval for task {task_id}, score={score}", "green" if score == 1.0 else "red")
    return {
        'task_id': task_id,
        'score': score,
        'feedback': feedback,
        'valid': result["valid"],
        'error_messages': result.get("error_messages", []),
        'error_details': result.get("error_details", []),
    }

# Test 1: Single evaluation (no async, just to make sure lean_interpreter works)
print_color("="*50, "magenta")
print_color("Test 1: Single evaluation (synchronous)", "magenta")
print_color("="*50, "magenta")

result = run_single_eval(INVALID_LEAN_CODE, 0)
print_color(f"Result valid: {result['valid']}, score: {result['score']}", "cyan")
print_color(f"Summary: {result['feedback']}", "cyan")
if result.get('error_messages'):
    print_color("Error Messages:", "yellow")
    for i, err in enumerate(result['error_messages']):
        print_color(f"  [{i+1}] {err}", "yellow")

# Test 2: Single evaluation with async_run (max_workers=1, runs sequentially)
print_color("\n" + "="*50, "magenta")
print_color("Test 2: async_run with max_workers=1 (sequential)", "magenta")
print_color("="*50, "magenta")

eval_runs = [run_single_eval]
eval_args = [(VALID_LEAN_CODE, 1)]
eval_kwargs = [{}]

results = async_run(eval_runs, eval_args, eval_kwargs, max_workers=1, description="Lean eval (seq)")
print_color(f"Results: {results}", "cyan")

# Test 3: 10 evaluations with async_run (max_workers=10, parallel)
print_color("\n" + "="*50, "magenta")
print_color("Test 3: async_run with max_workers=10 (10 parallel evals)", "magenta")
print_color("="*50, "magenta")

# Create 10 test cases (mix of valid, invalid, and coupling error)
# Tasks 0,3,6,9: valid code
# Tasks 1,4,7: invalid code (type error)
# Tasks 2,5,8: coupling error code
def get_test_code(i):
    if i % 3 == 0:
        return VALID_LEAN_CODE
    elif i % 3 == 1:
        return INVALID_LEAN_CODE
    else:
        return COUPLING_ERROR_CODE

test_codes = [get_test_code(i) for i in range(10)]

eval_runs = [run_single_eval] * 10
eval_args = [(test_codes[i], i) for i in range(10)]
eval_kwargs = [{}] * 10

results = async_run(eval_runs, eval_args, eval_kwargs, max_workers=10, description="Lean eval (10 parallel)")
print_color(f"Completed {len(results)} evaluations", "cyan")

for r in results:
    status = "✓ PASS" if r['valid'] else "✗ FAIL"
    if r['valid']:
        print_color(f"  Task {r['task_id']}: {status}", "green")
    else:
        print_color(f"  Task {r['task_id']}: {status}", "red")
        print_color(f"    Summary: {r['feedback']}", "yellow")
        # Print full error messages
        if r.get('error_messages'):
            print_color(f"    Error Messages:", "yellow")
            for i, err in enumerate(r['error_messages']):
                print_color(f"      [{i+1}] {err}", "yellow")
        # Print error details with context
        if r.get('error_details'):
            print_color(f"    Error Details:", "yellow")
            for detail in r['error_details']:
                for line in detail.split('\n'):
                    print_color(f"      {line}", "yellow")

print_color("\n" + "="*50, "magenta")
print_color("All tests completed!", "green")
print_color("="*50, "magenta")

