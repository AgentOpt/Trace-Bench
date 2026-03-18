from optimize_veribench_agent import VeribenchGuide
from opto.trainer.utils import async_run
from opto.optimizers.utils import print_color
import json

# Load the agent_responses.json file
with open('agent_responses.json', 'r') as f:
    data = json.load(f)

# Get the lean codes from results
lean_codes = [result['lean_code'] for result in data['results']]

# Initialize guide
guide = VeribenchGuide()

def evaluate_single(lean_code, task_info):
    """Evaluate a single lean code response."""
    score, feedback = guide.get_feedback(task=None, response=lean_code, info=task_info)
    return {
        'task_info': task_info,
        'score': score,
        'feedback': feedback
    }

# Run 10 evaluations asynchronously
runs = [evaluate_single] * 1
args_list = [(lean_codes[i], i) for i in range(1)]
kwargs_list = [{}] * 1

print_color("Evaluating 1 responses asynchronously...", "cyan")
results = async_run(runs, args_list, kwargs_list, max_workers=1, description="Evaluating responses")

# Print results
total_score = 0
for result in results:
    total_score += result['score']
    print_color(f"\nTask {result['task_info']}: Score = {result['score']}", 
                "green" if result['score'] == 1.0 else "red")
    if result['score'] < 1.0:
        print_color(f"Feedback: {result['feedback']}...", "yellow")

print_color(f"\n{'='*50}", "cyan")
print_color(f"Total Score: {total_score}/1 ({total_score*10:.0f}%)", "magenta")

# Save evaluation results
with open('evaluation_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print_color(f"Saved evaluation results to evaluation_results.json", "green")
