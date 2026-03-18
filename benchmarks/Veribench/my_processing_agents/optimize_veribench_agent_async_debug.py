
from opto import trace
from opto.trainer.guide import Guide
from opto.utils.llm import LLM
from opto.optimizers.utils import print_color
from my_processing_agents.lean_interpretor import lean_interpreter
from my_processing_agents.system_prompts import SYSTEM_PROMPT_WITH_EXAMPLES, SYSTEM_PROMPT, EXAMPLES
import re,os


from opto.optimizers.utils import print_color
os.environ["TRACE_LITELLM_MODEL"] = f"gemini/gemini-2.5-flash-lite"
from opto.trainer.utils import async_run
# import nest_asyncio
# nest_asyncio.apply()
from my_processing_agents.lean_interpretor import remove_import_error
# lean_interpreter("def main : IO Unit := IO.println \"Hello, world!\"")
import litellm 
litellm.drop_params = True
litellm.suppress_debug_info = True
@trace.model
class VeribenchAgent:
    """
    A simple traced agent for solving Veribench tasks.
    Uses trainable system_prompt to optimize LLM prompting.
    The system prompt is a trainable node that is optimized to produce correct Lean code.
    The task is user_query in the dataset.
    """

    def __init__(self, model: str = "gemini/gemini-2.5-flash-lite"):
        self.model = model
        self.llm = LLM(model=model)
        self.system_prompt = SYSTEM_PROMPT
        self.examples = EXAMPLES
        self.additional_instructions = trace.node("Here are the additional instructions to help the agent solve the task: ", trainable=True)

    @trace.bundle()
    def solve(self, additional_instructions: str, task_input: dict) -> str:
        """
        Solve a Veribench task with the given system_prompt.
        
        Args:
            system_prompt: Trainable system prompt
            task: Veribench task dictionary with 'system_prompt' and 'user_query'
            
        Returns:
            LLM response content
        """
        # system prompt = system prompt + additional instructions + examples  
        system_prompt = self.system_prompt + "\n\n" + additional_instructions + "\n\n" + self.examples
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_input}
        ]

        response = self.llm(messages=messages, max_tokens=65536)
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
        return self.solve(self.additional_instructions, task)


class VeribenchGuide(Guide):
    """
    Guide that uses lean_interpreter to evaluate Veribench responses
    and provide feedback.
    """

    def __init__(self):
        super().__init__() 

    def get_feedback(self, task, response, info=None, **kwargs):
        """
        Get feedback from the agent's Lean code response.
        
        Args:
            task: The task being evaluated
            response: The LLM-generated Lean code response
            info: Additional info (optional)
            
        Returns:
            Tuple of (score, feedback)
        """
        response = remove_import_error(response)
        try:
            result = lean_interpreter(response)
            correctness = result["valid"]
            score = 1.0 if correctness else 0.0

            if correctness:
                feedback = "The answer is correct! No need to change anything."
            else:
                error_message = "\n\n".join(result["error_messages"])
                summary_message = result["summary"]
                feedback = f"The answer is wrong. {summary_message}\n\nErrors:\n{error_message}\n\nPlease modify the prompt to help LLM produce correct Lean code."

            return score, feedback

        except Exception as e:
            error_str = str(e)
            # Check for system errors that should be raised, not returned as feedback
            # system_errors = [
            #     "event loop is already running",
            #     "event loop",
            #     "asyncio",
            #     "RuntimeError",
            # ]
            # if any(err.lower() in error_str.lower() for err in system_errors):
            #     print_color(f"System error (not a Lean compilation error): {error_str}. This is likely due to asyncio conflicts.", "red")
            #     # breakpoint()
            #     raise RuntimeError(
            #         f"System error (not a Lean compilation error): {error_str}. "
            #         "This is likely due to asyncio conflicts."
            #     ) from e
            return 0.0, f"Error occurred: {error_str}. Please fix the error and try again."

    def metric(self, task, response, info=None, **kwargs):
        """Metric for the agent's performance."""
        score, _ = self.get_feedback(task, response, info, **kwargs)
        return score

from datasets import load_dataset

def create_dataset(num_tasks: int):
    """Load the first num_tasks tasks from the Veribench dataset."""
    dataset = load_dataset("allenanie/veribench_with_prompts")
    tasks = dataset['train'][:num_tasks]
    
    # HuggingFace slice returns dict of lists: {'user_query': [...], ...}
    inputs = tasks['user_query']
    infos = list(range(num_tasks))
    
    return {'inputs': inputs, 'infos': infos}

import json
import copy


def run_single_agent(agent, task_input, task_info):
    """Run a single agent forward pass."""
    response = agent(task_input)
    lean_code = response.data if hasattr(response, 'data') else response
    return {
        'task_info': task_info,
        'lean_code': lean_code
    }


def run_single_eval(guide, lean_code, task_info):
    """Run a single evaluation using the guide."""
    score, feedback = guide.get_feedback(task=None, response=lean_code, info=task_info)
    return {
        'task_info': task_info,
        'score': score,
        'feedback': feedback
    }


if __name__ == "__main__":
    agent = VeribenchAgent()
    guide = VeribenchGuide()

    dataset = create_dataset(10)

    # Step 1: Run 10 agent forward passes asynchronously
    print_color("Step 1: Running 10 agent forward passes asynchronously...", "cyan")
    agent_runs = [lambda task_input, task_info: run_single_agent(agent, task_input, task_info)] * 10
    agent_args = [(dataset['inputs'][i], dataset['infos'][i]) for i in range(10)]
    agent_kwargs = [{}] * 10

    agent_results = async_run(agent_runs, agent_args, agent_kwargs, max_workers=10, description="Agent forward")

    # Extract lean codes from agent results
    lean_codes = [result['lean_code'] for result in agent_results]

    # Step 2: Run 10 evaluations asynchronously
    print_color("\nStep 2: Running 10 evaluations asynchronously...", "cyan")
    eval_runs = [lambda lean_code, task_info: run_single_eval(guide, lean_code, task_info)] * 10
    eval_args = [(lean_codes[i], i) for i in range(10)]
    eval_kwargs = [{}] * 10

    eval_results = async_run(eval_runs, eval_args, eval_kwargs, max_workers=1, description="Lean evaluation")

    # Print results
    print_color(f"\n{'='*50}", "cyan")
    print_color("Results:", "magenta")
    total_score = 0
    for i, (agent_res, eval_res) in enumerate(zip(agent_results, eval_results)):
        total_score += eval_res['score']
        status = "✓" if eval_res['score'] == 1.0 else "✗"
        color = "green" if eval_res['score'] == 1.0 else "red"
        print_color(f"Task {i}: {status} Score = {eval_res['score']}", color)
        if eval_res['score'] < 1.0:
            print_color(f"  Feedback: {eval_res['feedback']}...", "yellow")

    print_color(f"\n{'='*50}", "cyan")
    print_color(f"Total Score: {total_score}/10 ({total_score*10:.0f}%)", "magenta")

    # Save combined results to JSON
    output_data = {
        'lean_codes': lean_codes,
        'agent_results': agent_results,
        'eval_results': eval_results,
        'total_score': total_score
    }

    with open('agent_responses.json', 'w') as f:
        json.dump(output_data, f, indent=2)

    print_color(f"Saved results to agent_responses.json", "green")



