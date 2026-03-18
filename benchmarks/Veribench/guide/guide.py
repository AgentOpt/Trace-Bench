import sys
import tempfile
from pathlib import Path
import requests

# Setup paths BEFORE importing project modules
_guide_dir = Path(__file__).resolve().parent
_project_root = _guide_dir.parent
_veribench_dataset_utils = _project_root / "veribench_dataset_utils"

# Add project root to path for my_processing_agents import
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Setup VeriBench paths for unit test integration (done at module level)
_veribench_root = _project_root / "self-opt-data-gen" / "veribench_bundle"
_py_src = _veribench_root / "veribench_dataset" / "py_src"
_experiments_path = _veribench_root / "experiments" / "12_unit_test_accuracy"

# Add veribench_dataset_utils to path for eval_utils import
if str(_veribench_dataset_utils) not in sys.path:
    sys.path.insert(0, str(_veribench_dataset_utils))

if str(_py_src) not in sys.path:
    sys.path.insert(0, str(_py_src))
if str(_experiments_path) not in sys.path:
    sys.path.insert(0, str(_experiments_path))

# Now import project modules that depend on the paths
from opto.trainer.guide import Guide
from my_processing_agents.lean_interpretor import lean_interpreter
from opto.optimizers.utils import print_color

# VeriBench utilities will be imported lazily in functions to avoid dependency issues

def remove_import_error(response,result):
    """Remove the import error from the response."""
    error_messages = result["error_messages"]
    has_import_errors = any("invalid 'import' command" in msg for msg in error_messages)

    if has_import_errors:
        # Remove import error lines
        code_lines = response.strip().split('\n')
        problematic_lines = set()
        
        for error_msg in error_messages:
            if "invalid 'import' command" in error_msg:
                try:
                    parts = error_msg.split(':')
                    if len(parts) >= 3:
                        line_num = int(parts[1])
                        if 1 <= line_num <= len(code_lines):
                            problematic_lines.add(line_num - 1)
                except Exception:
                    pass
        
        # Remove problematic import lines and re-evaluate
        if problematic_lines:
            filtered_lines = [line for i, line in enumerate(code_lines) if i not in problematic_lines]
            response = '\n'.join(filtered_lines)
    return response

def compile(response):
    """Compile the lean code, return the score and feedback.
    
    Args:
        task: The task being evaluated (user query)
        response: The LLM-generated Lean code response
        
    Returns:
        Tuple of (score, feedback)
        The score and feedback only contains the information about the compilation error.
    """
    try:
        result = lean_interpreter(response)
        correctness = result["valid"]
        score = 1.0 if correctness else 0.0

        if correctness:
            feedback = "The answer is correct! No need to change anything."
            return score, feedback
        
        num_errors = result["num_errors"]
        
        # Get error details with context if available, otherwise raw messages
        error_details = result.get("error_details", result["error_messages"])
        errors_str = "\n\n".join(error_details)
        
        # Build feedback
        feedback = f"Lean compilation FAILED with {num_errors} error(s). Please make the Lean code correct and as simple as possible.\n\nErrors:\n{errors_str}"

        if 'TimeoutError' in errors_str:
            # Time out error is not a simple Lean compilation error, so we return 0.0.
            print_color("Lean code compilation TIMEOUT. Return timeout feedback.", "yellow")
            return 0.0, "Lean code compilation TIMEOUT. The generated code is either incorrect or too complex for the interpreter to compile within the time limit. Please make the Lean code correct and as simple as possible."
        
        # I assume there will not be timeout error after removing import errors...

        cleaned_code = remove_import_error(response,result)
        if cleaned_code != response:
            # Re-run interpreter on cleaned code
            result = lean_interpreter(cleaned_code)
            # assert current no import errors
            while any("invalid 'import' command" in msg for msg in result["error_messages"]):
                cleaned_code = remove_import_error(cleaned_code,result)
                result = lean_interpreter(cleaned_code)

            assert not any("invalid 'import' command" in msg for msg in result["error_messages"]), "There are still import errors after removing import errors."
        
            correctness = result["valid"]
            score = 1.0 if correctness else 0.0
        
            # Update feedback based on new result
            if correctness:
                feedback = "The answer is correct after removing invalid import statements!"
            else:
                num_errors = result["num_errors"]
                error_details = result.get("error_details", result["error_messages"])
                errors_str = "\n\n".join(error_details)
                feedback = f"Lean compilation FAILED with {num_errors} error(s) after removing invalid imports. Please make the Lean code correct and as simple as possible.\n\nErrors:\n{errors_str}"

        return score, feedback

    except Exception as e:
        print_color(f"Error: {e}", "red")
        raise e
        # error_str = str(e)
        # # Check for system errors that should be raised, not returned as feedback
        # system_errors = [
        #     "event loop is already running",
        #     "event loop",
        #     "asyncio",
        #     "RuntimeError",
        # ]
        # if any(err.lower() in error_str.lower() for err in system_errors):
        #     print_color(f"System error (not a Lean compilation error): {error_str}. This is likely due to asyncio conflicts.", "red")
        #     raise RuntimeError(
        #         f"System error (not a Lean compilation error): {error_str}. "
        #         "This is likely due to asyncio conflicts."
        #     ) from e
        # return 0.0, f"Error occurred: {error_str}. Please fix the error and try again."

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
            task: The task being evaluated (user query)
            response: The LLM-generated Lean code response
            info: Additional info (optional)
            
        Returns:
            Tuple of (score, feedback)
        """
        score, feedback = compile(response)
        return score, feedback

    def metric(self, task, response, info=None, **kwargs):
        """Metric for the agent's performance."""
        score, _ = self.get_feedback(task, response, info, **kwargs)
        return score

def combine_code_with_tests(task_id: int, lean_code: str) -> str:
    """
    Combine the lean code with the unit tests from gold reference.
    
    This function:
    1. Extracts implementation from generated lean_code
    2. Extracts unit tests from gold reference
    3. Automatically matches function names and creates aliases
    4. Combines them into a compilable Lean file
    
    Args:
        task_id: Task ID (0-139)
        lean_code: Generated Lean code
        
    Returns:
        Combined Lean code with matched unit tests
    """
    # Lazy import to avoid dependency issues
    from veribench.metrics.lean_test_extractor import extract_unit_tests as extract_tests
    from main_eval_unit_test_acc import create_combined_file
    import re
    import logging
    
    # Suppress INFO logging from lean_test_extractor
    logging.getLogger('veribench.metrics.lean_test_extractor').setLevel(logging.WARNING)
    
    # Alternative: Load from our local dataset JSON to avoid fastchat dependency
    dataset_root = Path(__file__).resolve().parent.parent / "veribench_dataset_utils" / "dataset"
    task_file = dataset_root / f"task_{task_id}.json"
    
    if not task_file.exists():
        raise FileNotFoundError(f"Task file not found: {task_file}")
    
    import json
    with open(task_file, 'r') as f:
        task_data = json.load(f)
    
    # Use gold reference code directly from our dataset
    gold_lean_code = task_data.get('gold_reference_lean4_code', '')
    if not gold_lean_code:
        raise ValueError(f"No gold reference Lean code found for task {task_id}")
    
    # Save gold reference to temp file for extraction
    with tempfile.NamedTemporaryFile(mode='w', suffix='.lean', delete=False, encoding='utf-8') as temp_gold:
        temp_gold.write(gold_lean_code)
        temp_gold_path = temp_gold.name
    
    gold_lean_file = Path(temp_gold_path)
    
    try:
        # More robust extraction: Extract all function definitions from generated code
        # This handles multi-line definitions and various formats better
        implementation = _extract_implementation_robust(lean_code)
        
        # Extract unit tests from gold reference
        unit_tests = extract_tests(str(gold_lean_file))
        
        # Use VeriBench's smart combining function
        # This automatically:
        # - Matches function names between implementation and tests
        # - Creates aliases (abbrev) to bridge naming differences
        # - Handles namespaces correctly
        # - Reorders definitions to resolve dependencies
        combined_code = create_combined_file(
            implementation=implementation,
            tests=unit_tests,
            imperative_impl="",  # No imperative implementation for now
            imperative_tests=[],
            disable_aliases=False,  # Enable automatic name matching
            keep_tests_exact=False,  # Allow test modification
            let_llm_order=False  # Use smart reordering
        )
        
        return combined_code
        
    finally:
        # Clean up temp file
        try:
            gold_lean_file.unlink()
        except:
            pass


def _extract_implementation_robust(lean_code: str) -> str:
    """
    Robust extraction of implementation code from generated Lean code.
    Extracts imports, namespaces, and all function definitions.
    Stops at test/example sections.
    """
    import re
    
    lines = lean_code.split('\n')
    extracted = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Stop at test/example sections
        if (re.match(r'^\s*/-!\s*#?\s*(Unit\s*)?Tests', line, re.IGNORECASE) or
            re.match(r'^\s*--\s*Unit\s*[Tt]ests', line) or
            re.match(r'^\s*example\s*:', line) or
            re.match(r'^\s*#eval\s+', line)):
            break
        
        # Skip comments and empty lines unless they're doc comments for functions
        if not stripped or stripped.startswith('--') or stripped.startswith('#'):
            i += 1
            continue
        
        # Skip block comments (but keep doc comments)
        if stripped.startswith('/-') and not stripped.startswith('/-!'):
            # Skip until closing -/
            while i < len(lines) and '-/' not in lines[i]:
                i += 1
            i += 1
            continue
        
        # Capture imports, namespaces, opens
        if (stripped.startswith('import ') or 
            stripped.startswith('namespace ') or 
            stripped.startswith('open ') or
            stripped.startswith('variable ') or
            stripped.startswith('universe ')):
            extracted.append(line)
            i += 1
            continue
        
        # Capture function definitions (def, partial def, noncomputable def, abbrev)
        if re.match(r'^\s*(partial\s+|noncomputable\s+)?(def|abbrev)\s+\w+', line):
            # Start capturing the function
            func_lines = [line]
            i += 1
            
            # Continue capturing until we find the end of the function
            # A function ends when we hit a blank line followed by a non-indented line,
            # or another top-level definition
            while i < len(lines):
                next_line = lines[i]
                next_stripped = next_line.strip()
                
                # Stop if we hit another definition or test section
                if (re.match(r'^\s*(partial\s+|noncomputable\s+)?(def|abbrev|theorem|example)\s+', next_line) or
                    re.match(r'^\s*/-!\s*#?\s*(Unit\s*)?Tests', next_line, re.IGNORECASE) or
                    re.match(r'^\s*#eval\s+', next_line)):
                    break
                
                # Include indented lines and empty lines within the function
                if next_stripped or next_line.startswith('  ') or next_line.startswith('\t'):
                    func_lines.append(next_line)
                    i += 1
                else:
                    # Empty line - check if next non-empty line is indented
                    j = i + 1
                    while j < len(lines) and not lines[j].strip():
                        j += 1
                    
                    if j < len(lines) and (lines[j].startswith('  ') or lines[j].startswith('\t')):
                        # Still part of the function
                        func_lines.append(next_line)
                        i += 1
                    else:
                        # End of function
                        break
            
            extracted.extend(func_lines)
            continue
        
        i += 1
    
    return '\n'.join(extracted) 

class VeribenchGuidewithUnitTests(VeribenchGuide):
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
            task: The task query string
            response: Generated Lean code
            info: Must contain 'task_id' (int) for unit test matching
        """
        
        task_id = info
        
        # Test step 1: Compile the lean code
        compile_score, compile_feedback = compile(response)

        if compile_score == 0.0: # compilation failed
            return 0.0, compile_feedback

        # Test step 2: Unit tests with automatic name matching
        lean_code_with_unit_tests = combine_code_with_tests(task_id, response)
        unit_test_score, unit_test_feedback = compile(lean_code_with_unit_tests)
        
        # for debugging, print the lean code before and after the unit tests
        print_color("Lean code before the unit tests: ", "green")
        print_color(response, "yellow")
        print_color("Lean code after the unit tests: ", "green")
        print_color(lean_code_with_unit_tests, "yellow")

        if unit_test_score == 1.0:
            return 1.0, "The answer is correct! All unit tests passed."
        elif unit_test_score == 0.0 and compile_score == 1.0:
            # print_color(unit_test_feedback, "yellow")
            return 0.5, f"The lean code compiled but the unit tests failed. Feedback from the compilation with unit tests: {unit_test_feedback}"
        
    
    def metric(self, task, response, info=None, **kwargs):
        """Metric for the agent's performance with unit tests."""
        score, _ = self.get_feedback(task, response, info, **kwargs)
        return score

from eval_utils import evaluate
class VeribenchGuidewithLLMJudge(VeribenchGuide):
    """
    Guide that uses LLM Judge to evaluate Veribench responses.
    """
    def __init__(self):
        super().__init__()
    
    def get_feedback(self, task, response, info=None, **kwargs):
        """Get feedback from the agent's Lean code response."""
        task_id = info
        score, feedback = evaluate(task_id, response)
        print_color(f"Score: {score}", "yellow")
        return score, feedback

# class WebGuide(Guide):
#     """
#     Guide that uses a web server to evaluate Veribench responses.
#     Calls the Lean Feedback Server via HTTP to get compilation feedback.
    
#     This avoids asyncio conflicts by delegating Lean compilation to a separate process.
    
#     Usage:
#         # First start the server:
#         # uv run python webserver/lean_feedback_server.py --port 8000
        
#         guide = WebGuide(server_url="http://localhost:8000")
#         score, feedback = guide.get_feedback(task, response)
#     """

#     def __init__(self, server_url: str = "http://localhost:8000", timeout: int = 20):
#         """
#         Initialize WebGuide.
        
#         Args:
#             server_url: URL of the Lean Feedback Server
#             timeout: Request timeout in seconds (default 70, Lean Server has 60s internal timeout)
#         """
#         super().__init__()
#         self.server_url = server_url.rstrip("/")
#         self.timeout = timeout
    
#     def _check_server(self) -> bool:
#         """Check if the server is running."""
#         try:
#             response = requests.get(f"{self.server_url}/health", timeout=5)
#             return response.status_code == 200
#         except requests.exceptions.RequestException:
#             return False

#     def get_feedback(self, task, response, info=None, **kwargs):
#         """
#         Get feedback from the agent's Lean code response via web server.
        
#         Args:
#             task: The task being evaluated (user query)
#             response: The LLM-generated Lean code response
#             info: Additional info (optional)
            
#         Returns:
#             Tuple of (score, feedback)
#         """
#         try:
#             # Call the web server
#             http_response = requests.post(
#                 f"{self.server_url}/feedback",
#                 json={
#                     "lean_code": response,
#                     "remove_import_errors": True
#                 },
#                 timeout=self.timeout
#             )
            
#             if http_response.status_code != 200:
#                 error_detail = http_response.json().get("detail", "Unknown error")
#                 return 0.0, f"Server error ({http_response.status_code}): {error_detail}"
            
#             data = http_response.json()
#             score = data["score"]
#             feedback = data["feedback"]
            
#             print_color(feedback, "yellow")
#             return score, feedback

#         except requests.exceptions.ConnectionError:
#             error_msg = (
#                 f"Cannot connect to Lean Feedback Server at {self.server_url}. "
#                 "Please start the server first:\n"
#                 "  uv run python webserver/lean_feedback_server.py --port 8000"
#             )
#             print_color(error_msg, "red")
#             return 0.0, error_msg
        
#         except requests.exceptions.Timeout:
#             error_msg = f"""Compilation TIMEOUT after {self.timeout} seconds.

#         This usually means the Lean code contains constructs that are extremely slow to compile:

#         COMMON CAUSES:
#         1. 'native_decide' tactic - This executes code at compile time and is VERY slow for recursive functions
#         2. 'decide' on large data - Similar issue with compile-time evaluation
#         3. Complex recursive functions without 'termination_by' - Lean struggles to prove termination

#         HOW TO FIX:
#         1. Replace 'native_decide' with 'sorry' or 'rfl' where applicable
#         2. Use '#eval!' instead of 'example ... := by native_decide' for testing
#         3. Add 'termination_by <measure>' and 'decreasing_by all_goals sorry' for recursive functions
#         4. Simplify proofs - use 'sorry' as placeholder for complex theorems

#         EXAMPLE FIX:
#         -- BAD (causes timeout):
#         example : myFunc #[1,2,3] = some 1 := by native_decide
        
#         -- GOOD (fast):
#         #eval! myFunc #[1,2,3]  -- just evaluate, don't prove
#         -- OR
#         example : myFunc #[1,2,3] = some 1 := sorry  -- placeholder proof
#         """
#             print_color(f"TIMEOUT: {self.timeout}s exceeded", "red")
#             print_color(f'Lean code that caused timeout: {response}', "red")
#             return 0.0, error_msg
            
#         except Exception as e:
#             error_str = str(e)
#             print_color(f"WebGuide error: {error_str}", "red")
#             return 0.0, f"Error occurred: {error_str}. Please fix the error and try again."

#     def metric(self, task, response, info=None, **kwargs):
#         """Metric for the agent's performance."""
#         score, _ = self.get_feedback(task, response, info, **kwargs)
#         return score
