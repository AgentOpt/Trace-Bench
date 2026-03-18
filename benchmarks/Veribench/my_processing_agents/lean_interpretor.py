import os
import traceback
from typing import Dict, Any, List
from pantograph.server import Server
from pantograph.data import CompilationUnit

from my_processing_agents.lean4_utils import default_server, get_list_lean4_all_mgs_and_error_mgs

def get_indentation_block(code_lines: List[str], line_num: int) -> List[str]:
    """
    Get the indentation block containing the given line number.
    Returns the lines that are at the same indentation level or more indented.
    For outermost level lines (no indentation), returns just 1 line above and below.
    """
    if not code_lines or line_num < 1 or line_num > len(code_lines):
        return []
    
    target_line = code_lines[line_num - 1]
    target_indent = len(target_line) - len(target_line.lstrip())
    
    # If the line has no indentation, just return 1 line above and below
    if target_indent == 0:
        start = max(0, line_num - 2)  # -2 because line_num is 1-based
        end = min(len(code_lines), line_num + 1)  # +1 because we want one line after
        return code_lines[start:end]
    
    # Find the start of the block (first line with same or less indentation)
    start = line_num - 1
    while start > 0:
        prev_line = code_lines[start - 1]
        prev_indent = len(prev_line) - len(prev_line.lstrip())
        if prev_indent < target_indent:
            break
        start -= 1
    
    # Find the end of the block (first line with less indentation)
    end = line_num
    while end < len(code_lines):
        next_line = code_lines[end]
        next_indent = len(next_line) - len(next_line.lstrip())
        if next_indent < target_indent:
            break
        end += 1
    
    return code_lines[start:end]

def format_error_context(error_msg: str, code_lines: List[str], line_num: int) -> str:
    """
    Format error message with code context in a human-readable way.
    Shows both the error line and the line where the error was discovered.
    """
    # Get the indentation block for the error line
    block_lines = get_indentation_block(code_lines, line_num)
    if not block_lines:
        return f"Error at line {line_num}: {error_msg}\nNo context available."
    
    # Format the output
    output = []
    output.append(f"\nError discovered at line {line_num}:")
    output.append(f"Message: {error_msg.strip()}")
    output.append("\nCode context (indentation block):")
    
    # Calculate the actual starting line number of the block
    target_line = code_lines[line_num - 1]
    target_indent = len(target_line) - len(target_line.lstrip())
    
    if target_indent == 0:
        # For unindented lines, we know exactly which lines we're showing
        start_line = max(1, line_num - 1)
    else:
        # For indented blocks, find the first line with less indentation
        start_line = line_num
        while start_line > 1:
            prev_line = code_lines[start_line - 2]  # -2 because line_num is 1-based and we want previous line
            prev_indent = len(prev_line) - len(prev_line.lstrip())
            if prev_indent < target_indent:
                break
            start_line -= 1
    
    # Add line numbers and highlight the error line
    for i, line in enumerate(block_lines):
        current_line = start_line + i
        # Highlight the actual error line
        is_error_line = (current_line == line_num)
        
        prefix = ">> " if is_error_line else "   "
        output.append(f"{prefix}{current_line:3d} | {line}")
    
    # Add a note about the error discovery
    output.append("\nNote: The error was discovered during compilation at the marked line, but the actual error might be in a different line within this block.")
    
    return "\n".join(output)

def lean_interpreter(lean4_code: str, 
                    #  server: Any = default_server, # Any type otherwise DSPy's pydantic type checker complains that it doesn't know PyPantograph Server type.
                     ) -> Dict[str, Any]:
    """
    Check and interpret Lean4 code using the PyPantograph server.
    Adds a 'summary' field to guide the ReAct agent.
    
    Args:
        lean4_code: String containing Lean4 code to check
        
    Returns:
        Dictionary containing:
        - valid: Boolean indicating if code compiled successfully
        - messages: All compiler messages (errors, warnings, info)
        - errors: List of error messages only
        - has_errors: Boolean indicating presence of errors
        - num_errors: String indicating the number of errors.
        - summary: Natural language summary of the outcome, advising the next step.
        - error_details: List of formatted error messages with code context
    """
    server: Server = Server(
        imports=["Init"], 
        # project_path=os.path.expanduser("~/mathlib_4_15_0_lfs"),
        timeout=60,
    )
    result: Dict[str, str] = get_list_lean4_all_mgs_and_error_mgs(lean4_code, server)
    all_messages: List[str] = result['all_messages']
    error_messages: List[str] = result['error_messages']
    num_errors: int = len(error_messages)
    has_errors: bool = num_errors > 0

    # Process error messages to include context
    error_details = []
    code_lines = lean4_code.strip().split('\n')
    
    for error_msg in error_messages:
        try:
            # Parse the error message to get line number
            parts = error_msg.split(':')
            if len(parts) >= 2:
                line_num = int(parts[1])
                formatted_error = format_error_context(error_msg, code_lines, line_num)
                error_details.append(formatted_error)
            else:
                error_details.append(f"Could not parse line number from error: {error_msg}")
        except Exception as e:
            error_details.append(f"Error processing message: {error_msg}\nException: {str(e)}")

    if has_errors:
        # Check for specific error types to provide more helpful feedback
        has_coupling_error = any("metavariable coupling" in msg or "Coupling is not allowed" in msg 
                                  for msg in error_messages)
        
        if has_coupling_error:
            summary = (
                f"Lean code compilation FAILED with {num_errors} errors including a METAVARIABLE COUPLING issue. "
                f"This means your proof has 'sorry' placeholders that depend on each other "
                f"(e.g., 'have h1 := sorry; have h2 : Type h1 := sorry'). "
                f"To fix: (1) Complete intermediate proofs instead of using sorry, "
                f"(2) Avoid 'have'/'let' where one depends on another's placeholder, or "
                f"(3) Restructure the proof to eliminate coupled dependencies."
            )
        else:
            summary = f"Lean code compilation FAILED with {num_errors} errors. Errors MUST be fixed before finishing. Review the 'errors' list and try again (for example,if in a React agent loop)."
    else:
        summary = "Lean code compiled successfully with 0 errors. Proceed to finish if the task is complete."
            
    return {
        "valid": not has_errors,
        "all_messages": all_messages,
        "error_messages": error_messages,
        "has_errors": has_errors,
        "num_errors": num_errors,
        "summary": summary,
        "error_details": error_details
    }


def remove_import_error(lean4_code: str) -> str:
    """
    Remove import statements that cause errors in the Lean 4 code.
    Since imports only happen at the beginning of the file, we can simplify
    the approach to just identify and remove problematic import lines.
    
    Args:
        lean4_code: The Lean 4 code string
        
    Returns:
        Modified Lean 4 code with all problematic import statements removed
    """
    # First check if there are any import errors
    result = lean_interpreter(lean4_code)
    if not result["valid"]:
        error_messages = result["error_messages"]
    else:
        # No errors to fix
        return lean4_code
    
    code_lines = lean4_code.strip().split('\n')
    problematic_lines = set()
    
    for error_msg in error_messages:
        # Look for import errors
        if "invalid 'import' command" in error_msg:
            try:
                # Parse the error message to get line number
                parts = error_msg.split(':')
                if len(parts) >= 3:
                    line_num = int(parts[1])
                    # Lean file starts with line 1, not line 0
                    if 1 <= line_num <= len(code_lines):
                        problematic_lines.add(line_num - 1)
            except Exception as e:
                print(f"Error processing import error: {error_msg}\nException: {str(e)}")
    
    # Remove all problematic import lines at once
    if problematic_lines:
        filtered_lines = [line for i, line in enumerate(code_lines) if i not in problematic_lines]
        modified_code = '\n'.join(filtered_lines)
        
        # Check if there are still import errors
        result = lean_interpreter(modified_code)
        if not result["valid"] and any("invalid 'import' command" in msg for msg in result["error_messages"]):
            # Recursively remove remaining import errors
            return remove_import_error(modified_code)
        
        return modified_code
                
    return lean4_code

if __name__ == "__main__":
    # Use a default example file
    lean4_code = None
    try:
        # with open('fs/my_add.lean', 'r') as f:
        # with open('trace_test/my_max.lean', 'r') as f:
        with open('dev_lean_from_trace_3_7_max_try_10/humaneval_7.lean', 'r') as f:
            lean4_code = f.read().strip()
    except Exception as e:
        print(f"Error reading default file fs/my_max.lean: {e}")
        exit(1)
    
    result = lean_interpreter(remove_import_error(lean4_code))
    for d in result['error_messages']:
        print(d)

    for d in result['error_details']:
        print(d)