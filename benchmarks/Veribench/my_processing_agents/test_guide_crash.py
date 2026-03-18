"""
Simple test to check if lean code with native_decide causes timeout.

Usage:
    cd /Users/xuanfeiren/Documents/Trace-Bench/Veribench
    uv run python my_processing_agents/test_guide_crash.py
"""

import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from my_processing_agents.lean_interpretor import lean_interpreter, remove_import_error

# Simple Lean code that compiles successfully
SIMPLE_LEAN_CODE = """
def hello : Nat := 42

def add (x y : Nat) : Nat := x + y

#eval hello
#eval add 3 5
"""

# Lean code with multiple native_decide examples
LEAN_CODE = """/-!
# Binary Search Implementation in Lean 4

This module implements binary search over a sorted list of natural numbers.

Edge cases:
- Empty list returns none
- Type safety enforced by Lean's type system
-/

namespace BinarySearch

/--
Binary search implementation that searches for a target value in a sorted list.
Returns `some index` if found, `none` if not found.

## Examples
```lean
#eval binarySearch [1, 2, 3, 4, 5] 3  -- expected: some 2
#eval binarySearch [1, 2, 3, 4, 5] 6  -- expected: none
#eval binarySearch [] 1               -- expected: none
```
-/
def binarySearch (arr : List Nat) (target : Nat) : Option Nat :=
  if arr.isEmpty then
    none
  else
    let rec search (left right : Nat) : Option Nat :=
      if left > right then
        none
      else
        let mid := (left + right) / 2
        match arr.get? mid with
        | none => none  
        | some midVal =>
          if midVal = target then
            some mid
          else if midVal < target then
            search (mid + 1) right
          else
            search left (if mid = 0 then 0 else mid - 1)
    termination_by search left right => right - left
    
    search 0 (arr.length - 1)

section Tests

/-- Basic test cases -/
example : binarySearch [1, 2, 3, 4, 5] 3 = some 2 := by native_decide
example : binarySearch [1, 2, 3, 4, 5] 6 = none := by native_decide
example : binarySearch ([] : List Nat) 1 = none := by native_decide
example : binarySearch [5] 5 = some 0 := by native_decide
example : binarySearch [5] 3 = none := by native_decide

/-- Tests with larger arrays -/
example : binarySearch [1, 3, 5, 7, 9] 3 = some 1 := by native_decide
example : binarySearch [1, 3, 5, 7, 9] 7 = some 3 := by native_decide
example : binarySearch [1, 3, 5, 7, 9] 4 = none := by native_decide
example : binarySearch [10, 20, 30, 40, 50] 50 = some 4 := by native_decide
example : binarySearch [10, 20, 30, 40, 50] 10 = some 0 := by native_decide

/-- Tests with two-element arrays -/
example : binarySearch [1, 2] 1 = some 0 := by native_decide
example : binarySearch [1, 2] 2 = some 1 := by native_decide
example : binarySearch [1, 2] 3 = none := by native_decide

end Tests

/-!
# Theorems
-/

/-- Property: if target is found, the index must be valid -/
theorem found_index_valid {arr : List Nat} {target idx : Nat} 
  (h : binarySearch arr target = some idx) : 
  idx < arr.length := sorry

/-- Property: if target is found, it must be at the returned index -/
theorem found_value_correct {arr : List Nat} {target idx : Nat}
  (h : binarySearch arr target = some idx) :
  arr.get? idx = some target := sorry

/-- Property: empty list always returns none -/
theorem empty_list_none : 
  ∀ (target : Nat), binarySearch ([] : List Nat) target = none := by 
  intro target
  native_decide

end BinarySearch"""

# LEAN_CODE = """/-!
# # Binary Search Implementation in Lean 4

# This module implements binary search over a sorted list of natural numbers.

# Edge cases:
# - Empty list returns none
# - Type safety enforced by Lean's type system
# -/

# namespace BinarySearch

# /--
# Binary search implementation that searches for a target value in a sorted list.
# Returns `some index` if found, `none` if not found.

# ## Examples
# ```lean
# #eval binarySearch [1, 2, 3, 4, 5] 3  -- expected: some 2
# #eval binarySearch [1, 2, 3, 4, 5] 6  -- expected: none
# #eval binarySearch [] 1               -- expected: none


# end BinarySearch"""

def main():
    # Test 1: Simple code
    print("=" * 60)
    print("TEST 1: Simple Lean code (should compile quickly)")
    print("=" * 60)
    print(f"Code ({len(SIMPLE_LEAN_CODE)} chars):")
    print(SIMPLE_LEAN_CODE)
    
    print("\nRunning lean_interpreter...")
    start = time.time()
    result = lean_interpreter(SIMPLE_LEAN_CODE)
    elapsed = time.time() - start
    
    print(f"\n✅ Valid: {result['valid']}")
    print(f"Num errors: {result['num_errors']}")
    print(f"Time: {elapsed:.2f}s")
    
    # Test 2: Complex code with native_decide
    print("\n" + "=" * 60)
    print("TEST 2: Complex code with native_decide (may timeout)")
    print("=" * 60)
    print(f"Code ({len(LEAN_CODE)} chars)")
    print(f"First 150 chars: {LEAN_CODE[:150]}...\n")
    
    print("Running lean_interpreter (60s timeout)...")
    start = time.time()
    
    try:
        result = lean_interpreter(LEAN_CODE)
        elapsed = time.time() - start
        
        print(f"\nValid: {result['valid']}")
        print(f"Num errors: {result['num_errors']}")
        print(f"Time: {elapsed:.2f}s")
        print(f"\nError messages: {result['error_messages'][:3]}")  # Show first 3
        if result.get('error_details'):
            print(f"\nError details (first 2):")
            for detail in result['error_details'][:2]:
                print(detail[:200] + "...")
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n❌ Exception after {elapsed:.2f}s: {type(e).__name__}")
        print(f"Message: {str(e)[:300]}")


if __name__ == "__main__":
    main()

