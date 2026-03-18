import sys
import os
import time
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from guide.guide import VeribenchGuide
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

CODE ="""/-!
# Binary Search Implementation

This module implements binary search over a sorted list of integers.
Returns the index if found, none if not found.
-/

namespace BinarySearch

/--
Binary search implementation that searches for a target value in a sorted array.
Returns `some idx` if found at index idx, `none` if not found.
-/
def binarySearch (arr : Array Int) (target : Int) : Option Nat :=
  if arr.isEmpty then
    none
  else
    let rec search (left right : Nat) : Option Nat :=
      if left > right then
        none
      else
        let mid := (left + right) / 2
        let midVal := arr[mid]!
        if midVal == target then
          some mid
        else if midVal < target then
          search (mid + 1) right
        else
          search left (mid - 1)
    search 0 (arr.size - 1)

/-!
# Essential Tests
-/

/-- Test: Basic search functionality -/
example : binarySearch #[1, 2, 3, 4, 5] 3 = some 2 := by native_decide
#eval binarySearch #[1, 2, 3, 4, 5] 3  -- expected: some 2

/-- Test: Element not found -/
example : binarySearch #[1, 2, 3, 4, 5] 6 = none := by native_decide

/-- Test: Empty array -/
example : binarySearch #[] 1 = none := by native_decide

/-- Test: Single element array -/
example : binarySearch #[5] 5 = some 0 := by native_decide

/-!
# Core Properties
-/

/-- Property: Found element is at correct index -/
def found_correct_index (arr : Array Int) (target : Int) (idx : Nat) : Prop :=
  (binarySearch arr target = some idx) → arr[idx]! = target

/-- Basic correctness theorem -/
theorem found_correct_index_thm (arr : Array Int) (target : Int) (idx : Nat) :
  found_correct_index arr target idx := sorry

end BinarySearch"""

def main():
    guide = VeribenchGuide()
    # score, feedback = guide.get_feedback(task=None, response=SIMPLE_LEAN_CODE, info=None)
    # print(f"Simple Lean code: {SIMPLE_LEAN_CODE}")
    # print(f"Score: {score}")
    # print(f"Feedback: {feedback}")

    score, feedback = guide.get_feedback(task=None, response=LEAN_CODE, info=None)
    print(f"Complex Lean code: {LEAN_CODE[:100]}...")
    print(f"Score: {score}")
    print(f"Feedback: {feedback}")

    # score, feedback = guide.get_feedback(task=None, response=CODE, info=None)
    # print(f"Code: {CODE[:100]}...")
    # print(f"Score: {score}")
    # print(f"Feedback: {feedback}")

if __name__ == "__main__":
    main()
