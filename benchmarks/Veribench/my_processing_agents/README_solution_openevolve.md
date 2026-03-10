# OpenEvolve Solution for Veribench

## Overview

This solution uses **OpenEvolve**, an evolutionary coding agent, to translate Python programs to Lean 4 code through iterative evolution. Unlike sequential refinement approaches (DSPy) or candidate-based search (PrioritySearch), OpenEvolve maintains a **diverse population** of solutions that evolve over time.

**Goal**: Generate compilable Lean 4 code that correctly translates a Python program from the Veribench dataset.

---

## How Evolution Works

### 1. Initialization

```
Start with:
- Python program from Veribench dataset
- Initial placeholder Lean code: "-- Lean 4 translation of the Python program"
- Empty population
- VeribenchGuide for evaluation
```

The initial placeholder is evaluated and added to the population as the first member.

---

### 2. Evolution Loop (Each Iteration)

Each iteration follows this process:

#### **Step 2.1: Sample Parent** 🎯

Select one program from the population to serve as the parent:
- **80% Exploitation**: Pick from best-performing programs (high scores)
- **20% Exploration**: Pick diverse programs (different structures)

**When all scores are 0** (early iterations):
- Selection based on code complexity and structural diversity
- Ensures different approaches are tried

#### **Step 2.2: Build Prompt** 📝

Construct a detailed prompt for the LLM (Claude) containing:

**System Message:**
- Task description: Translate Python to Lean 4
- Requirements: Complete, compilable code
- Instructions from Veribench prompts

**User Message:**
- Current parent code (Lean 4)
- Top 3-5 programs from population (inspiration)
- **Compilation feedback from previous attempt** ← Critical!
  - Error messages from Lean compiler
  - Specific issues to fix
  - Success indicators

**Example of what Claude sees:**
```
Previous attempt resulted in:
stderr: "Lean compilation FAILED with 3 error(s).

Errors:
error: unknown identifier 'List.map'
  at line 5: result := List.map f xs
error: type mismatch at application
  at line 12: return result"

Now improve the code to fix these errors...
```

#### **Step 2.3: Generate New Code** 🤖

Claude generates a **complete rewrite** of the Lean code:
- Not a diff/patch, but full new implementation
- Informed by compilation errors from parent
- Attempts to fix specific issues identified

#### **Step 2.4: Evaluate** ⚖️

New code is evaluated with VeribenchGuide:

```
Call: guide.get_feedback(lean_code)
  ↓
Lean 4 compiler attempts compilation
  ↓
Returns:
  score = 1.0 if successful compilation
  score = 0.0 if compilation errors
  feedback = Full error messages or success confirmation
```

#### **Step 2.5: Add to Population** 📊

The new program is added to the population using **MAP-Elites algorithm**:

**MAP-Elites Grid:**
- Population organized in multi-dimensional grid
- Dimensions: **Complexity** (code length) × **Diversity** (structural difference)
- Each grid cell holds ONE program
- New program replaces cell occupant if better

**Example Grid (simplified):**
```
           │ Low Diversity │ Medium Diversity │ High Diversity
───────────┼───────────────┼──────────────────┼────────────────
Short Code │ Program A     │ Program B        │ Program C
           │ (score=0.0)   │ (score=0.0)      │ (score=1.0)✓
───────────┼───────────────┼──────────────────┼────────────────
Medium     │ Program D     │ (empty)          │ Program E
           │ (score=0.0)   │                  │ (score=0.0)
───────────┼───────────────┼──────────────────┼────────────────
Long Code  │ Program F     │ Program G        │ (empty)
           │ (score=0.0)   │ (score=0.0)      │
```

**Key Properties:**
- Maximum 30 programs maintained (population size limit)
- Diverse approaches preserved even with same scores
- Best program globally tracked separately

#### **Step 2.6: Check Early Stopping** 🛑

After each iteration, check if evolution should stop:

**Stop Conditions:**
1. **Success achieved**: Score = 1.0
   - Continue for 10 more iterations (to find simpler solutions)
   - Then stop
   
2. **No improvement**: 10 iterations without score increase
   - Evolution has converged
   - Return best program found

**Otherwise**: Continue to next iteration

---

### 3. Key Evolutionary Mechanisms

#### **Diversity Maintenance (MAP-Elites)**

Why maintain diverse programs when only score matters?

1. **Exploration of Solution Space**
   - Different structural approaches to translation
   - Some may be closer to success than others
   - Prevents premature convergence

2. **Different Error Patterns**
   - Each program has unique compilation errors
   - Claude learns from variety of failure modes
   - Combines insights from multiple approaches

3. **Complementary Progress**
   - Program A: Correct imports, wrong types
   - Program B: Wrong imports, correct types  
   - Program C: Combines insights → Success!

#### **Artifact Feedback Loop**

The most critical feature for Lean code generation:

**Without Artifacts (score only):**
```
Iteration N:   Score = 0.0
Iteration N+1: Score = 0.0  [LLM guesses blindly]
Iteration N+2: Score = 0.0  [No learning]
```

**With Artifacts (score + feedback):**
```
Iteration N:   Score = 0.0
               Feedback: "error: unknown identifier 'List.map'"
               
Iteration N+1: Score = 0.0  [LLM adds import]
               Feedback: "error: type mismatch at line 12"
               
Iteration N+2: Score = 1.0✓ [LLM fixes types]
               Feedback: "SUCCESS!"
```

**What gets passed:**
- Full compilation error messages (unmodified from VeribenchGuide)
- Line numbers and specific issues
- Type errors, syntax errors, import errors
- Success confirmation when compilation succeeds

#### **Exploitation vs Exploration**

The algorithm balances two competing goals:

**Exploitation (80%):** 
- Sample from best-performing programs
- Refine successful approaches
- Make incremental improvements

**Exploration (20%):**
- Sample diverse, different programs  
- Try novel approaches
- Escape local optima

This ratio is configurable but 80/20 works well for single-objective optimization.

---

## Why This Works for Lean Translation

### 1. **Compilation is Binary but Errors are Rich**

- Score: 0 or 1 (pass/fail)
- Feedback: Detailed error messages with context
- Evolution uses feedback to guide search

### 2. **Multiple Valid Solutions**

- Different ways to translate Python to Lean
- Population explores alternatives simultaneously
- Finds simplest/best among valid solutions

### 3. **Complex Error Landscapes**

- One fix may reveal new errors
- Sequential approaches can get stuck
- Population maintains multiple "hypothesis" approaches

### 4. **Incremental Learning**

- Each generation builds on previous insights
- Errors guide next mutations
- Converges faster than random exploration

---

## Comparison with Other Methods

| Aspect | DSPy | PrioritySearch | OpenEvolve |
|--------|------|----------------|------------|
| **Population** | 1 (current only) | 1-5 candidates | 30 diverse programs |
| **Diversity** | None | Limited | High (MAP-Elites) |
| **Feedback Used** | ✓ Last attempt | ✓ Top candidates | ✓ All programs |
| **Exploration** | Sequential | Best-first | 20% random |
| **Convergence** | Linear | Fast to local optimum | Balanced |
| **Best For** | Simple refinement | Quick convergence | Complex search spaces |

**When to use OpenEvolve:**
- Multiple valid solutions exist
- Error patterns are complex
- Want to explore solution space thoroughly
- Have budget for 50+ iterations

---

## Evolution Parameters

### Population Management
- **Population Size**: 30 programs
- **Islands**: 1 (single population)
- **Feature Dimensions**: Complexity, Diversity (built-in)

### Selection Strategy
- **Exploitation Ratio**: 0.8 (80% best programs)
- **Exploration Ratio**: 0.2 (20% diverse)

### Early Stopping
- **Patience**: 10 iterations without improvement
- **Target Score**: 1.0 (successful compilation)
- **Strategy**: Continue 10 iterations after success

### Evolution Strategy
- **Type**: Full rewrites (not diff-based)
- **Reason**: Lean syntax is complex, diffs can break structure
- **Artifacts**: Enabled (critical for feedback)

---

## Example Evolution Trace

```
Iteration 1:
  Parent: "-- Lean 4 translation..." (placeholder)
  → Generate new code
  → Score: 0.0, Error: "unknown identifier 'List'"
  → Add to population [1 program]

Iteration 2:
  Sample parent from population (only 1 choice)
  See error about List
  → Generate with imports
  → Score: 0.0, Error: "type mismatch: expected Nat, got Int"
  → Add to population [2 programs]

Iteration 3:
  Sample parent (80% chance: best = most recent)
  See type mismatch error
  → Generate with correct types
  → Score: 1.0, Success! ✓
  → Add to population [3 programs]

Iterations 4-13:
  Continue searching for simpler solutions
  No better than 1.0 found
  → Early stopping triggered at iteration 13

Result: Return program from iteration 3
```

---

## Key Takeaways

1. **Population-Based**: Maintains 30 diverse solutions, not just one
2. **Artifact-Driven**: Compilation errors guide evolution, not just scores
3. **Balanced Search**: 80% exploit best, 20% explore alternatives
4. **MAP-Elites**: Preserves diversity across complexity and structure
5. **Early Stopping**: Stops automatically when optimal solution found
6. **Single Objective**: Optimizes one metric (compilation success)

The evolution process is designed to systematically explore the space of Lean translations, learning from compilation errors at each step, until a valid solution is found.

