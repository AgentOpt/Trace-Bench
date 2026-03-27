# Task Inventory

This page lists all available tasks in Trace-Bench, organized by suite.

To discover tasks at runtime, use the CLI:

```bash
# All tasks
trace-bench list-tasks

# Filter by suite
trace-bench list-tasks --bench llm4ad
trace-bench list-tasks --bench veribench
trace-bench list-tasks --bench trace_examples
trace-bench list-tasks --bench internal
```

---

## Suite Overview

| Suite | Description | Task Count | Source |
|-------|-------------|------------|--------|
| `llm4ad` | Algorithm design problems from [LLM4AD](https://github.com/Optima-CityU/LLM4AD) | ~65 | `benchmarks/LLM4AD/benchmark_tasks/` |
| `veribench` | Python-to-Lean4 formal verification | ~140 | [HuggingFace dataset](https://huggingface.co/datasets/allenanie/veribench_with_prompts) or local entrypoint |
| `trace_examples` | Small example tasks shipped with Trace-Bench | 4 | `trace_bench/examples/` |
| `internal` | Synthetic tasks for testing and validation | 7 | `trace_bench/examples/` |

---

## LLM4AD Tasks (~65)

Tasks from the LLM4AD benchmark, covering algorithm design for optimization,
machine learning, and scientific discovery. Each task asks the optimizer to
produce a Python function (heuristic) that solves a specific problem.

**Categories:**

### Optimization -- Constructive Heuristics

These tasks ask the optimizer to produce a construction heuristic for
combinatorial optimization problems.

| Task ID | Problem | Entry Function |
|---------|---------|---------------|
| `llm4ad:optimization/tsp_construct` | Traveling Salesman Problem | `select_next_node` |
| `llm4ad:optimization/vrptw_construct` | Vehicle Routing (Time Windows) | `select_next_node` |
| `llm4ad:optimization/jssp_construct` | Job Shop Scheduling | `heuristic` |
| `llm4ad:optimization/knapsack_construct` | 0/1 Knapsack Problem | `heuristic` |
| `llm4ad:optimization/bp_1d_construct` | 1D Bin Packing | `heuristic` |
| `llm4ad:optimization/bp_2d_construct` | 2D Bin Packing | `heuristic` |
| `llm4ad:optimization/set_cover_construct` | Set Cover Problem | `select_next_subset` |
| `llm4ad:optimization/cvrp_construct` | Capacitated VRP | `select_next_node` |
| `llm4ad:optimization/ovrp_construct` | Open VRP | `select_next_node` |
| `llm4ad:optimization/qap_construct` | Quadratic Assignment | `heuristic` |
| `llm4ad:optimization/cflp_construct` | Capacitated Facility Location | `heuristic` |

### Optimization -- Improvement Heuristics

| Task ID | Problem |
|---------|---------|
| `llm4ad:optimization/tsp_gls_2O` | TSP via Guided Local Search |
| `llm4ad:optimization/online_bin_packing` | Online Bin Packing |
| `llm4ad:optimization/online_bin_packing_2O` | Online Bin Packing (2-opt) |
| `llm4ad:optimization/graph_colouring` | Graph Coloring |
| `llm4ad:optimization/set_covering` | Set Covering |
| `llm4ad:optimization/set_partitioning` | Set Partitioning |
| `llm4ad:optimization/maximal_independent_set` | Maximum Independent Set |
| `llm4ad:optimization/pymoo_moead` | Multi-objective (MOEA/D) |

### Optimization -- Scheduling & Routing

| Task ID | Problem |
|---------|---------|
| `llm4ad:optimization/flow_shop_scheduling` | Flow Shop Scheduling |
| `llm4ad:optimization/job_shop_scheduling` | Job Shop Scheduling |
| `llm4ad:optimization/hybrid_reentrant_shop_scheduling` | Hybrid Reentrant Shop |
| `llm4ad:optimization/open_shop_scheduling` | Open Shop Scheduling |
| `llm4ad:optimization/common_due_date_scheduling` | Common Due Date |
| `llm4ad:optimization/crew_scheduling` | Crew Scheduling |
| `llm4ad:optimization/vehicle_routing_period_routing` | Periodic VRP |
| `llm4ad:optimization/aircraft_landing` | Aircraft Landing |

### Optimization -- Packing & Location

| Task ID | Problem |
|---------|---------|
| `llm4ad:circle_packing` | Circle Packing |
| `llm4ad:optimization/packing_unequal_circles` | Unequal Circle Packing |
| `llm4ad:optimization/packing_unequal_circles_area` | Circle Packing (area) |
| `llm4ad:optimization/packing_unequal_rectangles_and_squares` | Rectangle Packing |
| `llm4ad:optimization/container_loading` | Container Loading |
| `llm4ad:optimization/container_loading_with_weight_restrictions` | Container Loading (weight) |
| `llm4ad:optimization/constrained_guillotine_cutting` | Guillotine Cutting |
| `llm4ad:optimization/unconstrained_guillotine_cutting` | Unconstrained Guillotine |
| `llm4ad:optimization/constrained_non_guillotine_cutting` | Non-Guillotine Cutting |
| `llm4ad:optimization/p_median_capacitated` | Capacitated P-Median |
| `llm4ad:optimization/p_median_uncapacitated` | Uncapacitated P-Median |
| `llm4ad:optimization/capacitated_warehouse_location` | Warehouse Location |
| `llm4ad:optimization/uncapacitated_warehouse_location` | Uncapacitated Warehouse |

### Optimization -- Other

| Task ID | Problem |
|---------|---------|
| `llm4ad:optimization/admissible_set` | Admissible Set |
| `llm4ad:optimization/assignment_problem` | Assignment Problem |
| `llm4ad:optimization/assortment_problem` | Assortment Problem |
| `llm4ad:optimization/corporate_structuring` | Corporate Structuring |
| `llm4ad:optimization/equitable_partitioning_problem` | Equitable Partitioning |
| `llm4ad:optimization/euclidean_steiner_problem` | Euclidean Steiner Tree |
| `llm4ad:optimization/multidimensional_knapsack_problem` | Multi-dim Knapsack |
| `llm4ad:optimization/multi_demand_multidimensional_knapsack_problem` | Multi-demand Knapsack |
| `llm4ad:optimization/resource_constrained_shortest_path` | Resource-Constrained SP |
| `llm4ad:optimization/travelling_salesman_problem` | Classic TSP |

### Machine Learning

| Task ID | Problem |
|---------|---------|
| `llm4ad:machine_learning/acrobot` | Acrobot control |
| `llm4ad:machine_learning/car_mountain` | Mountain Car |
| `llm4ad:machine_learning/car_mountain_continue` | Mountain Car (continuous) |
| `llm4ad:machine_learning/moon_lander` | Lunar Lander |
| `llm4ad:machine_learning/pendulum` | Inverted Pendulum |

### Scientific Discovery

| Task ID | Problem |
|---------|---------|
| `llm4ad:science_discovery/bactgrow` | Bacterial Growth |
| `llm4ad:science_discovery/feynman_srsd` | Feynman SRSD |
| `llm4ad:science_discovery/ode_1d` | 1D ODE |
| `llm4ad:science_discovery/oscillator1` | Oscillator 1 |
| `llm4ad:science_discovery/oscillator2` | Oscillator 2 |
| `llm4ad:science_discovery/stresstrain` | Stress-Strain |

---

## VeriBench Tasks (~140)

Tasks from the VeriBench benchmark: each presents a Python function and asks
the optimizer to produce a correct Lean 4 translation with proof.

Tasks are discovered dynamically from the HuggingFace dataset
[`allenanie/veribench_with_prompts`](https://huggingface.co/datasets/allenanie/veribench_with_prompts)
or from a locally installed VeriBench entrypoint.

Task ID format: `veribench:<task_name>`

```bash
trace-bench list-tasks --bench veribench
```

---

## Trace Examples (4 tasks)

Small, self-contained tasks shipped with Trace-Bench for quick testing and
documentation. These require OpenTrace to be installed.

| Task ID | Description | Agent Class |
|---------|-------------|-------------|
| `trace_examples:greeting_stub` | Exact-match greeting optimization | `GreetingAgent` |
| `trace_examples:opentrace_greeting` | Multilingual greeting (EN/ES) | `OpenTraceGreetingAgent` |
| `trace_examples:opentrace_train_single_node` | Single trainable node optimization | (single `ParameterNode`) |
| `trace_examples:train_single_node_stub` | Stub version of single-node training | (single `ParameterNode`) |

---

## Internal Tasks (7 tasks)

Synthetic tasks for testing the framework itself. Useful for CI, validation,
and debugging.

| Task ID | Description |
|---------|-------------|
| `internal:numeric_param` | Numeric parameter optimization |
| `internal:code_param` | Code string parameter optimization |
| `internal:multi_param` | Multiple parameters simultaneously |
| `internal:non_trainable` | Non-trainable agent (for testing validation) |
| `internal:multiobjective_convex` | Multi-objective convex optimization |
| `internal:multiobjective_bbeh` | Multi-objective BIG-Bench Hard |
| `internal:multiobjective_gsm8k` | Multi-objective GSM8K math |

---

## Related

- [Agents and Tasks](agents-and-tasks.md) -- understanding the agent/task distinction
- [Adding a Task](adding-task.md) -- how to create and register a new task
- [Adding a Benchmark](adding-benchmark.md) -- integrating an external benchmark suite
- [Running Experiments](running-experiments.md) -- CLI reference for running tasks
