# Trace-Bench
Benchmark to evaluate LLM as an optimizer.

Currently, we are adding problems/domains one folder at a time. 

The instructions to run each task are located inside the task folder.

## Problem Sets

### General Problem Sets
- Simple QA Problem
- A problem set that uses a ReAct agent
- A problem set that uses a tool-calling agent
- Code writing/generation
- Math proof generation
- A **reasoning** problem set that uses multi-agent (Learning to reason)

## Agent Architecture
- ReAct agent

### LLM4AD problems set
A comprehensive collection of **60 benchmark tasks** derived from the [LLM4AD (Large Language Models for Algorithm Design)](https://github.com/Optima-CityU/LLM4AD).
- **Optimization - Basic** (18 tasks): `circle_packing`, `online_bin_packing_local`, etc.
- **Optimization - Constructive** (15 tasks): `optimization_tsp_construct`, `optimization_knapsack_construct`, `optimization_set_cover_construct`, etc.
- **Optimization - CO-Bench** (21 tasks): `optimization_travelling_salesman_problem`, `optimization_job_shop_scheduling`, `optimization_container_loading`, etc.
- **Machine Learning** (5 tasks): `machine_learning_acrobot`, `machine_learning_pendulum`, `machine_learning_moon_lander`, etc.
- **Scientific Discovery** (1 task): `science_discovery_ode_1d`

**Supported Algorithms:** PrioritySearch, GEPA-Base, GEPA-UCB, GEPA-Beam

📖 **[See detailed usage guide →](LM4AD/readme.md)**

