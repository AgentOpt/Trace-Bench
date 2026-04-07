# Agent implementations for the hf: task suite.
#
# Each module in this package implements a specific framework's agents and
# must expose a ``make_agent(agent_class, **kwargs)`` factory function.
#
# Supported modules:
#   trace_agent.py  — OpenTrace (opto) agents
#   dspy_agent.py   — DSPy agents
