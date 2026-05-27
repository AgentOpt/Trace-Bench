# OpenEvolveTrainer

`OpenEvolveTrainer` is an external Trace-Bench trainer wrapper for `openevolve.run_evolution`.

- Evolves a **safe literal** candidate mapping of trainable parameter values.
- Never executes candidate code via `exec`.
- Parses candidates using `ast.parse` and `ast.literal_eval` only.
- Can optionally keep only improving updates (`ensure_improvement=True`).
