# Config & Outputs Reference

This page documents the full YAML configuration schema, matrix expansion
semantics, resume modes, and the output artifact tree.

## YAML Schema

A Trace-Bench config is a YAML (or JSON) file with the following top-level
fields. All fields are optional unless noted.

```yaml
# --- Run identity & paths ---
run_id: null              # Optional fixed run ID. Generated if omitted.
runs_dir: runs            # Root directory for output. Alias: runs_root.

# --- Execution ---
mode: stub                # "stub" (dry-run) or "real" (actual LLM calls)
seeds: [123]              # List of random seeds. Legacy: single `seed: 123`.
max_workers: 1            # Parallel job slots. Aliases: n_concurrent, n-concurrent.
fail_fast: false          # Abort entire run on first job failure.
resume: auto              # Resume mode: "auto" | "failed" | "none".
job_timeout: null         # Per-job timeout in seconds. Null = unspecified (CLI applies defaults).

# --- LLM provider ---
llm:
  provider: openrouter    # UI uses this block; runner also applies it (sets env).
  base_url: https://openrouter.ai/api/v1
  model: openrouter/openai/gpt-4o-mini
  api_key_env: OPENROUTER_API_KEY

# --- Global defaults ---
eval_kwargs: {}           # Merged into every task's eval_kwargs.
trainer_kwargs: {}        # Merged into every trainer's params_variants.
tags: []                  # Freeform labels for organizing runs.

# --- Tasks ---
tasks:
  - id: "llm4ad:circle_packing"          # Short form: just an ID string.
  - id: "llm4ad:optimization/tsp_gls_2O" # Long form with overrides:
    eval_kwargs:
      timeout: 30

# --- Trainers ---
trainers:
  - id: PrioritySearch             # Short form: ID string only.
  - id: GEPA-Base                  # Long form with full options:
    params_variants:
      - gepa_iters: 1
        gepa_train_bs: 2
      - gepa_iters: 5
        gepa_train_bs: 4
    optimizer: null                 # Optional optimizer class name.
    optimizer_kwargs: {}
    guide: null                     # Optional guide class name.
    guide_kwargs: {}
    logger: null                    # Optional logger class name.
    logger_kwargs: {}
```

### Field details

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `run_id` | string or null | auto-generated | Format: `YYYYMMDD-HHMMSS-<hash8>` (timestamp + hash) |
| `runs_dir` | string | `"runs"` | Accepts `runs_root` as alias |
| `mode` | string | `"stub"` | `stub` skips real LLM calls |
| `seeds` | list[int] | `[123]` | Each seed creates a separate job per task/trainer/variant |
| `max_workers` | int | `1` | `0` or `1` = sequential execution |
| `fail_fast` | bool | `false` | Stop on first failure |
| `resume` | string | `"auto"` | See Resume Modes below |
| `job_timeout` | float or null | `null` | Seconds; null = unspecified (CLI defaults to 0 stub / 600 real) |
| `tags` | list[string] | `[]` | Informational only |

### Run IDs and `config_hash`

`run_id` is generated as `YYYYMMDD-HHMMSS-<hash8>` using the current timestamp and a hash
of the config snapshot plus git SHA. Identical configs do **not** guarantee identical
`run_id` values. To dedupe or compare runs, use `config_hash` in `meta/manifest.json`.

If you want a stable identifier, set `run_id` explicitly in the config.

### Task config

Each task entry requires an `id` (or `key`/`task` as aliases). An optional
`eval_kwargs` dict is merged on top of the global `eval_kwargs`.

### Trainer config

Each trainer entry requires an `id` (or `name`/`trainer`/`key`). Additional
fields:

| Field | Purpose |
|-------|---------|
| `params_variants` | List of param dicts. Each variant produces separate jobs. Falls back to a single `params`/`trainer_kwargs` dict. |
| `optimizer` | Class name of an optimizer to use |
| `optimizer_kwargs` | Kwargs passed to the optimizer |
| `guide` | Class name of a guide to use |
| `guide_kwargs` | Kwargs passed to the guide |
| `logger` | Logger class name (e.g. `TensorboardLogger`) |
| `logger_kwargs` | Kwargs passed to the logger |

Global `trainer_kwargs` and LLM4AD-specific knobs (`ps_steps`, `ps_batches`,
`gepa_iters`, etc.) are merged into every variant as defaults.

### LLM Block Behavior

The `llm` block is consumed by the runner (it sets `OPENAI_API_KEY`, `OPENAI_API_BASE`,
`TRACE_LITELLM_MODEL`, etc.) **and** by the UI (for defaults and form prefill).
If you omit this block, the runner relies on environment variables only.

### Job Timeout Defaults

- `job_timeout: null` means "unspecified".
- CLI defaults are mode-dependent when unspecified:
  - stub mode: 0 (no timeout)
  - real mode: 600 seconds
- Programmatic usage (`BenchRunner(...)`) uses `None` unless you pass a timeout.

---

## Matrix Expansion

The job matrix is the Cartesian product:

```
jobs = tasks x trainers x params_variants x seeds
```

For example, with 3 tasks, 2 trainers (each with 2 param variants), and
2 seeds:

```
3 x 2 x 2 x 2 = 24 jobs
```

Each job gets a deterministic `job_id` computed as a SHA-256 hash of
`(task_id, trainer_id, resolved_kwargs, seed)`. This means re-running the
same config produces the same job IDs, enabling resume.

See `trace_bench/matrix.py` for the `expand_matrix()` implementation.

---

## Resume Modes

| Mode | Behavior |
|------|----------|
| `auto` | Skip jobs whose `job_meta.json` already has `status: ok`. Re-run everything else. |
| `failed` | Only re-run jobs with `status: failed`. Skip `ok` and `skipped`. |
| `none` | Run all jobs unconditionally, overwriting previous results. |

Resume works by matching `job_id` values. Since job IDs are deterministic,
the runner can detect which jobs from a prior run correspond to the current
matrix.

---

## Example Configs

**Minimal smoke test** (`configs/smoke.yaml`):

```yaml
runs_dir: runs
mode: stub
seeds: [123]

tasks:
  - id: internal:numeric_param

trainers:
  - id: PrioritySearch
    params_variants:
      - ps_steps: 1
        ps_batches: 1
```

**Multi-trainer coverage** (`configs/m2_coverage.yaml`):

```yaml
mode: stub
seeds: [123]
max_workers: 2
resume: auto

trainers:
  - id: PrioritySearch
    params_variants:
      - ps_steps: 1
        ps_batches: 1
        ps_candidates: 2
        ps_proposals: 2
  - id: GEPA-Base
    params_variants:
      - gepa_iters: 1
        gepa_train_bs: 2
        gepa_merge_every: 2
        gepa_pareto_subset: 2

tasks:
  - id: "llm4ad:circle_packing"
  - id: "llm4ad:optimization/tsp_gls_2O"
  # ... 29 tasks total
```

---

## Output Artifact Tree

Every run produces the following directory structure:

```
runs/<run_id>/
  meta/
    config.snapshot.yaml   # Frozen copy of the resolved config
    manifest.json          # Run manifest (generated_at, job list, git info)
    env.json               # Captured environment variables (redacted)
    git.json               # Git commit and branch at run time
    files_index.json       # Index of all output files
  results.csv              # One row per job with scores, timing, tokens
  summary.json             # Aggregate counts (ok/failed/skipped) and tokens
  leaderboard.csv          # Best score per task, ranked
  jobs/
    <job_id>/
      job_meta.json        # Task, trainer, seed, resolved kwargs, status
      results.json         # Detailed result payload
      events.jsonl         # Structured event log (one JSON object per line)
      stdout.log           # Captured standard output
      artifacts/
        initial_state.json # Starting state (also .yaml)
        best_state.json    # Best state found (also .yaml)
        final_state.json   # Final state at end of optimization (also .yaml)
        state_history.jsonl # Full state evolution over time
      tb/                  # TensorBoard log files for this job
```

---

## Related

- [UI Guide](ui-guide.md) -- launching runs from the Gradio interface
- [MLflow Integration](mlflow.md) -- what gets mirrored to MLflow
- [Result Analysis](result-analysis.md) -- how to work with the output files
