# MLflow Integration

Trace-Bench can mirror run metadata and job metrics to an MLflow tracking
server. The filesystem remains the canonical source of truth; MLflow provides
a searchable, visual layer on top.

## Enabling MLflow

Two conditions must be met:

1. The `mlflow` Python package is installed.
2. The `MLFLOW_TRACKING_URI` environment variable is set.

```bash
pip install mlflow
export MLFLOW_TRACKING_URI=http://localhost:5000
# or a remote URI like https://mlflow.example.com
```

If either condition is missing, all MLflow functions silently no-op.
No configuration flags or YAML keys are needed -- presence of the environment
variable is the only switch.

---

## What is logged

### Run-level (`log_run_start`)

When a benchmark run begins, Trace-Bench creates an MLflow run and logs:

| Category | Detail |
|----------|--------|
| **Run name** | The Trace-Bench `run_id` (timestamp + hash) |
| **Tags** | `trace_bench.run_id`, `trace_bench.run_dir` |
| **Params** | Shallow scalar fields from the config snapshot, prefixed `cfg.*` (e.g. `cfg.mode`, `cfg.resume`, `cfg.max_workers`) |
| **Artifacts** | `meta/config.snapshot.yaml`, `meta/manifest.json`, `results.csv`, `summary.json`, `meta/env.json`, `meta/git.json` -- uploaded under the `trace_bench` artifact path |

Only scalar config values (string, int, float, bool, None) are logged as
params. Nested structures like `tasks` and `trainers` are captured in the
artifact files instead.

### Job-level (`log_job_result`)

Each completed job is logged as a **nested MLflow run** under the parent:

| Category | Detail |
|----------|--------|
| **Nested run name** | `job:<job_id>` |
| **Tags** | `trace_bench.parent_run_id`, `trace_bench.job_id`, `trace_bench.task_id`, `trace_bench.trainer_id`, `trace_bench.suite` |
| **Metrics** | `score_initial`, `score_final`, `score_best`, `time_seconds` (numeric values only) |
| **Params** | `status`, `resolved_trainer_kwargs` (JSON string), `resolved_optimizer_kwargs` (JSON string) |

Nested runs ensure that repeated keys (e.g. multiple jobs reporting
`score_best`) do not collide on the parent run.

### Run-level summary (`log_run_end`)

When the run finishes, aggregate counts are logged as metrics on the parent
run:

- `run.count_ok`
- `run.count_failed`
- `run.count_skipped`

The MLflow run is then ended.

---

## What is NOT logged

- **Job-level artifacts** (events.jsonl, state snapshots, stdout.log) are not
  uploaded to MLflow. They remain only on the local filesystem.
- **Sensitive values** -- API keys and tokens are redacted before any data
  leaves the process (see `artifacts.py` sanitization).
- **Guide kwargs and logger kwargs** -- only trainer and optimizer kwargs are
  logged as params on nested runs.

---

## What remains canonical on the filesystem

All data lives under `runs/<run_id>/`:

```
runs/<run_id>/
  meta/config.snapshot.yaml
  meta/manifest.json
  meta/env.json
  meta/git.json
  results.csv
  summary.json
  leaderboard.csv
  jobs/<job_id>/
    job_meta.json
    results.json
    events.jsonl
    stdout.log
    artifacts/
    tb/
```

MLflow mirrors a subset of this tree. If the MLflow server is unavailable or
loses data, the filesystem copy is authoritative.

---

## No-op behavior

When MLflow is not configured (`MLFLOW_TRACKING_URI` unset or `mlflow` not
installed):

- `log_run_start` returns `None`.
- `log_job_result` and `log_run_end` check for a `None` context and return
  immediately.
- No imports of `mlflow` are attempted at module load time; the check is
  deferred to first use.
- Benchmark execution is never affected by MLflow availability.

All MLflow calls are wrapped in try/except blocks so that a tracking server
error cannot break a benchmark run.

---

## Viewing results

```bash
# Start the MLflow UI
mlflow ui --backend-store-uri $MLFLOW_TRACKING_URI

# Or use the MLflow CLI to list runs
mlflow runs list --experiment-id 0
```

In the MLflow UI, parent runs correspond to Trace-Bench runs and nested runs
correspond to individual jobs. Use the search bar to filter by tags such as
`trace_bench.trainer_id = "PrioritySearch"`.

---

## Related

- [Config Reference](config-reference.md) -- YAML schema for run configuration
- [Result Analysis](result-analysis.md) -- filesystem-based analysis workflows
- [UI Guide](ui-guide.md) -- the Gradio UI shows MLflow status at the bottom
