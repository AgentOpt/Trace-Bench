# Running Experiments

## CLI Command Reference

Trace-Bench provides five commands via the `trace-bench` entry point.

### list-tasks

List all discoverable tasks.

```bash
trace-bench list-tasks --root benchmarks/LLM4AD/benchmark_tasks
trace-bench list-tasks --root benchmarks/LLM4AD/benchmark_tasks --bench llm4ad
trace-bench list-tasks --root benchmarks/LLM4AD/benchmark_tasks --bench veribench
trace-bench list-tasks --root benchmarks/LLM4AD/benchmark_tasks --bench trace_examples
```

The `--bench` flag filters by suite. Accepted values: `llm4ad`, `trace_examples`, `internal`, `veribench`. Combine with commas: `--bench llm4ad,internal`.

### list-trainers

List available optimization algorithms.

```bash
trace-bench list-trainers
trace-bench list-trainers --all   # include unavailable trainers
```

Output is tab-separated: `trainer_id\tavailable|unavailable`.

### validate

Dry-run a config: checks that all trainers exist and all tasks can build their bundles.

```bash
trace-bench validate --config configs/smoke.yaml --root benchmarks/LLM4AD/benchmark_tasks
trace-bench validate --config configs/smoke.yaml --root benchmarks/LLM4AD/benchmark_tasks --strict
```

With `--strict`, validation also checks that every task has trainable parameters, verifies trainer kwargs against the allow-list, expands the full job matrix, and writes a manifest to `--runs-dir`.

### run

Execute a benchmark configuration.

```bash
trace-bench run --config configs/smoke.yaml --root benchmarks/LLM4AD/benchmark_tasks
trace-bench run --config configs/m2_coverage.yaml --root benchmarks/LLM4AD/benchmark_tasks \
  --runs-dir results/m2 --max-workers 4
trace-bench run --config configs/smoke_real.yaml --root benchmarks/LLM4AD/benchmark_tasks \
  --job-timeout 300 --logger ConsoleLogger
```

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--runs-dir` | `runs` | Output directory for run artifacts |
| `--max-workers` | `1` | Parallel job threads |
| `--force` | off | Re-run all jobs even if results exist |
| `--resume` | `auto` | Resume mode: `auto`, `failed`, `none` |
| `--job-timeout` | mode-dependent | Per-job timeout in seconds (0 = stub, 600 = real) |
| `--logger` | config | Override logger for all trainers (`ConsoleLogger`, `none`, etc.) |

### ui

Launch the Gradio results dashboard.

```bash
trace-bench ui --runs-dir runs
trace-bench ui --runs-dir runs --share --port 7860
```

## Making Fair Comparisons

A benchmark run is a matrix: tasks x trainers x params_variants x seeds. To isolate one variable, change only that axis.

| Comparison | What to change | What to hold fixed |
|------------|----------------|-------------------|
| LLM provider/model | `llm.model` / `llm.base_url` in config | Same tasks, trainers, seeds |
| Trainer algorithm | `trainers[].id` | Same tasks, LLM, seeds |
| Optimizer params | `trainers[].params_variants` | Same trainer, tasks, LLM, seeds |
| Task set | `tasks[]` | Same trainers, LLM, seeds |
| Seed | `seeds[]` | Everything else identical |

The `run_id` is a deterministic hash of the config snapshot. Identical configs produce identical `run_id` values, making it easy to detect duplicate runs.

## Run Artifacts

After `trace-bench run` completes, the run directory contains:

```
runs/<run_id>/
  meta/
    config.yaml          # resolved config snapshot
    manifest.json        # full job matrix with resolved kwargs
    env.json             # Python version, platform, packages
    git.json             # repo commit info (if in a git repo)
    files_index.json     # index of all produced files
  jobs/<job_id>/
    meta.json            # job metadata (task, trainer, seed, params)
    results.json         # score, feedback, timing, token usage
    artifacts/
      initial_state.yaml # model state before training
      best_state.yaml    # best model state observed
      final_state.yaml   # model state after training
      state_history.jsonl # step-by-step state changes
      events.jsonl       # timestamped event log
  results.csv            # one row per job, all scores
  summary.json           # aggregate stats (mean, std, pass rate)
  leaderboard.csv        # trainers ranked by mean score
```

### Reading Results

- **run_id**: deterministic hash from config. Find it in `summary.json` or the directory name.
- **job_id**: unique per (task, trainer, params, seed). Found in `manifest.json` and `results.csv`.
- **results.csv**: the primary results table. One row per job with columns for task, trainer, seed, score, duration, tokens.
- **summary.json**: aggregate statistics grouped by trainer.
- **leaderboard.csv**: trainers ranked by mean score across all tasks/seeds.
- **manifest.json**: the full expanded matrix with resolved kwargs -- useful for debugging which parameters were actually used.

## Notebooks

- `notebooks/03_task_coverage.ipynb` -- explore which tasks are available and their properties
- `notebooks/05_full_benchmark.ipynb` -- running and analyzing a full benchmark matrix
