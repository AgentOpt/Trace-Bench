# Result Analysis

After a Trace-Bench run completes, results are available in three forms:
filesystem artifacts (CSV/JSON), the Gradio UI, and programmatic access via
pandas. This page covers all three.

---

## CLI / Filesystem

Every run writes its outputs to `runs/<run_id>/`. The key files for analysis:

| File | Format | Contents |
|------|--------|----------|
| `results.csv` | CSV | One row per job: scores, timing, token usage, trainer kwargs |
| `summary.json` | JSON | Aggregate counts (`ok`, `failed`, `skipped`) and total tokens |
| `leaderboard.csv` | CSV | Best score per task (ranked), with the winning job and trainer |

### Quick inspection

```bash
# View the summary
cat runs/<run_id>/summary.json | python -m json.tool

# Check status counts
python -c "
import json, pathlib
s = json.loads(pathlib.Path('runs/<run_id>/summary.json').read_text())
print(s['counts'])
print(f\"Total tokens: {s['token_totals']['total_tokens']}\")
"

# Preview the leaderboard
column -t -s, runs/<run_id>/leaderboard.csv | head -20
```

### Results CSV columns

The full column list (from `trace_bench/results.py`):

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | string | Parent run identifier |
| `job_id` | string | Deterministic job hash |
| `task_id` | string | Task identifier (e.g. `llm4ad:circle_packing`) |
| `suite` | string | Benchmark suite (e.g. `llm4ad`, `veribench`) |
| `trainer_id` | string | Trainer name (e.g. `PrioritySearch`) |
| `seed` | int | Random seed used |
| `status` | string | `ok`, `failed`, or `skipped` |
| `score_initial` | float | Score before optimization |
| `score_final` | float | Score at end of optimization |
| `score_best` | float | Best score seen during optimization |
| `time_seconds` | float | Wall-clock time for the job |
| `prompt_tokens` | int | LLM prompt tokens consumed |
| `completion_tokens` | int | LLM completion tokens consumed |
| `total_tokens` | int | Total LLM tokens consumed |
| `feedback` | string | Trainer feedback message (if any) |

Additional columns include `resolved_trainer_kwargs`, `resolved_optimizer_kwargs`,
`eval_kwargs` (JSON strings), state file paths, and LLM provider details.

---

## Gradio UI

The **Browse Runs** tab provides interactive filtering without writing code.

### Workflow

1. Launch the UI: `trace-bench ui --runs-dir runs`
2. Go to the **Browse Runs** tab.
3. Select a run from the dropdown.
4. Use the filter dropdowns to narrow by **suite**, **status**, **trainer_id**,
   or **task_id**. The results table and leaderboard update instantly.
5. Open the **Config & metadata** accordion to see the frozen config, environment,
   and manifest.

### Job Inspector

For deeper investigation of a single job:

1. Switch to the **Job Inspector** tab.
2. Select a run, then click **Load job list**.
3. Pick a job from the dropdown.
4. Review `job_meta.json`, `results.json`, `events.jsonl`, `stdout.log`, and
   state snapshots (initial, best, final).

See [UI Guide](ui-guide.md) for full tab descriptions and screenshots.

---

## Programmatic Analysis

### Loading results with pandas

```python
import pandas as pd

df = pd.read_csv("runs/<run_id>/results.csv")

# Filter to successful jobs
ok = df[df["status"] == "ok"]

# Mean score by trainer
print(ok.groupby("trainer_id")["score_best"].mean())

# Mean score by trainer and suite
print(ok.groupby(["suite", "trainer_id"])["score_best"].mean().unstack())
```

### Comparing trainers across tasks

```python
import pandas as pd

df = pd.read_csv("runs/<run_id>/results.csv")
ok = df[df["status"] == "ok"]

# Pivot: tasks as rows, trainers as columns, score_best as values
pivot = ok.pivot_table(
    index="task_id",
    columns="trainer_id",
    values="score_best",
    aggfunc="max",  # best across seeds
)
print(pivot)

# Win rate: how often each trainer has the highest score
wins = pivot.idxmax(axis=1).value_counts()
print("\nWin counts per trainer:")
print(wins)
```

### Comparing across seeds

```python
import pandas as pd

df = pd.read_csv("runs/<run_id>/results.csv")
ok = df[df["status"] == "ok"]

# Score variance across seeds for each task+trainer
variance = ok.groupby(["task_id", "trainer_id"])["score_best"].agg(["mean", "std", "count"])
print(variance.sort_values("std", ascending=False).head(20))
```

### Aggregating multiple runs

```python
import pandas as pd
from pathlib import Path

runs_dir = Path("runs")
frames = []
for run_path in runs_dir.iterdir():
    csv_path = run_path / "results.csv"
    if csv_path.exists():
        frames.append(pd.read_csv(csv_path))

all_results = pd.concat(frames, ignore_index=True)
print(f"Total jobs across all runs: {len(all_results)}")
print(all_results.groupby("trainer_id")["score_best"].describe())
```

### Token usage analysis

```python
import pandas as pd

df = pd.read_csv("runs/<run_id>/results.csv")
ok = df[df["status"] == "ok"]

# Tokens per trainer
token_summary = ok.groupby("trainer_id")[["prompt_tokens", "completion_tokens", "total_tokens"]].sum()
print(token_summary)

# Score-per-token efficiency
ok["efficiency"] = ok["score_best"] / ok["total_tokens"].clip(lower=1)
print(ok.groupby("trainer_id")["efficiency"].mean())
```

---

## Benchmark Analysis Patterns

### Multi-objective comparison

Several notebooks demonstrate multi-objective analysis across different
benchmark suites:

- `notebooks/06_multiobjective_convex.ipynb` -- convex optimization tasks
- `notebooks/07_multiobjective_bbeh.ipynb` -- BIG-Bench Hard tasks
- `notebooks/08_multiobjective_gsm8k.ipynb` -- GSM8K math reasoning tasks

These notebooks load `results.csv`, group by trainer, and plot Pareto
frontiers of score vs. cost (tokens or time).

### Leaderboard interpretation

The `leaderboard.csv` file picks the best-scoring job per task (among jobs
with `status: ok`) and ranks tasks alphabetically. Each row contains:

| Column | Description |
|--------|-------------|
| `rank` | Position (1-indexed, alphabetical by task_id) |
| `task_id` | Task identifier |
| `suite` | Benchmark suite |
| `job_id` | Winning job hash |
| `trainer_id` | Winning trainer |
| `score_best` | Highest score achieved |
| `time_seconds` | Time taken by the winning job |

To build a custom leaderboard (e.g. best by trainer rather than by task),
use the pivot table approach shown above.

---

## Related

- [Config Reference](config-reference.md) -- output artifact tree and field definitions
- [UI Guide](ui-guide.md) -- interactive browsing and filtering
- [MLflow Integration](mlflow.md) -- querying results via MLflow
