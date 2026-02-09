# M0 — Trace‑Bench Technical Plan (Two Approaches + Explicit Acceptance)

## Milestone definitions (fixed)

- **M0**: Technical plan + locked contracts (**this document**).
- **M1**: **Full Trace‑Bench API implementation** + **minimal runnable coverage** for each bench type, trainer type, and parameter/optimization target type.
- **M2**: **Full coverage** across benches/trainers/tasks/parameters with **efficient** matrix execution + aggregation (meet coverage targets).
- **M3**: **UI + MLflow + TensorBoard + Gradio** (operational wiring, not placeholders).
- **M4**: **GEPA + Curriculum** trainers integration.

---

## Executive Summary

This M0 provides two implementation approaches (A/B) and locks the core contracts needed to de-risk M1–M4:

- Run/job identity (no overwrites; deterministic job IDs)
- Matrix semantics (tasks × trainers × parameter variants × seeds)
- Canonical artifacts schema (filesystem is source of truth)
- Task & trainer discovery contracts (stable IDs + validation)
- Parameter pass‑through contracts (compatible with Trace/OpenTrace Trainer/Optimizer/Guide/Logger APIs)
- Security rules for environment capture (allowlist + redaction)
- All validation logic/config/notebooks are delivered and reviewed via **PRs to Trace-Bench repo** (no out-of-band validation).

Trace‑Bench **does not implement trainers**: it **consumes** trainer algorithms from Trace/OpenTrace and focuses on orchestration, validation, reproducibility, and reporting.

---

## 0) Purpose of M0

M0 is **not** an implementation milestone. It is a **contract-locking** milestone:

- Define the plan variants, acceptance targets, and validation approach
- Lock the run/job/matrix/artifacts/discovery/security contracts so M1 can implement with minimal ambiguity
- Ensure the plan is aligned with the current Trace‑Bench + Trace code realities (trainer.train API, task loaders, LLM4AD runner knobs)

---

## 1) Two Plan Variants (Pick A or B)

Both variants respect the fixed milestone definitions above. The difference is how much breadth is proven **early in M1** vs deferred to M2.

### Plan A — Lean / staged (recommended: Plan A+ Pareto from Plan B)
Plan A+ keeps the smaller M1 surface, but **adds a bounded compatibility harness** (low cost / high gain).

- **M1**: Implement full API + prove minimal runnable coverage **and** a bounded matrix smoke:
  - 1 internal example task bundle
  - 1 LLM4AD task bundle
  - 1 VeriBench task bundle **if** entrypoint is available (otherwise “skipped with reason” is valid)
  - OpenTrace examples smoke (import/`--help`) wired in CI
  - Run each supported trainer at least once with at least one non-default parameter
  - **Minimal matrix smoke (bounded):** 2 tasks × 2 trainers × 1 seed (4 jobs) end-to-end
  - **Edge hardening:** `validate --strict` fails fast on unknown kwargs, missing trainable params, and task build errors
- **M2**: Expand to full coverage targets + efficiency improvements (parallelism, resume, aggregation)
- **M3**: UI + MLflow/TB + Gradio
- **M4**: GEPA + Curriculum

### Plan B — Compatibility‑first (more breadth earlier, higher integration risk)
- **M1**: Everything in Plan A+, plus higher-risk breadth:
  - broader early discovery/coverage (especially VeriBench task inventory)
  - larger matrices (more tasks/trainers/seeds) earlier
- **M2–M4**: same as Plan A

**Trade-off**: Plan A+ minimizes early integration complexity while still catching most incompatibilities early (via strict validate + 2×2 smoke). Plan B buys earlier breadth at the cost of significantly higher M1 integration risk.

---

## 2) Scope and Coverage Targets (Explicit)

### LLM4AD coverage (M2 acceptance target)
- Target: **≥80% functional in Real mode** over the *current discovered LLM4AD task inventory*.
- “Functional” means:
  - job runs to completion (or hits configured timeout) **without crashing the runner**
  - at least one training/optimization step is executed
  - canonical artifacts are written (`results.json`, `events.jsonl`, `results.csv`, `summary.json`)
- “Optimizing” means (M2):
  - on a defined subset of tasks, **best_score > initial_score + ε** (ε default: 1e‑9) OR non‑finite → finite transition.

Proposed “optimizing” subset (10 tasks; tunable):
- `circle_packing`
- `optimization_knapsack_construct`
- `optimization_bp_1d_construct`
- `optimization_tsp_construct`
- `optimization_cvrp_construct`
- `optimization_jssp_construct`
- `optimization_qap_construct`
- `optimization_set_cover_construct`
- `optimization_vrptw_construct`
- `optimization_cflp_construct`

### VeriBench coverage (M2 acceptance target)
- Target: **≥80% functional in Real mode** over the *current discovered VeriBench task inventory*.
- Dependency: Trace team provides the canonical entrypoint/task index (kept in scope).
- If entrypoint is not available in a given environment, jobs must be marked `skipped` with a structured reason (not a crash).

### OpenTrace examples (CI enforced from M1; acceptance target in M2)
- Target (SMART): **No unexpected failures in CI / 100% smoke tests pass**.
- Rule: every example must be either:
  - **PASS**: imports (and `--help` works if applicable) under `TRACE_BENCH_SMOKE=1`, or
  - **EXCEPTIONNAL SKIP (explicit)**: listed in a small `smoke_skip_allowlist.yaml` with a clear reason (missing optional dependency/dataset/credential).
- This prevents “silent skips” while keeping CI lightweight.

- CI requirement:
  - import every script in `OpenTrace/examples/` in a bounded subprocess
  - for argparse scripts: run `python <script> --help`
  - set `TRACE_BENCH_SMOKE=1` for smoke runs; examples should early-exit if they do heavy work at import time

---

## 3) Repository Grounding (Current Reality)

### LLM4AD tasks and runner knobs
- LLM4AD tasks are discoverable under `LLM4AD/benchmark_tasks/` and/or `LLM4AD/benchmark_tasks/index.json` (when present).
- The existing benchmark runner supports these knobs (must be representable in Trace‑Bench config and passed through):
  - `threads`
  - `optimizer_kwargs` (JSON dict merge)
  - `eval_kwargs` (JSON dict passed to evaluator)
  - `ps-*` knobs: `ps-steps`, `ps-batches`, `ps-candidates`, `ps-proposals`, `ps-mem-update`
  - `gepa-*` knobs: `gepa-iters`, `gepa-train-bs`, `gepa-merge-every`, `gepa-pareto-subset`

### Trace trainer entrypoint contract
Trace’s high-level entrypoint is (conceptually):
- `trainer.train(model=..., algorithm=..., optimizer=..., guide=..., logger=..., optimizer_kwargs=..., guide_kwargs=..., logger_kwargs=..., **trainer_kwargs)`
- `model` can be a `ParameterNode` (wrapped into a single-node module) or a Trace `Module`.

**Implication for Trace‑Bench**: config must support:
- algorithm selection (trainer)
- optimizer/guide/logger selection
- optimizer/guide/logger kwargs
- arbitrary trainer kwargs pass-through (with schema validation when available)

---

## 4) Contracts Locked in M0

### 4.1 Run vs Job identity (no overwrites)
- **Run** = one invocation of `trace-bench run --config ...`
- **Job** = one concrete combination: `(task_id, trainer_id, params_variant, seed)`

ID scheme:
- `run_id`: `YYYYMMDD-HHMMSS-<short_hash>` where the hash is derived from:
  - normalized config snapshot + git SHA
- `job_id`: deterministic hash of:
  - `task_id + trainer_id + resolved_kwargs + seed`

Guarantees:
- no overwrite of past results
- stable aggregation and comparisons
- reproducible job naming

### 4.2 Matrix semantics
Matrix expansion is the cartesian product:

`jobs = tasks × trainers × params_variants × seeds`

Rules:
- each job is independent
- failures do not stop the run unless `fail_fast: true`
- aggregation happens at run-level (`results.csv`, `summary.json`)

### 4.3 Canonical artifacts (filesystem is source of truth)
Filesystem under `runs/<run_id>/` is canonical. MLflow/TB mirror it (never replace it).

Canonical layout:

```
runs/<run_id>/
  meta/
    config.snapshot.yaml
    env.json
    git.json
    manifest.json
  jobs/
    <job_id>/
      job_meta.json
      results.json
      events.jsonl
      artifacts/
      tb/
  results.csv
  summary.json
```

Minimum `results.csv` columns:
- `run_id`, `job_id`, `task_id`, `suite`, `trainer_id`, `seed`
- `status` (`ok` | `failed` | `skipped`)
- `score_initial`, `score_final`, `score_best` (when available)
- `time_seconds`
- `resolved_trainer_kwargs` (JSON), `resolved_optimizer_kwargs` (JSON), `eval_kwargs` (JSON)
- `feedback` (string; error or evaluator feedback summary)
- `tb_logdir` (relative path)

### 4.4 env.json security policy
- allowlist-only environment capture (no full dumps)
- redact any key containing `KEY`, `TOKEN`, `SECRET`, `PASSWORD` (case-insensitive)
- record key runtime facts (python version, platform, package versions, git sha, selected config)

### 4.5 Discovery contracts
**TaskSpec**
- stable `task_id` with namespace prefix (e.g., `llm4ad:circle_packing`, `veribench:<name>`, `example:<name>`, `internal:<name>`)
- suite label (`llm4ad`, `veribench`, `trace_examples`, `internal`)
- loader/factory reference
- can build a runnable job bundle

**TrainerSpec**
- stable `trainer_id`
- resolution source (Trace core, Trace features, OpenTrace)
- parameter schema (best-effort; required when available)

---

## 5) Architecture (Minimal Surface, Contract‑driven)

**Target modules** (M1 implementation target):
- `trace_bench/config.py` — RunConfig parsing + validation (YAML/JSON)
- `trace_bench/registry.py` — task & trainer discovery + stable IDs
- `trace_bench/matrix.py` — matrix expansion + `run_id`/`job_id` + manifest
- `trace_bench/runner.py` — orchestration (stub/real, per-job isolation, timeouts)
- `trace_bench/artifacts.py` — canonical artifact writers (env redaction, job_meta, events.jsonl, results.json)
- `trace_bench/results.py` — run-level aggregation into `results.csv` + `summary.json`
- `trace_bench/logging.py` — MLflow/TB adapters mirroring the filesystem (M3 full wiring)
- `trace_bench/cli.py` — `trace-bench` entrypoint + subcommands
- `trace_bench/ui/` — Gradio UI package (M3)

**Public API**
```python
from trace_bench.config import load_config
from trace_bench.runner import BenchRunner

cfg = load_config("configs/m1_validation.yaml")
summary = BenchRunner(cfg).run()
```

---

## 6) CLI and Config Schema (including LLM4AD knob coverage)

### CLI
```
trace-bench list-tasks [--bench llm4ad|veribench|trace_examples|internal] [--pattern <glob>]
trace-bench list-trainers
trace-bench validate --config <path> [--strict]
trace-bench run --config <path> --runs-dir <abs_path>
trace-bench ui --runs-dir <abs_path>
```

### RunConfig YAML (conceptual)

```yaml
runs_dir: /abs/path/runs
mode: real                # real | stub
seeds: [123]
max_workers: 1            # job-level parallelism (matrix execution)
fail_fast: false

tasks:
  - id: llm4ad:circle_packing
    eval_kwargs:
      timeout_seconds: 10
  - id: internal:toy_numeric_param
  - id: internal:toy_code_param

trainers:
  - id: PrioritySearch
    params_variants:
      - threads: 2
        ps_steps: 2
        ps_batches: 2
        ps_candidates: 3
        ps_proposals: 3
        ps_mem_update: 2
    optimizer: OPROv2
    optimizer_kwargs: {}
    guide: LLMJudge
    guide_kwargs: {}
    logger: TensorboardLogger
    logger_kwargs: {}

  - id: GEPA-Base
    params_variants:
      - threads: 2
        gepa_iters: 2
        gepa_train_bs: 2
        gepa_merge_every: 2
        gepa_pareto_subset: 3
    optimizer: OPROv2
    optimizer_kwargs: {}

eval_kwargs:
  timeout_seconds: 10
```

**Parameter coverage rule (M1 contract)**:
- Trace‑Bench must accept **all current LLM4AD runner knobs** (threads, optimizer_kwargs, eval_kwargs, ps_*, gepa_*) as pass-through trainer kwargs, and record the resolved kwargs in:
  - `meta/manifest.json`
  - `jobs/<job_id>/job_meta.json`
  - `results.csv` (`resolved_trainer_kwargs`, etc.)

**Alias / compatibility mapping (best-effort, validated in M1 notebook)**:
- For LLM4AD-style configs, the runner must support reading the conventional keys:
  - `threads`, `optimizer_kwargs`, `eval_kwargs`
  - `ps_steps`, `ps_batches`, `ps_candidates`, `ps_proposals`, `ps_mem_update`
  - `gepa_iters`, `gepa_train_bs`, `gepa_merge_every`, `gepa_pareto_subset`
…and forward them to the selected trainer algorithm. If a trainer rejects a kwarg, validation must fail with a clear message naming the offending kwarg.

---

## 7) Compatibility Notes (Trace / Trainers / Optimizers / Guides)

The implementation must explicitly handle these realities:

1. **Trainer resolution across modules**
   - Some trainers live in `opto.trainer.algorithms` (e.g., `BasicSearchAlgorithm`, `UCBSearchAlgorithm`, `BeamsearchAlgorithm`)
   - Some live in `opto.features.*` (e.g., PrioritySearch)
   - Trace‑Bench registry must resolve trainer IDs from multiple sources (and record where they came from).

2. **ParameterNode vs Module differences**
   - Trace’s `trainer.train` wraps a `ParameterNode` into a single-node model and uses a default optimizer (`OPROv2` for ParameterNode, `OptoPrimeV2` for Module).
   - Trace‑Bench validation must ensure “trainable parameters exist”; otherwise jobs are marked `failed` with a precise reason.

3. **Timeout behavior**
   - Per-evaluation timeouts belong in `eval_kwargs.timeout_seconds`.
   - Runner must also enforce a job-level hard timeout (process/thread) to avoid hangs.

4. **Logging**
   - Trace loggers are pluggable; the runner must set TensorBoard logdir per job (`jobs/<job_id>/tb/`).

---

## 8) Tests / CI (Milestone placement)

### M1 (minimum)
- Unit:
  - config parsing + validation
  - matrix expansion
  - artifact writer schema (meta + per-job)
  - LLM4AD knob parsing coverage
- Integration:
  - run minimal internal examples (multiple optimization target types)
  - run 1 LLM4AD task (bounded eval_kwargs)
  - VeriBench: discovery + run 1 task if available, else `skipped` with reason
- CI:
  - OpenTrace examples smoke (import/--help) bounded by timeout
  - **PR CI policy:** keep PR CI **offline/stub** by default (no paid LLM calls).
  - **Real-LLM sanity check (separate workflow):**
    - Add a workflow (nightly + manual trigger) that runs **one tiny real job**:
      - 1 task × 1 trainer × 1 seed, minimal steps/iters, short timeout
    - It runs **only when secrets exist** (e.g., OpenRouter key via env vars).
    - This is the “real world” check without making PRs expensive.

### M2 (expanded)
- matrix performance features (bounded parallelism, skip-existing/resume semantics)
- broad suite coverage to meet ≥80% functional targets
- for VeriBench, it will be important to use the latest version from https://github.com/xuanfeiren/Trace-Bench (if not yet merged to Trace-Bench)

---

## 9) Notebooks (Deliverables from M1 onward)

Rule: each milestone delivers a notebook that is:
- committed with **executed outputs** (so reviewers can inspect results immediately)
- includes an **“Open in Colab”** badge in the first markdown cell
- writes to a deterministic `runs_dir` and commits a small artifact snapshot:
  - `results.csv`, `summary.json`, plus one representative `jobs/<job_id>/results.json`

**Notebook rule (no “smoke-only” => real by default)**
- Notebooks default to **`mode: real`** and use the configured LLM backend (OpenRouter recommended).
- If no key is present, the notebook must:
  - print a clear warning, and
  - switch to stub mode if explictely validated by user, and
  - label outputs as **STUB** (so results are not mistaken for real runs during our test for validation).

### Notebooks
- **M1**: `notebooks/01_m1_minimal_api.ipynb`
- **M2**: `notebooks/02_m2_coverage.ipynb`
- **M3**: `notebooks/03_m3_ui_mlflow_gradio.ipynb`
- **M4**: `notebooks/04_m4_gepa_curriculum.ipynb`

---

## 10) Acceptance Criteria (SMART, verifiable)

### M0 (this document)
- Milestone definitions are consistent with delivery expectations (M0..M4 fixed mapping).
- Contracts are explicitly stated and unambiguous:
  - run_id/job_id
  - matrix semantics
  - canonical artifacts schema
  - discovery contracts
  - env.json security policy
  - LLM4AD knob coverage contract

### M1 (full API + minimal runnable coverage)
Goal: implement Trace‑Bench (CLI + Python API) and prove the runner can execute at least one job of each required type.

**Validation must demonstrate all of:**

1) **CLI surface**
- `trace-bench list-tasks` works for `llm4ad`, `veribench`, `trace_examples`, `internal`
- `trace-bench list-trainers` lists available trainer IDs
- `trace-bench validate --config configs/m1_validation.yaml --strict`:
  - tasks resolve and build runnable bundles
  - trainers resolve
  - optimizer/guide/logger resolve
  - matrix expands deterministically and writes `meta/manifest.json`
  - unknown kwargs / missing trainables / task-build failures are surfaced as clear validation errors (no silent ignore)

2) **Minimal runnable suite coverage**
- Internal examples: at least these optimization target types are executed:
  - trainable **code** ParameterNode
  - trainable **numeric** ParameterNode
  - multi-parameter **Module**
  - **non-trainable** parameter case is rejected cleanly (job `failed` with explicit reason)
- LLM4AD: run **≥1 task** with bounded `eval_kwargs.timeout_seconds`
  - **M1 notebook:** real-mode by default (uses key if present); artifacts show `mode=real`.
  - **PR CI:** stub/offline is acceptable to control cost.
- VeriBench: run **≥1 task** if entrypoint available; otherwise job is `skipped` with structured reason
- OpenTrace examples: smoke import/`--help` wired in CI (bounded by timeout)

2b) **Bounded matrix smoke (Plan A+ Pareto)**
- Execute exactly **2 tasks × 2 trainers × 1 seed** (4 jobs) and verify `results.csv` has 4 rows and `summary.json` aggregates them.

3) **Trainer coverage**
- Each “supported” trainer ID is executed at least once with at least one non-default kwarg.
- At least one run uses explicit optimizer override and records `resolved_optimizer_kwargs`.

4) **Artifacts**
- canonical layout exists and is populated:
  - `meta/config.snapshot.yaml`, `meta/manifest.json`, `meta/env.json` (redacted)
  - at least one `jobs/<job_id>/results.json` and `jobs/<job_id>/events.jsonl`
  - `results.csv` + `summary.json`

5) **Notebook**
- `notebooks/01_m1_minimal_api.ipynb` is committed, executed, includes “Open in Colab” badge, and shows inspection of produced artifacts.

### M2 (full coverage + efficiency)
Goal: expand breadth and efficiency while meeting coverage targets.

- Matrix engine supports:
  - multi-task × multi-trainer × params_variants × seeds
  - per-job isolation, partial failure tolerance
  - aggregation into `results.csv` + `summary.json`
  - skip-existing/resume semantics (re-running does not overwrite; it reuses existing completed jobs)
- Coverage targets achieved:
  - LLM4AD: ≥80% functional in Real mode over discovered tasks; optimizing subset shows improvement definition satisfied
  - VeriBench: ≥80% functional in Real mode over discovered tasks (when available)
- OpenTrace examples smoke: **100% pass** in CI (PASS or explicit allowlisted SKIP)

### M3 (UI + MLflow + Gradio)
Goal: operational interface + monitoring.

- Gradio UI supports configuring runner options (or selecting existing configs), launches runs, shows recent runs, and browses results
- MLflow is wired and shows run/job tags and metrics
- TensorBoard logdir and open command are visible per job
- UI reads filesystem canonical artifacts (`manifest.json`, `results.csv`, `summary.json`) and does not rely on MLflow as source-of-truth

### M4 (GEPA + Curriculum)
Goal: trainer integration and demonstration.

- GEPA + Curriculum trainers are integrated (when provided) and runnable in both stub and real modes
- Demonstrated on ≥2 tasks with artifacts and logs consistent with schema

---

## 11) Next Step

Adopt **Plan A+ (Pareto)**:
- keep M1 small, but include:
  - the **compatibility harness** (`configs/m1_validation.yaml`)
  - the **bounded 2×2 matrix smoke**
  - **strict edge validation** (`validate --strict`)
- defer **broad VeriBench discovery/coverage** and **large matrices** to M2.

Then lock:
- `configs/m1_validation.yaml` (the minimal M1 proof configuration)
- the list of trainer IDs considered “supported” for M1 (as exposed by `trace-bench list-trainers`)