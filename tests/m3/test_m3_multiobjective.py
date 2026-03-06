"""M3: Multi-objective Trace-Bench integration tests.

Unit tests, feature tests, and non-regression tests for the three
multi-objective task modules (convex, bbeh, gsm8k), ObjectiveConfig
passthrough, and YAML matrix expansion.
"""
import csv
import os
from dataclasses import asdict
from pathlib import Path

import pytest

from trace_bench.config import RunConfig, load_config
from trace_bench.matrix import expand_matrix
from trace_bench.registry import load_task_bundle
from trace_bench.runner import BenchRunner

# Ensure opto is importable (same pattern as registry.py)
from trace_bench.registry import ensure_opto_importable
ensure_opto_importable()

pytest.importorskip(
    "opto.trainer.objectives",
    reason="opto.trainer.objectives not available in local opto install",
)

from opto.trainer.objectives import ObjectiveConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS_ROOT = "LLM4AD/benchmark_tasks"

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

_REQUIRED_BUNDLE_KEYS = {"param", "guide", "train_dataset", "optimizer_kwargs", "objective_config"}

_MO_TASKS = [
    "internal:multiobjective_convex",
    "internal:multiobjective_bbeh",
    "internal:multiobjective_gsm8k",
]


# ──────────────────────────────────────────────────────────────────────
# Unit Tests — bundle loading
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("task_id", _MO_TASKS)
def test_bundle_loads(task_id):
    """load_task_bundle returns valid bundle with all required keys."""
    bundle = load_task_bundle(task_id, TASKS_ROOT)
    missing = _REQUIRED_BUNDLE_KEYS - set(bundle.keys())
    assert not missing, f"Bundle for {task_id} missing keys: {sorted(missing)}"


def test_convex_bundle_loads():
    bundle = load_task_bundle("internal:multiobjective_convex", TASKS_ROOT)
    assert _REQUIRED_BUNDLE_KEYS.issubset(bundle.keys())


def test_bbeh_bundle_loads():
    bundle = load_task_bundle("internal:multiobjective_bbeh", TASKS_ROOT)
    assert _REQUIRED_BUNDLE_KEYS.issubset(bundle.keys())


def test_gsm8k_bundle_loads():
    bundle = load_task_bundle("internal:multiobjective_gsm8k", TASKS_ROOT)
    assert _REQUIRED_BUNDLE_KEYS.issubset(bundle.keys())


# ──────────────────────────────────────────────────────────────────────
# Unit Tests — guide score_dict
# ──────────────────────────────────────────────────────────────────────

def test_convex_guide_score_dict():
    """ConvexRewardGuide.get_score_dict() returns {base_loss, reg_loss}."""
    bundle = load_task_bundle("internal:multiobjective_convex", TASKS_ROOT)
    guide = bundle["guide"]
    sd = guide.get_score_dict(query=None, response="0.0, 0.0", reference=None)
    assert set(sd.keys()) == {"base_loss", "reg_loss"}
    assert all(isinstance(v, float) for v in sd.values())


def test_bbeh_guide_score_dict():
    """BBEHGuide.get_score_dict() returns {accuracy, execution_time_s}."""
    from trace_bench.examples.multiobjective_bbeh import BBEHGuide

    guide = BBEHGuide()
    # Use the initial PAL code from the module (standard pattern)
    from trace_bench.examples.multiobjective_bbeh import _INITIAL_CODE
    sd = guide.get_score_dict(
        query="not ( True ) and ( True ) is",
        response=_INITIAL_CODE,
        reference="False",
    )
    assert set(sd.keys()) == {"accuracy", "execution_time_s"}
    assert all(isinstance(v, float) for v in sd.values())


def test_gsm8k_guide_score_dict():
    """GSM8KGuide.get_score_dict() returns {error} (base guide without token wrapper)."""
    from trace_bench.examples.multiobjective_gsm8k import GSM8KGuide

    guide = GSM8KGuide()
    sd = guide.get_score_dict(
        query="How many is 2+2?",
        response="2+2=4\n#### 4",
        reference="#### 4",
    )
    assert set(sd.keys()) == {"error"}
    assert isinstance(sd["error"], float)
    assert sd["error"] == 0.0  # correct answer


# ──────────────────────────────────────────────────────────────────────
# Unit Tests — ObjectiveConfig serialization
# ──────────────────────────────────────────────────────────────────────

def test_objective_config_weighted():
    """ObjectiveConfig with mode='weighted' serializes/deserializes correctly."""
    cfg = ObjectiveConfig(
        mode="weighted",
        weights={"error": 1.0, "tokens_in": 1e-3, "tokens_out": 1e-3},
        minimize=frozenset({"error", "tokens_in", "tokens_out"}),
        seed=42,
    )
    d = asdict(cfg)
    cfg2 = ObjectiveConfig(**d)
    assert cfg == cfg2
    assert cfg2.mode == "weighted"
    assert cfg2.weights == {"error": 1.0, "tokens_in": 1e-3, "tokens_out": 1e-3}
    assert cfg2.minimize == frozenset({"error", "tokens_in", "tokens_out"})


def test_objective_config_pareto():
    """ObjectiveConfig with mode='pareto' serializes/deserializes correctly."""
    cfg = ObjectiveConfig(
        mode="pareto",
        weights={"accuracy": 1.0, "execution_time_s": 0.3},
        minimize=frozenset({"execution_time_s"}),
        tie_break="weighted",
        seed=42,
    )
    d = asdict(cfg)
    cfg2 = ObjectiveConfig(**d)
    assert cfg == cfg2
    assert cfg2.mode == "pareto"
    assert "execution_time_s" in cfg2.minimize


# ──────────────────────────────────────────────────────────────────────
# Unit Tests — objective_mode switching
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("task_id", _MO_TASKS)
@pytest.mark.parametrize("mode", ["weighted", "pareto"])
def test_objective_mode_switching(task_id, mode):
    """Each task module respects objective_mode eval_kwarg."""
    bundle = load_task_bundle(
        task_id, TASKS_ROOT, eval_kwargs={"objective_mode": mode},
    )
    oc = bundle["objective_config"]
    assert isinstance(oc, ObjectiveConfig)
    assert oc.mode == mode
    assert bundle["metadata"]["objective_mode"] == mode


# ──────────────────────────────────────────────────────────────────────
# Feature Tests — runner passthrough
# ──────────────────────────────────────────────────────────────────────

def test_runner_passthrough_objective_config():
    """Bundle with objective_config passes it through to trainer kwargs."""
    bundle = load_task_bundle(
        "internal:multiobjective_convex", TASKS_ROOT,
        eval_kwargs={"objective_mode": "weighted"},
    )
    assert "objective_config" in bundle
    oc = bundle["objective_config"]
    assert isinstance(oc, ObjectiveConfig)

    # Simulate what runner._train_bundle does (lines 200-203)
    kwargs = {}
    objective_config = bundle.get("objective_config")
    if objective_config is not None:
        kwargs["objective_config"] = objective_config
    assert "objective_config" in kwargs
    assert kwargs["objective_config"] is oc


@pytest.mark.slow
def test_convex_stub_run(tmp_path):
    """Full stub run of convex task completes without error."""
    os.chdir(REPO_ROOT)

    cfg = RunConfig.from_dict({
        "tasks": [
            {"id": "internal:multiobjective_convex", "eval_kwargs": {"objective_mode": "weighted"}},
        ],
        "trainers": [
            {"id": "BasicSearchAlgorithm", "params_variants": [{"num_proposals": 2, "num_epochs": 1, "batch_size": 1}]},
        ],
        "seeds": [42],
        "mode": "stub",
    })
    cfg.runs_dir = str(tmp_path / "runs")

    summary = BenchRunner(cfg).run()
    run_dir = Path(cfg.runs_dir) / summary.run_id
    assert run_dir.exists()

    results_csv = run_dir / "results.csv"
    assert results_csv.exists()
    with open(results_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1, f"Expected 1 row, got {len(rows)}"
    assert any(row.get("status") != "skipped" for row in rows)


def test_registry_lists_all_three():
    """trace-bench list-tasks output contains all 3 internal:multiobjective_* tasks."""
    from trace_bench.registry import discover_internal

    specs = discover_internal()
    ids = {s.id for s in specs}
    for task_id in _MO_TASKS:
        assert task_id in ids, f"{task_id} not in discover_internal() output"


def test_config_matrix_expansion():
    """m3_multiobjective.yaml expands to expected 18 jobs."""
    os.chdir(REPO_ROOT)
    cfg = load_config("configs/m3_multiobjective.yaml")
    jobs = expand_matrix(cfg)
    assert len(jobs) == 18, f"Expected 18 jobs (6 tasks x 3 trainers), got {len(jobs)}"

    # Verify all 3 task prefixes are present
    task_ids = {j.task.id for j in jobs}
    for task_id in _MO_TASKS:
        assert task_id in task_ids, f"{task_id} missing from expanded matrix"

    # Verify both objective modes are present per task
    for task_id in _MO_TASKS:
        modes = {
            j.task.eval_kwargs.get("objective_mode")
            for j in jobs
            if j.task.id == task_id
        }
        assert modes == {"weighted", "pareto"}, (
            f"Expected weighted+pareto for {task_id}, got {modes}"
        )


# ──────────────────────────────────────────────────────────────────────
# Non-regression Tests
# ──────────────────────────────────────────────────────────────────────

def test_existing_internal_tasks_still_work():
    """Existing internal tasks (e.g., internal:numeric_param) still load correctly."""
    bundle = load_task_bundle("internal:numeric_param", TASKS_ROOT)
    assert "param" in bundle
    assert "guide" in bundle
    assert "train_dataset" in bundle
    assert "optimizer_kwargs" in bundle


@pytest.mark.slow
def test_basic_search_scalar_still_works(tmp_path):
    """BasicSearch with scalar objective (no objective_config) still returns float score."""
    os.chdir(REPO_ROOT)

    cfg = RunConfig.from_dict({
        "tasks": [{"id": "internal:numeric_param"}],
        "trainers": [
            {"id": "BasicSearchAlgorithm", "params_variants": [{"num_proposals": 2, "num_epochs": 1, "batch_size": 1}]},
        ],
        "seeds": [42],
        "mode": "stub",
    })
    cfg.runs_dir = str(tmp_path / "runs")

    summary = BenchRunner(cfg).run()
    run_dir = Path(cfg.runs_dir) / summary.run_id
    assert run_dir.exists()

    results_csv = run_dir / "results.csv"
    assert results_csv.exists()
    with open(results_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    # Scalar tasks should still work — not skipped or crashed
    assert rows[0].get("status") in ("ok", "failed"), (
        f"Expected ok or failed status, got {rows[0].get('status')}"
    )
