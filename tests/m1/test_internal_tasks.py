from trace_bench.config import RunConfig
from trace_bench.registry import load_task_bundle
from trace_bench.runner import BenchRunner


def test_internal_tasks_load():
    bundle = load_task_bundle("internal:code_param", "LLM4AD/benchmark_tasks")
    assert "param" in bundle
    bundle2 = load_task_bundle("internal:numeric_param", "LLM4AD/benchmark_tasks")
    assert "param" in bundle2


def test_internal_non_trainable_fails(tmp_path):
    cfg = RunConfig.from_dict(
        {
            "tasks": [{"id": "internal:non_trainable"}],
            "trainers": [{"id": "PrioritySearch", "params_variants": [{"ps_steps": 1}]}],
            "seeds": [123],
        }
    )
    cfg.runs_dir = str(tmp_path / "runs")
    summary = BenchRunner(cfg).run()
    assert any(row.get("status") == "failed" for row in summary.results)
