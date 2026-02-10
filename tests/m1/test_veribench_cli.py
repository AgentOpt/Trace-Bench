import pytest

from trace_bench.cli import cmd_list_tasks, cmd_validate


def test_veribench_list_tasks_explicit_failure():
    with pytest.raises(NotImplementedError) as exc:
        cmd_list_tasks("LLM4AD/benchmark_tasks", bench="veribench")
    assert "awaiting trace team entrypoint/task list" in str(exc.value).lower()


def test_veribench_validate_explicit_failure(tmp_path):
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("tasks: []\n", encoding="utf-8")
    with pytest.raises(NotImplementedError) as exc:
        cmd_validate(str(config_path), "LLM4AD/benchmark_tasks", bench="veribench")
    assert "awaiting trace team entrypoint/task list" in str(exc.value).lower()
