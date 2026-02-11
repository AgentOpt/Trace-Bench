from trace_bench.cli import cmd_list_tasks, cmd_validate


def test_veribench_list_tasks_does_not_fail():
    assert cmd_list_tasks("LLM4AD/benchmark_tasks", bench="veribench") == 0


def test_veribench_validate_does_not_fail(tmp_path, capsys):
    config_path = tmp_path / "veribench.yaml"
    config_path.write_text(
        "tasks:\n  - id: veribench:smoke_placeholder\n", encoding="utf-8"
    )
    assert cmd_validate(str(config_path), "LLM4AD/benchmark_tasks", bench="veribench") == 0
    out = capsys.readouterr().out
    assert "[SKIP]" in out
