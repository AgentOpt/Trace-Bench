from pathlib import Path

import pytest

from trace_bench.terminal_bench.harbor_runner import build_harbor_cmd


def test_build_harbor_cmd_requires_dataset_or_path(tmp_path: Path):
    with pytest.raises(ValueError):
        build_harbor_cmd(
            job_name="x",
            jobs_dir=tmp_path,
            model_name="m",
            env_type="docker",
            task_name="t",
            dataset=None,
            task_path=None,
        )

    with pytest.raises(ValueError):
        build_harbor_cmd(
            job_name="x",
            jobs_dir=tmp_path,
            model_name="m",
            env_type="docker",
            task_name="t",
            dataset="terminal-bench@2.0",
            task_path=tmp_path,
        )


def test_build_harbor_cmd_dataset(tmp_path: Path):
    cmd, _env = build_harbor_cmd(
        job_name="job",
        jobs_dir=tmp_path,
        model_name="m",
        env_type="daytona",
        task_name="regex-log",
        dataset="terminal-bench@2.0",
        task_path=None,
        agent_kwargs={"prompt_template_path": "/tmp/prompt.txt"},
        n_concurrent=1,
        quiet=True,
    )
    assert cmd[:2] == ["harbor", "run"]
    assert "--dataset" in cmd
    assert "--task-name" in cmd
    assert "regex-log" in cmd
    assert "--env" in cmd
    assert "daytona" in cmd


def test_build_harbor_cmd_path(tmp_path: Path):
    task_path = tmp_path / "task"
    task_path.mkdir()
    cmd, _env = build_harbor_cmd(
        job_name="job",
        jobs_dir=tmp_path,
        model_name="m",
        env_type="docker",
        task_name="ignored",
        dataset=None,
        task_path=task_path,
        agent_kwargs={},
        n_concurrent=1,
        quiet=False,
    )
    assert "--path" in cmd
    assert str(task_path) in cmd
    assert "--quiet" not in cmd


def test_run_harbor_eval_marks_trial_exception_failed(tmp_path: Path, monkeypatch):
    from types import SimpleNamespace

    from trace_bench.terminal_bench import harbor_runner

    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "job"
    trial_dir = job_dir / "regex-log__abc123"
    trial_dir.mkdir(parents=True)
    (job_dir / "result.json").write_text(
        '{"stats":{"n_errored_trials":1}}', encoding="utf-8"
    )
    (trial_dir / "result.json").write_text(
        '{"exception_info":{"exception_type":"BadRequestError"}}', encoding="utf-8"
    )

    monkeypatch.setattr(harbor_runner, "harbor_available", lambda: True)
    monkeypatch.setattr(
        harbor_runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    result = harbor_runner.run_harbor_eval(
        job_name="job",
        jobs_dir=jobs_dir,
        model_name="m",
        env_type="daytona",
        task_name="regex-log",
        dataset="terminal-bench@2.0",
    )

    assert result.status == "failed"
    assert result.trial_dir == trial_dir
    assert result.trial_result["exception_info"]["exception_type"] == "BadRequestError"


def test_build_harbor_cmd_agent_kwargs_are_forwarded(tmp_path: Path):
    cmd, _env = build_harbor_cmd(
        job_name="job",
        jobs_dir=tmp_path,
        model_name="m",
        env_type="daytona",
        task_name="regex-log",
        dataset="terminal-bench@2.0",
        task_path=None,
        agent_kwargs={"max_turns": 3, "parser_name": "json"},
        n_concurrent=1,
        quiet=True,
    )
    joined = " ".join(cmd)
    assert "--agent-kwarg max_turns=3" in joined
    assert "--agent-kwarg parser_name=json" in joined


def test_run_harbor_eval_without_job_result_returns_failure_payload(
    tmp_path: Path, monkeypatch
):
    from types import SimpleNamespace

    from trace_bench.terminal_bench import harbor_runner

    monkeypatch.setattr(harbor_runner, "harbor_available", lambda: True)
    monkeypatch.setattr(
        harbor_runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=2, stdout="", stderr="bad"),
    )

    result = harbor_runner.run_harbor_eval(
        job_name="job",
        jobs_dir=tmp_path / "jobs",
        model_name="m",
        env_type="daytona",
        task_name="regex-log",
        dataset="terminal-bench@2.0",
    )

    assert result.status == "failed"
    assert (
        result.job_result["trace_bench"]["failure_reason"]
        == "harbor_failed_without_job_result"
    )
    assert result.job_result["trace_bench"]["returncode"] == 2


def test_run_harbor_eval_retries_include_task_name_for_older_harbor(
    tmp_path: Path, monkeypatch
):
    from types import SimpleNamespace

    from trace_bench.terminal_bench import harbor_runner

    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "job"
    job_dir.mkdir(parents=True)
    (job_dir / "result.json").write_text('{"score": 1}', encoding="utf-8")
    seen_cmds = []

    def fake_run(cmd, **kwargs):
        seen_cmds.append(cmd)
        if "--task-name" in cmd:
            return SimpleNamespace(returncode=2, stdout="", stderr="No such option")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(harbor_runner, "harbor_available", lambda: True)
    monkeypatch.setattr(harbor_runner.subprocess, "run", fake_run)

    result = harbor_runner.run_harbor_eval(
        job_name="job",
        jobs_dir=jobs_dir,
        model_name="m",
        env_type="daytona",
        task_name="regex-log",
        dataset="terminal-bench@2.0",
    )

    assert result.status == "ok"
    assert "--task-name" in seen_cmds[0]
    assert "--include-task-name" in seen_cmds[1]


def test_run_harbor_eval_timeout_returns_failure_payload(tmp_path: Path, monkeypatch):
    from trace_bench.terminal_bench import harbor_runner

    def raise_timeout(cmd, **kwargs):
        raise harbor_runner.subprocess.TimeoutExpired(cmd=cmd, timeout=1, output="out")

    monkeypatch.setattr(harbor_runner, "harbor_available", lambda: True)
    monkeypatch.setattr(harbor_runner.subprocess, "run", raise_timeout)

    result = harbor_runner.run_harbor_eval(
        job_name="job",
        jobs_dir=tmp_path / "jobs",
        model_name="m",
        env_type="daytona",
        task_name="regex-log",
        dataset="terminal-bench@2.0",
        timeout_seconds=1,
    )

    assert result.status == "failed"
    assert (
        result.job_result["trace_bench"]["failure_reason"]
        == "harbor_failed_without_job_result"
    )
    assert result.job_result["trace_bench"]["returncode"] == 124
    assert "timed out" in result.stderr
