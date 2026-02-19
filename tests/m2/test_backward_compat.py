"""M2: Backward-compatible trainers_benchmark.py wrapper."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Import the wrapper module
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "LLM4AD"))
from trainers_benchmark import _build_config_dict, _parse_args, main


def test_trainers_benchmark_deprecation_warning(capsys):
    """Running trainers_benchmark.py should print a deprecation warning."""
    # Patch cmd_run at the source (trace_bench.cli) since trainers_benchmark
    # imports it lazily inside main()
    with patch("trace_bench.cli.cmd_run", return_value=0):
        try:
            main(["--tasks", ".", "--task", "circle_packing"])
        except Exception:
            pass  # May fail due to config issues, but warning should print

    captured = capsys.readouterr()
    assert "deprecated" in captured.err.lower() or "DEPRECATED" in captured.err


def test_trainers_benchmark_builds_config():
    """_build_config_dict should generate a valid RunConfig-compatible dict."""
    args = _parse_args([
        "--tasks", "./benchmark_tasks",
        "--task", "circle_packing,knapsack",
        "--algos", "PrioritySearch,GEPA-Base",
        "--threads", "4",
        "--ps-steps", "10",
        "--gepa-iters", "5",
    ])

    config = _build_config_dict(args)

    # Check tasks are prefixed with llm4ad:
    assert len(config["tasks"]) == 2
    assert config["tasks"][0]["id"] == "llm4ad:circle_packing"
    assert config["tasks"][1]["id"] == "llm4ad:knapsack"

    # Check trainers
    assert len(config["trainers"]) == 2
    assert config["trainers"][0]["id"] == "PrioritySearch"
    assert config["trainers"][1]["id"] == "GEPA-Base"

    # Check params mapping
    ps_params = config["trainers"][0]["params"]
    assert ps_params["ps_steps"] == 10
    assert ps_params["threads"] == 4

    gepa_params = config["trainers"][1]["params"]
    assert gepa_params["gepa_iters"] == 5
    assert gepa_params["threads"] == 4

    # Check mode and resume
    assert config["mode"] == "real"
    assert config["resume"] == "auto"


def test_trainers_benchmark_task_prefix_passthrough():
    """Tasks already containing ':' should not be double-prefixed."""
    args = _parse_args([
        "--tasks", ".",
        "--task", "internal:numeric_param",
    ])
    config = _build_config_dict(args)
    assert config["tasks"][0]["id"] == "internal:numeric_param"


def test_trainers_benchmark_eval_kwargs():
    """--eval-kwargs JSON should be passed through to task eval_kwargs."""
    args = _parse_args([
        "--tasks", ".",
        "--task", "test_task",
        "--eval-kwargs", '{"timeout_seconds": 60}',
    ])
    config = _build_config_dict(args)
    assert config["tasks"][0]["eval_kwargs"]["timeout_seconds"] == 60
