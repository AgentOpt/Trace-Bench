from pathlib import Path

from trace_bench.cli import cmd_validate


def test_validate_writes_manifest_to_runs_dir(tmp_path):
    config_path = tmp_path / "validate.yaml"
    config_path.write_text(
        "\n".join(
            [
                "mode: stub",
                "tasks:",
                "  - id: internal:numeric_param",
                "trainers:",
                "  - id: PrioritySearch",
                "    params_variants:",
                "      - threads: 2",
            ]
        ),
        encoding="utf-8",
    )

    runs_dir = tmp_path / "colab_runs"
    rc = cmd_validate(
        str(config_path),
        "benchmarks/LLM4AD/benchmark_tasks",
        bench=None,
        strict=True,
        runs_dir=str(runs_dir),
    )
    assert rc == 0

    run_dirs = [path for path in runs_dir.iterdir() if path.is_dir()]
    assert run_dirs, "validate should create one run directory under --runs-dir"
    manifest_path = run_dirs[0] / "meta" / "manifest.json"
    assert manifest_path.exists()

