from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import os
import subprocess
import sys
import traceback

from trace_bench.config import RunConfig, load_config
from trace_bench.runner import BenchRunner
from trace_bench.ui.discovery import (
    discover_runs,
    filter_results_rows,
    load_job_details,
    load_run_summary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_gradio():
    try:
        import gradio as gr  # type: ignore
        return gr
    except Exception:
        return None


def _list_configs(configs_dir: str) -> List[str]:
    """List YAML/JSON configs in *configs_dir*."""
    root = Path(configs_dir)
    if not root.exists():
        return []
    out: List[str] = []
    for pat in ("*.yaml", "*.yml", "*.json"):
        out.extend([str(p) for p in sorted(root.glob(pat))])
    return out


def _default_configs_dir() -> str:
    """Best-effort default: repo configs/ directory."""
    repo_configs = Path(__file__).resolve().parents[2] / "configs"
    return str(repo_configs) if repo_configs.exists() else "configs"


def _default_tasks_root() -> str:
    """Best-effort default: LLM4AD/benchmark_tasks if it exists, else '.'."""
    candidates = [
        Path(__file__).resolve().parents[2] / "LLM4AD" / "benchmark_tasks",
        Path("LLM4AD/benchmark_tasks"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return "LLM4AD/benchmark_tasks"


def _apply_overrides(
    cfg: RunConfig,
    mode: str,
    max_workers: int,
    resume: str,
    job_timeout: float,
    force: bool,
) -> Tuple[RunConfig, bool]:
    rerun_all = bool(force)
    if mode:
        cfg.mode = mode
    if max_workers and int(max_workers) > 0:
        cfg.max_workers = int(max_workers)
    if resume:
        cfg.resume = resume
    if job_timeout and float(job_timeout) > 0:
        cfg.job_timeout = float(job_timeout)
    return cfg, rerun_all


def _tb_command(run_dir: str) -> str:
    return f"tensorboard --logdir \"{Path(run_dir) / 'jobs'}\""


def _try_launch_tensorboard(run_dir: str) -> str:
    cmd = _tb_command(run_dir)
    try:
        subprocess.Popen(cmd, shell=True)  # noqa: S603,S607
        return f"Started TensorBoard.\n{cmd}"
    except Exception as exc:
        return f"Could not launch TensorBoard ({exc}).\n{cmd}"


def _discover_tasks_safe(tasks_root: str, bench: str) -> str:
    """Discover tasks and return a newline-separated list."""
    try:
        from trace_bench.registry import discover_tasks
        specs = discover_tasks(tasks_root, bench or None)
        return "\n".join(f"{s.id}  ({s.suite})" for s in specs)
    except Exception as exc:
        return f"Error discovering tasks: {exc}"


def _discover_trainers_safe() -> str:
    """Discover trainers and return a newline-separated list."""
    try:
        from trace_bench.registry import discover_trainers
        specs = discover_trainers()
        lines = []
        for s in specs:
            tag = "available" if s.available else "unavailable"
            lines.append(f"{s.id}  ({tag})")
        return "\n".join(lines)
    except Exception as exc:
        return f"Error discovering trainers: {exc}"


def _discover_tasks_for_ui(tasks_root: str, bench: str):
    """Return (log_text, checkbox update) for task discovery."""
    gr = _import_gradio()
    if gr is None:
        return "Gradio unavailable", None
    try:
        from trace_bench.registry import discover_tasks

        specs = discover_tasks(tasks_root, bench or None)
        ids = [s.id for s in specs]
        lines = [f"{s.id}  ({s.suite})" for s in specs]
        selected = ids[: min(12, len(ids))]
        return "\n".join(lines), gr.CheckboxGroup(choices=ids, value=selected)
    except Exception as exc:
        return f"Error discovering tasks: {exc}", gr.CheckboxGroup(choices=[], value=[])


def _discover_trainers_for_ui():
    """Return (log_text, checkbox update) for trainer discovery."""
    gr = _import_gradio()
    if gr is None:
        return "Gradio unavailable", None
    try:
        from trace_bench.registry import discover_trainers

        specs = discover_trainers()
        ids = [s.id for s in specs if s.available]
        lines = [f"{s.id}  ({'available' if s.available else 'unavailable'})" for s in specs]
        selected = ids[: min(3, len(ids))]
        return "\n".join(lines), gr.CheckboxGroup(choices=ids, value=selected)
    except Exception as exc:
        return f"Error discovering trainers: {exc}", gr.CheckboxGroup(choices=[], value=[])


def _default_params_for_trainer(trainer_id: str) -> Dict[str, Any]:
    if trainer_id == "PrioritySearch":
        return {"ps_steps": 1, "ps_batches": 1, "ps_candidates": 2, "ps_proposals": 2}
    if trainer_id.startswith("GEPA"):
        return {"gepa_iters": 1, "gepa_train_bs": 2}
    return {}


def _compose_editor_config(mode: str, selected_tasks: List[str], selected_trainers: List[str]) -> str:
    if not selected_tasks:
        return "# Select at least one task first."
    if not selected_trainers:
        return "# Select at least one trainer first."
    payload = {
        "mode": mode or "stub",
        "seeds": [123],
        "max_workers": 2,
        "resume": "auto",
        "tasks": [{"id": t} for t in selected_tasks],
        "trainers": [
            {"id": tid, "params_variants": [_default_params_for_trainer(tid)]}
            for tid in selected_trainers
        ],
    }
    try:
        import yaml

        return yaml.safe_dump(payload, sort_keys=False)
    except Exception:
        return json.dumps(payload, indent=2)


def _apply_llm_env(api_base: str, api_key: str, model_name: str) -> None:
    """Apply optional runtime LLM settings from UI controls."""
    if api_key and api_key.strip():
        os.environ["OPENAI_API_KEY"] = api_key.strip()
        os.environ["OPENROUTER_API_KEY"] = api_key.strip()
    if api_base and api_base.strip():
        os.environ["OPENAI_API_BASE"] = api_base.strip()
        os.environ["OPENAI_BASE_URL"] = api_base.strip()
    if model_name and model_name.strip():
        os.environ["TRACE_LITELLM_MODEL"] = model_name.strip()


# ---------------------------------------------------------------------------
# Unified run handler
# ---------------------------------------------------------------------------


def _run_config(
    cfg: RunConfig,
    runs_dir_text: str,
    tasks_root: str,
    mode: str,
    max_workers: int,
    resume: str,
    job_timeout: float,
    force: bool,
) -> Tuple[str, str]:
    """Run a config and return (log_message, run_id)."""
    if runs_dir_text:
        cfg.runs_dir = runs_dir_text
    cfg, rerun_all = _apply_overrides(cfg, mode, max_workers, resume, job_timeout, force)
    runner = BenchRunner(cfg, tasks_root=tasks_root, job_timeout=cfg.job_timeout)
    summary = runner.run(force=rerun_all)
    return f"Run complete: {summary.run_id} ({len(summary.results)} results)", summary.run_id


def _unified_run(
    runs_dir_text: str,
    tasks_root: str,
    editor_text: str,
    uploaded,
    config_path: str,
    mode: str,
    max_workers: int,
    resume: str,
    job_timeout: float,
    force: bool,
    api_base: str = "",
    api_key: str = "",
    model_name: str = "",
) -> Tuple[str, str]:
    """Single run handler with priority: editor > upload > picker."""
    try:
        _apply_llm_env(api_base, api_key, model_name)
        # Priority 1: editor text (if non-empty)
        if editor_text and editor_text.strip():
            try:
                import yaml
                data = yaml.safe_load(editor_text)
            except Exception:
                data = json.loads(editor_text)
            if not isinstance(data, dict):
                return "Config must be a YAML/JSON mapping.", ""
            cfg = RunConfig.from_dict(data)
            return _run_config(cfg, runs_dir_text, tasks_root, mode, max_workers, resume, job_timeout, force)

        # Priority 2: uploaded file
        if uploaded is not None:
            path = getattr(uploaded, "name", None) or str(uploaded)
            cfg = load_config(path)
            return _run_config(cfg, runs_dir_text, tasks_root, mode, max_workers, resume, job_timeout, force)

        # Priority 3: selected config path
        if config_path:
            cfg = load_config(config_path)
            return _run_config(cfg, runs_dir_text, tasks_root, mode, max_workers, resume, job_timeout, force)

        return "No config provided. Use the editor, upload a file, or select from configs.", ""
    except Exception as exc:
        return f"Run failed: {exc}\n{traceback.format_exc()}", ""


# ---------------------------------------------------------------------------
# Resume handler
# ---------------------------------------------------------------------------


def _resume_run(
    runs_dir_text: str,
    tasks_root: str,
    run_id: str,
    new_run_id: bool,
) -> Tuple[str, str]:
    """Resume a previous run using its config snapshot."""
    if not run_id:
        return "No run selected to resume.", ""
    try:
        run_path = Path(runs_dir_text) / run_id
        config_path = run_path / "meta" / "config.snapshot.yaml"
        if not config_path.exists():
            config_path = run_path / "config.snapshot.yaml"
        if not config_path.exists():
            return f"No config snapshot found in {run_path}", ""

        cfg = load_config(str(config_path))
        cfg.runs_dir = runs_dir_text
        cfg.resume = "auto"
        if not new_run_id:
            cfg.run_id = run_id  # preserve original run directory
        else:
            cfg.run_id = None  # generate fresh run_id

        runner = BenchRunner(cfg, tasks_root=tasks_root, job_timeout=cfg.job_timeout)
        summary = runner.run(force=False)
        return f"Resume complete: {summary.run_id} ({len(summary.results)} results)", summary.run_id
    except Exception as exc:
        return f"Resume failed: {exc}\n{traceback.format_exc()}", ""


# ---------------------------------------------------------------------------
# Main UI builder
# ---------------------------------------------------------------------------


def launch_ui(
    runs_dir: str,
    tasks_root: Optional[str] = None,
    share: bool = False,
    port: Optional[int] = None,
) -> int:
    gr = _import_gradio()
    if gr is None:
        print("Gradio is not installed. Install with: pip install gradio")
        return 1

    initial_runs_dir = runs_dir or "runs"
    initial_tasks_root = tasks_root or _default_tasks_root()
    initial_configs_dir = _default_configs_dir()
    initial_configs = _list_configs(initial_configs_dir)

    # Auto-detect Colab for share
    if not share and "google.colab" in sys.modules:
        share = True

    # --- Internal callbacks ---

    def _refresh_configs(configs_dir_text: str):
        choices = _list_configs(configs_dir_text)
        return gr.Dropdown(choices=choices, value=(choices[0] if choices else None))

    def _load_config_to_editor(config_path: str):
        if not config_path:
            return ""
        try:
            return Path(config_path).read_text(encoding="utf-8")
        except Exception as exc:
            return f"# Error reading config: {exc}"

    def _save_config(editor_text: str, save_path: str):
        if not editor_text.strip():
            return "Nothing to save (editor is empty)."
        if not save_path.strip():
            return "No save path specified."
        try:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(editor_text, encoding="utf-8")
            return f"Saved to {p}"
        except Exception as exc:
            return f"Save failed: {exc}"

    def _refresh_runs(runs_dir_text: str):
        recs = discover_runs(runs_dir_text)
        choices = [r.run_id for r in recs]
        return gr.Dropdown(choices=choices, value=(choices[0] if choices else None))

    def _load_selected_run(runs_dir_text: str, run_id: str):
        if not run_id:
            return "", [], "", "{}", "{}", ""
        run_path = Path(runs_dir_text) / run_id
        d = load_run_summary(run_path)
        return (
            d["config_text"],
            d["results_rows"],
            d["env_text"],
            json.dumps(d["summary"], indent=2, ensure_ascii=False),
            json.dumps(d["manifest"], indent=2, ensure_ascii=False),
            _tb_command(str(run_path)),
        )

    def _filter_rows(
        rows: List[Dict[str, Any]],
        suite: str,
        status: str,
        trainer_id: str,
        task_substring: str,
    ):
        return filter_results_rows(rows, suite=suite, status=status, trainer_id=trainer_id, task_substring=task_substring)

    def _job_ids_from_rows(rows: List[Dict[str, Any]]) -> List[str]:
        ids: List[str] = []
        for r in rows or []:
            jid = r.get("job_id")
            if jid:
                ids.append(str(jid))
        return ids

    def _load_job(runs_dir_text: str, run_id: str, job_id: str):
        if not run_id or not job_id:
            return "", "{}", "{}", "", ""
        run_path = Path(runs_dir_text) / run_id
        d = load_job_details(run_path, job_id)
        return (
            d["job_dir"],
            json.dumps(d["job_meta"], indent=2, ensure_ascii=False),
            json.dumps(d["job_results"], indent=2, ensure_ascii=False),
            d["events_head"],
            d["tb_dir"],
        )

    def _load_and_filter(runs_dir_text: str, run_id: str, suite: str, status: str, trainer_id: str, task_sub: str):
        cfg_text, rows, env_text, summary, manifest, tb = _load_selected_run(runs_dir_text, run_id)
        filtered = _filter_rows(rows, suite, status, trainer_id, task_sub)
        return cfg_text, filtered, env_text, summary, manifest, tb

    def _load_jobs(runs_dir_text: str, run_id: str):
        if not run_id:
            return gr.Dropdown(choices=[], value=None)
        run_path = Path(runs_dir_text) / run_id
        d = load_run_summary(run_path)
        ids = _job_ids_from_rows(d["results_rows"])
        return gr.Dropdown(choices=ids, value=(ids[0] if ids else None))

    # --- Build UI ---

    ui_css = """
    .tb-shell {border: 1px solid #c7d2e0; border-radius: 10px; padding: 8px; background: linear-gradient(180deg, #f8fbff 0%, #f3f6fb 100%);}
    .tb-left {border-right: 1px solid #d6deea; padding-right: 8px;}
    .tb-title {font-weight: 700; margin: 0 0 4px 0;}
    """

    with gr.Blocks(title="Trace-Bench UI", css=ui_css) as demo:
        gr.Markdown("# Trace-Bench UI (M3)")
        gr.Markdown(
            "Artifacts under `runs/<run_id>/` are canonical. "
            "UI reads filesystem; MLflow/TB mirror it."
        )

        with gr.Tabs():
            # ========== TAB 1: Launch Run ==========
            with gr.Tab("Launch Run"):
                with gr.Row(elem_classes=["tb-shell"]):
                    with gr.Column(scale=4, elem_classes=["tb-left"]):
                        gr.Markdown("### LLM setups")
                        with gr.Row():
                            api_base = gr.Textbox(
                                value=os.environ.get("OPENAI_BASE_URL", ""),
                                label="host/base URL",
                                placeholder="https://openrouter.ai/api/v1",
                            )
                            api_key = gr.Textbox(
                                value="",
                                label="key",
                                type="password",
                                placeholder="optional override",
                            )
                        model_name = gr.Textbox(
                            value=os.environ.get("TRACE_LITELLM_MODEL", ""),
                            label="model",
                            placeholder="openrouter/openai/gpt-4o-mini",
                        )

                        with gr.Row():
                            runs_dir_text = gr.Textbox(
                                value=initial_runs_dir, label="runs_dir",
                            )
                            tasks_root_box = gr.Textbox(
                                value=initial_tasks_root, label="tasks_root",
                            )
                        bench_selector = gr.Dropdown(
                            choices=["", "llm4ad", "trace_examples", "internal", "veribench"],
                            value="", label="bench filter",
                        )

                        with gr.Row():
                            discover_tasks_btn = gr.Button("Discover Tasks", variant="secondary")
                            discover_trainers_btn = gr.Button("Discover Trainers", variant="secondary")
                        selected_trainers = gr.CheckboxGroup(label="Methods", choices=[])
                        selected_tasks = gr.CheckboxGroup(label="Tasks", choices=[])
                        discovery_output = gr.Textbox(label="Discovery results", lines=6, interactive=False)

                        with gr.Row():
                            configs_dir_box = gr.Textbox(value=initial_configs_dir, label="configs dir")
                            refresh_configs_btn = gr.Button("Refresh configs", variant="secondary")
                        config_picker = gr.Dropdown(choices=initial_configs, label="config picker")
                        config_upload = gr.File(label="upload config (YAML/JSON)")
                        load_to_editor_btn = gr.Button("Load picked config", variant="secondary")

                        config_editor = gr.Code(
                            label="Config editor (priority: editor > upload > picker)",
                            language="yaml",
                            lines=16,
                        )
                        build_editor_btn = gr.Button("Build config from selected Methods/Tasks", variant="secondary")

                        with gr.Row():
                            save_path_box = gr.Textbox(label="save as", placeholder="configs/my_custom.yaml")
                            save_btn = gr.Button("Save config", variant="secondary")
                        save_log = gr.Textbox(label="Save status", interactive=False)

                    with gr.Column(scale=5):
                        gr.Markdown("### Run controls")
                        with gr.Row():
                            mode = gr.Dropdown(choices=["", "stub", "real"], value="", label="mode override")
                            max_workers = gr.Number(value=0, label="max_workers (0=keep)")
                            resume = gr.Dropdown(choices=["", "auto", "failed", "none"], value="", label="resume")
                        with gr.Row():
                            job_timeout = gr.Number(value=0, label="job_timeout seconds (0=keep)")
                            force = gr.Checkbox(value=False, label="force rerun")

                        with gr.Row():
                            run_btn = gr.Button("Run", variant="primary")
                            stop_btn = gr.Button("Stop (manual)", variant="stop")
                        run_log = gr.Textbox(label="Run log", lines=4)
                        last_run_id = gr.Textbox(label="Last run_id", interactive=False)

                        gr.Markdown("### Current best / metrics")
                        latest_summary = gr.Code(label="Latest summary.json", language="json")
                        latest_results = gr.Dataframe(label="Latest results.csv rows")
                        refresh_latest_btn = gr.Button("Refresh latest run view", variant="secondary")

                def _latest_run_view(runs_dir_text: str):
                    recs = discover_runs(runs_dir_text)
                    if not recs:
                        return "{}", []
                    d = load_run_summary(recs[0].run_dir)
                    return json.dumps(d["summary"], indent=2), d["results_rows"][:20]

                discover_tasks_btn.click(
                    _discover_tasks_for_ui,
                    inputs=[tasks_root_box, bench_selector],
                    outputs=[discovery_output, selected_tasks],
                )
                discover_trainers_btn.click(
                    _discover_trainers_for_ui,
                    inputs=[],
                    outputs=[discovery_output, selected_trainers],
                )
                build_editor_btn.click(
                    _compose_editor_config,
                    inputs=[mode, selected_tasks, selected_trainers],
                    outputs=[config_editor],
                )
                refresh_configs_btn.click(_refresh_configs, inputs=[configs_dir_box], outputs=[config_picker])
                load_to_editor_btn.click(_load_config_to_editor, inputs=[config_picker], outputs=[config_editor])
                save_btn.click(_save_config, inputs=[config_editor, save_path_box], outputs=[save_log])
                run_btn.click(
                    _unified_run,
                    inputs=[
                        runs_dir_text, tasks_root_box, config_editor, config_upload, config_picker,
                        mode, max_workers, resume, job_timeout, force, api_base, api_key, model_name,
                    ],
                    outputs=[run_log, last_run_id],
                )
                stop_btn.click(lambda: "Use notebook interrupt to stop current process.", outputs=[run_log])
                refresh_latest_btn.click(_latest_run_view, inputs=[runs_dir_text], outputs=[latest_summary, latest_results])

            # ========== TAB 2: Browse Runs ==========
            with gr.Tab("Browse Runs"):
                runs_dir_text_b = gr.Textbox(value=initial_runs_dir, label="runs_dir")
                with gr.Row():
                    refresh_btn = gr.Button("Refresh runs")
                    run_selector = gr.Dropdown(
                        choices=[r.run_id for r in discover_runs(initial_runs_dir)],
                        label="run_id",
                    )

                with gr.Accordion("Config & metadata", open=False):
                    config_box = gr.Code(label="config.snapshot.yaml", language="yaml")
                    env_box = gr.Code(label="env.json", language="json")
                    summary_box = gr.Code(label="summary.json", language="json")
                    manifest_box = gr.Code(label="meta/manifest.json", language="json")

                gr.Markdown("#### Filters")
                with gr.Row():
                    suite_filter = gr.Textbox(value="", label="suite (exact)")
                    status_filter = gr.Textbox(value="", label="status (exact)")
                    trainer_filter = gr.Textbox(value="", label="trainer_id (exact)")
                    task_filter = gr.Textbox(value="", label="task_id contains")

                results_df = gr.Dataframe(label="results.csv (filtered)")

                gr.Markdown("#### TensorBoard")
                tb_cmd = gr.Textbox(label="TensorBoard command", lines=2)
                with gr.Row():
                    tb_launch_btn = gr.Button("Launch TensorBoard (best-effort)")
                    tb_launch_log = gr.Textbox(label="TensorBoard status", lines=2)

                gr.Markdown("#### Resume selected run")
                with gr.Row():
                    resume_new_id = gr.Checkbox(
                        value=False,
                        label="Create new run_id (uncheck to resume in-place)",
                    )
                    resume_tasks_root = gr.Textbox(
                        value=initial_tasks_root, label="tasks_root for resume",
                    )
                with gr.Row():
                    resume_btn = gr.Button("Resume selected run", variant="primary")
                    resume_log = gr.Textbox(label="Resume log", lines=2)
                    resume_run_id = gr.Textbox(label="Resumed run_id", interactive=False)

                # Wire browse callbacks
                refresh_btn.click(_refresh_runs, inputs=[runs_dir_text_b], outputs=[run_selector])
                run_selector.change(
                    _load_and_filter,
                    inputs=[runs_dir_text_b, run_selector, suite_filter, status_filter, trainer_filter, task_filter],
                    outputs=[config_box, results_df, env_box, summary_box, manifest_box, tb_cmd],
                )
                for widget in (suite_filter, status_filter, trainer_filter, task_filter):
                    widget.change(
                        _load_and_filter,
                        inputs=[runs_dir_text_b, run_selector, suite_filter, status_filter, trainer_filter, task_filter],
                        outputs=[config_box, results_df, env_box, summary_box, manifest_box, tb_cmd],
                    )

                tb_launch_btn.click(
                    lambda rid, d: _try_launch_tensorboard(str(Path(d) / rid)),
                    inputs=[run_selector, runs_dir_text_b],
                    outputs=[tb_launch_log],
                )

                resume_btn.click(
                    _resume_run,
                    inputs=[runs_dir_text_b, resume_tasks_root, run_selector, resume_new_id],
                    outputs=[resume_log, resume_run_id],
                )

            # ========== TAB 3: Job Inspector ==========
            with gr.Tab("Job Inspector"):
                runs_dir_text_j = gr.Textbox(value=initial_runs_dir, label="runs_dir")
                run_selector_j = gr.Dropdown(
                    choices=[r.run_id for r in discover_runs(initial_runs_dir)],
                    label="run_id",
                )
                load_jobs_btn = gr.Button("Load job list")
                job_selector = gr.Dropdown(choices=[], label="job_id")

                job_dir = gr.Textbox(label="job_dir", interactive=False)
                job_meta = gr.Code(label="job_meta.json", language="json")
                job_results = gr.Code(label="results.json", language="json")
                events_head = gr.Code(label="events.jsonl (head)", language="json")
                tb_dir = gr.Textbox(label="tb_dir", interactive=False)

                load_jobs_btn.click(
                    _load_jobs,
                    inputs=[runs_dir_text_j, run_selector_j],
                    outputs=[job_selector],
                )
                job_selector.change(
                    _load_job,
                    inputs=[runs_dir_text_j, run_selector_j, job_selector],
                    outputs=[job_dir, job_meta, job_results, events_head, tb_dir],
                )

        # MLflow status
        try:
            import mlflow  # noqa: F401
            mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
            if mlflow_uri:
                gr.Markdown(f"**MLflow** detected (tracking URI: `{mlflow_uri}`). Mirroring enabled.")
            else:
                gr.Markdown("**MLflow** installed but `MLFLOW_TRACKING_URI` not set. Mirroring disabled.")
        except Exception:
            gr.Markdown("MLflow not installed. UI works without it.")

    launch_kwargs: Dict[str, Any] = {"share": share}
    if port is not None:
        launch_kwargs["server_port"] = port
    demo.launch(**launch_kwargs)
    return 0


__all__ = ["launch_ui"]
