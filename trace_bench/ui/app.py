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


def _colab_secret(name: str) -> str:
    try:
        from google.colab import userdata  # type: ignore
        value = userdata.get(name)
        return value or ""
    except Exception:
        return ""


def _provider_defaults(provider: str, current_base: str = "", current_model: str = "") -> Tuple[str, str, str]:
    provider = (provider or "custom").lower()
    if provider == "openai":
        base = current_base or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE") or "https://api.openai.com/v1"
        key = os.environ.get("OPENAI_API_KEY", "") or _colab_secret("OPENAI_API_KEY")
        model = current_model or os.environ.get("TRACE_LITELLM_MODEL", "") or "gpt-4o-mini"
        return base, key, model
    if provider == "openrouter":
        base = current_base or "https://openrouter.ai/api/v1"
        key = (
            os.environ.get("OPENROUTER_API_KEY", "")
            or _colab_secret("OPENROUTER_API_KEY")
            or os.environ.get("OPENAI_API_KEY", "")
            or _colab_secret("OPENAI_API_KEY")
        )
        model = current_model or os.environ.get("TRACE_LITELLM_MODEL", "") or "openrouter/openai/gpt-4o-mini"
        return base, key, model
    return (
        current_base or os.environ.get("OPENAI_BASE_URL", ""),
        "",
        current_model or os.environ.get("TRACE_LITELLM_MODEL", ""),
    )


def _logger_choices() -> List[str]:
    base = ["default", "none"]
    try:
        import importlib

        mod = importlib.import_module("opto.trainer.loggers")
        base_cls = getattr(mod, "BaseLogger", None)
        if base_cls is None:
            return base + ["ConsoleLogger"]
        names: List[str] = []
        for key, value in vars(mod).items():
            if isinstance(value, type) and issubclass(value, base_cls) and value is not base_cls:
                if key == "NullLogger":
                    continue
                names.append(key)
        return base + sorted(set(names))
    except Exception:
        return base + ["ConsoleLogger", "TensorboardLogger", "WandbLogger"]


def _normalize_logger_override(raw: str | None) -> Optional[str]:
    if raw is None:
        return None
    name = str(raw).strip()
    if not name or name.lower() in {"default", "config"}:
        return None
    if name.lower() in {"none", "null", "off", "disable", "disabled"}:
        return "none"
    try:
        import importlib

        mod = importlib.import_module("opto.trainer.loggers")
        if hasattr(mod, name):
            return name
    except Exception:
        pass
    print(f"[WARN] Unknown logger override '{name}'. Keeping config/default logger.")
    return None


def _load_uploaded_config_to_editor(uploaded) -> str:
    if uploaded is None:
        return ""
    path = getattr(uploaded, "name", None) or str(uploaded)
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception as exc:
        return f"# Error reading uploaded config: {exc}"


def _apply_overrides(
    cfg: RunConfig,
    mode: str,
    max_workers: int,
    resume: str,
    job_timeout: float,
    force: bool,
    logger_override: str | None = None,
) -> Tuple[RunConfig, bool]:
    rerun_all = bool(force)
    validated_workers = _validate_non_negative("max_workers", max_workers)
    if validated_workers is not None and not float(validated_workers).is_integer():
        raise ValueError("max_workers must be an integer >= 0")
    validated_timeout = _validate_non_negative("job_timeout", job_timeout)
    if mode:
        cfg.mode = mode
    if validated_workers and int(validated_workers) > 0:
        cfg.max_workers = int(validated_workers)
    if resume:
        cfg.resume = resume
    if validated_timeout and float(validated_timeout) > 0:
        cfg.job_timeout = float(validated_timeout)
    override = _normalize_logger_override(logger_override)
    if override is not None:
        for trainer in cfg.trainers:
            trainer.logger = override
    return cfg, rerun_all


def _tb_command(run_dir: str) -> str:
    return f"tensorboard --logdir \"{Path(run_dir) / 'jobs'}\" --bind_all"


def _try_launch_tensorboard(run_dir: str) -> str:
    cmd = _tb_command(run_dir)
    default_url = "http://localhost:6006"
    try:
        subprocess.Popen(cmd, shell=True)  # noqa: S603,S607
        return (
            f"TensorBoard started. Open: {default_url}\n"
            f"If not reachable from your environment, run:\n{cmd}"
        )
    except Exception as exc:
        return (
            f"Could not launch TensorBoard automatically ({exc}).\n"
            f"Run this command manually:\n{cmd}"
        )


def _validate_non_negative(name: str, value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    numeric = float(value)
    if numeric < 0:
        raise ValueError(f"{name} must be >= 0")
    return numeric


def _cell_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    except Exception:
        return str(value)


def _rows_to_table(rows: List[Dict[str, Any]]) -> Tuple[List[str], List[List[str]]]:
    headers: List[str] = []
    seen = set()
    for row in rows or []:
        for key in row.keys():
            if key not in seen:
                headers.append(str(key))
                seen.add(key)
    data: List[List[str]] = []
    for row in rows or []:
        data.append([_cell_to_text(row.get(h, "")) for h in headers])
    return headers, data


def _dropdown_choices(rows: List[Dict[str, Any]], column: str) -> List[str]:
    values = {
        _cell_to_text(r.get(column, "")).strip()
        for r in (rows or [])
        if _cell_to_text(r.get(column, "")).strip()
    }
    return [""] + sorted(values)


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
        return "\n".join(lines), gr.CheckboxGroup(choices=ids, value=[])
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
        return "\n".join(lines), gr.CheckboxGroup(choices=ids, value=[])
    except Exception as exc:
        return f"Error discovering trainers: {exc}", gr.CheckboxGroup(choices=[], value=[])


def _default_params_for_trainer(trainer_id: str) -> Dict[str, Any]:
    if trainer_id == "PrioritySearch":
        return {"ps_steps": 1, "ps_batches": 1, "ps_candidates": 2, "ps_proposals": 2}
    if trainer_id.startswith("GEPA"):
        return {"gepa_iters": 1, "gepa_train_bs": 2}
    return {}


def _compose_editor_config(mode: str, selected_tasks: List[str], selected_trainers: List[str], provider: str, api_base: str, model_name: str, verbose: bool) -> str:
    if not selected_tasks:
        return "# Select at least one task first."
    if not selected_trainers:
        return "# Select at least one trainer first."
    base, _key, model = _provider_defaults(provider, api_base, model_name)
    payload = {
        "mode": mode or "real",
        "seeds": [123],
        "max_workers": 2,
        "resume": "auto",
        "llm": {
            "provider": provider,
            "base_url": base,
            "model": model,
            "api_key_env": "OPENROUTER_API_KEY" if provider == "openrouter" else "OPENAI_API_KEY",
        },
        "tags": ["ui-generated"],
        "trainer_kwargs": {"verbose": bool(verbose)},
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


def _apply_llm_env(provider: str, api_base: str, api_key: str, model_name: str) -> None:
    """Apply optional runtime LLM settings from UI controls."""
    provider = (provider or "custom").lower()
    if api_key and api_key.strip():
        if provider == "openrouter":
            os.environ["OPENROUTER_API_KEY"] = api_key.strip()
            os.environ["OPENAI_API_KEY"] = api_key.strip()
        elif provider == "openai":
            os.environ["OPENAI_API_KEY"] = api_key.strip()
        else:
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
    logger_override: str | None = None,
) -> Tuple[str, str]:
    """Run a config and return (log_message, run_id)."""
    if runs_dir_text:
        cfg.runs_dir = runs_dir_text
    cfg, rerun_all = _apply_overrides(cfg, mode, max_workers, resume, job_timeout, force, logger_override)
    runner = BenchRunner(cfg, tasks_root=tasks_root, job_timeout=cfg.job_timeout)
    summary = runner.run(force=rerun_all)
    return f"Run complete: {summary.run_id} ({len(summary.results)} results)", summary.run_id


def _unified_run(
    runs_dir_text: str,
    tasks_root: str,
    editor_text: str,
    uploaded,
    config_path: str,
    config_source: str = "",
    mode: str = "",
    max_workers: int = 0,
    resume: str = "",
    job_timeout: float = 0,
    force: bool = False,
    logger_override: str | None = None,
    provider: str = "custom",
    api_base: str = "",
    api_key: str = "",
    model_name: str = "",
) -> Tuple[str, str]:
    """Single run handler with explicit source selection."""
    try:
        _apply_llm_env(provider, api_base, api_key, model_name)
        source = (config_source or "").lower()

        # Backward-compatible priority mode (legacy callers/tests):
        # editor > upload > picker when source is omitted.
        if source in {"", "auto"}:
            if editor_text and editor_text.strip():
                source = "editor"
            elif uploaded is not None:
                source = "upload"
            elif config_path:
                source = "picker"
            else:
                return "No config provided. Use the editor, upload a file, or select from configs.", ""

        if source == "editor":
            if not editor_text or not editor_text.strip():
                return "No config provided (editor is empty).", ""
            try:
                import yaml
                data = yaml.safe_load(editor_text)
            except Exception:
                data = json.loads(editor_text)
            if not isinstance(data, dict):
                return "Config must be a YAML/JSON mapping.", ""
            cfg = RunConfig.from_dict(data)
            return _run_config(cfg, runs_dir_text, tasks_root, mode, max_workers, resume, job_timeout, force, logger_override)

        if source == "upload":
            if uploaded is None:
                return "No config uploaded.", ""
            path = getattr(uploaded, "name", None) or str(uploaded)
            cfg = load_config(path)
            return _run_config(cfg, runs_dir_text, tasks_root, mode, max_workers, resume, job_timeout, force, logger_override)

        if source == "picker":
            if not config_path:
                return "No config selected in picker.", ""
            cfg = load_config(config_path)
            return _run_config(cfg, runs_dir_text, tasks_root, mode, max_workers, resume, job_timeout, force, logger_override)

        return "No config provided. Use the editor, upload a file, or select from configs.", ""
    except ValueError as exc:
        return f"Invalid input: {exc}", ""
    except Exception as exc:
        return f"Run failed: {exc}\n{traceback.format_exc()}", ""


# ---------------------------------------------------------------------------
# Resume handler
# ---------------------------------------------------------------------------


def _resume_run(
    runs_dir_text: str,
    tasks_root: str,
    selected_run: str = "",
    new_run_id: bool = False,
    run_id: str | None = None,
) -> Tuple[str, str]:
    """Resume a previous run using its config snapshot."""
    # Backward compatibility for older callers/tests that pass `run_id=...`.
    if not selected_run and run_id:
        selected_run = run_id
    if not selected_run:
        return "No run selected to resume.", ""
    try:
        run_path = Path(selected_run)
        if not run_path.exists():
            run_path = Path(runs_dir_text) / selected_run
        run_id = run_path.name
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
        choices = [(r.label or r.run_id, str(r.run_dir)) for r in recs]
        return gr.Dropdown(choices=choices, value=(choices[0][1] if choices else None))

    def _load_selected_run(runs_dir_text: str, selected_run: str):
        if not selected_run:
            return "", [], "", "{}", "{}", "", [], "{}"
        run_path = Path(selected_run)
        if not run_path.exists():
            run_path = Path(runs_dir_text) / selected_run
        d = load_run_summary(run_path)
        return (
            d["config_text"],
            d["results_rows"],
            d["env_text"],
            json.dumps(d["summary"], indent=2, ensure_ascii=False),
            json.dumps(d["manifest"], indent=2, ensure_ascii=False),
            _tb_command(str(run_path)),
            d.get("leaderboard_rows", []),
            d.get("files_index_text", "{}"),
        )

    def _filter_rows(
        rows: List[Dict[str, Any]],
        suite: str,
        status: str,
        trainer_id: str,
        task_id: str,
    ):
        filtered = filter_results_rows(rows, suite=suite, status=status, trainer_id=trainer_id, task_substring="")
        if not task_id:
            return filtered
        return [row for row in filtered if _cell_to_text(row.get("task_id", "")) == task_id]

    def _job_ids_from_rows(rows: List[Dict[str, Any]]) -> List[str]:
        ids: List[str] = []
        for r in rows or []:
            jid = r.get("job_id")
            if jid:
                ids.append(str(jid))
        return ids

    def _load_job(runs_dir_text: str, selected_run: str, job_id: str):
        if not selected_run or not job_id:
            return "", "{}", "{}", "", "", "", "", "", "", "", ""
        run_path = Path(selected_run)
        if not run_path.exists():
            run_path = Path(runs_dir_text) / selected_run
        d = load_job_details(run_path, job_id)
        return (
            d["job_dir"],
            json.dumps(d["job_meta"], indent=2, ensure_ascii=False),
            json.dumps(d["job_results"], indent=2, ensure_ascii=False),
            d["events_head"],
            d["tb_dir"],
            d.get("stdout_tail", ""),
            d.get("initial_state_yaml", json.dumps(d.get("initial_state", {}), indent=2, ensure_ascii=False)),
            d.get("best_state_yaml", json.dumps(d.get("best_state", {}), indent=2, ensure_ascii=False)),
            d.get("final_state_yaml", json.dumps(d.get("final_state", {}), indent=2, ensure_ascii=False)),
            d.get("state_history_tail", ""),
            json.dumps(d.get("initial_state", {}), indent=2, ensure_ascii=False),
        )

    def _load_and_filter(runs_dir_text: str, selected_run: str, suite: str, status: str, trainer_id: str, task_sub: str):
        cfg_text, rows, env_text, summary, manifest, tb, leaderboard, files_index = _load_selected_run(runs_dir_text, selected_run)

        suite_choices = _dropdown_choices(rows, "suite")
        status_choices = _dropdown_choices(rows, "status")
        trainer_choices = _dropdown_choices(rows, "trainer_id")
        task_choices = _dropdown_choices(rows, "task_id")

        suite_selected = suite if suite in suite_choices else ""
        status_selected = status if status in status_choices else ""
        trainer_selected = trainer_id if trainer_id in trainer_choices else ""
        task_selected = task_sub if task_sub in task_choices else ""

        filtered = _filter_rows(rows, suite_selected, status_selected, trainer_selected, task_selected)
        headers, data = _rows_to_table(filtered)
        return (
            cfg_text,
            gr.Dataframe(headers=headers, value=data),
            env_text,
            summary,
            manifest,
            tb,
            gr.Dataframe(value=leaderboard),
            files_index,
            gr.Dropdown(choices=suite_choices, value=suite_selected),
            gr.Dropdown(choices=status_choices, value=status_selected),
            gr.Dropdown(choices=trainer_choices, value=trainer_selected),
            gr.Dropdown(choices=task_choices, value=task_selected),
        )

    def _load_jobs(runs_dir_text: str, selected_run: str):
        if not selected_run:
            recs = discover_runs(runs_dir_text)
            if not recs:
                return gr.Dropdown(choices=[], value=None)
            selected_run = str(recs[0].run_dir)
        run_path = Path(selected_run)
        if not run_path.exists():
            run_path = Path(runs_dir_text) / selected_run
        d = load_run_summary(run_path)
        ids = _job_ids_from_rows(d["results_rows"])
        return gr.Dropdown(choices=ids, value=(ids[0] if ids else None))

    def _init_launch_state(tasks_root: str, bench: str, configs_dir: str, provider: str, current_base: str, current_model: str):
        tasks_text, tasks_update = _discover_tasks_for_ui(tasks_root, bench)
        trainers_text, trainers_update = _discover_trainers_for_ui()
        configs = _list_configs(configs_dir)
        first = configs[0] if configs else None
        editor_text = _load_config_to_editor(first) if first else ""
        base, key, model = _provider_defaults(provider, current_base, current_model)
        combined = "\n".join([x for x in [tasks_text, "", trainers_text] if x])
        return (
            combined,
            tasks_update,
            trainers_update,
            gr.Dropdown(choices=configs, value=first),
            editor_text,
            base,
            key,
            model,
        )

    # --- Build UI ---

    ui_css = """
    .tb-shell {border: 1px solid #c7d2e0; border-radius: 10px; padding: 8px; background: linear-gradient(180deg, #f8fbff 0%, #f3f6fb 100%);}
    .tb-left {border-right: 1px solid #d6deea; padding-right: 8px;}
    .tb-title {font-weight: 700; margin: 0 0 4px 0;}
    /* Keep top-level tabs visible while scrolling long content */
    .gradio-container [role="tablist"] {
        position: sticky;
        top: 0;
        z-index: 20;
        background: #f8fbff;
        border-bottom: 1px solid #d6deea;
        padding: 4px 0;
    }
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
                            provider = gr.Dropdown(
                                choices=["custom", "openai", "openrouter"],
                                value="openrouter",
                                label="provider",
                            )
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
                        config_source = gr.Radio(
                            choices=["picker", "upload", "editor"],
                            value="picker",
                            label="config source",
                        )
                        config_picker = gr.Dropdown(choices=initial_configs, label="config picker (select from list)")
                        config_upload = gr.File(label="upload config (YAML/JSON)")

                        config_editor = gr.Code(
                            label="Config editor",
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
                            mode = gr.Dropdown(choices=["", "stub", "real"], value="real", label="mode override")
                            max_workers = gr.Number(value=0, label="max_workers (0=keep)", minimum=0, precision=0)
                            resume = gr.Dropdown(choices=["", "auto", "failed", "none"], value="", label="resume")
                            logger_override = gr.Dropdown(
                                choices=_logger_choices(),
                                value="default",
                                label="logger override",
                            )
                        with gr.Row():
                            verbose_trainer = gr.Checkbox(value=False, label="verbose trainer/log output")
                        with gr.Row():
                            job_timeout = gr.Number(
                                value=0, label="job_timeout seconds (0=keep)", minimum=0, precision=0
                            )
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
                        return "{}", gr.Dataframe(headers=[], value=[])
                    d = load_run_summary(recs[0].run_dir)
                    headers, data = _rows_to_table(d["results_rows"][:20])
                    return json.dumps(d["summary"], indent=2), gr.Dataframe(headers=headers, value=data)

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
                    inputs=[mode, selected_tasks, selected_trainers, provider, api_base, model_name, verbose_trainer],
                    outputs=[config_editor],
                )
                refresh_configs_btn.click(_refresh_configs, inputs=[configs_dir_box], outputs=[config_picker])
                config_picker.change(_load_config_to_editor, inputs=[config_picker], outputs=[config_editor])
                config_upload.change(_load_uploaded_config_to_editor, inputs=[config_upload], outputs=[config_editor])
                save_btn.click(_save_config, inputs=[config_editor, save_path_box], outputs=[save_log])
                provider.change(
                    _provider_defaults,
                    inputs=[provider, api_base, model_name],
                    outputs=[api_base, api_key, model_name],
                )
                run_btn.click(
                    _unified_run,
                    inputs=[
                        runs_dir_text, tasks_root_box, config_editor, config_upload, config_picker,
                        config_source, mode, max_workers, resume, job_timeout, force, logger_override,
                        provider, api_base, api_key, model_name,
                    ],
                    outputs=[run_log, last_run_id],
                )
                stop_btn.click(lambda: "Use notebook interrupt to stop current process.", outputs=[run_log])
                bench_selector.change(
                    _discover_tasks_for_ui, inputs=[tasks_root_box, bench_selector], outputs=[discovery_output, selected_tasks]
                )
                refresh_latest_btn.click(_latest_run_view, inputs=[runs_dir_text], outputs=[latest_summary, latest_results])
                demo.load(
                    _init_launch_state,
                    inputs=[tasks_root_box, bench_selector, configs_dir_box, provider, api_base, model_name],
                    outputs=[discovery_output, selected_tasks, selected_trainers, config_picker, config_editor, api_base, api_key, model_name],
                )

            # ========== TAB 2: Browse Runs ==========
            with gr.Tab("Browse Runs"):
                runs_dir_text_b = gr.Textbox(value=initial_runs_dir, label="runs_dir")
                with gr.Row():
                    refresh_btn = gr.Button("Refresh runs")
                    run_selector = gr.Dropdown(
                        choices=[(r.label or r.run_id, str(r.run_dir)) for r in discover_runs(initial_runs_dir)],
                        label="Select run from list",
                    )

                with gr.Accordion("Config & metadata", open=False):
                    config_box = gr.Code(label="config.snapshot.yaml", language="yaml")
                    env_box = gr.Code(label="env.json", language="json")
                    summary_box = gr.Code(label="summary.json", language="json")
                    manifest_box = gr.Code(label="meta/manifest.json", language="json")
                    files_index_box = gr.Code(label="meta/files_index.json", language="json")

                gr.Markdown("#### Filters")
                with gr.Row():
                    suite_filter = gr.Dropdown(choices=[""], value="", label="suite (exact)")
                    status_filter = gr.Dropdown(choices=[""], value="", label="status (exact)")
                    trainer_filter = gr.Dropdown(choices=[""], value="", label="trainer_id (exact)")
                    task_filter = gr.Dropdown(choices=[""], value="", label="task_id (exact)")

                results_df = gr.Dataframe(label="results.csv (filtered)")
                leaderboard_df = gr.Dataframe(label="leaderboard.csv")

                gr.Markdown("#### TensorBoard")
                tb_cmd = gr.Textbox(label="TensorBoard command", lines=2)
                with gr.Row():
                    tb_launch_btn = gr.Button("Start TensorBoard + show URL/command")
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
                    outputs=[
                        config_box,
                        results_df,
                        env_box,
                        summary_box,
                        manifest_box,
                        tb_cmd,
                        leaderboard_df,
                        files_index_box,
                        suite_filter,
                        status_filter,
                        trainer_filter,
                        task_filter,
                    ],
                )
                for widget in (suite_filter, status_filter, trainer_filter, task_filter):
                    widget.change(
                        _load_and_filter,
                        inputs=[runs_dir_text_b, run_selector, suite_filter, status_filter, trainer_filter, task_filter],
                        outputs=[
                            config_box,
                            results_df,
                            env_box,
                            summary_box,
                            manifest_box,
                            tb_cmd,
                            leaderboard_df,
                            files_index_box,
                            suite_filter,
                            status_filter,
                            trainer_filter,
                            task_filter,
                        ],
                    )

                tb_launch_btn.click(
                    lambda selected_run, d: _try_launch_tensorboard(str(Path(selected_run) if selected_run else Path(d))),
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
                with gr.Row():
                    refresh_runs_j_btn = gr.Button("Refresh runs")
                    run_selector_j = gr.Dropdown(
                        choices=[(r.label or r.run_id, str(r.run_dir)) for r in discover_runs(initial_runs_dir)],
                        label="Select run from list",
                    )
                load_jobs_btn = gr.Button("Load job list")
                job_selector = gr.Dropdown(choices=[], label="Select job from list")

                job_dir = gr.Textbox(label="job_dir", interactive=False)
                job_meta = gr.Code(label="job_meta.json", language="json")
                job_results = gr.Code(label="results.json", language="json")
                events_head = gr.Code(label="events.jsonl (head)", language="json")
                stdout_tail = gr.Code(label="stdout.log (tail)", language="text")
                initial_state = gr.Code(label="initial_state.yaml", language="yaml")
                best_state = gr.Code(label="best_state.yaml", language="yaml")
                final_state = gr.Code(label="final_state.yaml", language="yaml")
                state_history = gr.Code(label="state_history.jsonl (tail)", language="json")
                initial_state_json = gr.Code(label="initial_state.json (raw)", language="json")
                tb_dir = gr.Textbox(label="tb_dir", interactive=False)

                refresh_runs_j_btn.click(_refresh_runs, inputs=[runs_dir_text_j], outputs=[run_selector_j])
                runs_dir_text_j.change(_refresh_runs, inputs=[runs_dir_text_j], outputs=[run_selector_j])
                run_selector_j.change(_load_jobs, inputs=[runs_dir_text_j, run_selector_j], outputs=[job_selector])
                load_jobs_btn.click(_load_jobs, inputs=[runs_dir_text_j, run_selector_j], outputs=[job_selector])
                job_selector.change(
                    _load_job,
                    inputs=[runs_dir_text_j, run_selector_j, job_selector],
                    outputs=[job_dir, job_meta, job_results, events_head, tb_dir, stdout_tail, initial_state, best_state, final_state, state_history, initial_state_json],
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
