from __future__ import annotations

from pathlib import Path
import csv
import json


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _read_csv(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def launch_ui(runs_dir: str) -> int:
    try:
        import gradio as gr
    except Exception:
        print("Gradio is not installed. Install with: pip install gradio")
        return 1

    runs_root = Path(runs_dir)
    runs = sorted([p.name for p in runs_root.iterdir() if p.is_dir()]) if runs_root.exists() else []

    def load_run(run_id: str):
        run_path = runs_root / run_id
        config_text = _read_text(run_path / "meta" / "config.snapshot.yaml")
        results = _read_csv(run_path / "results.csv")
        env_text = _read_text(run_path / "meta" / "env.json")
        return config_text, results, env_text

    with gr.Blocks() as demo:
        gr.Markdown("# Trace-Bench UI (Stub)")
        gr.Markdown("Select a run to view config, results, and env info.")
        run_selector = gr.Dropdown(choices=runs, label="Run ID")
        config_box = gr.Code(label="config.snapshot.yaml", language="yaml")
        results_df = gr.Dataframe(label="results.csv")
        env_box = gr.Code(label="env.json", language="json")

        run_selector.change(load_run, inputs=run_selector, outputs=[config_box, results_df, env_box])

        try:
            import mlflow  # noqa: F401
            gr.Markdown("MLflow detected. Full integration is pending (M3).")
        except Exception:
            gr.Markdown("MLflow not installed. Install if you want UI-linked runs.")

    demo.launch()
    return 0


__all__ = ["launch_ui"]
