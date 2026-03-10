# Screenshot Manifest

This file lists all screenshots referenced in the documentation.
Each screenshot should be saved as a PNG file in `docs/assets/`.

## How to capture

1. Launch the UI: `trace-bench ui --runs-dir runs`
2. Complete at least one run so Browse Runs and Job Inspector have data.
3. Capture each screenshot according to the table below.
4. Save files in this directory (`docs/assets/`) with the exact filenames.

## Screenshot list

| Filename | What to Capture | Used In |
|----------|----------------|---------|
| `ui-launch-tab.png` | Launch Run tab with config dropdown, provider selector, and Run button visible | [ui-guide.md](../ui-guide.md) |
| `ui-launch-config-editor.png` | YAML config editor with a loaded config showing tasks and trainers | [ui-guide.md](../ui-guide.md) |
| `ui-browse-runs-list.png` | Browse Runs tab showing the run selector dropdown with at least one completed run | [ui-guide.md](../ui-guide.md) |
| `ui-results-table.png` | Results table (results.csv filtered) with suite/status/trainer filter dropdowns visible | [ui-guide.md](../ui-guide.md) |
| `ui-leaderboard.png` | Leaderboard table showing ranked scores per task with trainer and timing columns | [ui-guide.md](../ui-guide.md) |
| `ui-job-inspector.png` | Job Inspector tab with a job selected, showing job_meta.json and events.jsonl panels | [ui-guide.md](../ui-guide.md) |
| `ui-job-state.png` | Job Inspector scrolled to the state snapshot section showing initial_state.yaml and best_state.yaml | [ui-guide.md](../ui-guide.md) |

## Recommended capture settings

- Browser window width: 1280px or wider.
- Use a light theme (Gradio default).
- Crop to the tab content area; include the tab bar for context.
- Preferred format: PNG, lossless.
- If capturing on Colab, use the browser's built-in screenshot tool or a
  screen capture extension.
