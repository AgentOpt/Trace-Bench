#!/usr/bin/env sh
set -eu

# Determine repository root (parent of this script)
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

# Ensure uv is installed
if command -v uv >/dev/null 2>&1; then
  echo "uv is already installed: $(command -v uv)"
else
  echo "uv not found; installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Ensure uv is available on PATH for this shell session
if ! command -v uv >/dev/null 2>&1; then
  if [ -f "$HOME/.local/bin/env" ]; then
    . "$HOME/.local/bin/env"
  elif [ -x "$HOME/.local/bin/uv" ]; then
    PATH="$HOME/.local/bin:$PATH"
    export PATH
  fi
  # refresh shell command hash, if supported
  hash -r 2>/dev/null || true
fi

# Locate uv binary (absolute path fallback)
if command -v uv >/dev/null 2>&1; then
  UV_BIN=$(command -v uv)
elif [ -x "$HOME/.local/bin/uv" ]; then
  UV_BIN="$HOME/.local/bin/uv"
else
  echo "uv not found after installation; ensure $HOME/.local/bin is on PATH"
  exit 1
fi

# Prepare external directory and clone KernelBench there
command -v git >/dev/null 2>&1 || { echo "git is required"; exit 1; }

EXTERNAL_DIR=${EXTERNAL_DIR:-"$REPO_ROOT/external"}
KERNELBENCH_DIR=${KERNELBENCH_DIR:-"$EXTERNAL_DIR/KernelBench"}
KERNELBENCH_URL=${KERNELBENCH_URL:-"https://github.com/ScalingIntelligence/KernelBench.git"}

mkdir -p "$EXTERNAL_DIR"

if [ -d "$KERNELBENCH_DIR/.git" ]; then
  echo "Updating KernelBench in $KERNELBENCH_DIR"
  if ! (
    git -C "$KERNELBENCH_DIR" fetch --prune
    git -C "$KERNELBENCH_DIR" pull --ff-only
  ); then
    echo "KernelBench update failed; recloning..."
    rm -rf "$KERNELBENCH_DIR"
    git clone "$KERNELBENCH_URL" "$KERNELBENCH_DIR"
  fi
else
  echo "Cloning KernelBench into $KERNELBENCH_DIR"
  git clone "$KERNELBENCH_URL" "$KERNELBENCH_DIR"
fi

# Install Python dependencies for this task using uv
echo "Installing Python dependencies with uv in $SCRIPT_DIR"
(cd "$SCRIPT_DIR" && "$UV_BIN" sync)
echo "Activating the Python environment at $SCRIPT_DIR/.venv"
. "$SCRIPT_DIR/.venv/bin/activate"

# Use uv to install upstream KernelBench as an editable package if it provides pyproject.toml or setup.py
if [ -f "$KERNELBENCH_DIR/pyproject.toml" ] || [ -f "$KERNELBENCH_DIR/setup.py" ]; then
  echo "Installing/updating upstream KernelBench package in editable mode with uv"
  "$UV_BIN" pip install -e "$KERNELBENCH_DIR"
fi

# If KernelBench has a requirements.txt, install those requirements explicitly.
if [ -f "$KERNELBENCH_DIR/requirements.txt" ]; then
  echo "Installing KernelBench requirements.txt into the uv environment"
  "$UV_BIN" pip install -r "$KERNELBENCH_DIR/requirements.txt"
fi

echo "KernelBench setup complete."

ACTIVATE_CMD="source \"$SCRIPT_DIR/.venv/bin/activate\""
echo "To activate this environment later, run: $ACTIVATE_CMD"
