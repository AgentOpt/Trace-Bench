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

## Defer Python dependency installation until after external deps and toolchains are ready

# Ensure PyPantograph is available in a repo-local directory (not in $HOME)
command -v git >/dev/null 2>&1 || { echo "git is required"; exit 1; }

EXTERNAL_DIR=${EXTERNAL_DIR:-"$REPO_ROOT/external"}
PYPANTO_DIR=${PYPANTO_DIR:-"$EXTERNAL_DIR/PyPantograph_Kai"}
REPO_URL=${REPO_URL:-"https://github.com/kaifronsdal/PyPantograph.git"}

mkdir -p "$EXTERNAL_DIR"

if [ -d "$PYPANTO_DIR/.git" ]; then
  echo "Updating PyPantograph in $PYPANTO_DIR"
  if ! (
    git -C "$PYPANTO_DIR" fetch --prune
    git -C "$PYPANTO_DIR" pull --ff-only
    git -C "$PYPANTO_DIR" submodule sync --recursive
    git -C "$PYPANTO_DIR" submodule update --init --recursive
  ); then
    echo "PyPantograph update failed; recloning..."
    rm -rf "$PYPANTO_DIR"
    git clone --recurse-submodules "$REPO_URL" "$PYPANTO_DIR"
  fi
else
  echo "Cloning PyPantograph into $PYPANTO_DIR"
  git clone --recurse-submodules "$REPO_URL" "$PYPANTO_DIR"
fi

# Setup Lean/elan/lake BEFORE building pantograph so 'lake' is available
curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y
# Ensure elan/lake are available in this shell
if [ -d "$HOME/.elan/bin" ]; then
  PATH="$HOME/.elan/bin:$PATH"
  export PATH
fi
# Update and set a default Lean toolchain
elan self update || true
elan default leanprover/lean4:v4.18.0 || true
elan --version || true
lean --version || true
lake --version || true

# Optional: verify PyPantograph toolchain and integration
cd "$PYPANTO_DIR"
if [ -f src/lean-toolchain ]; then
  TOOLCHAIN=$(cut -d ' ' -f1 src/lean-toolchain | tr -d '\n')
  echo "Detected PyPantograph src toolchain: $TOOLCHAIN"
  # Ensure the submodule's toolchain is installed and overridden for src/
  elan toolchain install "$TOOLCHAIN" || true
  elan override set --path "$PYPANTO_DIR/src" "$TOOLCHAIN" || true
  echo "lean/lake versions within $PYPANTO_DIR/src:"
  (cd "$PYPANTO_DIR/src" && lean --version || true)
  (cd "$PYPANTO_DIR/src" && lake --version || true)
else
  echo "No src/lean-toolchain found under $PYPANTO_DIR; skipping override"
fi
cd "$SCRIPT_DIR"

# Install base Python dependencies now that external deps exist
echo "Installing base Python dependencies with uv in $SCRIPT_DIR"
(cd "$SCRIPT_DIR" && "$UV_BIN" sync)
echo "Activating the Python environment at $SCRIPT_DIR/.venv"
. "$SCRIPT_DIR/.venv/bin/activate"

# Install pantograph via the lean4 extra (requires lake/toolchain ready)
echo "Installing lean4 extra (pantograph) with uv in $SCRIPT_DIR"
(cd "$SCRIPT_DIR" && "$UV_BIN" sync --extra lean4)

# check that pypantograph is installed correctly
python -c "from pantograph import Server; server = Server(imports=['Init']); print(server)"
python3 -m pantograph.server