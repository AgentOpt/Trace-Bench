#!/usr/bin/env sh
set -eu

# Determine repository root (parent of this script)
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

echo "=== Veribench Uninstall Script ==="
echo "This will remove:"
echo "  - Virtual environment (.venv)"
echo "  - External dependencies (../external, includes PyPantograph)"
echo "  - Build artifacts (*.egg-info, __pycache__, etc.)"
echo ""
echo "Note: This will NOT remove Lean/elan toolchains from ~/.elan"
echo ""
read -p "Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "Uninstall cancelled."
  exit 0
fi

# Remove virtual environment
if [ -d "$SCRIPT_DIR/.venv" ]; then
  echo "Removing virtual environment at $SCRIPT_DIR/.venv"
  rm -rf "$SCRIPT_DIR/.venv"
else
  echo "No virtual environment found at $SCRIPT_DIR/.venv"
fi

# Remove external directory (shared with other benchmarks, includes PyPantograph)
EXTERNAL_DIR=${EXTERNAL_DIR:-"$REPO_ROOT/external"}
if [ -d "$EXTERNAL_DIR" ]; then
  echo "Removing external dependencies at $EXTERNAL_DIR"
  rm -rf "$EXTERNAL_DIR"
else
  echo "No external directory found at $EXTERNAL_DIR"
fi

# Remove egg-info and build artifacts in Veribench directory
echo "Removing build artifacts in $SCRIPT_DIR"
find "$SCRIPT_DIR" -depth -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
find "$SCRIPT_DIR" -depth -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$SCRIPT_DIR" -depth -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
find "$SCRIPT_DIR" -depth -type d -name "build" -exec rm -rf {} + 2>/dev/null || true
find "$SCRIPT_DIR" -depth -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true

# Remove uv cache (optional - uncomment if you want to clear uv cache)
# echo "Removing uv cache"
# uv cache clean

echo ""
echo "=== Uninstall Complete ==="
echo "Veribench environment has been cleaned up."
echo ""
echo "Note: This script does NOT uninstall:"
echo "  - uv (to uninstall: rm -rf ~/.local/bin/uv ~/.cargo/bin/uv)"
echo "  - elan/Lean toolchains (to uninstall: rm -rf ~/.elan)"
