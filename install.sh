#!/usr/bin/env bash
# install.sh — One-line setup for XNAV Cold Start Simulator
#
# Usage:
#   bash install.sh          # install and launch
#   bash install.sh --no-run # install only, don't launch
#
# Requirements: Python 3.11+, git (already cloned)
# Tested on: macOS 13+, Ubuntu 22.04+

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_ROOT/.venv"
APP="$REPO_ROOT/xnav_simulator/app.py"
REQUIREMENTS="$REPO_ROOT/xnav_simulator/requirements.txt"
LAUNCH=true

# ── Argument handling ──────────────────────────────────────────────────────────
for arg in "$@"; do
  case $arg in
    --no-run) LAUNCH=false ;;
    -h|--help)
      echo "Usage: bash install.sh [--no-run]"
      echo "  --no-run   Install dependencies but do not launch the app"
      exit 0 ;;
  esac
done

# ── Colour helpers ─────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()    { echo -e "${CYAN}▸${NC} $*"; }
success() { echo -e "${GREEN}✓${NC} $*"; }
error()   { echo -e "${RED}✗ ERROR:${NC} $*" >&2; exit 1; }

echo ""
echo -e "${BOLD}  XNAV Cold Start Simulator — Setup${NC}"
echo "  ─────────────────────────────────"
echo ""

# ── Python version check ───────────────────────────────────────────────────────
info "Checking Python version..."

PYTHON=""
for candidate in python3.12 python3.11 python3; do
  if command -v "$candidate" &>/dev/null; then
    ver=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    major=$(echo "$ver" | cut -d. -f1)
    minor=$(echo "$ver" | cut -d. -f2)
    if [ "$major" -eq 3 ] && [ "$minor" -ge 11 ]; then
      PYTHON="$candidate"
      break
    fi
  fi
done

if [ -z "$PYTHON" ]; then
  error "Python 3.11 or later is required but was not found.
       Install it from https://www.python.org/downloads/ then re-run this script."
fi

PYVER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
success "Found Python $PYVER at $(command -v "$PYTHON")"

# ── Virtual environment ────────────────────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
  info "Virtual environment already exists at .venv/ — skipping creation"
else
  info "Creating virtual environment at .venv/ ..."
  "$PYTHON" -m venv "$VENV_DIR"
  success "Virtual environment created"
fi

# Activate
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
success "Virtual environment activated"

# ── Dependencies ───────────────────────────────────────────────────────────────
info "Installing dependencies from requirements.txt ..."
pip install --quiet --upgrade pip
pip install --quiet -r "$REQUIREMENTS"
success "All dependencies installed"

# ── Numba cache warm-up (optional, non-fatal) ─────────────────────────────────
info "Pre-compiling Numba JIT functions (first-run warm-up) ..."
python -c "
import warnings; warnings.filterwarnings('ignore')
import sys; sys.path.insert(0, '$(dirname "$APP")')
try:
    from core.estimator import _systematic_resample_nb
    import numpy as np
    _systematic_resample_nb(np.ones(10)/10)
    print('  Numba warm-up: OK')
except Exception as e:
    print(f'  Numba warm-up skipped: {e}')
" || true
success "Pre-compilation done"

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}  Installation complete!${NC}"
echo ""
echo "  To run the app manually in future:"
echo "    source .venv/bin/activate"
echo "    streamlit run xnav_simulator/app.py"
echo ""

# ── Launch ─────────────────────────────────────────────────────────────────────
if [ "$LAUNCH" = true ]; then
  info "Launching XNAV Simulator at http://localhost:8501 ..."
  echo ""
  streamlit run "$APP"
fi
