#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Finder から起動した場合でも Homebrew の ffmpeg を検出できるようにする。
export PATH="/opt/homebrew/bin:/usr/local/bin:${PATH}"

# Bound native pools before NumPy/OpenCV/PyTorch are imported.  Without this,
# each library may create one thread per logical core; multiple H.264 jobs then
# multiply that count and can destabilize Tcl/Tk and the native video backend.
export METEOR_NATIVE_THREADS="${METEOR_NATIVE_THREADS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$METEOR_NATIVE_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$METEOR_NATIVE_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$METEOR_NATIVE_THREADS}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-$METEOR_NATIVE_THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$METEOR_NATIVE_THREADS}"
export OPENCV_FOR_THREADS_NUM="${OPENCV_FOR_THREADS_NUM:-$METEOR_NATIVE_THREADS}"
export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-PASSIVE}"
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

PYTHON_RUNNER=""
if [[ -n "${METEOR_PYTHON:-}" ]]; then
  if [[ -x "${METEOR_PYTHON}" ]]; then
    PYTHON_RUNNER="$METEOR_PYTHON"
  elif command -v "$METEOR_PYTHON" >/dev/null 2>&1; then
    PYTHON_RUNNER="$(command -v "$METEOR_PYTHON")"
  else
    echo "METEOR_PYTHON is not executable; falling back to the managed macOS environment." >&2
  fi
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_RUNNER" && -z "$PYTHON_BIN" ]]; then
  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.11)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "Python 3 is not installed. Install Python 3.11, then run this script again." >&2
    exit 1
  fi
fi

if [[ -z "$PYTHON_RUNNER" && ! -d .venv-mac ]]; then
  "$PYTHON_BIN" -m venv .venv-mac
fi

if [[ -z "$PYTHON_RUNNER" ]]; then
  source .venv-mac/bin/activate
  PYTHON_RUNNER="$(command -v python)"
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install -r requirements-mac.txt
fi

mkdir -p meteor not_meteor temp_clips rtsp lighten_blend_cache

export METEOR_DETECTOR_ROOT="$(pwd)"
export METEOR_PYTHON="$PYTHON_RUNNER"

if [[ "${METEOR_UI_MODE:-swiftui}" == "legacy" ]]; then
  exec "$PYTHON_RUNNER" main_gui.py "$@"
fi

exec swift run --package-path swiftui MeteorDetectorSwiftUI "$@"
