#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.11)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "Python 3 is not installed. Install Python 3.11, then run this script again." >&2
    exit 1
  fi
fi

if [[ ! -d .venv-mac ]]; then
  "$PYTHON_BIN" -m venv .venv-mac
fi

source .venv-mac/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-mac.txt

mkdir -p meteor not_meteor temp_clips rtsp lighten_blend_cache
python main_gui.py
