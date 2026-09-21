#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export METEOR_DETECTOR_ROOT="${METEOR_DETECTOR_ROOT:-$ROOT_DIR}"
if [[ -z "${METEOR_PYTHON:-}" ]]; then
  if [[ -x "$ROOT_DIR/.venv-mac/bin/python" ]]; then
    export METEOR_PYTHON="$ROOT_DIR/.venv-mac/bin/python"
  elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    export METEOR_PYTHON="$ROOT_DIR/.venv/bin/python"
  fi
fi

exec swift run --package-path "$ROOT_DIR/swiftui" MeteorDetectorSwiftUI "$@"
