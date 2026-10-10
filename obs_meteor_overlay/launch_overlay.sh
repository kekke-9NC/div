#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$BASE_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv-mac/bin/python}"
FFMPEG_BIN="${FFMPEG_BIN:-/opt/homebrew/bin/ffmpeg}"
PID_FILE="$BASE_DIR/overlay.pid"
LOG_FILE="$BASE_DIR/overlay.log"
FOREGROUND=false

if [[ "${1:-}" == "--foreground" ]]; then
  FOREGROUND=true
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi
if [[ ! -x "$FFMPEG_BIN" ]]; then
  FFMPEG_BIN="$(command -v ffmpeg)"
fi

overlay_process_running() {
  while read -r pid command; do
    if [[ "$pid" != "$$" && "$command" == *"$BASE_DIR/latest_meteor_gif.py"* ]]; then
      echo "$pid" > "$PID_FILE"
      return 0
    fi
  done < <(ps -axo pid=,command=)
  return 1
}

if overlay_process_running; then
  echo "OBS meteor overlay is already running."
  exit 0
fi

if $FOREGROUND; then
  echo "$$" > "$PID_FILE"
  exec "$PYTHON_BIN" "$BASE_DIR/latest_meteor_gif.py" \
    --meteor-dir "$PROJECT_DIR/meteor" \
    --output-dir "$BASE_DIR" \
    --ffmpeg "$FFMPEG_BIN"
fi

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(<"$PID_FILE")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] &&
      ps -p "$old_pid" -o command= | grep -Fq "$BASE_DIR/latest_meteor_gif.py"; then
    echo "OBS meteor overlay is already running (PID $old_pid)."
    exit 0
  fi
  rm -f "$PID_FILE"
fi

nohup "$PYTHON_BIN" "$BASE_DIR/latest_meteor_gif.py" \
  --meteor-dir "$PROJECT_DIR/meteor" \
  --output-dir "$BASE_DIR" \
  --ffmpeg "$FFMPEG_BIN" \
  >> "$LOG_FILE" 2>&1 &
pid=$!
echo "$pid" > "$PID_FILE"
echo "Started OBS meteor overlay (PID $pid)."
