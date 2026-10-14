"""Bounded, cancellable review preview decoding; never calls Tk."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import queue
import sys
import threading
import time

import cv2
from PIL import Image


class PreviewStream:
    def __init__(self, event, *, trace_id=None):
        self.event = Path(event)
        self.trace_id = trace_id
        self.messages = queue.Queue(maxsize=2)
        self.cancel = threading.Event()
        self.closed = threading.Event()
        self.playing = threading.Event()
        self.playing.set()
        self.thread = threading.Thread(target=self._run, name="MLReviewPreview", daemon=False)
        self.thread.start()

    def stop(self):
        self.cancel.set()
        self.playing.set()

    def _put(self, message):
        while not self.cancel.is_set():
            try:
                self.messages.put(message, timeout=.05)
                return True
            except queue.Full:
                pass
        return False

    def _trace(self, stage, **values):
        if self.trace_id is not None:
            record_latency(dict(stage=stage, preview_id=self.trace_id,
                                at=time.perf_counter(), **values))

    @staticmethod
    def _photo(path):
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
                image.thumbnail((400, 420), Image.Resampling.LANCZOS)
                return image
        except Exception:
            return None

    def _run(self):
        cap = None
        try:
            with (self.event / "metadata.json").open(encoding="utf-8") as source:
                metadata = json.load(source)
            if not isinstance(metadata, dict):
                raise ValueError("メタデータの形式が不正です。")
            diff = self._photo(self.event / "diff.png")
            temporal = self._photo(self.event / "temporal_rgb.png")
            if self.cancel.is_set():
                return
            if not self._put(("ready", (metadata, diff, temporal))):
                return
            self._trace("preview_metadata_queued")
            open_started = time.perf_counter()
            threads = getattr(cv2, "CAP_PROP_N_THREADS", None)
            if threads is not None:
                try:
                    cap = cv2.VideoCapture(str(self.event / "clip.mp4"), cv2.CAP_FFMPEG,
                                           [int(threads), 1])
                except (cv2.error, TypeError, ValueError):
                    cap = cv2.VideoCapture(str(self.event / "clip.mp4"))
            else:
                cap = cv2.VideoCapture(str(self.event / "clip.mp4"))
            self._trace("preview_video_opened", open_started=open_started)
            fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else 15.
            if not math.isfinite(fps) or fps <= 0:
                fps = 15.
            index = 0
            if not cap.isOpened():
                self._put(("video_missing", None))
                return
            while not self.cancel.is_set():
                if not self.playing.wait(.05):
                    continue
                if self.cancel.is_set():
                    break
                ok, frame = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    index = 0
                    ok, frame = cap.read()
                if not ok:
                    self._put(("video_missing", None))
                    return
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image.thumbnail((420, 420), Image.Resampling.LANCZOS)
                if index == 0:
                    self._trace("preview_loop_first_frame_decoded")
                if not self._put(("frame", (index, image, fps))):
                    break
                index += 1
        except Exception as exc:
            self._put(("error", str(exc)))
        finally:
            try:
                if cap is not None:
                    cap.release()
            finally:
                self.closed.set()


_trace_queue = queue.Queue(maxsize=128)
_trace_started = False
_trace_lock = threading.Lock()


def record_latency(record):
    """Nonblocking action diagnostics, separate from the authoritative journal."""
    global _trace_started
    with _trace_lock:
        if not _trace_started:
            threading.Thread(target=_write_traces, name="MLReviewLatency", daemon=True).start()
            _trace_started = True
    try:
        _trace_queue.put_nowait(dict(record, pid=os.getpid(), recorded_at=time.time()))
    except queue.Full:
        pass


def _write_traces():
    directory = (Path.home() / "Library/Logs/MeteorDetector" if sys.platform == "darwin"
                 else Path.home() / ".meteor_detector/logs")
    path = directory / "review_latency.jsonl"
    while True:
        record = _trace_queue.get()
        try:
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            # Keep timing diagnostics small; classification history is untouched.
            if path.exists() and path.stat().st_size > 2_000_000:
                path.replace(path.with_suffix(".previous.jsonl"))
            descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass
