"""Bounded, cancellable review preview decoding; never calls Tk."""
from __future__ import annotations

import json
import math
import os
from fractions import Fraction
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time

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
        self._process_lock = threading.Lock()
        self._process = None
        self._read_deadline = 0.
        self._read_timed_out = False
        self.thread = threading.Thread(target=self._run, name="MLReviewPreview", daemon=False)
        threading.Thread(target=self._watch_decoder, name="MLReviewDecoderWatch", daemon=True).start()
        self.thread.start()

    def stop(self):
        self.cancel.set()
        self.playing.set()
        with self._process_lock:
            process = self._process
        self._signal(process)

    @staticmethod
    def _signal(process, *, kill=False):
        if process is not None:
            try:
                process.kill() if kill else process.terminate()
            except OSError:
                pass

    def _watch_decoder(self):
        # A stalled pipe read must be interruptible, including on Windows.
        while not self.closed.wait(.1):
            with self._process_lock:
                process, deadline = self._process, self._read_deadline
            if self.cancel.is_set():
                self._signal(process, kill=True)
            elif deadline and time.perf_counter() >= deadline:
                self._read_timed_out = True
                self._signal(process, kill=True)

    def _launch(self, arguments):
        if self.cancel.is_set():
            raise InterruptedError()
        process = subprocess.Popen(arguments, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                   bufsize=0, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        with self._process_lock:
            self._process = process
        # stop() can race with Popen; the creator handles that race as well.
        if self.cancel.is_set():
            self._signal(process)
        return process

    def _close_process(self, process):
        self._signal(process)
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            self._signal(process, kill=True)
            process.wait()  # Reap before releasing the source for file moves.
        finally:
            try:
                if process.stdout is not None:
                    process.stdout.close()
            finally:
                with self._process_lock:
                    if self._process is process:
                        self._process = None
                        self._read_deadline = 0.

    def _video_info(self, executable, clip):
        process = self._launch([executable, "-v", "error", "-select_streams", "v:0",
                                "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate",
                                "-of", "json", str(clip)])
        try:
            output, _ = process.communicate(timeout=5)
            if self.cancel.is_set():
                raise InterruptedError()
            if process.returncode:
                raise ValueError("動画の情報を読み込めません。")
            info = json.loads(output)["streams"][0]
            width, height = int(info["width"]), int(info["height"])
            if not 0 < width <= 32768 or not 0 < height <= 32768:
                raise ValueError("動画の画像サイズが不正です。")
            fps = 15.
            for key in ("avg_frame_rate", "r_frame_rate"):
                try:
                    value = float(Fraction(info.get(key, "0/1")))
                    if math.isfinite(value) and value > 0:
                        fps = value
                        break
                except (ValueError, ZeroDivisionError, TypeError):
                    pass
            factor = min(1., 420 / width, 420 / height)
            return max(1, round(width * factor)), max(1, round(height * factor)), fps
        finally:
            self._close_process(process)

    def _read_frame(self, process, size):
        data = bytearray()
        with self._process_lock:
            self._read_deadline = time.perf_counter() + 5
        try:
            while len(data) < size and not self.cancel.is_set():
                chunk = process.stdout.read(size - len(data))
                if not chunk:
                    if self._read_timed_out:
                        raise TimeoutError("動画の復号がタイムアウトしました。再読み込みしてください。")
                    if self.cancel.is_set():
                        raise InterruptedError()
                    raise ValueError("動画を復号できません（途中で終了しました）。")
                data.extend(chunk)
            if self.cancel.is_set():
                raise InterruptedError()
            return bytes(data)
        finally:
            with self._process_lock:
                self._read_deadline = 0.

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
        decoder = None
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
            clip = (self.event / "clip.mp4").resolve()
            if not clip.is_file():
                self._put(("video_missing", None))
                return
            ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
            if not ffmpeg or not ffprobe:
                raise RuntimeError("動画表示に必要なFFmpeg/ffprobeが見つかりません。")
            width, height, fps = self._video_info(ffprobe, clip)
            # OpenCV's FFmpeg open mutex is process-wide. Do not let RTSP
            # connection retries delay local review files in that same process.
            decoder = self._launch([
                ffmpeg, "-nostdin", "-v", "error", "-threads", "1", "-filter_threads", "1",
                "-stream_loop", "-1", "-noautorotate", "-i", str(clip), "-map", "0:v:0",
                "-an", "-sn", "-dn", "-vf", f"scale={width}:{height}:flags=lanczos",
                "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-threads", "1",
                "-f", "rawvideo", "pipe:1"])
            self._trace("preview_decoder_started", open_started=open_started,
                        decoder_pid=decoder.pid, backend="ffmpeg_process")
            index = 0
            while not self.cancel.is_set():
                if not self.playing.wait(.05):
                    continue
                if self.cancel.is_set():
                    break
                frame = self._read_frame(decoder, width * height * 3)
                image = Image.frombytes("RGB", (width, height), frame)
                if index == 0:
                    self._trace("preview_loop_first_frame_decoded")
                if not self._put(("frame", (index, image, fps))):
                    break
                index += 1
        except InterruptedError:
            pass
        except Exception as exc:
            self._put(("error", str(exc)))
        finally:
            try:
                if decoder is not None:
                    self._close_process(decoder)
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
