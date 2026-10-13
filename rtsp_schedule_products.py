"""Durable scheduled RTSP products, prepared independently of recording.

Only disposable entries live in the bounded cache. Job manifests and original
archives remain outside it so eviction or a manual clear can be recovered.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time

import cv2
import numpy as np

import config
import rtsp_live
import rtsp_segments
import rtsp_startup
from video_io import open_video_capture

CACHE_LIMIT = 150_000_000_000
_CACHE_LOCK = threading.RLock()
_CACHE_BUILDS = {}
_CACHE_PINS = Counter()
_CACHE_CLEAR_PENDING = set()
CONCAT_GROUP_SIZE = 10
DEFAULTS = dict(timelapse=False, blend_video=False, blend_image=False, concatenate=False,
                duration=15.0, fps=60.0, constellations=True, timestamp=True,
                timestamp_position="bottom_right", timestamp_size=3.5, mean_radius=0,
                crop_margin=10, ai_crop=False, sky_config="", output_dir="")


def normalize_options(value):
    options = dict(DEFAULTS, **(value or {}))
    for name, minimum, maximum in (("duration", .1, 300), ("fps", 1, 120),
                                    ("timestamp_size", .1, 20), ("mean_radius", 0, 10),
                                    ("crop_margin", 0, 500)):
        options[name] = float(options[name])
        if not math.isfinite(options[name]) or not minimum <= options[name] <= maximum:
            raise ValueError(f"自動作成設定 {name}: {minimum}〜{maximum}の値を指定してください")
    if options["duration"] * options["fps"] > 18000:
        raise ValueError("タイムラプスは18000フレーム以内で設定してください")
    for name in ("mean_radius", "crop_margin"):
        options[name] = int(options[name])
    if options["timestamp_position"] not in ("bottom_right", "bottom_left", "top_right", "top_left"):
        raise ValueError("タイムラプスの時刻位置が不正です")
    return options


def enabled(options):
    return any(options.get(name) for name in ("timelapse", "blend_video", "blend_image", "concatenate"))


def processing_options(value):
    """Disabled outputs must not make ordinary RTSP depend on optional fields."""
    return normalize_options(value) if enabled(value or {}) else dict(DEFAULTS)


def schedule_window(when, start_hour, start_minute, end_hour, end_minute):
    """Return the daily window containing when, including a midnight crossing."""
    start = when.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
    end = when.replace(hour=end_hour, minute=end_minute, second=0, microsecond=0)
    if end == start:
        return None
    if end < start:
        if when < end:
            start -= timedelta(days=1)
        else:
            end += timedelta(days=1)
    return (start, end) if start <= when < end else None


def overlapping_schedule_windows(started_at, ended_at, clock):
    """Return scheduled windows that overlap the half-open interval [start, end)."""
    start_hour, start_minute, end_hour, end_minute = clock
    if ended_at <= started_at or (start_hour, start_minute) == (end_hour, end_minute):
        return []
    windows = []
    day = (started_at - timedelta(days=1)).date()
    last_day = ended_at.date()
    while day <= last_day:
        start = started_at.replace(year=day.year, month=day.month, day=day.day,
                                   hour=start_hour, minute=start_minute, second=0, microsecond=0)
        end = started_at.replace(year=day.year, month=day.month, day=day.day,
                                 hour=end_hour, minute=end_minute, second=0, microsecond=0)
        if end < start:
            end += timedelta(days=1)
        if start < ended_at and end > started_at:
            windows.append((start, end))
        day += timedelta(days=1)
    return windows


class ProductCache:
    """Atomic cache entries with oldest-created-first eviction and shared locking."""
    def __init__(self, root, limit=CACHE_LIMIT):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.limit = min(CACHE_LIMIT, max(0, int(limit)))
        with _CACHE_LOCK:
            # A crashed writer can leave an unpublished entry. It is disposable.
            for temporary in self.root.glob(".entry_*"):
                if temporary.is_dir() and temporary not in _CACHE_BUILDS:
                    shutil.rmtree(temporary)
            self._trim(0)

    def entries(self):
        return sorted((p for p in self.root.iterdir() if p.is_dir() and not p.name.startswith(".")),
                      key=lambda p: (p.stat().st_mtime_ns, p.name))

    def size(self):
        with _CACHE_LOCK:
            return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())

    def _trim(self, incoming):
        used = self.size()
        for temporary, (reserved, target) in _CACHE_BUILDS.items():
            if target.parent == self.root:
                written = sum(p.stat().st_size for p in temporary.rglob("*") if p.is_file())
                used += max(0, reserved-written)
        for entry in self.entries():
            if used + incoming <= self.limit:
                break
            if _CACHE_PINS[entry]:
                continue
            size = sum(p.stat().st_size for p in entry.rglob("*") if p.is_file())
            shutil.rmtree(entry)
            used -= size
        return used + incoming <= self.limit

    def put(self, key, files):
        """Files are already encoded bytes; no transient on-disk quota overshoot."""
        size = sum(len(data) for data in files.values())
        with _CACHE_LOCK:
            if size > self.limit:
                return False
            target = self.root / key
            if target.exists():
                return True
            if not self._trim(size):
                return False
            temporary = Path(tempfile.mkdtemp(prefix=".entry_", dir=self.root))
            try:
                for name, data in files.items():
                    (temporary / name).write_bytes(data)
                temporary.rename(target)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
            return True

    def build(self, key, reserve, writer):
        """Reserve capacity and stream a video into an atomic disposable entry.

        Heavy I/O runs without the cache lock so meteor delivery stays live.
        Reservations include unpublished files and prevent concurrent writers
        from taking each other's capacity. The writer must respect its bound.
        """
        with _CACHE_LOCK:
            target = self.root / key
            if target.exists():
                return True
            if reserve > self.limit or not self._trim(reserve):
                return False
            temporary = Path(tempfile.mkdtemp(prefix=".entry_", dir=self.root))
            _CACHE_BUILDS[temporary] = (reserve, target)
        try:
            writer(temporary)
            written = sum(p.stat().st_size for p in temporary.rglob("*") if p.is_file())
            if written > reserve:
                raise IOError("連結キャッシュが予約容量を超えました。元録画は保持します")
            with _CACHE_LOCK:
                if target in _CACHE_CLEAR_PENDING:
                    _CACHE_CLEAR_PENDING.discard(target)
                    return False
                if not target.exists():
                    temporary.rename(target)
                return True
        finally:
            with _CACHE_LOCK:
                _CACHE_BUILDS.pop(temporary, None)
                _CACHE_CLEAR_PENDING.discard(target)
                if temporary.exists():
                    shutil.rmtree(temporary)

    @contextmanager
    def pin(self, keys):
        """Keep input videos readable during eviction or a manual cache clear."""
        with _CACHE_LOCK:
            available = {key: self.root/key for key in keys if (self.root/key).is_dir()}
            for path in available.values():
                _CACHE_PINS[path] += 1
        try:
            yield available
        finally:
            with _CACHE_LOCK:
                for path in available.values():
                    _CACHE_PINS[path] -= 1
                    if not _CACHE_PINS[path]:
                        del _CACHE_PINS[path]
                        if path in _CACHE_CLEAR_PENDING:
                            _CACHE_CLEAR_PENDING.discard(path)
                            if path.exists():
                                shutil.rmtree(path)

    def read(self, key, name):
        with _CACHE_LOCK:
            path = self.root / key / name
            return path.read_bytes() if path.is_file() else None

    def clear(self, keys=None):
        with _CACHE_LOCK:
            allowed = set(keys) if keys is not None else None
            for entry in self.entries():
                if allowed is None or entry.name in allowed:
                    if _CACHE_PINS[entry]:
                        _CACHE_CLEAR_PENDING.add(entry)
                    else:
                        shutil.rmtree(entry)
            for _temporary, (_reserved, target) in _CACHE_BUILDS.items():
                if target.parent == self.root and (allowed is None or target.name in allowed):
                    _CACHE_CLEAR_PENDING.add(target)


def png_bytes(frame):
    ok, data = cv2.imencode(".png", frame)
    if not ok:
        raise IOError("キャッシュ画像の保存に失敗しました")
    return data.tobytes()


def decode_png(data):
    if data is None:
        return None
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise IOError("キャッシュ画像が破損しています")
    return frame


def line_rect(line, size, margin):
    width, height = size
    (x1, y1), (x2, y2) = line
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
    return [max(0, min(x1, x2) - margin), max(0, min(y1, y2) - margin),
            min(width, max(x1, x2) + margin + 1), min(height, max(y1, y2) + margin + 1)]


def coordinate_rect(frames, line, margin, seed_rect=None):
    """Follow temporal changes attached to the detected coordinates, without AI.

    A short detection can describe only part of a moving meteor. Grow a local
    search when its connected trail reaches an edge, then crop to the trail's
    actual bounds plus margin. Never pad/resize to the classifier's 256 square.
    """
    height, width = frames[0].shape[:2]
    seed = line_rect(line, (width, height), 0) if line is not None else seed_rect
    if seed is None or seed[2] <= seed[0] or seed[3] <= seed[1]:
        raise ValueError("流星の検出座標が不正です")
    # A few search pixels remain necessary even when output margin is zero.
    search_margin = max(10, margin)
    bounds = [max(0, seed[0]-search_margin), max(0, seed[1]-search_margin),
              min(width, seed[2]+search_margin), min(height, seed[3]+search_margin)]
    while True:
        x1, y1, x2, y2 = bounds
        minimum = maximum = None
        for frame in frames:
            gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            if minimum is None:
                minimum, maximum = gray.copy(), gray.copy()
            else:
                np.minimum(minimum, gray, out=minimum)
                np.maximum(maximum, gray, out=maximum)
        residual = cv2.subtract(maximum, minimum)
        anchor = np.zeros(residual.shape, np.uint8)
        if line is None:
            a, b, c, d = seed
            anchor[b-y1:d-y1, a-x1:c-x1] = 255
        else:
            points = [tuple(int(v-origin) for v, origin in zip(point, (x1, y1))) for point in line]
            cv2.line(anchor, points[0], points[1], 255, 5)
        peak = int(residual[anchor != 0].max(initial=0))
        median = float(np.median(residual))
        noise = float(np.median(np.abs(residual.astype(np.float32)-median)))
        threshold = max(8, peak*.15, median+6*noise)
        signal = (residual >= threshold).astype(np.uint8)
        # Join small gaps between consecutive observed meteor positions.
        joined = cv2.dilate(signal, np.ones((7, 7), np.uint8))
        _count, labels = cv2.connectedComponents(joined, connectivity=8)
        identities = np.unique(labels[(anchor != 0) & (joined != 0)])
        selected = (signal != 0) & np.isin(labels, identities[identities != 0])
        yy, xx = np.nonzero(selected)
        if not len(xx):
            return [max(0, seed[0]-margin), max(0, seed[1]-margin),
                    min(width, seed[2]+margin), min(height, seed[3]+margin)]
        left, top, right, bottom = int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())
        grow_x, grow_y = max(10, (x2-x1)//2), max(10, (y2-y1)//2)
        enlarged = [max(0, x1-grow_x) if left < search_margin else x1,
                    max(0, y1-grow_y) if top < search_margin else y1,
                    min(width, x2+grow_x) if right >= x2-x1-search_margin else x2,
                    min(height, y2+grow_y) if bottom >= y2-y1-search_margin else y2]
        if enlarged == bounds:
            return [max(0, min(seed[0], x1+left)-margin),
                    max(0, min(seed[1], y1+top)-margin),
                    min(width, max(seed[2], x1+right+1)+margin),
                    min(height, max(seed[3], y1+bottom+1)+margin)]
        bounds = enlarged


def local_boxes(image, log):
    import bright_area_detector as detector
    return detector.detect_meteor_boxes_local(image, log)


class ScheduleProducts:
    def __init__(self, root, options, clock, cancel, status=print, mask=None,
                 cache_limit=CACHE_LIMIT, box_detector=local_boxes):
        self.root = Path(root).resolve()
        self.options = normalize_options(options)
        self.clock = clock
        self.cancel = cancel
        self.status = status
        self.mask = None if mask is None else mask.copy()
        self.cache = ProductCache(self.root / ".schedule_cache", cache_limit)
        self.job_root = self.root / ".schedule_jobs"
        self.job_root.mkdir(parents=True, exist_ok=True)
        self.box_detector = box_detector
        self.lock = threading.RLock()
        self.jobs = {}
        self.flush_archives = lambda: None
        self.thread = None
        self.overlay_module = None
        self.source_data = {}
        self.source_stamps = {}
        self.source_indexes = {}
        self.sample_gaps = {}
        self.timelapse_revisions = {}
        self.prepared_timelapses = {}
        self.ai_unavailable = None
        self.stream_signatures = {}
        self.source_properties = {}
        self.checked_jobs = set()
        self.existing_only = False
        self.fresh_start = rtsp_startup.fresh_started_at(self.root)
        deferred = rtsp_startup.deferred_job_ids(self.root)
        setup_error = self.job_root / "capture_setup.failed"
        self.capture_setup_error = setup_error.read_text(encoding="utf-8") if setup_error.exists() and self.fresh_start is None else None
        if self.fresh_start is None and any(not (guard.parent / "job.json").exists() for guard in self.job_root.glob("*/capture.guard")):
            self.capture_setup_error = "流星素材のジョブ登録が中断されています"
        for path in self.job_root.glob("*/job.json"):
            job = json.loads(path.read_text(encoding="utf-8"))
            if job["state"] != "complete" and job["id"] not in deferred:
                self.jobs[job["id"]] = job
                guard = path.parent / "capture.guard"
                if not guard.exists():
                    guard.write_bytes(b" " * 64)
                pending = self._capture_guard(job)
                if pending:
                    event = next((event for event in job["events"] if event["id"] == pending), None)
                    if event is not None:
                        event["capture_pending"] = True
                    else:
                        job["capture_unknown"] = pending

    def _save(self, job):
        with self.lock:
            rtsp_live.write_json(self.job_root / job["id"] / "job.json", job)

    def _capture_guard(self, job):
        path = self.job_root / job["id"] / "capture.guard"
        return path.read_text(encoding="ascii").strip() if path.exists() else ""

    def _set_capture_guard(self, job, identity):
        # Reserve the small guard when creating the job. Updating it in place
        # does not need a second file allocation when the disk fills up. A torn
        # or interrupted write leaves a nonempty guard and fails closed.
        path = self.job_root / job["id"] / "capture.guard"
        with path.open("r+b") as handle:
            handle.write(identity.encode("ascii").ljust(64, b" "))
            handle.flush()
            os.fsync(handle.fileno())

    def _assert_capture_ready(self, job):
        with self.lock:
            if self.capture_setup_error or job.get("capture_unknown") or any(
                    event.get("capture_pending") for event in job["events"]):
                raise IOError("流星素材の保存・登録を復旧するまで、生成完了と元録画の削除を保留します")
            if self._capture_guard(job):
                raise IOError("流星素材の登録が未完了です。元録画を保持します")

    def _job(self, when):
        with self.lock:
            return self._job_locked(when)

    def _job_locked(self, when):
        window = schedule_window(when, *self.clock)
        if window is None:
            return None
        start, end = window
        key = start.strftime("%Y%m%dT%H%M") + "_" + end.strftime("%Y%m%dT%H%M")
        if self.fresh_start is not None and start < self.fresh_start:
            if when < self.fresh_start:
                return None
            start = self.fresh_start
            key += "_fresh_" + start.strftime("%Y%m%dT%H%M%S%f")
        if key not in self.jobs:
            directory = self.job_root / key
            directory.mkdir(parents=True, exist_ok=True)
            marker = directory / "job.json"
            if marker.exists():
                job = json.loads(marker.read_text(encoding="utf-8"))
            else:
                job = dict(id=key, start=start.isoformat(), end=end.isoformat(),
                           options=self.options, state="recording", archives=[], events=[], samples={}, outputs={})
                guard = directory / "capture.guard"
                with guard.open("wb") as handle:
                    handle.write(b" " * 64)
                    handle.flush()
                    os.fsync(handle.fileno())
                if self.mask is not None:
                    np.save(directory / "mask.npy", self.mask)
                    job["mask_path"] = str(directory / "mask.npy")
                self._save(job)
            self.jobs[key] = job
            # Compatibility with jobs created before the capture guard existed.
            guard = directory / "capture.guard"
            if not guard.exists():
                guard.write_bytes(b" " * 64)
        return self.jobs[key]

    def _overlapping_job(self, started_at, ended_at):
        overlapping = []
        for job in self.jobs.values():
            if job["state"] == "complete":
                continue
            start, end = datetime.fromisoformat(job["start"]), datetime.fromisoformat(job["end"])
            if start < ended_at and end > started_at:
                overlapping.append((min(ended_at, end)-max(started_at, start), job))
        return max(overlapping, key=lambda item: item[0])[1] if overlapping else None

    def _event_job(self, started_at, ended_at):
        # A resumed window keeps its saved settings even if the UI has changed
        # (or was closed before saving its checkboxes).
        job = self._overlapping_job(started_at, ended_at)
        if job is not None or self.existing_only:
            return job
        for start, end in overlapping_schedule_windows(started_at, ended_at, self.clock):
            job = self._job(max(started_at, start))
            if job is not None and job["state"] != "complete":
                return job
        return None

    def capture_event(self, frames, frame_rate, started_at, line):
        """Persist a small lossless line crop before detector buffers are reused.

        VLM inference and all rendering run on the product worker. A cache miss
        can reconstruct these frames from original archives using this clock.
        """
        if not frames:
            return
        with self.lock:
            ended_at = started_at + timedelta(seconds=len(frames)/frame_rate)
            job = self._event_job(started_at, ended_at)
            if job is None or job["state"] == "complete":
                return
            if not (job["options"]["blend_image"] or job["options"]["blend_video"]):
                return
            size = [frames[0].shape[1], frames[0].shape[0]]
            rect = coordinate_rect(frames, line, job["options"]["crop_margin"])
            if rect[2] <= rect[0] or rect[3] <= rect[1]:
                raise ValueError("流星の検出座標が不正です")
            identity = hashlib.sha256((started_at.isoformat() + repr(line)).encode()).hexdigest()[:24]
            previous = next((event for event in job["events"] if event["id"] == identity), None)
            if previous is not None and not previous.get("capture_pending"):
                return
            # Exclude padding outside the schedule itself.
            start, end = datetime.fromisoformat(job["start"]), datetime.fromisoformat(job["end"])
            indices = [i for i in range(len(frames)) if start <= started_at + timedelta(seconds=i/frame_rate) < end]
            if not indices:
                return
            key = job["id"] + "_event_" + identity
            event = dict(id=identity, key=key, start=(started_at + timedelta(seconds=indices[0]/frame_rate)).isoformat(),
                         count=len(indices), fps=frame_rate, size=size, rect=rect, ai_done=False,
                         detected_line=[[float(x), float(y)] for x, y in line],
                         coordinate_rect_ready=True,
                         capture_pending=True)
            if previous is None:
                job["events"].append(event)
            else:
                event = previous
            # Publish recovery coordinates before any image encoding/cache I/O.
            # Even if the manifest write fails, the preallocated guard records
            # that this event must be recovered before deleting source footage.
            self._set_capture_guard(job, identity)
            self._save(job)
            x1, y1, x2, y2 = rect
            files = {f"{j:06d}.png": png_bytes(frames[i][y1:y2, x1:x2]) for j, i in enumerate(indices)}
            if not self.cache.put(key, files):
                raise IOError("流星素材がキャッシュ上限を超えました。元録画からの復旧を待ちます")
            event["capture_pending"] = False
            try:
                self._save(job)
                self._set_capture_guard(job, "")
            except Exception:
                event["capture_pending"] = True
                raise

    def capture_event_failed(self, frames, frame_rate, started_at, line, error):
        """Latch capture failure even when the manifest cannot currently write."""
        with self.lock:
            ended_at = started_at + timedelta(seconds=len(frames)/frame_rate)
            job = self._overlapping_job(started_at, ended_at)
            if job is None:
                windows = overlapping_schedule_windows(started_at, ended_at, self.clock)
                if not windows or self.existing_only:
                    return
            if job is None:
                # Job setup failed: the worker must fail closed instead of
                # treating an empty event list as a successful observation.
                self.capture_setup_error = str(error)
                try:
                    (self.job_root / "capture_setup.failed").write_text(str(error), encoding="utf-8")
                except OSError:
                    pass
                return
            identity = hashlib.sha256((started_at.isoformat() + repr(line)).encode()).hexdigest()[:24]
            event = next((event for event in job["events"] if event["id"] == identity), None)
            if event is not None:
                event["capture_pending"] = True
                event["capture_error"] = str(error)
            else:
                job["capture_unknown"] = identity
            try:
                self._set_capture_guard(job, identity)
                self._save(job)
            except Exception:
                # Keep the in-memory latch; the worker retries persistence.
                pass

    def _recover_captures(self, job):
        with self.lock:
            if job.get("capture_unknown"):
                self._save(job)
                self._assert_capture_ready(job)
            pending = [event for event in job["events"] if event.get("capture_pending")]
        for event in pending:
            with self.lock:
                # Save recovery metadata again before clearing any failure.
                self._save(job)
            for index in range(event["count"]):
                self._event_frame(job, event, index)
            self._prepare_event(job, event)
            with self.lock:
                event["capture_pending"] = False
                event.pop("capture_error", None)
                # A recovered event invalidates any previously rendered blend.
                job["blends_done"] = False
                try:
                    self._save(job)
                except Exception:
                    event["capture_pending"] = True
                    raise
        with self.lock:
            if self.capture_setup_error or job.get("capture_unknown") or any(event.get("capture_pending") for event in job["events"]):
                self._assert_capture_ready(job)
            self._set_capture_guard(job, "")
            self._assert_capture_ready(job)

    def _mask(self, job):
        path = job.get("mask_path")
        return np.load(path) if path else None

    def _archives(self, job):
        found = []
        start, end = datetime.fromisoformat(job["start"]), datetime.fromisoformat(job["end"])
        for day in (start.date(), end.date()):
            directory = self.root / day.strftime("%Y%m%d")
            for marker in directory.glob("*/*.mp4.capture.json"):
                path = Path(str(marker)[:-len(".capture.json")])
                if rtsp_startup.is_deferred_recording(path) or not path.is_file() or not rtsp_segments.is_finalized_mp4(path):
                    continue
                data = self._timing(str(path))
                when = datetime.fromisoformat(data["start_time"])
                stop = when + timedelta(seconds=data["frame_count"]/data["fps"])
                if when < end and stop > start:
                    found.append((when, str(path)))
        paths = [path for _, path in sorted(set(found))]
        if paths != job["archives"]:
            job["archives"] = paths
            self._save(job)
        return paths

    def _timing(self, source):
        stamp = self._source_stamp(source)
        if source not in self.source_data or self.source_stamps.get(source) != stamp:
            self.source_data[source] = rtsp_live.timing(source)
            self.source_stamps[source] = self._source_stamp(source)
        return self.source_data[source]

    @staticmethod
    def _source_stamp(source):
        result = []
        for path in (Path(source), Path(str(source) + ".capture.json")):
            try:
                stat = path.stat()
                result.append((stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
            except FileNotFoundError:
                result.append(None)
        return tuple(result)

    def _source_revision(self, job):
        return tuple((source, self._source_stamp(source)) for source in job["archives"])

    def _source_frame(self, job, when, radius=0):
        sources = job["archives"]
        revision = self._source_revision(job)
        cached = self.source_indexes.get(job["id"])
        if cached is None or cached[0] != revision:
            starts = [datetime.fromisoformat(self._timing(path)["start_time"]) for path in sources]
            self.source_indexes[job["id"]] = revision, starts
        else:
            starts = cached[1]
        position = bisect_right(starts, when)-1
        if position < 0:
            return None
        source = sources[position]
        data = self._timing(source)
        index = int((when-starts[position]).total_seconds()*data["fps"] + 1e-6)
        if not 0 <= index < data["frame_count"]:
            return None
        cap = open_video_capture(source)
        try:
            first, last = max(0, index-radius), min(data["frame_count"]-1, index+radius)
            cap.set(cv2.CAP_PROP_POS_FRAMES, first)
            accumulator = None
            for _ in range(first, last+1):
                ok, frame = cap.read()
                if not ok:
                    raise IOError(f"録画のフレームを読めません: {source}")
                if accumulator is None:
                    accumulator = frame.astype(np.float32)
                else:
                    accumulator += frame
            return np.rint(accumulator/(last-first+1)).astype(np.uint8)
        finally:
            cap.release()

    def _event_frame(self, job, event, index):
        try:
            frame = decode_png(self.cache.read(event["key"], f"{index:06d}.png"))
        except IOError:
            frame = None
        if frame is not None and frame.shape[1::-1] == (
                event["rect"][2]-event["rect"][0], event["rect"][3]-event["rect"][1]):
            return frame
        when = datetime.fromisoformat(event["start"]) + timedelta(seconds=index/event["fps"])
        frame = self._source_frame(job, when)
        if frame is None:
            raise IOError("流星キャッシュの再作成に必要な録画がまだありません")
        mask = self._mask(job)
        if mask is not None:
            frame = cv2.bitwise_and(frame, frame, mask=cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST))
        x1, y1, x2, y2 = event["rect"]
        return frame[y1:y2, x1:x2]

    @staticmethod
    def _crop_method(job):
        return "ai" if job["options"].get("ai_crop", False) else "coordinates"

    def _cutout_key(self, job, event):
        method = self._crop_method(job)
        suffix = method
        if method == "ai":
            suffix += "_" + hashlib.sha256(json.dumps(event.get("boxes", [])).encode()).hexdigest()[:12]
        return event["key"] + "_cutouts_" + suffix

    def _prepare_event(self, job, event):
        method = self._crop_method(job)
        if method == "coordinates":
            if not event.get("coordinate_rect_ready"):
                # Old caches may already truncate a short detection's trail.
                # Rebuild once from retained archives, not from that crop.
                started = datetime.fromisoformat(event["start"])
                frames = []
                for index in range(event["count"]):
                    if self.cancel.is_set():
                        raise InterruptedError("座標範囲の復旧を中断しました")
                    frame = self._source_frame(job, started+timedelta(seconds=index/event["fps"]))
                    if frame is None:
                        raise IOError("流星の座標範囲を補完する録画がまだありません")
                    frames.append(frame)
                margin = job["options"]["crop_margin"]
                a, b, c, d = event["rect"]
                width, height = event["size"]
                seed = [min(a+margin, c-1) if a > 0 else 0,
                        min(b+margin, d-1) if b > 0 else 0,
                        max(a+1, c-margin) if c < width else width,
                        max(b+1, d-margin) if d < height else height]
                event["rect"] = coordinate_rect(frames, event.get("detected_line"), margin, seed)
                event["coordinate_rect_ready"] = True
                event["key"] += "_coordinates_v1"
                self._save(job)
                mask = self._mask(job)
                a, b, c, d = event["rect"]
                if mask is not None:
                    mask = cv2.resize(mask, (frames[0].shape[1], frames[0].shape[0]), interpolation=cv2.INTER_NEAREST)[b:d, a:c]
                files = {}
                for index, frame in enumerate(frames):
                    crop = frame[b:d, a:c]
                    if mask is not None:
                        crop = cv2.bitwise_and(crop, crop, mask=mask)
                    files[f"{index:06d}.png"] = png_bytes(crop)
                self.cache.put(event["key"], files)
            # Coordinates seed the local temporal trail search; retain the
            # complete native-sized region, including its requested margin.
            width = event["rect"][2]-event["rect"][0]
            height = event["rect"][3]-event["rect"][1]
            event["boxes"] = [[0, 0, width, height]]
            event["box_source"] = "detected_line"
            event["ai_done"] = False
        elif not event.get("ai_done") or event.get("crop_method") not in (None, "ai"):
            composite = None
            for index in range(event["count"]):
                frame = self._event_frame(job, event, index)
                composite = frame if composite is None else np.maximum(composite, frame)
            height, width = composite.shape[:2]
            boxes = []
            try:
                if self.ai_unavailable is None:
                    boxes = self.box_detector(composite, self.status)
            except Exception as exc:
                self.ai_unavailable = str(exc)
                self.status(f"[自動作成] 内蔵AIの枠取得に失敗。線検出の枠を使用: {exc}")
            valid = []
            for box in boxes:
                if len(box) != 4 or not all(math.isfinite(v) and 0 <= v <= 1000 for v in box):
                    continue
                x1, y1, x2, y2 = box
                if x2 > x1 and y2 > y1:
                    valid.append([max(0, int(x1*width/1000)-3), max(0, int(y1*height/1000)-3),
                                  min(width, math.ceil(x2*width/1000)+3), min(height, math.ceil(y2*height/1000)+3)])
            event["boxes"] = valid or [[0, 0, width, height]]
            event["box_source"] = "local_ai" if valid else "detected_line"
            event["ai_done"] = True
        event["crop_method"] = method
        event["cutouts_prepared"] = False
        self._save(job)
        # Distinct keys prevent previously cached AI crops from silently
        # surviving a migration to the default coordinate method.
        key = self._cutout_key(job, event)
        if self._cutouts_valid(event, key):
            event["cutout_key"] = key
            event["cutouts_prepared"] = True
            self._save(job)
            return
        files = {}
        for index in range(event["count"]):
            if self.cancel.is_set():
                raise InterruptedError("切り抜きキャッシュを中断しました。元録画は保持します")
            files[f"{index:06d}.png"] = png_bytes(self._masked_crop(job, event, index))
        self.cache.clear([key])
        event["cutouts_prepared"] = self.cache.put(key, files)
        event["cutout_key"] = key
        self._save(job)

    def _cutouts_valid(self, event, key):
        # Pin the directory against eviction while reading, but release the
        # shared cache lock between files and before decoding each PNG.
        with self.cache.pin([key]) as available:
            if key not in available:
                return False
            # Decode to detect a damaged/incomplete cache after interruption.
            for index in range(event["count"]):
                try:
                    frame = decode_png(self.cache.read(key, f"{index:06d}.png"))
                except IOError:
                    return False
                if frame is None or frame.shape[1::-1] != (
                        event["rect"][2]-event["rect"][0], event["rect"][3]-event["rect"][1]):
                    return False
            return True

    def _masked_crop(self, job, event, index):
        crop = self._event_frame(job, event, index)
        keep = np.zeros(crop.shape[:2], np.uint8)
        for a, b, c, d in event["boxes"]:
            keep[b:d, a:c] = 255
        return cv2.bitwise_and(crop, crop, mask=keep)

    def _cutout_frame(self, job, event, index):
        key = self._cutout_key(job, event)
        try:
            frame = decode_png(self.cache.read(key, f"{index:06d}.png"))
            if frame is not None and frame.shape[1::-1] == (
                    event["rect"][2]-event["rect"][0], event["rect"][3]-event["rect"][1]):
                return frame
        except IOError:
            pass
        return self._masked_crop(job, event, index)

    def _annotate(self, job, frame, when):
        options = job["options"]
        if options["constellations"]:
            # Use the actual OBS renderer and its approved calibration/opacity.
            # Snapshot configuration and model once, then keep the same revision.
            if "sky_payload" not in job:
                from obs_meteor_overlay import sky_overlay
                module = sky_overlay
                self.overlay_module = module
                config_path = Path(options["sky_config"] or Path(config.EXE_DIR) / "obs_meteor_overlay/sky_config.json")
                configuration, payload, revision = module.approved_model(config_path)
                job.update(sky_payload=payload, sky_revision=revision, sky_opacity=configuration.get("line_opacity", .45))
                self._save(job)
            if self.overlay_module is None:
                from obs_meteor_overlay import sky_overlay
                self.overlay_module = sky_overlay
            payload = job["sky_payload"]
            if frame.shape[:2] != (payload["height"], payload["width"]):
                raise ValueError("配信の星座モデルと録画の解像度が一致しません")
            layer = self.overlay_module.render_layer(payload, when, self._mask(job), job["sky_opacity"])
            alpha = layer[:, :, 3:4].astype(np.float32)/255
            frame = np.rint(frame*(1-alpha) + layer[:, :, :3][:, :, ::-1]*alpha).astype(np.uint8)
        if options["timestamp"]:
            import timelapse_creator
            frame = timelapse_creator._draw_timestamp(frame, when, timelapse_creator._normalize_timestamp_settings({
                "enabled": True, "position": options["timestamp_position"], "size_percent": options["timestamp_size"]}))
        return frame

    def _prepare_samples(self, job):
        options = job["options"]
        if not options["timelapse"]:
            return
        start, end = datetime.fromisoformat(job["start"]), datetime.fromisoformat(job["end"])
        count = max(1, round(options["duration"]*options["fps"]))
        sources = self._source_revision(job)
        source_versions = dict(sources)
        # Persist the identities behind the disposable PNGs. Replacing an
        # existing archive must rebuild its samples as well as the final MP4.
        previous = job.get("sample_source_versions", {})
        changed = bool(set(previous) - set(source_versions)) or any(
            source in source_versions and json.dumps(stamp) != json.dumps(source_versions[source])
            for source, stamp in previous.items())
        sample_times = sorted(datetime.fromisoformat(when) for when in job["samples"].values())
        for source in set(source_versions) - set(previous):
            if not sample_times:
                break
            data = self._timing(source)
            low = datetime.fromisoformat(data["start_time"])
            high = low + timedelta(seconds=data["frame_count"] / data["fps"])
            position = bisect_right(sample_times, low-timedelta(microseconds=1))
            if position < len(sample_times) and sample_times[position] < high:
                changed = True
                break
        if changed:
            self.cache.clear([job["id"] + f"_sample_{int(index):06d}" for index in job["samples"]])
            job["samples"].clear()
            self.sample_gaps.pop(job["id"], None)
            self.status("[自動作成] 録画の差し替えを検知。タイムラプス素材を再作成します")
        job["sample_source_versions"] = source_versions
        revision = (sources, job["start"], job["end"], count,
                    options["mean_radius"])
        cached = self.sample_gaps.get(job["id"])
        if cached is None or cached[0] != revision:
            gaps = set()
            self.sample_gaps[job["id"]] = revision, gaps
        else:
            gaps = cached[1]
        # Sampling uses the full schedule clock, never a dusk/dawn filter.
        for index in range(count):
            if self.cancel.is_set():
                return
            key = job["id"] + f"_sample_{index:06d}"
            if (self.cache.root / key / "frame.png").is_file():
                continue
            if index in gaps:
                continue
            when = start + (end-start)*(index/max(1, count-1))
            if index == count-1:
                when = end-timedelta(milliseconds=40)
            frame = self._source_frame(job, when, options["mean_radius"])
            if frame is None:
                gaps.add(index)
                continue  # Real connection gaps remain gaps, not fabricated frames.
            mask = self._mask(job)
            if mask is not None:
                frame = cv2.bitwise_and(frame, frame, mask=cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST))
            frame = self._annotate(job, frame, when)
            self.cache.put(key, {"frame.png": png_bytes(frame)})
            job["samples"][str(index)] = when.isoformat()
        self._save(job)

    def _drained(self, job):
        start, end = datetime.fromisoformat(job["start"]), datetime.fromisoformat(job["end"])
        for path in (self.root / ".rtsp_live").rglob("chunk_*.mp4"):
            if "_innovation" in path.name or "_temp_" in path.name or rtsp_startup.is_deferred_recording(path):
                continue
            session = json.loads((path.parent / ".session.json").read_text(encoding="utf-8"))
            when = datetime.fromisoformat(session["start_time"])
            if when < end and when + timedelta(days=1) > start:
                # Active/incomplete tails and queued analysis must finish first.
                if not rtsp_segments.is_finalized_mp4(path):
                    return False
                data = rtsp_live.timing(path)
                captured = datetime.fromisoformat(data["start_time"])
                if start <= captured < end:
                    # Wait until the archiver has removed consumed spool inputs.
                    # Otherwise replacing a minute archive can orphan its chunks.
                    return False
        return True

    def _output_root(self, job):
        path = Path(job["options"]["output_dir"] or self.root / "schedule_products") / job["id"]
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _write_video(self, path, frames, fps):
        from video_encoding import FFmpegFrameWriter
        writer = None
        count = 0
        try:
            for frame in frames:
                if self.cancel.is_set():
                    raise InterruptedError("自動作成を中断しました。元録画は保持します")
                if writer is None:
                    writer = FFmpegFrameWriter(str(path), fps, (frame.shape[1], frame.shape[0]),
                                               {"codec": "h264", "quality": "standard"})
                writer.write(frame)
                count += 1
        finally:
            if writer is not None:
                writer.release()
        if not count or rtsp_segments._properties(path)[1] != count:
            raise IOError("自動作成動画のフレーム数検証に失敗しました")

    def _timelapse(self, job, output):
        options = job["options"]
        available = sorted(map(int, job["samples"]))
        if not available:
            raise IOError("スケジュール内のタイムラプス素材がありません")
        count = max(1, round(options["duration"]*options["fps"]))
        # Stretch only actual captured samples when a connection gap removed slots.
        def frames():
            previous_index, frame = None, None
            for position in np.linspace(0, len(available)-1, count):
                index = available[int(round(position))]
                if index == previous_index:
                    yield frame
                    continue
                key = job["id"] + f"_sample_{index:06d}"
                frame = decode_png(self.cache.read(key, "frame.png"))
                if frame is None:
                    when = datetime.fromisoformat(job["samples"][str(index)])
                    frame = self._source_frame(job, when, options["mean_radius"])
                    if frame is None:
                        raise IOError("タイムラプス素材を再作成できません")
                    mask = self._mask(job)
                    if mask is not None:
                        frame = cv2.bitwise_and(frame, frame, mask=cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST))
                    frame = self._annotate(job, frame, when)
                previous_index = index
                yield frame
        from video_encoding import write_repeated_images
        written = write_repeated_images(output, frames(), options["fps"], self.cancel,
                                       {"codec": "h264", "quality": "standard"})
        if written != count or rtsp_segments._properties(output)[1] != count:
            raise IOError("タイムラプス動画のフレーム数検証に失敗しました")

    def _timelapse_key(self, job):
        samples = []
        for index, when in sorted(job["samples"].items()):
            path = self.cache.root / (job["id"] + f"_sample_{int(index):06d}") / "frame.png"
            samples.append((index, when, self._source_stamp(path)))
        revision = (job["options"], self._source_revision(job), samples)
        digest = hashlib.sha256(json.dumps(revision, sort_keys=True).encode()).hexdigest()[:24]
        return job["id"] + "_timelapse_" + digest

    def _prepare_timelapse(self, job):
        # This product depends only on captured samples. During a backlog it
        # can render while the detector is still finding events for the blends.
        # A changed archive/sample revision produces a different cache key;
        # final publication still waits for detection and validates that key.
        if not job["options"]["timelapse"] or not job["samples"] or "timelapse" in job["outputs"]:
            return
        key = self._timelapse_key(job)
        if self.timelapse_revisions.get(job["id"]) != key:
            # Observe the same revision on two product ticks before spending
            # encoding time; archives may still arrive during backlog recovery.
            self.timelapse_revisions[job["id"]] = key
            return
        if not (self.cache.root / key / "timelapse.mp4").exists():
            self.status(f"[自動作成] タイムラプスを並行準備: {len(job['archives'])}録画・{len(job['samples'])}素材 ({key})")
        reserve = int(job["options"]["duration"] * 48_000_000 / 8) + 16 * 1024**2
        if self.cache.build(key, reserve, lambda directory: self._timelapse(job, directory / "timelapse.mp4")):
            previous = self.prepared_timelapses.get(job["id"])
            self.prepared_timelapses[job["id"]] = key
            if previous and previous != key:
                self.cache.clear([previous])

    def _blends(self, job, directory):
        events = job["events"]
        if not events:
            self.status("[自動作成] このスケジュールには流星がありません。比較明合成をスキップします")
            return {}
        size = events[0]["size"]
        fps = events[0]["fps"]
        image = np.zeros((size[1], size[0], 3), np.uint8)
        count = 0
        with tempfile.TemporaryDirectory(prefix=".blend_", dir=directory) as temporary:
            temporary = Path(temporary)
            for event in events:
                if event["size"] != size:
                    raise IOError("流星クリップの解像度が途中で変わっています")
                if not event.get("cutouts_prepared") or event.get("crop_method") != self._crop_method(job):
                    self._prepare_event(job, event)
                x, y, _, _ = event["rect"]
                frame_count = max(1, round(event["count"]*fps/event["fps"]))
                count = max(count, frame_count)
                for index in range(frame_count):
                    if self.cancel.is_set():
                        raise InterruptedError("比較明合成を中断しました")
                    crop = self._cutout_frame(job, event, min(event["count"]-1, int(index*event["fps"]/fps)))
                    height, width = crop.shape[:2]
                    region = image[y:y+height, x:x+width]
                    np.maximum(region, crop, out=region)
                    if job["options"]["blend_video"]:
                        frame_path = temporary / f"{index:06d}.png"
                        frame = cv2.imread(str(frame_path)) if frame_path.exists() else np.zeros_like(image)
                        region = frame[y:y+height, x:x+width]
                        np.maximum(region, crop, out=region)
                        frame_path.write_bytes(png_bytes(frame))
            outputs = {}
            if job["options"]["blend_image"]:
                path = directory / "lighten.png"
                path.write_bytes(png_bytes(image))
                outputs["blend_image"] = str(path)
            if job["options"]["blend_video"]:
                path = directory / "lighten.mp4"
                self._write_video(path, (cv2.imread(str(temporary / f"{i:06d}.png")) for i in range(count)), fps)
                outputs["blend_video"] = str(path)
            return outputs

    @staticmethod
    def _stream_signature(path):
        probe = shutil.which("ffprobe")
        if not probe:
            return None
        result = subprocess.run([probe, "-v", "error", "-select_streams", "v:0",
            "-show_data_hash", "sha256", "-show_entries",
            "stream=codec_name,profile,level,pix_fmt,width,height,extradata_hash",
            "-of", "json", str(path)], capture_output=True, text=True, timeout=15)
        if result.returncode:
            return None
        return json.loads(result.stdout).get("streams", [None])[0]

    def _concat_plan(self, job, paths):
        start, end = datetime.fromisoformat(job["start"]), datetime.fromisoformat(job["end"])
        data = []
        for path in paths:
            stat = Path(path).stat()
            key = (str(path), stat.st_size, stat.st_mtime_ns)
            if key not in self.source_properties:
                self.source_properties[key] = rtsp_segments._properties(path)
            rate, count, dimensions = self.source_properties[key]
            # Recovery metadata supplies the clock, never the deletion bounds.
            data.append(dict(self._timing(path), fps=rate, frame_count=count, size=list(dimensions)))
        fps, size = data[0]["fps"], data[0]["size"]
        if any(abs(item["fps"]-fps) > .05 or item["size"] != size for item in data):
            raise IOError("録画のFPSまたは解像度が途中で変わっています。元動画を保持します")
        ranges, deletable = [], []
        for path, item in zip(paths, data):
            when = datetime.fromisoformat(item["start_time"])
            first = max(0, math.ceil((start-when).total_seconds()*fps-1e-6))
            last = min(item["frame_count"], math.ceil((end-when).total_seconds()*fps-1e-6))
            ranges.append([first, last])
            if first == 0 and last == item["frame_count"]:
                deletable.append(path)
        return dict(fps=fps, count=sum(last-first for first, last in ranges),
                    size=size, ranges=ranges, deletable=deletable)

    def _signature(self, path):
        stat = Path(path).stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in self.stream_signatures:
            self.stream_signatures[key] = self._stream_signature(path)
        return self.stream_signatures[key]

    def _run_concat_copy(self, paths, output):
        # Poll a real process so closing the app cannot orphan a video writer.
        with tempfile.TemporaryDirectory(prefix=".concat_list_", dir=output.parent) as temporary:
            listing = Path(temporary)/"inputs.txt"
            listing.write_text("".join("file '" + str(Path(path).resolve()).replace("'", "'\\''") + "'\n" for path in paths), encoding="utf-8")
            with tempfile.TemporaryFile() as errors:
                process = subprocess.Popen([shutil.which("ffmpeg") or "ffmpeg", "-nostdin", "-v", "error", "-y",
                    "-f", "concat", "-safe", "0", "-i", str(listing), "-map", "0:v:0", "-an", "-c", "copy",
                    "-movflags", "+faststart", str(output)], stdout=subprocess.DEVNULL, stderr=errors)
                try:
                    while process.poll() is None:
                        if self.cancel.wait(.1):
                            raise InterruptedError("連結を中断しました。元録画は保持します")
                    return process.returncode == 0
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()

    def _render_concat(self, paths, expected, output):
        signatures = [self._signature(path) for path in paths]
        whole = all(first == 0 and last == rtsp_segments._properties(path)[1]
                    for path, (first, last) in zip(paths, expected["ranges"]))
        compatible = whole and signatures and signatures[0] is not None and all(
            item == signatures[0] for item in signatures)
        if compatible and self._run_concat_copy(paths, output):
            try:
                self._validate_concat(output, expected)
                return
            except IOError:
                pass
        def frames():
            for path, (first, last) in zip(paths, expected["ranges"]):
                cap = open_video_capture(path)
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
                    for _ in range(first, last):
                        ok, frame = cap.read()
                        if not ok:
                            raise IOError(f"連結元の動画を読めません: {path}")
                        yield frame
                finally:
                    cap.release()
        self._write_video(output, frames(), expected["fps"])
        self._validate_concat(output, expected)

    def _prepare_concatenation(self, job, max_builds=1):
        if not job["options"]["concatenate"]:
            return
        old = {item["key"]: item for item in job.get("concat_parts", [])}
        parts = []
        builds = 0
        for offset in range(0, len(job["archives"]), CONCAT_GROUP_SIZE):
            paths = job["archives"][offset:offset+CONCAT_GROUP_SIZE]
            expected = self._concat_plan(job, paths)
            sources = [(path, Path(path).stat().st_size, Path(path).stat().st_mtime_ns) for path in paths]
            identity = hashlib.sha256(json.dumps([sources, expected], sort_keys=True).encode()).hexdigest()[:24]
            key = job["id"] + "_concat_" + identity
            part = dict(key=key, sources=paths, expected=expected)
            previous = old.get(key)
            if previous:
                part.update(previous)
            parts.append(part)
            if part.get("prepared") or part.get("unavailable") or builds >= max_builds:
                continue
            if self.cancel.is_set():
                raise InterruptedError("連結キャッシュを中断しました。元録画は保持します")
            duration = expected["count"]/expected["fps"]
            # Stream copy stays below input bytes + container overhead. Frame
            # encoding uses 40 Mbps, maxrate 48 Mbps; reserve a generous bound.
            reserve = max(sum(size for _path, size, _mtime in sources), math.ceil(duration*8_000_000)) + 8_000_000
            def writer(directory):
                self._render_concat(paths, expected, directory/"recording.mp4")
            prepared = self.cache.build(key, reserve, writer)
            part["prepared"] = prepared
            part["unavailable"] = not prepared
            builds += 1
        with self.lock:
            job["concat_parts"] = parts
            self._save(job)
        current_keys = {part["key"] for part in parts}
        self.cache.clear([key for key in old if key not in current_keys])

    def _check_existing_cache(self, job):
        missing_cuts = missing_parts = missing_samples = 0
        for index in job.get("samples", {}):
            key = job["id"] + f"_sample_{int(index):06d}"
            try:
                frame = decode_png(self.cache.read(key, "frame.png"))
                if frame is None:
                    raise IOError("Missing sample")
            except IOError:
                self.cache.clear([key])
                missing_samples += 1
        for event in list(job["events"]):
            key = self._cutout_key(job, event)
            if event.get("crop_method") != self._crop_method(job):
                job["blends_done"] = False
            if event.get("crop_method") != self._crop_method(job) or not self._cutouts_valid(event, key):
                event["cutouts_prepared"] = False
                missing_cuts += 1
        for part in job.get("concat_parts", []):
            with self.cache.pin([part["key"]]) as available:
                directory = available.get(part["key"])
                try:
                    if directory is None:
                        raise IOError("Missing concat cache")
                    self._validate_concat(directory/"recording.mp4", part["expected"])
                except (IOError, OSError, cv2.error):
                    part["prepared"] = False
                    part.pop("unavailable", None)
                    self.cache.clear([part["key"]])
                    missing_parts += 1
        self._save(job)
        self.status(f"[自動作成] 起動時のキャッシュ確認: タイムラプス {missing_samples}枚・切り抜き {missing_cuts}件・連結 {missing_parts}区間を補完")

    def _concatenate(self, job, output):
        """Reuse prepared parts; publish/validate before deleting any input."""
        self._assert_capture_ready(job)
        paths = list(job["archives"])
        if not paths:
            raise IOError("連結する録画がありません")
        expected = job.get("concat_expected")
        if expected is None:
            expected = self._concat_plan(job, paths)
            job["concat_expected"] = expected
            self._save(job)
        if not output.exists():
            with self.cache.pin([part["key"] for part in job.get("concat_parts", [])]) as available:
                inputs, ranges = [], []
                covered = []
                for part in job.get("concat_parts", []):
                    directory = available.get(part["key"])
                    valid = False
                    if directory is not None:
                        try:
                            self._validate_concat(directory/"recording.mp4", part["expected"])
                            valid = True
                        except (IOError, OSError, cv2.error):
                            pass
                    if valid:
                        inputs.append(str(directory/"recording.mp4"))
                        ranges.append([0, part["expected"]["count"]])
                    else:
                        inputs.extend(part["sources"])
                        ranges.extend(part["expected"]["ranges"])
                    covered.extend(part["sources"])
                if covered != paths:
                    inputs, ranges = paths, expected["ranges"]
                render_plan = dict(expected, ranges=ranges)
                with tempfile.TemporaryDirectory(prefix=".concat_", dir=output.parent) as temporary:
                    joined = Path(temporary)/"joined.mp4"
                    self._render_concat(inputs, render_plan, joined)
                    if self.cancel.is_set():
                        raise InterruptedError("連結を中断しました。元録画は保持します")
                    captured_at = datetime.fromisoformat(self._timing(paths[0])["start_time"]) + timedelta(
                        seconds=expected["ranges"][0][0]/expected["fps"])
                    rtsp_live.write_json(str(output)+".capture.json", dict(start_time=captured_at.isoformat(),
                        fps=expected["fps"], frame_count=expected["count"], size=expected["size"], sources=paths))
                    os.replace(joined, output)
        self._validate_concat(output, expected)
        self._assert_capture_ready(job)
        job["outputs"]["concatenate"] = str(output)
        job["state"] = "replacing"
        self._save(job)
        for path in expected["deletable"]:
            Path(path).unlink(missing_ok=True)
            for suffix in (".capture.json", ".preprocessed.json"):
                Path(path+suffix).unlink(missing_ok=True)

    @staticmethod
    def _validate_concat(path, expected):
        fps, count, size = rtsp_segments._properties(path)
        if count != expected["count"] or abs(fps-expected["fps"]) > .05 or list(size) != expected["size"]:
            raise IOError("連結動画の検証に失敗しました。元動画を保持します")
        # Decode both ends as well as inspecting the container metadata.
        cap = open_video_capture(str(path))
        try:
            for index in (0, count-1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                if not cap.read()[0]:
                    raise IOError("連結動画の端のフレームを読めません。元動画を保持します")
        finally:
            cap.release()

    def _finish(self, job):
        self._assert_capture_ready(job)
        output = self._output_root(job)
        if job["state"] == "replacing":
            self._concatenate(job, Path(job["outputs"]["concatenate"]))
        else:
            if job["options"]["timelapse"] and "timelapse" not in job["outputs"]:
                with tempfile.TemporaryDirectory(prefix=".timelapse_", dir=output) as temporary:
                    path = Path(temporary) / "timelapse.mp4"
                    key = self._timelapse_key(job)
                    with self.cache.pin([key]):
                        prepared = self.cache.root / key / "timelapse.mp4"
                        if prepared.exists():
                            shutil.copyfile(prepared, path)
                            fps, count, _ = rtsp_segments._properties(path)
                            if count != round(job["options"]["duration"] * job["options"]["fps"]) or abs(fps-job["options"]["fps"]) > .05:
                                raise IOError("準備済みタイムラプスのフレーム数/FPSが一致しません")
                        else:
                            self.status(f"[自動作成] タイムラプス素材更新のため再生成: {key}")
                            self._timelapse(job, path)
                    os.replace(path, output / path.name)
                job["outputs"]["timelapse"] = str(output / "timelapse.mp4")
                self._save(job)
            if (job["options"]["blend_video"] or job["options"]["blend_image"]) and not job.get("blends_done"):
                with tempfile.TemporaryDirectory(prefix=".products_", dir=output) as temporary:
                    results = self._blends(job, Path(temporary))
                    for kind, source in results.items():
                        target = output / Path(source).name
                        os.replace(source, target)
                        job["outputs"][kind] = str(target)
                job["blends_done"] = True
                self._save(job)
            if job["options"]["concatenate"]:
                self._concatenate(job, output / "recording.mp4")
        self._assert_capture_ready(job)
        job["state"] = "complete"
        self._save(job)
        self.source_indexes.pop(job["id"], None)
        self.sample_gaps.pop(job["id"], None)
        self.timelapse_revisions.pop(job["id"], None)
        self.prepared_timelapses.pop(job["id"], None)
        self.status(f"[自動作成] スケジュール終了後の作成完了: {output}")

    def tick(self, now=None):
        now = now or datetime.now()
        if self.capture_setup_error:
            raise IOError(f"流星素材のジョブ登録を復旧するまで元録画を保持します: {self.capture_setup_error}")
        if not self.existing_only and schedule_window(now, *self.clock):
            self._job(now)
        for job in list(self.jobs.values()):
            if job["state"] == "complete":
                continue
            if job["state"] == "replacing":
                self._finish(job)
                continue
            self._archives(job)
            if job["id"] not in self.checked_jobs:
                self._check_existing_cache(job)
                self.checked_jobs.add(job["id"])
            self._recover_captures(job)
            # Do not wait for detection to end before preparing shared crops.
            pending = [event for event in list(job["events"]) if not event.get("cutouts_prepared") or
                       event.get("crop_method") != self._crop_method(job)]
            for event in pending[:2]:
                self._prepare_event(job, event)
            self._prepare_samples(job)
            self._prepare_concatenation(job)
            if now >= datetime.fromisoformat(job["end"]):
                self.flush_archives()
                self._archives(job)
                self._prepare_samples(job)
                self._prepare_timelapse(job)
                if not self._drained(job):
                    continue
                self._archives(job)
                self._prepare_samples(job)
                if not job["archives"]:
                    self._assert_capture_ready(job)
                    job["state"] = "complete"
                    self._save(job)
                    self.status("[自動作成] スケジュール内の録画がありません")
                    continue
                self._prepare_concatenation(job, max_builds=len(job["archives"])+1)
                job["state"] = "rendering"
                self._save(job)
                self._finish(job)

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        def run():
            while not self.cancel.is_set():
                try:
                    self.tick()
                except Exception as exc:
                    self.status(f"[自動作成] 再試行予定（元録画は保持）: {exc}")
                self.cancel.wait(2)
        self.thread = threading.Thread(target=run, daemon=True, name="RTSPScheduleProducts")
        self.thread.start()

    def stop(self):
        self.cancel.set()
        if self.thread:
            self.thread.join(timeout=5)


_SERVICE_LOCK = threading.RLock()
_SERVICES = {}


def pending_recording_products(root, status):
    """Keep detecting for saved windows when unsaved UI switches reset to OFF."""
    with _SERVICE_LOCK:
        service = _SERVICES.get(str(Path(root).resolve()))
        if service is not None:
            with service.lock:
                service.existing_only = True
                if any(job["state"] != "complete" for job in service.jobs.values()):
                    service.status = status
                    return service
    return None


def recording_products(root, options, clock, status, mask=None):
    """Attach recording to the startup worker rather than launching a duplicate."""
    key = str(Path(root).resolve())
    with _SERVICE_LOCK:
        service = _SERVICES.get(key)
        if service is None:
            service = ScheduleProducts(root, options, clock, threading.Event(), status, mask)
            _SERVICES[key] = service
        with service.lock:
            service.options = normalize_options(options)
            service.clock = clock
            service.status = status
            service.mask = None if mask is None else mask.copy()
            service.existing_only = False
        service.start()
        return service


def resume_pending_products(root, status=print):
    """On app launch, verify/rebuild unfinished jobs using their saved settings."""
    key = str(Path(root).resolve())
    with _SERVICE_LOCK:
        if key in _SERVICES:
            return _SERVICES[key]
        pending = []
        for marker in (Path(root)/".schedule_jobs").glob("*/job.json"):
            try:
                job = json.loads(marker.read_text(encoding="utf-8"))
                if job["state"] != "complete" and job["id"] not in rtsp_startup.deferred_job_ids(root):
                    pending.append(job)
            except (OSError, ValueError, KeyError) as exc:
                status(f"[自動作成] 復旧情報を読み取れません。元録画は保持: {exc}")
                return None
        if not pending:
            return None
        first = pending[0]
        start, end = datetime.fromisoformat(first["start"]), datetime.fromisoformat(first["end"])
        service = ScheduleProducts(root, first["options"],
            (start.hour, start.minute, end.hour, end.minute), threading.Event(), status)
        service.existing_only = True
        _SERVICES[key] = service
        service.start()
        return service


def stop_background_products():
    with _SERVICE_LOCK:
        services = list(_SERVICES.values())
        _SERVICES.clear()
    for service in services:
        service.cancel.set()
    for service in services:
        service.stop()
