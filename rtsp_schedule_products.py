"""Durable scheduled RTSP products, prepared independently of recording.

Only disposable entries live in the bounded cache. Job manifests and original
archives remain outside it so eviction or a manual clear can be recovered.
"""
from __future__ import annotations

from bisect import bisect_right
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
from video_io import open_video_capture

CACHE_LIMIT = 150_000_000_000
_CACHE_LOCK = threading.RLock()
DEFAULTS = dict(timelapse=False, blend_video=False, blend_image=False, concatenate=False,
                duration=15.0, fps=60.0, constellations=True, timestamp=True,
                timestamp_position="bottom_right", timestamp_size=3.5, mean_radius=0,
                crop_margin=20, sky_config="", output_dir="")


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


class ProductCache:
    """Atomic cache entries with oldest-created-first eviction and shared locking."""
    def __init__(self, root, limit=CACHE_LIMIT):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.limit = min(CACHE_LIMIT, max(0, int(limit)))
        with _CACHE_LOCK:
            # A crashed writer can leave an unpublished entry. It is disposable.
            for temporary in self.root.glob(".entry_*"):
                if temporary.is_dir():
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
        for entry in self.entries():
            if used + incoming <= self.limit:
                break
            size = sum(p.stat().st_size for p in entry.rglob("*") if p.is_file())
            shutil.rmtree(entry)
            used -= size

    def put(self, key, files):
        """Files are already encoded bytes; no transient on-disk quota overshoot."""
        size = sum(len(data) for data in files.values())
        with _CACHE_LOCK:
            if size > self.limit:
                return False
            target = self.root / key
            if target.exists():
                return True
            self._trim(size)
            temporary = Path(tempfile.mkdtemp(prefix=".entry_", dir=self.root))
            try:
                for name, data in files.items():
                    (temporary / name).write_bytes(data)
                temporary.rename(target)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
            return True

    def read(self, key, name):
        with _CACHE_LOCK:
            path = self.root / key / name
            return path.read_bytes() if path.is_file() else None

    def clear(self, keys=None):
        with _CACHE_LOCK:
            allowed = set(keys) if keys is not None else None
            for entry in self.entries():
                if allowed is None or entry.name in allowed:
                    shutil.rmtree(entry)


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
        self.ai_unavailable = None
        setup_error = self.job_root / "capture_setup.failed"
        self.capture_setup_error = setup_error.read_text(encoding="utf-8") if setup_error.exists() else None
        if any(not (guard.parent / "job.json").exists() for guard in self.job_root.glob("*/capture.guard")):
            self.capture_setup_error = "流星素材のジョブ登録が中断されています"
        for path in self.job_root.glob("*/job.json"):
            job = json.loads(path.read_text(encoding="utf-8"))
            if job["state"] != "complete":
                self.jobs[job["id"]] = job
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

    def capture_event(self, frames, frame_rate, started_at, line):
        """Persist a small lossless line crop before detector buffers are reused.

        VLM inference and all rendering run on the product worker. A cache miss
        can reconstruct these frames from original archives using this clock.
        """
        if not (self.options["blend_image"] or self.options["blend_video"]) or not frames:
            return
        with self.lock:
            job = self._job(started_at + timedelta(seconds=(len(frames)-1)/frame_rate/2))
            if job is None or job["state"] == "complete":
                return
            size = [frames[0].shape[1], frames[0].shape[0]]
            rect = line_rect(line, size, self.options["crop_margin"])
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
            window = schedule_window(started_at + timedelta(seconds=(len(frames)-1)/frame_rate/2), *self.clock)
            if window is None:
                return
            start, end = window
            key = start.strftime("%Y%m%dT%H%M") + "_" + end.strftime("%Y%m%dT%H%M")
            job = self.jobs.get(key)
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
                if not path.is_file() or not rtsp_segments.is_finalized_mp4(path):
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
        if source not in self.source_data:
            self.source_data[source] = rtsp_live.timing(source)
        return self.source_data[source]

    def _source_frame(self, job, when, radius=0):
        sources = job["archives"]
        starts = [datetime.fromisoformat(self._timing(path)["start_time"]) for path in sources]
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
        frame = decode_png(self.cache.read(event["key"], f"{index:06d}.png"))
        if frame is not None:
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

    def _prepare_event(self, job, event):
        if event["ai_done"]:
            return
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
        self._save(job)

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
        # Sampling uses the full schedule clock, never a dusk/dawn filter.
        for index in range(count):
            if self.cancel.is_set():
                return
            key = job["id"] + f"_sample_{index:06d}"
            if (self.cache.root / key / "frame.png").is_file():
                continue
            when = start + (end-start)*(index/max(1, count-1))
            if index == count-1:
                when = end-timedelta(milliseconds=40)
            frame = self._source_frame(job, when, options["mean_radius"])
            if frame is None:
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
            if "_innovation" in path.name or "_temp_" in path.name:
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
            for position in np.linspace(0, len(available)-1, count):
                index = available[int(round(position))]
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
                yield frame
        self._write_video(output, frames(), options["fps"])

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
                self._prepare_event(job, event)
                x, y, _, _ = event["rect"]
                frame_count = max(1, round(event["count"]*fps/event["fps"]))
                count = max(count, frame_count)
                for index in range(frame_count):
                    if self.cancel.is_set():
                        raise InterruptedError("比較明合成を中断しました")
                    crop = self._event_frame(job, event, min(event["count"]-1, int(index*event["fps"]/fps)))
                    keep = np.zeros(crop.shape[:2], np.uint8)
                    for a, b, c, d in event["boxes"]:
                        keep[b:d, a:c] = 255
                    crop = cv2.bitwise_and(crop, crop, mask=keep)
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

    def _concatenate(self, job, output):
        """Publish and journal a validated result before deleting any input.

        Whole compatible archives use stream copy. Rare boundary overruns or
        changed codecs use a frame-accurate encode; out-of-window originals are
        retained, since they contain footage outside this replacement scope.
        """
        self._assert_capture_ready(job)
        paths = list(job["archives"])
        if not paths:
            raise IOError("連結する録画がありません")
        expected = job.get("concat_expected")
        if expected is None:
            start, end = datetime.fromisoformat(job["start"]), datetime.fromisoformat(job["end"])
            properties = [rtsp_segments._properties(path) for path in paths]
            fps, _, size = properties[0]
            if any(abs(rate-fps) > .05 or dimensions != size for rate, _, dimensions in properties):
                raise IOError("録画のFPSまたは解像度が途中で変わっています。元動画を保持します")
            ranges, deletable = [], []
            for path, (rate, count, _) in zip(paths, properties):
                when = datetime.fromisoformat(self._timing(path)["start_time"])
                first = max(0, math.ceil((start-when).total_seconds()*rate-1e-6))
                last = min(count, math.ceil((end-when).total_seconds()*rate-1e-6))
                ranges.append([first, last])
                if first == 0 and last == count:
                    deletable.append(path)
            expected = dict(fps=fps, count=sum(last-first for first, last in ranges),
                            size=list(size), ranges=ranges, deletable=deletable)
            job["concat_expected"] = expected
            self._save(job)
        if not output.exists():
            with tempfile.TemporaryDirectory(prefix=".concat_", dir=output.parent) as temporary:
                listing = Path(temporary) / "inputs.txt"
                listing.write_text("".join("file '" + str(Path(path).resolve()).replace("'", "'\\''") + "'\n" for path in paths), encoding="utf-8")
                joined = Path(temporary) / "joined.mp4"
                full_archives = len(expected["deletable"]) == len(paths)
                copied = False
                signatures = [self._stream_signature(path) for path in paths] if full_archives else []
                compatible = signatures and signatures[0] is not None and all(signature == signatures[0] for signature in signatures)
                if full_archives and compatible:
                    result = subprocess.run([shutil.which("ffmpeg") or "ffmpeg", "-nostdin", "-v", "error", "-y",
                        "-f", "concat", "-safe", "0", "-i", str(listing), "-map", "0:v:0", "-an", "-c", "copy",
                        "-movflags", "+faststart", str(joined)], capture_output=True, text=True)
                    if not result.returncode:
                        try:
                            self._validate_concat(joined, expected)
                            copied = True
                        except IOError:
                            pass
                if not copied:
                    self.status("[自動作成] スケジュールの境界をフレーム単位で整えて連結します")
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
                    self._write_video(joined, frames(), expected["fps"])
                    self._validate_concat(joined, expected)
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
        self.status(f"[自動作成] スケジュール終了後の作成完了: {output}")

    def tick(self, now=None):
        now = now or datetime.now()
        if self.capture_setup_error:
            raise IOError(f"流星素材のジョブ登録を復旧するまで元録画を保持します: {self.capture_setup_error}")
        if schedule_window(now, *self.clock):
            self._job(now)
        for job in list(self.jobs.values()):
            if job["state"] == "complete":
                continue
            if job["state"] == "replacing":
                self._finish(job)
                continue
            self._archives(job)
            self._recover_captures(job)
            # Do not wait for detection to end before preparing shared crops.
            for event in list(job["events"]):
                if not event["ai_done"]:
                    self._prepare_event(job, event)
            self._prepare_samples(job)
            if now >= datetime.fromisoformat(job["end"]):
                self.flush_archives()
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
                job["state"] = "rendering"
                self._save(job)
                self._finish(job)

    def start(self):
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
        if self.thread:
            self.thread.join(timeout=5)
