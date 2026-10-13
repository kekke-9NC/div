"""Continuous detection from a durable subsecond spool, with minute archives.

The recorder never waits for detection. The detector sees one frame clock per
connection, not a succession of short movies. Archives use stream-copy and
spool files are removed only after both publication and decoder consumption.
"""
from collections import deque
import csv
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import tempfile
import threading
import time

import cv2

import config
import media_time
import rtsp_segments
import rtsp_startup
import temporal_mean
from fixed_pattern import apply_fixed_pattern_correction
from video_io import open_video_capture


def write_json(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=".meteor_meta_", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def capture_directory(root, when=None):
    when = when or datetime.now()
    directory = Path(root) / ".rtsp_live" / when.strftime("%Y%m%d/%H") / (
        when.strftime("%M_%S_") + rtsp_segments.new_session_id())
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / ".session.json", {"start_time": when.isoformat()})
    return directory


def capture_options(directory):
    return ["-segment_list", str(Path(directory) / "segments.csv"),
            "-segment_list_type", "csv"]


def timing(path):
    marker = Path(str(path) + ".capture.json")
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        started, _ = media_time.get_media_start_time(str(path))
        if started is None:
            raise IOError(f"No capture clock: {path}")
        data = {"start_time": started.isoformat()}
    if not all(key in data for key in ("fps", "frame_count", "size")):
        fps, count, size = rtsp_segments._properties(path)
        data.update(fps=fps, frame_count=count, size=list(size))
        write_json(marker, data)
    return data


class ChunkCatalog:
    """Read only newly appended, closed FFmpeg CSV records."""
    def __init__(self, root, ready=None):
        self.root = Path(root) / ".rtsp_live"
        self.ready = ready or (lambda _path: True)
        self.csv_offsets = {}
        self.entries = {}

    def _stamp_csv(self, listing):
        offset = self.csv_offsets.get(listing, 0)
        with listing.open(encoding="utf-8") as handle:
            handle.seek(offset)
            while True:
                position = handle.tell()
                line = handle.readline()
                if not line or not line.endswith("\n"):
                    self.csv_offsets[listing] = position
                    break
                row = next(csv.reader([line]))
                if len(row) < 3:
                    continue
                path = listing.parent / Path(row[0]).name
                if not path.is_file():
                    continue
                session_path = listing.parent / ".session.json"
                session = json.loads(session_path.read_text(encoding="utf-8"))
                if "first_pts" not in session:
                    session.update(first_pts=float(row[1]), start_time=(
                        media_time._filesystem_creation_time(str(path))
                        or datetime.fromisoformat(session["start_time"])).isoformat())
                    write_json(session_path, session)
                started = datetime.fromisoformat(session["start_time"]) + timedelta(
                    seconds=float(row[1]) - session["first_pts"])
                if not Path(str(path) + ".capture.json").exists():
                    write_json(str(path) + ".capture.json", {"start_time": started.isoformat()})

    def paths(self):
        if not self.root.exists():
            return []
        for listing in self.root.rglob("segments.csv"):
            try:
                self._stamp_csv(listing)
            except (OSError, ValueError, KeyError):
                continue
        found = []
        present = set()
        for path in self.root.rglob("chunk_*.mp4"):
            if "_temp_" in path.name or "_innovation" in path.name or rtsp_startup.is_deferred_recording(path):
                continue
            try:
                present.add(path)
                if not self.ready(path):
                    continue
                # A CSV recorder publishes timing only after closing the file.
                if (path.parent / "segments.csv").exists() and not Path(
                        str(path) + ".capture.json").exists():
                    continue
                marker = Path(str(path) + '.capture.json')
                stat = path.stat()
                stamp = marker.stat() if marker.exists() else None
                identity = (stat.st_ino, stat.st_size, stat.st_mtime_ns,
                            (stamp.st_size, stamp.st_mtime_ns) if stamp else None)
                cached = self.entries.get(path)
                if cached is None or cached[0] != identity:
                    if not rtsp_segments.is_finalized_mp4(path):
                        continue
                    data = timing(path)
                    self.entries[path] = (identity, datetime.fromisoformat(data['start_time']))
                found.append((self.entries[path][1], path))
            except (OSError, ValueError, cv2.error):
                continue
        self.entries = {path: entry for path, entry in self.entries.items() if path in present}
        return [path for _, path in sorted(found)]


class SegmentCapture:
    """VideoCapture interface over a queue of finalized, adjacent chunks."""
    def __init__(self, first, incoming, cancel, consumed, evidence=False,
                 mean_frames=0, correction=None, idle_seconds=2.0, track_chunks=False):
        self.continuous_rtsp = True
        self.analysis_final_frame = False
        self.first_path = Path(first)
        self.incoming = incoming
        self.cancel = cancel
        self.consumed = consumed
        self.pending = None
        self.path = self.first_path
        self.data = timing(self.path)
        self.started_at = datetime.fromisoformat(self.data["start_time"])
        self.fps = self.data["fps"]
        self.size = tuple(self.data["size"])
        self.evidence = evidence
        self.last_evidence = None
        self.cap = None
        self.evidence_cap = None
        self.read_count = 0
        self.opened = True
        self.eof = False
        self.input_ended = False
        self.idle_seconds = idle_seconds
        self.mean = temporal_mean.TemporalMeanStream(mean_frames) if mean_frames else None
        self.correction = correction
        self.outputs = deque()
        self.decoded = deque()
        self.total_input_frames = 0
        self.track_chunks = track_chunks
        self.new_chunks = deque([(self.path, dict(self.data))] if track_chunks else [])

    def isOpened(self):
        return self.opened

    def get(self, property_id):
        return {cv2.CAP_PROP_FPS: self.fps, cv2.CAP_PROP_FRAME_WIDTH: self.size[0],
                cv2.CAP_PROP_FRAME_HEIGHT: self.size[1],
                cv2.CAP_PROP_FRAME_COUNT: 0}.get(property_id, 0)

    def _next_path(self):
        deadline = time.monotonic() + self.idle_seconds
        while not self.cancel.is_set() and time.monotonic() < deadline:
            try:
                successor = Path(self.incoming.get(timeout=0.1))
            except queue.Empty:
                continue
            data = timing(successor)
            expected = datetime.fromisoformat(self.data["start_time"]) + timedelta(
                seconds=self.data["frame_count"] / self.fps)
            adjacent = (successor.parent == self.path.parent
                        and tuple(data["size"]) == self.size
                        and abs(data["fps"] - self.fps) < 0.05
                        and abs((datetime.fromisoformat(data["start_time"]) - expected
                                 ).total_seconds()) <= max(0.1, 2 / self.fps))
            if not adjacent:
                self.pending = successor
                return False
            self.path, self.data = successor, data
            if self.track_chunks:
                self.new_chunks.append((successor, dict(data)))
            self.read_count = 0
            return True
        return False

    def _read_native(self):
        while not self.cancel.is_set():
            if self.cap is None:
                self.cap = open_video_capture(str(self.path))
                if self.evidence:
                    self.evidence_cap = open_video_capture(str(
                        self.path.with_name(self.path.stem + "_innovation.mp4")))
            ok, frame = self.cap.read()
            if ok:
                self.read_count += 1
                self.total_input_frames += 1
                if self.evidence_cap is not None:
                    evidence_ok, self.last_evidence = self.evidence_cap.read()
                    if not evidence_ok:
                        raise IOError(f"Incomplete innovation stream: {self.path}")
                return True, frame
            self.cap.release()
            self.cap = None
            if self.evidence_cap is not None:
                self.evidence_cap.release()
                self.evidence_cap = None
            if self.read_count != self.data["frame_count"]:
                raise IOError(f"Incomplete chunk: {self.path}")
            self.decoded.append((self.path, self.total_input_frames))
            if not self._next_path():
                break
        return False, None

    def read(self):
        while not self.outputs and not self.input_ended:
            ok, frame = self._read_native()
            if not ok:
                self.input_ended = True
                if self.mean:
                    self.outputs.extend(self.mean.flush())
                break
            if self.mean:
                if self.correction is not None:
                    correction = cv2.resize(self.correction, self.size)
                    frame = apply_fixed_pattern_correction(frame, correction)
                self.outputs.extend(self.mean.push(frame))
            else:
                self.outputs.append(frame)
        if self.outputs:
            return True, self.outputs.popleft()
        self.eof = True
        return False, None

    def acknowledge(self, end_frame, final=False):
        # Retain the recent tail on interruption: an active track can begin
        # before the latest difference window has emitted its final candidate.
        retain = int(round(self.fps * (config.FINER_DETECT_WINDOW_SECONDS
                                      + config.FINER_DETECT_PADDING_SECONDS)))
        watermark = end_frame + 1 - (0 if final else retain)
        while self.decoded and self.decoded[0][1] <= watermark:
            path, _ = self.decoded.popleft()
            self.consumed(path)

    def release(self):
        self.opened = False
        for capture in (self.cap, self.evidence_cap):
            if capture is not None:
                capture.release()


class EvidenceCapture:
    def __init__(self, main):
        self.main = main

    def read(self):
        return self.main.last_evidence is not None, self.main.last_evidence

    def release(self):
        pass


class MinuteArchiver:
    def __init__(self, root, status=None, seconds=config.RTSP_ARCHIVE_DURATION, ready=None):
        self.root = Path(root)
        self.status = status or print
        self.seconds = seconds
        self.catalog = ChunkCatalog(root, ready)
        self.recovered = set()

    def _recover(self):
        for marker in self.root.glob("[0-9]*/[0-9]*/*_minute_*.mp4.capture.json"):
            output = Path(str(marker)[:-len(".capture.json")])
            if output in self.recovered or not rtsp_segments.is_finalized_mp4(output):
                continue
            data = json.loads(marker.read_text(encoding="utf-8"))
            for relative in data.get("sources", []):
                path = self.root / relative
                if path.exists():
                    write_json(str(path) + ".archived.json", {"archive": str(output.relative_to(self.root))})
            self.recovered.add(output)

    def _publish(self, paths):
        first = timing(paths[0])
        when = datetime.fromisoformat(first["start_time"])
        directory = self.root / when.strftime("%Y%m%d/%H")
        directory.mkdir(parents=True, exist_ok=True)
        session = paths[0].parent.name.rsplit("_", 1)[-1]
        sequence = paths[0].stem.split("_")[1]
        output = directory / f"{when:%M_%S}_{session}_minute_{sequence}.mp4"
        # A published manifest is also a crash-recovery transaction record.
        marker = Path(str(output) + ".capture.json")
        relative = [str(path.relative_to(self.root)) for path in paths]
        if not output.exists():
            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                raise IOError("FFmpeg is required for minute archives; spool retained")
            with tempfile.TemporaryDirectory(prefix=".rtsp_concat_", dir=directory) as temporary:
                listing = Path(temporary) / "inputs.txt"
                listing.write_text("".join("file '" + str(path.resolve()).replace(
                    "'", "'\\''") + "'\n" for path in paths), encoding="utf-8")
                joined = Path(temporary) / "minute.mp4"
                result = subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-y",
                    "-f", "concat", "-safe", "0", "-i", str(listing), "-map", "0:v:0",
                    "-an", "-c", "copy", "-movflags", "+faststart", str(joined)],
                    capture_output=True, text=True, timeout=120)
                if result.returncode:
                    raise IOError(result.stderr[-500:])
                fps, count, size = rtsp_segments._properties(joined)
                expected = sum(timing(path)["frame_count"] for path in paths)
                if count != expected or abs(fps - first["fps"]) > .05 or list(size) != first["size"]:
                    raise IOError("Minute archive validation failed; spool retained")
                write_json(marker, dict(first, frame_count=count, sources=relative))
                preprocessing = Path(str(paths[0]) + ".preprocessed.json")
                if preprocessing.exists():
                    data = json.loads(preprocessing.read_text(encoding="utf-8"))
                    data["analyzed"] = True
                    write_json(str(output) + ".preprocessed.json", data)
                os.replace(joined, output)
        else:
            data = json.loads(marker.read_text(encoding="utf-8"))
            if data.get("sources") != relative or not rtsp_segments.is_finalized_mp4(output):
                raise IOError(f"Conflicting minute archive: {output}")
        for path in paths:
            write_json(str(path) + ".archived.json", {"archive": str(output.relative_to(self.root))})
        self.status(f"[RTSP保存] 1分アーカイブ保存: {output.name} ({len(paths)}区間)")

    def _cleanup(self, path):
        consumed = Path(str(path) + ".analysis_consumed.json")
        archived = Path(str(path) + ".archived.json")
        if not consumed.exists() or not archived.exists():
            return
        archive = self.root / json.loads(archived.read_text(encoding="utf-8"))["archive"]
        if not archive.exists() or not rtsp_segments.is_finalized_mp4(archive):
            return
        for item in (path, path.with_name(path.stem + "_innovation.mp4"),
                     *(Path(str(path) + suffix) for suffix in (
                         ".capture.json", ".preprocessed.json", ".analysis_pending.json",
                         ".analysis_consumed.json", ".archived.json"))):
            item.unlink(missing_ok=True)

    def compact(self, flush=False):
        self._recover()
        paths = self.catalog.paths()
        groups = {}
        for path in paths:
            archived = Path(str(path) + ".archived.json")
            if archived.exists():
                target = self.root / json.loads(archived.read_text(encoding='utf-8'))['archive']
                if target.exists():
                    self._cleanup(path)
                    continue
                # A stale publication marker is not proof that pixels survive.
                # Remove only that marker, then republish the retained originals
                # and validate all frames before normal consumed cleanup.
                archived.unlink()
                self.status(f"[RTSP保存] 欠落した連結先を元区間から復旧: {target.name}")
            groups.setdefault(path.parent, []).append(path)
        for session_paths in groups.values():
            batch, duration = [], 0.0
            for path in sorted(session_paths):
                data = timing(path)
                if batch:
                    last = timing(batch[-1])
                    expected = datetime.fromisoformat(last["start_time"]) + timedelta(
                        seconds=last["frame_count"] / last["fps"])
                    gap = abs((datetime.fromisoformat(data["start_time"]) - expected).total_seconds())
                    if (gap > max(.1, 2 / last["fps"]) or data["fps"] != last["fps"]
                            or data["size"] != last["size"]):
                        self._publish(batch)
                        for old in batch:
                            self._cleanup(old)
                        batch, duration = [], 0.0
                batch.append(path)
                duration += data["frame_count"] / data["fps"]
                if duration + 1e-6 >= self.seconds:
                    self._publish(batch)
                    for old in batch:
                        self._cleanup(old)
                    batch, duration = [], 0.0
            # Recover an old, disconnected session, or publish a stop's tail.
            if batch and (flush or time.time() - batch[-1].stat().st_mtime > 60):
                self._publish(batch)
                for old in batch:
                    self._cleanup(old)


class RealtimeRunner:
    def __init__(self, root, analyze, cancel, status=None, ready=None,
                 evidence=False, mean_frames=0, correction=None):
        self.cancel = cancel
        self.analyze = analyze
        self.status = status or print
        self.catalog = ChunkCatalog(root, ready)
        self.archiver = MinuteArchiver(root, self.status)
        self.evidence = evidence
        self.mean_frames = mean_frames
        self.correction = correction
        self.flush_requested = threading.Event()
        self.queue = queue.Queue()
        self.seen = set()
        self.seen_lock = threading.Lock()
        self.threads = []

    def _scan(self):
        last_status = 0.0
        while not self.cancel.is_set():
            with self.seen_lock:
                self.seen = {path for path in self.seen if path.exists()}
            for path in self.catalog.paths():
                with self.seen_lock:
                    if path not in self.seen and not Path(str(path) + ".analysis_consumed.json").exists():
                        self.seen.add(path)
                        self.queue.put(path)
            if self.queue.qsize() >= 10 and time.monotonic() - last_status >= 10:
                self.status(f"[RTSP解析] 解析待ち: {self.queue.qsize()}区間（録画は継続）")
                last_status = time.monotonic()
            self.cancel.wait(config.RTSP_SCAN_INTERVAL)

    def _consume(self, path):
        write_json(str(path) + ".analysis_consumed.json", {"consumed": True})

    def _detect(self):
        pending = None
        while not self.cancel.is_set():
            if pending is None:
                try:
                    pending = self.queue.get(timeout=.1)
                except queue.Empty:
                    continue
            capture = SegmentCapture(pending, self.queue, self.cancel, self._consume,
                evidence=self.evidence, mean_frames=self.mean_frames, correction=self.correction)
            try:
                self.analyze(capture, EvidenceCapture(capture) if self.evidence else None)
            except Exception as exc:
                self.status(f"[RTSP解析] 連続解析エラー（録画は保持）: {exc}")
            finally:
                capture.release()
            if not capture.eof and not self.cancel.is_set():
                with self.seen_lock:
                    self.seen.discard(capture.path)
                self.cancel.wait(.5)
            pending = capture.pending

    def _archive(self):
        while not self.cancel.is_set():
            try:
                flush = self.flush_requested.is_set()
                self.flush_requested.clear()
                self.archiver.compact(flush=flush)
            except Exception as exc:
                self.status(f"[RTSP保存] 連結再試行予定（元動画は保持）: {exc}")
            self.cancel.wait(1)

    def start(self):
        self.threads = [threading.Thread(target=target, daemon=True, name=name)
            for name, target in (("RTSPContinuousDetection", self._detect),
                                 ("RTSPChunkDiscovery", self._scan),
                                 ("RTSPMinuteArchive", self._archive))]
        for thread in self.threads:
            thread.start()

    def stop(self):
        self.cancel.set()
        for thread in self.threads:
            thread.join(timeout=5)
        # Do not race an archive command that is still finishing.
        if not self.threads[-1].is_alive():
            try:
                self.archiver.compact(flush=True)
            except Exception as exc:
                self.status(f"[RTSP保存] 未連結の録画を保持: {exc}")
