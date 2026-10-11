"""Capture timing and bounded, lossless analysis windows for RTSP archives.

Archive files never overlap. Only the temporary detection window includes
neighbouring frames; events belong to the segment containing their first frame.
"""

from contextlib import contextmanager
from datetime import datetime, timedelta
import json
from pathlib import Path
import re
import struct
import tempfile
import time
import uuid

import cv2

import media_time
from video_io import open_video_capture


_NAME = re.compile(r"^\d{2}_\d{2}_([0-9a-f]{12})$")


class SegmentNotReady(Exception):
    """The next segment is still being recorded or transformed."""


def new_session_id():
    return uuid.uuid4().hex[:12]


def write_capture_time(path, started_at):
    marker = Path(str(path) + ".capture.json")
    temporary = marker.with_suffix(marker.suffix + ".tmp")
    temporary.write_text(json.dumps({"start_time": started_at.isoformat()}), encoding="utf-8")
    temporary.replace(marker)


def is_segment(path):
    return bool(_NAME.fullmatch(Path(path).stem))


def is_finalized_mp4(path):
    """Check for a closed MP4 index without opening an unfinished decoder."""
    try:
        with open(path, "rb") as handle:
            length = handle.seek(0, 2)
            handle.seek(0)
            while handle.tell() + 8 <= length:
                start = handle.tell()
                size, kind = struct.unpack(">I4s", handle.read(8))
                if size == 1:
                    if handle.tell() + 8 > length:
                        return False
                    size = struct.unpack(">Q", handle.read(8))[0]
                if size < 8 or start + size > length:
                    return False
                if kind == b"moov":
                    return True
                handle.seek(start + size)
    except (OSError, struct.error):
        pass
    return False


def _properties(path):
    cap = open_video_capture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if not cap.isOpened() or fps <= 0 or count <= 0:
            raise IOError(f"Invalid RTSP segment: {path}")
        return fps, count, size
    finally:
        cap.release()


def _neighbours(path, ready_paths=None):
    path = Path(path)
    match = _NAME.fullmatch(path.stem)
    if not match:
        return None, None
    timestamp = media_time._path_time(str(path))
    if timestamp is None:
        return None, None
    if ready_paths is None:
        # Inspect only nearby hour directories, never the whole archive.
        root = path.parent.parent.parent
        candidates = []
        for offset in (-1, 0, 1):
            hour = timestamp + timedelta(hours=offset)
            directory = root / hour.strftime("%Y%m%d") / hour.strftime("%H")
            candidates.extend(directory.glob(f"*_{match.group(1)}.mp4"))
    else:
        candidates = map(Path, ready_paths)
    ordered = sorted({p for p in candidates if _NAME.fullmatch(p.stem)
                      and _NAME.fullmatch(p.stem).group(1) == match.group(1)})
    if path not in ordered:
        return None, None
    index = ordered.index(path)
    return (ordered[index - 1] if index else None,
            ordered[index + 1] if index + 1 < len(ordered) else None)


def _adjacent(first, second, properties):
    if first is None or second is None:
        return False
    a, _ = media_time.get_media_start_time(str(first))
    b, _ = media_time.get_media_start_time(str(second))
    if a is None or b is None:
        return False
    fps, count, size = properties[first]
    other_fps, _, other_size = properties[second]
    # File birth times and strftime are only accurate to about one second
    # with FFmpeg. A different capture session is already excluded above.
    return (size == other_size and abs(fps - other_fps) < 0.05
            and abs((b - a).total_seconds() - count / fps) <= 1.1)


def _write_window(output, ranges, fps, size):
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"FFV1"), fps, size)
    try:
        if not writer.isOpened():
            raise IOError("Cannot open lossless RTSP analysis writer")
        for path, start, end in ranges:
            cap = open_video_capture(str(path))
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                for _ in range(start, end):
                    ok, frame = cap.read()
                    if not ok:
                        raise IOError(f"Incomplete RTSP segment: {path}")
                    writer.write(frame)
            finally:
                cap.release()
    finally:
        writer.release()


@contextmanager
def analysis_window(path, ready_paths=None, evidence_path="", context_seconds=5.0,
                    next_wait_seconds=15.0):
    """Yield detector options, retaining context but saving each event once.

    Wait for one completed successor. At a recording pause or disconnection,
    the last segment is processed after a bounded wait without future context.
    Legacy minute recordings keep their existing independent analysis.
    """
    if not is_segment(path):
        yield {}
        return
    current = Path(path)
    previous, following = _neighbours(current, ready_paths)
    if following is None and time.time() - current.stat().st_mtime < next_wait_seconds:
        raise SegmentNotReady(str(current))
    properties = {p: _properties(p) for p in (previous, current, following) if p is not None}
    if not _adjacent(previous, current, properties):
        previous = None
    if not _adjacent(current, following, properties):
        following = None
    fps, count, size = properties[current]
    context_frames = max(1, int(round(context_seconds * fps)))
    before = min(properties[previous][1], context_frames) if previous else 0
    ranges = []
    if previous:
        ranges.append((previous, properties[previous][1] - before, properties[previous][1]))
    ranges.append((current, 0, count))
    if following:
        ranges.append((following, 0, min(properties[following][1], context_frames)))
    started_at, _ = media_time.get_media_start_time(str(current))
    if started_at is None:
        raise IOError(f"Cannot determine capture time: {current}")
    with tempfile.TemporaryDirectory(prefix="meteor_rtsp_window_") as directory:
        joined = Path(directory) / "analysis.avi"
        _write_window(joined, ranges, fps, size)
        options = {"analysis_source": str(joined), "base_datetime": started_at - timedelta(seconds=before / fps),
                   "core_start_frame": before, "core_end_frame": before + count}
        if evidence_path:
            evidence_ranges = [(p.with_name(p.stem + "_innovation.mp4"), start, end)
                               for p, start, end in ranges]
            if not all(p.is_file() for p, _, _ in evidence_ranges):
                raise IOError("Missing neighbouring NoiseTwin innovation evidence")
            joined_evidence = Path(directory) / "innovation.avi"
            _write_window(joined_evidence, evidence_ranges, fps, size)
            options["evidence_path"] = str(joined_evidence)
        yield options


def owns_event(options, start_frame):
    context = options.get("rtsp_analysis_context", {})
    return (not context or context["core_start_frame"] <= start_frame < context["core_end_frame"])
