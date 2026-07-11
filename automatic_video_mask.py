"""Build conservative, cloud-robust masks for fixed all-sky cameras.

The camera is assumed to keep the same framing during an observing night.  A
mask is therefore estimated from frames spread over an hour and reused for all
clips in that hour.  Only pixel-aligned, persistent structure is excluded;
moving clouds remain part of the usable sky.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import threading
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


ALGORITHM_VERSION = "persistent-hourly-v2"
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
_HOUR_DIRECTORY = re.compile(r"^(?:[01]\d|2[0-3])$")
_DATE_DIRECTORY = re.compile(r"^(?:19|20)\d{6}$")
_DATE_TIME_NAME = re.compile(
    r"(?P<date>(?:19|20)\d{6})[^0-9]?(?P<hour>[01]\d|2[0-3])(?:[0-5]\d)?"
)
_PROCESS_LOCKS: Dict[str, threading.Lock] = {}
_PROCESS_LOCKS_GUARD = threading.Lock()


def _open_video(video_path: str) -> cv2.VideoCapture:
    """Open one decoder with a small thread budget.

    FFmpeg otherwise creates a large decoder pool for every concurrently
    processed clip.  Some OpenCV builds do not expose the open-only property,
    so retain a normal VideoCapture fallback.
    """
    thread_property = getattr(cv2, "CAP_PROP_N_THREADS", None)
    ffmpeg_backend = getattr(cv2, "CAP_FFMPEG", None)
    if thread_property is not None and ffmpeg_backend is not None:
        try:
            cap = cv2.VideoCapture(
                video_path,
                ffmpeg_backend,
                [int(thread_property), 2],
            )
            if cap.isOpened():
                return cap
            cap.release()
        except (cv2.error, TypeError):
            pass
    return cv2.VideoCapture(video_path)


def _evenly_spaced(items: Sequence[Path], count: int) -> List[Path]:
    if len(items) <= count:
        return list(items)
    indices = np.linspace(0, len(items) - 1, count, dtype=int)
    return [items[int(index)] for index in np.unique(indices)]


def _hour_scope(video_path: str) -> Tuple[str, str]:
    """Return a stable human-readable label and cache identity for an hour."""
    path = Path(video_path).expanduser().resolve()
    parent = path.parent
    if _HOUR_DIRECTORY.fullmatch(parent.name) and _DATE_DIRECTORY.fullmatch(parent.parent.name):
        label = f"{parent.parent.name}/{parent.name}"
        return label, f"directory:{parent}"

    match = _DATE_TIME_NAME.search(path.stem)
    if match:
        bucket = f"{match.group('date')}/{match.group('hour')}"
        return bucket, f"filename:{parent}|{bucket}"

    # A generic directory has no timestamp convention.  File modification time
    # is the safest available one-hour bucket and still allows adjacent clips to
    # share a mask.
    timestamp = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y%m%d/%H")
    return timestamp, f"mtime:{parent}|{timestamp}"


def _cache_key(video_path: str) -> str:
    _, identity = _hour_scope(video_path)
    raw = f"{ALGORITHM_VERSION}|{identity}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20]


def representative_hour_videos(video_path: str, count: int = 7) -> List[str]:
    """Choose clips spread across the same hour as ``video_path``."""
    target = Path(video_path).expanduser().resolve()
    _, target_identity = _hour_scope(str(target))
    candidates: List[Path] = []
    try:
        for item in target.parent.iterdir():
            if not item.is_file() or item.suffix.lower() not in VIDEO_SUFFIXES:
                continue
            try:
                if _hour_scope(str(item))[1] == target_identity:
                    candidates.append(item.resolve())
            except OSError:
                continue
    except OSError:
        candidates = []
    if target not in candidates:
        candidates.append(target)
    candidates.sort(key=lambda item: item.name)
    return [str(item) for item in _evenly_spaced(candidates, max(1, count))]


def _sample_one_video(
    video_path: str,
    sample_count: int,
    max_width: int,
) -> Tuple[List[np.ndarray], Optional[Tuple[int, int]]]:
    cap = _open_video(video_path)
    if not cap.isOpened():
        return [], None
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            positions = list(range(sample_count))
        else:
            positions = np.linspace(0, max(0, total - 1), sample_count + 2, dtype=int)[1:-1]
        frames: List[np.ndarray] = []
        original_size: Optional[Tuple[int, int]] = None
        for position in positions:
            if total > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(position))
            ok, frame = cap.read()
            if not ok:
                continue
            original_size = (frame.shape[1], frame.shape[0])
            gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if gray.shape[1] > max_width:
                scale = max_width / gray.shape[1]
                gray = cv2.resize(
                    gray,
                    (max_width, max(1, round(gray.shape[0] * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            frames.append(gray)
        return frames, original_size
    finally:
        cap.release()


def sample_video_median(video_path: str, sample_count: int = 7, max_width: int = 640):
    """Compatibility helper returning the median of one clip."""
    frames, original_size = _sample_one_video(video_path, sample_count, max_width)
    if not frames or original_size is None:
        raise IOError(f"マスク用フレームを読み込めません: {video_path}")
    shape = frames[0].shape
    frames = [frame for frame in frames if frame.shape == shape]
    return np.median(np.stack(frames), axis=0).astype(np.uint8), original_size


def sample_hour_frames(
    video_path: str,
    videos_per_hour: int = 7,
    samples_per_video: int = 3,
    max_width: int = 640,
) -> Tuple[np.ndarray, Tuple[int, int], List[str]]:
    """Read a compact set of frames distributed across one camera-hour."""
    representatives = representative_hour_videos(video_path, videos_per_hour)
    all_frames: List[np.ndarray] = []
    original_size: Optional[Tuple[int, int]] = None
    sample_paths: List[str] = []
    reference_shape: Optional[Tuple[int, int]] = None
    for path in representatives:
        frames, size = _sample_one_video(path, samples_per_video, max_width)
        if not frames or size is None:
            continue
        if original_size is None:
            original_size = size
            reference_shape = frames[0].shape
        if size != original_size or reference_shape is None:
            continue
        accepted = [frame for frame in frames if frame.shape == reference_shape]
        if accepted:
            all_frames.extend(accepted)
            sample_paths.append(path)
    if not all_frames or original_size is None:
        raise IOError(f"マスク用フレームを読み込めません: {video_path}")
    return np.stack(all_frames), original_size, sample_paths


def build_mask_from_samples(samples: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """Return 255 for sky and 0 for high-confidence static obstructions.

    Moving clouds can have strong edges in an individual image.  A physical
    obstruction, lens edge, wire, tree, or text overlay instead produces an
    edge at the *same pixel* throughout the hour.  The low temporal percentile
    of gradient magnitude isolates that persistence without assuming that an
    obstruction is connected to the bottom of the frame.
    """
    stack = np.asarray(samples)
    if stack.ndim == 2:
        stack = stack[np.newaxis, ...]
    if stack.ndim != 3 or stack.shape[0] == 0:
        raise ValueError("samples must have shape (frames, height, width)")
    stack = np.clip(stack, 0, 255).astype(np.uint8)
    height, width = stack.shape[1:]

    gradients = []
    for gray in stack:
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        gradients.append(cv2.magnitude(gx, gy))
    persistence_percentile = 5.0 if stack.shape[0] >= 8 else 0.0
    persistent_gradient = np.percentile(
        np.stack(gradients), persistence_percentile, axis=0
    )

    # Absolute contrast prevents a cloudy image with no obstruction from
    # manufacturing a mask merely because every image has a percentile tail.
    # The capped adaptive part accommodates both low- and high-gain cameras.
    gradient_threshold = max(
        10.0,
        min(16.0, float(np.percentile(persistent_gradient, 98.8))),
    )
    core = (persistent_gradient >= gradient_threshold).astype(np.uint8) * 255
    core = cv2.morphologyEx(core, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    # Thicken wires/branches and close outlines.  Enclosed, reasonably sized
    # contours are filled so solid obstacles are masked rather than only their
    # boundaries.  Large diffuse regions are deliberately never filled.
    dilation_size = max(3, min(7, int(round(min(height, width) * 0.012)) | 1))
    dilation_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation_size, dilation_size)
    )
    obstacle = cv2.dilate(core, dilation_kernel, iterations=1)
    close_size = max(5, min(11, dilation_size + 4))
    closed = cv2.morphologyEx(
        core,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)),
    )
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(core)
    frame_area = height * width
    minimum_area = max(12.0, frame_area * 0.00003)
    filled_components = 0
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        _, _, box_width, box_height = cv2.boundingRect(contour)
        box_area = box_width * box_height
        if (
            minimum_area <= contour_area <= frame_area * 0.20
            and box_area <= frame_area * 0.25
        ):
            cv2.drawContours(filled, [contour], -1, 255, thickness=-1)
            filled_components += 1
    obstacle = cv2.bitwise_or(obstacle, filled)
    obstacle = cv2.morphologyEx(
        obstacle, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )

    # Remove isolated decoder/noise specks, while preserving thin connected
    # wires and text strokes after dilation.
    component_count, labels, component_stats, _ = cv2.connectedComponentsWithStats(
        (obstacle > 0).astype(np.uint8), connectivity=8
    )
    cleaned = np.zeros_like(obstacle)
    minimum_component = max(6, int(frame_area * 0.000015))
    retained_components = 0
    for label in range(1, component_count):
        if component_stats[label, cv2.CC_STAT_AREA] >= minimum_component:
            cleaned[labels == label] = 255
            retained_components += 1

    mask = cv2.bitwise_not(cleaned)
    obstacle_fraction = float(np.count_nonzero(cleaned) / cleaned.size)
    stats = {
        "sky_fraction": 1.0 - obstacle_fraction,
        "obstacle_fraction": obstacle_fraction,
        "sample_count": float(stack.shape[0]),
        "persistent_gradient_percentile": persistence_percentile,
        "persistent_gradient_threshold": float(gradient_threshold),
        "retained_components": float(retained_components),
        "filled_components": float(filled_components),
    }
    return mask, stats


def build_mask_from_median(median_gray: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """Single-image compatibility fallback.

    New masks use :func:`build_mask_from_samples`; callers with only one image
    still get a conservative all-frame static-feature mask.
    """
    return build_mask_from_samples(np.asarray(median_gray)[np.newaxis, ...])


@contextmanager
def _generation_lock(cache: Path, key: str) -> Iterable[None]:
    """Serialize one hourly generation across threads and worker processes."""
    with _PROCESS_LOCKS_GUARD:
        process_lock = _PROCESS_LOCKS.setdefault(key, threading.Lock())
    with process_lock:
        lock_path = cache / f"{key}.lock"
        with lock_path.open("a+b") as handle:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            except (ImportError, OSError):
                fcntl = None  # type: ignore[assignment]
            try:
                yield
            finally:
                if fcntl is not None:
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    except OSError:
                        pass


def _read_cached_mask(
    mask_path: Path, preview_path: Path, metadata_path: Path
) -> Optional[Tuple[np.ndarray, str, Dict[str, float]]]:
    if not mask_path.exists() or not metadata_path.exists():
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if metadata.get("algorithm_version") != ALGORITHM_VERSION:
        return None
    return mask, str(preview_path), metadata.get("stats", {})


def _atomic_imwrite(path: Path, image: np.ndarray, params: Optional[List[int]] = None) -> None:
    temporary = path.with_name(f"{path.stem}.tmp-{os.getpid()}-{threading.get_ident()}{path.suffix}")
    try:
        if not cv2.imwrite(str(temporary), image, params or []):
            raise IOError(f"画像を書き込めません: {path}")
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_json_write(path: Path, payload: Dict[str, object]) -> None:
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def create_auto_mask(video_path: str, cache_dir: str) -> Tuple[np.ndarray, str, Dict[str, float]]:
    """Create/load one cloud-robust mask shared by a camera-hour."""
    cache = Path(cache_dir).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    key = _cache_key(video_path)
    mask_path = cache / f"{key}_mask.png"
    preview_path = cache / f"{key}_preview.jpg"
    metadata_path = cache / f"{key}.json"
    cached = _read_cached_mask(mask_path, preview_path, metadata_path)
    if cached is not None:
        return cached

    with _generation_lock(cache, key):
        # Another worker may have completed the hour while this worker waited.
        cached = _read_cached_mask(mask_path, preview_path, metadata_path)
        if cached is not None:
            return cached

        samples, (width, height), sample_paths = sample_hour_frames(video_path)
        mask_small, stats = build_mask_from_samples(samples)
        mask = cv2.resize(mask_small, (width, height), interpolation=cv2.INTER_NEAREST)
        median_small = np.median(samples, axis=0).astype(np.uint8)
        median_full = cv2.resize(
            median_small, (width, height), interpolation=cv2.INTER_LINEAR
        )
        overlay = cv2.cvtColor(median_full, cv2.COLOR_GRAY2BGR)
        excluded = mask == 0
        overlay[excluded] = (
            0.35 * overlay[excluded] + 0.65 * np.array([0, 0, 255])
        ).astype(np.uint8)
        boundary = cv2.morphologyEx(
            mask, cv2.MORPH_GRADIENT, np.ones((7, 7), np.uint8)
        )
        overlay[boundary > 0] = (0, 255, 255)

        _atomic_imwrite(mask_path, mask)
        _atomic_imwrite(preview_path, overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
        scope_label, scope_identity = _hour_scope(video_path)
        metadata: Dict[str, object] = {
            "algorithm_version": ALGORITHM_VERSION,
            "generated_at": datetime.now().astimezone().isoformat(),
            "hour_scope": scope_label,
            "scope_identity": scope_identity,
            "requested_video_path": str(Path(video_path).resolve()),
            "sample_video_paths": sample_paths,
            "mask_path": str(mask_path),
            "preview_path": str(preview_path),
            "stats": stats,
        }
        _atomic_json_write(metadata_path, metadata)
        return mask, str(preview_path), stats


def combine_masks(manual_mask: Optional[np.ndarray], automatic_mask: np.ndarray) -> np.ndarray:
    if manual_mask is None:
        return automatic_mask
    resized = manual_mask
    if resized.shape[:2] != automatic_mask.shape[:2]:
        resized = cv2.resize(
            resized,
            (automatic_mask.shape[1], automatic_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    if resized.ndim == 3:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return cv2.bitwise_and(resized.astype(np.uint8), automatic_mask)
