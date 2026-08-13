"""Detect short moving point sources without requiring a Hough line.

The primary detector works on a one-second frame window and is intentionally
independent from the Hough implementation.  A meteor can be visible as a
small bright blob in each frame while never forming a sufficiently strong
edge in the one-second max/min composite.  This module tracks those blobs in
time and returns a line-like candidate only after the movement itself is
coherent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]


@dataclass(frozen=True)
class MovingPointTrack:
    """A temporally coherent moving bright-point candidate."""

    start_frame: int
    end_frame: int
    start: Point
    end: Point
    center: Point
    displacement_px: float
    linearity: float
    speed_px_per_frame: float
    active_frames: int
    mean_score: float

    @property
    def line(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return (
            (int(round(self.start[0])), int(round(self.start[1]))),
            (int(round(self.end[0])), int(round(self.end[1]))),
        )


def _gray_small(frame: np.ndarray, scale: float) -> np.ndarray:
    if frame.ndim == 2:
        gray = frame
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if scale == 1.0:
        return gray
    height, width = gray.shape[:2]
    return cv2.resize(
        gray,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def _frame_detections(
    residual: np.ndarray,
    threshold: float,
    border: int,
    max_components: int,
) -> List[dict]:
    """Extract small bright components from one residual frame."""
    smoothed = cv2.GaussianBlur(residual, (3, 3), 0)
    binary = (smoothed >= float(threshold)).astype(np.uint8)
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    height, width = residual.shape[:2]
    detections = []
    for index in range(1, component_count):
        x, y, component_width, component_height, area = stats[index]
        if area < 2 or area > 120:
            continue
        if component_width > 30 or component_height > 30:
            continue
        if (
            x < border
            or y < border
            or x + component_width >= width - border
            or y + component_height >= height - border
        ):
            continue
        component_values = smoothed[labels == index]
        peak = float(component_values.max())
        total = float(component_values.sum())
        if peak < float(threshold) + 1.0:
            continue
        detections.append(
            {
                "x": float(centroids[index][0]),
                "y": float(centroids[index][1]),
                "score": peak + 0.05 * total,
                "area": int(area),
            }
        )
    detections.sort(key=lambda item: item["score"], reverse=True)
    return detections[:max_components]


def _finish_track(track: dict, scale: float) -> MovingPointTrack | None:
    points = np.asarray(track["points"], dtype=np.float32)
    times = np.asarray(track["times"], dtype=np.int32)
    if len(points) < 3 or int(times[-1] - times[0]) < 2:
        return None
    centered = points - points.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    linearity = float(
        singular_values[0] / max(1.0e-6, float(singular_values.sum()))
    )
    displacement_small = float(np.linalg.norm(points[-1] - points[0]))
    span = max(1, int(times[-1] - times[0]))
    displacement = displacement_small / max(scale, 1.0e-6)
    speed = displacement / span
    active_ratio = len(points) / max(1, span + 1)
    if displacement < 8.0 or linearity < 0.80 or active_ratio < 0.50:
        return None
    if speed < 0.5 or speed > 80.0:
        return None
    start = tuple(float(value / scale) for value in points[0])
    end = tuple(float(value / scale) for value in points[-1])
    center = tuple(float(value / scale) for value in points.mean(axis=0))
    return MovingPointTrack(
        start_frame=int(times[0]),
        end_frame=int(times[-1]),
        start=start,
        end=end,
        center=center,
        displacement_px=displacement,
        linearity=linearity,
        speed_px_per_frame=speed,
        active_frames=len(points),
        mean_score=float(np.mean(track["scores"])),
    )


def _deduplicate_tracks(tracks: Sequence[MovingPointTrack]) -> List[MovingPointTrack]:
    result: List[MovingPointTrack] = []
    for candidate in sorted(
        tracks,
        key=lambda item: (item.mean_score, item.displacement_px, item.active_frames),
        reverse=True,
    ):
        is_duplicate = False
        for existing in result:
            center_distance = math.hypot(
                candidate.center[0] - existing.center[0],
                candidate.center[1] - existing.center[1],
            )
            start_distance = math.hypot(
                candidate.start[0] - existing.start[0],
                candidate.start[1] - existing.start[1],
            )
            end_distance = math.hypot(
                candidate.end[0] - existing.end[0],
                candidate.end[1] - existing.end[1],
            )
            if center_distance < 80.0 and start_distance < 100.0 and end_distance < 100.0:
                is_duplicate = True
                break
        if not is_duplicate:
            result.append(candidate)
    return result


def detect_moving_point_tracks(
    frames: Sequence[np.ndarray],
    *,
    frame_rate: float = 25.0,
    scale: float = 0.5,
    threshold: float = 8.0,
    max_gap: int = 1,
    max_step: float = 14.0,
    border: int = 15,
    max_components_per_frame: int = 150,
    max_candidates: int = 20,
) -> List[MovingPointTrack]:
    """Return coherent moving bright-point tracks in ``frames``.

    The default scale makes the detector practical on 1920x1080 RTSP clips.
    Thresholds are in the downsampled grayscale residual domain, so they are
    deliberately modest; coherence over multiple frames supplies the main
    false-positive rejection instead of a high single-frame threshold.
    """
    del frame_rate  # reserved for future speed priors; speed is frame-based here
    if len(frames) < 3:
        return []
    if not 0.25 <= scale <= 1.0:
        raise ValueError("scale must be between 0.25 and 1.0")

    gray_frames = np.stack([_gray_small(frame, scale) for frame in frames]).astype(
        np.float32
    )
    background = np.median(gray_frames, axis=0)
    residuals = np.maximum(gray_frames - background[None, :, :], 0.0)
    small_border = max(3, int(round(border * scale)))
    detections = [
        _frame_detections(
            residual,
            threshold,
            small_border,
            max_components_per_frame,
        )
        for residual in residuals
    ]

    active: List[dict] = []
    finished: List[dict] = []
    for frame_index, frame_detections in enumerate(detections):
        used = set()
        next_active: List[dict] = []
        for track in sorted(active, key=lambda item: item["scores"][-1], reverse=True):
            last_frame = int(track["times"][-1])
            gap = frame_index - last_frame
            if gap > max_gap + 1:
                finished.append(track)
                continue
            previous_x, previous_y = track["points"][-1]
            best = None
            for detection_index, detection in enumerate(frame_detections):
                if detection_index in used:
                    continue
                dx = detection["x"] - previous_x
                dy = detection["y"] - previous_y
                distance = math.hypot(dx, dy)
                if distance > max_step * max(1, gap):
                    continue
                if len(track["points"]) >= 2:
                    old_x, old_y = track["points"][-2]
                    vx, vy = previous_x - old_x, previous_y - old_y
                    if dx * vx + dy * vy < -4.0 * max(1.0, distance):
                        continue
                cost = distance - min(20.0, detection["score"]) * 0.02
                if best is None or cost < best[0]:
                    best = (cost, detection_index, detection)
            if best is None:
                track["gaps"] += 1
                if track["gaps"] <= max_gap:
                    next_active.append(track)
                else:
                    finished.append(track)
                continue
            _, detection_index, detection = best
            used.add(detection_index)
            track["points"].append((detection["x"], detection["y"]))
            track["times"].append(frame_index)
            track["scores"].append(detection["score"])
            track["gaps"] = 0
            next_active.append(track)

        for detection_index, detection in enumerate(frame_detections):
            if detection_index in used:
                continue
            next_active.append(
                {
                    "points": [(detection["x"], detection["y"])],
                    "times": [frame_index],
                    "scores": [detection["score"]],
                    "gaps": 0,
                }
            )
        active = next_active
    finished.extend(active)

    tracks = []
    for track in finished:
        completed = _finish_track(track, scale)
        if completed is not None:
            tracks.append(completed)
    return _deduplicate_tracks(tracks)[:max_candidates]

