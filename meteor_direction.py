"""Estimate the time-ordered motion of a detected meteor line.

The line detector returns an unoriented geometric segment.  This module uses
the chronological event frames to recover the missing orientation by tracking
the positive temporal residual along that segment.  The returned endpoints are
therefore explicitly ordered as early observation -> late observation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]
Line = Tuple[Point, Point]


@dataclass(frozen=True)
class MotionEstimate:
    """Time-ordered motion information for one detected event."""

    status: str
    method: str
    start_pixel: Optional[Point]
    end_pixel: Optional[Point]
    start_frame: Optional[int]
    end_frame: Optional[int]
    vector_px: Optional[Point]
    speed_px_per_second: Optional[float]
    direction_angle_deg: Optional[float]
    displacement_fraction: Optional[float]
    fit_r2: Optional[float]
    active_frame_count: int
    note: str = ""

    @property
    def is_known(self) -> bool:
        return self.status in {"high", "medium"}

    def as_info_fields(self) -> Dict[str, Any]:
        """Return scalar values ready for the detector's info.txt writer."""
        fields: Dict[str, Any] = {
            "status": self.status,
            "method": self.method,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "speed_px_per_second": self.speed_px_per_second,
            "direction_angle_deg": self.direction_angle_deg,
            "displacement_fraction": self.displacement_fraction,
            "fit_r2": self.fit_r2,
            "active_frame_count": self.active_frame_count,
            "note": self.note,
        }
        if self.start_pixel is not None:
            fields["start_pixel"] = self.start_pixel
        if self.end_pixel is not None:
            fields["end_pixel"] = self.end_pixel
        if self.vector_px is not None:
            fields["vector_px"] = self.vector_px
        return fields


def _as_line(line: Sequence[Sequence[float]]) -> Line:
    if len(line) != 2:
        raise ValueError("検出線は2端点で指定してください")
    first = (float(line[0][0]), float(line[0][1]))
    second = (float(line[1][0]), float(line[1][1]))
    return first, second


def _grayscale(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame.astype(np.float32, copy=False)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32, copy=False)


def _sample_line_corridor(
    frames: Sequence[np.ndarray],
    line: Line,
    sample_count: int,
    corridor_half_width: float,
) -> np.ndarray:
    """Sample a narrow line corridor from each frame.

    The output shape is ``(frame, line_position, corridor_offset)``.  Using
    only this corridor avoids allocating a full-HD temporal cube for every
    candidate while retaining the spatial information needed for direction.
    """
    (x1, y1), (x2, y2) = line
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux
    t = np.linspace(0.0, 1.0, sample_count, dtype=np.float32)
    offsets = np.linspace(
        -float(corridor_half_width),
        float(corridor_half_width),
        max(5, int(round(corridor_half_width * 2.0)) + 1),
        dtype=np.float32,
    )
    xs = x1 + t[:, None] * dx + offsets[None, :] * nx
    ys = y1 + t[:, None] * dy + offsets[None, :] * ny

    values = []
    for frame in frames:
        gray = _grayscale(frame)
        height, width = gray.shape[:2]
        xi = np.clip(np.rint(xs).astype(np.int32), 0, width - 1)
        yi = np.clip(np.rint(ys).astype(np.int32), 0, height - 1)
        values.append(gray[yi, xi])
    return np.stack(values, axis=0).astype(np.float32)


def _weighted_linear_fit(frame_indices: np.ndarray, positions: np.ndarray, weights: np.ndarray):
    weights = np.asarray(weights, dtype=np.float64)
    x = np.asarray(frame_indices, dtype=np.float64)
    y = np.asarray(positions, dtype=np.float64)
    total = float(weights.sum())
    if total <= 1e-9:
        return None
    x_mean = float(np.sum(weights * x) / total)
    y_mean = float(np.sum(weights * y) / total)
    denominator = float(np.sum(weights * (x - x_mean) ** 2))
    if denominator <= 1e-9:
        return None
    slope = float(np.sum(weights * (x - x_mean) * (y - y_mean)) / denominator)
    intercept = y_mean - slope * x_mean
    predicted = intercept + slope * x
    ss_res = float(np.sum(weights * (y - predicted) ** 2))
    ss_tot = float(np.sum(weights * (y - y_mean) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0
    return slope, intercept, float(np.clip(r2, -1.0, 1.0))


def estimate_motion(
    frames: Sequence[np.ndarray],
    detected_line: Sequence[Sequence[float]],
    frame_start: int,
    frame_rate: float,
) -> MotionEstimate:
    """Estimate early-to-late motion along ``detected_line``.

    ``frames`` must be in chronological order and ``frame_start`` must be the
    absolute index of ``frames[0]``.  The result is conservative: short or
    weakly ordered signals are marked ``unknown`` instead of inventing a
    direction.
    """
    method = "temporal_residual_line_centroid_v1"
    if len(frames) < 3:
        return MotionEstimate(
            "unknown", method, None, None, None, None, None, None, None, None,
            None, 0, "方向推定に必要な時間フレームが不足しています",
        )
    try:
        line = _as_line(detected_line)
    except (TypeError, ValueError, IndexError) as exc:
        return MotionEstimate(
            "unknown", method, None, None, None, None, None, None, None, None,
            None, 0, f"検出線を解釈できません: {exc}",
        )

    (x1, y1), (x2, y2) = line
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < 8.0:
        return MotionEstimate(
            "unknown", method, None, None, None, None, None, None, None, None,
            None, 0, "検出線が短すぎます",
        )

    sample_count = max(96, min(512, int(round(length * 1.5))))
    corridor = max(3.0, min(12.0, length * 0.025))
    try:
        sampled = _sample_line_corridor(frames, line, sample_count, corridor)
    except (cv2.error, ValueError, IndexError) as exc:
        return MotionEstimate(
            "unknown", method, None, None, None, None, None, None, None, None,
            None, 0, f"時間信号を抽出できません: {exc}",
        )

    # A temporal median removes stars and fixed scene structure.  Only the
    # positive residual is used because a meteor is an additive light source.
    background = np.median(sampled, axis=0)
    deviation = np.abs(sampled - background[None, :, :])
    local_sigma = 1.4826 * np.median(deviation, axis=0)
    frame_delta_sigma = float(np.median(np.abs(np.diff(sampled, axis=0)))) / 0.954
    global_sigma = max(1.0, frame_delta_sigma)
    sigma = np.maximum(local_sigma, global_sigma)
    response = np.maximum((sampled - background[None, :, :]) / sigma[None, :, :] - 1.5, 0.0)

    t = np.linspace(0.0, 1.0, sample_count, dtype=np.float32)[None, :]
    weights = response.sum(axis=2)
    signal = weights.sum(axis=1)
    position = np.divide(
        (weights * t).sum(axis=1),
        signal,
        out=np.full(len(frames), np.nan, dtype=np.float32),
        where=signal > 1e-6,
    )

    positive_signal = signal[signal > 1e-6]
    if len(positive_signal) < 3:
        return MotionEstimate(
            "unknown", method, None, None, None, None, None, None, None, None,
            None, int(len(positive_signal)), "有効な時間信号が3フレーム未満です",
        )
    signal_floor = max(float(np.percentile(positive_signal, 25.0)), 1.0)
    valid = np.isfinite(position) & (signal >= signal_floor)
    if int(valid.sum()) < 3:
        return MotionEstimate(
            "unknown", method, None, None, None, None, None, None, None, None,
            None, int(valid.sum()), "時間信号が弱く、方向を確定できません",
        )

    local_frames = np.flatnonzero(valid).astype(np.float64)
    local_positions = position[valid].astype(np.float64)
    local_weights = np.maximum(signal[valid].astype(np.float64), 1.0)
    fit = _weighted_linear_fit(local_frames, local_positions, local_weights)
    if fit is None:
        return MotionEstimate(
            "unknown", method, None, None, None, None, None, None, None, None,
            None, int(valid.sum()), "時間位置の回帰に失敗しました",
        )
    slope, intercept, fit_r2 = fit
    first_local = int(local_frames.min())
    last_local = int(local_frames.max())
    fitted_start = float(np.clip(intercept + slope * first_local, 0.0, 1.0))
    fitted_end = float(np.clip(intercept + slope * last_local, 0.0, 1.0))
    displacement = fitted_end - fitted_start
    active_start_frame = int(frame_start + first_local)
    active_end_frame = int(frame_start + last_local)

    if (
        abs(displacement) < 0.06
        or (active_end_frame - active_start_frame) < 2
        or fit_r2 < 0.10
    ):
        return MotionEstimate(
            "unknown", method, None, None, active_start_frame, active_end_frame,
            None, None, None, abs(displacement), fit_r2, int(valid.sum()),
            "時間による位置変化または直線性が不十分で、方向を確定できません",
        )

    # ``fitted_start`` and ``fitted_end`` already correspond to the first and
    # last active frames.  Keep those temporal positions directly; this is
    # what makes the result independent of the Hough transform's arbitrary
    # endpoint order.
    start_t, end_t = fitted_start, fitted_end
    start = (x1 + start_t * dx, y1 + start_t * dy)
    end = (x1 + end_t * dx, y1 + end_t * dy)
    vector = (end[0] - start[0], end[1] - start[1])
    elapsed_seconds = max(1e-6, (active_end_frame - active_start_frame) / max(1.0, float(frame_rate)))
    speed = float(math.hypot(*vector) / elapsed_seconds)
    angle = math.degrees(math.atan2(vector[1], vector[0])) % 360.0
    confidence = "high" if abs(displacement) >= 0.18 and fit_r2 >= 0.35 else "medium"
    note = (
        "早い時刻の線上重心から遅い時刻の線上重心を回帰して決定。"
        "画像座標の角度は+X=右、+Y=下。"
    )
    return MotionEstimate(
        confidence,
        method,
        (float(start[0]), float(start[1])),
        (float(end[0]), float(end[1])),
        active_start_frame,
        active_end_frame,
        (float(vector[0]), float(vector[1])),
        speed,
        angle,
        abs(displacement),
        fit_r2,
        int(valid.sum()),
        note,
    )
