"""Observation-safe temporal denoising for detected meteor clips.

The original clip is always kept.  This module creates a view-oriented copy
using a short temporal median, then restores transient bright pixels along the
detected trajectory so a meteor is not averaged away with the sensor noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional, Sequence, Tuple

import cv2
import numpy as np


Line = Tuple[Tuple[int, int], Tuple[int, int]]


def _clamped_window(frames: Sequence[np.ndarray], index: int, radius: int) -> np.ndarray:
    """Return an odd-sized temporal window, repeating edge frames as needed."""
    indices = [min(len(frames) - 1, max(0, index + offset)) for offset in range(-radius, radius + 1)]
    return np.stack([frames[item] for item in indices], axis=0)


def _temporal_median(frames: Sequence[np.ndarray], index: int, radius: int) -> np.ndarray:
    window = _clamped_window(frames, index, radius)
    middle = window.shape[0] // 2
    # np.partition keeps uint8 data and uses much less memory than np.median,
    # which promotes a 1080p stack to float64.
    return np.partition(window, middle, axis=0)[middle]


def _temporal_mean(frames: Sequence[np.ndarray], index: int, radius: int) -> np.ndarray:
    """Return an 8-bit temporal mean without promoting the full stack to float64."""
    window = _clamped_window(frames, index, radius)
    total = np.sum(window, axis=0, dtype=np.uint32)
    return np.rint(total / window.shape[0]).astype(np.uint8)


def _temporal_reference(
    frames: Sequence[np.ndarray],
    index: int,
    radius: int,
    method: str,
) -> np.ndarray:
    normalized = str(method).strip().lower()
    if normalized == "mean":
        return _temporal_mean(frames, index, radius)
    if normalized != "median":
        raise ValueError(f"Unsupported temporal denoise method: {method}")
    return _temporal_median(frames, index, radius)


def transient_restore_mask(
    current: np.ndarray,
    temporal_median: np.ndarray,
    detected_line: Optional[Line],
    threshold: float,
    line_width: int,
    protect_global_transients: bool = False,
) -> Optional[np.ndarray]:
    """Build a soft mask for bright temporal transients near the meteor line."""
    if not detected_line and not protect_global_transients:
        return None

    height, width = current.shape[:2]
    if detected_line:
        (x1, y1), (x2, y2) = detected_line
        x1, x2 = int(np.clip(x1, 0, width - 1)), int(np.clip(x2, 0, width - 1))
        y1, y2 = int(np.clip(y1, 0, height - 1)), int(np.clip(y2, 0, height - 1))
        corridor = np.zeros((height, width), dtype=np.uint8)
        cv2.line(corridor, (x1, y1), (x2, y2), 255, max(3, int(line_width)), cv2.LINE_AA)
    else:
        corridor = np.full((height, width), 255, dtype=np.uint8)

    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY).astype(np.float32)
    median_gray = cv2.cvtColor(temporal_median, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bright_residual = current_gray - median_gray

    # Soft transition: small random fluctuations stay denoised, while a clear
    # positive transient is increasingly restored from the current raw frame.
    soft_range = max(4.0, threshold)
    restore = np.clip((bright_residual - threshold) / soft_range, 0.0, 1.0)
    restore *= corridor.astype(np.float32) / 255.0
    restore = cv2.GaussianBlur(restore, (0, 0), sigmaX=1.0, sigmaY=1.0)
    return np.clip(restore, 0.0, 1.0)


def denoise_frame(
    frames: Sequence[np.ndarray],
    index: int,
    detected_line: Optional[Line] = None,
    temporal_radius: int = 2,
    original_blend: float = 0.12,
    transient_threshold: float = 10.0,
    protect_line_width: int = 32,
    temporal_method: str = "median",
    protect_global_transients: bool = False,
) -> np.ndarray:
    """Denoise one frame while protecting a detected meteor trajectory."""
    if not frames:
        raise ValueError("frames must not be empty")
    radius = max(1, int(temporal_radius))
    original_weight = float(np.clip(original_blend, 0.0, 1.0))
    current = frames[index]
    reference = _temporal_reference(frames, index, radius, temporal_method)
    denoised = cv2.addWeighted(current, original_weight, reference, 1.0 - original_weight, 0.0)

    restore_mask = transient_restore_mask(
        current,
        reference,
        detected_line,
        float(transient_threshold),
        int(protect_line_width),
        protect_global_transients=protect_global_transients,
    )
    if restore_mask is not None and np.any(restore_mask > 0):
        alpha = restore_mask[..., None]
        denoised = np.clip(
            denoised.astype(np.float32) * (1.0 - alpha) + current.astype(np.float32) * alpha,
            0,
            255,
        ).astype(np.uint8)
    return denoised


def iter_denoised_frames(
    frames: Sequence[np.ndarray],
    detected_line: Optional[Line] = None,
    temporal_radius: int = 2,
    original_blend: float = 0.12,
    transient_threshold: float = 10.0,
    protect_line_width: int = 32,
    temporal_method: str = "median",
    protect_global_transients: bool = False,
) -> Iterator[np.ndarray]:
    for index in range(len(frames)):
        yield denoise_frame(
            frames,
            index,
            detected_line=detected_line,
            temporal_radius=temporal_radius,
            original_blend=original_blend,
            transient_threshold=transient_threshold,
            protect_line_width=protect_line_width,
            temporal_method=temporal_method,
            protect_global_transients=protect_global_transients,
        )


@dataclass
class TemporalNoiseMeter:
    """Robust temporal high-frequency noise estimate in 8-bit intensity units."""

    sample_step: int = 4
    _previous: Optional[np.ndarray] = None
    _estimates: list[float] = field(default_factory=list)

    def update(self, frame: np.ndarray) -> None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if self._previous is not None:
            temporal_difference = gray - self._previous
            slow_change = cv2.GaussianBlur(temporal_difference, (0, 0), sigmaX=1.4, sigmaY=1.4)
            high_frequency = (temporal_difference - slow_change)[:: self.sample_step, :: self.sample_step]
            center = float(np.median(high_frequency))
            mad = float(np.median(np.abs(high_frequency - center)))
            # Divide by sqrt(2): a frame difference contains noise from two frames.
            self._estimates.append(mad / 0.67448975 / np.sqrt(2.0))
        self._previous = gray

    @property
    def sigma(self) -> float:
        return float(np.median(self._estimates)) if self._estimates else 0.0


def estimate_temporal_noise(frames: Iterable[np.ndarray]) -> float:
    meter = TemporalNoiseMeter()
    for frame in frames:
        meter.update(frame)
    return meter.sigma


def noise_reduction_percent(before_sigma: float, after_sigma: float) -> float:
    if before_sigma <= 0:
        return 0.0
    return max(0.0, (1.0 - after_sigma / before_sigma) * 100.0)
