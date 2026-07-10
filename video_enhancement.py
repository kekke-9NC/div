"""Shared fixed-pattern + temporal-mean enhancement pipeline.

This is used by detected-item saving and by the analysis-tab video
concatenator so RTSP, folders and individual video files behave identically.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence, Tuple

import cv2
import numpy as np

from fixed_pattern import apply_fixed_pattern_correction
import video_denoising


Line = Tuple[Tuple[int, int], Tuple[int, int]]


@dataclass(frozen=True)
class EnhancementResult:
    frames: list[np.ndarray]
    correction_strength: float
    noise_before: float
    noise_after: float

    @property
    def noise_reduction_percent(self) -> float:
        return video_denoising.noise_reduction_percent(self.noise_before, self.noise_after)


def _resize_correction(correction: np.ndarray, frame: np.ndarray) -> np.ndarray:
    if correction.shape[:2] == frame.shape[:2]:
        return correction
    return cv2.resize(
        correction,
        (frame.shape[1], frame.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )


def _normalized_temporal_pattern(frames: Sequence[np.ndarray], max_samples: int = 31) -> np.ndarray:
    indices = np.linspace(0, len(frames) - 1, min(max_samples, len(frames)), dtype=int)
    gray_frames = [cv2.cvtColor(frames[index], cv2.COLOR_BGR2GRAY) for index in indices]
    height, width = gray_frames[0].shape
    levels = []
    for gray in gray_frames:
        crop = gray[int(height * 0.12):int(height * 0.78), int(width * 0.12):int(width * 0.88)]
        valid = crop[crop > 8]
        levels.append(float(np.median(valid)) if valid.size else float(np.median(crop)))
    target = float(np.median(levels))
    stack = np.stack(
        [
            cv2.convertScaleAbs(gray, alpha=target / max(level, 1.0))
            for gray, level in zip(gray_frames, levels)
        ],
        axis=0,
    )
    middle = stack.shape[0] // 2
    return np.partition(stack, middle, axis=0)[middle].astype(np.float32)


def estimate_correction_strength(
    frames: Sequence[np.ndarray],
    correction: Optional[np.ndarray],
    minimum: float = 0.6,
    maximum: float = 1.0,
) -> float:
    """Estimate only the amplitude of a known fixed-pattern shape."""
    if not frames or correction is None:
        return 0.0
    correction = _resize_correction(correction, frames[0]).astype(np.float32)
    if correction.ndim == 3:
        correction = cv2.cvtColor(correction, cv2.COLOR_BGR2GRAY)
    pattern = _normalized_temporal_pattern(frames)
    pattern_hp = pattern - cv2.GaussianBlur(pattern, (0, 0), 18)
    correction_hp = correction - cv2.GaussianBlur(correction, (0, 0), 18)
    height, width = pattern.shape
    region = np.s_[int(height * 0.12):int(height * 0.78), int(width * 0.12):int(width * 0.88)]
    observed = pattern_hp[region].ravel()
    reference = correction_hp[region].ravel()
    observed = observed - np.mean(observed)
    reference = reference - np.mean(reference)
    power = float(np.dot(reference, reference))
    if power <= 1e-6:
        return float(minimum)
    strength = float(np.dot(observed, reference) / power)
    return float(np.clip(strength, minimum, maximum))


def enhance_frames(
    frames: list[np.ndarray],
    correction: np.ndarray,
    detected_line: Optional[Line] = None,
    temporal_radius: int = 10,
    original_blend: float = 0.12,
    transient_threshold: float = 10.0,
    protect_line_width: int = 32,
) -> EnhancementResult:
    """Apply adaptive fixed correction followed by a 21-frame temporal mean."""
    if not frames:
        return EnhancementResult([], 0.0, 0.0, 0.0)
    correction = _resize_correction(correction, frames[0])
    strength = estimate_correction_strength(frames, correction)
    before_meter = video_denoising.TemporalNoiseMeter()
    for index, frame in enumerate(frames):
        before_meter.update(frame)
        frames[index] = apply_fixed_pattern_correction(frame, correction, strength=strength)

    enhanced: list[np.ndarray] = []
    after_meter = video_denoising.TemporalNoiseMeter()
    for frame in video_denoising.iter_denoised_frames(
        frames,
        detected_line=detected_line,
        temporal_radius=temporal_radius,
        original_blend=original_blend,
        transient_threshold=transient_threshold,
        protect_line_width=protect_line_width,
        temporal_method="mean",
        protect_global_transients=detected_line is None,
    ):
        after_meter.update(frame)
        enhanced.append(frame)
    return EnhancementResult(enhanced, strength, before_meter.sigma, after_meter.sigma)


def iter_enhanced_video_frames(
    video_path: str,
    correction: np.ndarray,
    temporal_radius: int = 10,
    transient_threshold: float = 12.0,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> tuple[Iterator[np.ndarray], float, float, int, Tuple[int, int]]:
    """Return a streaming 21-frame-mean iterator for long concatenation inputs."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"動画を開けません: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    samples = []
    sample_count = min(31, max(1, total))
    for index in np.linspace(0, max(0, total - 1), sample_count, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if ok:
            samples.append(frame)
    if not samples:
        cap.release()
        raise IOError(f"補正強度推定用フレームを取得できません: {video_path}")
    resized_correction = _resize_correction(correction, samples[0])
    strength = estimate_correction_strength(samples, resized_correction)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def generator() -> Iterator[np.ndarray]:
        try:
            initial = []
            for _ in range(temporal_radius + 1):
                ok, frame = cap.read()
                if not ok:
                    break
                initial.append(apply_fixed_pattern_correction(frame, resized_correction, strength))
            if not initial:
                return
            buffer = [initial[0]] * temporal_radius + initial
            last_real = initial[-1]
            emitted = 0
            while emitted < max(total, len(initial)):
                current = buffer[temporal_radius]
                stack = np.stack(buffer, axis=0)
                reference = np.rint(np.sum(stack, axis=0, dtype=np.uint32) / len(buffer)).astype(np.uint8)
                denoised = cv2.addWeighted(current, 0.12, reference, 0.88, 0.0)
                restore = video_denoising.transient_restore_mask(
                    current,
                    reference,
                    detected_line=None,
                    threshold=transient_threshold,
                    line_width=32,
                    protect_global_transients=True,
                )
                if restore is not None and np.any(restore > 0):
                    alpha = restore[..., None]
                    denoised = np.clip(
                        denoised.astype(np.float32) * (1.0 - alpha)
                        + current.astype(np.float32) * alpha,
                        0,
                        255,
                    ).astype(np.uint8)
                yield denoised
                emitted += 1
                if progress_callback:
                    progress_callback(emitted, total)
                ok, next_frame = cap.read()
                if ok:
                    last_real = apply_fixed_pattern_correction(next_frame, resized_correction, strength)
                buffer.pop(0)
                buffer.append(last_real)
        finally:
            cap.release()

    return generator(), strength, fps, total, (width, height)


def load_fixed_correction(path: str) -> np.ndarray:
    with np.load(Path(path), allow_pickle=False) as data:
        if "fixed_correction" in data:
            return data["fixed_correction"].astype(np.int16)
        if "dark_frame" in data:
            return data["dark_frame"].astype(np.uint8)
    raise ValueError(f"固定パターン補正データがありません: {path}")


def enhance_video_file(
    input_path: str,
    output_path: str,
    correction: np.ndarray,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> float:
    """Stream-enhance a long video and return the adaptive correction strength."""
    frames, strength, fps, total, (width, height) = iter_enhanced_video_frames(
        input_path,
        correction,
        temporal_radius=10,
        progress_callback=progress_callback,
    )
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"avc1"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise IOError(f"補正動画の書き込みを開始できません: {output_path}")
    written = 0
    try:
        for frame in frames:
            writer.write(frame)
            written += 1
    finally:
        writer.release()
    if written == 0:
        raise IOError(f"補正動画へフレームを書き込めませんでした: {input_path}")
    return strength
