"""Camera-invariant event representation and lightweight meteor classifier.

The model deliberately never receives a camera name, date, source path, or a
prediction made by an older model.  It operates on robust temporal residuals,
a track-aligned space-time representation, and dimensionless event features.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn


ARCHITECTURE_NAME = "meteor_fusion_universal_v1"
PREPROCESS_VERSION = 1
IMAGE_SIZE = 160
KYMO_HEIGHT = 96
KYMO_WIDTH = 128
FEATURE_COUNT = 12


def _gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame.astype(np.uint8, copy=False)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _crop_square(
    frame: np.ndarray,
    rect: Optional[Sequence[int]],
    size: int = 256,
) -> np.ndarray:
    gray = _gray(frame)
    if rect is None:
        crop = gray
    else:
        x1, y1, x2, y2 = (int(v) for v in rect)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(gray.shape[1], x2), min(gray.shape[0], y2)
        crop = gray[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((size, size), dtype=np.uint8)
    if crop.shape != (size, size):
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return crop


def _line_in_crop(
    detected_line: Optional[Sequence[Sequence[float]]],
    rect: Optional[Sequence[int]],
    source_size: int,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    if not detected_line or len(detected_line) != 2:
        return (source_size * 0.3, source_size * 0.5), (
            source_size * 0.7,
            source_size * 0.5,
        )
    p1 = detected_line[0]
    p2 = detected_line[1]
    offset_x = float(rect[0]) if rect is not None else 0.0
    offset_y = float(rect[1]) if rect is not None else 0.0
    rect_w = max(1.0, float(rect[2] - rect[0])) if rect is not None else source_size
    rect_h = max(1.0, float(rect[3] - rect[1])) if rect is not None else source_size
    sx = source_size / rect_w
    sy = source_size / rect_h
    return (
        ((float(p1[0]) - offset_x) * sx, (float(p1[1]) - offset_y) * sy),
        ((float(p2[0]) - offset_x) * sx, (float(p2[1]) - offset_y) * sy),
    )


def _robust_residual_cube(stack: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return positive temporal residuals in local noise-sigma units."""
    values = stack.astype(np.float32)
    background = np.median(values, axis=0)
    abs_deviation = np.abs(values - background)
    temporal_sigma = 1.4826 * np.median(abs_deviation, axis=0)

    if len(values) > 1:
        frame_delta = np.diff(values, axis=0)
        global_sigma = float(np.median(np.abs(frame_delta))) / 0.954
    else:
        global_sigma = float(np.median(abs_deviation)) * 1.4826
    global_sigma = max(1.0, global_sigma)
    sigma = np.maximum(temporal_sigma, global_sigma)
    positive = np.maximum(values - background, 0.0)
    residual_z = np.clip(positive / sigma[None, :, :], 0.0, 30.0)
    return residual_z.astype(np.float32), global_sigma


def _compress_response(residual_z: np.ndarray) -> np.ndarray:
    scale = math.asinh(6.0)
    return np.clip(np.arcsinh(residual_z / 2.0) / scale, 0.0, 1.0)


def _track_aligned_cube(
    response: np.ndarray,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
) -> np.ndarray:
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    line_length = max(8.0, math.hypot(dx, dy))
    center = ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)
    angle = math.degrees(math.atan2(dy, dx))
    scale = float(np.clip(84.0 / line_length, 0.45, 2.5))
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[0, 2] += KYMO_WIDTH * 0.5 - center[0]
    matrix[1, 2] += 64.0 * 0.5 - center[1]
    aligned = [
        cv2.warpAffine(
            frame,
            matrix,
            (KYMO_WIDTH, 64),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        for frame in response
    ]
    return np.stack(aligned).astype(np.float32)


def _centroid_track(response: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, weights = [], [], []
    height, width = response.shape[1:]
    xx = np.arange(width, dtype=np.float32)[None, :]
    yy = np.arange(height, dtype=np.float32)[:, None]
    for frame in response:
        signal = np.maximum(frame - 0.08, 0.0)
        weight = float(signal.sum())
        weights.append(weight)
        if weight > 1e-5:
            xs.append(float((signal * xx).sum() / weight))
            ys.append(float((signal * yy).sum() / weight))
        else:
            xs.append(float("nan"))
            ys.append(float("nan"))
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(weights, dtype=np.float32),
    )


def _physical_features(
    response: np.ndarray,
    residual_z: np.ndarray,
    line_length: float,
    fps: float,
) -> np.ndarray:
    frame_count, height, width = response.shape
    duration = frame_count / max(1.0, float(fps))
    xs, ys, weights = _centroid_track(response)
    valid = np.isfinite(xs) & (weights > np.percentile(weights, 40))

    travel = 0.0
    linearity = 0.0
    speed_cv = 0.0
    if valid.sum() >= 3:
        vx, vy = xs[valid], ys[valid]
        travel = math.hypot(float(vx[-1] - vx[0]), float(vy[-1] - vy[0]))
        points = np.column_stack([vx, vy])
        centered = points - points.mean(axis=0, keepdims=True)
        singular = np.linalg.svd(centered, compute_uv=False)
        linearity = float(singular[0] / max(1e-6, singular.sum()))
        steps = np.hypot(np.diff(vx), np.diff(vy))
        if steps.size and float(steps.mean()) > 1e-5:
            speed_cv = float(steps.std() / steps.mean())

    active_by_frame = (residual_z > 3.0).mean(axis=(1, 2))
    active_time_fraction = float((active_by_frame > 0.0005).mean())
    active_area = float((residual_z.max(axis=0) > 3.0).mean())
    peak_weights = weights / max(1e-6, float(weights.sum()))
    temporal_peakiness = float(peak_weights.max()) if peak_weights.size else 0.0

    periodicity = 0.0
    if len(weights) >= 8 and float(weights.std()) > 1e-5:
        normalized = (weights - weights.mean()) / weights.std()
        autocorr = np.correlate(normalized, normalized, mode="full")[len(weights) - 1 :]
        autocorr /= max(1.0, float(autocorr[0]))
        upper = min(len(autocorr), max(3, int(float(fps) * 1.5)))
        if upper > 2:
            periodicity = float(np.max(autocorr[2:upper]))

    features = np.asarray(
        [
            np.clip(duration / 6.0, 0.0, 2.0),
            np.clip(line_length / 256.0, 0.0, 2.0),
            np.clip(float(residual_z.max()) / 20.0, 0.0, 2.0),
            np.clip(float(np.percentile(residual_z, 99.5)) / 10.0, 0.0, 2.0),
            np.clip(active_area * 20.0, 0.0, 2.0),
            active_time_fraction,
            np.clip(temporal_peakiness * 5.0, 0.0, 2.0),
            np.clip(travel / max(8.0, line_length), 0.0, 2.0),
            linearity,
            np.clip(speed_cv / 2.0, 0.0, 2.0),
            np.clip((periodicity + 1.0) * 0.5, 0.0, 1.0),
            np.clip(valid.mean(), 0.0, 1.0),
        ],
        dtype=np.float32,
    )
    return features


def build_universal_inputs(
    frames: Sequence[np.ndarray],
    rect: Optional[Sequence[int]] = None,
    detected_line: Optional[Sequence[Sequence[float]]] = None,
    frame_rate: float = 15.0,
    source_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create image, track-time, and physical inputs from an event clip."""
    if not frames:
        raise ValueError("At least one frame is required")
    crops = np.stack([_crop_square(frame, rect, source_size) for frame in frames])
    residual_z, _global_sigma = _robust_residual_cube(crops)
    response = _compress_response(residual_z)

    groups = np.array_split(np.arange(len(response)), 3)
    temporal = [
        response[group].max(axis=0) if len(group) else np.zeros_like(response[0])
        for group in groups
    ]
    maximum = response.max(axis=0)
    peak_index = residual_z.argmax(axis=0).astype(np.float32)
    time_peak = peak_index / max(1, len(response) - 1)
    time_peak[residual_z.max(axis=0) < 3.0] = 0.0
    image = np.stack([maximum, temporal[0], temporal[1], temporal[2], time_peak])
    image = np.stack(
        [
            cv2.resize(channel, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
            for channel in image
        ]
    ).astype(np.float32)

    p1, p2 = _line_in_crop(detected_line, rect, source_size)
    line_length = max(8.0, math.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    aligned = _track_aligned_cube(response, p1, p2)
    longitudinal = aligned.max(axis=1)
    transverse = aligned.max(axis=2)
    transverse = np.stack(
        [
            cv2.resize(row[None, :], (KYMO_WIDTH, 1), interpolation=cv2.INTER_LINEAR)[0]
            for row in transverse
        ]
    )
    energy = np.percentile(aligned, 99.0, axis=(1, 2))
    energy_map = np.repeat(energy[:, None], KYMO_WIDTH, axis=1)
    kymograph = np.stack([longitudinal, transverse, energy_map])
    kymograph = np.stack(
        [
            cv2.resize(
                channel,
                (KYMO_WIDTH, KYMO_HEIGHT),
                interpolation=cv2.INTER_LINEAR,
            )
            for channel in kymograph
        ]
    ).astype(np.float32)

    features = _physical_features(
        response=response,
        residual_z=residual_z,
        line_length=line_length,
        fps=frame_rate,
    )
    return image, kymograph, features

def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class SeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.depth_norm = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.point_norm = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.activation = nn.SiLU(inplace=True)
        self.skip = None
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(_group_count(out_channels), out_channels),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs if self.skip is None else self.skip(inputs)
        output = self.activation(self.depth_norm(self.depthwise(inputs)))
        output = self.point_norm(self.pointwise(output))
        return self.activation(output + residual)


class UniversalEncoder(nn.Module):
    def __init__(self, in_channels: int, widths: Sequence[int]):
        super().__init__()
        first = int(widths[0])
        layers = [
            nn.Conv2d(in_channels, first, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(_group_count(first), first),
            nn.SiLU(inplace=True),
        ]
        current = first
        for width in widths:
            width = int(width)
            layers.append(SeparableBlock(current, width, stride=2 if current != width else 1))
            layers.append(SeparableBlock(width, width, stride=1))
            current = width
        self.network = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output_channels = current

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.pool(self.network(inputs)).flatten(1)


class MeteorFusionUniversal(nn.Module):
    """Small GroupNorm network intended to be robust to unseen cameras."""

    def __init__(self, feature_count: int = FEATURE_COUNT):
        super().__init__()
        self.image_encoder = UniversalEncoder(5, (24, 40, 72, 128, 192))
        self.kymo_encoder = UniversalEncoder(3, (16, 32, 64, 96))
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_count, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        total = self.image_encoder.output_channels + self.kymo_encoder.output_channels + 64
        self.classifier = nn.Sequential(
            nn.Linear(total, 160),
            nn.LayerNorm(160),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(160, 1),
        )

    def forward(
        self,
        image: torch.Tensor,
        kymograph: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        fused = torch.cat(
            [
                self.image_encoder(image),
                self.kymo_encoder(kymograph),
                self.feature_encoder(features),
            ],
            dim=1,
        )
        return self.classifier(fused).squeeze(1)


def augment_universal_inputs(
    image: torch.Tensor,
    kymograph: torch.Tensor,
    features: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sensor and geometry augmentation that preserves temporal semantics."""
    if torch.rand(()) < 0.5:
        image = torch.flip(image, dims=(-1,))
        kymograph = torch.flip(kymograph, dims=(-1,))
    if torch.rand(()) < 0.5:
        image = torch.flip(image, dims=(-2,))
    rotations = int(torch.randint(0, 4, ()).item())
    if rotations:
        image = torch.rot90(image, rotations, dims=(-2, -1))
    if torch.rand(()) < 0.25:
        image = image.clone()
        image[[1, 3]] = image[[3, 1]]
        signal = image[0] > 0.02
        image[4] = torch.where(signal, 1.0 - image[4], image[4])
        kymograph = torch.flip(kymograph, dims=(-2,))

    strength = float(torch.empty(()).uniform_(0.75, 1.25))
    gamma = float(torch.empty(()).uniform_(0.8, 1.25))
    image = image.clone()
    image[:4] = torch.clamp(image[:4] * strength, 0.0, 1.0).pow(gamma)
    kymograph = torch.clamp(kymograph * strength, 0.0, 1.0).pow(gamma)
    noise_sigma = float(torch.empty(()).uniform_(0.0, 0.035))
    if noise_sigma > 0.0:
        image[:4] = torch.clamp(
            image[:4] + torch.randn_like(image[:4]) * noise_sigma,
            0.0,
            1.0,
        )
        kymograph = torch.clamp(
            kymograph + torch.randn_like(kymograph) * noise_sigma,
            0.0,
            1.0,
        )
    return image, kymograph, features
