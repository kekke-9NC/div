"""Build conservative per-video detection masks for all-sky monochrome videos."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np


def sample_video_median(video_path: str, sample_count: int = 7, max_width: int = 640):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"動画を開けません: {video_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            positions = list(range(sample_count))
        else:
            positions = np.linspace(0, max(0, total - 1), sample_count + 2, dtype=int)[1:-1]
        frames = []
        original_size = None
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
                gray = cv2.resize(gray, (max_width, max(1, round(gray.shape[0] * scale))))
            frames.append(gray)
        if not frames or original_size is None:
            raise IOError(f"マスク用フレームを読み込めません: {video_path}")
        return np.median(np.stack(frames), axis=0).astype(np.uint8), original_size
    finally:
        cap.release()


def build_mask_from_median(median_gray: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """Return 255 for sky and 0 for bottom-connected structures/overlays."""
    gray = median_gray.astype(np.uint8)
    height, width = gray.shape
    lower_start = int(height * 0.52)
    lower = gray[lower_start:]

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    mean = cv2.boxFilter(gray.astype(np.float32), -1, (15, 15))
    mean_sq = cv2.boxFilter(gray.astype(np.float32) ** 2, -1, (15, 15))
    local_std = np.sqrt(np.maximum(0.0, mean_sq - mean ** 2))

    grad_threshold = max(12.0, float(np.percentile(gradient[lower_start:], 82)))
    std_threshold = max(5.0, float(np.percentile(local_std[lower_start:], 78)))
    bright_threshold = max(145.0, float(np.percentile(lower, 91)))
    candidates = (
        (gradient >= grad_threshold)
        | (local_std >= std_threshold)
        | (gray >= bright_threshold)
    ).astype(np.uint8) * 255
    candidates[:lower_start] = 0

    # Join fragmented lights/building edges, then retain only structures connected to the bottom.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 9))
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_CLOSE, kernel, iterations=2)
    candidates = cv2.dilate(candidates, np.ones((9, 9), np.uint8), iterations=1)
    candidates[-max(3, height // 35):] = 255
    count, labels = cv2.connectedComponents(candidates)
    bottom_labels = np.unique(labels[-max(3, height // 30):])
    connected = np.isin(labels, bottom_labels[bottom_labels != 0])

    # Convert the connected structure into a smooth skyline and mask everything below it.
    default_horizon = int(height * 0.94)
    horizon = np.full(width, default_horizon, dtype=np.float32)
    for x in range(width):
        ys = np.flatnonzero(connected[:, x])
        if ys.size:
            horizon[x] = max(lower_start, int(ys[0]) - max(3, height // 90))
    horizon_u8 = np.round(horizon * 255.0 / max(1, height - 1)).astype(np.uint8)
    horizon = (
        cv2.medianBlur(horizon_u8.reshape(1, -1), 31).ravel().astype(np.float32)
        * max(1, height - 1) / 255.0
    )
    # Prevent single narrow towers/lights from removing an excessive part of the sky.
    baseline = cv2.GaussianBlur(horizon.reshape(1, -1), (81, 1), 0).ravel()
    horizon = np.maximum(horizon, baseline - height * 0.10)

    mask = np.full((height, width), 255, dtype=np.uint8)
    for x, y in enumerate(horizon.astype(int)):
        mask[max(0, y):, x] = 0

    # Timestamp overlay is fixed in the lower-right corner in this camera archive.
    timestamp_top = int(height * 0.90)
    timestamp_left = int(width * 0.74)
    mask[timestamp_top:, timestamp_left:] = 0
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)
    stats = {
        "sky_fraction": float(np.count_nonzero(mask) / mask.size),
        "median_horizon_fraction": float(np.median(horizon) / height),
        "minimum_horizon_fraction": float(np.min(horizon) / height),
    }
    return mask, stats


def _cache_key(video_path: str) -> str:
    stat = os.stat(video_path)
    raw = f"{Path(video_path).resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20]


def create_auto_mask(video_path: str, cache_dir: str) -> Tuple[np.ndarray, str, Dict[str, float]]:
    """Create or load a cached full-resolution mask and its visual preview."""
    cache = Path(cache_dir).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    key = _cache_key(video_path)
    mask_path = cache / f"{key}_mask.png"
    preview_path = cache / f"{key}_preview.jpg"
    metadata_path = cache / f"{key}.json"
    if mask_path.exists() and metadata_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            return mask, str(preview_path), metadata.get("stats", {})

    median_small, (width, height) = sample_video_median(video_path)
    mask_small, stats = build_mask_from_median(median_small)
    mask = cv2.resize(mask_small, (width, height), interpolation=cv2.INTER_NEAREST)
    median_full = cv2.resize(median_small, (width, height), interpolation=cv2.INTER_LINEAR)
    overlay = cv2.cvtColor(median_full, cv2.COLOR_GRAY2BGR)
    excluded = mask == 0
    overlay[excluded] = (0.35 * overlay[excluded] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
    boundary = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((7, 7), np.uint8))
    overlay[boundary > 0] = (0, 255, 255)
    if not cv2.imwrite(str(mask_path), mask):
        raise IOError(f"自動マスクを書き込めません: {mask_path}")
    if not cv2.imwrite(str(preview_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise IOError(f"自動マスクプレビューを書き込めません: {preview_path}")
    metadata = {
        "video_path": str(Path(video_path).resolve()),
        "mask_path": str(mask_path),
        "preview_path": str(preview_path),
        "stats": stats,
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return mask, str(preview_path), stats


def combine_masks(manual_mask: Optional[np.ndarray], automatic_mask: np.ndarray) -> np.ndarray:
    if manual_mask is None:
        return automatic_mask
    resized = manual_mask
    if resized.shape[:2] != automatic_mask.shape[:2]:
        resized = cv2.resize(
            resized, (automatic_mask.shape[1], automatic_mask.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    if resized.ndim == 3:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return cv2.bitwise_and(resized.astype(np.uint8), automatic_mask)
