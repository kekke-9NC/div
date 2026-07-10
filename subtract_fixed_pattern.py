#!/usr/bin/env python3
"""Subtract a fixed-pattern estimate from a video without changing its mean level."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def robust_level(image: np.ndarray) -> float:
    height, width = image.shape
    crop = image[int(height * 0.12):int(height * 0.78),
                 int(width * 0.12):int(width * 0.88)]
    valid = crop[crop > 8]
    return float(np.median(valid)) if valid.size else float(np.median(crop))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("fixed_component", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot read video: {args.video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    pattern = cv2.imread(str(args.fixed_component), cv2.IMREAD_GRAYSCALE)
    if pattern is None:
        raise SystemExit(f"Cannot read fixed component: {args.fixed_component}")
    pattern = cv2.resize(pattern, (width, height), interpolation=cv2.INTER_LINEAR)
    correction = pattern.astype(np.float32) - robust_level(pattern)

    # The lower-right timestamp changes every frame and must not participate in
    # the calibration image.  Feather the boundary to avoid a visible seam.
    timestamp_mask = np.zeros((height, width), np.float32)
    timestamp_mask[int(height * 0.94):, int(width * 0.72):] = 1.0
    timestamp_mask = cv2.GaussianBlur(timestamp_mask, (0, 0), 12)
    correction *= 1.0 - timestamp_mask

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height)
    )
    if not writer.isOpened():
        raise SystemExit("Cannot initialise H.264 output writer.")
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        corrected = np.clip(frame.astype(np.float32) - correction[..., None], 0, 255)
        writer.write(corrected.astype(np.uint8))
        count += 1
    writer.release()
    cap.release()
    print(f"Wrote {count} frames: {args.output}")


if __name__ == "__main__":
    main()
