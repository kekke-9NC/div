#!/usr/bin/env python3
"""Extract a best-effort fixed-pattern map from a timelapse video.

The source frames are exposure-normalized before a robust temporal median is
calculated.  Moving clouds and stars are therefore suppressed, while sensor
fixed-pattern noise, vignetting and static overlays remain visible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=90,
                        help="Evenly spaced frames to analyse (default: 90)")
    return parser.parse_args()


def robust_level(gray: np.ndarray) -> float:
    """Sky brightness estimate, excluding the black edge and timestamp area."""
    height, width = gray.shape
    crop = gray[int(height * 0.12):int(height * 0.78),
                int(width * 0.12):int(width * 0.88)]
    valid = crop[crop > 8]
    return float(np.median(valid)) if valid.size else float(np.median(crop))


def main() -> None:
    args = parse_args()
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot read video: {args.video}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < 1:
        raise SystemExit("The video contains no readable frames.")
    positions = np.linspace(0, frame_count - 1,
                            min(max(3, args.samples), frame_count), dtype=int)
    frames: list[np.ndarray] = []
    levels: list[float] = []
    for position in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(position))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        frames.append(gray)
        levels.append(robust_level(gray))
    cap.release()
    if len(frames) < 3:
        raise SystemExit("Too few frames could be decoded.")

    target_level = float(np.median(levels))
    normalized = np.stack([frame * (target_level / max(level, 1.0))
                           for frame, level in zip(frames, levels)])
    fixed = np.median(normalized, axis=0)
    deviation = np.median(np.abs(normalized - fixed), axis=0)

    # The fixed field is displayed in two complementary ways: its absolute
    # brightness (vignetting / masking) and a contrast-expanded residual
    # (pixel-scale or compression-pattern noise).
    fixed_u8 = np.clip(fixed, 0, 255).astype(np.uint8)
    residual = fixed - cv2.GaussianBlur(fixed, (0, 0), 25)
    scale = max(float(np.percentile(np.abs(residual), 99.5)), 1.0)
    residual_u8 = np.clip(127.5 + residual * (110.0 / scale), 0, 255).astype(np.uint8)
    stability_u8 = np.clip(255 - deviation * 20, 0, 255).astype(np.uint8)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.video.stem
    cv2.imwrite(str(args.output_dir / f"{stem}_fixed_component.png"), fixed_u8)
    cv2.imwrite(str(args.output_dir / f"{stem}_fixed_noise_xray.png"), residual_u8)
    cv2.imwrite(str(args.output_dir / f"{stem}_stability.png"), stability_u8)
    print(f"Analysed {len(frames)} frames; normalised median brightness: {target_level:.1f}")
    print(f"Fixed component: {args.output_dir / f'{stem}_fixed_component.png'}")
    print(f"Noise X-ray: {args.output_dir / f'{stem}_fixed_noise_xray.png'}")
    print(f"Stability map: {args.output_dir / f'{stem}_stability.png'}")


if __name__ == "__main__":
    main()
