#!/usr/bin/env python3
"""Measure whether a timelapse's estimated fixed pattern changes over time."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def sky_level(image: np.ndarray) -> float:
    h, w = image.shape
    crop = image[int(h * .12):int(h * .78), int(w * .12):int(w * .88)]
    valid = crop[crop > 8]
    return float(np.median(valid)) if valid.size else float(np.median(crop))


def pattern_for_range(cap: cv2.VideoCapture, start: int, stop: int, samples: int) -> np.ndarray:
    frames: list[np.ndarray] = []
    levels: list[float] = []
    for index in np.linspace(start, stop - 1, samples, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if ok:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            frames.append(gray)
            levels.append(sky_level(gray))
    target = float(np.median(levels))
    return np.median(np.stack([frame * target / max(level, 1.0)
                               for frame, level in zip(frames, levels)]), axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--segments", type=int, default=6)
    parser.add_argument("--samples-per-segment", type=int, default=30)
    args = parser.parse_args()
    cap = cv2.VideoCapture(str(args.video))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    if count < args.segments * 3:
        raise SystemExit("Video is too short for the requested segmentation.")
    patterns = [pattern_for_range(cap, i * count // args.segments,
                                  (i + 1) * count // args.segments,
                                  args.samples_per_segment)
                for i in range(args.segments)]
    cap.release()
    reference = np.median(np.stack(patterns), axis=0)
    h, w = reference.shape
    valid = np.ones((h, w), bool)
    valid[:int(h * .04)] = False
    valid[int(h * .94):, int(w * .70):] = False  # changing timestamp
    # Each segment is assessed after its sky-level offset is removed.
    ref = reference[valid] - np.mean(reference[valid])
    reference_highpass = reference - cv2.GaussianBlur(reference, (0, 0), 18)
    ref_highpass = reference_highpass[valid] - np.mean(reference_highpass[valid])
    rows = []
    panels = []
    for i, pattern in enumerate(patterns):
        value = pattern[valid] - np.mean(pattern[valid])
        diff = value - ref
        correlation = float(np.corrcoef(value, ref)[0, 1])
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        mae = float(np.mean(np.abs(diff)))
        highpass = pattern - cv2.GaussianBlur(pattern, (0, 0), 18)
        highpass_value = highpass[valid] - np.mean(highpass[valid])
        highpass_correlation = float(np.corrcoef(highpass_value, ref_highpass)[0, 1])
        highpass_rmse = float(np.sqrt(np.mean((highpass_value - ref_highpass) ** 2)))
        start_s = i * count / args.segments / fps
        end_s = (i + 1) * count / args.segments / fps
        rows.append([i + 1, f"{start_s:.2f}", f"{end_s:.2f}",
                     f"{sky_level(pattern):.2f}", f"{correlation:.6f}",
                     f"{rmse:.4f}", f"{mae:.4f}", f"{highpass_correlation:.6f}",
                     f"{highpass_rmse:.4f}"])
        image_diff = pattern - reference
        image_diff -= np.mean(image_diff[valid])
        scale = max(float(np.percentile(np.abs(image_diff[valid]), 99)), 1.0)
        panel = np.clip(127.5 + image_diff * 100 / scale, 0, 255).astype(np.uint8)
        cv2.putText(panel, f"Segment {i + 1}: {start_s:.1f}-{end_s:.1f}s",
                    (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, cv2.LINE_AA)
        panels.append(panel)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "fixed_pattern_drift.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["segment", "start_seconds", "end_seconds", "normalised_sky_level",
                         "correlation_to_whole_video", "rmse_levels", "mae_levels",
                         "fine_pattern_correlation", "fine_pattern_rmse_levels"])
        writer.writerows(rows)
    thumb_w = 640
    thumbs = [cv2.resize(panel, (thumb_w, int(panel.shape[0] * thumb_w / panel.shape[1])))
              for panel in panels]
    montage = np.vstack([np.hstack(thumbs[:3]), np.hstack(thumbs[3:])])
    cv2.imwrite(str(args.output_dir / "fixed_pattern_drift_montage.png"), montage)
    print("segment,start-end(s),correlation,rmse,mae,fine-pattern-correlation,fine-pattern-rmse")
    for row in rows:
        print(f"{row[0]},{row[1]}-{row[2]},{row[4]},{row[5]},{row[6]},{row[7]},{row[8]}")


if __name__ == "__main__":
    main()
