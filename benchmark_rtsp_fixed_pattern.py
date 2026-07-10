#!/usr/bin/env python3
"""Benchmark fixed-pattern extraction from the RTSP URL in app_settings.json."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np


def level(gray: np.ndarray) -> float:
    h, w = gray.shape
    crop = gray[int(h * .12):int(h * .78), int(w * .12):int(w * .88)]
    valid = crop[crop > 8]
    return float(np.median(valid)) if valid.size else float(np.median(crop))


def read_pattern(cap: cv2.VideoCapture, frames_required: int, interval_seconds: float) -> tuple[np.ndarray, float]:
    frames: list[np.ndarray] = []
    values: list[float] = []
    started = time.monotonic()
    last_accepted = started - interval_seconds
    attempts = 0
    # RTSP can drop individual H.264 packets.  Allow ample retries; timing is
    # based on successfully acquired frames, not decode failures.
    while len(frames) < frames_required and attempts < frames_required * 20:
        attempts += 1
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        # Keep consuming the RTSP stream continuously.  Pausing reads lets its
        # small live buffer overflow, so temporal spacing is applied only to
        # which decoded frames enter the estimator.
        if time.monotonic() - last_accepted < interval_seconds:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        values.append(level(gray))
        last_accepted = time.monotonic()
    if len(frames) != frames_required:
        raise RuntimeError(f"Only {len(frames)}/{frames_required} RTSP frames were read")
    target = float(np.median(values))
    # Work in uint8 to bound peak memory; this is sufficient for an 8-bit H.264
    # RTSP stream and preserves the same robust-median estimator used above.
    normalized = [cv2.convertScaleAbs(frame, alpha=target / max(value, 1.0))
                  for frame, value in zip(frames, values)]
    pattern = np.median(np.stack(normalized), axis=0).astype(np.uint8)
    return pattern, time.monotonic() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, default=Path("app_settings.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--interval-seconds", type=float, default=0.0,
                        help="Wait after each captured frame to spread samples over time")
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())
    urls = settings.get("rtsp_urls", [])
    if not urls:
        raise SystemExit("No RTSP URL is configured.")
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    connect_started = time.monotonic()
    cap = cv2.VideoCapture(str(urls[0]), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise SystemExit("Cannot open the configured RTSP stream.")
    # Discard startup-buffer frames before timing the measurement.
    for _ in range(10):
        cap.read()
    connection_seconds = time.monotonic() - connect_started
    patterns: list[np.ndarray] = []
    times: list[float] = []
    for _ in range(args.repeats):
        pattern, elapsed = read_pattern(cap, args.frames, args.interval_seconds)
        patterns.append(pattern)
        times.append(elapsed)
    cap.release()

    reference = np.median(np.stack(patterns), axis=0)
    h, w = reference.shape
    # Central upper sky excludes the timestamp, frame edge, and usual cloud bank.
    region = np.s_[int(h * .12):int(h * .62), int(w * .12):int(w * .88)]
    ref = reference[region].astype(np.float32)
    ref -= ref.mean()
    rows = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, (pattern, elapsed) in enumerate(zip(patterns, times), 1):
        value = pattern[region].astype(np.float32)
        value -= value.mean()
        correlation = float(np.corrcoef(value.ravel(), ref.ravel())[0, 1])
        rmse = float(np.sqrt(np.mean((value - ref) ** 2)))
        rows.append((index, elapsed, correlation, rmse))
        cv2.imwrite(str(args.output_dir / f"rtsp_pattern_{index}.png"), pattern)
    csv = args.output_dir / "rtsp_fixed_pattern_benchmark.csv"
    csv.write_text(
        "repeat,capture_seconds,central_upper_correlation,central_upper_rmse_levels\n"
        + "\n".join(f"{i},{seconds:.3f},{corr:.6f},{rmse:.4f}"
                      for i, seconds, corr, rmse in rows)
        + f"\nconnection_and_warmup_seconds,{connection_seconds:.3f}\n",
        encoding="utf-8",
    )
    print(f"connection_and_warmup_seconds={connection_seconds:.3f}")
    for index, seconds, correlation, rmse in rows:
        print(f"repeat={index} capture_seconds={seconds:.3f} correlation={correlation:.6f} rmse={rmse:.4f}")


if __name__ == "__main__":
    main()
