#!/usr/bin/env python3
"""Isolated training worker used by the Tk application."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np

import noise_twin
import noise_twin_training


def _load_correction(path: str):
    if not path or not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=False) as data:
        if "fixed_correction" in data:
            return data["fixed_correction"].astype(np.int16)
        if "dark_frame" in data:
            return data["dark_frame"].astype(np.uint8)
    return None


def _progress(phase: str, done: int, total: int) -> None:
    print(f"PROGRESS {phase} {done} {total}", flush=True)


def _first_sequence(path: str):
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(noise_twin.TEMPORAL_WINDOW):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if len(frames) != noise_twin.TEMPORAL_WINDOW:
        raise noise_twin.NoiseTwinError("速度測定用フレームを取得できません。")
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", action="append", default=[])
    parser.add_argument("--rtsp-url", default="")
    parser.add_argument("--rtsp-duration", type=int, default=600)
    parser.add_argument("--output", required=True)
    parser.add_argument("--correction", default="")
    parser.add_argument("--background-steps", type=int, default=1500)
    parser.add_argument("--gate-steps", type=int, default=750)
    parser.add_argument("--validation-injections", type=int, default=10_000)
    parser.add_argument("--rtsp-benchmark-seconds", type=int, default=1800)
    args = parser.parse_args()

    captured_path = ""
    try:
        paths = list(args.video)
        if args.rtsp_url:
            capture_dir = Path(args.output).parent
            capture_dir.mkdir(parents=True, exist_ok=True)
            captured_path = str(capture_dir / f"{Path(args.output).stem}_training_capture.mp4")
            print("STATUS RTSP学習映像を収録しています", flush=True)
            noise_twin_training.capture_rtsp_training_video(
                args.rtsp_url,
                captured_path,
                duration_seconds=args.rtsp_duration,
                progress_callback=_progress,
            )
            paths.append(captured_path)
        if not paths:
            raise noise_twin.NoiseTwinError("学習動画が指定されていません。")
        correction = _load_correction(args.correction)
        print("STATUS NoiseTwin背景モデルを学習しています", flush=True)
        metadata = noise_twin_training.train_noise_twin(
            paths,
            args.output,
            correction=correction,
            source_identifier=args.rtsp_url or paths[0],
            background_steps=args.background_steps,
            gate_steps=args.gate_steps,
            validation_injections=args.validation_injections,
            progress_callback=_progress,
        )
        dropped_frames = 0
        realtime_test_seconds = 0.0
        if args.rtsp_url:
            print("STATUS RTSP 30分連続・欠落なし検証を実行しています", flush=True)
            measured_fps, dropped_frames, realtime_test_seconds = (
                noise_twin_training.benchmark_rtsp_stream(
                    args.output,
                    args.rtsp_url,
                    correction=correction,
                    duration_seconds=args.rtsp_benchmark_seconds,
                    progress_callback=_progress,
                )
            )
            measured_fps = max(
                measured_fps,
                noise_twin_training.benchmark_full_frame(
                    args.output, _first_sequence(paths[0])
                ),
            )
        else:
            print("STATUS 実フレーム速度を測定しています", flush=True)
            measured_fps = noise_twin_training.benchmark_full_frame(
                args.output, _first_sequence(paths[0])
            )
        validation = replace(
            metadata.validation,
            realtime_fps=measured_fps,
            realtime_test_seconds=realtime_test_seconds,
            dropped_frames=dropped_frames,
        )
        metadata = replace(metadata, validation=validation)
        noise_twin.save_metadata(args.output, metadata)
        print("RESULT " + json.dumps({
            "model_path": args.output,
            "model_id": metadata.model_id,
            "validated": metadata.validation.validated,
            "realtime_fps": metadata.validation.realtime_fps,
            "flux_retention": metadata.validation.flux_retention,
            "missed_fraction": metadata.validation.missed_fraction,
            "realtime_test_seconds": metadata.validation.realtime_test_seconds,
            "dropped_frames": metadata.validation.dropped_frames,
        }, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        print(f"ERROR {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
