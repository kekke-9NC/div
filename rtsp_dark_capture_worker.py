"""Capture or reprocess a 30-second RTSP fixed-pattern calibration video."""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


SAMPLE_COUNT = 90
GRADIENT_SIGMA = 25
# Camera-generated timestamps/overlays seen in supported RTSP recordings.
# These are deliberately excluded from calibration: a temporal median preserves
# the date and hour digits and would otherwise burn their inverse into every
# corrected frame.
OVERLAY_REGIONS = (
    # AtomCam timestamp used by the recorded-folder input.
    (0.00, 0.00, 0.25, 0.09),
    # App/generated timestamp or status overlay used by older recordings.
    (0.70, 0.94, 1.00, 1.00),
)


def log(message: str) -> None:
    print(message, flush=True)


def robust_level(gray: np.ndarray) -> float:
    """Estimate the typical level while ignoring the frame edge."""
    height, width = gray.shape
    crop = gray[int(height * .12):int(height * .78),
                int(width * .12):int(width * .88)]
    valid = crop[crop > 8]
    return float(np.median(valid)) if valid.size else float(np.median(crop))


def build_fixed_pattern(samples: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, float]:
    """Use the same normalised temporal-median method as timelapse analysis."""
    levels = [robust_level(frame) for frame in samples]
    # A capped lens can have a level close to zero.  Keep its native level
    # rather than scaling every frame to black.
    target = max(1.0, float(np.median(levels)))
    normalised = [
        cv2.convertScaleAbs(frame, alpha=target / max(level, 1.0))
        for frame, level in zip(samples, levels)
    ]
    pattern = np.median(np.stack(normalised), axis=0).astype(np.uint8)
    # Remove only broad, smooth illumination (including a cap light leak).
    # What remains is a signed, zero-centred fine fixed-pattern correction.
    smooth = cv2.GaussianBlur(pattern.astype(np.float32), (0, 0), GRADIENT_SIGMA)
    correction = np.rint(pattern.astype(np.float32) - smooth).astype(np.int16)
    # Camera and generated RTSP videos commonly have changing timestamps.
    # They are not fixed sensor patterns and must never be calibrated into the
    # correction map.
    height, width = correction.shape
    for left, top, right, bottom in OVERLAY_REGIONS:
        correction[
            int(height * top):int(np.ceil(height * bottom)),
            int(width * left):int(np.ceil(width * right)),
        ] = 0
    return pattern, correction, target


def sample_video(video_path: Path, sample_count: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise RuntimeError("動画に読み取り可能なフレームがありません。")
    positions = set(np.linspace(0, frame_count - 1, min(sample_count, frame_count), dtype=int))
    samples: list[np.ndarray] = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index in positions:
            samples.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if len(samples) % 10 == 0 or len(samples) == len(positions):
                log(f"PROGRESS {len(samples)} {len(positions)}")
        index += 1
    cap.release()
    return samples


def choose_sample_locations(
    frame_counts: list[int],
    sample_count: int,
    random_seed: int,
) -> list[tuple[int, int]]:
    """Choose reproducible, time-uniform random frames across multiple clips."""
    usable_counts = np.asarray([max(0, int(value)) for value in frame_counts], dtype=np.int64)
    total_frames = int(usable_counts.sum())
    if total_frames <= 0:
        return []
    count = min(max(1, int(sample_count)), total_frames)
    positions = np.sort(
        np.random.default_rng(int(random_seed)).choice(
            total_frames,
            size=count,
            replace=False,
        )
    )
    ends = np.cumsum(usable_counts)
    video_indices = np.searchsorted(ends, positions, side="right")
    starts = np.concatenate(([0], ends[:-1]))
    return [
        (int(video_index), int(position - starts[video_index]))
        for position, video_index in zip(positions, video_indices)
    ]


def sample_video_directory(
    directory: Path,
    sample_count: int,
    random_seed: int,
) -> tuple[list[np.ndarray], int, int]:
    """Randomly sample an entire recording folder without concatenating it."""
    extensions = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}
    paths = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )
    if not paths:
        raise RuntimeError(f"動画が見つかりません: {directory}")

    readable_paths: list[Path] = []
    frame_counts: list[int] = []
    for path in paths:
        cap = cv2.VideoCapture(str(path))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        cap.release()
        if count > 0:
            readable_paths.append(path)
            frame_counts.append(count)
    if not readable_paths:
        raise RuntimeError(f"読み取り可能な動画がありません: {directory}")

    locations = choose_sample_locations(frame_counts, sample_count, random_seed)
    locations_by_video: dict[int, list[int]] = {}
    for video_index, frame_index in locations:
        locations_by_video.setdefault(video_index, []).append(frame_index)

    samples: list[np.ndarray] = []
    for video_index, frame_indices in locations_by_video.items():
        cap = cv2.VideoCapture(str(readable_paths[video_index]))
        if not cap.isOpened():
            continue
        for frame_index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if ok and frame is not None:
                samples.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                if len(samples) % 10 == 0 or len(samples) == len(locations):
                    log(f"PROGRESS {len(samples)} {len(locations)}")
        cap.release()
    return samples, len(readable_paths), len(locations_by_video)


def capture_rtsp_video(url: str, video_path: Path, duration: float,
                       sample_count: int) -> list[np.ndarray]:
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
    if not cap.isOpened():
        raise RuntimeError("RTSPストリームを開けません。")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    log(f"STATUS opened {width}x{height} @ {fps:.2f} fps")
    for _ in range(10):
        cap.read()

    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))
    if not writer.isOpened():
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("ダーク動画の保存を開始できません。")

    samples: list[np.ndarray] = []
    started = time.monotonic()
    next_sample_at = 0.0
    last_report = -1
    while time.monotonic() - started < duration:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        writer.write(frame)
        elapsed = time.monotonic() - started
        if elapsed >= next_sample_at and len(samples) < sample_count:
            samples.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            next_sample_at = duration * len(samples) / sample_count
        whole_seconds = int(elapsed)
        if whole_seconds != last_report:
            last_report = whole_seconds
            log(f"CAPTURE {whole_seconds} {int(duration)} {len(samples)} {sample_count}")
    writer.release()
    cap.release()
    return samples


def save_result(
    output: Path,
    preview: Path,
    source_path: Path,
    samples: list[np.ndarray],
    *,
    source_file_count: int = 1,
    sampled_file_count: int = 1,
    sampling_strategy: str = "evenly_spaced_single_video",
    random_seed: int = -1,
) -> None:
    if len(samples) < 10:
        raise RuntimeError(f"十分なフレームを取得できませんでした ({len(samples)}枚)。")
    pattern, correction, target = build_fixed_pattern(samples)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        fixed_pattern=pattern,
        fixed_correction=correction,
        source_video=np.array(str(source_path)),
        source_file_count=np.array(source_file_count),
        sampled_file_count=np.array(sampled_file_count),
        sampling_strategy=np.array(sampling_strategy),
        random_seed=np.array(random_seed),
        frames=np.array(len(samples)),
        method=np.array("normalised_temporal_median_fine_pattern_v1"),
        gradient_sigma=np.array(GRADIENT_SIGMA),
        created_at=np.array(datetime.now().isoformat(timespec="seconds")),
    )
    scale = max(float(np.percentile(np.abs(correction), 99.5)), 1.0)
    xray = np.clip(127.5 + correction * 110.0 / scale, 0, 255).astype(np.uint8)
    cv2.imwrite(str(preview), xray)
    log(
        f"RESULT frames={len(samples)} reference={target:.2f} "
        f"shape={pattern.shape[1]}x{pattern.shape[0]} source={source_path} "
        f"files={sampled_file_count}/{source_file_count} sampling={sampling_strategy}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url")
    source.add_argument("--input-video", type=Path)
    source.add_argument("--input-directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    parser.add_argument("--video", type=Path,
                        help="Where a newly captured 30-second video is saved")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--samples", type=int, default=SAMPLE_COUNT)
    parser.add_argument("--random-seed", type=int, default=0)
    args = parser.parse_args()
    try:
        source_file_count = 1
        sampled_file_count = 1
        sampling_strategy = "evenly_spaced_single_video"
        random_seed = -1
        if args.input_directory:
            video_path = args.input_directory.resolve()
            log(f"STATUS analysing recording folder {video_path}")
            samples, source_file_count, sampled_file_count = sample_video_directory(
                video_path,
                args.samples,
                args.random_seed,
            )
            sampling_strategy = "random_frames_across_recording_folder"
            random_seed = args.random_seed
        elif args.input_video:
            video_path = args.input_video.resolve()
            log(f"STATUS analysing saved video {video_path}")
            samples = sample_video(video_path, args.samples)
        else:
            if args.video is None:
                raise RuntimeError("RTSP撮影には --video が必要です。")
            video_path = args.video.resolve()
            log("STATUS recording 30-second fixed-pattern video")
            samples = capture_rtsp_video(args.url, video_path, args.duration, args.samples)
        save_result(
            args.output.resolve(),
            args.preview.resolve(),
            video_path,
            samples,
            source_file_count=source_file_count,
            sampled_file_count=sampled_file_count,
            sampling_strategy=sampling_strategy,
            random_seed=random_seed,
        )
        return 0
    except Exception as exc:
        log(f"ERROR {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
