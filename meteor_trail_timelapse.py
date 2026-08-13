"""Meteor-preserving lighten-composite timelapse generation.

The ordinary timelapse path samples one frame at a time.  That is efficient,
but a short meteor can fall between two samples.  This module provides a
separate presentation mode: it takes the pixel-wise maximum over a short
source-time window, then lets that composite fade into the next output frame.
Static stars remain visible while transient streaks survive long enough to be
seen in a fast, X-style observing timelapse.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Callable, Iterable, Optional, Sequence

import cv2
import numpy as np

import video_encoding


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
_RTSP_PATH = re.compile(
    r"(?:^|/)(?P<date>20\d{6})/(?P<hour>[01]\d|2[0-3])/(?P<minute>[0-5]\d)\.\w+$"
)


@dataclass(frozen=True)
class TrailTimelapseSettings:
    """Presentation settings for the meteor-trail timelapse."""

    source_seconds_per_output_frame: float = 2.0
    output_fps: float = 25.0
    output_size: tuple[int, int] = (1280, 720)
    gamma: float = 1.35
    contrast: float = 1.12
    brightness: float = 1.0
    trail_decay: float = 0.78

    def validate(self) -> "TrailTimelapseSettings":
        width, height = (int(self.output_size[0]), int(self.output_size[1]))
        if width < 16 or height < 16:
            raise ValueError("output_size must be at least 16x16")
        if width % 2 or height % 2:
            raise ValueError("output_size must contain even dimensions")
        if self.source_seconds_per_output_frame <= 0:
            raise ValueError("source_seconds_per_output_frame must be positive")
        if self.output_fps <= 0:
            raise ValueError("output_fps must be positive")
        if self.gamma <= 0 or self.contrast <= 0 or self.brightness <= 0:
            raise ValueError("gamma, contrast, and brightness must be positive")
        if not 0.0 <= self.trail_decay < 1.0:
            raise ValueError("trail_decay must be in [0, 1)")
        return TrailTimelapseSettings(
            source_seconds_per_output_frame=float(self.source_seconds_per_output_frame),
            output_fps=float(self.output_fps),
            output_size=(width, height),
            gamma=float(self.gamma),
            contrast=float(self.contrast),
            brightness=float(self.brightness),
            trail_decay=float(self.trail_decay),
        )


def _rtsp_sort_key(path: str) -> tuple:
    normalized = str(Path(path).expanduser().resolve()).replace(os.sep, "/")
    match = _RTSP_PATH.search(normalized)
    if match:
        return (
            0,
            match.group("date"),
            int(match.group("hour")),
            int(match.group("minute")),
            normalized,
        )
    try:
        stat = Path(path).stat()
        return (1, stat.st_mtime_ns, normalized)
    except OSError:
        return (2, normalized)


def discover_video_files(inputs: Sequence[str | os.PathLike[str]]) -> list[str]:
    """Expand files/directories into a chronological, de-duplicated list."""

    found: set[str] = set()
    for raw in inputs:
        path = Path(raw).expanduser()
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = [item for item in path.rglob("*") if item.is_file()]
        else:
            continue
        for item in candidates:
            if item.suffix.lower() not in VIDEO_SUFFIXES:
                continue
            if "_temp" in item.stem.lower() or item.name.startswith("."):
                continue
            found.add(str(item.resolve()))
    return sorted(found, key=_rtsp_sort_key)


def _tone_lut(settings: TrailTimelapseSettings) -> np.ndarray:
    values = np.arange(256, dtype=np.float32) / 255.0
    values = np.power(values, 1.0 / settings.gamma) * 255.0
    values = (values - 32.0) * settings.contrast + 32.0
    values *= settings.brightness
    return np.clip(values, 0.0, 255.0).astype(np.uint8)


def _resize_frame(frame: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    width, height = output_size
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _iter_frames(paths: Iterable[str]):
    """Yield (frame, source_fps) while keeping only one decoder open."""

    for path in paths:
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            capture.release()
            continue
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            if not 1.0 <= fps <= 120.0:
                fps = 25.0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                yield frame, fps
        finally:
            capture.release()


def _count_frames(paths: Sequence[str]) -> int:
    total = 0
    for path in paths:
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            capture.release()
            continue
        try:
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if count > 0:
                total += count
        finally:
            capture.release()
    return total


def create_meteor_trail_timelapse(
    video_paths: Sequence[str],
    output_path: str,
    *,
    settings: Optional[TrailTimelapseSettings] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> bool:
    """Create a meteor-preserving lighten-composite timelapse.

    One output frame represents ``source_seconds_per_output_frame`` seconds of
    source video.  The source frames in that interval are combined with a
    maximum operation, so a meteor remains visible even when it lasts only a
    few source frames.  ``trail_decay`` adds a restrained afterglow to the
    following output frame without drawing artificial markers.
    """

    normalized_paths = discover_video_files(video_paths)
    if not normalized_paths:
        if progress_callback:
            progress_callback("有効な動画ファイルがありません。")
        return False
    options = (settings or TrailTimelapseSettings()).validate()
    total_frames = _count_frames(normalized_paths)
    lut = _tone_lut(options)
    writer = video_encoding.FFmpegFrameWriter(
        output_path,
        options.output_fps,
        options.output_size,
        {"codec": "h264", "quality": "standard", "bitrate_mbps": 12},
    )
    if not writer.isOpened():
        if progress_callback:
            progress_callback("動画エンコーダーを起動できません。")
        return False

    window_max: Optional[np.ndarray] = None
    trail: Optional[np.ndarray] = None
    elapsed = 0.0
    processed = 0
    emitted = 0
    success = False

    def emit_window() -> None:
        nonlocal window_max, trail, elapsed, emitted
        if window_max is None:
            return
        composite = window_max
        if trail is None:
            trail = composite.copy()
        else:
            faded = np.clip(trail.astype(np.float32) * options.trail_decay, 0, 255)
            trail = np.maximum(composite, faded.astype(np.uint8))
        writer.write(trail)
        emitted += 1
        window_max = None
        elapsed = 0.0

    try:
        for frame, fps in _iter_frames(normalized_paths):
            toned = cv2.LUT(_resize_frame(frame, options.output_size), lut)
            window_max = toned.copy() if window_max is None else np.maximum(window_max, toned)
            elapsed += 1.0 / fps
            processed += 1
            if elapsed >= options.source_seconds_per_output_frame:
                emit_window()
            if progress_callback and (processed == 1 or processed % 250 == 0):
                fraction = processed / max(1, total_frames)
                progress_callback(
                    f"流星トレイル合成中: {fraction * 100:.1f}% "
                    f"（入力{processed}/{total_frames or '?'}フレーム、出力{emitted}フレーム）"
                )
        emit_window()
        writer.release()
        success = True
    except Exception as exc:
        try:
            writer.release()
        except Exception:
            pass
        if progress_callback:
            progress_callback(f"流星トレイルタイムラプスに失敗しました: {exc}")
        return False

    if progress_callback:
        speedup = options.source_seconds_per_output_frame * options.output_fps
        progress_callback(
            f"流星トレイルタイムラプスを保存しました: {output_path} "
            f"（約{speedup:.1f}倍速、{emitted}フレーム）"
        )
    return success


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="動画ファイルまたはフォルダ")
    parser.add_argument("-o", "--output", required=True, help="出力MP4")
    parser.add_argument(
        "--source-seconds-per-frame", type=float, default=2.0,
        help="入力の何秒を1出力フレームに合成するか（既定: 2）",
    )
    parser.add_argument("--fps", type=float, default=25.0, help="出力FPS（既定: 25）")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    settings = TrailTimelapseSettings(
        source_seconds_per_output_frame=args.source_seconds_per_frame,
        output_fps=args.fps,
        output_size=(args.width, args.height),
    )
    return 0 if create_meteor_trail_timelapse(
        args.inputs, args.output, settings=settings, progress_callback=print
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
