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
from datetime import datetime, timedelta
import os
from pathlib import Path
import re
from typing import Callable, Iterable, Optional, Sequence

import cv2
import numpy as np

import video_encoding
import media_time


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
    # Applied once per emitted frame, not once per source frame.  A high
    # value lets the fixed camera's slowly moving stars build real trails.
    trail_decay: float = 0.985
    timestamp_enabled: bool = True
    timestamp_position: str = "bottom_right"
    timestamp_size_percent: float = 2.6

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
        if not 0.0 < self.trail_decay < 1.0:
            raise ValueError("trail_decay must be in (0, 1)")
        position = {
            "右下": "bottom_right",
            "左下": "bottom_left",
            "右上": "top_right",
            "左上": "top_left",
        }.get(str(self.timestamp_position), str(self.timestamp_position))
        if position not in {"bottom_right", "bottom_left", "top_right", "top_left"}:
            raise ValueError("timestamp_position is invalid")
        if self.timestamp_size_percent <= 0:
            raise ValueError("timestamp_size_percent must be positive")
        return TrailTimelapseSettings(
            source_seconds_per_output_frame=float(self.source_seconds_per_output_frame),
            output_fps=float(self.output_fps),
            output_size=(width, height),
            gamma=float(self.gamma),
            contrast=float(self.contrast),
            brightness=float(self.brightness),
            trail_decay=float(self.trail_decay),
            timestamp_enabled=bool(self.timestamp_enabled),
            timestamp_position=position,
            timestamp_size_percent=float(self.timestamp_size_percent),
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


def _source_start_time(path: str) -> datetime:
    """Return the capture start time, preferring the RTSP path timestamp."""

    path_timestamp = getattr(media_time, "_path_time", lambda _path: None)(path)
    if path_timestamp is not None:
        return path_timestamp
    timestamp, _source = media_time.get_media_start_time(path)
    return timestamp or datetime.now()


def _iter_frames(paths: Iterable[str]):
    """Yield (frame, source_fps, capture_time) while keeping one decoder open."""

    for path in paths:
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            capture.release()
            continue
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            if not 1.0 <= fps <= 120.0:
                fps = 25.0
            start_time = _source_start_time(path)
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                yield frame, fps, start_time + timedelta(seconds=frame_index / fps)
                frame_index += 1
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


def _draw_timestamp(
    frame: np.ndarray,
    timestamp: datetime,
    position: str,
    size_percent: float,
) -> np.ndarray:
    """Draw the same readable timestamp style used by ordinary timelapses."""

    output = frame.copy()
    height, width = output.shape[:2]
    text = timestamp.strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
    font = cv2.FONT_HERSHEY_SIMPLEX
    desired_height = max(12, int(round(height * size_percent / 100.0)))
    unit_height = max(1, cv2.getTextSize("Ag", font, 1.0, 1)[0][1])
    font_scale = desired_height / unit_height
    thickness = max(1, int(round(desired_height / 18)))
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    margin = max(8, int(round(desired_height * 0.55)))
    x = margin if position.endswith("left") else width - text_width - margin
    y = text_height + margin if position.startswith("top") else height - margin
    x = max(0, min(x, max(0, width - text_width)))
    y = max(text_height, min(y, max(text_height, height - baseline)))

    padding = max(3, desired_height // 4)
    left, top = max(0, x - padding), max(0, y - text_height - padding)
    right, bottom = min(width, x + text_width + padding), min(height, y + baseline + padding)
    if right > left and bottom > top:
        roi = output[top:bottom, left:right]
        output[top:bottom, left:right] = cv2.addWeighted(
            roi, 0.45, np.zeros_like(roi), 0.55, 0
        )
    cv2.putText(output, text, (x, y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(output, text, (x, y), font, font_scale, (245, 245, 245), thickness, cv2.LINE_AA)
    return output


def create_meteor_trail_timelapse(
    video_paths: Sequence[str],
    output_path: str,
    *,
    settings: Optional[TrailTimelapseSettings] = None,
    target_duration_seconds: Optional[float] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> bool:
    """Create a meteor-preserving lighten-composite timelapse.

    One output frame represents ``source_seconds_per_output_frame`` seconds of
    source video.  The source frames in that interval are combined with a
    maximum operation, so a meteor remains visible even when it lasts only a
    few source frames.  ``trail_decay`` controls how long real star positions
    remain in the composite.  It is close to one by default so the camera's
    slow stellar motion becomes a continuous trail rather than dots; no
    synthetic line or marker is drawn.
    """

    normalized_paths = discover_video_files(video_paths)
    if not normalized_paths:
        if progress_callback:
            progress_callback("有効な動画ファイルがありません。")
        return False
    options = (settings or TrailTimelapseSettings()).validate()
    if target_duration_seconds is not None:
        try:
            target_duration_seconds = float(target_duration_seconds)
        except (TypeError, ValueError):
            if progress_callback:
                progress_callback("エラー: 動画の長さは数値で指定してください。")
            return False
        if target_duration_seconds <= 0:
            if progress_callback:
                progress_callback("エラー: 動画の長さは0秒より大きくしてください。")
            return False
    total_frames = _count_frames(normalized_paths)
    target_frame_count: Optional[int] = None
    if target_duration_seconds is not None and total_frames > 0:
        target_frame_count = max(
            1, int(round(target_duration_seconds * options.output_fps))
        )
        # A source shorter than the requested duration cannot provide more
        # unique frames without inventing repeated output frames.  Match the
        # ordinary timelapse behavior and emit each available source frame.
        target_frame_count = min(target_frame_count, total_frames)
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
    window_first_time: Optional[datetime] = None
    window_last_time: Optional[datetime] = None
    previous_capture_time: Optional[datetime] = None
    elapsed = 0.0
    processed = 0
    emitted = 0
    current_target_frame: Optional[int] = None
    success = False

    def emit_window() -> None:
        nonlocal window_max, trail, elapsed, emitted, window_first_time, window_last_time
        if window_max is None:
            return
        composite = window_max
        if trail is None:
            trail = composite.copy()
        else:
            faded = np.clip(trail.astype(np.float32) * options.trail_decay, 0, 255)
            trail = np.maximum(composite, faded.astype(np.uint8))
        rendered = trail
        if options.timestamp_enabled and window_first_time is not None and window_last_time is not None:
            timestamp = window_first_time + (window_last_time - window_first_time) / 2
            rendered = _draw_timestamp(
                rendered,
                timestamp,
                options.timestamp_position,
                options.timestamp_size_percent,
            )
        writer.write(rendered)
        emitted += 1
        window_max = None
        window_first_time = None
        window_last_time = None
        elapsed = 0.0

    try:
        for frame, fps, capture_time in _iter_frames(normalized_paths):
            # Do not blend across a missing source interval.  Apart from
            # making the timestamp ambiguous, carrying the old trail across
            # a gap can make a cloud appear to reverse direction at a segment
            # boundary.  Ordinary minute-to-minute RTSP segments differ by
            # one frame period; a larger jump is a real archive gap.
            if previous_capture_time is not None:
                capture_step = (capture_time - previous_capture_time).total_seconds()
                expected_step = 1.0 / fps
                if capture_step > expected_step * 1.5:
                    if target_frame_count is None:
                        emit_window()
                        trail = None
                        elapsed = 0.0
                    else:
                        # Keep the requested output frame count fixed even
                        # when a source archive has a time gap.  Do not carry
                        # the previous trail across that gap.
                        window_max = None
                        window_first_time = None
                        window_last_time = None
                        trail = None
                        elapsed = 0.0
            previous_capture_time = capture_time
            toned = cv2.LUT(_resize_frame(frame, options.output_size), lut)

            if target_frame_count is not None:
                # The duration selector describes the final video duration.
                # Map the complete input range into the requested number of
                # output frames, just like the ordinary timelapse path.  Each
                # mapped interval still uses a lighten composite, preserving
                # the meteor-trail behavior while guaranteeing the duration.
                source_index = processed
                target_frame = min(
                    target_frame_count - 1,
                    source_index * target_frame_count // max(1, total_frames),
                )
                if current_target_frame is None:
                    current_target_frame = target_frame
                elif target_frame != current_target_frame:
                    emit_window()
                    current_target_frame = target_frame

            window_max = toned.copy() if window_max is None else np.maximum(window_max, toned)
            if window_first_time is None:
                window_first_time = capture_time
            window_last_time = capture_time
            elapsed += 1.0 / fps
            processed += 1
            if target_frame_count is None and elapsed >= options.source_seconds_per_output_frame:
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
        if target_duration_seconds is None:
            speedup = options.source_seconds_per_output_frame * options.output_fps
            detail = f"約{speedup:.1f}倍速、{emitted}フレーム"
        else:
            actual_duration = emitted / options.output_fps
            detail = (
                f"指定{target_duration_seconds:g}秒、実測約{actual_duration:g}秒、"
                f"{emitted}フレーム"
            )
        progress_callback(
            f"流星トレイルタイムラプスを保存しました: {output_path} "
            f"（{detail}）"
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
    parser.add_argument(
        "--trail-decay", type=float, default=0.985,
        help="星の残像を次の出力フレームへ残す割合（既定: 0.985）",
    )
    parser.add_argument(
        "--no-timestamp", action="store_true",
        help="右下の実時刻表示を無効にする",
    )
    parser.add_argument(
        "--duration", type=float, default=None,
        help="最終動画の長さ（秒）。省略時は入力時間から決定",
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
        trail_decay=args.trail_decay,
        timestamp_enabled=not args.no_timestamp,
    )
    return 0 if create_meteor_trail_timelapse(
        args.inputs,
        args.output,
        settings=settings,
        target_duration_seconds=args.duration,
        progress_callback=print,
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
