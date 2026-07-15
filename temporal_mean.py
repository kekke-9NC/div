"""Model-free centered three/five-frame temporal averaging."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import tempfile
import time
from typing import Callable, Optional, Sequence

import cv2
import numpy as np

from fixed_pattern import apply_fixed_pattern_correction
import noise_twin
import video_encoding


def normalize_window(value: int) -> int:
    return int(value) if int(value) in (3, 5) else 0


def marker_path(video_path: str | os.PathLike[str]) -> str:
    return str(video_path) + ".preprocessed.json"


def write_processing_marker(video_path: str, window: int, analyzed: bool = False) -> None:
    with open(marker_path(video_path), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "processed": True,
                "method": "temporal_mean",
                "frames": normalize_window(window),
                "analyzed": bool(analyzed),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


def load_processing_marker(video_path: str) -> Optional[dict]:
    try:
        with open(marker_path(video_path), encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) and value.get("processed") is True else None


def mean_frame(frames: Sequence[np.ndarray]) -> np.ndarray:
    if not frames:
        raise ValueError("frames must not be empty")
    accumulator = np.zeros(frames[0].shape, dtype=np.float32)
    for frame in frames:
        cv2.accumulate(frame, accumulator)
    return cv2.convertScaleAbs(accumulator, alpha=1.0 / len(frames))


class TemporalMeanStream:
    def __init__(self, window: int):
        self.window = normalize_window(window)
        if not self.window:
            raise ValueError("temporal mean window must be 3 or 5")
        self.radius = self.window // 2
        self.buffer: list[np.ndarray] = []
        self.started = False

    def push(self, frame: np.ndarray) -> list[np.ndarray]:
        self.buffer.append(frame)
        if len(self.buffer) < self.window:
            return []
        if not self.started:
            first = self.buffer[0]
            padded = [first] * self.radius + self.buffer[: self.window]
            outputs = [
                mean_frame(padded[index : index + self.window])
                for index in range(self.radius + 1)
            ]
        else:
            outputs = [mean_frame(self.buffer[: self.window])]
        self.buffer.pop(0)
        self.started = True
        return outputs

    def flush(self) -> list[np.ndarray]:
        if not self.buffer:
            return []
        outputs: list[np.ndarray] = []
        if not self.started:
            first = self.buffer[0]
            padded = [first] * self.radius + self.buffer
            padded += [self.buffer[-1]] * max(0, self.window - len(padded))
            for index in range(len(self.buffer)):
                window = padded[index : index + self.window]
                if len(window) < self.window:
                    window += [window[-1]] * (self.window - len(window))
                outputs.append(mean_frame(window))
        else:
            tail = self.buffer + [self.buffer[-1]] * self.radius
            for index in range(min(self.radius, len(self.buffer))):
                window = tail[index : index + self.window]
                if len(window) < self.window:
                    window += [window[-1]] * (self.window - len(window))
                outputs.append(mean_frame(window))
        self.buffer.clear()
        return outputs


@dataclass(frozen=True)
class PreparedTemporalVideo:
    video_path: str
    temporary_paths: tuple[str, ...]
    metrics: dict[str, float]

    def cleanup(self) -> None:
        for path in self.temporary_paths:
            try:
                os.remove(path)
            except OSError:
                pass


def prepare_video(
    input_path: str,
    window: int,
    correction: Optional[np.ndarray] = None,
    temp_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    encoding_options: Optional[dict] = None,
) -> PreparedTemporalVideo:
    window = normalize_window(window)
    if not window:
        raise ValueError("時間平均は3または5フレームを指定してください。")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"動画を開けません: {input_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    directory = temp_dir or tempfile.gettempdir()
    os.makedirs(directory, exist_ok=True)
    output_path = os.path.join(
        directory, f"temporal_mean_{window}_{time.time_ns()}_{os.getpid()}.mp4"
    )
    resolved_encoding = video_encoding.resolve_for_source(encoding_options, input_path)
    writer = video_encoding.FFmpegFrameWriter(
        output_path, fps, (width, height), resolved_encoding
    )
    if not writer.isOpened():
        cap.release()
        raise IOError("時間平均動画の書き込みを開始できません。")
    async_writer = noise_twin.AsyncVideoPairWriter(writer)
    processor = TemporalMeanStream(window)
    count = 0

    def write_frame(frame: np.ndarray) -> None:
        nonlocal count
        result = noise_twin.NoiseTwinResult(frame, frame, 0.0, 0.0, 0.0, 1.0)
        async_writer.submit(result)
        count += 1
        if progress_callback:
            progress_callback(count, total)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            corrected = apply_fixed_pattern_correction(frame, correction)
            for averaged in processor.push(corrected):
                write_frame(averaged)
        for averaged in processor.flush():
            write_frame(averaged)
    except Exception:
        try:
            os.remove(output_path)
        except OSError:
            pass
        raise
    finally:
        cap.release()
        async_writer.close()
    return PreparedTemporalVideo(
        output_path,
        (output_path,),
        {
            "window": float(window),
            "theoretical_noise_reduction_percent": (1.0 - 1.0 / np.sqrt(window)) * 100.0,
            "encoding_bitrate_mbps": float(resolved_encoding.bitrate_mbps),
        },
    )
