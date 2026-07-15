"""Configurable FFmpeg frame writer for processed continuous videos."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Optional

import numpy as np


class VideoEncodingError(RuntimeError):
    pass


@dataclass(frozen=True)
class EncodingSettings:
    codec: str = "hevc"
    quality: str = "source"
    bitrate_mbps: int = 80

    @classmethod
    def from_value(cls, value: Any) -> "EncodingSettings":
        data: Mapping[str, Any] = value if isinstance(value, Mapping) else {}
        codec = str(data.get("codec", "hevc")).lower()
        if codec not in ("hevc", "h264", "mpeg4"):
            codec = "hevc"
        quality = str(data.get("quality", "source")).lower()
        if quality not in ("source", "maximum", "high", "standard", "compact", "custom", "lossless"):
            quality = "source"
        presets = {"maximum": 80, "high": 60, "standard": 40, "compact": 20}
        try:
            requested = int(float(data.get("bitrate_mbps", presets.get(quality, 80))))
        except (TypeError, ValueError):
            requested = presets.get(quality, 80)
        return cls(codec=codec, quality=quality, bitrate_mbps=max(5, min(200, requested)))

    @property
    def is_lossless(self) -> bool:
        return self.quality == "lossless"

    @property
    def label(self) -> str:
        codec_label = {"hevc": "H.265/HEVC", "h264": "H.264/AVC", "mpeg4": "MPEG-4"}[self.codec]
        if self.quality == "source":
            return f"{codec_label} 入力品質基準"
        return f"{codec_label} 可逆" if self.is_lossless else f"{codec_label} {self.bitrate_mbps} Mbps"


def source_bitrate_mbps(path: str) -> Optional[float]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            result = subprocess.run(
                [
                    ffprobe, "-v", "error", "-show_entries", "format=bit_rate,duration,size",
                    "-of", "json", path,
                ],
                capture_output=True, text=True, timeout=15, check=False,
            )
            data = json.loads(result.stdout or "{}").get("format", {})
            bit_rate = float(data.get("bit_rate") or 0.0)
            if bit_rate > 0:
                return bit_rate / 1_000_000.0
            duration = float(data.get("duration") or 0.0)
            size = float(data.get("size") or 0.0)
            if duration > 0 and size > 0:
                return size * 8.0 / duration / 1_000_000.0
        except (OSError, ValueError, TypeError, json.JSONDecodeError, subprocess.TimeoutExpired):
            pass
    return None


def resolve_for_source(
    settings: Optional[EncodingSettings | Mapping[str, Any]], input_path: str
) -> EncodingSettings:
    normalized = settings if isinstance(settings, EncodingSettings) else EncodingSettings.from_value(settings)
    if normalized.quality != "source":
        return normalized
    measured = source_bitrate_mbps(input_path)
    if measured is None:
        measured = 40.0
    # Re-encoding cannot add detail that is absent in the camera stream. Match
    # its measured rate; add a small allowance for H.264 generation loss and a
    # larger one for legacy MPEG-4 Part 2.
    multiplier = {"hevc": 1.0, "h264": 1.05, "mpeg4": 1.5}[normalized.codec]
    target = int(round(measured * multiplier / 5.0) * 5)
    return EncodingSettings(
        codec=normalized.codec,
        quality="custom",
        bitrate_mbps=max(10, min(120, target)),
    )


class FFmpegFrameWriter:
    """Small cv2.VideoWriter-compatible wrapper around an FFmpeg raw pipe."""

    def __init__(
        self,
        output_path: str,
        fps: float,
        frame_size: tuple[int, int],
        settings: Optional[EncodingSettings | Mapping[str, Any]] = None,
    ):
        self.output_path = str(output_path)
        self.fps = max(0.1, float(fps))
        self.width, self.height = (int(frame_size[0]), int(frame_size[1]))
        self.settings = (
            settings if isinstance(settings, EncodingSettings)
            else EncodingSettings.from_value(settings)
        )
        self.process: Optional[subprocess.Popen] = None
        self._error: Optional[str] = None
        self._open()

    def _encoder_args(self) -> list[str]:
        settings = self.settings
        if settings.is_lossless:
            if settings.codec == "h264":
                return ["-c:v", "libx264", "-preset", "fast", "-crf", "0"]
            # MPEG-4 Part 2 has no suitable mathematically lossless MP4 mode.
            return [
                "-c:v", "libx265", "-preset", "fast",
                "-x265-params", "lossless=1", "-tag:v", "hvc1",
            ]
        bitrate = f"{settings.bitrate_mbps}M"
        maxrate = f"{max(settings.bitrate_mbps + 5, round(settings.bitrate_mbps * 1.2))}M"
        bufsize = f"{settings.bitrate_mbps * 2}M"
        if settings.codec == "hevc":
            return [
                "-c:v", "hevc_videotoolbox", "-b:v", bitrate,
                "-maxrate", maxrate, "-bufsize", bufsize, "-tag:v", "hvc1",
            ]
        if settings.codec == "h264":
            return [
                "-c:v", "h264_videotoolbox", "-b:v", bitrate,
                "-maxrate", maxrate, "-bufsize", bufsize,
            ]
        return ["-c:v", "mpeg4", "-b:v", bitrate, "-q:v", "1"]

    def _open(self) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            self._error = "FFmpegが見つかりません。"
            return
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        command = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s:v", f"{self.width}x{self.height}", "-r", f"{self.fps:.8f}",
            "-i", "pipe:0", "-an",
            *self._encoder_args(),
            "-pix_fmt", "yuv444p" if self.settings.is_lossless else "yuv420p",
            "-movflags", "+faststart", self.output_path,
        ]
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            self._error = str(exc)

    def isOpened(self) -> bool:
        return self.process is not None and self.process.stdin is not None and self.process.poll() is None

    def write(self, frame: np.ndarray) -> None:
        if not self.isOpened():
            raise VideoEncodingError(self._error or "FFmpegエンコーダが終了しています。")
        if frame.shape[:2] != (self.height, self.width):
            raise VideoEncodingError(
                f"フレーム解像度 {frame.shape[1]}x{frame.shape[0]} が保存設定と一致しません。"
            )
        try:
            self.process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        except (BrokenPipeError, OSError) as exc:
            self._error = str(exc)
            raise VideoEncodingError(f"FFmpegへのフレーム送信に失敗しました: {exc}") from exc

    def release(self) -> None:
        process = self.process
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        try:
            return_code = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            return_code = process.wait()
        stderr = b""
        if process.stderr is not None:
            stderr = process.stderr.read()
            process.stderr.close()
        self.process = None
        if return_code != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()[-2000:]
            try:
                os.remove(self.output_path)
            except OSError:
                pass
            raise VideoEncodingError(f"FFmpeg保存に失敗しました: {detail or return_code}")


def estimated_megabytes_per_minute(settings: EncodingSettings | Mapping[str, Any]) -> Optional[float]:
    normalized = settings if isinstance(settings, EncodingSettings) else EncodingSettings.from_value(settings)
    if normalized.is_lossless or normalized.quality == "source":
        return None
    return normalized.bitrate_mbps * 60.0 / 8.0
