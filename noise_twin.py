"""Camera-specific, self-supervised video noise separation.

The centre frame is deliberately excluded from the background predictor.  A
separate gate can therefore only *select* positive evidence present in the
observed frame; it cannot invent a meteor.  Torch is imported lazily so the
legacy application remains usable when Camera Digital Twin is disabled.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import tempfile
import threading
import time
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence

import cv2
import numpy as np

from fixed_pattern import apply_fixed_pattern_correction


MODEL_FORMAT_VERSION = 3
DEFAULT_TILE_SIZE = 512
DEFAULT_TILE_OVERLAP = 32
TEMPORAL_RADIUS = 3
TEMPORAL_WINDOW = TEMPORAL_RADIUS * 2 + 1


class NoiseTwinError(RuntimeError):
    pass


@dataclass(frozen=True)
class NoiseTwinValidation:
    injection_count: int = 0
    missed_fraction: float = 1.0
    flux_retention: float = 0.0
    peak_retention: float = 0.0
    trajectory_retention: float = 0.0
    false_positive_reduction: float = 0.0
    realtime_fps: float = 0.0
    realtime_test_seconds: float = 0.0
    dropped_frames: int = 0
    validated: bool = False

    @classmethod
    def from_dict(cls, value: Any) -> "NoiseTwinValidation":
        data = value if isinstance(value, dict) else {}
        allowed = cls.__dataclass_fields__
        return cls(**{key: data[key] for key in allowed if key in data})


@dataclass(frozen=True)
class NoiseTwinMetadata:
    model_id: str
    created_at: str
    width: int
    height: int
    fps: float
    source_id: str
    fixed_pattern_sha256: str = ""
    architecture: str = "blind-temporal-unet32-quarter-physical-gate"
    inference_scale: float = 0.25
    tile_size: int = DEFAULT_TILE_SIZE
    tile_overlap: int = DEFAULT_TILE_OVERLAP
    format_version: int = MODEL_FORMAT_VERSION
    validation: NoiseTwinValidation = field(default_factory=NoiseTwinValidation)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NoiseTwinMetadata":
        payload = dict(data)
        payload["validation"] = NoiseTwinValidation.from_dict(payload.get("validation"))
        allowed = cls.__dataclass_fields__
        return cls(**{key: payload[key] for key in allowed if key in payload})


@dataclass(frozen=True)
class NoiseTwinOptions:
    enabled: bool = False
    model_path: str = ""
    already_processed: bool = False
    require_validated: bool = True

    @classmethod
    def from_value(cls, value: Any) -> "NoiseTwinOptions":
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            return cls()
        return cls(
            enabled=bool(value.get("enabled", False)),
            model_path=str(value.get("model_path", "") or ""),
            already_processed=bool(value.get("already_processed", False)),
            require_validated=bool(value.get("require_validated", True)),
        )


@dataclass
class NoiseTwinResult:
    frame: np.ndarray
    innovation: np.ndarray
    noise_sigma: float
    innovation_max: float
    protected_fraction: float
    flux_retention: float


@dataclass(frozen=True)
class PreparedVideo:
    video_path: str
    innovation_path: str
    temporary_paths: tuple[str, ...]
    metrics: dict[str, float]

    def cleanup(self) -> None:
        for path in self.temporary_paths:
            try:
                os.remove(path)
            except OSError:
                pass


class AsyncVideoPairWriter:
    """Bounded writer stage so codec work overlaps MPS inference."""

    def __init__(self, video_writer, innovation_writer=None, queue_size: int = 12):
        self.video_writer = video_writer
        self.innovation_writer = innovation_writer
        self.queue: queue.Queue = queue.Queue(maxsize=max(2, int(queue_size)))
        self.errors: list[BaseException] = []
        self.closed = False
        self.thread = threading.Thread(target=self._run, name="NoiseTwinWriter", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while True:
            item = self.queue.get()
            try:
                if item is None:
                    return
                if not self.errors:
                    self.video_writer.write(item.frame)
                    if self.innovation_writer is not None:
                        self.innovation_writer.write(item.innovation)
            except BaseException as exc:
                self.errors.append(exc)
            finally:
                self.queue.task_done()

    def submit(self, result: NoiseTwinResult) -> None:
        if self.closed:
            raise NoiseTwinError("NoiseTwin書き込みパイプラインは終了しています。")
        if self.errors:
            raise NoiseTwinError(f"NoiseTwin動画の書き込みに失敗しました: {self.errors[0]}")
        self.queue.put(result)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.queue.put(None)
        self.queue.join()
        self.thread.join(timeout=10)
        self.video_writer.release()
        if self.innovation_writer is not None:
            self.innovation_writer.release()
        if self.errors:
            raise NoiseTwinError(f"NoiseTwin動画の書き込みに失敗しました: {self.errors[0]}")


def metadata_path(model_path: str | os.PathLike[str]) -> Path:
    path = Path(model_path)
    return path.with_suffix(path.suffix + ".json")


def processing_marker_path(video_path: str | os.PathLike[str]) -> Path:
    path = Path(video_path)
    return path.with_suffix(path.suffix + ".noisetwin.json")


def write_processing_marker(
    video_path: str | os.PathLike[str], metadata: "NoiseTwinMetadata"
) -> None:
    marker = {
        "processed": True,
        "model_id": metadata.model_id,
        "format_version": metadata.format_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path = processing_marker_path(video_path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def load_processing_marker(video_path: str | os.PathLike[str]) -> Optional[dict[str, Any]]:
    try:
        value = json.loads(processing_marker_path(video_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) and value.get("processed") is True else None


def load_metadata(model_path: str | os.PathLike[str]) -> NoiseTwinMetadata:
    path = metadata_path(model_path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NoiseTwinError(f"NoiseTwinメタデータを読み込めません: {path}: {exc}") from exc
    metadata = NoiseTwinMetadata.from_dict(data)
    if metadata.format_version != MODEL_FORMAT_VERSION:
        raise NoiseTwinError(
            f"未対応のNoiseTwin形式です: {metadata.format_version}"
        )
    return metadata


def save_metadata(model_path: str | os.PathLike[str], metadata: NoiseTwinMetadata) -> None:
    payload = asdict(metadata)
    path = metadata_path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def correction_sha256(correction: Optional[np.ndarray]) -> str:
    if correction is None:
        return ""
    digest = hashlib.sha256()
    digest.update(str(correction.dtype).encode("ascii"))
    digest.update(np.asarray(correction.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(correction).tobytes())
    return digest.hexdigest()


def camera_source_id(source: str) -> str:
    value = str(source).strip()
    if value.lower().startswith(("rtsp://", "rtsps://")):
        # Do not leak credentials into metadata or filenames.
        try:
            from urllib.parse import urlsplit, urlunsplit

            parts = urlsplit(value)
            hostname = parts.hostname or "camera"
            port = f":{parts.port}" if parts.port else ""
            value = urlunsplit((parts.scheme, hostname + port, parts.path, "", ""))
        except Exception:
            value = "rtsp-camera"
    else:
        value = str(Path(value).resolve().parent)
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:16]


def _torch_modules():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as exc:
        raise NoiseTwinError(
            "Camera Digital TwinにはPyTorchが必要です。アプリ用環境を起動してください。"
        ) from exc
    return torch, nn, functional


def select_device():
    torch, _nn, _functional = _torch_modules()
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    raise NoiseTwinError("NoiseTwinは現在MPSまたはCUDA環境でのみ有効です。")


def build_model():
    """Build the blind-temporal background predictor and residual gate."""
    torch, nn, functional = _torch_modules()

    class ConvBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            groups = max(1, min(8, out_channels // 4))
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.GroupNorm(groups, out_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.GroupNorm(groups, out_channels),
                nn.SiLU(inplace=True),
            )

        def forward(self, value):
            return self.layers(value)

    class BackgroundUNet(nn.Module):
        def __init__(self):
            super().__init__()
            channels = (32, 64, 128, 256)
            self.enc1 = ConvBlock(18, channels[0])
            self.enc2 = ConvBlock(channels[0], channels[1])
            self.enc3 = ConvBlock(channels[1], channels[2])
            self.enc4 = ConvBlock(channels[2], channels[3])
            self.pool = nn.AvgPool2d(2)
            self.mid = ConvBlock(channels[3], 512)
            self.up4 = nn.ConvTranspose2d(512, channels[3], 2, stride=2)
            self.dec4 = ConvBlock(channels[3] * 2, channels[3])
            self.up3 = nn.ConvTranspose2d(channels[3], channels[2], 2, stride=2)
            self.dec3 = ConvBlock(channels[2] * 2, channels[2])
            self.up2 = nn.ConvTranspose2d(channels[2], channels[1], 2, stride=2)
            self.dec2 = ConvBlock(channels[1] * 2, channels[1])
            self.up1 = nn.ConvTranspose2d(channels[1], channels[0], 2, stride=2)
            self.dec1 = ConvBlock(channels[0] * 2, channels[0])
            self.output = nn.Conv2d(channels[0], 3, 1)

        def forward(self, value):
            e1 = self.enc1(value)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            e4 = self.enc4(self.pool(e3))
            middle = self.mid(self.pool(e4))
            d4 = self.dec4(torch.cat((self.up4(middle), e4), dim=1))
            d3 = self.dec3(torch.cat((self.up3(d4), e3), dim=1))
            d2 = self.dec2(torch.cat((self.up2(d3), e2), dim=1))
            d1 = self.dec1(torch.cat((self.up1(d2), e1), dim=1))
            return torch.sigmoid(self.output(d1))

    class SignalGate(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(7, 32, 5, padding=2), nn.SiLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(inplace=True),
                nn.Conv2d(32, 16, 3, padding=1), nn.SiLU(inplace=True),
                nn.Conv2d(16, 1, 1),
            )

        def forward(self, centre, background, innovation):
            return torch.sigmoid(self.layers(torch.cat((centre, background, innovation), dim=1)))

    class NoiseTwinNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.background = BackgroundUNet()
            self.gate = SignalGate()

        def predict_background(self, neighbors):
            return self.background(neighbors)

        def forward(self, neighbors, centre):
            background = self.background(neighbors)
            residual = torch.clamp(centre - background, min=0.0)
            gray = residual.mean(dim=1, keepdim=True)
            scale = torch.median(gray.flatten(2), dim=2).values[:, :, None, None]
            innovation = torch.clamp(gray / torch.clamp(scale * 1.4826, min=1.0 / 255.0), 0, 16) / 16
            gate = self.gate(centre, background, innovation)
            clean = torch.clamp(background + gate * residual, 0.0, 1.0)
            # A strict upper bound prevents the network from creating positive light.
            clean = torch.minimum(clean, torch.maximum(centre, background))
            return clean, background, gate, innovation

    return NoiseTwinNet()


def _load_torch_model(model_path: str, require_validated: bool = True):
    torch, _nn, _functional = _torch_modules()
    metadata = load_metadata(model_path)
    if require_validated and not metadata.validation.validated:
        raise NoiseTwinError("検証基準を満たしていないNoiseTwinモデルです。")
    device = select_device()
    network = build_model().to(device)
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)
    network.load_state_dict(state)
    network.eval()
    return network, metadata, device


def _tile_positions(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    step = max(1, tile - overlap * 2)
    positions = list(range(0, max(1, length - tile + 1), step))
    last = length - tile
    if positions[-1] != last:
        positions.append(last)
    return positions


def _blend_window(height: int, width: int, overlap: int) -> np.ndarray:
    y = np.ones(height, dtype=np.float32)
    x = np.ones(width, dtype=np.float32)
    edge_y = min(overlap, height // 2)
    edge_x = min(overlap, width // 2)
    if edge_y:
        ramp = np.linspace(0.05, 1.0, edge_y, dtype=np.float32)
        y[:edge_y], y[-edge_y:] = ramp, ramp[::-1]
    if edge_x:
        ramp = np.linspace(0.05, 1.0, edge_x, dtype=np.float32)
        x[:edge_x], x[-edge_x:] = ramp, ramp[::-1]
    return y[:, None] * x[None, :]


def _physical_trajectory_gate(
    corrected_frames: Sequence[np.ndarray],
    background_u8: np.ndarray,
    evidence: np.ndarray,
    noise_sigma: float,
) -> np.ndarray:
    """Protect only observed centre-frame light supported by a line/trajectory.

    The temporal maximum is evaluated at half resolution for speed. A candidate
    is restored only where that trajectory overlaps positive evidence in the
    original centre frame, so this gate cannot introduce a synthetic pixel.
    """
    height, width = evidence.shape
    small_size = (max(16, width // 2), max(16, height // 2))
    background_small = cv2.resize(background_u8, small_size, interpolation=cv2.INTER_AREA)
    background_gray = cv2.cvtColor(background_small, cv2.COLOR_BGR2GRAY)
    temporal_max = np.zeros(background_gray.shape, dtype=np.uint8)
    threshold = max(2, int(round(noise_sigma * 2.25)))
    for frame in corrected_frames:
        small = cv2.resize(frame, small_size, interpolation=cv2.INTER_AREA)
        residual = cv2.subtract(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), background_gray)
        temporal_max = cv2.max(temporal_max, residual)
    binary = np.where(temporal_max >= threshold, 255, 0).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    lines = cv2.HoughLinesP(
        binary,
        1,
        np.pi / 180.0,
        threshold=10,
        minLineLength=max(6, min(small_size) // 120),
        maxLineGap=max(3, min(small_size) // 180),
    )
    if lines is None:
        return np.zeros((height, width), dtype=np.float32)
    line_mask = np.zeros_like(binary)
    for line in lines[:, 0]:
        cv2.line(line_mask, tuple(line[:2]), tuple(line[2:]), 255, 3, cv2.LINE_AA)
    line_mask = cv2.resize(line_mask, (width, height), interpolation=cv2.INTER_LINEAR)
    observed = np.where(evidence >= 2.5, 255, 0).astype(np.uint8)
    observed = cv2.dilate(observed, np.ones((3, 3), np.uint8))
    protected = cv2.bitwise_and(line_mask, observed)
    return cv2.GaussianBlur(protected, (0, 0), 1.0).astype(np.float32) / 255.0


class NoiseTwinEngine:
    def __init__(
        self,
        model_path: str,
        correction: Optional[np.ndarray] = None,
        require_validated: bool = True,
    ):
        self.network, self.metadata, self.device = _load_torch_model(
            model_path, require_validated=require_validated
        )
        actual_correction_hash = correction_sha256(correction)
        if self.metadata.fixed_pattern_sha256 != actual_correction_hash:
            raise NoiseTwinError(
                "学習時と現在の固定パターン補正が一致しません。"
                "同じ補正設定でNoiseTwinを使用してください。"
            )
        self.correction = correction
        self._torch, _nn, self._functional = _torch_modules()
        self.inference_dtype = (
            self._torch.float16 if self.device.type in ("mps", "cuda") else self._torch.float32
        )
        self.network.to(dtype=self.inference_dtype)

    def _correct(self, frame: np.ndarray) -> np.ndarray:
        return apply_fixed_pattern_correction(frame, self.correction)

    def infer(self, frames: Sequence[np.ndarray]) -> NoiseTwinResult:
        if len(frames) != TEMPORAL_WINDOW:
            raise ValueError(f"NoiseTwin requires {TEMPORAL_WINDOW} frames")
        corrected = [self._correct(frame) for frame in frames]
        centre = corrected[TEMPORAL_RADIUS]
        height, width = centre.shape[:2]
        neighbors = corrected[:TEMPORAL_RADIUS] + corrected[TEMPORAL_RADIUS + 1 :]
        scale = float(np.clip(self.metadata.inference_scale, 0.125, 1.0))
        scaled_width = max(16, int(round(width * scale)))
        scaled_height = max(16, int(round(height * scale)))
        scaled = [
            cv2.resize(frame, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
            for frame in corrected
        ]
        scaled_centre = scaled[TEMPORAL_RADIUS]
        scaled_neighbors = scaled[:TEMPORAL_RADIUS] + scaled[TEMPORAL_RADIUS + 1 :]
        background_output = np.zeros((scaled_height, scaled_width, 3), dtype=np.float32)
        gate_output = np.zeros((scaled_height, scaled_width), dtype=np.float32)
        weights = np.zeros((scaled_height, scaled_width), dtype=np.float32)
        tile = int(self.metadata.tile_size or DEFAULT_TILE_SIZE)
        overlap = int(self.metadata.tile_overlap or DEFAULT_TILE_OVERLAP)

        torch = self._torch
        with torch.inference_mode():
            for y in _tile_positions(scaled_height, tile, overlap):
                for x in _tile_positions(scaled_width, tile, overlap):
                    y2, x2 = min(scaled_height, y + tile), min(scaled_width, x + tile)
                    h, w = y2 - y, x2 - x
                    pad_h = int(math.ceil(h / 16) * 16)
                    pad_w = int(math.ceil(w / 16) * 16)
                    neighbor_array = np.concatenate(
                        [item[y:y2, x:x2, ::-1].transpose(2, 0, 1) for item in scaled_neighbors], axis=0
                    ).astype(np.float32) / 255.0
                    centre_array = scaled_centre[y:y2, x:x2, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
                    neighbor_tensor = torch.from_numpy(neighbor_array[None]).to(
                        self.device, dtype=self.inference_dtype
                    )
                    centre_tensor = torch.from_numpy(centre_array[None]).to(
                        self.device, dtype=self.inference_dtype
                    )
                    if pad_h != h or pad_w != w:
                        neighbor_tensor = self._functional.pad(neighbor_tensor, (0, pad_w - w, 0, pad_h - h), mode="replicate")
                        centre_tensor = self._functional.pad(centre_tensor, (0, pad_w - w, 0, pad_h - h), mode="replicate")
                    _clean, background, gate, _innovation = self.network(neighbor_tensor, centre_tensor)
                    background_np = background[0, :, :h, :w].float().cpu().numpy().transpose(1, 2, 0)[..., ::-1]
                    gate_np = gate[0, 0, :h, :w].float().cpu().numpy()
                    blend = _blend_window(h, w, overlap)
                    background_output[y:y2, x:x2] += background_np * blend[..., None]
                    gate_output[y:y2, x:x2] += gate_np * blend
                    weights[y:y2, x:x2] += blend

        background_output /= np.maximum(weights[..., None], 1e-6)
        gate_output /= np.maximum(weights, 1e-6)
        background = cv2.resize(background_output, (width, height), interpolation=cv2.INTER_LINEAR)
        gate = cv2.resize(gate_output, (width, height), interpolation=cv2.INTER_LINEAR)
        background_u8 = np.clip(np.rint(background * 255.0), 0, 255).astype(np.uint8)
        gray_centre = cv2.cvtColor(centre, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_background = cv2.cvtColor(background_u8, cv2.COLOR_BGR2GRAY).astype(np.float32)
        signed_residual = gray_centre - gray_background
        sigma_sample = signed_residual[::4, ::4]
        residual_centre = float(np.median(sigma_sample))
        noise_sigma_normalized = float(
            np.median(np.abs(sigma_sample - residual_centre)) / 0.67448975
        )
        noise_sigma_normalized = max(noise_sigma_normalized, 1.0)
        evidence = np.maximum(signed_residual, 0.0) / noise_sigma_normalized
        physical_gate = _physical_trajectory_gate(
            corrected, background_u8, evidence, noise_sigma_normalized
        )
        gate = np.maximum(gate, physical_gate)
        gate_u8 = np.clip(np.rint(gate * 255.0), 0, 255).astype(np.uint8)
        positive_residual_u8 = cv2.subtract(centre, background_u8)
        gated_residual = cv2.multiply(
            positive_residual_u8,
            cv2.cvtColor(gate_u8, cv2.COLOR_GRAY2BGR),
            scale=1.0 / 255.0,
        )
        # OpenCV's saturated add plus subtract semantics give the same strict
        # bound as B + gate*max(I-B, 0): no positive value can exceed I.
        clean_u8 = cv2.add(background_u8, gated_residual)
        innovation_u8 = np.clip(np.rint(evidence * 16.0), 0, 255).astype(np.uint8)
        protected = evidence > 3.0
        source_positive = np.maximum(signed_residual, 0.0)[protected].sum()
        output_positive = (np.maximum(signed_residual, 0.0) * gate)[protected].sum()
        flux_retention = (
            float(min(1.0, output_positive / max(source_positive, 1e-6)))
            if np.any(protected)
            else 1.0
        )
        return NoiseTwinResult(
            frame=clean_u8,
            innovation=cv2.cvtColor(innovation_u8, cv2.COLOR_GRAY2BGR),
            noise_sigma=noise_sigma_normalized,
            innovation_max=float(evidence.max(initial=0.0)),
            protected_fraction=float(np.mean(protected)),
            flux_retention=flux_retention,
        )


class NoiseTwinStreamProcessor:
    """Seven-frame centred streaming inference with a deterministic flush."""
    def __init__(self, engine: NoiseTwinEngine):
        self.engine = engine
        self.buffer: list[np.ndarray] = []
        self.started = False

    def push(self, frame: np.ndarray) -> list[NoiseTwinResult]:
        self.buffer.append(frame)
        if len(self.buffer) < TEMPORAL_WINDOW:
            return []
        if not self.started:
            first = self.buffer[0]
            padded = [first] * TEMPORAL_RADIUS + self.buffer[:TEMPORAL_WINDOW]
            results = [
                self.engine.infer(padded[index : index + TEMPORAL_WINDOW])
                for index in range(TEMPORAL_RADIUS + 1)
            ]
        else:
            results = [self.engine.infer(self.buffer[:TEMPORAL_WINDOW])]
        self.buffer.pop(0)
        self.started = True
        return results

    def flush(self) -> list[NoiseTwinResult]:
        if not self.buffer:
            return []
        results: list[NoiseTwinResult] = []
        if not self.started:
            first = self.buffer[0]
            padded = [first] * TEMPORAL_RADIUS + self.buffer
            padded += [self.buffer[-1]] * max(0, TEMPORAL_WINDOW - len(padded))
            for index in range(len(self.buffer)):
                window = padded[index : index + TEMPORAL_WINDOW]
                if len(window) < TEMPORAL_WINDOW:
                    window += [window[-1]] * (TEMPORAL_WINDOW - len(window))
                results.append(self.engine.infer(window))
        else:
            # The remaining buffer contains the last six source frames.  Only
            # the final three centres have not yet been emitted.
            tail = self.buffer + [self.buffer[-1]] * TEMPORAL_RADIUS
            for index in range(min(TEMPORAL_RADIUS, len(self.buffer))):
                window = tail[index : index + TEMPORAL_WINDOW]
                if len(window) < TEMPORAL_WINDOW:
                    window += [window[-1]] * (TEMPORAL_WINDOW - len(window))
                results.append(self.engine.infer(window))
        self.buffer.clear()
        return results


def validate_model_for_video(
    model_path: str,
    width: int,
    height: int,
    fps: float,
    require_realtime: bool = False,
) -> NoiseTwinMetadata:
    metadata = load_metadata(model_path)
    if not metadata.validation.validated:
        raise NoiseTwinError("このNoiseTwinモデルは採用基準を満たしていません。")
    if (metadata.width, metadata.height) != (int(width), int(height)):
        raise NoiseTwinError(
            f"モデル解像度 {metadata.width}x{metadata.height} と入力 {width}x{height} が一致しません。"
        )
    if require_realtime and metadata.validation.realtime_fps + 1e-6 < float(fps):
        raise NoiseTwinError(
            f"検証速度 {metadata.validation.realtime_fps:.1f} fps が入力 {fps:.1f} fps 未満です。"
        )
    if require_realtime and metadata.validation.realtime_test_seconds < 1800.0:
        raise NoiseTwinError(
            "RTSP本番利用には30分間の連続速度検証が必要です。"
        )
    if require_realtime and metadata.validation.dropped_frames != 0:
        raise NoiseTwinError(
            f"連続速度検証で {metadata.validation.dropped_frames} フレーム欠落したため利用できません。"
        )
    return metadata


def prepare_video(
    input_path: str,
    model_path: str,
    correction: Optional[np.ndarray] = None,
    temp_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    require_validated: bool = True,
) -> PreparedVideo:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise NoiseTwinError(f"動画を開けません: {input_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if require_validated:
        validate_model_for_video(model_path, width, height, fps, require_realtime=False)
    directory = temp_dir or tempfile.gettempdir()
    os.makedirs(directory, exist_ok=True)
    token = f"{time.time_ns()}_{os.getpid()}"
    video_path = os.path.join(directory, f"noise_twin_{token}.mp4")
    innovation_path = os.path.join(directory, f"noise_twin_{token}_innovation.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    evidence_writer = cv2.VideoWriter(innovation_path, fourcc, fps, (width, height))
    if not writer.isOpened() or not evidence_writer.isOpened():
        cap.release()
        writer.release()
        evidence_writer.release()
        raise NoiseTwinError("NoiseTwin一時動画の書き込みを開始できません。")
    engine = NoiseTwinEngine(model_path, correction, require_validated=require_validated)
    processor = NoiseTwinStreamProcessor(engine)
    async_writer = AsyncVideoPairWriter(writer, evidence_writer)
    metrics = {"noise_sigma": 0.0, "innovation_max": 0.0, "flux_retention": 0.0}
    count = 0

    def write_result(result: NoiseTwinResult) -> None:
        nonlocal count
        async_writer.submit(result)
        count += 1
        metrics["noise_sigma"] += result.noise_sigma
        metrics["innovation_max"] = max(metrics["innovation_max"], result.innovation_max)
        metrics["flux_retention"] += result.flux_retention
        if progress_callback:
            progress_callback(count, total)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            for result in processor.push(frame):
                write_result(result)
        for result in processor.flush():
            write_result(result)
    except Exception:
        for path in (video_path, innovation_path):
            try:
                os.remove(path)
            except OSError:
                pass
        raise
    finally:
        cap.release()
        async_writer.close()
    if count:
        metrics["noise_sigma"] /= count
        metrics["flux_retention"] /= count
    return PreparedVideo(video_path, innovation_path, (video_path, innovation_path), metrics)


def new_metadata(
    model_path: str,
    source: str,
    width: int,
    height: int,
    fps: float,
    correction: Optional[np.ndarray],
    validation: NoiseTwinValidation,
) -> NoiseTwinMetadata:
    identifier = hashlib.sha256(
        f"{source}|{width}|{height}|{fps}|{time.time_ns()}".encode("utf-8")
    ).hexdigest()[:16]
    return NoiseTwinMetadata(
        model_id=identifier,
        created_at=datetime.now(timezone.utc).isoformat(),
        width=int(width),
        height=int(height),
        fps=float(fps),
        source_id=camera_source_id(source),
        fixed_pattern_sha256=correction_sha256(correction),
        validation=validation,
    )
