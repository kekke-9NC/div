"""Self-supervised training and synthetic-meteor validation for NoiseTwin."""

from __future__ import annotations

import math
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Callable, Optional, Sequence

import cv2
import numpy as np

import noise_twin


Progress = Optional[Callable[[str, int, int], None]]


def _report(callback: Progress, phase: str, done: int, total: int) -> None:
    if callback:
        callback(phase, int(done), int(total))


def _video_properties(path: str) -> tuple[int, int, float, int]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise noise_twin.NoiseTwinError(f"学習動画を開けません: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if width <= 0 or height <= 0 or total < noise_twin.TEMPORAL_WINDOW:
        raise noise_twin.NoiseTwinError(f"学習に使用できない動画です: {path}")
    return width, height, fps, total


def capture_rtsp_training_video(
    url: str,
    output_path: str,
    duration_seconds: int = 600,
    progress_callback: Progress = None,
) -> str:
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise noise_twin.NoiseTwinError("RTSP学習ストリームへ接続できません。")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise noise_twin.NoiseTwinError("RTSP学習動画を保存できません。")
    total = max(1, int(round(fps * duration_seconds)))
    written = 0
    failures = 0
    try:
        while written < total:
            ok, frame = cap.read()
            if not ok:
                failures += 1
                if failures >= 50:
                    raise noise_twin.NoiseTwinError("RTSP学習収録中に接続が切れました。")
                continue
            failures = 0
            writer.write(frame)
            written += 1
            if written % max(1, int(fps)) == 0:
                _report(progress_callback, "capture", written, total)
    finally:
        writer.release()
        cap.release()
    return output_path


class VideoPatchSampler:
    def __init__(
        self,
        paths: Sequence[str],
        patch_size: int = 256,
        correction: Optional[np.ndarray] = None,
        inference_scale: float = 0.25,
        seed: int = 1234,
    ):
        if not paths:
            raise ValueError("paths must not be empty")
        self.paths = [str(path) for path in paths]
        self.properties = [_video_properties(path) for path in self.paths]
        first = self.properties[0]
        for path, props in zip(self.paths, self.properties):
            if props[:2] != first[:2]:
                raise noise_twin.NoiseTwinError(
                    f"学習動画の解像度が一致しません: {path}"
                )
        self.inference_scale = float(np.clip(inference_scale, 0.125, 1.0))
        self.scaled_width = max(16, int(round(first[0] * self.inference_scale)))
        self.scaled_height = max(16, int(round(first[1] * self.inference_scale)))
        # Four pooling stages require a multiple of 16. Training in the same
        # resolution domain as inference is both faster and avoids a train/run
        # spatial-scale mismatch.
        maximum_patch = min(int(patch_size), self.scaled_width, self.scaled_height)
        self.patch_size = max(16, (maximum_patch // 16) * 16)
        self.correction = correction
        self.random = random.Random(seed)

    @property
    def width(self) -> int:
        return self.properties[0][0]

    @property
    def height(self) -> int:
        return self.properties[0][1]

    @property
    def fps(self) -> float:
        return self.properties[0][2]

    def sample(self, retries: int = 8) -> list[np.ndarray]:
        for _ in range(retries):
            item = self.random.randrange(len(self.paths))
            path = self.paths[item]
            _width, _height, _fps, total = self.properties[item]
            centre = self.random.randint(
                noise_twin.TEMPORAL_RADIUS,
                total - noise_twin.TEMPORAL_RADIUS - 1,
            )
            cap = cv2.VideoCapture(path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, centre - noise_twin.TEMPORAL_RADIUS)
            frames = []
            for _index in range(noise_twin.TEMPORAL_WINDOW):
                ok, frame = cap.read()
                if not ok:
                    break
                if self.correction is not None:
                    from fixed_pattern import apply_fixed_pattern_correction

                    frame = apply_fixed_pattern_correction(frame, self.correction)
                frame = cv2.resize(
                    frame,
                    (self.scaled_width, self.scaled_height),
                    interpolation=cv2.INTER_AREA,
                )
                frames.append(frame)
            cap.release()
            if len(frames) != noise_twin.TEMPORAL_WINDOW:
                continue
            levels = np.array([float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()) for frame in frames])
            if levels.max() - levels.min() > 35.0:
                continue
            # Reject obvious moving lines so real meteors never become a noise target.
            maximum = frames[0].copy()
            minimum = frames[0].copy()
            for frame in frames[1:]:
                cv2.max(maximum, frame, dst=maximum)
                cv2.min(minimum, frame, dst=minimum)
            diff = cv2.cvtColor(cv2.absdiff(maximum, minimum), cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(cv2.GaussianBlur(diff, (5, 5), 0), 60, 180)
            if cv2.HoughLinesP(edges, 1, np.pi / 180, 20, minLineLength=20, maxLineGap=5) is not None:
                continue
            x = self.random.randint(0, max(0, self.scaled_width - self.patch_size))
            y = self.random.randint(0, max(0, self.scaled_height - self.patch_size))
            return [frame[y : y + self.patch_size, x : x + self.patch_size] for frame in frames]
        raise noise_twin.NoiseTwinError("流星候補を含まない安定した学習パッチを取得できません。")


def _frames_to_tensors(frames: Sequence[np.ndarray], device):
    torch, _nn, _functional = noise_twin._torch_modules()
    arrays = [
        frame[..., ::-1].transpose(2, 0, 1).copy().astype(np.float32) / 255.0
        for frame in frames
    ]
    centre = torch.from_numpy(arrays[noise_twin.TEMPORAL_RADIUS][None]).to(device)
    neighbors = arrays[: noise_twin.TEMPORAL_RADIUS] + arrays[noise_twin.TEMPORAL_RADIUS + 1 :]
    neighbor_tensor = torch.from_numpy(np.concatenate(neighbors, axis=0)[None]).to(device)
    return neighbor_tensor, centre


def inject_synthetic_meteor(centre, random_state: random.Random):
    """Inject a PSF-softened, positive trajectory and return its exact mask."""
    torch, _nn, _functional = noise_twin._torch_modules()
    batch, _channels, height, width = centre.shape
    mask_np = np.zeros((batch, 1, height, width), dtype=np.float32)
    for item in range(batch):
        angle = random_state.uniform(0, math.pi)
        length = random_state.randint(max(18, width // 12), max(20, width // 2))
        # At quarter scale this represents roughly 1--8 full-resolution pixels.
        maximum_thickness = max(1, min(2, int(round(8 * 0.25))))
        thickness = random_state.randint(1, maximum_thickness)
        cx = random_state.randint(length // 2 + 2, max(length // 2 + 2, width - length // 2 - 3))
        cy = random_state.randint(8, max(8, height - 9))
        dx = math.cos(angle) * length / 2
        dy = math.sin(angle) * length / 2
        p1 = (int(round(cx - dx)), int(round(cy - dy)))
        p2 = (int(round(cx + dx)), int(round(cy + dy)))
        cv2.line(mask_np[item, 0], p1, p2, 1.0, thickness, cv2.LINE_AA)
        sigma = max(0.6, thickness / 2.5)
        mask_np[item, 0] = cv2.GaussianBlur(mask_np[item, 0], (0, 0), sigma)
        if random_state.random() < 0.25:
            mask_np[item, 0, :, :: random_state.randint(5, 12)] *= random_state.uniform(0.15, 0.7)
    mask = torch.from_numpy(mask_np).to(centre.device)
    amplitude = torch.empty((batch, 1, 1, 1), device=centre.device).uniform_(3.0 / 255.0, 150.0 / 255.0)
    color = torch.empty((batch, 3, 1, 1), device=centre.device).uniform_(0.72, 1.0)
    injected = torch.clamp(centre + mask * amplitude * color, 0.0, 1.0)
    return injected, mask


def inject_synthetic_sequence(neighbors, centre, random_state: random.Random):
    """Inject a 1--20 frame moving meteor into the seven-frame observation."""
    torch, _nn, _functional = noise_twin._torch_modules()
    injected_centre, centre_mask = inject_synthetic_meteor(centre, random_state)
    signal = torch.clamp(injected_centre - centre, min=0.0)
    batch, _channels, height, width = centre.shape
    sequence = neighbors.reshape(batch, 6, 3, height, width).clone()
    offsets = (-3, -2, -1, 1, 2, 3)
    duration = random_state.randint(1, 20)
    speed_x = random_state.uniform(-9.0, 9.0)
    speed_y = random_state.uniform(-5.0, 5.0)

    def shifted_without_wrap(value, dx: int, dy: int):
        shifted = torch.roll(value, shifts=(dy, dx), dims=(-2, -1))
        if dy > 0:
            shifted[..., :dy, :] = 0
        elif dy < 0:
            shifted[..., dy:, :] = 0
        if dx > 0:
            shifted[..., :, :dx] = 0
        elif dx < 0:
            shifted[..., :, dx:] = 0
        return shifted

    half_duration = (duration - 1) / 2.0
    for index, offset in enumerate(offsets):
        if abs(offset) > half_duration:
            continue
        dx = int(round(offset * speed_x))
        dy = int(round(offset * speed_y))
        fade = max(0.25, 1.0 - abs(offset) / max(half_duration + 1.0, 2.0))
        sequence[:, index] = torch.clamp(
            sequence[:, index] + shifted_without_wrap(signal, dx, dy) * fade,
            0.0,
            1.0,
        )
    return sequence.reshape(batch, 18, height, width), injected_centre, centre_mask


def _load_frozen_detector(device):
    try:
        import config
        import model as detector_module
        import torch

        if not os.path.exists(config.MODEL_PATH):
            return None
        detector = detector_module.ComplexCNN(num_classes=2).to(device)
        try:
            state = torch.load(config.MODEL_PATH, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(config.MODEL_PATH, map_location=device)
        detector.load_state_dict(state)
        detector.eval()
        for parameter in detector.parameters():
            parameter.requires_grad_(False)
        return detector
    except Exception:
        return None


def _detector_probability(detector, residual):
    torch, _nn, functional = noise_twin._torch_modules()
    if detector is None:
        return None
    image = residual.repeat(1, 3, 1, 1) if residual.shape[1] == 1 else residual
    image = functional.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device)[None, :, None, None]
    return torch.softmax(detector((image - mean) / std), dim=1)[:, 0]


def train_noise_twin(
    video_paths: Sequence[str],
    output_path: str,
    correction: Optional[np.ndarray] = None,
    source_identifier: Optional[str] = None,
    background_steps: int = 1500,
    gate_steps: int = 750,
    validation_injections: int = 10_000,
    progress_callback: Progress = None,
) -> noise_twin.NoiseTwinMetadata:
    torch, _nn, functional = noise_twin._torch_modules()
    device = noise_twin.select_device()
    sampler = VideoPatchSampler(video_paths, correction=correction)
    network = noise_twin.build_model().to(device)
    optimizer = torch.optim.AdamW(network.background.parameters(), lr=2e-4, weight_decay=1e-5)
    network.train()
    for step in range(max(1, int(background_steps))):
        neighbors, centre = _frames_to_tensors(sampler.sample(), device)
        prediction = network.predict_background(neighbors)
        difference = prediction - centre
        loss = torch.sqrt(difference * difference + (1.0 / 255.0) ** 2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 10 == 0 or step + 1 == background_steps:
            _report(progress_callback, "background", step + 1, background_steps)

    for parameter in network.background.parameters():
        parameter.requires_grad_(False)
    gate_optimizer = torch.optim.AdamW(network.gate.parameters(), lr=3e-4, weight_decay=1e-5)
    detector = _load_frozen_detector(device)
    rng = random.Random(7734)
    for step in range(max(1, int(gate_steps))):
        neighbors, centre = _frames_to_tensors(sampler.sample(), device)
        injected_neighbors, injected, target_mask = inject_synthetic_sequence(neighbors, centre, rng)
        with torch.no_grad():
            background = network.predict_background(injected_neighbors)
        residual = torch.clamp(injected - background, min=0.0)
        gray = residual.mean(dim=1, keepdim=True)
        robust_scale = torch.median(gray.flatten(2), dim=2).values[:, :, None, None]
        innovation = torch.clamp(gray / torch.clamp(robust_scale * 1.4826, min=1.0 / 255.0), 0, 16) / 16
        gate = network.gate(injected, background, innovation)
        output = torch.clamp(background + gate * residual, 0.0, 1.0)
        gate_loss = functional.binary_cross_entropy(gate, target_mask)
        source_flux = (residual * target_mask).sum().detach()
        kept_flux = (torch.clamp(output - background, min=0.0) * target_mask).sum()
        flux_loss = torch.abs(kept_flux - source_flux) / torch.clamp(source_flux, min=1e-6)
        detector_loss = torch.tensor(0.0, device=device)
        raw_probability = _detector_probability(detector, gray)
        if raw_probability is not None:
            output_residual = torch.clamp(output - background, min=0.0).mean(dim=1, keepdim=True)
            output_probability = _detector_probability(detector, output_residual)
            detector_loss = torch.relu(raw_probability.detach() - output_probability).mean()
        loss = gate_loss + 2.0 * flux_loss + 0.5 * detector_loss
        gate_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gate_optimizer.step()
        if step % 10 == 0 or step + 1 == gate_steps:
            _report(progress_callback, "gate", step + 1, gate_steps)

    network.eval()
    validation = validate_trained_model(
        network,
        sampler,
        device,
        injection_count=validation_injections,
        progress_callback=progress_callback,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(network.state_dict(), output)
    metadata = noise_twin.new_metadata(
        str(output),
        source_identifier or video_paths[0],
        sampler.width,
        sampler.height,
        sampler.fps,
        correction,
        validation,
    )
    noise_twin.save_metadata(output, metadata)
    return metadata


def validate_trained_model(
    network,
    sampler: VideoPatchSampler,
    device,
    injection_count: int,
    progress_callback: Progress = None,
) -> noise_twin.NoiseTwinValidation:
    torch, _nn, _functional = noise_twin._torch_modules()
    rng = random.Random(9917)
    flux_values = []
    peak_values = []
    trajectory_values = []
    misses = 0
    raw_noise = []
    clean_noise = []
    started = time.perf_counter()
    count = max(1, int(injection_count))
    with torch.inference_mode():
        for index in range(count):
            neighbors, centre = _frames_to_tensors(sampler.sample(), device)
            injected_neighbors, injected, mask = inject_synthetic_sequence(neighbors, centre, rng)
            clean, background, gate, innovation = network(injected_neighbors, injected)
            source = torch.clamp(injected - background, min=0.0) * mask
            kept = torch.clamp(clean - background, min=0.0) * mask
            source_sum = float(source.sum().item())
            kept_sum = float(kept.sum().item())
            flux_values.append(min(1.0, kept_sum / max(source_sum, 1e-8)))
            peak_values.append(min(1.0, float(kept.max().item()) / max(float(source.max().item()), 1e-8)))
            line_pixels = mask > 0.2
            trajectory = float((gate[line_pixels] > 0.5).float().mean().item()) if line_pixels.any() else 0.0
            trajectory_values.append(trajectory)
            if trajectory < 0.5 or float(innovation[mask > 0.2].mean().item()) < 0.08:
                misses += 1
            raw_noise.append(float((centre - background).abs().median().item()))
            clean_noise.append(float((clean - background).abs().median().item()))
            if index % 25 == 0 or index + 1 == count:
                _report(progress_callback, "validation", index + 1, count)
    elapsed = max(1e-6, time.perf_counter() - started)
    # Patch throughput is deliberately not presented as full-frame realtime FPS.
    realtime_fps = 0.0
    missed_fraction = misses / count
    flux = float(np.median(flux_values))
    peak = float(np.median(peak_values))
    trajectory = float(np.median(trajectory_values))
    before = float(np.median(raw_noise))
    after = float(np.median(clean_noise))
    reduction = max(0.0, 1.0 - after / max(before, 1e-8))
    validated = (
        count >= 10_000
        and missed_fraction < 0.005
        and flux >= 0.95
        and peak >= 0.90
        and trajectory >= 0.98
        and reduction >= 0.30
    )
    return noise_twin.NoiseTwinValidation(
        injection_count=count,
        missed_fraction=missed_fraction,
        flux_retention=flux,
        peak_retention=peak,
        trajectory_retention=trajectory,
        false_positive_reduction=reduction,
        realtime_fps=realtime_fps,
        validated=validated,
    )


def benchmark_full_frame(model_path: str, frame_sequence: Sequence[np.ndarray], repeats: int = 8) -> float:
    engine = noise_twin.NoiseTwinEngine(model_path, require_validated=False)
    for _ in range(2):
        engine.infer(frame_sequence)
    started = time.perf_counter()
    for _ in range(max(1, repeats)):
        engine.infer(frame_sequence)
    return max(1, repeats) / max(1e-6, time.perf_counter() - started)


def benchmark_rtsp_stream(
    model_path: str,
    rtsp_url: str,
    correction: Optional[np.ndarray] = None,
    duration_seconds: int = 1800,
    progress_callback: Progress = None,
) -> tuple[float, int, float]:
    """Run a sustained live benchmark and report FPS, missing frames and duration."""
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        raise noise_twin.NoiseTwinError("30分連続検証用RTSPへ接続できません。")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    engine = noise_twin.NoiseTwinEngine(model_path, correction, require_validated=False)
    processor = noise_twin.NoiseTwinStreamProcessor(engine)
    requested = max(1, int(duration_seconds))
    started = time.perf_counter()
    outputs = 0
    failures = 0
    last_report = -1
    try:
        while time.perf_counter() - started < requested:
            ok, frame = cap.read()
            if not ok:
                failures += 1
                if failures >= 30:
                    break
                continue
            failures = 0
            outputs += len(processor.push(frame))
            elapsed_whole = int(time.perf_counter() - started)
            if elapsed_whole != last_report and elapsed_whole % 10 == 0:
                last_report = elapsed_whole
                _report(progress_callback, "realtime", elapsed_whole, requested)
        outputs += len(processor.flush())
    finally:
        cap.release()
    elapsed = max(1e-6, time.perf_counter() - started)
    expected = int(round(source_fps * min(elapsed, requested)))
    dropped = max(0, expected - outputs)
    if failures:
        dropped += failures
    _report(progress_callback, "realtime", min(int(elapsed), requested), requested)
    return outputs / elapsed, dropped, elapsed
