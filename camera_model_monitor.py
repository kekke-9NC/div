"""Periodic RTSP clear-sky monitor for automatic camera-model creation."""

from __future__ import annotations

from datetime import datetime, timedelta
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

import config
from camera_model_builder import CameraModelBuildRequest, CameraModelBuildResult, build_camera_model, discover_video_paths
from cloud_coverage import CloudClassification, classify_cloud_fraction
import utils


class RTSPCameraModelMonitor:
    """Read one RTSP probe per interval and build once a clear run is ready.

    The monitor intentionally consumes completed files from the app's normal
    RTSP recorder.  It never starts a second recorder, so there is only one
    network connection responsible for writing minute segments.
    """

    def __init__(
        self,
        rtsp_url: str,
        *,
        save_root: str = config.RTSP_SAVE_ROOT,
        interval_seconds: int = 60,
        cloud_threshold: float = 0.10,
        minimum_clear_segments: int = 3,
        backend: str = "lmstudio_qwen3_5_2b",
        lm_studio_url: str = "http://localhost:1234/v1",
        lm_studio_model_id: str = "qwen/qwen3-vl-4b",
        lm_studio_api_key: str = "",
        cache_root: Optional[str] = None,
        classifier: Optional[Callable[..., CloudClassification]] = None,
        builder: Optional[Callable[..., CameraModelBuildResult]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
    ):
        self.rtsp_url = rtsp_url
        self.save_root = save_root
        self.interval_seconds = max(10, int(interval_seconds))
        self.cloud_threshold = float(cloud_threshold)
        self.minimum_clear_segments = max(1, int(minimum_clear_segments))
        self.backend = backend
        self.lm_studio_url = lm_studio_url
        self.lm_studio_model_id = lm_studio_model_id
        self.lm_studio_api_key = lm_studio_api_key
        self.cache_root = cache_root
        self.classifier = classifier or classify_cloud_fraction
        self.builder = builder or build_camera_model
        self.status_callback = status_callback
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.last_classification: Optional[CloudClassification] = None
        self.last_build_result: Optional[CameraModelBuildResult] = None
        self._last_build_signature = ""
        self._busy = threading.Lock()

    def _emit(self, message: str) -> None:
        if self.status_callback:
            try:
                self.status_callback(message)
            except Exception:
                pass

    def _probe_frame(self) -> Optional[np.ndarray]:
        cap = utils.create_rtsp_capture(self.rtsp_url)
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            return None
        frames = []
        try:
            for _ in range(5):
                ok, frame = cap.read()
                if ok and frame is not None:
                    frames.append(frame)
            if not frames:
                return None
            # A median of a few probes suppresses a transient meteor/light.
            return np.median(np.stack(frames), axis=0).astype(np.uint8)
        finally:
            cap.release()

    def _recent_segments(self) -> list[str]:
        paths = discover_video_paths(self.save_root)
        paths.sort(key=lambda item: Path(item).stat().st_mtime)
        return paths[-max(self.minimum_clear_segments, 6):]

    def _capture_fallback_segment(self) -> Optional[str]:
        """Record a short sample when the normal RTSP segmenter is not running."""
        cap = utils.create_rtsp_capture(self.rtsp_url)
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            return None
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            width = width if width > 0 else 1920
            height = height if height > 0 else 1080
            fps = fps if 1.0 <= fps <= 120.0 else float(config.RTSP_FPS)
            frame_limit = max(30, round(fps * 8.0))
            stamp = datetime.now()
            folder = Path(self.save_root) / "_camera_model_samples" / stamp.strftime("%Y%m%d") / stamp.strftime("%H")
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{stamp.strftime('%M%S')}_{time.time_ns()}.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not writer.isOpened():
                return None
            written = 0
            try:
                for _ in range(frame_limit):
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                    if frame.shape[1] != width or frame.shape[0] != height:
                        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                    writer.write(frame)
                    written += 1
            finally:
                writer.release()
            if written < max(10, round(fps)):
                try:
                    path.unlink()
                except OSError:
                    pass
                return None
            return str(path)
        finally:
            cap.release()

    def _build_if_ready(self) -> Optional[CameraModelBuildResult]:
        segments = self._recent_segments()
        if len(segments) < self.minimum_clear_segments:
            fallback = self._capture_fallback_segment()
            if fallback:
                segments = self._recent_segments()
                self._emit(f"RTSP保存セグメントが少ないため短時間サンプルを保存しました: {Path(fallback).name}")
        if len(segments) < self.minimum_clear_segments:
            self._emit(f"高精度モデル待機中: RTSP保存済みセグメント {len(segments)}/{self.minimum_clear_segments}")
            return None
        signature = "|".join(segments)
        if signature == self._last_build_signature:
            self._emit("高精度モデル監視: 同じセグメントは作成済みです")
            return None
        stamps = [datetime.fromtimestamp(Path(path).stat().st_mtime) for path in segments]
        start = min(stamps).strftime("%Y-%m-%d %H:%M:%S")
        end = max(stamps).strftime("%Y-%m-%d %H:%M:%S")
        request = CameraModelBuildRequest(
            source=self.save_root,
            start=start,
            end=end,
            cache_root=self.cache_root,
            cloud_threshold=self.cloud_threshold,
            backend=self.backend,
            lm_studio_url=self.lm_studio_url,
            lm_studio_model_id=self.lm_studio_model_id,
            lm_studio_api_key=self.lm_studio_api_key,
            maximum_videos=max(len(segments), self.minimum_clear_segments),
        )
        self._emit("雲量条件を満たしました。高精度プレートソルブモデルを作成します")
        result = self.builder(request, progress_callback=self._emit, classifier=self.classifier)
        self.last_build_result = result
        self._last_build_signature = signature
        if result.success and result.enabled:
            self._emit(f"RTSP自動モデル作成完了: {result.model_path}")
        elif result.success:
            self._emit(f"RTSP自動モデル候補を保存しました（未適用）: {result.report_path}")
        return result

    def run_once(self) -> Optional[CameraModelBuildResult]:
        frame = self._probe_frame()
        if frame is None:
            self._emit("高精度モデル監視: RTSPプローブを取得できません")
            return None
        result = self.classifier(
            frame, backend=self.backend, lm_studio_url=self.lm_studio_url,
            lm_studio_model_id=self.lm_studio_model_id, lm_studio_api_key=self.lm_studio_api_key,
        )
        self.last_classification = result
        self._emit(f"RTSP雲量判定: {result.cloud_fraction * 100:.1f}% ({result.source})")
        if result.source == "qwen-vlm" and (result.error or result.confidence < 0.60):
            self._emit("Qwenの判定信頼度が不足しているため、自動モデル作成を見送ります")
            return None
        if result.cloud_fraction >= self.cloud_threshold:
            self._emit("雲量条件未達。次の監視まで待機します")
            return None
        if not self._busy.acquire(blocking=False):
            self._emit("高精度モデル作成中のため、今回の自動作成をスキップしました")
            return None
        try:
            return self._build_if_ready()
        finally:
            self._busy.release()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.run_once()
            except Exception as exc:
                self._emit(f"高精度モデル監視エラー: {type(exc).__name__}: {exc}")
            self.stop_event.wait(self.interval_seconds)

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, name="camera-model-rtsp-monitor", daemon=True)
        self.thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self.stop_event.set()
        thread = self.thread
        if thread and thread.is_alive():
            thread.join(timeout=max(0.1, float(timeout)))
        self.thread = None
