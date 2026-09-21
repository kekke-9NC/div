#!/usr/bin/env python3
"""JSON-lines bridge used by the SwiftUI front end.

The bridge deliberately does not import Tkinter or the legacy GUI modules.
That keeps the new UI process independent from Tcl/Tk while reusing the
tested video discovery and processing pipeline.

Protocol:
  request:  {"id": "...", "command": "...", "payload": {...}}
  response: {"type": "response", "id": "...", "ok": true, "payload": {...}}
  event:    {"type": "event", "event": "log|progress|status|run_state", ...}

Only JSON is written to stdout. Diagnostics go to stderr so a Swift client
can safely parse every stdout line.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import threading
import traceback
from contextlib import contextmanager
from urllib.parse import urlparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TextIO

from process_lock import AppProcessLock


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov")


class _NonProtocolStdout:
    """Route legacy ``print`` calls away from the NDJSON stdout channel."""

    def __init__(self, fallback: TextIO) -> None:
        self._fallback = fallback

    def write(self, value: str) -> int:
        self._fallback.write(value)
        self._fallback.flush()
        return len(value)

    def flush(self) -> None:
        self._fallback.flush()

    def isatty(self) -> bool:
        return False


class Bridge:
    def __init__(self, root: Path, protocol_stdout: Optional[TextIO] = None) -> None:
        self.root = root.resolve()
        self._protocol_stdout = protocol_stdout or sys.stdout
        self._write_lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._run_thread: Optional[threading.Thread] = None

    def emit(self, payload: Dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with self._write_lock:
            self._protocol_stdout.write(line + "\n")
            self._protocol_stdout.flush()

    def response(
        self,
        request_id: str,
        payload: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        body: Dict[str, Any] = {
            "type": "response",
            "id": request_id,
            "ok": error is None,
            "payload": payload or {},
        }
        if error is not None:
            body["error"] = error
        self.emit(body)

    def event(self, name: str, payload: Optional[Dict[str, Any]] = None) -> None:
        body: Dict[str, Any] = {
            "type": "event",
            "event": name,
            "payload": payload or {},
        }
        self.emit(body)

    def log(self, message: str, level: str = "info") -> None:
        self.event("log", {"message": str(message), "level": level})

    def handle(self, request: Dict[str, Any]) -> None:
        request_id = str(request.get("id", ""))
        command = str(request.get("command", ""))
        payload = request.get("payload") or {}
        try:
            if command == "ping":
                self.response(
                    request_id,
                    {
                        "name": "Meteor Detector",
                        "bridgeVersion": 1,
                        "root": str(self.root),
                    },
                )
            elif command == "load_settings":
                settings = self._load_settings()
                self.response(
                    request_id,
                    {
                        "settings": settings,
                        "unsupportedFeatures": self._unsupported_features(settings),
                    },
                )
            elif command == "validate_mask":
                normalized_path = self._validated_path(payload.get("maskPath"), name="maskPath")
                mask = self._load_detection_mask(
                    {"applyMask": True, "maskPath": normalized_path}
                )
                self.response(
                    request_id,
                    {
                        "valid": True,
                        "height": int(mask.shape[0]),
                        "width": int(mask.shape[1]),
                    },
                )
            elif command == "save_mask":
                saved_path = self._save_mask_from_strokes(payload)
                self.response(request_id, {"saved": True, "path": saved_path})
            elif command == "save_settings":
                settings = payload.get("settings") or {}
                self._save_settings(settings)
                self.response(request_id, {"saved": True})
            elif command == "discover_sources":
                self._discover(request_id, payload)
            elif command == "run_detection":
                self._start_run(request_id, payload)
            elif command == "run_periodic":
                self._start_periodic_run(request_id, payload)
            elif command == "run_rtsp":
                self._start_rtsp_run(request_id, payload)
            elif command == "build_camera_model":
                self._start_camera_model_build(request_id, payload)
            elif command == "cancel":
                self._cancel()
                self.response(request_id, {"accepted": True})
            else:
                self.response(request_id, error=f"Unknown command: {command}")
        except Exception as exc:  # boundary for the UI process
            self.log(f"バックエンドエラー: {exc}", "error")
            self.response(request_id, error=str(exc))

    def _settings_path(self) -> Path:
        return self.root / "app_settings.json"

    def _load_settings(self) -> Dict[str, Any]:
        path = self._settings_path()
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict):
                raise ValueError("設定ファイルのトップレベルはオブジェクトである必要があります")
            return data
        except (OSError, json.JSONDecodeError) as exc:
            self.log(f"設定を読み込めませんでした: {exc}", "error")
            raise ValueError("app_settings.jsonが壊れているため、設定を保存せずに停止しました") from exc
        except ValueError as exc:
            self.log(f"設定を読み込めませんでした: {exc}", "error")
            raise ValueError("app_settings.jsonの形式が不正なため、設定を保存せずに停止しました") from exc

    @staticmethod
    def _unsupported_features(settings: Dict[str, Any]) -> List[str]:
        unsupported: List[str] = []
        if settings.get("apply_rtsp_dark"):
            unsupported.append("RTSP固定パターン補正")
        if settings.get("advanced_settings"):
            unsupported.append("詳細パラメータ")
        if settings.get("camera_control_base_url") or settings.get("camera_control_ev_target"):
            unsupported.append("カメラ制御")
        if settings.get("ml_training_export_enabled") or settings.get("auto_video_mask_enabled"):
            unsupported.append("学習データ / 自動マスク")
        if settings.get("ai_vlm_backend") not in (None, "", "local_qwen3_vl_4b", "lmstudio_qwen3_5_2b"):
            unsupported.append("AIアシスタント設定")
        if settings.get("lm_studio_vlm_url") not in (None, "", "http://localhost:1234/v1"):
            unsupported.append("AIアシスタント設定")
        concat_defaults = {
            "bitrate": "Auto",
            "codec": "h264",
            "fps": "Auto",
            "safe_mode": True,
            "apply_enhancement": False,
            "timestamp_enabled": True,
            "timestamp_position": "右下",
            "timestamp_size_percent": "1.8",
            "timestamp_offset_seconds": "0.0",
        }
        if settings.get("video_concat_settings") not in (None, concat_defaults):
            unsupported.append("動画連結設定")
        timestamp_defaults = {
            "enabled": True,
            "position": "右下",
            "size_percent": "1.8",
        }
        if settings.get("full_video_timestamp") not in (None, timestamp_defaults):
            unsupported.append("フルサイズ動画の時刻表示")
        encoding_defaults = {
            "codec": "H.265 / HEVC (推奨)",
            "quality": "入力品質基準（推奨）",
            "bitrate_mbps": "40",
        }
        if settings.get("processed_video_encoding") not in (None, encoding_defaults):
            unsupported.append("動画エンコード設定")
        return list(dict.fromkeys(unsupported))

    def _save_settings(self, settings: Dict[str, Any]) -> None:
        if not isinstance(settings, dict):
            raise ValueError("settings must be an object")
        path = self._settings_path()
        existing = self._load_settings()
        existing.update(settings)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Replace atomically so a quit or power loss cannot leave a truncated
        # app_settings.json that the legacy UI can no longer read.
        fd, temporary_name = tempfile.mkstemp(
            prefix=".app_settings.", suffix=".json", dir=str(path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(existing, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        finally:
            if os.path.exists(temporary_name):
                try:
                    os.unlink(temporary_name)
                except OSError:
                    pass

    def _discover(self, request_id: str, payload: Dict[str, Any]) -> None:
        selected_paths = [str(item) for item in payload.get("paths", []) if str(item)]
        if not selected_paths:
            self.response(request_id, {"sources": []})
            return

        with self._run_lock:
            if self._run_thread is not None and self._run_thread.is_alive():
                self.response(request_id, error="別の処理が実行中です")
                return
            self._cancel_event = threading.Event()
            cancel_event = self._cancel_event
            self._run_thread = threading.Thread(
                target=self._discover_worker,
                args=(request_id, payload, selected_paths, cancel_event),
                name="swiftui-source-discovery",
                daemon=True,
            )
            self._run_thread.start()

    def _discover_worker(
        self,
        request_id: str,
        payload: Dict[str, Any],
        selected_paths: List[str],
        cancel_event: threading.Event,
    ) -> None:
        try:
            # Import lazily so ping/settings and cancellation stay responsive.
            import folder_source_discovery

            self.event("run_state", {"state": "preparing"})
            self.log(f"{len(selected_paths)}個の入力を走査しています…")

            def progress(message: str) -> None:
                self.event("log", {"message": message, "level": "info"})

            sources = folder_source_discovery.discover_sources(
                selected_paths,
                VIDEO_EXTENSIONS,
                twilight_filter_enabled=bool(payload.get("twilightFilter", True)),
                latitude=float(payload.get("latitude", 35.0)),
                longitude=float(payload.get("longitude", 135.0)),
                progress_callback=progress,
                cancel_flag=cancel_event,
            )
            if cancel_event.is_set():
                self._release_run_thread()
                self.response(request_id, {"sources": [], "cancelled": True})
                self.event("run_state", {"state": "cancelled"})
                return
            serialized = [
                {"path": str(item.get("path", "")), "is_rtsp": False}
                for item in sources
                if item.get("path")
            ]
            self._release_run_thread()
            self.response(request_id, {"sources": serialized})
            if not serialized:
                self.event("run_state", {"state": "idle"})
                self.log("選択された場所に対応する動画が見つかりませんでした。", "warning")
            else:
                self.event("run_state", {"state": "discovery_completed"})
                self.log(f"{len(serialized)}本の動画を検出しました。")
        except Exception as exc:
            self._release_run_thread()
            self.log(f"入力の走査中にエラーが発生しました: {exc}", "error")
            self.response(request_id, error=str(exc))
            self.event("run_state", {"state": "failed", "error": str(exc)})

    def _release_run_thread(self) -> None:
        with self._run_lock:
            if self._run_thread is threading.current_thread():
                self._run_thread = None

    def _start_run(self, request_id: str, payload: Dict[str, Any]) -> None:
        with self._run_lock:
            if self._run_thread is not None and self._run_thread.is_alive():
                self.response(request_id, error="別の処理が実行中です")
                return
            sources = payload.get("sources") or []
            if not isinstance(sources, list) or not sources:
                self.response(request_id, error="処理対象の動画がありません")
                return
            try:
                normalized_payload = dict(payload)
                normalized_payload["saveOptions"] = self._validated_save_options(
                    payload.get("saveOptions")
                )
                normalized_payload["summaryConfig"] = self._validated_summary_config(
                    payload.get("summaryConfig")
                )
                normalized_payload["applyMask"] = self._validated_bool(
                    payload.get("applyMask"), default=False, name="applyMask"
                )
                normalized_payload["maskPath"] = self._validated_path(
                    payload.get("maskPath"), name="maskPath"
                )
                normalized_payload["modelPath"] = self._validated_model_path(
                    payload.get("modelPath")
                )
                normalized_payload["globalWCSInfo"] = self._validated_wcs_info(
                    payload.get("plateSolveWCSPath")
                )
                normalized_payload["noiseTwinOptions"] = self._validated_noise_twin_options(
                    payload.get("noiseTwinOptions")
                )
            except ValueError as exc:
                self.response(request_id, error=str(exc))
                return
            self._cancel_event = threading.Event()
            cancel_event = self._cancel_event
            self._run_thread = threading.Thread(
                target=self._run_pipeline,
                args=(normalized_payload, cancel_event),
                name="swiftui-detection-pipeline",
                daemon=True,
            )
            self._run_thread.start()
        self.response(request_id, {"accepted": True, "count": len(sources)})

    def _start_rtsp_run(self, request_id: str, payload: Dict[str, Any]) -> None:
        with self._run_lock:
            if self._run_thread is not None and self._run_thread.is_alive():
                self.response(request_id, error="別の処理が実行中です")
                return
            url = str(payload.get("url", "")).strip()
            parsed = urlparse(url)
            if parsed.scheme.lower() not in {"rtsp", "rtsps"} or not parsed.hostname:
                self.response(request_id, error="有効なRTSP URLがありません")
                return
            try:
                normalized_payload = dict(payload)
                normalized_payload["saveOptions"] = self._validated_save_options(
                    payload.get("saveOptions")
                )
                normalized_payload["summaryConfig"] = self._validated_summary_config(
                    payload.get("summaryConfig")
                )
                normalized_payload["timeLimitEnabled"] = self._validated_bool(
                    payload.get("timeLimitEnabled"), default=False, name="timeLimitEnabled"
                )
                normalized_payload["notifyOnDetection"] = self._validated_bool(
                    payload.get("notifyOnDetection"), default=True, name="notifyOnDetection"
                )
                normalized_payload["applyMask"] = self._validated_bool(
                    payload.get("applyMask"), default=False, name="applyMask"
                )
                normalized_payload["maskPath"] = self._validated_path(
                    payload.get("maskPath"), name="maskPath"
                )
                normalized_payload["modelPath"] = self._validated_model_path(
                    payload.get("modelPath")
                )
                normalized_payload["globalWCSInfo"] = self._validated_wcs_info(
                    payload.get("plateSolveWCSPath")
                )
                normalized_payload["rtspPreset"] = self._validated_choice(
                    payload.get("rtspPreset"), {"cloudy", "clear"}, "cloudy", "rtspPreset"
                )
                normalized_payload["rtspFps"] = self._validated_int(
                    payload.get("rtspFps"), 1, 120, 25, "rtspFps"
                )
                normalized_payload["noiseTwinOptions"] = self._validated_noise_twin_options(
                    payload.get("noiseTwinOptions")
                )
                for key, lower, upper, default in (
                    ("startHour", 0, 23, 17),
                    ("startMinute", 0, 59, 0),
                    ("endHour", 0, 23, 7),
                    ("endMinute", 0, 59, 0),
                ):
                    normalized_payload[key] = self._validated_int(
                        payload.get(key), lower, upper, default, key
                    )
                self._validate_time_window(normalized_payload)
            except ValueError as exc:
                self.response(request_id, error=str(exc))
                return
            self._cancel_event = threading.Event()
            cancel_event = self._cancel_event
            self._run_thread = threading.Thread(
                target=self._run_rtsp,
                args=(normalized_payload, cancel_event),
                name="swiftui-rtsp-pipeline",
                daemon=True,
            )
            self._run_thread.start()
        self.response(request_id, {"accepted": True, "url": url})

    def _start_periodic_run(self, request_id: str, payload: Dict[str, Any]) -> None:
        with self._run_lock:
            if self._run_thread is not None and self._run_thread.is_alive():
                self.response(request_id, error="別の処理が実行中です")
                return
            raw_directory = payload.get("directory")
            if not isinstance(raw_directory, str) or not raw_directory.strip():
                self.response(request_id, error="有効な定期スキャンフォルダがありません")
                return
            directory = self._safe_path(raw_directory, self.root)
            if not directory.is_dir():
                self.response(request_id, error="有効な定期スキャンフォルダがありません")
                return
            try:
                normalized_payload = dict(payload)
                normalized_payload["saveOptions"] = self._validated_save_options(
                    payload.get("saveOptions")
                )
                normalized_payload["summaryConfig"] = self._validated_summary_config(
                    payload.get("summaryConfig")
                )
                normalized_payload["applyMask"] = self._validated_bool(
                    payload.get("applyMask"), default=False, name="applyMask"
                )
                normalized_payload["maskPath"] = self._validated_path(
                    payload.get("maskPath"), name="maskPath"
                )
                normalized_payload["modelPath"] = self._validated_model_path(
                    payload.get("modelPath")
                )
                normalized_payload["globalWCSInfo"] = self._validated_wcs_info(
                    payload.get("plateSolveWCSPath")
                )
                normalized_payload["noiseTwinOptions"] = self._validated_noise_twin_options(
                    payload.get("noiseTwinOptions")
                )
                normalized_payload["timeLimitEnabled"] = self._validated_bool(
                    payload.get("timeLimitEnabled"), default=False, name="timeLimitEnabled"
                )
                normalized_payload["scanInterval"] = self._validated_int(
                    payload.get("scanInterval"), 5, 3600, 60, "scanInterval"
                )
                for key, lower, upper, default in (
                    ("startHour", 0, 23, 17),
                    ("startMinute", 0, 59, 0),
                    ("endHour", 0, 23, 7),
                    ("endMinute", 0, 59, 0),
                ):
                    normalized_payload[key] = self._validated_int(
                        payload.get(key), lower, upper, default, key
                    )
                self._validate_time_window(normalized_payload)
            except ValueError as exc:
                self.response(request_id, error=str(exc))
                return
            self._cancel_event = threading.Event()
            cancel_event = self._cancel_event
            self._run_thread = threading.Thread(
                target=self._run_periodic,
                args=(normalized_payload, cancel_event),
                name="swiftui-periodic-scan",
                daemon=True,
            )
            self._run_thread.start()
        self.response(request_id, {"accepted": True, "directory": str(directory)})

    def _validated_camera_model_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        source_value = payload.get("source")
        if not isinstance(source_value, str) or not source_value.strip():
            raise ValueError("モデル作成対象の動画またはフォルダを選択してください")
        source = self._safe_path(source_value, self.root)
        if not source.is_file() and not source.is_dir():
            raise ValueError(f"モデル作成対象が見つかりません: {source}")

        def text_value(name: str, default: str = "") -> str:
            value = payload.get(name)
            if value is None:
                return default
            if not isinstance(value, str) or len(value) > 128:
                raise ValueError(f"{name} must be a short string")
            return value.strip()

        cloud_threshold = payload.get("cloudThreshold", 0.10)
        if isinstance(cloud_threshold, bool) or not isinstance(cloud_threshold, (int, float)):
            raise ValueError("cloudThreshold must be a number")
        cloud_threshold = float(cloud_threshold)
        if not math.isfinite(cloud_threshold) or not 0.0 <= cloud_threshold <= 1.0:
            raise ValueError("cloudThreshold is out of range")

        def finite_float(name: str, lower: float, upper: float, default: float) -> float:
            value = payload.get(name, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number")
            value = float(value)
            if not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"{name} is out of range")
            return value

        cache_root = text_value("cacheRoot")
        return {
            "source": str(source),
            "start": text_value("start"),
            "end": text_value("end"),
            "autoSelect": self._validated_bool(
                payload.get("autoSelect"), default=True, name="autoSelect"
            ),
            "cloudThreshold": cloud_threshold,
            "useCloudFilter": self._validated_bool(
                payload.get("useCloudFilter"), default=False, name="useCloudFilter"
            ),
            "maximumVideos": self._validated_int(
                payload.get("maximumVideos"), 1, 50, 12, "maximumVideos"
            ),
            "observationLatitude": finite_float(
                "observationLatitude", -90.0, 90.0, 35.0
            ),
            "observationLongitude": finite_float(
                "observationLongitude", -180.0, 180.0, 135.0
            ),
            "cacheRoot": str(self._safe_path(cache_root, self.root)) if cache_root else "",
            "backend": text_value("backend", "lmstudio_qwen3_5_2b"),
            "lmStudioURL": text_value("lmStudioURL", "http://localhost:1234/v1"),
            "lmStudioModelID": text_value("lmStudioModelID", "qwen/qwen3-vl-4b"),
            "lmStudioAPIKey": text_value("lmStudioAPIKey"),
        }

    def _start_camera_model_build(self, request_id: str, payload: Dict[str, Any]) -> None:
        with self._run_lock:
            if self._run_thread is not None and self._run_thread.is_alive():
                self.response(request_id, error="別の処理が実行中です")
                return
            try:
                normalized_payload = self._validated_camera_model_payload(payload)
            except ValueError as exc:
                self.response(request_id, error=str(exc))
                return
            self._cancel_event = threading.Event()
            cancel_event = self._cancel_event
            self._run_thread = threading.Thread(
                target=self._build_camera_model_worker,
                args=(request_id, normalized_payload, cancel_event),
                name="swiftui-camera-model-builder",
                daemon=True,
            )
            self._run_thread.start()
        self.response(request_id, {"accepted": True, "source": normalized_payload["source"]})

    def _build_camera_model_worker(
        self,
        request_id: str,
        payload: Dict[str, Any],
        cancel_event: threading.Event,
    ) -> None:
        self.event("run_state", {"state": "preparing"})
        try:
            from camera_model_builder import CameraModelBuildRequest, build_camera_model

            cancellation_abort_sent = False

            def progress(message: str) -> None:
                nonlocal cancellation_abort_sent
                if cancel_event.is_set():
                    if cancellation_abort_sent:
                        return
                    cancellation_abort_sent = True
                    raise RuntimeError("高精度カメラ補正の停止が要求されました")
                self.event(
                    "progress",
                    {
                        "current": 0,
                        "total": 0,
                        "message": f"高精度モデル: {message}",
                    },
                )

            self.event("run_state", {"state": "running"})
            request = CameraModelBuildRequest(
                source=payload["source"],
                start=payload["start"],
                end=payload["end"],
                auto_select=payload["autoSelect"],
                cache_root=payload["cacheRoot"] or None,
                cloud_threshold=payload["cloudThreshold"],
                use_cloud_filter=payload["useCloudFilter"],
                backend=payload["backend"],
                lm_studio_url=payload["lmStudioURL"],
                lm_studio_model_id=payload["lmStudioModelID"],
                lm_studio_api_key=payload["lmStudioAPIKey"],
                maximum_videos=payload["maximumVideos"],
                observation_latitude=payload["observationLatitude"],
                observation_longitude=payload["observationLongitude"],
            )
            result = build_camera_model(request, progress_callback=progress)
            result_payload = result.as_dict()
            result_payload["cancelled"] = cancel_event.is_set()
            self.event("camera_model_result", result_payload)
            if cancel_event.is_set():
                self.log("高精度カメラ補正の作成を停止しました。", "warning")
                self.event("run_state", {"state": "cancelled"})
            elif result.success:
                self.log(f"高精度カメラ補正を登録しました: {result.model_path}")
                self.event("run_state", {"state": "completed"})
            else:
                self.log(f"高精度カメラ補正を作成できませんでした: {result.error}", "error")
                self.event("run_state", {"state": "failed", "error": result.error})
            self._release_run_thread()
        except Exception as exc:
            self.log(f"高精度カメラ補正中にエラーが発生しました: {exc}", "error")
            traceback.print_exc(file=sys.stderr)
            self.event("run_state", {"state": "failed", "error": str(exc)})
            self._release_run_thread()

    @classmethod
    def _validated_save_options(cls, value: Any) -> Dict[str, bool]:
        defaults = cls._default_save_options()
        if value is None:
            return defaults
        if not isinstance(value, dict):
            raise ValueError("saveOptions must be an object")
        normalized = dict(defaults)
        for key, item in value.items():
            if key not in normalized or not isinstance(item, bool):
                raise ValueError("saveOptionsの形式が不正です")
            normalized[key] = item
        return normalized

    @staticmethod
    def _validated_summary_config(value: Any) -> List[Dict[str, Any]]:
        if value is None:
            return Bridge._default_summary_config()
        if not isinstance(value, list):
            raise ValueError("summaryConfig must be an array")
        normalized: List[Dict[str, Any]] = []
        allowed_names = {
            "Composite Image",
            "Annotated Image",
            "Full Size Video",
            "Zoom Sequence",
            "Cutout Video",
        }
        duration_defaults = {
            "Composite Image": 1.0,
            "Annotated Image": 2.0,
            "Zoom Sequence": 2.0,
        }
        seen_names = set()
        for item in value:
            if not isinstance(item, dict):
                raise ValueError("summaryConfigの形式が不正です")
            name = item.get("name")
            enabled = item.get("enabled")
            if not isinstance(name, str) or not name.strip() or not isinstance(enabled, bool):
                raise ValueError("summaryConfigの形式が不正です")
            if name not in allowed_names or name in seen_names:
                raise ValueError("summaryConfigに未対応または重複した項目があります")
            entry: Dict[str, Any] = {"name": name, "enabled": enabled}
            if "duration" in item:
                duration = item["duration"]
                if isinstance(duration, bool) or not isinstance(duration, (int, float)):
                    raise ValueError("summaryConfigのdurationが不正です")
                if not math.isfinite(float(duration)) or not 0.05 <= float(duration) <= 60:
                    raise ValueError("summaryConfigのdurationが範囲外です")
                entry["duration"] = float(duration)
            elif name in duration_defaults:
                entry["duration"] = duration_defaults[name]
            normalized.append(entry)
            seen_names.add(name)
        if not normalized or not any(item["enabled"] for item in normalized):
            raise ValueError("summaryConfigは1つ以上の出力を有効にしてください")
        return normalized

    @staticmethod
    def _validated_bool(value: Any, default: bool, name: str) -> bool:
        if value is None:
            return default
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean")
        return value

    @staticmethod
    def _validated_path(value: Any, name: str) -> str:
        if value is None:
            return ""
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        return value.strip()

    def _validated_model_path(self, value: Any) -> str:
        """Return an existing model path, or an empty path for the default model."""
        if value is None:
            return ""
        if not isinstance(value, str):
            raise ValueError("modelPath must be a string")
        raw_path = value.strip()
        if not raw_path:
            return ""
        model_path = self._safe_path(raw_path, self.root)
        if not model_path.is_file():
            raise ValueError(f"検出モデルが見つかりません: {model_path}")
        return str(model_path)

    def _validated_wcs_info(self, value: Any) -> Optional[Dict[str, Any]]:
        """Validate an existing FITS WCS or fixed-camera JSON calibration."""
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("plateSolveWCSPath must be a string")
        raw_path = value.strip()
        if not raw_path:
            return None
        wcs_path = self._safe_path(raw_path, self.root)
        if not wcs_path.is_file():
            raise ValueError(f"カメラ補正データが見つかりません: {wcs_path}")
        if wcs_path.suffix.lower() not in {".json", ".wcs", ".fits", ".fit"}:
            raise ValueError("カメラ補正データは .json / .wcs / .fits / .fit を選択してください")
        info: Dict[str, Any] = {
            "wcs_file": str(wcs_path),
            "job_id": "local-wideangle-camera-model" if wcs_path.suffix.lower() == ".json" else "manual-wcs",
        }
        if wcs_path.suffix.lower() == ".json":
            info["calibration_path"] = str(wcs_path)
            info["model_path"] = str(wcs_path)
        return info

    def _validated_noise_twin_options(self, value: Any) -> Dict[str, Any]:
        if value is None:
            return {
                "enabled": False,
                "model_path": "",
                "require_validated": True,
                "temporal_mean_frames": 0,
                "save_temporal_mean_video": True,
            }
        if not isinstance(value, dict):
            raise ValueError("noiseTwinOptions must be an object")

        enabled = self._validated_bool(
            value.get("enabled"), default=False, name="noiseTwinOptions.enabled"
        )
        temporal_mean_frames = self._validated_int(
            value.get("temporalMeanFrames"), 0, 5, 0, "noiseTwinOptions.temporalMeanFrames"
        )
        if temporal_mean_frames not in (0, 3, 5):
            raise ValueError("noiseTwinOptions.temporalMeanFrames must be 0, 3, or 5")
        save_temporal_mean_video = self._validated_bool(
            value.get("saveTemporalMeanVideo"),
            default=True,
            name="noiseTwinOptions.saveTemporalMeanVideo",
        )
        model_path = ""
        if enabled:
            model_path = self._validated_path(
                value.get("modelPath"), name="noiseTwinOptions.modelPath"
            )
            model_file = self._safe_path(model_path, self.root)
            if not model_file.is_file():
                raise ValueError(f"NoiseTwinモデルが見つかりません: {model_file}")
            try:
                import noise_twin

                metadata = noise_twin.load_metadata(str(model_file))
            except Exception as exc:
                raise ValueError(f"NoiseTwinモデルを読み込めませんでした: {exc}") from exc
            if not metadata.validation.validated:
                raise ValueError("NoiseTwinモデルが採用基準を満たしていません")
        if enabled and temporal_mean_frames:
            raise ValueError("NoiseTwinと時間平均は同時に使用できません")
        return {
            "enabled": enabled,
            "model_path": model_path,
            "require_validated": True,
            "temporal_mean_frames": temporal_mean_frames,
            "save_temporal_mean_video": save_temporal_mean_video,
        }

    @staticmethod
    def _validated_choice(value: Any, choices: set[str], default: str, name: str) -> str:
        if value is None:
            return default
        if not isinstance(value, str) or value not in choices:
            raise ValueError(f"{name} is invalid")
        return value

    @staticmethod
    def _validate_time_window(payload: Dict[str, Any]) -> None:
        if not payload.get("timeLimitEnabled"):
            return
        start = (payload["startHour"], payload["startMinute"])
        end = (payload["endHour"], payload["endMinute"])
        if start == end:
            raise ValueError("時間制限の開始と終了を同じにはできません。24時間運用では時間制限をオフにしてください")

    @staticmethod
    def _validated_int(value: Any, lower: int, upper: int, default: int, name: str) -> int:
        if value is None:
            return default
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        parsed = value
        if not lower <= parsed <= upper:
            raise ValueError(f"{name} is out of range")
        return parsed

    def _run_pipeline(self, payload: Dict[str, Any], cancel_event: threading.Event) -> None:
        self.event("run_state", {"state": "running"})
        try:
            import config
            import download_pipeline

            self._load_selected_model(payload)
            mask = self._load_detection_mask(payload)

            sources = [
                {"path": str(item.get("path", "")), "is_rtsp": False}
                for item in payload.get("sources", [])
                if isinstance(item, dict) and item.get("path")
            ]
            if not sources:
                raise ValueError("有効な動画がありません")

            meteor_path = self._safe_path(
                payload.get("meteorSavePath"), self.root / "meteor"
            )
            not_meteor_path = self._safe_path(
                payload.get("notMeteorSavePath"), self.root / "not_meteor"
            )
            meteor_path.mkdir(parents=True, exist_ok=True)
            not_meteor_path.mkdir(parents=True, exist_ok=True)

            max_workers = self._bounded_int(payload.get("maxWorkers", 4), 1, 6)
            interval = self._bounded_float(payload.get("interval", 1.0), 0.05, 60.0)
            duration = self._bounded_float(payload.get("duration", 1.0), 0.05, 30.0)
            save_options = payload.get("saveOptions") or self._default_save_options()
            summary_config = payload.get("summaryConfig") or self._default_summary_config()
            tmp_root = self.root / "temp_video"
            tmp_root.mkdir(parents=True, exist_ok=True)

            def progress_callback(item: Any) -> None:
                message = None
                value: Any = None
                if isinstance(item, (tuple, list)) and len(item) == 2:
                    message, value = item
                if isinstance(value, (tuple, list)) and len(value) == 2:
                    current = int(value[0])
                    total = max(0, int(value[1]))
                    self.event(
                        "progress",
                        {
                            "current": current,
                            "total": total,
                            "message": str(message or ""),
                        },
                    )
                elif message:
                    self.event("log", {"message": str(message), "level": "info"})

            def status_callback(status: Dict[str, Any]) -> None:
                self.event("status", status)

            self.log(f"{len(sources)}本の動画の解析を開始しました。")
            download_pipeline.run_pipeline(
                sources=sources,
                max_workers=max_workers,
                interval=interval,
                duration=duration,
                mask=mask,
                global_wcs_info=payload.get("globalWCSInfo"),
                plate_solve_mask=None,
                meteor_save_path=str(meteor_path),
                not_meteor_save_path=str(not_meteor_path),
                cancel_flag=cancel_event,
                progress_callback=progress_callback,
                save_options=save_options,
                summary_video_config=summary_config,
                tmp_root=str(tmp_root),
                status_callback=status_callback,
                fixed_pattern_correction=None,
                noise_twin_options=payload.get("noiseTwinOptions") or {"enabled": False},
            )
            if cancel_event.is_set():
                self.log("処理をキャンセルしました。", "warning")
                self._release_run_thread()
                self.event("run_state", {"state": "cancelled"})
            else:
                self.event("progress", {"current": len(sources), "total": len(sources), "message": "完了"})
                self.log("すべての処理が完了しました。")
                self._release_run_thread()
                self.event("run_state", {"state": "completed"})
        except Exception as exc:
            self.log(f"処理中にエラーが発生しました: {exc}", "error")
            traceback.print_exc(file=sys.stderr)
            self._release_run_thread()
            self.event("run_state", {"state": "failed", "error": str(exc)})

    def _run_periodic(self, payload: Dict[str, Any], cancel_event: threading.Event) -> None:
        self.event("run_state", {"state": "running"})
        try:
            import config
            import file_utils

            self._load_selected_model(payload)
            mask = self._load_detection_mask(payload)

            directory = self._safe_path(payload.get("directory"), self.root)
            meteor_path = self._safe_path(
                payload.get("meteorSavePath"), self.root / "meteor"
            )
            not_meteor_path = self._safe_path(
                payload.get("notMeteorSavePath"), self.root / "not_meteor"
            )
            meteor_path.mkdir(parents=True, exist_ok=True)
            not_meteor_path.mkdir(parents=True, exist_ok=True)

            def progress_callback(item: Any) -> None:
                message = None
                value: Any = None
                if isinstance(item, (tuple, list)) and len(item) == 2:
                    message, value = item
                if isinstance(value, (tuple, list)) and len(value) == 2:
                    self.event(
                        "progress",
                        {
                            "current": int(value[0]),
                            "total": max(0, int(value[1])),
                            "message": str(message or ""),
                        },
                    )
                elif message:
                    self.event("log", {"message": str(message), "level": "info"})

            self.log(f"定期スキャンを開始しました: {directory}")
            file_utils.monitor_directory(
                directory=str(directory),
                scan_interval=self._bounded_int(payload.get("scanInterval", 60), 5, 3600),
                progress_callback=progress_callback,
                mask=mask,
                global_wcs_info=payload.get("globalWCSInfo"),
                plate_solve_mask=None,
                meteor_save_path=str(meteor_path),
                not_meteor_save_path=str(not_meteor_path),
                cancel_flag=cancel_event,
                save_options=payload.get("saveOptions") or self._default_save_options(),
                interval=self._bounded_float(payload.get("interval", 1.0), 0.05, 60.0),
                duration=self._bounded_float(payload.get("duration", 1.0), 0.05, 30.0),
                min_length=config.MIN_LINE_LENGTH,
                summary_video_config=payload.get("summaryConfig") or self._default_summary_config(),
                time_limit_enabled=bool(payload.get("timeLimitEnabled", False)),
                start_hour=self._bounded_int(payload.get("startHour", 17), 0, 23),
                start_minute=self._bounded_int(payload.get("startMinute", 0), 0, 59),
                end_hour=self._bounded_int(payload.get("endHour", 7), 0, 23),
                end_minute=self._bounded_int(payload.get("endMinute", 0), 0, 59),
                fixed_pattern_correction=None,
                noise_twin_options=payload.get("noiseTwinOptions") or {"enabled": False},
            )
            if cancel_event.is_set():
                self.log("定期スキャンを停止しました。", "warning")
                self._release_run_thread()
                self.event("run_state", {"state": "cancelled"})
            else:
                self._release_run_thread()
                self.event("run_state", {"state": "completed"})
        except Exception as exc:
            self.log(f"定期スキャン中にエラーが発生しました: {exc}", "error")
            traceback.print_exc(file=sys.stderr)
            self._release_run_thread()
            self.event("run_state", {"state": "failed", "error": str(exc)})

    def _run_rtsp(self, payload: Dict[str, Any], cancel_event: threading.Event) -> None:
        self.event("run_state", {"state": "running"})
        try:
            import config
            import file_utils

            self._load_selected_model(payload)
            mask = self._load_detection_mask(payload)

            url = str(payload.get("url", "")).strip()
            meteor_path = self._safe_path(
                payload.get("meteorSavePath"), self.root / "meteor"
            )
            not_meteor_path = self._safe_path(
                payload.get("notMeteorSavePath"), self.root / "not_meteor"
            )
            rtsp_root = self.root / "rtsp"
            meteor_path.mkdir(parents=True, exist_ok=True)
            not_meteor_path.mkdir(parents=True, exist_ok=True)
            rtsp_root.mkdir(parents=True, exist_ok=True)

            def progress_callback(item: Any) -> None:
                message = None
                value: Any = None
                if isinstance(item, (tuple, list)) and len(item) == 2:
                    message, value = item
                if isinstance(value, (tuple, list)) and len(value) == 2:
                    self.event(
                        "progress",
                        {
                            "current": int(value[0]),
                            "total": max(0, int(value[1])),
                            "message": str(message or ""),
                        },
                    )
                elif message:
                    self.event("log", {"message": str(message), "level": "info"})

            self.log("RTSP録画と解析を開始しました。停止するまで監視します。")
            # Legacy RTSP code still writes diagnostic ``print`` messages.
            # main() replaces sys.stdout with a stderr sink, while emit()
            # writes directly to the preserved protocol stream.
            with self._temporary_rtsp_config(config, payload):
                file_utils.rtsp_save_and_process_thread_target(
                    rtsp_url=url,
                    save_root=str(rtsp_root),
                    segment_duration=config.RTSP_SEGMENT_DURATION,
                    scan_interval=60,
                    progress_callback=progress_callback,
                    mask=mask,
                    global_wcs_info=payload.get("globalWCSInfo"),
                    plate_solve_mask=None,
                    meteor_save_path=str(meteor_path),
                    not_meteor_save_path=str(not_meteor_path),
                    cancel_flag=cancel_event,
                    save_options=payload.get("saveOptions") or self._default_save_options(),
                    interval=self._bounded_float(payload.get("interval", 1.0), 0.05, 60.0),
                    duration=self._bounded_float(payload.get("duration", 1.0), 0.05, 30.0),
                    min_length=config.MIN_LINE_LENGTH,
                    summary_video_config=payload.get("summaryConfig") or self._default_summary_config(),
                    time_limit_enabled=bool(payload.get("timeLimitEnabled", False)),
                    start_hour=self._bounded_int(payload.get("startHour", 17), 0, 23),
                    start_minute=self._bounded_int(payload.get("startMinute", 0), 0, 59),
                    end_hour=self._bounded_int(payload.get("endHour", 7), 0, 23),
                    end_minute=self._bounded_int(payload.get("endMinute", 0), 0, 59),
                    max_workers=self._bounded_int(payload.get("maxWorkers", 1), 1, 6),
                    preview_callback=None,
                    dark_frame=None,
                    notify_on_detection=bool(payload.get("notifyOnDetection", True)),
                    noise_twin_options=payload.get("noiseTwinOptions") or {"enabled": False},
                    rtsp_fps=self._bounded_int(payload.get("rtspFps", 25), 1, 120),
            )
            if cancel_event.is_set():
                self.log("RTSP処理を停止しました。")
                self._release_run_thread()
                self.event("run_state", {"state": "cancelled"})
            else:
                self._release_run_thread()
                self.event("run_state", {"state": "completed"})
        except Exception as exc:
            self.log(f"RTSP処理中にエラーが発生しました: {exc}", "error")
            traceback.print_exc(file=sys.stderr)
            self._release_run_thread()
            self.event("run_state", {"state": "failed", "error": str(exc)})

    def _cancel(self) -> None:
        self._cancel_event.set()
        self.log("キャンセルを要求しました。")
        self.event("run_state", {"state": "cancelling"})

    def _safe_path(self, value: Any, fallback: Path) -> Path:
        if not isinstance(value, str) or not value.strip():
            return fallback
        path = Path(os.path.expanduser(value)).resolve()
        return path

    @staticmethod
    @contextmanager
    def _temporary_rtsp_config(config: Any, payload: Dict[str, Any]):
        presets = {
            "clear": {
                "min_line_length": 20,
                "hough_threshold": 25,
                "canny_thresh1": 75,
                "canny_thresh2": 180,
            },
            "cloudy": {
                "min_line_length": 25,
                "hough_threshold": 35,
                "canny_thresh1": 100,
                "canny_thresh2": 240,
            },
        }
        preset_name = payload.get("rtspPreset", "cloudy")
        preset = presets.get(preset_name, presets["cloudy"])
        config_preset_name = "RTSP_PRESET_CLEAR_SKY" if preset_name == "clear" else "RTSP_PRESET_CLOUDY"
        config_preset = getattr(config, config_preset_name, None)
        if isinstance(config_preset, dict):
            preset = config_preset
        values = {
            "RTSP_MIN_LINE_LENGTH": preset["min_line_length"],
            "RTSP_HOUGH_THRESHOLD": preset["hough_threshold"],
            "RTSP_CANNY_THRESH1": preset["canny_thresh1"],
            "RTSP_CANNY_THRESH2": preset["canny_thresh2"],
            "RTSP_FPS": int(payload.get("rtspFps", 25)),
        }
        missing = object()
        previous = {name: getattr(config, name, missing) for name in values}
        for name, value in values.items():
            setattr(config, name, value)
        try:
            yield
        finally:
            for name, value in previous.items():
                if value is missing:
                    delattr(config, name)
                else:
                    setattr(config, name, value)

    def _load_detection_mask(self, payload: Dict[str, Any]) -> Any:
        if not payload.get("applyMask", False):
            return None
        mask_path = self._safe_path(
            payload.get("maskPath"), self.root / "app_masks.npz"
        )
        if not mask_path.is_file():
            raise ValueError(f"検出マスクファイルが見つかりません: {mask_path}")
        try:
            import numpy as np

            with np.load(mask_path, allow_pickle=False) as archive:
                if "mask_image" not in archive.files:
                    raise ValueError("検出マスクファイルにmask_imageがありません")
                mask = archive["mask_image"]
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"検出マスクを読み込めませんでした: {exc}") from exc

        if getattr(mask, "ndim", 0) != 2 or getattr(mask, "size", 0) == 0:
            raise ValueError("検出マスクは空でない2次元画像である必要があります")
        height, width = (int(mask.shape[0]), int(mask.shape[1]))
        if height > 16_384 or width > 16_384 or int(mask.size) > 100_000_000:
            raise ValueError("検出マスクのサイズが大きすぎます")
        if mask.dtype == np.bool_:
            return mask.astype(np.uint8) * 255
        if not np.issubdtype(mask.dtype, np.number):
            raise ValueError("検出マスクの画素形式が不正です")
        if not np.isfinite(mask).all():
            raise ValueError("検出マスクに有限でない画素があります")
        return np.clip(mask, 0, 255).astype(np.uint8)

    def _save_mask_from_strokes(self, payload: Dict[str, Any]) -> str:
        """Rasterize SwiftUI's normalized brush strokes into a legacy NPZ mask."""
        output_value = payload.get("maskPath")
        if not isinstance(output_value, str) or not output_value.strip():
            raise ValueError("maskPath must be a non-empty string")
        output_path = self._safe_path(output_value, self.root / "app_masks.npz")
        if output_path.suffix.lower() != ".npz":
            raise ValueError("検出マスクの保存先は.npz形式で指定してください")
        width = self._validated_int(payload.get("width"), 1, 16_384, 1, "width")
        height = self._validated_int(payload.get("height"), 1, 16_384, 1, "height")
        if width * height > 100_000_000:
            raise ValueError("検出マスクのサイズが大きすぎます")
        brush_size = payload.get("brushSize", 0.03)
        if isinstance(brush_size, bool) or not isinstance(brush_size, (int, float)):
            raise ValueError("brushSize must be a number")
        brush_size = float(brush_size)
        if not math.isfinite(brush_size) or not 0.001 <= brush_size <= 0.5:
            raise ValueError("brushSize is out of range")
        raw_strokes = payload.get("strokes")
        if raw_strokes is None:
            raw_strokes = []
        if not isinstance(raw_strokes, list):
            raise ValueError("strokes must be an array")

        try:
            import cv2
            import numpy as np

            mask = np.full((height, width), 255, dtype=np.uint8)
            radius = max(1, int(round(min(width, height) * brush_size / 2.0)))
            for raw_stroke in raw_strokes:
                if not isinstance(raw_stroke, dict):
                    raise ValueError("strokesの形式が不正です")
                mode = raw_stroke.get("mode", "exclude")
                if mode not in {"exclude", "restore"}:
                    raise ValueError("strokesのmodeが不正です")
                raw_points = raw_stroke.get("points") or []
                if not isinstance(raw_points, list) or not raw_points:
                    continue
                points = []
                for raw_point in raw_points:
                    if (
                        not isinstance(raw_point, (list, tuple))
                        or len(raw_point) != 2
                    ):
                        raise ValueError("strokesの座標が不正です")
                    x = float(raw_point[0])
                    y = float(raw_point[1])
                    if not math.isfinite(x) or not math.isfinite(y):
                        raise ValueError("strokesの座標が不正です")
                    points.append(
                        [
                            max(0, min(width - 1, int(round(x * (width - 1))))),
                            max(0, min(height - 1, int(round(y * (height - 1))))),
                        ]
                    )
                color = 0 if mode == "exclude" else 255
                polyline = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
                if len(points) == 1:
                    cv2.circle(mask, tuple(points[0]), radius, color, -1)
                else:
                    cv2.polylines(mask, [polyline], False, color, radius * 2, cv2.LINE_AA)
                    for point in points:
                        cv2.circle(mask, tuple(point), radius, color, -1)

            # The legacy settings file can contain more than the detection
            # mask (for example ``plate_solve_mask_image``). Preserve every
            # readable companion array when editing an existing NPZ so the
            # SwiftUI editor cannot silently remove legacy state.
            preserved_arrays: Dict[str, Any] = {}
            if output_path.is_file():
                try:
                    with np.load(output_path, allow_pickle=False) as archive:
                        preserved_arrays = {
                            name: archive[name]
                            for name in archive.files
                            if name != "mask_image"
                        }
                except Exception as exc:
                    raise ValueError(
                        f"既存の検出マスクを読み込めないため上書きできません: {exc}"
                    ) from exc

            output_path.parent.mkdir(parents=True, exist_ok=True)
            file_descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{output_path.stem}.",
                suffix=".npz",
                dir=str(output_path.parent),
            )
            os.close(file_descriptor)
            try:
                preserved_arrays["mask_image"] = mask
                np.savez_compressed(temporary_name, **preserved_arrays)
                os.replace(temporary_name, output_path)
            finally:
                if os.path.exists(temporary_name):
                    try:
                        os.unlink(temporary_name)
                    except OSError:
                        pass
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"検出マスクを保存できませんでした: {exc}") from exc
        return str(output_path)

    def _load_selected_model(self, payload: Dict[str, Any]) -> None:
        """Apply the optional SwiftUI-selected classifier before processing starts."""
        raw_path = payload.get("modelPath")
        if not isinstance(raw_path, str) or not raw_path.strip():
            return

        model_path = self._validated_model_path(raw_path)
        try:
            import model
            import model_catalog

            metadata = model_catalog.load_model_metadata(model_path)
            ok, message = model.reload_model(model_path=model_path, metadata=metadata)
        except Exception as exc:
            raise ValueError(f"検出モデルを読み込めませんでした: {exc}") from exc
        if not ok:
            raise ValueError(f"検出モデルを読み込めませんでした: {message}")
        self.log(f"検出モデルを適用しました: {Path(model_path).name}")

    @staticmethod
    def _bounded_int(value: Any, lower: int, upper: int) -> int:
        try:
            return max(lower, min(upper, int(value)))
        except (TypeError, ValueError):
            return lower

    @staticmethod
    def _bounded_float(value: Any, lower: float, upper: float) -> float:
        try:
            return max(lower, min(upper, float(value)))
        except (TypeError, ValueError):
            return lower

    @staticmethod
    def _default_save_options() -> Dict[str, bool]:
        return {
            "video": True,
            "cutout": True,
            "full": False,
            "composite": True,
            "info": True,
            "summary": True,
            "full_video": False,
        }

    @staticmethod
    def _default_summary_config() -> List[Dict[str, Any]]:
        return [
            {"name": "Composite Image", "enabled": True, "duration": 1.0},
            {"name": "Annotated Image", "enabled": False, "duration": 2.0},
            {"name": "Full Size Video", "enabled": True},
            {"name": "Zoom Sequence", "enabled": False, "duration": 2.0},
            {"name": "Cutout Video", "enabled": True},
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Meteor Detector SwiftUI bridge")
    parser.add_argument("--stdio", action="store_true", help="serve JSON-lines requests")
    parser.add_argument("--root", type=Path, default=None, help="repository/application root")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.stdio:
        print("Use --stdio for the SwiftUI JSON bridge.", file=sys.stderr)
        return 2
    root = args.root or Path(os.environ.get("METEOR_DETECTOR_ROOT", Path.cwd()))
    process_lock = AppProcessLock(root)
    if not process_lock.acquire():
        print("Meteor Detector is already running.", file=sys.stderr)
        return 3
    protocol_stdout = sys.stdout
    sys.stdout = _NonProtocolStdout(sys.stderr)
    try:
        bridge = Bridge(root, protocol_stdout=protocol_stdout)
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("request must be an object")
                bridge.handle(request)
            except Exception as exc:
                bridge.log(f"リクエストを解釈できませんでした: {exc}", "error")
        return 0
    finally:
        process_lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
