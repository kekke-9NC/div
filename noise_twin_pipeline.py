"""Disk-backed three-stage RTSP pipeline for NoiseTwin.

Capture never waits for MPS or meteor analysis. Completed one-minute source
segments are queued on disk, transformed by exactly one NoiseTwin worker, and
then handed to a separate analysis worker.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time
from typing import Callable, Optional

import cv2

import noise_twin
import temporal_mean
import video_encoding


StatusCallback = Optional[Callable[[str], None]]
AnalyzeCallback = Callable[[str, str], bool]


@dataclass(frozen=True)
class PipelineResourcePlan:
    capture_processes: int = 1
    noise_twin_workers: int = 1
    noise_twin_cpu_threads: int = 8
    analysis_workers: int = 1
    ui_reserved_cores: int = 2

    @classmethod
    def for_host(cls) -> "PipelineResourcePlan":
        cores = max(1, os.cpu_count() or 1)
        ui = 2 if cores >= 8 else 1
        # MPS inference is deliberately singular. Remaining CPU capacity is
        # shared by OpenCV, the codec writer, analysis and the UI.
        twin_threads = max(2, min(8, cores - ui - 2))
        return cls(noise_twin_cpu_threads=twin_threads, ui_reserved_cores=ui)


class RtspNoiseTwinPipeline:
    def __init__(
        self,
        rtsp_url: str,
        save_root: str,
        model_path: str,
        analyze_callback: AnalyzeCallback,
        cancel_event: threading.Event,
        correction=None,
        segment_seconds: int = 60,
        require_validated: bool = True,
        status_callback: StatusCallback = None,
        recording_allowed: Optional[Callable[[], bool]] = None,
        preview_callback=None,
        resource_plan: Optional[PipelineResourcePlan] = None,
        temporal_mean_frames: int = 0,
        encoding_options: Optional[dict] = None,
    ):
        self.rtsp_url = rtsp_url
        self.save_root = Path(save_root)
        self.spool_root = self.save_root / ".noise_twin_spool"
        self.work_root = self.spool_root / ".work"
        self.model_path = model_path
        self.analyze_callback = analyze_callback
        self.cancel_event = cancel_event
        self.correction = correction
        self.segment_seconds = max(10, int(segment_seconds))
        self.require_validated = require_validated
        self.status_callback = status_callback
        self.recording_allowed = recording_allowed or (lambda: True)
        self.preview_callback = preview_callback
        self.resource_plan = resource_plan or PipelineResourcePlan.for_host()
        self.temporal_mean_frames = temporal_mean.normalize_window(temporal_mean_frames)
        self.encoding_settings = video_encoding.EncodingSettings.from_value(encoding_options)
        if not self.model_path and not self.temporal_mean_frames:
            raise ValueError("NoiseTwinモデルまたは3/5フレーム平均が必要です。")
        self.raw_queue: queue.Queue[str] = queue.Queue()
        self.analysis_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.capture_running = threading.Event()
        self.enqueued: set[str] = set()
        self.retry_counts: dict[str, int] = {}
        self.analysis_retry_counts: dict[str, int] = {}
        self.threads: list[threading.Thread] = []
        self.capture_process: Optional[subprocess.Popen] = None
        self._process_lock = threading.Lock()

    def _status(self, message: str) -> None:
        text = f"[NoiseTwin Pipeline] {message}"
        print(text)
        if self.status_callback:
            self.status_callback(text)

    def _capture_directory(self, when: Optional[datetime] = None) -> Path:
        current = when or datetime.now()
        directory = self.spool_root / current.strftime("%Y%m%d") / current.strftime("%H")
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _ffmpeg_command(self, capture_directory: Optional[Path] = None) -> list[str]:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise noise_twin.NoiseTwinError("RTSPパイプラインにはFFmpegが必要です。")
        # Some FFmpeg builds (including the Homebrew build used by the app)
        # do not provide -strftime_mkdir. Create the date/hour directory here
        # and only let FFmpeg expand the filename within that directory.
        directory = capture_directory or self._capture_directory()
        directory.mkdir(parents=True, exist_ok=True)
        pattern = str(directory / "%M_%S.mp4")
        command = [ffmpeg, "-y", "-loglevel", "warning"]
        command.extend(("-rtsp_transport", "tcp", "-i", self.rtsp_url))
        command.extend(
            (
                "-map", "0:v:0", "-an", "-c:v", "copy",
                "-f", "segment", "-segment_time", str(self.segment_seconds),
                "-segment_atclocktime", "1", "-reset_timestamps", "1",
                "-segment_format", "mp4", "-strftime", "1",
                pattern,
            )
        )
        return command

    @staticmethod
    def _drain_stderr(stream, tail: deque[str]) -> None:
        if stream is None:
            return
        try:
            for line in stream:
                message = line.strip()
                if message:
                    tail.append(message)
        finally:
            stream.close()

    def _capture_loop(self) -> None:
        while not self.cancel_event.is_set():
            if not self.recording_allowed():
                time.sleep(1)
                continue
            try:
                free_bytes = shutil.disk_usage(self.spool_root).free
                if free_bytes < 5 * 1024**3:
                    self._status(
                        "空き容量が5GB未満のため受信を一時停止しました。"
                        "変換・分析キューの解消を待ちます。"
                    )
                    time.sleep(30)
                    continue
            except OSError:
                pass
            try:
                started_at = datetime.now()
                capture_directory = self._capture_directory(started_at)
                capture_hour = started_at.strftime("%Y%m%d%H")
                command = self._ffmpeg_command(capture_directory)
                self._status(
                    f"RTSP stream-copy受信を開始しました（{self.segment_seconds}秒セグメント）。"
                )
                stderr_tail: deque[str] = deque(maxlen=20)
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
                stderr_thread = threading.Thread(
                    target=self._drain_stderr,
                    args=(process.stderr, stderr_tail),
                    name="rtsp-ffmpeg-stderr",
                    daemon=True,
                )
                stderr_thread.start()
                with self._process_lock:
                    self.capture_process = process
                self.capture_running.set()
                hour_rollover = False
                while process.poll() is None and not self.cancel_event.is_set():
                    if not self.recording_allowed():
                        process.terminate()
                        break
                    if datetime.now().strftime("%Y%m%d%H") != capture_hour:
                        hour_rollover = True
                        process.terminate()
                        break
                    time.sleep(0.5)
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                stderr_thread.join(timeout=1)
                if not self.cancel_event.is_set() and self.recording_allowed():
                    if hour_rollover:
                        self._status("保存先の時刻が切り替わったため受信を継続します。")
                    else:
                        detail = " / ".join(list(stderr_tail)[-3:])
                        suffix = f" FFmpeg: {detail}" if detail else ""
                        self._status(f"受信が終了したため2秒後に再接続します。{suffix}")
                        time.sleep(2)
            except Exception as exc:
                self._status(f"受信エラー: {exc}。2秒後に再接続します。")
                time.sleep(2)
            finally:
                self.capture_running.clear()
                with self._process_lock:
                    self.capture_process = None

    def _completed_raw_segments(self) -> list[Path]:
        candidates = [
            path for path in self.spool_root.rglob("*.mp4")
            if self.work_root not in path.parents and path.is_file()
        ]
        candidates.sort(key=lambda path: path.stat().st_mtime)
        if not candidates:
            return []
        newest = candidates[-1]
        now = time.time()
        completed = []
        for path in candidates:
            key = str(path)
            if key in self.enqueued:
                continue
            # While FFmpeg is alive, its newest segment is considered open.
            if path == newest and self.capture_running.is_set():
                continue
            try:
                if now - path.stat().st_mtime < 2.0:
                    continue
                if path.stat().st_size < 10_000:
                    continue
            except OSError:
                continue
            completed.append(path)
        return completed

    def _spool_loop(self) -> None:
        while not self.cancel_event.is_set():
            for path in self._completed_raw_segments():
                key = str(path)
                self.enqueued.add(key)
                self.raw_queue.put(key)
                self._status(
                    f"受信完了: {path.name} / NoiseTwin待ち={self.raw_queue.qsize()}"
                )
            time.sleep(1)

    def _final_paths(self, raw_path: Path) -> tuple[Path, Path]:
        relative = raw_path.relative_to(self.spool_root)
        final_video = self.save_root / relative
        final_evidence = final_video.with_name(final_video.stem + "_innovation.mp4")
        return final_video, final_evidence

    def _write_failure(self, raw_path: Path, exc: BaseException) -> None:
        marker = raw_path.with_suffix(raw_path.suffix + ".error.json")
        marker.write_text(
            json.dumps(
                {"error": str(exc), "time": time.time(), "attempts": self.retry_counts.get(str(raw_path), 0)},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _denoise_loop(self) -> None:
        try:
            cv2.setNumThreads(self.resource_plan.noise_twin_cpu_threads)
        except Exception:
            pass
        while not self.cancel_event.is_set():
            try:
                raw_value = self.raw_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            raw_path = Path(raw_value)
            try:
                transform_label = (
                    f"{self.temporal_mean_frames}フレーム平均"
                    if self.temporal_mean_frames
                    else "NoiseTwin"
                )
                self._status(
                    f"{transform_label}開始: {raw_path.name} / "
                    f"分析待ち={self.analysis_queue.qsize()}"
                )
                self.work_root.mkdir(parents=True, exist_ok=True)
                final_video, final_evidence = self._final_paths(raw_path)
                final_video.parent.mkdir(parents=True, exist_ok=True)
                if self.temporal_mean_frames:
                    prepared_mean = temporal_mean.prepare_video(
                        str(raw_path),
                        self.temporal_mean_frames,
                        correction=self.correction,
                        temp_dir=str(self.work_root),
                        encoding_options=dict(vars(self.encoding_settings)),
                    )
                    os.replace(prepared_mean.video_path, final_video)
                    final_evidence_value = ""
                    temporal_mean.write_processing_marker(
                        str(final_video), self.temporal_mean_frames, analyzed=False
                    )
                else:
                    prepared = noise_twin.prepare_video(
                        str(raw_path),
                        self.model_path,
                        correction=self.correction,
                        temp_dir=str(self.work_root),
                        require_validated=self.require_validated,
                        encoding_options=dict(vars(self.encoding_settings)),
                    )
                    os.replace(prepared.video_path, final_video)
                    os.replace(prepared.innovation_path, final_evidence)
                    metadata = noise_twin.load_metadata(self.model_path)
                    noise_twin.write_processing_marker(final_video, metadata)
                    final_evidence_value = str(final_evidence)
                if self.preview_callback is not None:
                    preview_cap = cv2.VideoCapture(str(final_video))
                    preview_ok, preview_frame = preview_cap.read()
                    preview_cap.release()
                    if preview_ok and preview_frame is not None:
                        try:
                            self.preview_callback(preview_frame)
                        except Exception:
                            pass
                try:
                    raw_path.unlink()
                except OSError:
                    pass
                error_marker = raw_path.with_suffix(raw_path.suffix + ".error.json")
                try:
                    error_marker.unlink()
                except OSError:
                    pass
                self.analysis_queue.put((str(final_video), final_evidence_value))
                self._status(
                    f"{transform_label}完了: {final_video.name} / "
                    f"分析待ち={self.analysis_queue.qsize()}"
                )
            except Exception as exc:
                key = str(raw_path)
                attempts = self.retry_counts.get(key, 0) + 1
                self.retry_counts[key] = attempts
                self._write_failure(raw_path, exc)
                self._status(f"変換失敗 ({attempts}/3): {raw_path.name}: {exc}")
                if attempts < 3 and not self.cancel_event.is_set():
                    time.sleep(min(10, attempts * 2))
                    self.raw_queue.put(key)
            finally:
                self.raw_queue.task_done()

    def _analysis_loop(self) -> None:
        while not self.cancel_event.is_set():
            try:
                video_path, evidence_path = self.analysis_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._status(f"流星分析開始: {Path(video_path).name}")
                success = bool(self.analyze_callback(video_path, evidence_path))
                if success:
                    self.analysis_retry_counts.pop(video_path, None)
                    if evidence_path:
                        try:
                            os.remove(evidence_path)
                        except OSError:
                            pass
                    mean_marker = Path(video_path).with_suffix(
                        Path(video_path).suffix + ".preprocessed.json"
                    )
                    if mean_marker.exists():
                        try:
                            marker_data = json.loads(mean_marker.read_text(encoding="utf-8"))
                            marker_data["analyzed"] = True
                            mean_marker.write_text(
                                json.dumps(marker_data, ensure_ascii=False, indent=2),
                                encoding="utf-8",
                            )
                        except (OSError, json.JSONDecodeError):
                            pass
                    self._status(f"流星分析完了: {Path(video_path).name}")
                else:
                    attempts = self.analysis_retry_counts.get(video_path, 0) + 1
                    self.analysis_retry_counts[video_path] = attempts
                    self._status(
                        f"流星分析失敗 ({attempts}/3): {Path(video_path).name}"
                    )
                    if attempts < 3 and not self.cancel_event.is_set():
                        time.sleep(min(10, attempts * 2))
                        self.analysis_queue.put((video_path, evidence_path))
            except Exception as exc:
                attempts = self.analysis_retry_counts.get(video_path, 0) + 1
                self.analysis_retry_counts[video_path] = attempts
                self._status(
                    f"流星分析エラー ({attempts}/3): {Path(video_path).name}: {exc}"
                )
                if attempts < 3 and not self.cancel_event.is_set():
                    time.sleep(min(10, attempts * 2))
                    self.analysis_queue.put((video_path, evidence_path))
            finally:
                self.analysis_queue.task_done()

    def start(self) -> None:
        self.spool_root.mkdir(parents=True, exist_ok=True)
        # Resume analysis after an app restart when denoising had completed but
        # the transient innovation sidecar had not yet been consumed.
        for evidence in self.save_root.rglob("*_innovation.mp4"):
            if self.spool_root in evidence.parents:
                continue
            video = evidence.with_name(evidence.name.replace("_innovation.mp4", ".mp4"))
            if video.exists():
                self.analysis_queue.put((str(video), str(evidence)))
        for marker in self.save_root.rglob("*.mp4.preprocessed.json"):
            if self.spool_root in marker.parents:
                continue
            try:
                data = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if data.get("method") != "temporal_mean" or data.get("analyzed") is True:
                continue
            video = Path(str(marker)[: -len(".preprocessed.json")])
            if video.exists():
                self.analysis_queue.put((str(video), ""))
        plan = self.resource_plan
        method = (
            f"時間平均{self.temporal_mean_frames}フレーム"
            if self.temporal_mean_frames
            else "NoiseTwin MPS"
        )
        self._status(
            f"リソース配分: 受信=FFmpeg copy 1、変換={method} 1 + "
            f"CPU {plan.noise_twin_cpu_threads}、分析={plan.analysis_workers}、"
            f"UI予約={plan.ui_reserved_cores}コア"
        )
        estimated = video_encoding.estimated_megabytes_per_minute(self.encoding_settings)
        if self.encoding_settings.quality == "source":
            size_text = "各入力セグメントから自動算出"
        else:
            size_text = "容量は映像内容依存" if estimated is None else f"目安 {estimated:.0f} MB/分"
        self._status(f"処理済み動画の保存: {self.encoding_settings.label}（{size_text}）")
        targets = (
            ("NoiseTwinCapture", self._capture_loop),
            ("NoiseTwinSpool", self._spool_loop),
            ("NoiseTwinTransform", self._denoise_loop),
            ("NoiseTwinAnalysis", self._analysis_loop),
        )
        self.threads = [
            threading.Thread(target=target, name=name, daemon=True)
            for name, target in targets
        ]
        for thread in self.threads:
            thread.start()

    def stop(self) -> None:
        self.cancel_event.set()
        with self._process_lock:
            process = self.capture_process
        if process is not None and process.poll() is None:
            process.terminate()
        for thread in self.threads:
            thread.join(timeout=10)
        self._status("停止しました。未処理原画はスプールへ保持されています。")

    def run(self) -> None:
        self.start()
        try:
            while not self.cancel_event.is_set():
                time.sleep(0.5)
        finally:
            self.stop()
