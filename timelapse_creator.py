"""
タイムラプス作成モジュール

フォルダ内の画像や動画ファイルからタイムラプス動画を作成する。
最終的な動画の長さを15秒、30秒、60秒から選択可能。

処理フロー:
1. まず総フレーム数を計算
2. 必要なフレーム数に基づいて等間隔でサンプリングするインデックスを計算
3. 並列処理でフレームを読み込み、ffmpegで動画を出力

メモリ制限: 最大1GB
"""

import os
import gc
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Callable, Tuple, Dict
import subprocess
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import config
import sys


_TIMESTAMP_POSITIONS = {
    "右下": "bottom_right", "左下": "bottom_left", "右上": "top_right", "左上": "top_left",
    "bottom_right": "bottom_right", "bottom_left": "bottom_left",
    "top_right": "top_right", "top_left": "top_left",
}


def _normalize_timestamp_settings(timestamp_settings: Optional[Dict]) -> Dict:
    settings = timestamp_settings or {}
    try:
        size_percent = float(settings.get("size_percent", config.TIMELAPSE_TIMESTAMP_SIZE_PERCENT))
    except (TypeError, ValueError):
        size_percent = config.TIMELAPSE_TIMESTAMP_SIZE_PERCENT
    return {
        "enabled": bool(settings.get("enabled", config.TIMELAPSE_TIMESTAMP_ENABLED)),
        "position": _TIMESTAMP_POSITIONS.get(
            str(settings.get("position", config.TIMELAPSE_TIMESTAMP_POSITION)),
            config.TIMELAPSE_TIMESTAMP_POSITION,
        ),
        "size_percent": max(0.8, min(4.0, size_percent)),
    }


def _source_created_datetime(path: str) -> datetime:
    """Use the source file's creation time, with mtime as a portable fallback."""
    stat = os.stat(path)
    return datetime.fromtimestamp(getattr(stat, "st_birthtime", stat.st_mtime))


def _draw_timestamp(frame: np.ndarray, timestamp: datetime, settings: Dict) -> np.ndarray:
    """Draw a small, readable timestamp in the configured frame corner."""
    if not settings["enabled"]:
        return frame
    output = frame.copy()
    height, width = output.shape[:2]
    if height <= 0 or width <= 0:
        return output

    text = timestamp.strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
    font = cv2.FONT_HERSHEY_SIMPLEX
    desired_height = max(12, int(round(height * settings["size_percent"] / 100.0)))
    unit_height = max(1, cv2.getTextSize("Ag", font, 1.0, 1)[0][1])
    font_scale = desired_height / unit_height
    thickness = max(1, int(round(desired_height / 18)))
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    margin = max(8, int(round(desired_height * 0.55)))
    position = settings["position"]
    x = margin if position.endswith("left") else width - text_width - margin
    y = text_height + margin if position.startswith("top") else height - margin
    x = max(0, min(x, max(0, width - text_width)))
    y = max(text_height, min(y, max(text_height, height - baseline)))

    padding = max(3, desired_height // 4)
    left, top = max(0, x - padding), max(0, y - text_height - padding)
    right, bottom = min(width, x + text_width + padding), min(height, y + baseline + padding)
    if right > left and bottom > top:
        roi = output[top:bottom, left:right]
        output[top:bottom, left:right] = cv2.addWeighted(roi, 0.45, np.zeros_like(roi), 0.55, 0)
    cv2.putText(output, text, (x, y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(output, text, (x, y), font, font_scale, (245, 245, 245), thickness, cv2.LINE_AA)
    return output


def _finish_ffmpeg_process(proc: subprocess.Popen) -> Tuple[int, bytes]:
    """Close stdin safely; do not call communicate after it has been closed."""
    if proc.stdin is not None and not proc.stdin.closed:
        try:
            proc.stdin.close()
        except (BrokenPipeError, OSError, ValueError):
            pass
    try:
        proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    try:
        stderr_output = proc.stderr.read() if proc.stderr is not None else b""
    except (OSError, ValueError):
        stderr_output = b""
    return proc.returncode if proc.returncode is not None else -1, stderr_output or b""

# サポートする画像・動画形式
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v'}

# 出力動画のFPS
OUTPUT_FPS = 60

# メモリ制限: 1GB (絶対に守る)
MAX_MEMORY_BYTES = 1 * 1024 * 1024 * 1024  # 1GB

# タイムラプスの各採用フレームは、この前後範囲の時間平均画像に置き換える。
TEMPORAL_MEAN_RADIUS_FRAMES = 50
# 平均算出時の一時配列を十分小さく保つための上限。画像は帯状に処理する。
TEMPORAL_MEAN_TILE_MEMORY_BYTES = 64 * 1024 * 1024
TEMPORAL_MEAN_AVAILABLE_MEMORY_FRACTION = 0.80


def get_available_memory_bytes() -> Optional[int]:
    """Return OS-reported available memory, preferring the platform API."""
    try:
        import psutil
        available = int(psutil.virtual_memory().available)
        if available > 0:
            return available
    except Exception:
        pass

    # psutil is normally bundled with the macOS runtime.  Keep a conservative
    # fallback for environments where it is unavailable.
    if sys.platform == "darwin":
        try:
            output = subprocess.check_output(["vm_stat"], text=True, timeout=3)
            page_size = 4096
            first_line = output.splitlines()[0]
            if "page size of" in first_line:
                page_size = int(first_line.split("page size of", 1)[1].split("bytes", 1)[0].strip())
            pages = 0
            for name in ("Pages free", "Pages inactive", "Pages speculative"):
                for line in output.splitlines():
                    if line.startswith(name):
                        pages += int(line.split(":", 1)[1].strip().rstrip("."))
                        break
            return pages * page_size if pages else None
        except Exception:
            return None
    return None


def _format_bytes(value: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    number = float(max(0, value))
    for unit in units:
        if number < 1024 or unit == units[-1]:
            return f"{number:.1f}{unit}"
        number /= 1024
    return f"{number:.1f}TB"

# NVENC利用可能フラグ（キャッシュ）
_nvenc_available: Optional[bool] = None


def is_nvenc_available() -> bool:
    """
    NVIDIA NVENCエンコーダー（h264_nvenc）が利用可能かチェックする。
    結果はキャッシュされる。
    """
    global _nvenc_available
    if _nvenc_available is not None:
        return _nvenc_available
    
    try:
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-encoders'],
            capture_output=True,
            text=True,
            timeout=10
        )
        _nvenc_available = 'h264_nvenc' in result.stdout
    except Exception:
        _nvenc_available = False
    
    return _nvenc_available


def get_files_from_path(path: str) -> Tuple[List[str], List[str]]:
    """
    パスから画像ファイルと動画ファイルのリストを取得する。
    """
    images = []
    videos = []
    
    if os.path.isfile(path):
        ext = Path(path).suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            images.append(path)
        elif ext in VIDEO_EXTENSIONS:
            videos.append(path)
    elif os.path.isdir(path):
        for ext in IMAGE_EXTENSIONS:
            pattern = os.path.join(path, f'*{ext}')
            images.extend(glob.glob(pattern, recursive=False))
            pattern_upper = os.path.join(path, f'*{ext.upper()}')
            images.extend(glob.glob(pattern_upper, recursive=False))
        
        for ext in VIDEO_EXTENSIONS:
            pattern = os.path.join(path, f'*{ext}')
            videos.extend(glob.glob(pattern, recursive=False))
            pattern_upper = os.path.join(path, f'*{ext.upper()}')
            videos.extend(glob.glob(pattern_upper, recursive=False))
    
    images = sorted(list(set(images)))
    videos = sorted(list(set(videos)))
    
    return images, videos


def get_video_frame_count(video_path: str) -> int:
    """動画のフレーム数を取得する。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(0, frame_count)


def count_total_frames(
    all_images: List[str],
    all_videos: List[str],
    progress_callback: Optional[Callable[[str], None]] = None
) -> Tuple[int, List[Tuple[str, int, int]]]:
    """総フレーム数をカウントし、各ソースの情報を返す。"""
    sources = []
    total_frames = 0
    
    for img_path in all_images:
        sources.append((img_path, total_frames, 1))
        total_frames += 1
    
    for i, video_path in enumerate(all_videos):
        if progress_callback and i % 10 == 0:
            progress_callback(f"動画情報を取得中: {i + 1}/{len(all_videos)}")
        
        frame_count = get_video_frame_count(video_path)
        if frame_count > 0:
            sources.append((video_path, total_frames, frame_count))
            total_frames += frame_count
    
    return total_frames, sources


def calculate_sample_indices(total_frames: int, target_duration_seconds: int) -> List[int]:
    """サンプリングするフレームのインデックスを計算する。"""
    target_frame_count = target_duration_seconds * OUTPUT_FPS
    
    if total_frames <= target_frame_count:
        return list(range(total_frames))
    
    step = total_frames / target_frame_count
    indices = [int(i * step) for i in range(target_frame_count)]
    
    return indices


def calculate_batch_size(frame_width: int, frame_height: int) -> int:
    """
    メモリ制限内で処理できるバッチサイズを計算する。
    1フレームのメモリ = width * height * 3 (BGR)
    """
    frame_size_bytes = frame_width * frame_height * 3
    
    # 安全マージンを取って、使用可能メモリの70%を使用
    available_memory = MAX_MEMORY_BYTES * 0.7
    
    # オーバーヘッド用に追加マージン（FFMPEGバッファ等）
    overhead_memory = 200 * 1024 * 1024  # 200MB
    usable_memory = available_memory - overhead_memory
    
    # バッチサイズを計算（最低1、最大でもCPUコア数×4）
    max_batch = max(1, int(usable_memory / frame_size_bytes))
    cpu_limit = min(os.cpu_count() or 4, 8) * 4  # 最大32
    
    batch_size = min(max_batch, cpu_limit)
    
    return max(1, batch_size)


class FrameLoader:
    """並列フレーム読み込み用クラス"""
    
    def __init__(self, sources: List[Tuple[str, int, int]]):
        self.sources = sources
        self._video_caps: Dict[str, cv2.VideoCapture] = {}
        self._video_positions: Dict[str, int] = {}
        self._video_fps: Dict[str, float] = {}
        self._source_times: Dict[str, datetime] = {}
        self._lock = threading.Lock()
    
    def _get_source_for_index(self, global_index: int) -> Optional[Tuple[str, int, int]]:
        """グローバルインデックスに対応するソースを探す"""
        for path, start_idx, frame_count in self.sources:
            if start_idx <= global_index < start_idx + frame_count:
                return (path, start_idx, frame_count)
        return None
    
    def load_frame(self, global_index: int, target_size: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        指定したグローバルインデックスのフレームを読み込む。
        
        Args:
            global_index: グローバルフレームインデックス
            target_size: (width, height) リサイズ先サイズ
            
        Returns:
            フレーム画像（BGRフォーマット）
        """
        source = self._get_source_for_index(global_index)
        if source is None:
            return None
        
        path, start_idx, frame_count = source
        local_index = global_index - start_idx
        
        frame = None
        
        if Path(path).suffix.lower() in IMAGE_EXTENSIONS:
            frame = cv2.imread(path)
        else:
            # 時間平均は順番に処理するため、VideoCaptureを再利用する。毎回
            # 開き直すより大幅に高速で、連続フレームはシークなしで読める。
            with self._lock:
                cap = self._video_caps.get(path)
                if cap is None or not cap.isOpened():
                    cap = cv2.VideoCapture(path)
                    self._video_caps[path] = cap
                    self._video_positions[path] = -1
                if cap.isOpened():
                    if self._video_positions.get(path, -1) + 1 != local_index:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, local_index)
                    ret, frame = cap.read()
                    if ret:
                        self._video_positions[path] = local_index
                    else:
                        frame = None
        
        if frame is not None and target_size[0] > 0 and target_size[1] > 0:
            if frame.shape[1] != target_size[0] or frame.shape[0] != target_size[1]:
                frame = cv2.resize(frame, target_size)
        
        return frame

    def load_frame_range(
        self,
        start_index: int,
        end_index: int,
        target_size: Tuple[int, int],
    ) -> Dict[int, np.ndarray]:
        """Decode one contiguous global-index range with a dedicated reader.

        This is intentionally independent from the rolling reader so several
        source ranges can be decoded in parallel during a full-memory preload.
        Each video range is read sequentially after one seek.
        """
        decoded: Dict[int, np.ndarray] = {}
        if end_index < start_index:
            return decoded
        for path, source_start, frame_count in self.sources:
            source_end = source_start + frame_count - 1
            read_start = max(start_index, source_start)
            read_end = min(end_index, source_end)
            if read_start > read_end:
                continue
            if Path(path).suffix.lower() in IMAGE_EXTENSIONS:
                frame = cv2.imread(path)
                if frame is not None:
                    if frame.shape[1] != target_size[0] or frame.shape[0] != target_size[1]:
                        frame = cv2.resize(frame, target_size)
                    decoded[read_start] = frame
                continue

            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                continue
            local_start = read_start - source_start
            cap.set(cv2.CAP_PROP_POS_FRAMES, local_start)
            for global_index in range(read_start, read_end + 1):
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if frame.shape[1] != target_size[0] or frame.shape[0] != target_size[1]:
                    frame = cv2.resize(frame, target_size)
                decoded[global_index] = frame
            cap.release()
        return decoded

    def timestamp_for_index(self, global_index: int) -> datetime:
        """Return capture time based on the input file creation time.

        For videos, each source frame advances from the file creation time by
        its native frame duration.  Images naturally keep their creation time.
        """
        source = self._get_source_for_index(global_index)
        if source is None:
            return datetime.now()
        path, start_idx, _ = source
        if path not in self._source_times:
            try:
                self._source_times[path] = _source_created_datetime(path)
            except OSError:
                self._source_times[path] = datetime.now()
        timestamp = self._source_times[path]
        if Path(path).suffix.lower() in VIDEO_EXTENSIONS:
            fps = self._video_fps.get(path)
            if fps is None:
                cap = cv2.VideoCapture(path)
                fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0.0
                cap.release()
                fps = fps if 0.1 <= fps <= 240.0 else OUTPUT_FPS
                self._video_fps[path] = fps
            timestamp += timedelta(seconds=(global_index - start_idx) / fps)
        return timestamp

    def load_temporal_mean_frame(
        self,
        global_index: int,
        target_size: Tuple[int, int],
        radius: int = TEMPORAL_MEAN_RADIUS_FRAMES,
    ) -> Optional[np.ndarray]:
        """Create an exact temporal mean using frames around ``global_index``.

        A full 101-frame 4K stack can require multiple gigabytes.  Each
        horizontal tile is therefore loaded and accumulated separately,
        keeping memory bounded while preserving the per-pixel temporal mean.
        """
        if not self.sources:
            return None
        total_frames = max(start + count for _, start, count in self.sources)
        start = max(0, global_index - radius)
        end = min(total_frames - 1, global_index + radius)
        window_indices = list(range(start, end + 1))
        if not window_indices:
            return None

        width, height = target_size
        if width <= 0 or height <= 0:
            return None
        bytes_per_row = max(1, len(window_indices) * width * 3)
        tile_height = max(1, min(height, TEMPORAL_MEAN_TILE_MEMORY_BYTES // bytes_per_row))
        mean_frame = np.empty((height, width, 3), dtype=np.uint8)

        for top in range(0, height, tile_height):
            bottom = min(height, top + tile_height)
            accumulated = np.zeros((bottom - top, width, 3), dtype=np.float64)
            valid_count = 0
            for index in window_indices:
                frame = self.load_frame(index, target_size)
                if frame is not None:
                    accumulated += frame[top:bottom]
                    valid_count += 1
            if valid_count == 0:
                return None
            mean_frame[top:bottom] = np.rint(accumulated / valid_count).astype(np.uint8)
            del accumulated
            gc.collect()

        return mean_frame

    def cleanup(self):
        """リソースを解放"""
        with self._lock:
            for cap in self._video_caps.values():
                cap.release()
            self._video_caps.clear()
            self._video_positions.clear()


class TemporalMeanFrameCache:
    """Rolling full-frame cache for fast temporal-mean generation.

    The cache retains the current ±N window and reuses its overlapping frames
    for the next output frame.  It is enabled only when its worst-case usage
    fits inside 80% of the memory that the OS currently reports as available.
    """

    def __init__(
        self,
        loader: FrameLoader,
        target_size: Tuple[int, int],
        radius: int,
        memory_budget_bytes: Optional[int],
    ):
        self.loader = loader
        self.target_size = target_size
        self.radius = radius
        self.memory_budget_bytes = memory_budget_bytes
        self.frames: Dict[int, np.ndarray] = {}
        width, height = target_size
        self.frame_bytes = max(1, width * height * 3)
        self.total_frames = max((start + count for _, start, count in loader.sources), default=0)
        maximum_window = min(self.total_frames, radius * 2 + 1)
        # Cached uint8 frames + float32 running sum + output/intermediate room.
        self.required_bytes = (maximum_window + 6) * self.frame_bytes
        self.full_preload_required_bytes = (self.total_frames + 6) * self.frame_bytes
        self.enabled = bool(
            memory_budget_bytes is not None and self.required_bytes <= memory_budget_bytes
        )
        self.full_preload_enabled = bool(
            memory_budget_bytes is not None
            and self.full_preload_required_bytes <= memory_budget_bytes
        )
        self._retain_all_frames = False
        self._window_start: Optional[int] = None
        self._window_end: Optional[int] = None
        self._running_sum = np.zeros((height, width, 3), dtype=np.float32)
        self._valid_count = 0

    def preload_all(self, progress_callback: Optional[Callable[[str], None]] = None) -> bool:
        """Read the complete source sequence once when it fits in the budget."""
        if not self.full_preload_enabled:
            return False

        worker_count = min(max(1, os.cpu_count() or 1), self.total_frames)
        chunk_size = max(1, (self.total_frames + worker_count - 1) // worker_count)
        ranges = [
            (start, min(self.total_frames - 1, start + chunk_size - 1))
            for start in range(0, self.total_frames, chunk_size)
        ]
        if progress_callback:
            progress_callback(f"時間平均用フレームを {worker_count} CPUワーカーで並列先読みします")

        requested = 0
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(self.loader.load_frame_range, start, end, self.target_size): (start, end)
                for start, end in ranges
            }
            for future in as_completed(futures):
                start, end = futures[future]
                requested += end - start + 1
                try:
                    self.frames.update(future.result())
                except Exception:
                    # Missing/corrupt chunks are allowed; mean_for_index uses
                    # the successfully decoded neighboring frames.
                    pass
                if progress_callback:
                    progress_callback(f"時間平均用フレームをメモリへ先読み中: {requested}/{self.total_frames}")
        self._retain_all_frames = True
        return bool(self.frames)

    def _get_or_load_frame(self, index: int) -> Optional[np.ndarray]:
        frame = self.frames.get(index)
        if frame is None:
            frame = self.loader.load_frame(index, self.target_size)
            if frame is not None:
                self.frames[index] = frame
        return frame

    def _add_index(self, index: int):
        frame = self._get_or_load_frame(index)
        if frame is not None:
            self._running_sum += frame
            self._valid_count += 1

    def _remove_index(self, index: int):
        frame = self.frames.get(index)
        if frame is not None:
            self._running_sum -= frame
            self._valid_count -= 1

    def mean_for_index(self, global_index: int) -> Optional[np.ndarray]:
        if not self.enabled:
            return self.loader.load_temporal_mean_frame(
                global_index, self.target_size, radius=self.radius
            )

        start = max(0, global_index - self.radius)
        end = min(self.total_frames - 1, global_index + self.radius)
        if self._window_start is None or self._window_end is None:
            for index in range(start, end + 1):
                self._add_index(index)
        else:
            # Sample indices are normally ascending.  Updating only the frames
            # that enter/leave the window makes each next mean O(1) in the
            # window size instead of summing all 101 frames again.
            for index in range(self._window_start, min(start, self._window_end + 1)):
                self._remove_index(index)
            for index in range(max(end + 1, self._window_start), self._window_end + 1):
                self._remove_index(index)
            for index in range(start, min(self._window_start, end + 1)):
                self._add_index(index)
            for index in range(max(self._window_end + 1, start), end + 1):
                self._add_index(index)

        self._window_start, self._window_end = start, end
        if self._valid_count <= 0:
            return None
        result = np.rint(self._running_sum / self._valid_count).astype(np.uint8)

        if not self._retain_all_frames:
            for index in tuple(self.frames):
                if index < start or index > end:
                    del self.frames[index]
        return result

    def clear(self):
        self.frames.clear()
        self._window_start = self._window_end = None
        self._valid_count = 0
        self._running_sum.fill(0)
        gc.collect()


def load_frame_wrapper(args):
    loader, idx, global_idx, target_size = args
    frame = loader.load_frame(global_idx, target_size)
    return (idx, frame)


def create_timelapse(
    input_paths: List[str],
    output_path: str,
    target_duration_seconds: int = 30,
    progress_callback: Optional[Callable[[str], None]] = None,
    mask: Optional[np.ndarray] = None,
    timestamp_settings: Optional[Dict] = None,
    temporal_mean_radius_frames: Optional[int] = None,
) -> bool:
    """
    タイムラプス動画を作成する（並列処理版）。
    
    処理手順:
    1. 総フレーム数を事前計算
    2. 必要なフレームのインデックスを等間隔で計算
    3. バッチ単位で並列にフレームを読み込み
    4. FFMPEGに順番に書き込み
    
    メモリ使用量は最大1GBに制限。
    """
    timestamp_settings = _normalize_timestamp_settings(timestamp_settings)
    try:
        temporal_mean_radius = int(
            config.TIMELAPSE_TEMPORAL_MEAN_RADIUS_FRAMES
            if temporal_mean_radius_frames is None else temporal_mean_radius_frames
        )
    except (TypeError, ValueError):
        temporal_mean_radius = config.TIMELAPSE_TEMPORAL_MEAN_RADIUS_FRAMES
    # A very wide range is valid but becomes disproportionately slow; keep a
    # generous safe limit while allowing the UI's 0-100 frame selection.
    temporal_mean_radius = max(0, min(300, temporal_mean_radius))

    if not input_paths:
        if progress_callback:
            progress_callback("入力パスが指定されていません。")
        return False
    
    if progress_callback:
        progress_callback(f"ファイルをスキャン中... ({len(input_paths)}個のパス)")
    
    # 全ての入力パスからファイルを収集
    all_images = []
    all_videos = []
    
    for path in input_paths:
        images, videos = get_files_from_path(path)
        all_images.extend(images)
        all_videos.extend(videos)
    
    all_images = sorted(list(set(all_images)))
    all_videos = sorted(list(set(all_videos)))
    
    if progress_callback:
        progress_callback(f"検出: 画像 {len(all_images)}枚, 動画 {len(all_videos)}本")
    
    if not all_images and not all_videos:
        if progress_callback:
            progress_callback("処理対象のファイルが見つかりませんでした。")
        return False
    
    # ステップ1: 総フレーム数を計算
    if progress_callback:
        progress_callback("総フレーム数を計算中...")
    
    total_frames, sources = count_total_frames(all_images, all_videos, progress_callback)
    
    if total_frames == 0:
        if progress_callback:
            progress_callback("フレームが見つかりませんでした。")
        return False
    
    if progress_callback:
        progress_callback(f"総フレーム数: {total_frames}")
    
    # ステップ2: サンプリングインデックスを計算
    target_frame_count = target_duration_seconds * OUTPUT_FPS
    sample_indices = calculate_sample_indices(total_frames, target_duration_seconds)
    
    if progress_callback:
        progress_callback(f"サンプリング: {len(sample_indices)}フレーム ({OUTPUT_FPS}fps × {target_duration_seconds}秒)")
        if total_frames > target_frame_count:
            interval = total_frames / len(sample_indices)
            progress_callback(f"サンプリング間隔: {interval:.2f}フレームごとに1フレーム抽出")
    
    # ステップ3: 最初の有効なフレームを取得して解像度を確認
    loader = FrameLoader(sources)
    first_frame = None
    first_valid_idx = 0
    
    # 最初の有効なフレームを探す（破損ファイルをスキップ）
    for try_idx, global_idx in enumerate(sample_indices[:min(len(sample_indices), 100)]):
        first_frame = loader.load_frame(global_idx, (0, 0))  # リサイズなし
        if first_frame is not None:
            first_valid_idx = try_idx
            if progress_callback and try_idx > 0:
                progress_callback(f"有効なフレームを発見: インデックス {global_idx} (試行 {try_idx + 1}回目)")
            break
    
    if first_frame is None:
        if progress_callback:
            progress_callback("有効なフレームを取得できませんでした。入力ファイルが破損している可能性があります。")
        loader.cleanup()
        return False
    
    base_height, base_width = first_frame.shape[:2]
    target_size = (base_width, base_height)

    if progress_callback:
        progress_callback(
            f"各採用フレームを前後{temporal_mean_radius}フレームの時間平均で作成します"
        )
    available_memory = get_available_memory_bytes()
    memory_budget = (
        int(available_memory * TEMPORAL_MEAN_AVAILABLE_MEMORY_FRACTION)
        if available_memory is not None else None
    )
    temporal_mean_cache = TemporalMeanFrameCache(
        loader, target_size, temporal_mean_radius, memory_budget
    )
    if progress_callback:
        if temporal_mean_cache.full_preload_enabled:
            progress_callback(
                f"全フレームキャッシュを使用します (必要量: 約"
                f"{_format_bytes(temporal_mean_cache.full_preload_required_bytes)})"
            )
        elif temporal_mean_cache.enabled:
            progress_callback(
                f"時間平均キャッシュ: 現在の空きメモリ {_format_bytes(available_memory)} の"
                f"{int(TEMPORAL_MEAN_AVAILABLE_MEMORY_FRACTION * 100)}%まで使用可能 "
                f"(必要量: 約{_format_bytes(temporal_mean_cache.required_bytes)})"
            )
        else:
            budget_text = _format_bytes(memory_budget) if memory_budget is not None else "取得不可"
            progress_callback(
                f"時間平均キャッシュは必要量 約{_format_bytes(temporal_mean_cache.required_bytes)} が"
                f"上限 {budget_text} を超えるため、低メモリ処理を使用します"
            )
    if temporal_mean_cache.full_preload_enabled:
        if not temporal_mean_cache.preload_all(progress_callback):
            temporal_mean_cache.full_preload_enabled = False
            temporal_mean_cache._retain_all_frames = False
            if progress_callback:
                progress_callback("警告: 全フレーム先読みを完了できないため、ローリングキャッシュへ切り替えます")
    mean_first_frame = temporal_mean_cache.mean_for_index(sample_indices[first_valid_idx])
    if mean_first_frame is None:
        if progress_callback:
            progress_callback("警告: 最初の時間平均を作成できないため、元フレームを使用します")
    else:
        first_frame = mean_first_frame
    
    # マスクをリサイズ（必要な場合）
    resized_mask = None
    if mask is not None:
        if mask.shape[:2] != (base_height, base_width):
            resized_mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        else:
            resized_mask = mask
        if progress_callback:
            progress_callback("マスクを適用します")
    
    # 最初のフレームにマスクを適用
    if resized_mask is not None:
        first_frame = cv2.bitwise_and(first_frame, first_frame, mask=resized_mask)

    if timestamp_settings["enabled"]:
        first_frame = _draw_timestamp(
            first_frame,
            loader.timestamp_for_index(sample_indices[first_valid_idx]),
            timestamp_settings,
        )
    
    # 有効なフレーム以降のサンプルインデックスを使用
    sample_indices = sample_indices[first_valid_idx:]
    
    if progress_callback:
        progress_callback(f"出力設定: {base_width}x{base_height}, {OUTPUT_FPS}fps")
        mode = "全フレーム先読み＋スライド平均" if temporal_mean_cache._retain_all_frames else "ローリングキャッシュ＋スライド平均"
        progress_callback(f"時間平均: 前後{temporal_mean_radius}フレーム、方式: {mode}")
    
    # FFMPEGを起動（GPU利用可能ならNVENCを使用）
    use_nvenc = is_nvenc_available()
    
    if use_nvenc:
        # NVIDIA GPU (NVENC) を使用 - RTX 4050等の場合高速エンコード
        if progress_callback:
            progress_callback("GPU エンコード (NVENC) を使用します")
        command = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{base_width}x{base_height}', '-pix_fmt', 'bgr24',
            '-r', str(OUTPUT_FPS), '-i', '-', '-an',
            '-c:v', 'h264_nvenc',
            '-preset', 'p5',  # p1(最速)～p7(最高品質), p5は高品質バランス
            '-rc', 'vbr',  # 可変ビットレート
            '-cq', '21',  # 品質レベル (0=ロスレス, 51=最低), 21は高品質
            '-b:v', '0',  # cqに任せる
            '-threads', '0',  # ffmpegに利用可能なCPUスレッド数を自動で全て使わせる
            '-pix_fmt', 'yuv420p',
            output_path
        ]
    else:
        # CPUエンコード (libx264)
        if progress_callback:
            progress_callback("CPU エンコード (libx264) を使用します")
        command = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{base_width}x{base_height}', '-pix_fmt', 'bgr24',
            '-r', str(OUTPUT_FPS), '-i', '-', '-an', '-c:v', 'libx264', '-threads', '0',
            '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p',
            output_path
        ]
    
    try:
        proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
    except FileNotFoundError:
        if progress_callback:
            progress_callback("エラー: ffmpegが見つかりません。ffmpegをインストールしてください。")
        temporal_mean_cache.clear()
        loader.cleanup()
        return False
    except Exception as e:
        if progress_callback:
            progress_callback(f"エラー: ffmpegの起動に失敗しました: {e}")
        temporal_mean_cache.clear()
        loader.cleanup()
        return False
    
    try:
        # 最初のフレームを書き込み
        if proc.stdin:
            proc.stdin.write(first_frame.tobytes())
        del first_frame
        gc.collect()
        
        # 各採用フレームを時間平均に置き換える。複数の101フレーム
        # ウィンドウを並列実行するとメモリが急増するため、ここは順番に処理する。
        remaining_indices = sample_indices[1:]
        total_output = len(sample_indices)
        processed = 1

        for global_idx in remaining_indices:
            frame = temporal_mean_cache.mean_for_index(global_idx)
            if frame is None:
                if progress_callback:
                    progress_callback(f"警告: 時間平均を作成できませんでした: フレーム {global_idx}")
            elif proc.stdin:
                if resized_mask is not None:
                    frame = cv2.bitwise_and(frame, frame, mask=resized_mask)
                if timestamp_settings["enabled"]:
                    frame = _draw_timestamp(
                        frame,
                        loader.timestamp_for_index(global_idx),
                        timestamp_settings,
                    )
                proc.stdin.write(frame.tobytes())
            processed += 1
            gc.collect()

            if progress_callback:
                progress = processed / max(1, total_output) * 100
                progress_callback(f"処理中: {processed}/{total_output} フレーム ({progress:.1f}%)")
        
        return_code, stderr_output = _finish_ffmpeg_process(proc)

        if return_code != 0:
            if progress_callback:
                progress_callback("警告: FFMPEGがエラーを返しました。")
                try:
                    error_msg = stderr_output.decode('utf-8', errors='ignore')[-500:]
                    progress_callback(f"FFMPEGエラー: {error_msg}")
                except:
                    pass
            return False
        
    except Exception as e:
        if progress_callback:
            progress_callback(f"エラー: フレーム処理中に問題が発生しました: {e}")
        return False
    finally:
        # 途中で書き込みに失敗した場合もstdinを閉じ、ffmpegを確実に回収する。
        # これを行わないと子プロセスが残り、次回の作成時に不安定になる。
        if (proc.stdin is not None and not proc.stdin.closed) or proc.poll() is None:
            _finish_ffmpeg_process(proc)
        temporal_mean_cache.clear()
        loader.cleanup()
        gc.collect()
    
    if progress_callback:
        progress_callback(f"タイムラプス動画を保存しました: {output_path}")
    
    return True


def get_default_output_path() -> str:
    """デフォルトの出力パスを取得する（ダウンロードフォルダ + 日時）。"""
    downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
    
    if not os.path.exists(downloads_path):
        downloads_path = os.path.join(os.path.expanduser("~"), "Desktop")
    
    now = datetime.now()
    date = now.strftime("%Y%m%d")
    time = now.strftime("%H%M%S")
    filename = f"{date}_timelapse_{time}.mp4"
    
    return os.path.join(downloads_path, filename)
