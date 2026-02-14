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
from datetime import datetime
from typing import List, Optional, Callable, Tuple, Dict
import subprocess
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# サポートする画像・動画形式
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v'}

# 出力動画のFPS
OUTPUT_FPS = 60

# メモリ制限: 1GB (絶対に守る)
MAX_MEMORY_BYTES = 1 * 1024 * 1024 * 1024  # 1GB

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
        
        # 画像ファイルの場合
        if Path(path).suffix.lower() in IMAGE_EXTENSIONS:
            frame = cv2.imread(path)
        else:
            # 動画ファイルの場合 - 各スレッドで独自のVideoCaptureを使用
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, local_index)
                ret, frame = cap.read()
                if not ret:
                    frame = None
                cap.release()
        
        # リサイズが必要な場合（target_sizeが有効な場合のみ）
        if frame is not None and target_size[0] > 0 and target_size[1] > 0:
            if frame.shape[1] != target_size[0] or frame.shape[0] != target_size[1]:
                frame = cv2.resize(frame, target_size)
        
        return frame
    
    def cleanup(self):
        """リソースを解放"""
        with self._lock:
            for cap in self._video_caps.values():
                cap.release()
            self._video_caps.clear()


def load_frame_wrapper(args):
    """ThreadPoolExecutor用のラッパー関数"""
    loader, idx, global_idx, target_size = args
    frame = loader.load_frame(global_idx, target_size)
    return (idx, frame)


def create_timelapse(
    input_paths: List[str],
    output_path: str,
    target_duration_seconds: int = 30,
    progress_callback: Optional[Callable[[str], None]] = None,
    mask: Optional[np.ndarray] = None
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
    
    # 有効なフレーム以降のサンプルインデックスを使用
    sample_indices = sample_indices[first_valid_idx:]
    
    # バッチサイズを計算（メモリ制限内）
    batch_size = calculate_batch_size(base_width, base_height)
    num_workers = min(batch_size, os.cpu_count() or 4)
    
    frame_size_mb = (base_width * base_height * 3) / (1024 * 1024)
    estimated_memory_mb = frame_size_mb * batch_size
    
    if progress_callback:
        progress_callback(f"出力設定: {base_width}x{base_height}, {OUTPUT_FPS}fps")
        progress_callback(f"並列処理: バッチサイズ={batch_size}, ワーカー数={num_workers}")
        progress_callback(f"推定メモリ使用量: {estimated_memory_mb:.1f}MB (制限: 1024MB)")
    
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
            '-r', str(OUTPUT_FPS), '-i', '-', '-an', '-c:v', 'libx264',
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
        loader.cleanup()
        return False
    except Exception as e:
        if progress_callback:
            progress_callback(f"エラー: ffmpegの起動に失敗しました: {e}")
        loader.cleanup()
        return False
    
    try:
        # 最初のフレームを書き込み
        if proc.stdin:
            proc.stdin.write(first_frame.tobytes())
        del first_frame
        gc.collect()
        
        # バッチ単位で並列処理
        remaining_indices = sample_indices[1:]
        total_remaining = len(remaining_indices)
        processed = 0
        
        for batch_start in range(0, len(remaining_indices), batch_size):
            batch_end = min(batch_start + batch_size, len(remaining_indices))
            batch_indices = remaining_indices[batch_start:batch_end]
            
            # バッチ内のフレームを並列で読み込み
            batch_frames = [None] * len(batch_indices)
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # タスクを投入
                futures = []
                for i, global_idx in enumerate(batch_indices):
                    future = executor.submit(
                        loader.load_frame, 
                        global_idx, 
                        target_size
                    )
                    futures.append((i, future))
                
                # 結果を収集
                for i, future in futures:
                    try:
                        frame = future.result(timeout=30)
                        batch_frames[i] = frame
                    except Exception as e:
                        pass  # エラーフレームはNoneのまま
            
            # バッチ内のフレームを順番にFFMPEGに書き込み
            for frame in batch_frames:
                if frame is not None and proc.stdin:
                    # マスクを適用
                    if resized_mask is not None:
                        frame = cv2.bitwise_and(frame, frame, mask=resized_mask)
                    proc.stdin.write(frame.tobytes())
                processed += 1
            
            # バッチのメモリを解放
            del batch_frames
            gc.collect()
            
            # 進捗報告
            if progress_callback:
                progress = (processed + 1) / total_remaining * 100
                progress_callback(f"処理中: {processed + 1}/{total_remaining + 1} フレーム ({progress:.1f}%)")
        
        if proc.stdin:
            proc.stdin.close()
        
        stderr_output = proc.communicate()[1]
        
        if proc.returncode != 0:
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
