"""
比較明合成動画作成モジュール

複数の動画ファイルから比較明合成動画を作成する。
動画の長さが異なる場合、最長の動画に合わせて処理し、
重なっているフレームは比較明、重なっていないフレームはそのまま残す。

メモリ使用量は最大1GBに制限される。
"""

import os
import gc
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Callable
import subprocess

# メモリ制限: 1GB = 1024 * 1024 * 1024 bytes
MAX_MEMORY_BYTES = 1 * 1024 * 1024 * 1024


def get_video_info(video_path: str) -> dict:
    """
    動画ファイルの情報を取得する。
    
    Args:
        video_path: 動画ファイルのパス
        
    Returns:
        dict: fps, frame_count, width, height を含む辞書
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    info = {
        'path': video_path,
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    }
    cap.release()
    return info


def calculate_frame_memory(width: int, height: int, num_videos: int) -> int:
    """
    1フレーム分の処理に必要なメモリを計算する。
    
    Args:
        width: フレーム幅
        height: フレーム高さ
        num_videos: 同時に処理する動画数
        
    Returns:
        int: 必要なメモリ（バイト）
    """
    # 1フレーム = width * height * 3 (BGR) bytes
    frame_size = width * height * 3
    # 各動画のフレーム + 合成用バッファ + 出力バッファ
    return frame_size * (num_videos + 2)


def create_lighten_blend_video(
    video_paths: List[str],
    output_path: str,
    progress_callback: Optional[Callable[[str], None]] = None
) -> bool:
    """
    複数の動画ファイルから比較明合成動画を作成する（メモリ効率版）。
    
    メモリ使用量を最大1GBに制限するため、動画数が多い場合は
    バッチ処理で段階的に合成を行う。
    
    動画の長さが異なる場合:
    - 最長の動画に合わせた長さの出力を生成
    - 各フレームで利用可能な全ての動画から比較明合成
    - 重なっていないフレームはそのまま使用
    
    Args:
        video_paths: 動画ファイルパスのリスト
        output_path: 出力動画のパス
        progress_callback: 進捗報告用コールバック関数
        
    Returns:
        bool: 成功したらTrue
    """
    if not video_paths:
        if progress_callback:
            progress_callback("動画ファイルが選択されていません。")
        return False
    
    if progress_callback:
        progress_callback(f"動画情報を取得中... ({len(video_paths)}個のファイル)")
    
    # 各動画の情報を取得
    video_infos = []
    for vp in video_paths:
        info = get_video_info(vp)
        if info and info['frame_count'] > 0:
            video_infos.append(info)
        else:
            if progress_callback:
                progress_callback(f"警告: 動画を読み込めません: {os.path.basename(vp)}")
    
    if not video_infos:
        if progress_callback:
            progress_callback("有効な動画ファイルがありませんでした。")
        return False
    
    # 最長フレーム数を取得
    max_frames = max(info['frame_count'] for info in video_infos)
    
    # 最初の動画の解像度を基準にする
    base_width = video_infos[0]['width']
    base_height = video_infos[0]['height']
    base_fps = video_infos[0]['fps']
    
    if base_fps <= 0 or base_fps > 120:
        base_fps = 30.0
    
    # メモリ計算: 1フレームあたりのメモリ使用量
    frame_memory = calculate_frame_memory(base_width, base_height, len(video_infos))
    
    # 1GBで同時に処理できる動画数を計算
    frame_size = base_width * base_height * 3
    max_concurrent_videos = max(1, (MAX_MEMORY_BYTES - frame_size * 2) // frame_size)
    
    if progress_callback:
        memory_per_frame_mb = frame_memory / (1024 * 1024)
        progress_callback(f"出力設定: {base_width}x{base_height}, {base_fps:.2f}fps, {max_frames}フレーム")
        progress_callback(f"メモリ使用量: 約{memory_per_frame_mb:.1f}MB/フレーム (制限: 1GB)")
    
    # 動画数がメモリ制限を超える場合はバッチ処理
    if len(video_infos) > max_concurrent_videos:
        if progress_callback:
            progress_callback(f"動画数({len(video_infos)})が多いためバッチ処理を使用します...")
        return _create_lighten_blend_video_batched(
            video_infos, output_path, base_width, base_height, base_fps, 
            max_frames, max_concurrent_videos, progress_callback
        )
    
    # 通常処理（メモリに収まる場合）
    return _create_lighten_blend_video_streaming(
        video_infos, output_path, base_width, base_height, base_fps,
        max_frames, progress_callback
    )


def _create_lighten_blend_video_streaming(
    video_infos: List[dict],
    output_path: str,
    base_width: int,
    base_height: int,
    base_fps: float,
    max_frames: int,
    progress_callback: Optional[Callable[[str], None]] = None
) -> bool:
    """
    ストリーミング方式で比較明合成動画を作成（メモリ効率版）。
    フレームを1つずつ読み込み、即座に合成して出力する。
    """
    # 全ての動画のVideoCaptureを開く
    caps = []
    for info in video_infos:
        cap = cv2.VideoCapture(info['path'])
        # バッファサイズを最小に設定してメモリ節約
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            caps.append({
                'cap': cap,
                'frame_count': info['frame_count'],
                'width': info['width'],
                'height': info['height']
            })
    
    if not caps:
        if progress_callback:
            progress_callback("動画ファイルを開けませんでした。")
        return False
    
    # FFMPEGを使用して動画を書き出す
    command = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{base_width}x{base_height}', '-pix_fmt', 'bgr24',
        '-r', str(base_fps), '-i', '-', '-an', '-c:v', 'libx264',
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
        for c in caps:
            c['cap'].release()
        return False
    except Exception as e:
        if progress_callback:
            progress_callback(f"エラー: ffmpegの起動に失敗しました: {e}")
        for c in caps:
            c['cap'].release()
        return False
    
    try:
        for frame_idx in range(max_frames):
            # 合成用のベースフレーム（最初のフレームで初期化）
            composite_frame = None
            frames_used = 0
            
            for cap_info in caps:
                cap = cap_info['cap']
                if frame_idx < cap_info['frame_count']:
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        # 解像度が異なる場合はリサイズ
                        if frame.shape[1] != base_width or frame.shape[0] != base_height:
                            frame = cv2.resize(frame, (base_width, base_height))
                        
                        if composite_frame is None:
                            composite_frame = frame
                        else:
                            # 比較明合成（その場で更新してメモリ節約）
                            np.maximum(composite_frame, frame, out=composite_frame)
                        
                        frames_used += 1
                        # 使用済みフレームの参照を削除
                        del frame
            
            if composite_frame is None:
                # フレームがない場合は黒フレーム
                composite_frame = np.zeros((base_height, base_width, 3), dtype=np.uint8)
            
            # FFMPEGに書き込み
            if proc.stdin:
                proc.stdin.write(composite_frame.tobytes())
            
            # 使用済みフレームを削除
            del composite_frame
            
            # 定期的にガベージコレクションを実行
            if frame_idx % 100 == 0:
                gc.collect()
            
            # 進捗報告（30フレームごと）
            if progress_callback and frame_idx % 30 == 0:
                progress = (frame_idx + 1) / max_frames * 100
                progress_callback(f"処理中: {frame_idx + 1}/{max_frames} フレーム ({progress:.1f}%)")
        
        if progress_callback:
            progress_callback(f"エンコード完了: {max_frames}フレーム処理しました。")
            
    except Exception as e:
        if progress_callback:
            progress_callback(f"エラー: フレーム処理中に問題が発生しました: {e}")
        return False
    finally:
        # リソース解放
        for c in caps:
            c['cap'].release()
        
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
        
        # 最終ガベージコレクション
        gc.collect()
    
    if progress_callback:
        progress_callback(f"比較明合成動画を保存しました: {output_path}")
    
    return True


def _create_lighten_blend_video_batched(
    video_infos: List[dict],
    output_path: str,
    base_width: int,
    base_height: int,
    base_fps: float,
    max_frames: int,
    batch_size: int,
    progress_callback: Optional[Callable[[str], None]] = None
) -> bool:
    """
    バッチ処理方式で比較明合成動画を作成。
    メモリ制限を超える数の動画がある場合に使用する。
    
    処理手順:
    1. 動画をバッチに分割
    2. 各バッチで中間ファイルを作成
    3. 中間ファイルを最終合成
    """
    import tempfile
    import shutil
    
    # 一時ディレクトリを作成
    temp_dir = tempfile.mkdtemp(prefix="lighten_blend_")
    intermediate_files = []
    
    try:
        # バッチに分割して処理
        num_batches = (len(video_infos) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(video_infos))
            batch_infos = video_infos[start_idx:end_idx]
            
            if progress_callback:
                progress_callback(f"バッチ {batch_idx + 1}/{num_batches} を処理中... ({len(batch_infos)}個の動画)")
            
            # 中間ファイルのパス
            intermediate_path = os.path.join(temp_dir, f"batch_{batch_idx}.mp4")
            intermediate_files.append(intermediate_path)
            
            # バッチを処理
            success = _create_lighten_blend_video_streaming(
                batch_infos, intermediate_path, base_width, base_height, base_fps,
                max_frames, None  # 中間処理では進捗を抑制
            )
            
            if not success:
                if progress_callback:
                    progress_callback(f"バッチ {batch_idx + 1} の処理に失敗しました。")
                return False
            
            # ガベージコレクション
            gc.collect()
        
        # 中間ファイルを最終合成
        if len(intermediate_files) == 1:
            # バッチが1つだけの場合はそのまま移動
            shutil.move(intermediate_files[0], output_path)
        else:
            if progress_callback:
                progress_callback("中間ファイルを最終合成中...")
            
            # 中間ファイルの情報を取得
            final_infos = [get_video_info(f) for f in intermediate_files]
            final_infos = [info for info in final_infos if info is not None]
            
            success = _create_lighten_blend_video_streaming(
                final_infos, output_path, base_width, base_height, base_fps,
                max_frames, progress_callback
            )
            
            if not success:
                return False
        
        if progress_callback:
            progress_callback(f"比較明合成動画を保存しました: {output_path}")
        
        return True
        
    finally:
        # 一時ディレクトリを削除
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass


def get_default_output_path() -> str:
    """
    デフォルトの出力パスを取得する（ダウンロードフォルダ + 日時）。
    
    Returns:
        str: デフォルトの出力ファイルパス
    """
    # Windowsのダウンロードフォルダを取得
    downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
    
    if not os.path.exists(downloads_path):
        # ダウンロードフォルダがない場合はデスクトップ
        downloads_path = os.path.join(os.path.expanduser("~"), "Desktop")
    
    # 日時でファイル名を生成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}.mp4"
    
    return os.path.join(downloads_path, filename)
