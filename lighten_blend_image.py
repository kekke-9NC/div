"""
比較明合成画像作成モジュール

複数の画像・動画ファイルから比較明合成画像を作成する。
メモリ使用量は最大1GBに制限される。
"""

import os
import gc
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Callable
import config


def _imread_unicode(path: str):
    """Read image with Unicode path support using np.fromfile + imdecode."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass
    return cv2.imread(path)


# メモリ制限: 1GB = 1024 * 1024 * 1024 bytes
MAX_MEMORY_BYTES = 1 * 1024 * 1024 * 1024


def get_supported_extensions() -> tuple:
    """サポートされている画像と動画の拡張子を返す"""
    image_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    video_ext = ('.mp4', '.avi', '.mov', '.mkv', '.wmv')
    return image_ext, video_ext


def get_default_output_path(base_folder: str = None) -> str:
    """
    デフォルトの出力パスを取得する。
    
    Args:
        base_folder: ベースフォルダ（指定がない場合はダウンロードフォルダ）
        
    Returns:
        str: デフォルトの出力ファイルパス
    """
    if base_folder and os.path.exists(base_folder):
        output_dir = base_folder
    else:
        # ダウンロードフォルダ
        output_dir = os.path.join(os.path.expanduser("~"), "Downloads")
        if not os.path.exists(output_dir):
            output_dir = os.path.join(os.path.expanduser("~"), "Desktop")
    
    now = datetime.now()
    date = now.strftime("%Y%m%d")
    time = now.strftime("%H%M%S")
    filename = f"{date}_lighten_blend_{time}.png"
    
    return os.path.join(output_dir, filename)


def collect_files_from_folder(folder_path: str) -> List[str]:
    """
    フォルダから画像と動画ファイルを収集する。
    
    Args:
        folder_path: フォルダパス
        
    Returns:
        List[str]: ファイルパスのリスト
    """
    image_ext, video_ext = get_supported_extensions()
    all_ext = image_ext + video_ext
    
    files = []
    folder = Path(folder_path)
    
    if folder.is_file():
        if folder.suffix.lower() in all_ext:
            files.append(str(folder))
    elif folder.is_dir():
        for f in sorted(folder.rglob('*')):
            if f.is_file() and f.suffix.lower() in all_ext:
                files.append(str(f))
    
    return files


def estimate_memory_usage(width: int, height: int, num_concurrent: int = 2) -> int:
    """
    メモリ使用量を推定する。
    
    Args:
        width: 画像幅
        height: 画像高さ
        num_concurrent: 同時に保持するフレーム数
        
    Returns:
        int: 推定メモリ使用量（バイト）
    """
    # 1フレーム = width * height * 3 (BGR) bytes
    frame_size = width * height * 3
    return frame_size * num_concurrent


def get_frame_from_video(video_path: str, frame_index: int = 0) -> Optional[np.ndarray]:
    """
    動画から指定フレームを取得する。
    
    Args:
        video_path: 動画ファイルパス
        frame_index: フレームインデックス (0 = 最初のフレーム)
        
    Returns:
        フレーム画像 or None
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    if frame_index > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    
    ret, frame = cap.read()
    cap.release()
    
    return frame if ret else None


def extract_frames_from_video(
    video_path: str,
    step: int = 1,
    max_frames: int = None,
    progress_callback: Optional[Callable[[str], None]] = None
) -> List[np.ndarray]:
    """
    動画からフレームを抽出する（メモリ制限付き）。
    
    Args:
        video_path: 動画ファイルパス
        step: フレーム間隔
        max_frames: 最大フレーム数
        progress_callback: 進捗コールバック
        
    Returns:
        フレームのリスト
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    frames = []
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while True:
        if max_frames and len(frames) >= max_frames:
            break
            
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % step == 0:
            frames.append(frame)
        
        frame_idx += 1
    
    cap.release()
    return frames


def create_lighten_blend_image(
    file_paths: List[str],
    output_path: str,
    progress_callback: Optional[Callable[[str], None]] = None,
    bright_area_mask: Optional[np.ndarray] = None,
    mask_generator: Optional[Callable[[np.ndarray], Optional[np.ndarray]]] = None,
    inclusion_mode: bool = False
) -> bool:
    """
    複数のファイルから比較明合成画像を作成する。
    
    メモリ使用量を最大1GBに制限するため、必要に応じて
    分割処理を行う。
    
    Args:
        file_paths: ファイルパスのリスト（フォルダも可）
        output_path: 出力画像のパス
        progress_callback: 進捗報告用コールバック関数
        bright_area_mask: 明るいエリアのマスク（0=マスク領域, 255=通常領域）
        mask_generator: 各フレームごとにマスクを生成するコールバック関数
        inclusion_mode: Trueの場合、マスク領域(255)のみ合成する（流星モード）
        
    Returns:
        bool: 成功したらTrue
    """
    if not file_paths:
        if progress_callback:
            progress_callback("ファイルが選択されていません。")
        return False
    
    # フォルダを展開してファイルリストを作成
    all_files = []
    image_ext, video_ext = get_supported_extensions()
    
    for path in file_paths:
        if os.path.isdir(path):
            all_files.extend(collect_files_from_folder(path))
        elif os.path.isfile(path):
            all_files.append(path)
    
    if not all_files:
        if progress_callback:
            progress_callback("有効なファイルが見つかりませんでした。")
        return False
    
    if progress_callback:
        progress_callback(f"合計 {len(all_files)} 個のファイルを処理します...")
    
    first_file = all_files[0]
    first_ext = Path(first_file).suffix.lower()
    
    if first_ext in image_ext:
        first_frame = _imread_unicode(first_file)
    else:
        first_frame = get_frame_from_video(first_file)
    
    if first_frame is None:
        if progress_callback:
            progress_callback(f"最初のファイルを読み込めませんでした: {first_file}")
        return False
    
    base_height, base_width = first_frame.shape[:2]
    
    # メモリ計算
    single_frame_memory = base_width * base_height * 3
    # 合成用バッファ + 現在のフレーム = 2フレーム分
    memory_per_operation = single_frame_memory * 2
    
    if progress_callback:
        memory_mb = memory_per_operation / (1024 * 1024)
        progress_callback(f"解像度: {base_width}x{base_height}, メモリ使用: 約{memory_mb:.1f}MB")
    
    # 動画からのフレーム抽出設定（メモリ制限に基づく）
    # 1GBで処理できるフレーム数を計算
    composite = np.zeros((base_height, base_width, 3), dtype=np.float32)
    processed = 0
    # first_frame is only used to decide the output size. Every selected file,
    # including the first video, is streamed through the loop below.
    
    # 最初のフレームにもマスクを適用
    if mask_generator is not None:
        first_mask = None
        if first_mask is not None:
            if first_mask.shape[0] != base_height or first_mask.shape[1] != base_width:
                first_mask = cv2.resize(first_mask, (base_width, base_height), interpolation=cv2.INTER_NEAREST)
            
            if inclusion_mode:
                # 包含モード（流星のみ合成）: マスク外（0の部分）を黒で初期化
                mask_bool = first_mask == 0
                composite[mask_bool] = 0
            else:
                # 除外モード（明るいエリアマスク）: マスク領域（0の部分）を黒で初期化
                mask_bool = first_mask == 0
                composite[mask_bool] = 0
    
    # マスクをリサイズ（必要な場合）
    resized_mask = None
    if bright_area_mask is not None:
        if bright_area_mask.shape[0] != base_height or bright_area_mask.shape[1] != base_width:
            resized_mask = cv2.resize(bright_area_mask, (base_width, base_height), interpolation=cv2.INTER_NEAREST)
        else:
            resized_mask = bright_area_mask

    del first_frame
    
    for idx, file_path in enumerate(all_files, start=1):
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext in image_ext:
                # 画像の場合
                frame = _imread_unicode(file_path)
                if frame is not None:
                    if frame.shape[1] != base_width or frame.shape[0] != base_height:
                        frame = cv2.resize(frame, (base_width, base_height))
                    
                    frame_float = frame.astype(np.float32)
                    
                    # 動的マスク生成（全画像処理）
                    current_mask = resized_mask
                    if mask_generator is not None:
                        generated = mask_generator(frame)
                        if generated is not None:
                            if generated.shape[0] != base_height or generated.shape[1] != base_width:
                                current_mask = cv2.resize(generated, (base_width, base_height), interpolation=cv2.INTER_NEAREST)
                            else:
                                current_mask = generated
                    
                    if current_mask is not None:
                        # マスク領域（0の部分）は合成しない
                        mask_bool = current_mask > 0
                        composite = np.where(mask_bool[:, :, np.newaxis], 
                                             np.maximum(composite, frame_float), 
                                             composite)
                    else:
                        np.maximum(composite, frame_float, out=composite)
                    del frame
                    del frame_float
                    processed += 1
                    
            else:
                # 動画の場合：フレームステップを動的に計算
                cap = cv2.VideoCapture(file_path)
                if cap.isOpened():
                    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    # メモリを超えないようにステップを計算
                    frame_idx = 0
                    video_frames_processed = 0
                    video_mask = resized_mask
                    video_mask_ready = False
                    
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Process every frame; memory stays bounded by the composite buffer plus one frame.
                        if frame is not None:
                            if frame.shape[1] != base_width or frame.shape[0] != base_height:
                                frame = cv2.resize(frame, (base_width, base_height))
                            
                            frame_float = frame.astype(np.float32)
                            
                            # 動的マスク生成（全画像処理）
                            current_mask = video_mask
                            if mask_generator is not None and not video_mask_ready:
                                generated = mask_generator(frame)
                                if generated is not None:
                                    if generated.shape[0] != base_height or generated.shape[1] != base_width:
                                        current_mask = cv2.resize(generated, (base_width, base_height), interpolation=cv2.INTER_NEAREST)
                                        video_mask = current_mask
                                    else:
                                        current_mask = generated
                                        video_mask = current_mask
                            video_mask_ready = True

                            if current_mask is not None:
                                # マスク領域（0の部分）は合成しない
                                mask_bool = current_mask > 0
                                composite = np.where(mask_bool[:, :, np.newaxis], 
                                                     np.maximum(composite, frame_float), 
                                                     composite)
                            else:
                                np.maximum(composite, frame_float, out=composite)
                            del frame_float
                            video_frames_processed += 1
                        
                        del frame
                        frame_idx += 1
                        
                        # メモリ解放
                        if frame_idx % 100 == 0:
                            gc.collect()
                    
                    cap.release()
                    processed += 1
                    
                    if progress_callback and video_frames_processed > 0:
                        progress_callback(f"動画処理: {os.path.basename(file_path)} ({video_frames_processed}フレーム使用)")
            
            if progress_callback and idx % 10 == 0:
                progress = idx / len(all_files) * 100
                progress_callback(f"処理中: {idx}/{len(all_files)} ({progress:.1f}%)")
            
            if idx % 50 == 0:
                gc.collect()
                
        except Exception as e:
            if progress_callback:
                progress_callback(f"警告: ファイル処理エラー: {os.path.basename(file_path)} - {e}")
            continue
    
    composite_uint8 = np.clip(composite, 0, 255).astype(np.uint8)
    del composite
    gc.collect()
    
    try:
        cv2.imwrite(output_path, composite_uint8)
        if progress_callback:
            progress_callback(f"比較明合成画像を保存しました: {output_path}")
        return True
    except Exception as e:
        if progress_callback:
            progress_callback(f"保存エラー: {e}")
        return False
