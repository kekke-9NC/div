import os
import cv2
import numpy as np
from pathlib import Path
import config

def create_long_exposure_map(sources, output_path, progress_callback=None):
    """
    Creates a long exposure map (comparative brightness composition) from the first frame of all videos in sources.
    
    Args:
        sources (list): List of folder paths or video file paths.
        output_path (str): Path to save the resulting image.
        progress_callback (function, optional): Callback function to report progress (message).
    """
    video_paths = []
    
    if progress_callback:
        progress_callback("動画ファイルを検索中...")

    for source in sources:
        path = Path(source)
        if path.is_dir():
            found = [p for p in path.rglob('*') if p.suffix.lower() in config.PERIODIC_VIDEO_EXTENSIONS]
            video_paths.extend(found)
        elif path.is_file() and path.suffix.lower() in config.PERIODIC_VIDEO_EXTENSIONS:
            video_paths.append(path)
    

    video_paths = sorted(list(set(video_paths)))
    
    if not video_paths:
        if progress_callback:
            progress_callback("処理対象の動画ファイルが見つかりませんでした。")
        return False

    total_videos = len(video_paths)
    if progress_callback:
        progress_callback(f"合計 {total_videos} 個の動画ファイルを処理します。")

    composite_image = None
    
    for i, video_path in enumerate(video_paths):
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                continue
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                continue
            
            if composite_image is None:
                composite_image = frame.astype(np.uint8)
            else:
                if frame.shape != composite_image.shape:
                    frame = cv2.resize(frame, (composite_image.shape[1], composite_image.shape[0]))
                
                composite_image = np.maximum(composite_image, frame)
                
            if progress_callback and i % 10 == 0:
                progress_callback(f"処理中: {i+1}/{total_videos}")
                
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            continue

    if composite_image is not None:
        try:
            cv2.imwrite(output_path, composite_image)
            if progress_callback:
                progress_callback(f"画像を保存しました: {output_path}")
            return True
        except Exception as e:
            if progress_callback:
                progress_callback(f"画像の保存に失敗しました: {e}")
            return False
    else:
        if progress_callback:
            progress_callback("有効なフレームを取得できませんでした。")
        return False
