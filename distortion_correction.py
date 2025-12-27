import os
import cv2
import numpy as np
from pathlib import Path
import config

def apply_distortion_correction(sources, output_path, map_x_path, map_y_path, progress_callback=None):
    """
    Applies distortion correction to a composite image created from the first 150 frames of the first video in sources.
    
    Args:
        sources (list): List of folder paths or video file paths.
        output_path (str): Path to save the resulting image.
        map_x_path (str): Path to the x distortion map (.npy).
        map_y_path (str): Path to the y distortion map (.npy).
        progress_callback (function, optional): Callback function to report progress (message).
    """
    
    # 1. Find the first video
    first_video_path = None
    
    if progress_callback:
        progress_callback("動画ファイルを検索中...")

    # Sort sources to ensure deterministic order
    sorted_sources = sorted(sources)

    for source in sorted_sources:
        path = Path(source)
        if path.is_dir():
            # Case insensitive search for extensions
            found = sorted([p for p in path.rglob('*') if p.suffix.lower() in config.PERIODIC_VIDEO_EXTENSIONS])
            if found:
                first_video_path = found[0]
                break
        elif path.is_file() and path.suffix.lower() in config.PERIODIC_VIDEO_EXTENSIONS:
            first_video_path = path
            break
    
    if not first_video_path:
        if progress_callback:
            progress_callback("処理対象の動画ファイルが見つかりませんでした。")
        return False

    if progress_callback:
        progress_callback(f"対象動画: {first_video_path}")

    # 2. Create composite image from first 150 frames
    composite_image = None
    frames_to_process = 150
    
    try:
        cap = cv2.VideoCapture(str(first_video_path))
        if not cap.isOpened():
            if progress_callback:
                progress_callback("動画ファイルを開けませんでした。")
            return False
        
        count = 0
        while count < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break
            
            if composite_image is None:
                composite_image = frame.astype(np.uint8)
            else:
                if frame.shape != composite_image.shape:
                    frame = cv2.resize(frame, (composite_image.shape[1], composite_image.shape[0]))
                composite_image = np.maximum(composite_image, frame)
            
            count += 1
            if progress_callback and count % 30 == 0:
                progress_callback(f"フレーム読み込み中: {count}/{frames_to_process}")
        
        cap.release()
        
        if composite_image is None:
            if progress_callback:
                progress_callback("有効なフレームを取得できませんでした。")
            return False

    except Exception as e:
        if progress_callback:
            progress_callback(f"動画処理中にエラーが発生しました: {e}")
        return False

    # 3. Load distortion maps and apply correction
    if progress_callback:
        progress_callback("ゆがみ補正マップを読み込み中...")
        
    try:
        if not os.path.exists(map_x_path) or not os.path.exists(map_y_path):
             if progress_callback:
                progress_callback(f"補正マップファイルが見つかりません。\n{map_x_path}\n{map_y_path}")
             return False

        map1 = np.load(map_x_path)
        map2 = np.load(map_y_path)
        
        if progress_callback:
            progress_callback("ゆがみ補正を適用中...")

        # Ensure map dimensions match image dimensions if possible, or resize image?
        # Usually maps are generated for a specific resolution.
        # If the video resolution is different from the map resolution, remap might fail or produce weird results.
        # We assume they match as per user context.
        
        corrected_img = cv2.remap(composite_image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        cv2.imwrite(output_path, corrected_img)
        
        if progress_callback:
            progress_callback(f"補正画像を保存しました: {output_path}")
        return True

    except Exception as e:
        if progress_callback:
            progress_callback(f"ゆがみ補正処理中にエラーが発生しました: {e}")
        return False
