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
import video_processor


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


def _has_same_aspect_ratio(width: int, height: int, base_width: int, base_height: int) -> bool:
    """Return whether two positive image sizes have aspect ratios within 0.1%."""
    return (
        width > 0
        and height > 0
        and base_width > 0
        and base_height > 0
        and np.isclose(width / height, base_width / base_height, rtol=1e-3, atol=0.0)
    )


def _aspect_ratios_match(first: float, second: float) -> bool:
    return (
        first is not None and second is not None
        and first > 0 and second > 0
        and np.isclose(first, second, rtol=1e-3, atol=0.0)
    )


def _enable_video_orientation_auto(cap, rotation_degrees: float) -> bool:
    orientation_property = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
    rotation_required = abs(rotation_degrees % 360.0) > 0.5
    if orientation_property is None:
        return not rotation_required
    cap.set(orientation_property, 1)
    return cap.get(orientation_property) >= 0.5 or not rotation_required


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


def get_frame_from_video(
    video_path: str, frame_index: int = 0, display_info: Optional[dict] = None
) -> Optional[np.ndarray]:
    """
    動画から指定フレームを取得する。
    
    Args:
        video_path: 動画ファイルパス
        frame_index: フレームインデックス (0 = 最初のフレーム)
        
    Returns:
        フレーム画像 or None
    """
    if display_info is None:
        display_info = video_processor.get_video_display_info(video_path)
    if display_info is None:
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    if not _enable_video_orientation_auto(cap, display_info['rotation_degrees']):
        cap.release()
        return None
    
    if frame_index > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return None
    if not _has_same_aspect_ratio(
        frame.shape[1], frame.shape[0], display_info['width'], display_info['height']
    ):
        return None
    return frame


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
    inclusion_mode: bool = False,
    mask_generator_for_path: Optional[Callable[[np.ndarray, str], Optional[np.ndarray]]] = None
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
        mask_generator_for_path: ファイルパスと画像からマスクを生成するコールバック関数
        
    Returns:
        bool: 成功したらTrue
    """
    if not file_paths:
        if progress_callback:
            progress_callback("ファイルが選択されていません。")
        return False
    
    # フォルダを展開してファイルリストを作成
    all_files = []
    invalid_inputs = []
    image_ext, video_ext = get_supported_extensions()
    
    for path in file_paths:
        if os.path.isdir(path):
            folder_files = collect_files_from_folder(path)
            if folder_files:
                all_files.extend(folder_files)
            else:
                invalid_inputs.append(path)
        elif os.path.isfile(path):
            all_files.append(path)
        else:
            invalid_inputs.append(path)

    if invalid_inputs:
        if progress_callback:
            failed_names = ", ".join(os.path.basename(path) or path for path in invalid_inputs)
            progress_callback(f"選択した入力が見つからないか、有効なファイルを含んでいません: {failed_names}")
        return False
    
    if not all_files:
        if progress_callback:
            progress_callback("有効なファイルが見つかりませんでした。")
        return False
    
    if progress_callback:
        progress_callback(f"合計 {len(all_files)} 個のファイルを処理します...")
    
    first_file = all_files[0]
    first_ext = Path(first_file).suffix.lower()
    first_video_raster_size = None
    first_video_display_info = None
    
    if first_ext in image_ext:
        first_frame = _imread_unicode(first_file)
        base_display_aspect_ratio = None
    else:
        first_video_display_info = video_processor.get_video_display_info(first_file)
        first_frame = get_frame_from_video(first_file, display_info=first_video_display_info)
        if first_frame is not None:
            first_video_raster_size = (first_frame.shape[1], first_frame.shape[0])
            base_display_aspect_ratio = first_video_display_info['display_aspect_ratio']
            base_height = first_frame.shape[0]
            base_width = max(1, int(round(base_height * base_display_aspect_ratio)))
            if (first_frame.shape[1], first_frame.shape[0]) != (base_width, base_height):
                first_frame = cv2.resize(first_frame, (base_width, base_height))
    
    if first_frame is None:
        if progress_callback:
            progress_callback(f"最初のファイルを読み込めませんでした: {first_file}")
        return False
    
    base_height, base_width = first_frame.shape[:2]
    if base_display_aspect_ratio is None:
        base_display_aspect_ratio = base_width / base_height

    if bright_area_mask is not None and not (
        _has_same_aspect_ratio(
            bright_area_mask.shape[1], bright_area_mask.shape[0], base_width, base_height
        )
        or (
            first_video_raster_size is not None
            and _has_same_aspect_ratio(
                bright_area_mask.shape[1], bright_area_mask.shape[0], *first_video_raster_size
            )
        )
    ):
        if progress_callback:
            progress_callback("エラー: マスクと入力画像の縦横比が一致しません。")
        return False
    
    # メモリ計算
    single_frame_memory = base_width * base_height * 3
    # 合成用バッファ + 現在のフレーム = 2フレーム分
    memory_per_operation = single_frame_memory * 2
    
    if progress_callback:
        memory_mb = memory_per_operation / (1024 * 1024)
        progress_callback(f"解像度: {base_width}x{base_height}, メモリ使用: 約{memory_mb:.1f}MB")
    
    # 動画からのフレーム抽出設定（メモリ制限に基づく）
    # 1GBで処理できるフレーム数を計算
    max_video_frames_in_memory = max(1, (MAX_MEMORY_BYTES // 2) // single_frame_memory)
    
    composite = first_frame.astype(np.float32)

    def generate_mask(frame: np.ndarray, source_path: str) -> Optional[np.ndarray]:
        if mask_generator_for_path is not None:
            return mask_generator_for_path(frame, source_path)
        if mask_generator is not None:
            return mask_generator(frame)
        return None
    
    # 最初のフレームにもマスクを適用
    if mask_generator is not None or mask_generator_for_path is not None:
        try:
            first_mask = generate_mask(first_frame, first_file)
        except Exception as e:
            del composite
            gc.collect()
            if progress_callback:
                progress_callback(f"最初のファイルのマスク生成に失敗しました: {first_file} - {e}")
            return False
        if first_mask is not None:
            if not _has_same_aspect_ratio(
                first_mask.shape[1], first_mask.shape[0], base_width, base_height
            ):
                del composite
                gc.collect()
                if progress_callback:
                    progress_callback(f"最初のファイルのマスクと入力の縦横比が一致しません: {first_file}")
                return False
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
    
    failed_sources = []
    for idx, file_path in enumerate(all_files, start=1):
        cap = None
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext in image_ext and idx == 1:
                # The first image was already used to initialize the composite.
                pass
            elif file_ext in image_ext:
                # 画像の場合
                frame = _imread_unicode(file_path)
                if frame is None:
                    failed_sources.append(file_path)
                    if progress_callback:
                        progress_callback(f"画像を読み込めませんでした: {file_path}")
                elif not _has_same_aspect_ratio(
                    frame.shape[1], frame.shape[0], base_width, base_height
                ):
                    failed_sources.append(file_path)
                    if progress_callback:
                        progress_callback(f"入力の縦横比が基準画像と一致しません: {file_path}")
                else:
                    if frame.shape[1] != base_width or frame.shape[0] != base_height:
                        frame = cv2.resize(frame, (base_width, base_height))

                    frame_float = frame.astype(np.float32)

                    # 動的マスク生成（全画像処理）
                    current_mask = resized_mask
                    if mask_generator is not None or mask_generator_for_path is not None:
                        generated = generate_mask(frame, file_path)
                        if generated is not None:
                            if generated.shape[0] != base_height or generated.shape[1] != base_width:
                                if not _has_same_aspect_ratio(
                                    generated.shape[1], generated.shape[0], base_width, base_height
                                ):
                                    raise ValueError("生成マスクと入力画像の縦横比が一致しません")
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
                    
            else:
                # 動画の場合：フレームステップを動的に計算
                cap = cv2.VideoCapture(file_path)
                if cap.isOpened():
                    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    video_display_info = (
                        first_video_display_info
                        if file_path == first_file
                        else video_processor.get_video_display_info(file_path)
                    )
                    if video_display_info is None or not _enable_video_orientation_auto(
                        cap,
                        video_display_info['rotation_degrees'] if video_display_info else 0.0
                    ):
                        failed_sources.append(file_path)
                        if progress_callback:
                            progress_callback(f"動画の表示情報または回転を取得できません: {file_path}")
                        continue
                    video_width = video_display_info['width']
                    video_height = video_display_info['height']
                    video_display_aspect_ratio = video_display_info['display_aspect_ratio']
                    if not _aspect_ratios_match(
                        video_display_aspect_ratio, base_display_aspect_ratio
                    ):
                        failed_sources.append(file_path)
                        if progress_callback:
                            progress_callback(f"動画の表示縦横比が基準入力と一致しません: {file_path}")
                        continue
                    
                    # メモリを超えないようにステップを計算
                    step = max(1, total_video_frames // max_video_frames_in_memory)
                    
                    frame_idx = 0
                    decoded_frame_count = 0
                    frame_geometry_mismatch = False
                    # Count frame 0, which seeded the composite before this loop.
                    video_frames_processed = 0
                    
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        decoded_frame_count += 1

                        if not _has_same_aspect_ratio(
                            frame.shape[1], frame.shape[0], video_width, video_height
                        ):
                            frame_geometry_mismatch = True
                            break

                        if frame.shape[1] != base_width or frame.shape[0] != base_height:
                            frame = cv2.resize(frame, (base_width, base_height))
                        
                        # The first video frame was already used to initialize the composite.
                        if frame_idx % step == 0 and not (idx == 1 and frame_idx == 0):
                            frame_float = frame.astype(np.float32)
                            
                            # 動的マスク生成（全画像処理）
                            current_mask = resized_mask
                            if mask_generator is not None or mask_generator_for_path is not None:
                                generated = generate_mask(frame, file_path)
                                if generated is not None:
                                    if generated.shape[0] != base_height or generated.shape[1] != base_width:
                                        if not _has_same_aspect_ratio(
                                            generated.shape[1], generated.shape[0],
                                            frame.shape[1], frame.shape[0]
                                        ):
                                            raise ValueError("生成マスクと動画フレームの縦横比が一致しません")
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
                            del frame_float
                            video_frames_processed += 1
                        
                        del frame
                        frame_idx += 1
                        
                        # メモリ解放
                        if frame_idx % 100 == 0:
                            gc.collect()
                    
                    if frame_geometry_mismatch:
                        failed_sources.append(file_path)
                        if progress_callback:
                            progress_callback(f"動画のフレームが基準入力と異なる縦横比です: {file_path}")
                    elif decoded_frame_count == 0 or (
                        total_video_frames > 0 and decoded_frame_count < total_video_frames
                    ):
                        failed_sources.append(file_path)
                        if progress_callback:
                            progress_callback(
                                f"動画の全フレームを読み込めませんでした: {file_path} "
                                f"({decoded_frame_count}/{total_video_frames}フレーム)"
                            )
                    elif progress_callback:
                        progress_callback(f"動画処理: {os.path.basename(file_path)} ({video_frames_processed + (1 if idx == 1 else 0)}フレーム使用)")
                else:
                    failed_sources.append(file_path)
                    if progress_callback:
                        progress_callback(f"動画を開けませんでした: {file_path}")
            
            if progress_callback and idx % 10 == 0:
                progress = idx / len(all_files) * 100
                progress_callback(f"処理中: {idx}/{len(all_files)} ({progress:.1f}%)")
            
            if idx % 50 == 0:
                gc.collect()
                
        except Exception as e:
            failed_sources.append(file_path)
            if progress_callback:
                progress_callback(f"警告: ファイル処理エラー: {os.path.basename(file_path)} - {e}")
        finally:
            if cap is not None:
                cap.release()
    
    if failed_sources:
        del composite
        gc.collect()
        if progress_callback:
            failed_names = ", ".join(os.path.basename(path) or path for path in failed_sources)
            progress_callback(f"入力の処理に失敗したため、合成画像を保存しませんでした: {failed_names}")
        return False

    composite_uint8 = np.clip(composite, 0, 255).astype(np.uint8)
    del composite
    gc.collect()
    
    try:
        if not cv2.imwrite(output_path, composite_uint8):
            if progress_callback:
                progress_callback(f"保存エラー: 画像を書き込めませんでした: {output_path}")
            return False
        if progress_callback:
            progress_callback(f"比較明合成画像を保存しました: {output_path}")
        return True
    except Exception as e:
        if progress_callback:
            progress_callback(f"保存エラー: {e}")
        return False
