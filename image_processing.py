# image_processing.py

import cv2
import numpy as np
from collections import deque
from PIL import Image
import config
import threading
from typing import Optional, Tuple, List, Deque, Generator, Union
import os


def _temporal_min_max(frames) -> Tuple[np.ndarray, np.ndarray]:
    """Compute extrema without stacking an entire full-HD time window.

    ``np.array(frames)`` temporarily duplicates every buffered frame.  Four
    archive workers at 1080p/25 fps could allocate hundreds of megabytes at
    the same instant.  OpenCV's in-place extrema keep only two accumulator
    frames regardless of window length.
    """
    iterator = iter(frames)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise ValueError("frames must not be empty") from exc
    maximum = first.copy()
    minimum = first.copy()
    for frame in iterator:
        cv2.max(maximum, frame, dst=maximum)
        cv2.min(minimum, frame, dst=minimum)
    return maximum, minimum

def detect_lines(
    img: np.ndarray,
    min_length: int = config.MIN_LINE_LENGTH,
    mask: Optional[np.ndarray] = None,
    roi: Optional[Tuple[int, int, int, int]] = None,
    cancel_flag: Optional[threading.Event] = None,
    canny_thresh1: int = 50,
    canny_thresh2: int = 150,
    hough_threshold: int = 25,
    hough_max_gap: int = 5
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    画像から直線を検出する関数。

    Args:
        img (numpy.ndarray): 入力画像（BGR形式）。
        min_length (int): 検出する直線の最小長さ。
        mask (numpy.ndarray, optional): マスク画像。
        roi (tuple, optional): 検出する領域（x_start, y_start, x_end, y_end）。
        cancel_flag (threading.Event, optional): キャンセル処理用のフラグ。
        canny_thresh1 (int): Cannyエッジ検出の下側閾値。
        canny_thresh2 (int): Cannyエッジ検出の上側閾値。
        hough_threshold (int): Hough変換の閾値。
        hough_max_gap (int): Hough変換で同一線分上とみなす最大ギャップ。

    Returns:
        list: 検出された直線のリスト。各直線は((x1, y1), (x2, y2))の形式。
    """
    height, width = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, canny_thresh1, canny_thresh2, apertureSize=3)

    if mask is not None:
        try:
            mask_resized = cv2.resize(mask, (edges.shape[1], edges.shape[0]))
            mask_resized = mask_resized.astype(np.uint8)
            edges = cv2.bitwise_and(edges, edges, mask=mask_resized)
        except cv2.error as e:
            print(f"マスク適用中にエラー: {e}")
            pass

    if roi:
        x_start_roi = max(roi[0], config.BORDER_SIZE)
        y_start_roi = max(roi[1], config.BORDER_SIZE)
        x_end_roi = min(roi[2], width - config.BORDER_SIZE)
        y_end_roi = min(roi[3], height - config.BORDER_SIZE)
        if x_start_roi >= x_end_roi or y_start_roi >= y_end_roi:
             print("警告: ROI領域が無効です。ROIなしで処理します。")
             edges_roi = edges[config.BORDER_SIZE:height - config.BORDER_SIZE, config.BORDER_SIZE:width - config.BORDER_SIZE]
             roi_offset_x, roi_offset_y = config.BORDER_SIZE, config.BORDER_SIZE
        else:
             edges_roi = edges[y_start_roi:y_end_roi, x_start_roi:x_end_roi]
             roi_offset_x, roi_offset_y = x_start_roi, y_start_roi
    else:
        edges_roi = edges[config.BORDER_SIZE:height - config.BORDER_SIZE, config.BORDER_SIZE:width - config.BORDER_SIZE]
        roi_offset_x, roi_offset_y = config.BORDER_SIZE, config.BORDER_SIZE

    lines = cv2.HoughLinesP(
        edges_roi,
        1,
        np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_length,
        maxLineGap=hough_max_gap
    )

    detected_lines: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    if lines is not None:
        lines = lines.reshape(-1, 1, 4)  # shape 正規化: 恰好1本の検出時にも (N,1,4) を保証
        for line in lines:
            if cancel_flag is not None and cancel_flag.is_set():
                print("線検出中にキャンセルされました。")
                break
            x1, y1, x2, y2 = line[0]
            x1 += roi_offset_x
            y1 += roi_offset_y
            x2 += roi_offset_x
            y2 += roi_offset_y
            detected_lines.append(((x1, y1), (x2, y2)))

    return detected_lines

# detect_lines_with_endpoints は detect_lines と同じロジックのためエイリアスとして定義
detect_lines_with_endpoints = detect_lines


def create_diff_images(
    cap: cv2.VideoCapture,
    interval: float = 1.0,
    duration: float = 1.0,
    median_duration: float = 1.0,
    buffer_duration: float = config.RTSP_BUFFER_DURATION,
    is_rtsp: bool = False,
    cancel_flag: Optional[threading.Event] = None,
    evidence_cap: Optional[cv2.VideoCapture] = None,
    include_frame_window: bool = False,
) -> Generator[
        Union[
            Tuple[np.ndarray, Tuple[int, int], np.ndarray, np.ndarray],
            Tuple[np.ndarray, Tuple[int, int], np.ndarray, np.ndarray, Deque[np.ndarray], int],
            Tuple[np.ndarray, Tuple[int, int], np.ndarray, np.ndarray, List[np.ndarray]],
            Tuple[np.ndarray, Tuple[int, int], np.ndarray, np.ndarray, Deque[np.ndarray], int, List[np.ndarray]],
        ], None, None]:
    """
    動画からフレームを読み込み、比較明合成と最小値画像の差分画像を生成するジェネレータ。
    RTSPの場合は、指定されたバッファ期間のフレームも一緒に返す。

    Args:
        cap (cv2.VideoCapture): 動画キャプチャオブジェクト。
        interval (float): 差分画像を生成する間隔（秒）。
        duration (float): 差分画像作成に使用するフレーム数（秒単位）。
        median_duration (float): 最小値画像作成に使用するフレーム数（秒単位）。
        buffer_duration (float): RTSPの場合にメモリ上に保持する秒数。
        is_rtsp (bool): RTSPストリームかどうか。
        cancel_flag (threading.Event, optional): キャンセル処理用のフラグ。

    Yields:
        通常時: Tuple[差分画像, (開始フレームindex, 終了フレームindex), 比較明合成画像, 最小値画像]
        RTSP時: Tuple[差分画像, (開始フレームindex, 終了フレームindex), 比較明合成画像, 最小値画像, フレームバッファ(deque), 現在のフレームindex]
    """
    if not cap.isOpened():
        print("VideoCaptureオブジェクトが開かれていません。")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = config.DEFAULT_FPS
        print(f"FPSが取得できなかったため、デフォルト値 {fps} を使用します。")

    required_frames = max(1, int(fps * duration))
    min_val_frames_count = max(1, int(fps * median_duration))
    frames_for_diff = deque(maxlen=required_frames)
    evidence_for_diff = deque(maxlen=required_frames) if evidence_cap is not None else None

    buffer_frames: Optional[Deque[np.ndarray]] = None
    if is_rtsp:
        buffer_maxlen = max(1, int(fps * buffer_duration))
        buffer_frames = deque(maxlen=buffer_maxlen)

    frame_index = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = float('inf')

    interval_frames = max(1, int(fps * interval))

    while frame_index < total_frames:
        if cancel_flag is not None and cancel_flag.is_set():
            print("差分画像生成中にキャンセルされました。")
            break

        ret, frame = cap.read()
        if not ret:
            print(f"フレーム {frame_index} の読み込みに失敗、またはストリームが終了しました。")
            break

        if evidence_cap is not None and evidence_for_diff is not None:
            evidence_ok, evidence_frame = evidence_cap.read()
            if evidence_ok and evidence_frame is not None:
                evidence_for_diff.append(evidence_frame)
            else:
                # Never desynchronise the main stream because of a damaged
                # optional evidence sidecar.  Falling back retains legacy
                # detection behaviour for this frame.
                evidence_for_diff.append(frame)

        frames_for_diff.append(frame)
        if is_rtsp and buffer_frames is not None:
            buffer_frames.append(frame)

        should_process = (frame_index > 0 and frame_index % interval_frames == 0) or \
                         (frame_index == total_frames - 1 and len(frames_for_diff) > 0)

        if is_rtsp and len(buffer_frames) < buffer_frames.maxlen:
            should_process = False

        if should_process and len(frames_for_diff) > 0:
            brightness_composite_image, min_val_image = _temporal_min_max(frames_for_diff)
            if evidence_for_diff is not None and len(evidence_for_diff) == len(frames_for_diff):
                diff_image, _unused_minimum = _temporal_min_max(evidence_for_diff)
            else:
                diff_image = cv2.absdiff(brightness_composite_image, min_val_image)

            end_idx = frame_index
            start_idx = max(0, end_idx - len(frames_for_diff) + 1)
            frame_indices = (start_idx, end_idx)

            frame_window = list(frames_for_diff) if include_frame_window else None
            if is_rtsp and buffer_frames is not None:
                if frame_window is not None:
                    yield (
                        diff_image,
                        frame_indices,
                        brightness_composite_image,
                        min_val_image,
                        buffer_frames.copy(),
                        frame_index,
                        frame_window,
                    )
                else:
                    yield diff_image, frame_indices, brightness_composite_image, min_val_image, buffer_frames.copy(), frame_index
            elif frame_window is not None:
                yield diff_image, frame_indices, brightness_composite_image, min_val_image, frame_window
            else:
                yield diff_image, frame_indices, brightness_composite_image, min_val_image

        frame_index += 1

    print("差分画像生成ジェネレータを終了します。")


def flip_image_vertically(image_path: str) -> Optional[str]:
    """
    指定された画像ファイルを読み込み、上下反転して別名で保存し、そのパスを返す。

    Args:
        image_path (str): 元の画像ファイルのパス。

    Returns:
        str: 上下反転された新しい画像のパス。エラー時は None。
    """
    try:
        image = Image.open(image_path)
        flipped_image = image.transpose(Image.FLIP_TOP_BOTTOM)

        base, ext = os.path.splitext(image_path)
        flipped_image_path = f"{base}_flipped{ext}"

        flipped_image.save(flipped_image_path)
        print(f"上下反転した画像を保存しました: {flipped_image_path}")
        return flipped_image_path
    except FileNotFoundError:
        print(f"エラー: 画像ファイルが見つかりません: {image_path}")
        return None
    except Exception as e:
        print(f"画像の上下反転中にエラーが発生しました: {e}")
        return None

def resize_preserve_aspect(image: np.ndarray, max_size: Tuple[int, int]) -> np.ndarray:
    """
    画像のアスペクト比を維持しながら、指定された (max_width, max_height) 内に収まるようにリサイズする。

    Args:
        image (numpy.ndarray): リサイズ対象の画像。
        max_size (tuple): (max_width, max_height) のタプル。

    Returns:
        numpy.ndarray: リサイズ後の画像。
    """
    h, w = image.shape[:2]
    max_w, max_h = max_size

    if h == 0 or w == 0:
        return np.zeros((max_h, max_w, 3), dtype=image.dtype)

    scale = min(max_w / w, max_h / h)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


if __name__ == '__main__':
    print("image_processing.py が直接実行されました。")

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.line(dummy_frame, (100, 100), (300, 150), (0, 255, 0), 2)

    print("\n--- detect_lines テスト ---")
    detected = detect_lines(dummy_frame, min_length=50)
    if detected:
        print(f"線が検出されました: {len(detected)} 本")
        print(f"最初の線: {detected[0]}")
    else:
        print("線は検出されませんでした。")

    print("\n--- resize_preserve_aspect テスト ---")
    resized_frame = resize_preserve_aspect(dummy_frame, (320, 240))
    print(f"リサイズ後の形状: {resized_frame.shape}")
    if resized_frame.shape[1] <= 320 and resized_frame.shape[0] <= 240:
        print("リサイズ成功。")
    else:
        print("リサイズ失敗。")

    print("\n--- create_diff_images (テスト省略) ---")

    print("\n--- flip_image_vertically テスト ---")
    dummy_image_path = "dummy_flip_test.png"
    cv2.imwrite(dummy_image_path, dummy_frame)
    flipped_path = flip_image_vertically(dummy_image_path)
    if flipped_path and os.path.exists(flipped_path):
        print(f"反転画像が作成されました: {flipped_path}")
        os.remove(flipped_path)
    else:
        print("反転画像の作成に失敗しました。")
    if os.path.exists(dummy_image_path):
        os.remove(dummy_image_path)
