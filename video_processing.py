import os
import math
import cv2
import numpy as np
from pathlib import Path
import time
from datetime import datetime, timedelta
import threading
import queue
from typing import List, Tuple, Optional, Callable, Dict, Any
from astropy.io import fits
from astropy.wcs import WCS
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

import config
import image_processing
import model
import astrometry
import tracking
import utils
import video_creation
import video_denoising
import video_enhancement
import ml_training_data
import automatic_video_mask
import noise_twin
import temporal_mean


def _open_video_capture(source: str, decoder_threads: int = 2) -> cv2.VideoCapture:
    """Open a capture without letting FFmpeg create a CPU-sized pool.

    OpenCV's macOS FFmpeg backend defaults to 16 codec threads for each H.264
    capture on the current machine, plus another helper pool after the first
    read.  Several videos in parallel therefore created around one hundred
    native threads.  ``CAP_PROP_N_THREADS`` is an *open-only* property, so it
    must be supplied to the constructor rather than set afterwards.
    """
    try:
        thread_count = max(1, min(8, int(decoder_threads)))
    except (TypeError, ValueError):
        thread_count = 2

    thread_property = getattr(cv2, "CAP_PROP_N_THREADS", None)
    ffmpeg_backend = getattr(cv2, "CAP_FFMPEG", None)
    if thread_property is not None and ffmpeg_backend is not None:
        try:
            capture = cv2.VideoCapture(
                source,
                ffmpeg_backend,
                [thread_property, thread_count],
            )
            if capture.isOpened():
                return capture
            capture.release()
        except (cv2.error, TypeError, ValueError):
            pass
    return cv2.VideoCapture(source)


_FULL_VIDEO_TIMESTAMP_POSITIONS = {
    "右下": "bottom_right",
    "左下": "bottom_left",
    "右上": "top_right",
    "左上": "top_left",
    "bottom_right": "bottom_right",
    "bottom_left": "bottom_left",
    "top_right": "top_right",
    "top_left": "top_left",
}


def _get_full_video_timestamp_settings(save_options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """保存設定からフルサイズ動画用の時刻表示設定を正規化する。"""
    options = save_options or {}
    position = _FULL_VIDEO_TIMESTAMP_POSITIONS.get(
        str(options.get('full_video_timestamp_position', config.FULL_VIDEO_TIMESTAMP_POSITION)),
        config.FULL_VIDEO_TIMESTAMP_POSITION,
    )
    try:
        size_percent = float(
            options.get('full_video_timestamp_size_percent', config.FULL_VIDEO_TIMESTAMP_SIZE_PERCENT)
        )
    except (TypeError, ValueError):
        size_percent = config.FULL_VIDEO_TIMESTAMP_SIZE_PERCENT
    return {
        'enabled': bool(options.get('full_video_timestamp_enabled', config.FULL_VIDEO_TIMESTAMP_ENABLED)),
        'position': position,
        # 画面を覆わず読み取り可能な範囲に制限する。
        'size_percent': max(0.8, min(4.0, size_percent)),
    }


def add_full_video_timestamp(
    frame: np.ndarray,
    timestamp: datetime,
    timestamp_settings: Dict[str, Any],
) -> np.ndarray:
    """フルサイズ動画フレームの隅に、読みやすい時刻表示を描画する。"""
    if not timestamp_settings.get('enabled', False):
        return frame

    output = frame.copy()
    height, width = output.shape[:2]
    if height <= 0 or width <= 0:
        return output

    text = timestamp.strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
    font = cv2.FONT_HERSHEY_SIMPLEX
    desired_height = max(12, int(round(height * timestamp_settings['size_percent'] / 100.0)))
    unit_height = max(1, cv2.getTextSize("Ag", font, 1.0, 1)[0][1])
    font_scale = desired_height / unit_height
    thickness = max(1, int(round(desired_height / 18)))
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    margin = max(8, int(round(desired_height * 0.55)))
    position = timestamp_settings['position']

    x = margin if position.endswith("left") else width - text_width - margin
    y = text_height + margin if position.startswith("top") else height - margin
    x = max(0, min(x, max(0, width - text_width)))
    y = max(text_height, min(y, max(text_height, height - baseline)))

    padding = max(3, desired_height // 4)
    left = max(0, x - padding)
    top = max(0, y - text_height - padding)
    right = min(width, x + text_width + padding)
    bottom = min(height, y + baseline + padding)
    if right > left and bottom > top:
        roi = output[top:bottom, left:right]
        shade = np.zeros_like(roi)
        output[top:bottom, left:right] = cv2.addWeighted(roi, 0.45, shade, 0.55, 0)

    # 黒い縁取りと白文字にして、雲や街明かりの上でも読めるようにする。
    cv2.putText(output, text, (x, y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(output, text, (x, y), font, font_scale, (245, 245, 245), thickness, cv2.LINE_AA)
    return output


def write_full_size_video(
    frames: List[np.ndarray],
    output_path: str,
    frame_rate: float,
    clip_start_datetime: datetime,
    save_options: Optional[Dict[str, Any]],
) -> bool:
    """時刻表示を適用してフルサイズ動画を書き出す。"""
    if not frames:
        return False
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*config.VIDEO_FOURCC),
        frame_rate,
        (width, height),
    )
    if not writer.isOpened():
        return False

    timestamp_settings = _get_full_video_timestamp_settings(save_options)
    safe_frame_rate = frame_rate if frame_rate > 0 else config.DEFAULT_FPS
    try:
        for index, frame in enumerate(frames):
            timestamp = clip_start_datetime + timedelta(seconds=index / safe_frame_rate)
            writer.write(add_full_video_timestamp(frame, timestamp, timestamp_settings))
    except Exception as exc:
        print(f"フルサイズ動画への時刻表示中にエラー: {exc}")
        return False
    finally:
        writer.release()
    return True


def write_denoised_full_size_video(
    frames: List[np.ndarray],
    output_path: str,
    frame_rate: float,
    clip_start_datetime: datetime,
    save_options: Optional[Dict[str, Any]],
    detected_line: Tuple[Tuple[int, int], Tuple[int, int]],
) -> Optional[Dict[str, float]]:
    """閲覧用ノイズ低減動画を書き出し、処理前後のノイズ推定値を返す。"""
    if not frames:
        return None
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*config.VIDEO_FOURCC),
        frame_rate,
        (width, height),
    )
    if not writer.isOpened():
        return None

    original_meter = video_denoising.TemporalNoiseMeter()
    denoised_meter = video_denoising.TemporalNoiseMeter()
    for frame in frames:
        original_meter.update(frame)

    timestamp_settings = _get_full_video_timestamp_settings(save_options)
    safe_frame_rate = frame_rate if frame_rate > 0 else config.DEFAULT_FPS
    try:
        denoised_frames = video_denoising.iter_denoised_frames(
            frames,
            detected_line=detected_line,
            temporal_radius=config.DENOISE_TEMPORAL_RADIUS,
            original_blend=config.DENOISE_ORIGINAL_BLEND,
            transient_threshold=config.DENOISE_TRANSIENT_THRESHOLD,
            protect_line_width=config.DENOISE_PROTECT_LINE_WIDTH,
        )
        for index, frame in enumerate(denoised_frames):
            denoised_meter.update(frame)
            timestamp = clip_start_datetime + timedelta(seconds=index / safe_frame_rate)
            writer.write(add_full_video_timestamp(frame, timestamp, timestamp_settings))
    except Exception as exc:
        print(f"ノイズ低減フルサイズ動画の生成中にエラー: {exc}")
        return None
    finally:
        writer.release()

    before_sigma = original_meter.sigma
    after_sigma = denoised_meter.sigma
    return {
        'before_sigma': before_sigma,
        'after_sigma': after_sigma,
        'reduction_percent': video_denoising.noise_reduction_percent(before_sigma, after_sigma),
    }


def _brightness_composite_and_diff(frames: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Build max/min composite difference without stacking every full-HD frame."""
    if not frames:
        raise ValueError("frames must not be empty")
    maximum = frames[0].copy()
    minimum = frames[0].copy()
    for frame in frames[1:]:
        maximum = cv2.max(maximum, frame)
        minimum = cv2.min(minimum, frame)
    return maximum, cv2.absdiff(maximum, minimum)


def _max_composite_cutout(
    frames: List[np.ndarray],
    cutout_rect: Tuple[int, int, int, int],
) -> np.ndarray:
    """Build a temporal maximum only for the required detail crop."""
    if not frames:
        raise ValueError("frames must not be empty")
    x_start, y_start, x_end, y_end = cutout_rect
    maximum = frames[0][y_start:y_end, x_start:x_end].copy()
    for frame in frames[1:]:
        cv2.max(
            maximum,
            frame[y_start:y_end, x_start:x_end],
            dst=maximum,
        )
    return maximum


def _extract_diff_cutout(
    diff_image: np.ndarray,
    cutout_rect: Tuple[int, int, int, int],
    cut_size: int,
) -> np.ndarray:
    """Extract a consistently sized classifier or saved-image diff cutout."""
    x_start, y_start, x_end, y_end = cutout_rect
    cutout = diff_image[y_start:y_end, x_start:x_end]
    if cutout.shape[0] != cut_size or cutout.shape[1] != cut_size:
        cutout = cv2.resize(cutout, (cut_size, cut_size))
    return cutout


def _write_noise_twin_info(
    handle,
    save_options: Optional[Dict[str, Any]],
    trajectory_score: float = 0.0,
    innovation_max: Optional[float] = None,
) -> None:
    options = save_options or {}
    mean_window = temporal_mean.normalize_window(options.get("temporal_mean_frames", 0))
    if mean_window:
        handle.write(f"Temporal Mean Frames: {mean_window}\n")
        handle.write("Temporal Mean Model Required: false\n")
    metadata = options.get("noise_twin_metadata")
    if not isinstance(metadata, dict):
        return
    handle.write("NoiseTwin Enabled: true\n")
    handle.write(f"NoiseTwin Model ID: {metadata.get('model_id', 'N/A')}\n")
    handle.write(f"NoiseTwin Architecture: {metadata.get('architecture', 'N/A')}\n")
    handle.write(f"NoiseTwin AI Background Version: {metadata.get('format_version', 'N/A')}\n")
    handle.write("NoiseTwin Residual Protection Version: physical-trajectory-gate-v1\n")
    handle.write(
        f"NoiseTwin Fixed Pattern SHA256: {metadata.get('fixed_pattern_sha256', '')}\n"
    )
    metrics = options.get("noise_twin_metrics", {})
    handle.write(f"NoiseTwin Trajectory Score: {float(trajectory_score):.4f}\n")
    if isinstance(metrics, dict):
        handle.write(f"NoiseTwin Noise Sigma: {float(metrics.get('noise_sigma', 0.0)):.4f}\n")
        effective_innovation_max = (
            float(innovation_max)
            if innovation_max is not None
            else float(metrics.get("innovation_max", 0.0))
        )
        handle.write(f"NoiseTwin Innovation Max: {effective_innovation_max:.4f}\n")
        handle.write(f"NoiseTwin Flux Retention: {float(metrics.get('flux_retention', 0.0)):.4f}\n")
    validation = metadata.get("validation", {})
    if isinstance(validation, dict):
        handle.write(
            "NoiseTwin Validated Noise Reduction: "
            f"{float(validation.get('false_positive_reduction', 0.0)) * 100.0:.2f}%\n"
        )


def get_rtsp_recording_start_datetime(
    source: str,
    total_frames: int,
    frame_rate: float,
) -> Tuple[Optional[datetime], str]:
    """RTSP保存ファイルの開始時刻を、作成日時を優先して決定する。

    RTSPセグメントは書き込み開始時に作成されるため、macOS/Windowsで取得できる
    作成日時（birth time）が最も直接的な開始時刻になる。ファイルをコピーした場合や
    作成日時を提供しないファイルシステムでは、更新日時から動画長を引く。
    """
    try:
        stat_result = os.stat(source)
        birth_time = getattr(stat_result, "st_birthtime", None)
        if birth_time is not None and birth_time > 0:
            return datetime.fromtimestamp(birth_time), "ファイル作成日時"

        duration_seconds = total_frames / frame_rate if total_frames > 0 and frame_rate > 0 else 0
        if stat_result.st_mtime > 0:
            return (
                datetime.fromtimestamp(stat_result.st_mtime) - timedelta(seconds=duration_seconds),
                "最終更新日時−動画長",
            )
    except OSError as exc:
        print(f"警告: RTSP保存ファイルの日時取得に失敗しました: {exc}")
    return None, "取得不可"


def _point_to_segment_distance(
    point: Tuple[float, float],
    start: Tuple[float, float],
    end: Tuple[float, float],
) -> float:
    """点から線分までの最短距離を返す。"""
    px, py = point
    sx, sy = start
    ex, ey = end
    dx, dy = ex - sx, ey - sy
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-6:
        return float(np.hypot(px - sx, py - sy))
    projection = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / length_sq))
    return float(np.hypot(px - (sx + projection * dx), py - (sy + projection * dy)))


def _finer_line_matches_candidate(
    finer_line: Tuple[Tuple[int, int], Tuple[int, int]],
    candidate_line: Tuple[Tuple[int, int], Tuple[int, int]],
    cutout_rect: Tuple[int, int, int, int],
) -> bool:
    """詳細検出の線が粗検出した流星候補と同じ軌跡か判定する。

    詳細検出はカットアウト内の線をすべて検出していたため、雲の縁やノイズでも
    検出期間が広がっていた。位置と向きを粗検出線分に照合してから採用する。
    """
    (fx1, fy1), (fx2, fy2) = finer_line
    offset_x, offset_y = cutout_rect[:2]
    fine_start = (fx1 + offset_x, fy1 + offset_y)
    fine_end = (fx2 + offset_x, fy2 + offset_y)
    (cx1, cy1), (cx2, cy2) = candidate_line
    candidate_start = (float(cx1), float(cy1))
    candidate_end = (float(cx2), float(cy2))

    candidate_angle = np.arctan2(cy2 - cy1, cx2 - cx1)
    fine_angle = np.arctan2(fine_end[1] - fine_start[1], fine_end[0] - fine_start[0])
    # 線分には向きがないため 180 度反転を同じ軌跡として扱う。
    angle_difference = abs((fine_angle - candidate_angle + np.pi / 2) % np.pi - np.pi / 2)
    if np.degrees(angle_difference) > config.FINER_LINE_MATCH_ANGLE_DEG:
        return False

    fine_center = ((fine_start[0] + fine_end[0]) / 2, (fine_start[1] + fine_end[1]) / 2)
    distance = min(
        _point_to_segment_distance(fine_start, candidate_start, candidate_end),
        _point_to_segment_distance(fine_end, candidate_start, candidate_end),
        _point_to_segment_distance(fine_center, candidate_start, candidate_end),
    )
    return distance <= config.FINER_LINE_MATCH_DISTANCE


def _has_matching_finer_line(
    lines: List[Tuple[Tuple[int, int], Tuple[int, int]]],
    candidate_line: Tuple[Tuple[int, int], Tuple[int, int]],
    cutout_rect: Tuple[int, int, int, int],
) -> bool:
    return any(
        _finer_line_matches_candidate(line, candidate_line, cutout_rect)
        for line in lines
    )


def _process_finer_detection_worker(
    frames_data: List[np.ndarray],
    cx: int,
    cy: int,
    img_h: int,
    img_w: int,
    actual_start_frame_index: int,
    candidate_line: Tuple[Tuple[int, int], Tuple[int, int]],
    composite_step: int,
    cut_size: int,
    border_margin: int,
    finer_min_length: int
) -> Tuple[List[int], Tuple[int, int, int, int]]:
    """
    並列処理用の詳細検出ワーカー関数。
    
    Args:
        frames_data: 詳細検出用フレームのリスト
        cx, cy: 粗検出の中心座標
        img_h, img_w: 元画像のサイズ
        actual_start_frame_index: 詳細検出開始フレームインデックス
        composite_step: 比較明合成ステップ
        cut_size: カットアウトサイズ
        border_margin: 画像端からのマージン
        finer_min_length: 詳細検出の最小線長
    
    Returns:
        Tuple[検出されたフレームインデックスのリスト, カットアウト領域座標]
    """
    # カットアウト領域を計算
    half_cut = cut_size // 2
    
    x_start_cut = max(border_margin, cx - half_cut)
    y_start_cut = max(border_margin, cy - half_cut)
    x_end_cut = min(img_w - border_margin, cx + half_cut)
    y_end_cut = min(img_h - border_margin, cy + half_cut)

    # サイズ調整
    current_w = x_end_cut - x_start_cut
    current_h = y_end_cut - y_start_cut
    if current_w < cut_size:
        diff_w = cut_size - current_w
        x_start_cut = max(border_margin, x_start_cut - diff_w // 2)
        x_end_cut = min(img_w - border_margin, x_start_cut + cut_size)
        if x_end_cut - x_start_cut < cut_size:
            x_start_cut = x_end_cut - cut_size
    if current_h < cut_size:
        diff_h = cut_size - current_h
        y_start_cut = max(border_margin, y_start_cut - diff_h // 2)
        y_end_cut = min(img_h - border_margin, y_start_cut + cut_size)
        if y_end_cut - y_start_cut < cut_size:
            y_start_cut = y_end_cut - cut_size
    x_start_cut = max(border_margin, x_end_cut - cut_size)
    y_start_cut = max(border_margin, y_end_cut - cut_size)
    
    cutout_rect = (x_start_cut, y_start_cut, x_end_cut, y_end_cut)
    
    # 比較明合成フレーム作成（カットアウト領域のみ）
    composite_frames = []
    composite_frame_indices = []
    
    for i in range(0, len(frames_data) - (composite_step - 1), composite_step):
        composite_cutout = _max_composite_cutout(
            frames_data[i : i + composite_step],
            cutout_rect,
        )
        composite_frames.append(composite_cutout)
        composite_frame_indices.append(actual_start_frame_index + i + (composite_step // 2))
    
    # 直線検出
    line_detected_indices = []
    
    for i in range(1, len(composite_frames)):
        diff_img_cutout = cv2.absdiff(composite_frames[i], composite_frames[i - 1])
        lines = image_processing.detect_lines(diff_img_cutout, min_length=finer_min_length)
        if lines and _has_matching_finer_line(lines, candidate_line, cutout_rect):
            line_detected_indices.append(composite_frame_indices[i])
    
    return line_detected_indices, cutout_rect


def create_line_video_clips(
    source: str,
    is_rtsp: bool = False,
    interval: float = config.DEFAULT_INTERVAL,
    duration: float = config.DEFAULT_DURATION,
    min_length: int = config.MIN_LINE_LENGTH,
    mask: Optional[np.ndarray] = None,
    roi: Optional[Tuple[int, int, int, int]] = None,
    progress_callback: Optional[Callable[[Tuple[str, Optional[float]]], None]] = None,
    meteor_save_path: str = config.DEFAULT_METEOR_SAVE_PATH,
    not_meteor_save_path: str = config.DEFAULT_NOT_METEOR_SAVE_PATH,
    use_plate_solve: bool = False,
    global_wcs_info: Optional[Dict] = None,
    plate_solve_mask: Optional[np.ndarray] = None,
    buffer_duration: float = config.RTSP_BUFFER_DURATION,
    cancel_flag: Optional[threading.Event] = None,
    save_options: Optional[Dict[str, bool]] = None,
    notify_on_detection: bool = False,
    summary_video_config: Optional[List[Dict[str, Any]]] = None,
    fixed_pattern_correction: Optional[np.ndarray] = None,
    noise_twin_options: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    動画ソースから流星候補の線分を検出し、詳細検出を行い、関連情報を保存する。
    """
    if cancel_flag is not None and cancel_flag.is_set():
        print("処理開始前にキャンセルされました。")
        return []

    default_save_options = {
        'video': config.DEFAULT_SAVE_VIDEO_CLIP, 'cutout': config.DEFAULT_SAVE_CUTOUT_DIFF,
        'full': config.DEFAULT_SAVE_FULL_DIFF, 'composite': config.DEFAULT_SAVE_COMPOSITE,
        'info': config.DEFAULT_SAVE_DETECTION_INFO, 'summary': True,
        'full_video': config.DEFAULT_SAVE_FULL_VIDEO,
        'denoised_full_video': False,
    }
    if save_options is None:
        save_options = default_save_options
    else:
        current_save_options = default_save_options.copy()
        current_save_options.update(save_options)
        save_options = current_save_options

    try:
        decoder_threads = max(1, min(8, int(save_options.get('video_decoder_threads', 2))))
    except (TypeError, ValueError):
        decoder_threads = 2

    twin_options = noise_twin.NoiseTwinOptions.from_value(noise_twin_options)
    prepared_video: Optional[noise_twin.PreparedVideo] = None
    prepared_mean: Optional[temporal_mean.PreparedTemporalVideo] = None
    mean_window = temporal_mean.normalize_window(
        (noise_twin_options or {}).get("temporal_mean_frames", 0)
    )
    if twin_options.enabled and mean_window:
        raise ValueError("NoiseTwinと3/5フレーム平均は同時に使用できません。")
    analysis_source = source
    evidence_path = ""
    active_fixed_pattern_correction = fixed_pattern_correction
    if twin_options.enabled:
        if not twin_options.model_path:
            raise noise_twin.NoiseTwinError("NoiseTwinモデルが選択されていません。")
        processed_marker = noise_twin.load_processing_marker(source)
        if processed_marker is not None and not twin_options.already_processed:
            selected_metadata = noise_twin.load_metadata(twin_options.model_path)
            if processed_marker.get("model_id") != selected_metadata.model_id:
                raise noise_twin.NoiseTwinError(
                    "この動画は別のNoiseTwinモデルで処理済みです。二重推論を防止しました。"
                )
            twin_options = noise_twin.NoiseTwinOptions(
                enabled=True,
                model_path=twin_options.model_path,
                already_processed=True,
                require_validated=twin_options.require_validated,
            )
        if twin_options.already_processed:
            evidence_path = str((noise_twin_options or {}).get("evidence_path", "") or "")
            if not evidence_path:
                candidate = os.path.splitext(source)[0] + "_innovation.mp4"
                if os.path.exists(candidate):
                    evidence_path = candidate
        else:
            message = f"Camera Digital Twin前処理を開始: {os.path.basename(source)}"
            print(message)
            if progress_callback:
                progress_callback((message, None))

            def twin_progress(done: int, total: int) -> None:
                if progress_callback and (done == total or done % 100 == 0):
                    progress_callback((f"NoiseTwin前処理 {done}/{max(total, 0)}", None))

            prepared_video = noise_twin.prepare_video(
                source,
                twin_options.model_path,
                correction=fixed_pattern_correction,
                temp_dir=config.TEMP_CLIP_DIR,
                progress_callback=twin_progress,
                require_validated=twin_options.require_validated,
                encoding_options=(noise_twin_options or {}).get("encoding"),
            )
            analysis_source = prepared_video.video_path
            evidence_path = prepared_video.innovation_path
            # The calibration map has already been applied before inference.
            active_fixed_pattern_correction = None
            save_options["noise_twin_metrics"] = dict(prepared_video.metrics)
        metadata = noise_twin.load_metadata(twin_options.model_path)
        if twin_options.require_validated and not metadata.validation.validated:
            raise noise_twin.NoiseTwinError("検証基準を満たしていないNoiseTwinモデルです。")
        save_options["noise_twin_metadata"] = {
            "model_id": metadata.model_id,
            "architecture": metadata.architecture,
            "format_version": metadata.format_version,
            "fixed_pattern_sha256": metadata.fixed_pattern_sha256,
            "validation": dict(vars(metadata.validation)),
        }
        save_options.setdefault(
            "noise_twin_metrics",
            {
                "noise_sigma": 0.0,
                "innovation_max": 0.0,
                "flux_retention": metadata.validation.flux_retention,
            },
        )
    elif mean_window:
        save_options["temporal_mean_frames"] = mean_window
        mean_marker = temporal_mean.load_processing_marker(source)
        already_mean_processed = twin_options.already_processed or (
            mean_marker is not None
            and mean_marker.get("method") == "temporal_mean"
            and int(mean_marker.get("frames", 0)) == mean_window
        )
        if not already_mean_processed:
            message = f"{mean_window}フレーム時間平均前処理を開始: {os.path.basename(source)}"
            print(message)
            if progress_callback:
                progress_callback((message, None))

            def mean_progress(done: int, total: int) -> None:
                if progress_callback and (done == total or done % 100 == 0):
                    progress_callback(
                        (f"{mean_window}フレーム平均 {done}/{max(total, 0)}", None)
                    )

            prepared_mean = temporal_mean.prepare_video(
                source,
                mean_window,
                correction=fixed_pattern_correction,
                temp_dir=config.TEMP_CLIP_DIR,
                progress_callback=mean_progress,
                encoding_options=(noise_twin_options or {}).get("encoding"),
            )
            analysis_source = prepared_mean.video_path
            active_fixed_pattern_correction = None
            save_options["temporal_mean_metrics"] = dict(prepared_mean.metrics)

    if not is_rtsp and save_options.get('auto_video_mask_enabled', False):
        try:
            automatic_mask, preview_path, mask_stats = automatic_video_mask.create_auto_mask(
                analysis_source,
                save_options.get('auto_video_mask_cache_dir', config.AUTO_VIDEO_MASK_CACHE_DIR),
            )
            mask = automatic_video_mask.combine_masks(mask, automatic_mask)
            message = (
                f"自動マスク適用: 空領域 {mask_stats.get('sky_fraction', 0.0) * 100:.1f}% "
                f"(プレビュー: {preview_path})"
            )
            print(message)
            if progress_callback:
                progress_callback((message, None))
        except Exception as auto_mask_error:
            message = f"自動マスク生成に失敗したため既存マスクで続行します: {auto_mask_error}"
            print(f"警告: {message}")
            if progress_callback:
                progress_callback((message, None))

    effective_use_plate_solve = use_plate_solve and global_wcs_info is not None
    if use_plate_solve and not effective_use_plate_solve:
        message = "WCS情報がないためプレートソルブは使用されません。"
        print(f"警告: {message}")
        if progress_callback:
            progress_callback((message, None))

    cap = _open_video_capture(analysis_source, decoder_threads)
    if not cap.isOpened():
        message = f"動画/ストリームを開けませんでした: {source}"
        print(f"エラー: {message}")
        if progress_callback:
            progress_callback((message, None))
        raise IOError(message)

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    if frame_rate <= 0:
        frame_rate = config.DEFAULT_FPS
        print(f"警告: FPS取得失敗、デフォルト値 {frame_rate} を使用します。 ({source})")

    total_frames = -1
    if not is_rtsp:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print(f"警告: 総フレーム数が取得できませんでした ({source})。")

    if is_rtsp:
        base_datetime = datetime.now()
        video_path_for_naming = "rtsp_stream"
    else:
        video_path_for_naming = source
        # RTSP録画ファイルは、書き込み開始時に作られるファイルの作成日時を開始時刻に使う。
        # 作成日時が取得できない場合のみ、更新日時から動画長を引く従来方式へフォールバックする。
        is_rtsp_recorded_file_for_time = config.RTSP_SAVE_ROOT in source
        if is_rtsp_recorded_file_for_time:
            base_datetime, time_source = get_rtsp_recording_start_datetime(
                source, total_frames, frame_rate
            )
            if base_datetime is not None:
                print(
                    f"[RTSP録画] {time_source}から基準時刻を取得: "
                    f"開始={base_datetime.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}"
                )
            else:
                print("警告: ファイル日時から基準時刻を取得できませんでした。パスから解析します。")
                base_datetime = astrometry.extract_datetime_from_video_path(video_path_for_naming)
        else:
            base_datetime = astrometry.extract_datetime_from_video_path(video_path_for_naming)
        if base_datetime is None:
            print(f"警告: 動画パスから基準日時を解析できませんでした。現在時刻を使用します。({video_path_for_naming})")
            base_datetime = datetime.now()

    if progress_callback:
        progress_callback((f'"{source}" の処理を開始...', None))

    processed_clips_info: List[Dict[str, Any]] = []
    detection_counter = 0

    try:
        evidence_cap = _open_video_capture(evidence_path, decoder_threads) if evidence_path else None
        diff_generator = image_processing.create_diff_images(
            cap, interval, duration, 1.0, buffer_duration, is_rtsp, cancel_flag,
            evidence_cap=evidence_cap,
        )

        last_read_frame_index_by_diff = -1

        for diff_data in diff_generator:
            try:
                if cancel_flag is not None and cancel_flag.is_set():
                    print("差分画像ループ中にキャンセルされました。")
                    break

                if is_rtsp:
                    diff_img, frame_idx_tuple, brightness_composite, median_image, rtsp_buffer, current_frame_index = diff_data
                else:
                    diff_img, frame_idx_tuple, brightness_composite, median_image = diff_data
                    rtsp_buffer = None
                    current_frame_index = frame_idx_tuple[1]
                    last_read_frame_index_by_diff = max(last_read_frame_index_by_diff, current_frame_index)

                start_index_diff, end_index_diff = frame_idx_tuple

                # RTSP時、またはRTSP録画ファイル時は専用パラメータを使用（ノイズ対策）
                # RTSP録画ファイルのパスにRTSP_SAVE_ROOTが含まれているかチェック
                is_rtsp_recorded_file = not is_rtsp and config.RTSP_SAVE_ROOT in source
                use_rtsp_params = is_rtsp or is_rtsp_recorded_file
            
                if use_rtsp_params:
                    effective_min_length = config.RTSP_MIN_LINE_LENGTH
                    effective_hough_threshold = config.RTSP_HOUGH_THRESHOLD
                    effective_canny_thresh1 = config.RTSP_CANNY_THRESH1
                    effective_canny_thresh2 = config.RTSP_CANNY_THRESH2
                else:
                    effective_min_length = min_length
                    effective_hough_threshold = 25  # デフォルト値
                    effective_canny_thresh1 = 50    # デフォルト値
                    effective_canny_thresh2 = 150   # デフォルト値

                lines = image_processing.detect_lines(
                    diff_img, min_length=effective_min_length, mask=mask, roi=roi, cancel_flag=cancel_flag,
                    canny_thresh1=effective_canny_thresh1, canny_thresh2=effective_canny_thresh2,
                    hough_threshold=effective_hough_threshold
                )

                if not lines:
                    continue

                # 並列処理が有効な場合: 候補を収集してからバッチ処理（動画ファイル・RTSPストリーム両対応）
                if config.RTSP_PARALLEL_ENABLED and len(lines) > 1:
                    # 重複を除外しながら候補を収集
                    valid_candidates = []
                    for line in lines:
                        if cancel_flag is not None and cancel_flag.is_set(): break
                        (x1, y1), (x2, y2) = line
                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                        is_duplicate = False
                        for existing_clip in processed_clips_info:
                            ex_cx, ex_cy = existing_clip.get('center', (None, None))
                            if ex_cx is not None:
                                dist = np.sqrt((cx - ex_cx)**2 + (cy - ex_cy)**2)
                                if dist < config.DUPLICATE_DETECTION_THRESHOLD:
                                    is_duplicate = True
                                    break
                        # 収集済み候補とも重複チェック
                        for vc in valid_candidates:
                            if np.sqrt((cx - vc['cx'])**2 + (cy - vc['cy'])**2) < config.DUPLICATE_DETECTION_THRESHOLD:
                                is_duplicate = True
                                break
                        if not is_duplicate:
                            line_length = int(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))
                            valid_candidates.append({
                                'line': line, 'cx': cx, 'cy': cy,
                                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                'length': line_length
                            })

                    if valid_candidates:
                        print(f"\n--- 並列処理: {len(valid_candidates)}件の候補を検出 (差分フレーム範囲 {start_index_diff}-{end_index_diff}) ---")
                        print(f"[DEBUG] 検出パラメータ: min_length={effective_min_length}, hough={effective_hough_threshold}, canny=({effective_canny_thresh1}, {effective_canny_thresh2})")
                        print(f"[DEBUG] 検出された候補一覧:")
                        for i, cand in enumerate(valid_candidates):
                            print(f"  [{i+1}] 中心:({cand['cx']}, {cand['cy']}) 線:({cand['x1']},{cand['y1']})-({cand['x2']},{cand['y2']}) 長さ:{cand['length']}px")
                    
                        # 詳細検出用フレーム範囲を計算
                        coarse_detection_center_frame = (start_index_diff + end_index_diff) // 2
                        finer_detect_duration_sec = config.FINER_DETECT_WINDOW_SECONDS
                        finer_detect_half_frames = int(finer_detect_duration_sec / 2 * frame_rate)
                        start_frame_finer = max(0, coarse_detection_center_frame - finer_detect_half_frames)
                        end_frame_finer = coarse_detection_center_frame + finer_detect_half_frames
                    
                        if not is_rtsp and total_frames > 0:
                            end_frame_finer = min(total_frames - 1, end_frame_finer)
                            start_frame_finer = max(0, end_frame_finer - int(finer_detect_duration_sec * frame_rate) + 1)
                    
                        # フレームデータを取得（RTSPバッファまたは動画ファイルから）
                        frames_for_finer_detect = []
                        actual_start_finer = start_frame_finer
                        buffer_start_frame_finer_parallel = -1
                        buffer_end_frame_finer_parallel = -1
                    
                        if is_rtsp and rtsp_buffer is not None:
                            # RTSPバッファから取得
                            buffer_end_frame_finer_parallel = current_frame_index
                            buffer_start_frame_finer_parallel = buffer_end_frame_finer_parallel - len(rtsp_buffer) + 1
                            actual_start_finer = max(start_frame_finer, buffer_start_frame_finer_parallel)
                            actual_end_finer = min(end_frame_finer, buffer_end_frame_finer_parallel)
                        
                            if actual_start_finer > actual_end_finer:
                                print(f"警告: 詳細検出用のフレーム範囲がRTSPバッファに含まれていません。スキップします。")
                                continue
                        
                            rel_start_finer = actual_start_finer - buffer_start_frame_finer_parallel
                            rel_end_finer = actual_end_finer - buffer_start_frame_finer_parallel
                            frames_for_finer_detect = list(rtsp_buffer)[rel_start_finer : rel_end_finer + 1]
                            print(f"RTSPバッファから詳細検出用フレームを取得: {len(frames_for_finer_detect)} フレーム")
                        else:
                            # 動画ファイルから取得
                            if not cap.isOpened():
                                cap = _open_video_capture(analysis_source, decoder_threads)
                                if not cap.isOpened():
                                    print(f"エラー: 動画を再オープンできませんでした: {source}")
                                    continue
                        
                            print(f"動画ファイルから詳細検出用フレームを取得: {start_frame_finer} - {end_frame_finer}")
                            if not cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_finer):
                                print(f"警告: フレーム {start_frame_finer} へのシークに失敗した可能性があります。")
                        
                            target_frame_count = end_frame_finer - start_frame_finer + 1
                            read_count = 0
                            max_read_attempts = target_frame_count + int(frame_rate)
                        
                            while len(frames_for_finer_detect) < target_frame_count and read_count < max_read_attempts:
                                if cancel_flag is not None and cancel_flag.is_set(): break
                                ret, frame = cap.read()
                                if not ret: break
                                read_count += 1
                                current_f_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                                if start_frame_finer <= current_f_idx <= end_frame_finer:
                                    frames_for_finer_detect.append(frame)
                                elif current_f_idx > end_frame_finer: break
                        
                            print(f"動画ファイルから {len(frames_for_finer_detect)} フレームを取得")
                    
                        if not frames_for_finer_detect:
                            print("警告: 詳細検出用のフレームを取得できませんでした。逐次処理にフォールバック。")
                            # フォールバックして逐次処理へ（continueしないで、下の for line in lines へ進む）
                        else:
                            # 並列処理実行
                            if mask is not None:
                                masked_frames_finer = []
                                for frame in frames_for_finer_detect:
                                    try:
                                        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0])).astype(np.uint8)
                                        masked_frames_finer.append(cv2.bitwise_and(frame, frame, mask=mask_resized))
                                    except cv2.error:
                                        masked_frames_finer.append(frame)
                                frames_for_finer_detect = masked_frames_finer
                        
                            img_h, img_w = diff_img.shape[:2]
                        
                            # Frames are already resident in this process.  A process
                            # pool copied them and, under spawn, imported model.py in
                            # every child (creating an additional MPS model each time).
                            # OpenCV/NumPy release the GIL for the heavy work, so a
                            # small shared-memory thread pool is both faster and safer.
                            num_workers = int(
                                save_options.get(
                                    'finer_detection_workers',
                                    config.RTSP_PARALLEL_WORKERS or max(1, cpu_count() // 2),
                                )
                            )
                            num_workers = max(1, min(num_workers, len(valid_candidates)))
                            print(f"並列処理開始: {num_workers}ワーカーで{len(valid_candidates)}件を処理中...")
                        
                            # 並列処理で確認された候補を収集
                            confirmed_candidates_parallel = []
                        
                            def collect_finer_result(cand, line_detected_indices, cutout_rect_finer):
                                if line_detected_indices:
                                    print(f"\n--- 詳細検出確認 中心: ({cand['cx']}, {cand['cy']}) ---")
                                    confirmed_candidates_parallel.append({
                                        'cand': cand,
                                        'line_detected_indices': line_detected_indices,
                                        'cutout_rect_finer': cutout_rect_finer,
                                    })
                                else:
                                    print(f"候補 ({cand['cx']}, {cand['cy']}): 詳細検出では流星は見つかりませんでした。")

                            worker_kwargs = {
                                'frames_data': frames_for_finer_detect,
                                'img_h': img_h,
                                'img_w': img_w,
                                'actual_start_frame_index': actual_start_finer,
                                'composite_step': max(1, int(frame_rate / 5)),
                                'cut_size': config.FINER_CUTOUT_SIZE,
                                'border_margin': config.BORDER_SIZE,
                                'finer_min_length': config.FINER_DETECT_MIN_LENGTH,
                            }
                            if num_workers == 1:
                                for cand in valid_candidates:
                                    if cancel_flag is not None and cancel_flag.is_set():
                                        break
                                    try:
                                        result = _process_finer_detection_worker(
                                            cx=cand['cx'], cy=cand['cy'], candidate_line=cand['line'],
                                            **worker_kwargs,
                                        )
                                        collect_finer_result(cand, *result)
                                    except Exception as e:
                                        print(f"詳細処理中にエラー ({cand['cx']}, {cand['cy']}): {e}")
                            else:
                                with ThreadPoolExecutor(
                                    max_workers=num_workers,
                                    thread_name_prefix="finer-detect",
                                ) as executor:
                                    futures = {
                                        executor.submit(
                                            _process_finer_detection_worker,
                                            cx=cand['cx'], cy=cand['cy'], candidate_line=cand['line'],
                                            **worker_kwargs,
                                        ): cand
                                        for cand in valid_candidates
                                    }
                                    for future in as_completed(futures):
                                        if cancel_flag is not None and cancel_flag.is_set():
                                            break
                                        cand = futures[future]
                                        try:
                                            collect_finer_result(cand, *future.result())
                                        except Exception as e:
                                            print(f"並列処理中にエラー ({cand['cx']}, {cand['cy']}): {e}")
                        
                            print(f"並列処理完了: {len(confirmed_candidates_parallel)}件の検出を確認")
                        
                            # 確認された候補に対して保存処理を実行
                            for confirmed in confirmed_candidates_parallel:
                                if cancel_flag is not None and cancel_flag.is_set():
                                    break
                            
                                cand = confirmed['cand']
                                line_detected_indices = confirmed['line_detected_indices']
                                cutout_rect_finer = confirmed['cutout_rect_finer']
                                cx, cy = cand['cx'], cand['cy']
                                x1, y1, x2, y2 = cand['x1'], cand['y1'], cand['x2'], cand['y2']
                            
                                detection_start_frame = min(line_detected_indices)
                                detection_end_frame = max(line_detected_indices)
                                print(f"詳細検出によるフレーム範囲: {detection_start_frame} - {detection_end_frame}")
                            
                                padding_seconds = config.FINER_DETECT_PADDING_SECONDS
                                padding_frames = int(padding_seconds * frame_rate)
                                adjusted_start_frame = max(0, detection_start_frame - padding_frames)
                                adjusted_end_frame = detection_end_frame + padding_frames
                            
                                if not is_rtsp and total_frames > 0:
                                    adjusted_end_frame = min(total_frames - 1, adjusted_end_frame)
                                    clip_actual_duration_frames = (detection_end_frame - detection_start_frame) + 2 * padding_frames
                                    adjusted_start_frame = max(0, adjusted_end_frame - clip_actual_duration_frames + 1)
                            
                                print(f"最終クリップフレーム範囲 (パディング込): {adjusted_start_frame} - {adjusted_end_frame}")
                            
                                # 最終クリップ用フレームを取得
                                final_frames_for_clip: List[np.ndarray] = []
                                if is_rtsp:
                                    actual_start_final = max(adjusted_start_frame, buffer_start_frame_finer_parallel)
                                    actual_end_final = min(adjusted_end_frame, buffer_end_frame_finer_parallel)
                                
                                    if actual_start_final > actual_end_final:
                                        print(f"警告: 最終クリップ用のフレーム範囲がRTSPバッファに含まれていません。スキップします。")
                                        continue
                                
                                    rel_start_final = actual_start_final - buffer_start_frame_finer_parallel
                                    rel_end_final = actual_end_final - buffer_start_frame_finer_parallel
                                    final_frames_for_clip = list(rtsp_buffer)[rel_start_final : rel_end_final + 1]
                                    print(f"RTSPバッファから最終クリップフレームを取得: {actual_start_final} - {actual_end_final}, {len(final_frames_for_clip)} フレーム")
                                else:
                                    if not cap.isOpened():
                                        cap = _open_video_capture(analysis_source, decoder_threads)
                                        if not cap.isOpened():
                                            continue
                                
                                    print(f"動画ファイルから最終クリップフレームを取得: {adjusted_start_frame} - {adjusted_end_frame}")
                                    if not cap.set(cv2.CAP_PROP_POS_FRAMES, adjusted_start_frame):
                                        print(f"警告: フレーム {adjusted_start_frame} へのシークに失敗した可能性があります。")
                                
                                    read_count = 0
                                    target_frame_indices = list(range(adjusted_start_frame, adjusted_end_frame + 1))
                                    max_read_attempts = len(target_frame_indices) + int(frame_rate)
                                
                                    while len(final_frames_for_clip) < len(target_frame_indices) and read_count < max_read_attempts:
                                        if cancel_flag is not None and cancel_flag.is_set(): break
                                        ret, frame = cap.read()
                                        if not ret: break
                                        read_count += 1
                                        current_f_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                                        if adjusted_start_frame <= current_f_idx <= adjusted_end_frame:
                                            final_frames_for_clip.append(frame)
                                        elif current_f_idx > adjusted_end_frame: break
                            
                                if not final_frames_for_clip:
                                    print("警告: 最終クリップ用のフレームを取得できませんでした。スキップします。")
                                    continue
                            
                                if mask is not None:
                                    masked_frames_final = []
                                    for frame in final_frames_for_clip:
                                        try:
                                            mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0])).astype(np.uint8)
                                            masked_frames_final.append(cv2.bitwise_and(frame, frame, mask=mask_resized))
                                        except cv2.error as e:
                                            print(f"最終クリップフレームへのマスク適用中にエラー: {e}")
                                            masked_frames_final.append(frame)
                                    final_frames_for_clip = masked_frames_final

                                # Keep the detector's original one-second difference image for
                                # inference.  The classifier was trained on this representation;
                                # the enhanced difference below is strictly for saved artifacts.
                                classification_diff_img = diff_img
                                saved_diff_img = diff_img
                                ml_source_frames = final_frames_for_clip
                                enhancement_result: Optional[video_enhancement.EnhancementResult] = None
                                if active_fixed_pattern_correction is not None:
                                    enhancement_result = video_enhancement.enhance_frames(
                                        final_frames_for_clip,
                                        active_fixed_pattern_correction,
                                        detected_line=((x1, y1), (x2, y2)),
                                    )
                                    final_frames_for_clip = enhancement_result.frames
                                    brightness_composite, saved_diff_img = _brightness_composite_and_diff(final_frames_for_clip)
                                    print(
                                        "保存物補正: 固定パターン "
                                        f"{enhancement_result.correction_strength:.3f}倍 + 21フレーム平均 "
                                        f"(時間ノイズ {enhancement_result.noise_reduction_percent:.1f}%低減)"
                                    )
                            
                                detection_counter += 1
                                print(f"--- 検出確定 {detection_counter} (並列処理) ---")
                            
                                cut_size = config.CUTOUT_SIZE
                                half_cut = cut_size // 2
                                border_margin = config.BORDER_SIZE
                            
                                x_start_cut = max(border_margin, cx - half_cut)
                                y_start_cut = max(border_margin, cy - half_cut)
                                x_end_cut = min(img_w - border_margin, cx + half_cut)
                                y_end_cut = min(img_h - border_margin, cy + half_cut)
                            
                                current_w = x_end_cut - x_start_cut
                                current_h = y_end_cut - y_start_cut
                                if current_w < cut_size:
                                    diff_w = cut_size - current_w
                                    x_start_cut = max(border_margin, x_start_cut - diff_w // 2)
                                    x_end_cut = min(img_w - border_margin, x_start_cut + cut_size)
                                    if x_end_cut - x_start_cut < cut_size: x_start_cut = x_end_cut - cut_size
                                if current_h < cut_size:
                                    diff_h = cut_size - current_h
                                    y_start_cut = max(border_margin, y_start_cut - diff_h // 2)
                                    y_end_cut = min(img_h - border_margin, y_start_cut + cut_size)
                                    if y_end_cut - y_start_cut < cut_size: y_start_cut = y_end_cut - cut_size
                                x_start_cut = max(border_margin, x_end_cut - cut_size)
                                y_start_cut = max(border_margin, y_end_cut - cut_size)
                                cutout_rect = (int(x_start_cut), int(y_start_cut), int(x_start_cut + cut_size), int(y_start_cut + cut_size))
                                x_start_cut, y_start_cut, x_end_cut, y_end_cut = cutout_rect
                            
                                diff_cutout = _extract_diff_cutout(saved_diff_img, cutout_rect, cut_size)
                                classification_diff_cutout = _extract_diff_cutout(
                                    classification_diff_img, cutout_rect, cut_size
                                )
                            
                                temp_dir = config.TEMP_CLIP_DIR
                                os.makedirs(temp_dir, exist_ok=True)
                                temp_diff_image_path = os.path.join(temp_dir, f"temp_diff_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.jpg")
                                cv2.imwrite(temp_diff_image_path, classification_diff_cutout)
                                probability = model.predict_meteor_probability(temp_diff_image_path)
                                try: os.remove(temp_diff_image_path)
                                except OSError as e: print(f"警告: 一時差分ファイルの削除に失敗: {e}")
                                print(f"流星確率: {probability:.4f}")
                            
                                is_meteor = probability >= config.METEOR_PROBABILITY_THRESHOLD
                                save_dir = meteor_save_path if is_meteor else not_meteor_save_path
                                detection_label = "meteor" if is_meteor else "not_meteor"
                                os.makedirs(save_dir, exist_ok=True)
                            
                                detection_frame_for_time = (adjusted_start_frame + adjusted_end_frame) // 2
                                detection_time_seconds_rel = detection_frame_for_time / frame_rate
                                detection_datetime = base_datetime + timedelta(seconds=detection_time_seconds_rel)
                                display_timestamp = detection_datetime.strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
                            
                                base_filename = (
                                    f"{detection_datetime.strftime('%Y%m%d_%H%M%S%f')[:-3]}"
                                    f"_{detection_label}_{detection_counter}_prob{probability:.2f}"
                                )
                                saved_paths = {}
                                denoise_metrics: Optional[Dict[str, float]] = None

                                if save_options.get('ml_training_export_enabled', False):
                                    try:
                                        ml_event_dir = ml_training_data.export_training_event(
                                            root_dir=save_options.get('ml_training_data_root', config.DEFAULT_ML_TRAINING_DATA_ROOT),
                                            predicted_label=detection_label,
                                            probability=probability,
                                            event_id=base_filename,
                                            source=source,
                                            detection_time=detection_datetime,
                                            frames=ml_source_frames,
                                            classification_diff=classification_diff_img,
                                            cutout_rect=cutout_rect,
                                            cutout_size=cut_size,
                                            frame_rate=frame_rate,
                                            detected_line=((x1, y1), (x2, y2)),
                                            frame_range=(adjusted_start_frame, adjusted_end_frame),
                                        )
                                        saved_paths['ml_training_event'] = ml_event_dir
                                        print(f"機械学習向けイベントを保存しました: {ml_event_dir}")
                                    except Exception as e_ml:
                                        print(f"機械学習向けイベントの保存に失敗: {e_ml}")
                            
                                if save_options.get('video', False):
                                    output_clip_path = os.path.join(save_dir, f"{base_filename}.mp4")
                                    fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_FOURCC)
                                    out_clip = cv2.VideoWriter(output_clip_path, fourcc, frame_rate, (cut_size, cut_size))
                                    if out_clip.isOpened():
                                        for frame in final_frames_for_clip:
                                            cut_frame = frame[y_start_cut:y_end_cut, x_start_cut:x_end_cut]
                                            if cut_frame.shape[0] != cut_size or cut_frame.shape[1] != cut_size:
                                                cut_frame_resized = cv2.resize(cut_frame, (cut_size, cut_size))
                                            else:
                                                cut_frame_resized = cut_frame
                                            out_clip.write(cut_frame_resized)
                                        out_clip.release()
                                        saved_paths['video'] = output_clip_path
                                        print(f"動画クリップを保存しました: {output_clip_path}")
                                    else:
                                        print(f"エラー: 動画クリップの書き込み開始に失敗 ({output_clip_path})")
                            
                                if save_options.get('full_video', False) and final_frames_for_clip:
                                    full_video_path = os.path.join(save_dir, f"{base_filename}_full.mp4")
                                    clip_start_frame = actual_start_final if is_rtsp else adjusted_start_frame
                                    clip_start_datetime = base_datetime + timedelta(
                                        seconds=clip_start_frame / frame_rate
                                    )
                                    if write_full_size_video(
                                        final_frames_for_clip, full_video_path, frame_rate,
                                        clip_start_datetime, save_options,
                                    ):
                                        saved_paths['full_video'] = full_video_path
                                        print(f"フルサイズ動画を保存しました: {full_video_path}")
                                    else:
                                        print(f"エラー: フルサイズ動画の書き込み開始に失敗 ({full_video_path})")

                                if (
                                    active_fixed_pattern_correction is None
                                    and save_options.get('denoised_full_video', False)
                                    and final_frames_for_clip
                                ):
                                    denoised_full_video_path = os.path.join(save_dir, f"{base_filename}_full_denoised.mp4")
                                    clip_start_frame = actual_start_final if is_rtsp else adjusted_start_frame
                                    clip_start_datetime = base_datetime + timedelta(
                                        seconds=clip_start_frame / frame_rate
                                    )
                                    denoise_metrics = write_denoised_full_size_video(
                                        final_frames_for_clip,
                                        denoised_full_video_path,
                                        frame_rate,
                                        clip_start_datetime,
                                        save_options,
                                        ((x1, y1), (x2, y2)),
                                    )
                                    if denoise_metrics is not None:
                                        saved_paths['denoised_full_video'] = denoised_full_video_path
                                        print(
                                            "ノイズ低減フルサイズ動画を保存しました: "
                                            f"{denoised_full_video_path} "
                                            f"(推定ノイズ {denoise_metrics['before_sigma']:.2f} -> "
                                            f"{denoise_metrics['after_sigma']:.2f}, "
                                            f"{denoise_metrics['reduction_percent']:.1f}%低減)"
                                        )
                                    else:
                                        print(f"エラー: ノイズ低減動画の書き込みに失敗 ({denoised_full_video_path})")
                            
                                if save_options.get('cutout', False):
                                    cutout_diff_path = os.path.join(save_dir, f"{base_filename}_cutout_diff.jpg")
                                    if cv2.imwrite(cutout_diff_path, diff_cutout):
                                        saved_paths['cutout'] = cutout_diff_path
                                        print(f"切り出し差分画像を保存しました: {cutout_diff_path}")
                                    else:
                                        print(f"エラー: 切り出し差分画像の保存に失敗 ({cutout_diff_path})")
                            
                                if save_options.get('full', False):
                                    full_diff_path = os.path.join(save_dir, f"{base_filename}_full_diff.jpg")
                                    if cv2.imwrite(full_diff_path, saved_diff_img):
                                        saved_paths['full'] = full_diff_path
                                        print(f"全体差分画像を保存しました: {full_diff_path}")
                                    else:
                                        print(f"エラー: 全体差分画像の保存に失敗 ({full_diff_path})")
                            
                                if save_options.get('composite', False) and final_frames_for_clip:
                                    try:
                                        composite_image, _ = _brightness_composite_and_diff(final_frames_for_clip)
                                        composite_path = os.path.join(save_dir, f"{base_filename}_composite.jpg")
                                        if cv2.imwrite(composite_path, composite_image):
                                            saved_paths['composite'] = composite_path
                                            print(f"比較明合成画像を保存しました: {composite_path}")
                                        
                                            wcs_data_for_annotation = global_wcs_info if effective_use_plate_solve else {}
                                            annotated_path = astrometry.annotate_image_with_wcs(
                                                image_path=composite_path, wcs_info=wcs_data_for_annotation,
                                                line_centers=[(cx, cy)], detection_datetime=detection_datetime,
                                                timestamp=display_timestamp, cancel_flag=cancel_flag,
                                                detected_line=((x1, y1), (x2, y2)),
                                            )
                                            if annotated_path:
                                                saved_paths['annotated'] = annotated_path
                                                print(f"注釈付き画像を保存しました: {annotated_path}")
                                        else:
                                            print(f"エラー: 比較明合成画像の保存に失敗 ({composite_path})")
                                    except Exception as e_comp:
                                        print(f"比較明合成または注釈処理中にエラー: {e_comp}")
                            
                                if save_options.get('info', False):
                                    info_path = os.path.join(save_dir, f"{base_filename}_info.txt")
                                    try:
                                        ra_start_str, dec_start_str, ra_end_str, dec_end_str = "N/A", "N/A", "N/A", "N/A"
                                        if effective_use_plate_solve and global_wcs_info.get('wcs_file'):
                                            try:
                                                with fits.open(global_wcs_info['wcs_file']) as hdul:
                                                    wcs = WCS(hdul[0].header, relax=True, fix=False)
                                                    sky_coord_start = wcs.pixel_to_world(x1, y1)
                                                    sky_coord_end = wcs.pixel_to_world(x2, y2)
                                                    ra_start_str, dec_start_str = f"{sky_coord_start.ra.deg:.6f}", f"{sky_coord_start.dec.deg:.6f}"
                                                    ra_end_str, dec_end_str = f"{sky_coord_end.ra.deg:.6f}", f"{sky_coord_end.dec.deg:.6f}"
                                            except Exception as e_wcs_read:
                                                print(f"情報ファイル作成中のWCS読み込み/変換エラー: {e_wcs_read}")
                                    
                                        with open(info_path, 'w', encoding='utf-8') as f:
                                            f.write(f"Source: {source}\n")
                                            f.write(f"Detection Time (UTC): {detection_datetime.isoformat()}Z\n")
                                            f.write(f"Frame Range (Clip): {adjusted_start_frame} - {adjusted_end_frame}\n")
                                            f.write(f"Meteor Probability: {probability:.6f}\n")
                                            f.write(f"Detected Line Center (px): ({cx:.2f}, {cy:.2f})\n")
                                            if denoise_metrics is not None:
                                                f.write(f"Denoise Noise Sigma Before: {denoise_metrics['before_sigma']:.4f}\n")
                                                f.write(f"Denoise Noise Sigma After: {denoise_metrics['after_sigma']:.4f}\n")
                                                f.write(f"Denoise Noise Reduction (%): {denoise_metrics['reduction_percent']:.2f}\n")
                                            if enhancement_result is not None:
                                                f.write(f"Adaptive Fixed Pattern Strength: {enhancement_result.correction_strength:.4f}\n")
                                                f.write("Temporal Enhancement: 21-frame mean\n")
                                                f.write(f"Enhancement Noise Sigma Before: {enhancement_result.noise_before:.4f}\n")
                                                f.write(f"Enhancement Noise Sigma After: {enhancement_result.noise_after:.4f}\n")
                                                f.write(f"Enhancement Noise Reduction (%): {enhancement_result.noise_reduction_percent:.2f}\n")
                                            _write_noise_twin_info(
                                                f,
                                                save_options,
                                                math.hypot(x2 - x1, y2 - y1)
                                                / max(1.0, math.hypot(img_w, img_h)),
                                                float(diff_img.max()) / 16.0 if evidence_path else None,
                                            )
                                            f.write(f"RA Start (deg): {ra_start_str}\nDec Start (deg): {dec_start_str}\n")
                                            f.write(f"RA End (deg): {ra_end_str}\nDec End (deg): {dec_end_str}\n")
                                            for key, path in saved_paths.items(): f.write(f"Saved {key.capitalize()} Path: {path}\n")
                                    
                                        saved_paths['info'] = info_path
                                        print(f"検出情報ファイルを保存しました: {info_path}")
                                    except Exception as e_info:
                                        print(f"エラー: 検出情報ファイルの保存中にエラー: {e_info}")
                            
                                if save_options.get('summary', False):
                                    if all(key in saved_paths for key in ['composite', 'annotated', 'video']):
                                        summary_video_path = os.path.join(save_dir, f"{base_filename}_summary.mp4")
                                        try:
                                            orig_h, orig_w = final_frames_for_clip[0].shape[:2]
                                            summary_created = video_creation.create_summary_video(
                                                summary_video_config=summary_video_config,
                                                composite_image_path=saved_paths['composite'],
                                                annotated_image_path=saved_paths['annotated'],
                                                cutout_video_path=saved_paths['video'],
                                                output_video_path=summary_video_path,
                                                cutout_rect=cutout_rect,
                                                frame_rate=frame_rate,
                                                output_resolution=(orig_w, orig_h),
                                                full_video_path=(
                                                    saved_paths.get('denoised_full_video')
                                                    or saved_paths.get('full_video')
                                                ),
                                                detected_line=((x1, y1), (x2, y2)),
                                            )
                                            if summary_created:
                                                saved_paths['summary'] = summary_video_path
                                                print(f"概要動画を保存しました: {summary_video_path}")
                                            else:
                                                print("警告: 概要動画を作成できなかったため、保存結果には追加しません。")
                                        except Exception as e_summary:
                                            print(f"エラー: 概要動画の生成に失敗しました: {e_summary}")
                                            import traceback; traceback.print_exc()
                                    else:
                                        print("警告: 概要動画の作成に必要なファイルが不足しているため、スキップします。")
                            
                                if progress_callback:
                                    progress_callback((f"検出 {detection_counter}: {detection_label} (Prob: {probability:.2f}) @ {display_timestamp}", None))
                                
                                    try:
                                        threshold = float(getattr(__import__('config'), 'METEOR_PROBABILITY_THRESHOLD', 0.5))
                                    except Exception:
                                        threshold = 0.5
                                
                                    if (not is_meteor) and (probability < threshold):
                                        progress_callback((f"  -> Not Meteor: Probability {probability:.2f}", None))
                                    else:
                                        for key, path in saved_paths.items():
                                            if path:
                                                progress_callback((f"  -> {key.capitalize()}: {os.path.basename(path)}", None))
                            
                                if is_meteor and notify_on_detection:
                                    try: utils.play_notification_sound()
                                    except Exception as e_sound: print(f"通知音の再生中にエラー: {e_sound}")
                            
                                processed_clips_info.append({
                                    'path': saved_paths.get('video'), 'center': (cx, cy), 'probability': probability,
                                    'label': detection_label, 'timestamp': detection_datetime,
                                    'start_frame': adjusted_start_frame, 'end_frame': adjusted_end_frame, 'saved_paths': saved_paths
                                })
                        
                            continue  # 並列処理による保存完了後は逐次ループをスキップ

                for line in lines:
                    if cancel_flag is not None and cancel_flag.is_set(): break

                    (x1, y1), (x2, y2) = line
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                    is_duplicate = False
                    for existing_clip in processed_clips_info:
                        ex_cx, ex_cy = existing_clip.get('center', (None, None))
                        if ex_cx is not None:
                            dist = np.sqrt((cx - ex_cx)**2 + (cy - ex_cy)**2)
                            if dist < config.DUPLICATE_DETECTION_THRESHOLD:
                                is_duplicate = True
                                break
                    if is_duplicate:
                        continue

                    current_detection_id = detection_counter + 1
                    print(f"\n--- 粗検出候補 {current_detection_id} (差分フレーム範囲 {start_index_diff}-{end_index_diff}, 中心: ({cx}, {cy})) ---")

                    coarse_detection_center_frame = (start_index_diff + end_index_diff) // 2
                    finer_detect_duration_sec = config.FINER_DETECT_WINDOW_SECONDS
                    finer_detect_half_frames = int(finer_detect_duration_sec / 2 * frame_rate)

                    start_frame_finer = max(0, coarse_detection_center_frame - finer_detect_half_frames)
                    end_frame_finer = coarse_detection_center_frame + finer_detect_half_frames

                    if not is_rtsp and total_frames > 0:
                        end_frame_finer = min(total_frames - 1, end_frame_finer)
                        start_frame_finer = max(0, end_frame_finer - int(finer_detect_duration_sec * frame_rate) + 1)

                    print(f"詳細検出用フレーム範囲 (推定): {start_frame_finer} - {end_frame_finer}")

                    frames_for_finer_detect: List[np.ndarray] = []
                    buffer_start_frame_finer = -1
                    buffer_end_frame_finer = -1

                    if is_rtsp:
                        if rtsp_buffer is None:
                            print("警告: RTSPバッファが存在しません。詳細検出をスキップします。")
                            continue
                        buffer_end_frame_finer = current_frame_index
                        buffer_start_frame_finer = buffer_end_frame_finer - len(rtsp_buffer) + 1
                        actual_start_finer = max(start_frame_finer, buffer_start_frame_finer)
                        actual_end_finer = min(end_frame_finer, buffer_end_frame_finer)

                        if actual_start_finer > actual_end_finer:
                            print(f"警告: 詳細検出用のフレーム範囲 ({start_frame_finer}-{end_frame_finer}) がRTSPバッファ ({buffer_start_frame_finer}-{buffer_end_frame_finer}) に含まれていません。スキップします。")
                            continue

                        rel_start_finer = actual_start_finer - buffer_start_frame_finer
                        rel_end_finer = actual_end_finer - buffer_start_frame_finer
                        frames_for_finer_detect = list(rtsp_buffer)[rel_start_finer : rel_end_finer + 1]
                        print(f"RTSPバッファから詳細検出用フレームを取得: {actual_start_finer} - {actual_end_finer}, {len(frames_for_finer_detect)} フレーム")

                    else:
                        if not cap.isOpened():
                            print("動画キャプチャが閉じています。再オープンします。")
                            cap = _open_video_capture(analysis_source, decoder_threads)
                            if not cap.isOpened():
                                print(f"エラー: 動画を再オープンできませんでした: {source}")
                                continue
                            last_read_frame_index_by_diff = -1

                        print(f"動画ファイルから詳細検出用フレームを取得: {start_frame_finer} - {end_frame_finer}")
                        if not cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_finer):
                            print(f"警告: フレーム {start_frame_finer} へのシークに失敗した可能性があります。")

                        read_count = 0
                        target_frame_indices = list(range(start_frame_finer, end_frame_finer + 1))
                        max_read_attempts = len(target_frame_indices) + int(frame_rate)

                        while len(frames_for_finer_detect) < len(target_frame_indices) and read_count < max_read_attempts:
                            if cancel_flag is not None and cancel_flag.is_set(): break
                            ret, frame = cap.read()
                            if not ret: break
                            read_count += 1
                            current_f_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                            if start_frame_finer <= current_f_idx <= end_frame_finer:
                                frames_for_finer_detect.append(frame)
                            elif current_f_idx > end_frame_finer: break

                    if not frames_for_finer_detect:
                        print("警告: 詳細検出用のフレームを取得できませんでした。スキップします。")
                        continue

                    if mask is not None:
                        masked_frames_finer = []
                        for frame in frames_for_finer_detect:
                            try:
                                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0])).astype(np.uint8)
                                masked_frames_finer.append(cv2.bitwise_and(frame, frame, mask=mask_resized))
                            except cv2.error as e:
                                print(f"詳細検出フレームへのマスク適用中にエラー: {e}")
                                masked_frames_finer.append(frame)
                        frames_for_finer_detect = masked_frames_finer

                    # 詳細検出の実行（RTSPで並列処理が有効な場合はワーカー関数を使用）
                    img_h, img_w = diff_img.shape[:2]
                    actual_start_frame_index_finer = actual_start_finer if is_rtsp else start_frame_finer
                
                    if is_rtsp and config.RTSP_PARALLEL_ENABLED:
                        # 並列処理モード: ワーカー関数を直接呼び出し（単一候補の場合）
                        # 複数候補の場合はバッチ処理が必要だが、現在の構造では各候補を逐次取得するため
                        # ここでは単一候補に対してワーカー関数を使用
                        print(f"詳細検出を並列処理モードで実行中...")
                        line_detected_indices, cutout_rect_finer = _process_finer_detection_worker(
                            frames_data=frames_for_finer_detect,
                            cx=cx,
                            cy=cy,
                            img_h=img_h,
                            img_w=img_w,
                            actual_start_frame_index=actual_start_frame_index_finer,
                            candidate_line=((x1, y1), (x2, y2)),
                            composite_step=max(1, int(frame_rate / 5)),  # FPSの1/5
                            cut_size=config.FINER_CUTOUT_SIZE,  # 詳細検出用カットアウトサイズ
                            border_margin=config.BORDER_SIZE,
                            finer_min_length=config.FINER_DETECT_MIN_LENGTH
                        )
                        x_start_cut_finer, y_start_cut_finer, x_end_cut_finer, y_end_cut_finer = cutout_rect_finer
                        print(f"詳細検出用カットアウト領域: ({x_start_cut_finer}, {y_start_cut_finer}) - ({x_end_cut_finer}, {y_end_cut_finer})")
                    
                        if line_detected_indices:
                            for idx in line_detected_indices:
                                print(f"  -> finer差分 (フレーム {idx} 付近) カットアウト領域で線検出")
                    else:
                        # 逐次処理モード（従来通り）
                        cut_size_finer = config.FINER_CUTOUT_SIZE  # 詳細検出用カットアウトサイズ
                        half_cut_finer = cut_size_finer // 2
                        border_margin = config.BORDER_SIZE

                        x_start_cut_finer = max(border_margin, cx - half_cut_finer)
                        y_start_cut_finer = max(border_margin, cy - half_cut_finer)
                        x_end_cut_finer = min(img_w - border_margin, cx + half_cut_finer)
                        y_end_cut_finer = min(img_h - border_margin, cy + half_cut_finer)

                        # サイズ調整（FINER_CUTOUT_SIZEを確保）
                        current_w_finer = x_end_cut_finer - x_start_cut_finer
                        current_h_finer = y_end_cut_finer - y_start_cut_finer
                        if current_w_finer < cut_size_finer:
                            diff_w = cut_size_finer - current_w_finer
                            x_start_cut_finer = max(border_margin, x_start_cut_finer - diff_w // 2)
                            x_end_cut_finer = min(img_w - border_margin, x_start_cut_finer + cut_size_finer)
                            if x_end_cut_finer - x_start_cut_finer < cut_size_finer:
                                x_start_cut_finer = x_end_cut_finer - cut_size_finer
                        if current_h_finer < cut_size_finer:
                            diff_h = cut_size_finer - current_h_finer
                            y_start_cut_finer = max(border_margin, y_start_cut_finer - diff_h // 2)
                            y_end_cut_finer = min(img_h - border_margin, y_start_cut_finer + cut_size_finer)
                            if y_end_cut_finer - y_start_cut_finer < cut_size_finer:
                                y_start_cut_finer = y_end_cut_finer - cut_size_finer
                        x_start_cut_finer = max(border_margin, x_end_cut_finer - cut_size_finer)
                        y_start_cut_finer = max(border_margin, y_end_cut_finer - cut_size_finer)

                        print(f"詳細検出用カットアウト領域: ({x_start_cut_finer}, {y_start_cut_finer}) - ({x_end_cut_finer}, {y_end_cut_finer})")

                        composite_frames = []
                        composite_frame_indices = []
                        step = max(1, int(frame_rate / 5))  # FPSの1/5
                        print(f"詳細検出: 比較明合成ステップ = {step}フレーム (FPS={frame_rate:.1f})")
                        for i in range(0, len(frames_for_finer_detect) - (step - 1), step):
                            composite_cutout = _max_composite_cutout(
                                frames_for_finer_detect[i : i + step],
                                (
                                    x_start_cut_finer,
                                    y_start_cut_finer,
                                    x_end_cut_finer,
                                    y_end_cut_finer,
                                ),
                            )
                            composite_frames.append(composite_cutout)
                            composite_frame_indices.append(actual_start_frame_index_finer + i + (step // 2))

                        line_detected_indices = []
                        finer_min_length = config.FINER_DETECT_MIN_LENGTH

                        for i in range(1, len(composite_frames)):
                            if cancel_flag is not None and cancel_flag.is_set(): break
                            # カットアウト領域の差分画像
                            diff_img_finer_cutout = cv2.absdiff(composite_frames[i], composite_frames[i - 1])
                            # カットアウト領域内で直線検出（マスクなし、ROIなし）
                            lines_finer = image_processing.detect_lines(
                                diff_img_finer_cutout, min_length=finer_min_length, cancel_flag=cancel_flag
                            )
                            if lines_finer and _has_matching_finer_line(
                                lines_finer,
                                ((x1, y1), (x2, y2)),
                                (x_start_cut_finer, y_start_cut_finer, x_end_cut_finer, y_end_cut_finer),
                            ):
                                line_detected_indices.append(composite_frame_indices[i])
                                print(f"  -> finer差分 (フレーム {composite_frame_indices[i]} 付近) カットアウト領域で線検出")

                    if cancel_flag is not None and cancel_flag.is_set(): break

                    if not line_detected_indices:
                        print("詳細検出では流星は見つかりませんでした。この候補をスキップします。")
                        continue

                    detection_start_frame = min(line_detected_indices)
                    detection_end_frame = max(line_detected_indices)
                    print(f"詳細検出によるフレーム範囲: {detection_start_frame} - {detection_end_frame}")

                    padding_seconds = config.FINER_DETECT_PADDING_SECONDS
                    padding_frames = int(padding_seconds * frame_rate)
                    adjusted_start_frame = max(0, detection_start_frame - padding_frames)
                    adjusted_end_frame = detection_end_frame + padding_frames

                    if not is_rtsp and total_frames > 0:
                        adjusted_end_frame = min(total_frames - 1, adjusted_end_frame)
                        clip_actual_duration_frames = (detection_end_frame - detection_start_frame) + 2 * padding_frames
                        adjusted_start_frame = max(0, adjusted_end_frame - clip_actual_duration_frames + 1)

                    print(f"最終クリップフレーム範囲 (パディング込): {adjusted_start_frame} - {adjusted_end_frame}")

                    final_frames_for_clip: List[np.ndarray] = []
                    if is_rtsp:
                        actual_start_final = max(adjusted_start_frame, buffer_start_frame_finer)
                        actual_end_final = min(adjusted_end_frame, buffer_end_frame_finer)

                        if actual_start_final > actual_end_final:
                            print(f"警告: 最終クリップ用のフレーム範囲 ({adjusted_start_frame}-{adjusted_end_frame}) がRTSPバッファ ({buffer_start_frame_finer}-{buffer_end_frame_finer}) に含まれていません。スキップします。")
                            continue

                        rel_start_final = actual_start_final - buffer_start_frame_finer
                        rel_end_final = actual_end_final - buffer_start_frame_finer
                        final_frames_for_clip = list(rtsp_buffer)[rel_start_final : rel_end_final + 1]
                        print(f"RTSPバッファから最終クリップフレームを取得: {actual_start_final} - {actual_end_final}, {len(final_frames_for_clip)} フレーム")

                    else:
                        if not cap.isOpened():
                            print("動画キャプチャが閉じています。再オープンします。")
                            cap = _open_video_capture(analysis_source, decoder_threads)
                            if not cap.isOpened(): continue
                            last_read_frame_index_by_diff = -1

                        print(f"動画ファイルから最終クリップフレームを取得: {adjusted_start_frame} - {adjusted_end_frame}")
                        if not cap.set(cv2.CAP_PROP_POS_FRAMES, adjusted_start_frame):
                            print(f"警告: フレーム {adjusted_start_frame} へのシークに失敗した可能性があります。")

                        read_count = 0
                        target_frame_indices = list(range(adjusted_start_frame, adjusted_end_frame + 1))
                        max_read_attempts = len(target_frame_indices) + int(frame_rate)

                        while len(final_frames_for_clip) < len(target_frame_indices) and read_count < max_read_attempts:
                            if cancel_flag is not None and cancel_flag.is_set(): break
                            ret, frame = cap.read()
                            if not ret: break
                            read_count += 1
                            current_f_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                            if adjusted_start_frame <= current_f_idx <= adjusted_end_frame:
                                final_frames_for_clip.append(frame)
                            elif current_f_idx > adjusted_end_frame: break

                    if not final_frames_for_clip:
                        print("警告: 最終クリップ用のフレームを取得できませんでした。スキップします。")
                        continue

                    if mask is not None:
                        masked_frames_final = []
                        for frame in final_frames_for_clip:
                            try:
                                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0])).astype(np.uint8)
                                masked_frames_final.append(cv2.bitwise_and(frame, frame, mask=mask_resized))
                            except cv2.error as e:
                                print(f"最終クリップフレームへのマスク適用中にエラー: {e}")
                                masked_frames_final.append(frame)
                        final_frames_for_clip = masked_frames_final

                    # The classifier sees the same detector evidence: legacy
                    # max/min difference when OFF, innovation when NoiseTwin is ON.
                    classification_diff_img = diff_img
                    saved_diff_img = diff_img
                    ml_source_frames = final_frames_for_clip
                    enhancement_result: Optional[video_enhancement.EnhancementResult] = None
                    if active_fixed_pattern_correction is not None:
                        enhancement_result = video_enhancement.enhance_frames(
                            final_frames_for_clip,
                            active_fixed_pattern_correction,
                            detected_line=((x1, y1), (x2, y2)),
                        )
                        final_frames_for_clip = enhancement_result.frames
                        brightness_composite, saved_diff_img = _brightness_composite_and_diff(final_frames_for_clip)
                        print(
                            "保存物補正: 固定パターン "
                            f"{enhancement_result.correction_strength:.3f}倍 + 21フレーム平均 "
                            f"(時間ノイズ {enhancement_result.noise_reduction_percent:.1f}%低減)"
                        )

                    detection_counter += 1
                    print(f"--- 検出確定 {detection_counter} ---")

                    img_h, img_w = diff_img.shape[:2]
                    cut_size = config.CUTOUT_SIZE
                    half_cut = cut_size // 2
                    border_margin = config.BORDER_SIZE

                    x_start_cut = max(border_margin, cx - half_cut)
                    y_start_cut = max(border_margin, cy - half_cut)
                    x_end_cut = min(img_w - border_margin, cx + half_cut)
                    y_end_cut = min(img_h - border_margin, cy + half_cut)

                    current_w = x_end_cut - x_start_cut
                    current_h = y_end_cut - y_start_cut
                    if current_w < cut_size:
                        diff_w = cut_size - current_w
                        x_start_cut = max(border_margin, x_start_cut - diff_w // 2)
                        x_end_cut = min(img_w - border_margin, x_start_cut + cut_size)
                        if x_end_cut - x_start_cut < cut_size: x_start_cut = x_end_cut - cut_size
                    if current_h < cut_size:
                        diff_h = cut_size - current_h
                        y_start_cut = max(border_margin, y_start_cut - diff_h // 2)
                        y_end_cut = min(img_h - border_margin, y_start_cut + cut_size)
                        if y_end_cut - y_start_cut < cut_size: y_start_cut = y_end_cut - cut_size
                    x_start_cut = max(border_margin, x_end_cut - cut_size)
                    y_start_cut = max(border_margin, y_end_cut - cut_size)
                    cutout_rect = (int(x_start_cut), int(y_start_cut), int(x_start_cut + cut_size), int(y_start_cut + cut_size))
                    x_start_cut, y_start_cut, x_end_cut, y_end_cut = cutout_rect

                    diff_cutout = _extract_diff_cutout(saved_diff_img, cutout_rect, cut_size)
                    classification_diff_cutout = _extract_diff_cutout(
                        classification_diff_img, cutout_rect, cut_size
                    )

                    temp_dir = config.TEMP_CLIP_DIR
                    os.makedirs(temp_dir, exist_ok=True)
                    temp_diff_image_path = os.path.join(temp_dir, f"temp_diff_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.jpg")
                    cv2.imwrite(temp_diff_image_path, classification_diff_cutout)
                    probability = model.predict_meteor_probability(temp_diff_image_path)
                    try: os.remove(temp_diff_image_path)
                    except OSError as e: print(f"警告: 一時差分ファイルの削除に失敗: {e}")
                    print(f"流星確率: {probability:.4f}")

                    is_meteor = probability >= config.METEOR_PROBABILITY_THRESHOLD
                    save_dir = meteor_save_path if is_meteor else not_meteor_save_path
                    detection_label = "meteor" if is_meteor else "not_meteor"
                    os.makedirs(save_dir, exist_ok=True)

                    detection_frame_for_time = (adjusted_start_frame + adjusted_end_frame) // 2
                    detection_time_seconds_rel = detection_frame_for_time / frame_rate
                    detection_datetime = base_datetime + timedelta(seconds=detection_time_seconds_rel)
                    display_timestamp = detection_datetime.strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]

                    base_filename = (
                        f"{detection_datetime.strftime('%Y%m%d_%H%M%S%f')[:-3]}"
                        f"_{detection_label}_{detection_counter}_prob{probability:.2f}"
                    )
                    saved_paths = {}
                    denoise_metrics: Optional[Dict[str, float]] = None

                    if save_options.get('ml_training_export_enabled', False):
                        try:
                            ml_event_dir = ml_training_data.export_training_event(
                                root_dir=save_options.get('ml_training_data_root', config.DEFAULT_ML_TRAINING_DATA_ROOT),
                                predicted_label=detection_label,
                                probability=probability,
                                event_id=base_filename,
                                source=source,
                                detection_time=detection_datetime,
                                frames=ml_source_frames,
                                classification_diff=classification_diff_img,
                                cutout_rect=cutout_rect,
                                cutout_size=cut_size,
                                frame_rate=frame_rate,
                                detected_line=((x1, y1), (x2, y2)),
                                frame_range=(adjusted_start_frame, adjusted_end_frame),
                            )
                            saved_paths['ml_training_event'] = ml_event_dir
                            print(f"機械学習向けイベントを保存しました: {ml_event_dir}")
                        except Exception as e_ml:
                            print(f"機械学習向けイベントの保存に失敗: {e_ml}")

                    if save_options.get('video', False):
                        output_clip_path = os.path.join(save_dir, f"{base_filename}.mp4")
                        fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_FOURCC)
                        out_clip = cv2.VideoWriter(output_clip_path, fourcc, frame_rate, (cut_size, cut_size))
                        if out_clip.isOpened():
                            for frame in final_frames_for_clip:
                                cut_frame = frame[y_start_cut:y_end_cut, x_start_cut:x_end_cut]
                                if cut_frame.shape[0] != cut_size or cut_frame.shape[1] != cut_size:
                                    cut_frame_resized = cv2.resize(cut_frame, (cut_size, cut_size))
                                else:
                                    cut_frame_resized = cut_frame
                                out_clip.write(cut_frame_resized)
                            out_clip.release()
                            saved_paths['video'] = output_clip_path
                            print(f"動画クリップを保存しました: {output_clip_path}")
                        else:
                            print(f"エラー: 動画クリップの書き込み開始に失敗 ({output_clip_path})")

                    if save_options.get('full_video', False) and final_frames_for_clip:
                        full_video_path = os.path.join(save_dir, f"{base_filename}_full.mp4")
                        clip_start_frame = actual_start_final if is_rtsp else adjusted_start_frame
                        clip_start_datetime = base_datetime + timedelta(
                            seconds=clip_start_frame / frame_rate
                        )
                        if write_full_size_video(
                            final_frames_for_clip, full_video_path, frame_rate,
                            clip_start_datetime, save_options,
                        ):
                            saved_paths['full_video'] = full_video_path
                            print(f"フルサイズ動画を保存しました: {full_video_path}")
                        else:
                            print(f"エラー: フルサイズ動画の書き込み開始に失敗 ({full_video_path})")

                    if (
                        active_fixed_pattern_correction is None
                        and save_options.get('denoised_full_video', False)
                        and final_frames_for_clip
                    ):
                        denoised_full_video_path = os.path.join(save_dir, f"{base_filename}_full_denoised.mp4")
                        clip_start_frame = actual_start_final if is_rtsp else adjusted_start_frame
                        clip_start_datetime = base_datetime + timedelta(
                            seconds=clip_start_frame / frame_rate
                        )
                        denoise_metrics = write_denoised_full_size_video(
                            final_frames_for_clip,
                            denoised_full_video_path,
                            frame_rate,
                            clip_start_datetime,
                            save_options,
                            ((x1, y1), (x2, y2)),
                        )
                        if denoise_metrics is not None:
                            saved_paths['denoised_full_video'] = denoised_full_video_path
                            print(
                                "ノイズ低減フルサイズ動画を保存しました: "
                                f"{denoised_full_video_path} "
                                f"(推定ノイズ {denoise_metrics['before_sigma']:.2f} -> "
                                f"{denoise_metrics['after_sigma']:.2f}, "
                                f"{denoise_metrics['reduction_percent']:.1f}%低減)"
                            )
                        else:
                            print(f"エラー: ノイズ低減動画の書き込みに失敗 ({denoised_full_video_path})")

                    if save_options.get('cutout', False):
                        cutout_diff_path = os.path.join(save_dir, f"{base_filename}_cutout_diff.jpg")
                        if cv2.imwrite(cutout_diff_path, diff_cutout):
                            saved_paths['cutout'] = cutout_diff_path
                            print(f"切り出し差分画像を保存しました: {cutout_diff_path}")
                        else:
                            print(f"エラー: 切り出し差分画像の保存に失敗 ({cutout_diff_path})")

                    if save_options.get('full', False):
                        full_diff_path = os.path.join(save_dir, f"{base_filename}_full_diff.jpg")
                        if cv2.imwrite(full_diff_path, saved_diff_img):
                            saved_paths['full'] = full_diff_path
                            print(f"全体差分画像を保存しました: {full_diff_path}")
                        else:
                            print(f"エラー: 全体差分画像の保存に失敗 ({full_diff_path})")

                    if save_options.get('composite', False) and final_frames_for_clip:
                        try:
                            composite_image, _ = _brightness_composite_and_diff(final_frames_for_clip)
                            composite_path = os.path.join(save_dir, f"{base_filename}_composite.jpg")
                            if cv2.imwrite(composite_path, composite_image):
                                saved_paths['composite'] = composite_path
                                print(f"比較明合成画像を保存しました: {composite_path}")

                                wcs_data_for_annotation = global_wcs_info if effective_use_plate_solve else {}
                                annotated_path = astrometry.annotate_image_with_wcs(
                                    image_path=composite_path, wcs_info=wcs_data_for_annotation,
                                    line_centers=[(cx, cy)], detection_datetime=detection_datetime,
                                    timestamp=display_timestamp, cancel_flag=cancel_flag,
                                    detected_line=((x1, y1), (x2, y2)),
                                )
                                if annotated_path:
                                    saved_paths['annotated'] = annotated_path
                                    print(f"注釈付き画像を保存しました: {annotated_path}")
                            else:
                                 print(f"エラー: 比較明合成画像の保存に失敗 ({composite_path})")
                        except Exception as e_comp:
                             print(f"比較明合成または注釈処理中にエラー: {e_comp}")

                    if save_options.get('info', False):
                        info_path = os.path.join(save_dir, f"{base_filename}_info.txt")
                        try:
                            ra_start_str, dec_start_str, ra_end_str, dec_end_str = "N/A", "N/A", "N/A", "N/A"
                            if effective_use_plate_solve and global_wcs_info.get('wcs_file'):
                                try:
                                    with fits.open(global_wcs_info['wcs_file']) as hdul:
                                        wcs = WCS(hdul[0].header, relax=True, fix=False)
                                        sky_coord_start = wcs.pixel_to_world(x1, y1)
                                        sky_coord_end = wcs.pixel_to_world(x2, y2)
                                        ra_start_str, dec_start_str = f"{sky_coord_start.ra.deg:.6f}", f"{sky_coord_start.dec.deg:.6f}"
                                        ra_end_str, dec_end_str = f"{sky_coord_end.ra.deg:.6f}", f"{sky_coord_end.dec.deg:.6f}"
                                except Exception as e_wcs_read:
                                    print(f"情報ファイル作成中のWCS読み込み/変換エラー: {e_wcs_read}")

                            with open(info_path, 'w', encoding='utf-8') as f:
                                f.write(f"Source: {source}\n")
                                f.write(f"Detection Time (UTC): {detection_datetime.isoformat()}Z\n")
                                f.write(f"Frame Range (Clip): {adjusted_start_frame} - {adjusted_end_frame}\n")
                                f.write(f"Meteor Probability: {probability:.6f}\n")
                                f.write(f"Detected Line Center (px): ({cx:.2f}, {cy:.2f})\n")
                                if denoise_metrics is not None:
                                    f.write(f"Denoise Noise Sigma Before: {denoise_metrics['before_sigma']:.4f}\n")
                                    f.write(f"Denoise Noise Sigma After: {denoise_metrics['after_sigma']:.4f}\n")
                                    f.write(f"Denoise Noise Reduction (%): {denoise_metrics['reduction_percent']:.2f}\n")
                                if enhancement_result is not None:
                                    f.write(f"Adaptive Fixed Pattern Strength: {enhancement_result.correction_strength:.4f}\n")
                                    f.write("Temporal Enhancement: 21-frame mean\n")
                                    f.write(f"Enhancement Noise Sigma Before: {enhancement_result.noise_before:.4f}\n")
                                    f.write(f"Enhancement Noise Sigma After: {enhancement_result.noise_after:.4f}\n")
                                    f.write(f"Enhancement Noise Reduction (%): {enhancement_result.noise_reduction_percent:.2f}\n")
                                _write_noise_twin_info(
                                    f,
                                    save_options,
                                    math.hypot(x2 - x1, y2 - y1)
                                    / max(1.0, math.hypot(img_w, img_h)),
                                    float(diff_img.max()) / 16.0 if evidence_path else None,
                                )
                                f.write(f"RA Start (deg): {ra_start_str}\nDec Start (deg): {dec_start_str}\n")
                                f.write(f"RA End (deg): {ra_end_str}\nDec End (deg): {dec_end_str}\n")
                                for key, path in saved_paths.items(): f.write(f"Saved {key.capitalize()} Path: {path}\n")

                            saved_paths['info'] = info_path
                            print(f"検出情報ファイルを保存しました: {info_path}")
                        except Exception as e_info:
                            print(f"エラー: 検出情報ファイルの保存中にエラー: {e_info}")

                    if save_options.get('summary', False):
                        if all(key in saved_paths for key in ['composite', 'annotated', 'video']):
                            summary_video_path = os.path.join(save_dir, f"{base_filename}_summary.mp4")
                            try:
                                orig_h, orig_w = final_frames_for_clip[0].shape[:2]
                                summary_created = video_creation.create_summary_video(
                                    summary_video_config=summary_video_config,
                                    composite_image_path=saved_paths['composite'],
                                    annotated_image_path=saved_paths['annotated'],
                                    cutout_video_path=saved_paths['video'],
                                    output_video_path=summary_video_path,
                                    cutout_rect=cutout_rect,
                                    frame_rate=frame_rate,
                                    output_resolution=(orig_w, orig_h),
                                    full_video_path=(
                                        saved_paths.get('denoised_full_video')
                                        or saved_paths.get('full_video')
                                    ),
                                    detected_line=((x1, y1), (x2, y2)),
                                )
                                if summary_created:
                                    saved_paths['summary'] = summary_video_path
                                    print(f"概要動画を保存しました: {summary_video_path}")
                                else:
                                    print("警告: 概要動画を作成できなかったため、保存結果には追加しません。")
                            except Exception as e_summary:
                                print(f"エラー: 概要動画の生成に失敗しました: {e_summary}")
                                import traceback; traceback.print_exc()
                        else:
                            print("警告: 概要動画の作成に必要なファイルが不足しているため、スキップします。")

                    if progress_callback:
                        progress_callback((f"検出 {detection_counter}: {detection_label} (Prob: {probability:.2f}) @ {display_timestamp}", None))

                        # If probability is below the configured meteor threshold, emit a concise
                        # Not Meteor summary line instead of listing all saved files.
                        try:
                            threshold = float(getattr(__import__('config'), 'METEOR_PROBABILITY_THRESHOLD', 0.5))
                        except Exception:
                            threshold = 0.5

                        if (not is_meteor) and (probability < threshold):
                            progress_callback((f"  -> Not Meteor: Probability {probability:.2f}", None))
                        else:
                            for key, path in saved_paths.items():
                                if path:
                                    progress_callback((f"  -> {key.capitalize()}: {os.path.basename(path)}", None))

                    if is_meteor and notify_on_detection:
                        try: utils.play_notification_sound()
                        except Exception as e_sound: print(f"通知音の再生中にエラー: {e_sound}")

                    processed_clips_info.append({
                        'path': saved_paths.get('video'), 'center': (cx, cy), 'probability': probability,
                        'label': detection_label, 'timestamp': detection_datetime,
                        'start_frame': adjusted_start_frame, 'end_frame': adjusted_end_frame, 'saved_paths': saved_paths
                    })

            except Exception as e_frame:
                print(f"警告: フレーム処理中にエラーが発生しました ({e_frame})。次のフレームを継続します。")
                continue
    except Exception as e_main_loop:
        print(f"動画処理ループ中に予期せぬエラーが発生しました: {e_main_loop}")
        import traceback; traceback.print_exc()
    finally:
        if cap.isOpened():
            cap.release()
            print("動画キャプチャを解放しました。")
        if 'evidence_cap' in locals() and evidence_cap is not None:
            evidence_cap.release()
        if prepared_video is not None:
            prepared_video.cleanup()
        if prepared_mean is not None:
            prepared_mean.cleanup()

    if progress_callback:
        # Emit a single concise completion line including full path and detection count.
        final_message = f'完了: {source} 検出: {detection_counter}件'
        if cancel_flag is not None and cancel_flag.is_set():
            final_message += " (キャンセルされました)"
        progress_callback((final_message, None))

    return processed_clips_info


if __name__ == '__main__':
    print("video_processing.py が直接実行されました。")
