"""Render a compact visual report for a saved camera model."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from camera_plate_model import FixedCameraPlateModel
import local_wideangle_astrometry as local_astrometry


DEFAULT_OUTPUT_NAME = "camera_model_visualization.png"


def _font(size: int, bold: bool = False):
    candidates = (
        ("/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc", "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc")
        if bold
        else ("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc", "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc")
    )
    for candidate in candidates:
        if os.path.isfile(candidate):
            try:
                return ImageFont.truetype(candidate, size=size)
            except (OSError, ValueError):
                pass
    return ImageFont.load_default()


def _as_naive(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.replace(tzinfo=None)


def _choose_reference_video(payload: dict, reference: datetime) -> Optional[Path]:
    videos = [Path(str(item)).expanduser() for item in payload.get("source_videos", [])]
    videos = [path for path in videos if path.is_file()]
    if not videos:
        return None
    return min(
        videos,
        key=lambda path: abs(local_astrometry._capture_datetime(str(path)) - reference),
    )


def _read_reference_frame(
    video_path: Path,
    reference: datetime,
    width: int,
    height: int,
) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        start = local_astrometry._capture_datetime(str(video_path))
        offset = max(0.0, (reference - start).total_seconds())
        index = int(round(offset * max(fps, 1.0)))
        if count > 0:
            index = min(max(0, index), count - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        return frame
    finally:
        cap.release()


def _read_frame_path(frame_path: str, width: int, height: int) -> Optional[np.ndarray]:
    path = Path(frame_path).expanduser()
    if not path.is_file():
        return None
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        return None
    if frame.shape[1] != width or frame.shape[0] != height:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame


def _projected_grid(
    image: np.ndarray,
    model: FixedCameraPlateModel,
) -> np.ndarray:
    output = image.copy()
    height, width = output.shape[:2]

    def draw_curve(ra_values: np.ndarray, dec_values: np.ndarray, color: tuple[int, int, int]):
        x, y = model.world_to_pixel_values(ra_values, dec_values)
        points = np.column_stack((x, y)).astype(float)
        valid = np.isfinite(points).all(axis=1)
        valid &= (points[:, 0] >= 0) & (points[:, 0] < width)
        valid &= (points[:, 1] >= 0) & (points[:, 1] < height)
        for index in range(len(points) - 1):
            if not (valid[index] and valid[index + 1]):
                continue
            if np.linalg.norm(points[index + 1] - points[index]) > 80:
                continue
            p1 = tuple(np.rint(points[index]).astype(int))
            p2 = tuple(np.rint(points[index + 1]).astype(int))
            cv2.line(output, p1, p2, color, 2, cv2.LINE_AA)

    # OpenCV receives BGR colors. Blue: declination. Magenta: right ascension.
    ra_samples = np.linspace(0.0, 360.0, 721)
    for dec in np.arange(-80.0, 81.0, 10.0):
        draw_curve(ra_samples, np.full_like(ra_samples, dec), (255, 180, 64))
    dec_samples = np.linspace(-89.0, 89.0, 361)
    for ra in np.arange(0.0, 360.0, 15.0):
        draw_curve(np.full_like(dec_samples, ra), dec_samples, (235, 100, 255))
    return output


def _support_overlay(image: np.ndarray, support: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    rows, columns = support.shape
    overlay = image.copy()
    cell_width = width / max(1, columns)
    cell_height = height / max(1, rows)
    for row in range(rows):
        for column in range(columns):
            x0 = int(round(column * cell_width))
            y0 = int(round(row * cell_height))
            x1 = int(round((column + 1) * cell_width))
            y1 = int(round((row + 1) * cell_height))
            if int(support[row, column]) > 0:
                cv2.rectangle(overlay, (x0, y0), (x1, y1), (55, 220, 100), -1)
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (215, 215, 215), 1)
    return cv2.addWeighted(overlay, 0.25, image, 0.75, 0.0)


def _support_panel(support: np.ndarray, width: int, height: int) -> Image.Image:
    rows, columns = support.shape
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (18, 29, 45)
    cell_width = width / max(1, columns)
    cell_height = height / max(1, rows)
    for row in range(rows):
        for column in range(columns):
            x0 = int(round(column * cell_width))
            y0 = int(round(row * cell_height))
            x1 = int(round((column + 1) * cell_width))
            y1 = int(round((row + 1) * cell_height))
            color = (57, 210, 103) if int(support[row, column]) > 0 else (45, 58, 75)
            cv2.rectangle(panel, (x0, y0), (x1, y1), color, -1)
            cv2.rectangle(panel, (x0, y0), (x1, y1), (120, 140, 160), 1)
    return Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))


def render_camera_model_visualization(
    model_path: str,
    *,
    output_path: Optional[str] = None,
    frame_path: Optional[str] = None,
) -> str:
    """Render the saved model and its validated support region as a PNG."""
    model_file = Path(model_path).expanduser().resolve()
    payload = json.loads(model_file.read_text(encoding="utf-8"))
    model = FixedCameraPlateModel(payload)
    reference = _as_naive(payload.get("reference_datetime"))
    original_width = int(payload["width"])
    original_height = int(payload["height"])
    main_width, main_height = 1280, 720

    selected_video = None
    frame = _read_frame_path(frame_path, original_width, original_height) if frame_path else None
    if frame is None:
        selected_video = _choose_reference_video(payload, reference)
        if selected_video is not None:
            frame = _read_reference_frame(selected_video, reference, original_width, original_height)
    if frame is None:
        frame = np.zeros((original_height, original_width, 3), dtype=np.uint8)
        cv2.putText(frame, "Reference frame unavailable", (40, max(60, original_height // 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (210, 220, 235), 2, cv2.LINE_AA)

    frame = cv2.resize(frame, (main_width, main_height), interpolation=cv2.INTER_AREA)
    support = np.asarray(payload.get("support_grid") or [[1]], dtype=np.uint8)
    if support.ndim != 2 or support.size == 0:
        support = np.ones((1, 1), dtype=np.uint8)
    frame = _support_overlay(frame, support)
    frame = _projected_grid(frame, model)

    canvas = Image.new("RGB", (1800, 1060), (8, 15, 27))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(34, True)
    sub_font = _font(20)
    section_font = _font(24, True)
    body_font = _font(20)
    small_font = _font(17)
    canvas.paste(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), (40, 112))
    draw.rectangle((40, 112, 1320, 832), outline=(95, 125, 155), width=2)
    draw.text((40, 28), "作成済みカメラモデル：有効領域と写像", fill=(240, 245, 250), font=title_font)
    source_text = selected_video.name if selected_video is not None else "背景フレームなし"
    draw.text(
        (40, 75),
        f"基準時刻 {reference:%Y-%m-%d %H:%M:%S}  /  背景: {source_text} の単一フレーム",
        fill=(166, 185, 205), font=sub_font,
    )
    draw.rounded_rectangle((64, 750, 700, 824), radius=10, fill=(8, 15, 27), outline=(80, 110, 135), width=1)
    draw.rectangle((80, 778, 102, 800), fill=(55, 220, 100))
    draw.text((112, 774), "緑: 軌跡で支持された有効セル", fill=(230, 240, 245), font=small_font)
    draw.line((390, 789, 425, 789), fill=(64, 180, 255), width=3)
    draw.text((435, 774), "青: Dec グリッド", fill=(230, 240, 245), font=small_font)
    draw.line((80, 813, 115, 813), fill=(235, 100, 255), width=3)
    draw.text((125, 798), "紫: RA グリッド", fill=(230, 240, 245), font=small_font)

    panel_x, panel_y, panel_width, panel_height = 1370, 112, 390, 350
    draw.text((panel_x, 28), "有効領域（支持グリッド）", fill=(240, 245, 250), font=section_font)
    panel = _support_panel(support, panel_width, panel_height)
    canvas.paste(panel, (panel_x, panel_y))
    draw.rectangle((panel_x, panel_y, panel_x + panel_width, panel_y + panel_height), outline=(95, 125, 155), width=2)

    fit_stats = payload.get("fit_stats") or {}
    trajectory_validation = payload.get("trajectory_validation") or {}
    supported_cells = int(np.count_nonzero(support))
    total_cells = int(support.size)
    support_fraction = float(payload.get("support_fraction", supported_cells / max(1, total_cells)))
    track_count = trajectory_validation.get("track_count", fit_stats.get("trajectory_count", "-"))
    raw_track_count = trajectory_validation.get("raw_track_count", "-")
    residual = fit_stats.get("residual_p95_px", "-")
    holdout = fit_stats.get("holdout_residual_p95_px", "-")
    target_text = "達成" if payload.get("target_met") else "未達"
    stats = [
        f"有効セル: {supported_cells}/{total_cells} ({support_fraction * 100:.1f}%)",
        f"恒星軌跡: {track_count}/{raw_track_count}本",
        f"残差 p95: {float(residual):.2f}px" if isinstance(residual, (float, int)) else f"残差 p95: {residual}",
        f"ホールドアウト p95: {float(holdout):.2f}px" if isinstance(holdout, (float, int)) else f"ホールドアウト p95: {holdout}",
        f"モデル状態: {'有効' if payload.get('enabled') else '候補'}（目標被覆率80%: {target_text}）",
    ]
    y = 500
    for index, line in enumerate(stats):
        draw.text((panel_x, y), line, fill=(255, 205, 110) if index == 4 else (215, 230, 240), font=body_font)
        y += 39
    draw.text((panel_x, 725), "緑: 3本以上の軌跡で検証されたセル", fill=(165, 185, 205), font=small_font)
    draw.text((panel_x, 754), "青紫: モデルから投影した空の座標グリッド", fill=(165, 185, 205), font=small_font)
    draw.text((panel_x, 783), "背景: 基準時刻付近の単一フレーム", fill=(165, 185, 205), font=small_font)
    draw.text((40, 900), "注意: 有効領域は画面全体ではなく、今回の動画で恒星軌跡が得られた範囲を示します。", fill=(180, 195, 210), font=body_font)
    draw.text((40, 945), f"モデル: {model_file.parent.name}", fill=(125, 150, 170), font=small_font)

    destination = Path(output_path).expanduser().resolve() if output_path else model_file.with_name(DEFAULT_OUTPUT_NAME)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".tmp" + destination.suffix)
    canvas.save(temporary, quality=95)
    os.replace(temporary, destination)
    return str(destination)
