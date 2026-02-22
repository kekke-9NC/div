import os
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import config

ProgressCallback = Optional[Callable[[str], None]]


def _emit(progress_callback: ProgressCallback, message: str) -> None:
    if progress_callback:
        try:
            progress_callback(message)
        except Exception:
            pass


def _find_first_video_path(sources: Sequence[str]) -> Optional[Path]:
    first_video_path = None
    sorted_sources = sorted(sources)

    for source in sorted_sources:
        path = Path(source)
        if path.is_dir():
            found = sorted([p for p in path.rglob('*') if p.suffix.lower() in config.PERIODIC_VIDEO_EXTENSIONS])
            if found:
                first_video_path = found[0]
                break
        elif path.is_file() and path.suffix.lower() in config.PERIODIC_VIDEO_EXTENSIONS:
            first_video_path = path
            break
    return first_video_path


def _list_video_paths_from_sources(sources: Sequence[str]) -> List[Path]:
    videos: List[Path] = []
    for source in sorted(sources):
        path = Path(source)
        if path.is_dir():
            found = sorted([p for p in path.rglob('*') if p.is_file() and p.suffix.lower() in config.PERIODIC_VIDEO_EXTENSIONS])
            videos.extend(found)
        elif path.is_file() and path.suffix.lower() in config.PERIODIC_VIDEO_EXTENSIONS:
            videos.append(path)
    # Deduplicate while preserving sorted order (case-insensitive on Windows)
    seen = set()
    unique: List[Path] = []
    for p in videos:
        key = str(p.resolve()) if p.exists() else str(p)
        key = os.path.normcase(os.path.normpath(key))
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return unique


def _select_video_sequence_from_sources(
    sources: Sequence[str],
    start_video_path: Optional[str] = None
) -> List[Path]:
    all_videos = _list_video_paths_from_sources(sources)
    if not all_videos:
        return []
    if not start_video_path:
        return all_videos

    start_norm = os.path.normcase(os.path.normpath(os.path.abspath(start_video_path)))
    for i, p in enumerate(all_videos):
        p_norm = os.path.normcase(os.path.normpath(os.path.abspath(str(p))))
        if p_norm == start_norm:
            return all_videos[i:]

    # Fallback: if the selected file is valid but outside current sources, use it alone.
    p = Path(start_video_path)
    if p.is_file() and p.suffix.lower() in config.PERIODIC_VIDEO_EXTENSIONS:
        return [p]
    return all_videos


def _estimate_video_duration_ms(cap: cv2.VideoCapture) -> Optional[float]:
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps > 0 and frame_count > 0 and np.isfinite(fps) and np.isfinite(frame_count):
            return (frame_count / fps) * 1000.0
    except Exception:
        pass
    return None


def _read_frame_at_ms(
    cap: cv2.VideoCapture,
    target_ms: float,
    fallback_offsets_ms: Tuple[int, ...] = (0, 250, -250, 500, -500, 1000, -1000),
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    for offset_ms in fallback_offsets_ms:
        seek_ms = max(0.0, float(target_ms + offset_ms))
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, seek_ms)
        except Exception:
            pass
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            continue
        actual_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if not np.isfinite(actual_ms) or actual_ms < 0:
            actual_ms = seek_ms
        return frame, float(actual_ms)
    return None, None


def _preprocess_star_frame(frame_bgr: np.ndarray) -> np.ndarray:
    if frame_bgr.ndim == 2:
        gray = frame_bgr.astype(np.float32)
    else:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape[:2]

    # Remove vertical banding bias (common in low-light security cameras).
    col_profile = np.median(gray, axis=0).astype(np.float32)
    k = max(9, (w // 64) | 1)
    col_profile_smooth = cv2.GaussianBlur(col_profile.reshape(1, -1), (k, 1), 0).reshape(-1)
    gray = gray - (col_profile_smooth[None, :] - float(np.median(col_profile_smooth)))

    # Keep point-like stars and suppress smooth background/glow.
    sigma = max(2.5, min(h, w) / 120.0)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    highpass = gray - background

    med = float(np.median(highpass))
    mad = float(np.median(np.abs(highpass - med))) + 1e-3
    z = (highpass - med) / (1.4826 * mad)
    z = np.clip(z, -2.0, 12.0)
    return ((z + 2.0) * (255.0 / 14.0)).astype(np.uint8)


def _connected_to_border_mask(binary_mask: np.ndarray, min_area: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=8)
    h, w = binary_mask.shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    for label_id in range(1, num_labels):
        x, y, bw, bh, area = stats[label_id]
        if area < min_area:
            continue
        touches_border = (x <= 0) or (y <= 0) or (x + bw >= w) or (y + bh >= h)
        if touches_border:
            out[labels == label_id] = 255
    return out


def build_auto_night_selfcal_mask(
    probe_gray_frames: Sequence[np.ndarray],
    progress_callback: ProgressCallback = None
) -> Optional[np.ndarray]:
    if not probe_gray_frames:
        return None

    frames = [f for f in probe_gray_frames if f is not None]
    if not frames:
        return None

    stack = np.stack(frames[: min(len(frames), 96)], axis=0).astype(np.uint8)
    h, w = stack.shape[1:]
    mask = np.full((h, w), 255, dtype=np.uint8)

    _emit(progress_callback, "Self-cal mask: building automatic mask from sampled night frames...")

    # 1) Timestamp / OSD in top-left.
    top_h = max(32, int(h * 0.16))
    left_w = max(160, int(w * 0.42))
    tl_max = np.max(stack[:, :top_h, :left_w], axis=0)
    tl_thr = max(170, int(np.percentile(tl_max, 99.0) * 0.75))
    timestamp_mask = (tl_max >= tl_thr).astype(np.uint8) * 255
    timestamp_mask = cv2.morphologyEx(timestamp_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    timestamp_mask = cv2.dilate(timestamp_mask, np.ones((5, 5), np.uint8), iterations=2)
    full_timestamp_mask = np.zeros((h, w), dtype=np.uint8)
    full_timestamp_mask[:top_h, :left_w] = timestamp_mask
    mask[full_timestamp_mask > 0] = 0

    # 2) Edge-connected glow / bright static regions.
    median_img = np.median(stack, axis=0).astype(np.float32)
    blur_sigma = max(8.0, min(h, w) / 80.0)
    median_blur = cv2.GaussianBlur(median_img, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    glow_thr = max(float(np.median(median_blur)) + 12.0, float(np.percentile(median_blur, 99.6)))
    glow_bin = (median_blur >= glow_thr).astype(np.uint8)
    edge_glow = _connected_to_border_mask(glow_bin, min_area=max(64, int(h * w * 0.002)))
    if np.any(edge_glow):
        edge_glow = cv2.dilate(edge_glow, np.ones((9, 9), np.uint8), iterations=1)
        mask[edge_glow > 0] = 0

    # 3) Saturated hot areas.
    max_img = np.max(stack, axis=0)
    hot = (max_img >= 252).astype(np.uint8) * 255
    if np.any(hot):
        hot = cv2.dilate(hot, np.ones((3, 3), np.uint8), iterations=1)
        mask[hot > 0] = 0

    border = 2
    mask[:border, :] = 0
    mask[-border:, :] = 0
    mask[:, :border] = 0
    mask[:, -border:] = 0

    masked_ratio = 1.0 - (float(np.count_nonzero(mask)) / float(mask.size))
    _emit(progress_callback, f"Self-cal mask: masked {masked_ratio * 100:.1f}% of pixels.")
    return mask


def _detect_star_points(
    preprocessed_gray: np.ndarray,
    mask: Optional[np.ndarray],
    max_points: int = 500,
    min_distance: int = 8,
) -> np.ndarray:
    pts = cv2.goodFeaturesToTrack(
        preprocessed_gray,
        maxCorners=int(max_points),
        qualityLevel=0.01,
        minDistance=float(min_distance),
        blockSize=5,
        useHarrisDetector=False,
        mask=mask,
    )
    if pts is None or len(pts) == 0:
        return np.empty((0, 2), dtype=np.float32)

    pts = pts.reshape(-1, 2).astype(np.float32)
    try:
        pts_refine = pts.reshape(-1, 1, 2).copy()
        cv2.cornerSubPix(
            preprocessed_gray,
            pts_refine,
            winSize=(3, 3),
            zeroZone=(-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.02),
        )
        pts = pts_refine.reshape(-1, 2)
    except Exception:
        pass
    return pts.astype(np.float32)


def _draw_exclusion_circles(base_mask: np.ndarray, points: Sequence[np.ndarray], radius: int = 10) -> np.ndarray:
    work = base_mask.copy()
    for pt in points:
        x, y = int(round(float(pt[0]))), int(round(float(pt[1])))
        if 0 <= x < work.shape[1] and 0 <= y < work.shape[0]:
            cv2.circle(work, (x, y), radius, 0, thickness=-1)
    return work


def _perspective_transform_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if pts.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    src = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(src, H.astype(np.float64))
    return dst.reshape(-1, 2).astype(np.float32)


def _seed_tracks(
    preprocessed_gray: np.ndarray,
    base_mask: np.ndarray,
    current_cumulative_to_segref: np.ndarray,
    next_track_id: int,
    existing_tracks: Optional[List[Dict]] = None,
    max_new_tracks: int = 200,
) -> Tuple[List[Dict], int]:
    if max_new_tracks <= 0:
        return [], next_track_id

    existing_tracks = existing_tracks or []
    seed_mask = _draw_exclusion_circles(base_mask, [t["pt"] for t in existing_tracks], radius=10)
    points = _detect_star_points(preprocessed_gray, seed_mask, max_points=max_new_tracks, min_distance=8)
    if points.size == 0:
        return [], next_track_id

    tracks: List[Dict] = []
    current_cumulative_to_segref = np.asarray(current_cumulative_to_segref, dtype=np.float64)
    inv_anchor = np.linalg.inv(current_cumulative_to_segref)
    for pt in points:
        tracks.append({
            "id": int(next_track_id),
            "pt": pt.astype(np.float32),
            "anchor_pt": pt.astype(np.float32).copy(),
            "anchor_cum_to_segref": current_cumulative_to_segref.copy(),
            "anchor_cum_inv": inv_anchor.copy(),
        })
        next_track_id += 1
    return tracks, next_track_id


def _weighted_affine_remove(
    x: np.ndarray,
    y: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    w: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    A = np.stack([np.ones_like(x), x, y], axis=1)
    ws = np.sqrt(np.clip(w, 1e-6, None))
    Aw = A * ws[:, None]
    dxw = dx * ws
    dyw = dy * ws
    try:
        coef_x, _, _, _ = np.linalg.lstsq(Aw, dxw, rcond=None)
        coef_y, _, _, _ = np.linalg.lstsq(Aw, dyw, rcond=None)
        dx = dx - A @ coef_x
        dy = dy - A @ coef_y
    except Exception:
        pass
    return dx, dy


def _fit_smooth_residual_field_to_maps(
    image_shape: Tuple[int, int],
    sample_x: Sequence[float],
    sample_y: Sequence[float],
    sample_dx: Sequence[float],
    sample_dy: Sequence[float],
    sample_w: Sequence[float],
    strength: float = 0.5,
    grid_step: int = 48,
    progress_callback: ProgressCallback = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    h, w = image_shape[:2]
    x = np.asarray(sample_x, dtype=np.float32)
    y = np.asarray(sample_y, dtype=np.float32)
    dx = np.asarray(sample_dx, dtype=np.float32)
    dy = np.asarray(sample_dy, dtype=np.float32)
    wt = np.asarray(sample_w, dtype=np.float32)

    if x.size == 0:
        raise ValueError("No residual samples available.")

    mag = np.hypot(dx, dy)
    if mag.size >= 10:
        cutoff = float(np.percentile(mag, 98.0))
        keep = mag <= max(0.75, cutoff)
        x, y, dx, dy, wt = x[keep], y[keep], dx[keep], dy[keep], wt[keep]

    dx, dy = _weighted_affine_remove(x, y, dx, dy, np.clip(wt, 1e-3, None))

    # Residuals represent inconsistency. Apply the opposite sign conservatively.
    dx = -dx
    dy = -dy

    gh = int(np.ceil(h / float(grid_step)))
    gw = int(np.ceil(w / float(grid_step)))
    sum_dx = np.zeros((gh, gw), dtype=np.float32)
    sum_dy = np.zeros((gh, gw), dtype=np.float32)
    sum_w = np.zeros((gh, gw), dtype=np.float32)

    _emit(progress_callback, f"Self-cal fit: accumulating {x.size} residual samples on a {gw}x{gh} grid...")

    for xi, yi, dxi, dyi, wi in zip(x, y, dx, dy, wt):
        fx = float(xi) / float(grid_step)
        fy = float(yi) / float(grid_step)
        ix = int(np.floor(fx))
        iy = int(np.floor(fy))
        tx = fx - ix
        ty = fy - iy

        ix0 = max(0, min(gw - 1, ix))
        iy0 = max(0, min(gh - 1, iy))
        ix1 = max(0, min(gw - 1, ix + 1))
        iy1 = max(0, min(gh - 1, iy + 1))

        w00 = (1.0 - tx) * (1.0 - ty)
        w10 = tx * (1.0 - ty)
        w01 = (1.0 - tx) * ty
        w11 = tx * ty

        for gx, gy, wb in ((ix0, iy0, w00), (ix1, iy0, w10), (ix0, iy1, w01), (ix1, iy1, w11)):
            if wb <= 0:
                continue
            wv = float(wi) * float(wb)
            sum_dx[gy, gx] += float(dxi) * wv
            sum_dy[gy, gx] += float(dyi) * wv
            sum_w[gy, gx] += wv

    sigma_cells = 1.4
    blur_dx = cv2.GaussianBlur(sum_dx, (0, 0), sigmaX=sigma_cells, sigmaY=sigma_cells)
    blur_dy = cv2.GaussianBlur(sum_dy, (0, 0), sigmaX=sigma_cells, sigmaY=sigma_cells)
    blur_w = cv2.GaussianBlur(sum_w, (0, 0), sigmaX=sigma_cells, sigmaY=sigma_cells)
    eps = 1e-5
    grid_dx = blur_dx / (blur_w + eps)
    grid_dy = blur_dy / (blur_w + eps)

    cy = min(gh - 1, max(0, gh // 2))
    cx = min(gw - 1, max(0, gw // 2))
    grid_dx -= grid_dx[cy, cx]
    grid_dy -= grid_dy[cy, cx]

    dx_full = cv2.resize(grid_dx, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    dy_full = cv2.resize(grid_dy, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)

    max_mag = np.hypot(dx_full, dy_full)
    clip_limit = 20.0
    over = max_mag > clip_limit
    if np.any(over):
        scale = (clip_limit / (max_mag[over] + 1e-6)).astype(np.float32)
        dx_full[over] *= scale
        dy_full[over] *= scale

    yy, xx = np.indices((h, w), dtype=np.float32)
    map_x = (xx + dx_full * float(strength)).astype(np.float32)
    map_y = (yy + dy_full * float(strength)).astype(np.float32)

    field_mag = np.hypot(dx_full, dy_full)
    stats = {
        "residual_samples_after_filter": int(x.size),
        "grid_width": int(gw),
        "grid_height": int(gh),
        "grid_step": int(grid_step),
        "strength": float(strength),
        "median_residual_mag_px": float(np.median(np.hypot(dx, dy))) if x.size else 0.0,
        "p95_residual_mag_px": float(np.percentile(np.hypot(dx, dy), 95.0)) if x.size else 0.0,
        "field_p95_mag_px_before_strength": float(np.percentile(field_mag, 95.0)) if field_mag.size else 0.0,
    }
    return map_x, map_y, stats


def _iter_sampled_frames_from_videos(
    video_paths: Sequence[str],
    duration_sec: float,
    sample_interval_sec: float,
):
    interval_ms = max(1.0, float(sample_interval_sec) * 1000.0)
    remaining_ms = max(0.0, float(duration_sec) * 1000.0)
    cumulative_ms = 0.0

    for video_idx, video_path in enumerate(video_paths):
        if remaining_ms <= 0:
            break

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            yield {
                "ok": False,
                "frame": None,
                "video_path": str(video_path),
                "video_index": int(video_idx),
                "global_actual_ms": cumulative_ms,
                "global_nominal_ms": cumulative_ms,
                "local_nominal_ms": 0.0,
                "error": "open_failed",
            }
            continue

        try:
            duration_ms_est = _estimate_video_duration_ms(cap)
            if duration_ms_est is None or duration_ms_est <= 0:
                # If duration metadata is missing, still try a conservative window.
                duration_ms_est = remaining_ms

            file_budget_ms = min(float(duration_ms_est), remaining_ms)
            local_t = 0.0
            while local_t < file_budget_ms and remaining_ms > 0:
                frame, actual_local_ms = _read_frame_at_ms(cap, float(local_t))
                global_nominal = cumulative_ms + local_t
                global_actual = cumulative_ms + (actual_local_ms if actual_local_ms is not None else local_t)
                if frame is None:
                    yield {
                        "ok": False,
                        "frame": None,
                        "video_path": str(video_path),
                        "video_index": int(video_idx),
                        "global_actual_ms": float(global_actual),
                        "global_nominal_ms": float(global_nominal),
                        "local_nominal_ms": float(local_t),
                        "error": "decode_failed",
                    }
                else:
                    yield {
                        "ok": True,
                        "frame": frame,
                        "video_path": str(video_path),
                        "video_index": int(video_idx),
                        "global_actual_ms": float(global_actual),
                        "global_nominal_ms": float(global_nominal),
                        "local_nominal_ms": float(local_t),
                        "error": None,
                    }

                local_t += interval_ms
                remaining_ms -= interval_ms
                if remaining_ms <= 0:
                    break
        finally:
            cap.release()

        cumulative_ms += float(duration_ms_est)


def _iter_median_composite_frames_from_videos(
    video_paths: Sequence[str],
    duration_sec: float,
    inner_sample_interval_sec: float,
    composite_window_sec: float = 20.0,
    composite_cycle_sec: float = 60.0,
):
    """
    Yield one median-composited frame per cycle.
    Uses all decodable frames within the first `composite_window_sec` of each cycle and skips the rest.
    """
    cycle_ms = max(1.0, float(composite_cycle_sec) * 1000.0)
    window_ms = max(1.0, min(float(composite_window_sec), float(composite_cycle_sec)) * 1000.0)
    # `inner_sample_interval_sec` is kept for compatibility/metadata, but this generator
    # now reads all frames in each composite window.
    _ = inner_sample_interval_sec
    total_duration_ms = max(0.0, float(duration_sec) * 1000.0)
    total_cycles = int(np.ceil(total_duration_ms / cycle_ms))

    if total_cycles <= 0:
        return

    def _composite_window_from_cap(
        cap: cv2.VideoCapture,
        local_start_ms: float,
        local_end_ms: float
    ) -> Tuple[Optional[np.ndarray], int, int]:
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(local_start_ms))
        except Exception:
            pass

        fps_meta = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps_meta) or fps_meta <= 0:
            fps_meta = 0.0

        gray_stack = None
        stack_capacity = 0
        stack_count = 0
        fail_count = 0
        shape_hw = None

        # Conservative initial capacity; expands if needed.
        if fps_meta > 0:
            stack_capacity = max(16, int(np.ceil((local_end_ms - local_start_ms) / 1000.0 * fps_meta)) + 8)
        else:
            stack_capacity = 64

        while True:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                fail_count += 1
                break

            pos_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
            if not np.isfinite(pos_ms) or pos_ms < 0:
                if fps_meta > 0:
                    pos_frames = float(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    pos_ms = max(0.0, (pos_frames - 1.0) * 1000.0 / fps_meta)
                else:
                    pos_ms = local_start_ms + (stack_count * 1000.0 / 25.0)

            # Exclude frames beyond the 20s window.
            if pos_ms > (local_end_ms + 1e-3):
                break

            if frame.ndim == 2:
                gray = frame
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if shape_hw is None:
                shape_hw = gray.shape[:2]
                gray_stack = np.empty((stack_capacity, shape_hw[0], shape_hw[1]), dtype=np.uint8)
            elif gray.shape[:2] != shape_hw:
                gray = cv2.resize(gray, (shape_hw[1], shape_hw[0]))

            if stack_count >= stack_capacity:
                new_capacity = int(max(stack_capacity * 1.5, stack_capacity + 32))
                new_stack = np.empty((new_capacity, shape_hw[0], shape_hw[1]), dtype=np.uint8)
                new_stack[:stack_count] = gray_stack[:stack_count]
                gray_stack = new_stack
                stack_capacity = new_capacity

            gray_stack[stack_count] = gray
            stack_count += 1

        if stack_count <= 0 or gray_stack is None:
            return None, 0, fail_count

        composite_gray = np.median(gray_stack[:stack_count], axis=0).astype(np.uint8)
        return composite_gray, stack_count, fail_count

    cumulative_ms = 0.0
    cycle_idx_global = 0

    for video_idx, video_path in enumerate(video_paths):
        if cumulative_ms >= total_duration_ms or cycle_idx_global >= total_cycles:
            break

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            # Emit a failed cycle placeholder if this file would have been used.
            center_ms = min(total_duration_ms, cumulative_ms + (window_ms * 0.5))
            yield {
                "ok": False,
                "frame": None,
                "video_path": str(video_path),
                "video_index": int(video_idx),
                "global_actual_ms": float(center_ms),
                "global_nominal_ms": float(center_ms),
                "local_nominal_ms": 0.0,
                "error": "open_failed",
                "cycle_index": int(cycle_idx_global),
                "window_ok_samples": 0,
                "window_failed_samples": 1,
                "window_total_samples": 1,
                "strategy": "median_window_composite_all_frames",
            }
            cycle_idx_global += 1
            cumulative_ms += cycle_ms
            continue

        try:
            duration_ms_est = _estimate_video_duration_ms(cap)
            if duration_ms_est is None or duration_ms_est <= 0:
                # Assume one cycle if metadata is unavailable.
                duration_ms_est = min(cycle_ms, total_duration_ms - cumulative_ms)

            file_budget_ms = min(duration_ms_est, total_duration_ms - cumulative_ms)
            local_cycle_start_ms = 0.0

            while local_cycle_start_ms < file_budget_ms and cycle_idx_global < total_cycles:
                local_window_end_ms = min(local_cycle_start_ms + window_ms, file_budget_ms)
                global_cycle_start_ms = cumulative_ms + local_cycle_start_ms
                center_ms = min(total_duration_ms, global_cycle_start_ms + (window_ms * 0.5))

                composite_gray, ok_count, fail_count = _composite_window_from_cap(
                    cap=cap,
                    local_start_ms=local_cycle_start_ms,
                    local_end_ms=local_window_end_ms
                )

                if composite_gray is not None:
                    yield {
                        "ok": True,
                        "frame": composite_gray,  # grayscale composite; downstream handles grayscale
                        "video_path": str(video_path),
                        "video_index": int(video_idx),
                        "global_actual_ms": float(center_ms),
                        "global_nominal_ms": float(center_ms),
                        "local_nominal_ms": float(local_cycle_start_ms),
                        "error": None,
                        "cycle_index": int(cycle_idx_global),
                        "window_ok_samples": int(ok_count),
                        "window_failed_samples": int(fail_count),
                        "window_total_samples": int(ok_count + fail_count),
                        "strategy": "median_window_composite_all_frames",
                    }
                else:
                    yield {
                        "ok": False,
                        "frame": None,
                        "video_path": str(video_path),
                        "video_index": int(video_idx),
                        "global_actual_ms": float(center_ms),
                        "global_nominal_ms": float(center_ms),
                        "local_nominal_ms": float(local_cycle_start_ms),
                        "error": "no_frames_in_window",
                        "cycle_index": int(cycle_idx_global),
                        "window_ok_samples": int(ok_count),
                        "window_failed_samples": int(fail_count),
                        "window_total_samples": int(ok_count + fail_count),
                        "strategy": "median_window_composite_all_frames",
                    }

                cycle_idx_global += 1
                local_cycle_start_ms += cycle_ms

        finally:
            cap.release()

        cumulative_ms += float(duration_ms_est)

    # If duration exceeds available videos, emit placeholders for remaining cycles.
    while cycle_idx_global < total_cycles:
        center_ms = min(total_duration_ms, (cycle_idx_global * cycle_ms) + (window_ms * 0.5))
        yield {
            "ok": False,
            "frame": None,
            "video_path": "",
            "video_index": -1,
            "global_actual_ms": float(center_ms),
            "global_nominal_ms": float(center_ms),
            "local_nominal_ms": 0.0,
            "error": "insufficient_video_coverage",
            "cycle_index": int(cycle_idx_global),
            "window_ok_samples": 0,
            "window_failed_samples": 0,
            "window_total_samples": 0,
            "strategy": "median_window_composite_all_frames",
        }
        cycle_idx_global += 1


def _collect_probe_frames_for_mask(
    video_path: str,
    duration_sec: float,
    probe_interval_sec: float,
    progress_callback: ProgressCallback = None
) -> List[np.ndarray]:
    probe_frames: List[np.ndarray] = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    try:
        duration_ms_est = _estimate_video_duration_ms(cap)
        end_ms = min(duration_sec * 1000.0, duration_ms_est) if duration_ms_est else (duration_sec * 1000.0)
        sample_times = np.arange(0.0, max(1.0, end_ms), max(1.0, probe_interval_sec * 1000.0), dtype=np.float64)
        for idx, t_ms in enumerate(sample_times):
            frame, _ = _read_frame_at_ms(cap, float(t_ms))
            if frame is None:
                continue
            probe_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if progress_callback and idx % 8 == 0:
                _emit(progress_callback, f"Self-cal mask probe: collected {len(probe_frames)} frames...")
    finally:
        cap.release()
    return probe_frames


def _collect_probe_frames_for_mask_from_videos(
    video_paths: Sequence[str],
    duration_sec: float,
    probe_interval_sec: float,
    progress_callback: ProgressCallback = None
) -> List[np.ndarray]:
    probe_frames: List[np.ndarray] = []
    for idx, sample in enumerate(_iter_sampled_frames_from_videos(video_paths, duration_sec, probe_interval_sec)):
        if not sample.get("ok") or sample.get("frame") is None:
            continue
        frame = sample["frame"]
        probe_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if progress_callback and idx % 8 == 0:
            _emit(progress_callback, f"Self-cal mask probe: collected {len(probe_frames)} frames...")
    return probe_frames


def estimate_distortion_map_from_night_video(
    video_path: str,
    map_x_path: str,
    map_y_path: str,
    duration_minutes: float = 20.0,
    sample_interval_sec: float = 2.0,
    progress_callback: ProgressCallback = None,
    auto_mask_output_path: Optional[str] = None,
    metadata_output_path: Optional[str] = None,
    strength: float = 0.5,
    manual_mask: Optional[np.ndarray] = None,
    use_auto_mask: bool = True,
) -> Dict[str, object]:
    """
    Estimate a distortion correction map from a fixed night-sky video using stars.

    This is an empirical self-calibration method based on star tracking and
    homography-consistency residuals (no SIP/WCS distortion terms required).
    It is designed to work even when some sampled timestamps cannot be decoded.
    """
    if duration_minutes <= 0:
        raise ValueError("duration_minutes must be > 0")
    if sample_interval_sec <= 0:
        raise ValueError("sample_interval_sec must be > 0")

    if isinstance(video_path, (list, tuple)):
        video_paths = [str(p) for p in video_path if p]
    else:
        video_paths = [str(video_path)]
    if not video_paths:
        raise ValueError("video_path is empty")

    duration_sec = float(duration_minutes) * 60.0
    _emit(progress_callback, f"Night self-calibration: loading probe frames from first {duration_minutes:.1f} min...")

    probe_interval_sec = max(10.0, sample_interval_sec * 6.0)
    probe_frames = _collect_probe_frames_for_mask_from_videos(
        video_paths=video_paths,
        duration_sec=duration_sec,
        probe_interval_sec=probe_interval_sec,
        progress_callback=progress_callback,
    )
    if not probe_frames:
        raise RuntimeError("Failed to collect probe frames for mask generation.")

    h, w = probe_frames[0].shape[:2]

    auto_mask = None
    if use_auto_mask:
        auto_mask = build_auto_night_selfcal_mask(probe_frames, progress_callback=progress_callback)
        if auto_mask is None:
            raise RuntimeError("Automatic mask generation failed.")
        if auto_mask_output_path:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(auto_mask_output_path)), exist_ok=True)
                cv2.imwrite(auto_mask_output_path, auto_mask)
                _emit(progress_callback, f"Night self-calibration: auto mask saved to {auto_mask_output_path}")
            except Exception as e:
                _emit(progress_callback, f"Night self-calibration: failed to save auto mask ({e})")
    else:
        _emit(progress_callback, "Night self-calibration: auto mask disabled (manual mask only mode).")

    manual_mask_resized = None
    if manual_mask is not None:
        try:
            mm = np.asarray(manual_mask, dtype=np.uint8)
            if mm.ndim == 3:
                mm = cv2.cvtColor(mm, cv2.COLOR_BGR2GRAY)
            if mm.shape[:2] != (h, w):
                mm = cv2.resize(mm, (w, h), interpolation=cv2.INTER_NEAREST)
            manual_mask_resized = np.where(mm > 0, 255, 0).astype(np.uint8)
            _emit(progress_callback, "Night self-calibration: manual mask loaded.")
        except Exception as e:
            raise RuntimeError(f"Failed to prepare manual mask: {e}")

    if auto_mask is not None and manual_mask_resized is not None:
        work_mask = cv2.bitwise_and(auto_mask, manual_mask_resized)
        _emit(progress_callback, "Night self-calibration: using combined mask (auto AND manual).")
    elif auto_mask is not None:
        work_mask = auto_mask
        _emit(progress_callback, "Night self-calibration: using auto mask only.")
    elif manual_mask_resized is not None:
        work_mask = manual_mask_resized
        _emit(progress_callback, "Night self-calibration: using manual mask only.")
    else:
        work_mask = np.full((h, w), 255, dtype=np.uint8)
        _emit(progress_callback, "Night self-calibration: no mask provided; using full frame.")

    sample_x: List[float] = []
    sample_y: List[float] = []
    sample_dx: List[float] = []
    sample_dy: List[float] = []
    sample_w: List[float] = []

    stats: Dict[str, object] = {
        "video_path": str(video_paths[0]),
        "video_path_start": str(video_paths[0]),
        "video_paths_selected_count": int(len(video_paths)),
        "duration_minutes_requested": float(duration_minutes),
        "sample_interval_sec": float(sample_interval_sec),
        "sampling_strategy": "median_composite_20s_every_60s",
        "composite_window_sec": 20.0,
        "composite_cycle_sec": 60.0,
        "composite_inner_sample_interval_sec": float(sample_interval_sec),
        "frames_sampled_success": 0,
        "frames_sampled_failed": 0,
        "samples_planned": int(np.ceil(duration_sec / 60.0)),
        "composite_windows_success": 0,
        "composite_windows_failed": 0,
        "composite_source_samples_success": 0,
        "composite_source_samples_failed": 0,
        "segments_started": 0,
        "segments_completed": 0,
        "homography_failures": 0,
        "gap_resets": 0,
        "track_observations_used": 0,
        "videos_touched_count": 0,
        "videos_touched_preview": [],
        "mask_shape": [int(h), int(w)],
        "use_auto_mask": bool(use_auto_mask),
        "manual_mask_used": bool(manual_mask_resized is not None),
        "auto_mask_nonzero_ratio": float(np.count_nonzero(auto_mask) / auto_mask.size) if auto_mask is not None else None,
        "manual_mask_nonzero_ratio": float(np.count_nonzero(manual_mask_resized) / manual_mask_resized.size) if manual_mask_resized is not None else None,
        "work_mask_nonzero_ratio": float(np.count_nonzero(work_mask) / work_mask.size),
        "algorithm": "star-homography-consistency-selfcal",
    }

    try:
        composite_cycle_sec = 60.0
        composite_window_sec = 20.0
        planned_samples = int(np.ceil(duration_sec / composite_cycle_sec))

        prev_pre = None
        prev_ms = None
        active_tracks: List[Dict] = []
        next_track_id = 0
        C_prev = np.eye(3, dtype=np.float64)  # previous frame -> segment reference frame
        segment_active = False
        frames_in_segment = 0

        effective_tracking_interval_sec = composite_cycle_sec
        max_track_gap_ms = max(6000.0, effective_tracking_interval_sec * 3000.0)
        target_active_tracks = 220
        min_tracks_for_h = 12
        min_seed_tracks = 30
        reseed_every_frames = 5

        _emit(
            progress_callback,
            f"Night self-calibration: sampling up to {planned_samples} median-composited frames "
            f"(20s use + 40s skip) over {duration_sec:.1f}s across {len(video_paths)} video(s)..."
        )

        touched_videos = []
        touched_set = set()
        for idx, sample in enumerate(
            _iter_median_composite_frames_from_videos(
                video_paths=video_paths,
                duration_sec=duration_sec,
                inner_sample_interval_sec=sample_interval_sec,
                composite_window_sec=composite_window_sec,
                composite_cycle_sec=composite_cycle_sec,
            )
        ):
            if progress_callback and idx % 25 == 0:
                _emit(progress_callback, f"Night self-calibration: sampled {idx}/{planned_samples} composite frames...")

            sample_video_path = str(sample.get("video_path", ""))
            sample_video_idx = int(sample.get("video_index", -1))
            sample_vid_key = (sample_video_idx, sample_video_path)
            if sample_vid_key not in touched_set:
                touched_set.add(sample_vid_key)
                touched_videos.append(sample_video_path)
                stats["videos_touched_count"] = int(len(touched_videos))
                stats["videos_touched_preview"] = touched_videos[:12]
                _emit(progress_callback, f"Night self-calibration: using video segment {sample_video_idx + 1}: {sample_video_path}")

            stats["composite_source_samples_success"] = int(stats["composite_source_samples_success"]) + int(sample.get("window_ok_samples", 0))
            stats["composite_source_samples_failed"] = int(stats["composite_source_samples_failed"]) + int(sample.get("window_failed_samples", 0))

            if not sample.get("ok") or sample.get("frame") is None:
                stats["frames_sampled_failed"] = int(stats["frames_sampled_failed"]) + 1
                stats["composite_windows_failed"] = int(stats["composite_windows_failed"]) + 1
                continue
            frame = sample["frame"]
            actual_ms = float(sample.get("global_actual_ms", 0.0))
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            stats["frames_sampled_success"] = int(stats["frames_sampled_success"]) + 1
            stats["composite_windows_success"] = int(stats["composite_windows_success"]) + 1

            pre = _preprocess_star_frame(frame)

            if prev_ms is not None and actual_ms is not None and (float(actual_ms) - float(prev_ms)) > max_track_gap_ms:
                if segment_active:
                    stats["segments_completed"] = int(stats["segments_completed"]) + 1
                stats["gap_resets"] = int(stats["gap_resets"]) + 1
                segment_active = False
                active_tracks = []
                prev_pre = None
                prev_ms = None
                C_prev = np.eye(3, dtype=np.float64)
                frames_in_segment = 0

            if not segment_active:
                seed_tracks, next_track_id = _seed_tracks(
                    preprocessed_gray=pre,
                    base_mask=work_mask,
                    current_cumulative_to_segref=np.eye(3, dtype=np.float64),
                    next_track_id=next_track_id,
                    existing_tracks=[],
                    max_new_tracks=target_active_tracks,
                )
                if len(seed_tracks) < min_seed_tracks:
                    continue
                active_tracks = seed_tracks
                segment_active = True
                stats["segments_started"] = int(stats["segments_started"]) + 1
                prev_pre = pre
                prev_ms = actual_ms
                C_prev = np.eye(3, dtype=np.float64)
                frames_in_segment = 1
                continue

            if prev_pre is None or not active_tracks:
                segment_active = False
                continue

            prev_pts = np.array([t["pt"] for t in active_tracks], dtype=np.float32).reshape(-1, 1, 2)
            cur_pts, st, err = cv2.calcOpticalFlowPyrLK(
                prev_pre, pre, prev_pts, None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
                flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
            )
            if cur_pts is None or st is None:
                stats["homography_failures"] = int(stats["homography_failures"]) + 1
                segment_active = False
                active_tracks = []
                prev_pre = None
                prev_ms = None
                C_prev = np.eye(3, dtype=np.float64)
                frames_in_segment = 0
                continue

            back_pts, st_back, _ = cv2.calcOpticalFlowPyrLK(
                pre, prev_pre, cur_pts, None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
            )

            st = st.reshape(-1).astype(bool)
            err_flat = err.reshape(-1) if err is not None else np.full((len(active_tracks),), 1.0, dtype=np.float32)
            if back_pts is not None and st_back is not None:
                st_back = st_back.reshape(-1).astype(bool)
                fb_err = np.linalg.norm(back_pts.reshape(-1, 2) - prev_pts.reshape(-1, 2), axis=1)
            else:
                st_back = np.zeros_like(st, dtype=bool)
                fb_err = np.full((len(active_tracks),), 1e9, dtype=np.float32)

            cur_pts_flat = cur_pts.reshape(-1, 2)
            good_idx: List[int] = []
            for i in range(len(active_tracks)):
                if not st[i]:
                    continue
                if not st_back[i]:
                    continue
                if fb_err[i] > 1.5:
                    continue
                x, y = float(cur_pts_flat[i, 0]), float(cur_pts_flat[i, 1])
                xi, yi = int(round(x)), int(round(y))
                if xi < 0 or yi < 0 or xi >= w or yi >= h:
                    continue
                if work_mask[yi, xi] == 0:
                    continue
                if err_flat[i] > 2.5:
                    continue
                good_idx.append(i)

            if len(good_idx) < min_tracks_for_h:
                stats["homography_failures"] = int(stats["homography_failures"]) + 1
                segment_active = False
                active_tracks = []
                prev_pre = None
                prev_ms = None
                C_prev = np.eye(3, dtype=np.float64)
                frames_in_segment = 0
                continue

            src_cur = cur_pts_flat[good_idx].astype(np.float32)
            dst_prev = prev_pts.reshape(-1, 2)[good_idx].astype(np.float32)
            H_cur_to_prev, inlier_mask = cv2.findHomography(src_cur, dst_prev, method=cv2.RANSAC, ransacReprojThreshold=2.5)
            if H_cur_to_prev is None or inlier_mask is None:
                stats["homography_failures"] = int(stats["homography_failures"]) + 1
                segment_active = False
                active_tracks = []
                prev_pre = None
                prev_ms = None
                C_prev = np.eye(3, dtype=np.float64)
                frames_in_segment = 0
                continue

            inlier_mask = inlier_mask.reshape(-1).astype(bool)
            if int(np.count_nonzero(inlier_mask)) < min_tracks_for_h:
                stats["homography_failures"] = int(stats["homography_failures"]) + 1
                segment_active = False
                active_tracks = []
                prev_pre = None
                prev_ms = None
                C_prev = np.eye(3, dtype=np.float64)
                frames_in_segment = 0
                continue

            H_cur_to_prev = np.asarray(H_cur_to_prev, dtype=np.float64)
            C_cur = C_prev @ H_cur_to_prev  # current -> segment reference

            new_active_tracks: List[Dict] = []
            for local_idx, global_idx in enumerate(good_idx):
                if not inlier_mask[local_idx]:
                    continue
                track = active_tracks[global_idx]
                cur_pt = src_cur[local_idx].astype(np.float32)
                H_cur_to_anchor = track["anchor_cum_inv"] @ C_cur
                try:
                    pred_anchor = _perspective_transform_points(H_cur_to_anchor, cur_pt.reshape(1, 2))[0]
                except Exception:
                    continue

                residual = track["anchor_pt"] - pred_anchor
                residual_norm = float(np.hypot(residual[0], residual[1]))
                if np.isfinite(residual_norm) and residual_norm <= 12.0:
                    ax, ay = float(track["anchor_pt"][0]), float(track["anchor_pt"][1])
                    if 0 <= ax < w and 0 <= ay < h:
                        sample_x.append(ax)
                        sample_y.append(ay)
                        sample_dx.append(float(residual[0]))
                        sample_dy.append(float(residual[1]))
                        sample_w.append(float(1.0 / (1.0 + max(0.0, float(err_flat[global_idx])))))
                        stats["track_observations_used"] = int(stats["track_observations_used"]) + 1

                track["pt"] = cur_pt
                new_active_tracks.append(track)

            active_tracks = new_active_tracks
            frames_in_segment += 1

            if (frames_in_segment % reseed_every_frames == 0) or (len(active_tracks) < target_active_tracks // 2):
                extra_tracks, next_track_id = _seed_tracks(
                    preprocessed_gray=pre,
                    base_mask=work_mask,
                    current_cumulative_to_segref=C_cur,
                    next_track_id=next_track_id,
                    existing_tracks=active_tracks,
                    max_new_tracks=max(0, target_active_tracks - len(active_tracks)),
                )
                active_tracks.extend(extra_tracks)

            if len(active_tracks) < min_tracks_for_h:
                segment_active = False
                stats["segments_completed"] = int(stats["segments_completed"]) + 1
                prev_pre = None
                prev_ms = None
                C_prev = np.eye(3, dtype=np.float64)
                frames_in_segment = 0
                active_tracks = []
                continue

            prev_pre = pre
            prev_ms = actual_ms
            C_prev = C_cur

        if segment_active:
            stats["segments_completed"] = int(stats["segments_completed"]) + 1

    finally:
        pass

    if len(sample_x) < 500:
        raise RuntimeError(
            f"Night self-calibration failed: too few residual samples ({len(sample_x)}). "
            "Use a clearer/longer night video or reduce sample interval."
        )

    _emit(progress_callback, f"Night self-calibration: fitting map from {len(sample_x)} residual samples...")
    map_x, map_y, fit_stats = _fit_smooth_residual_field_to_maps(
        image_shape=(h, w),
        sample_x=sample_x,
        sample_y=sample_y,
        sample_dx=sample_dx,
        sample_dy=sample_dy,
        sample_w=sample_w,
        strength=float(strength),
        grid_step=48,
        progress_callback=progress_callback,
    )

    os.makedirs(os.path.dirname(os.path.abspath(map_x_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(map_y_path)), exist_ok=True)
    np.save(map_x_path, map_x.astype(np.float32))
    np.save(map_y_path, map_y.astype(np.float32))
    _emit(progress_callback, f"Night self-calibration: saved map_x -> {map_x_path}")
    _emit(progress_callback, f"Night self-calibration: saved map_y -> {map_y_path}")

    stats.update(fit_stats)
    stats["residual_samples_before_fit"] = len(sample_x)
    stats["map_shape"] = [int(h), int(w)]
    stats["map_x_path"] = str(map_x_path)
    stats["map_y_path"] = str(map_y_path)
    stats["sampling_notes"] = "Uses one median-composited frame per minute (first 20s only, next 40s skipped)."
    if auto_mask_output_path:
        stats["auto_mask_path"] = str(auto_mask_output_path)

    if metadata_output_path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(metadata_output_path)), exist_ok=True)
            with open(metadata_output_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            _emit(progress_callback, f"Night self-calibration: saved metadata -> {metadata_output_path}")
        except Exception as e:
            _emit(progress_callback, f"Night self-calibration: failed to save metadata ({e})")

    return {
        "success": True,
        "map_x_path": map_x_path,
        "map_y_path": map_y_path,
        "auto_mask": auto_mask,
        "manual_mask": manual_mask_resized,
        "work_mask": work_mask,
        "stats": stats,
    }


def estimate_distortion_map_from_night_sources(
    sources: Sequence[str],
    map_x_path: str,
    map_y_path: str,
    duration_minutes: float = 20.0,
    sample_interval_sec: float = 2.0,
    progress_callback: ProgressCallback = None,
    auto_mask_output_path: Optional[str] = None,
    metadata_output_path: Optional[str] = None,
    strength: float = 0.5,
    start_video_path: Optional[str] = None,
    manual_mask: Optional[np.ndarray] = None,
    use_auto_mask: bool = True,
) -> Dict[str, object]:
    video_paths = _select_video_sequence_from_sources(sources, start_video_path=start_video_path)
    if not video_paths:
        raise FileNotFoundError("No video file found in sources.")
    _emit(progress_callback, f"Night self-calibration: selected start video -> {video_paths[0]}")
    return estimate_distortion_map_from_night_video(
        video_path=[str(p) for p in video_paths],
        map_x_path=map_x_path,
        map_y_path=map_y_path,
        duration_minutes=duration_minutes,
        sample_interval_sec=sample_interval_sec,
        progress_callback=progress_callback,
        auto_mask_output_path=auto_mask_output_path,
        metadata_output_path=metadata_output_path,
        strength=strength,
        manual_mask=manual_mask,
        use_auto_mask=use_auto_mask,
    )

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
