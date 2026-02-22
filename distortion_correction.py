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

    duration_sec = float(duration_minutes) * 60.0
    _emit(progress_callback, f"Night self-calibration: loading probe frames from first {duration_minutes:.1f} min...")

    probe_interval_sec = max(10.0, sample_interval_sec * 6.0)
    probe_frames = _collect_probe_frames_for_mask(
        video_path=video_path,
        duration_sec=duration_sec,
        probe_interval_sec=probe_interval_sec,
        progress_callback=progress_callback,
    )
    if not probe_frames:
        raise RuntimeError("Failed to collect probe frames for automatic mask generation.")

    auto_mask = build_auto_night_selfcal_mask(probe_frames, progress_callback=progress_callback)
    if auto_mask is None:
        raise RuntimeError("Automatic mask generation failed.")

    h, w = auto_mask.shape[:2]
    if auto_mask_output_path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(auto_mask_output_path)), exist_ok=True)
            cv2.imwrite(auto_mask_output_path, auto_mask)
            _emit(progress_callback, f"Night self-calibration: auto mask saved to {auto_mask_output_path}")
        except Exception as e:
            _emit(progress_callback, f"Night self-calibration: failed to save auto mask ({e})")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    sample_x: List[float] = []
    sample_y: List[float] = []
    sample_dx: List[float] = []
    sample_dy: List[float] = []
    sample_w: List[float] = []

    stats: Dict[str, object] = {
        "video_path": str(video_path),
        "duration_minutes_requested": float(duration_minutes),
        "sample_interval_sec": float(sample_interval_sec),
        "frames_sampled_success": 0,
        "frames_sampled_failed": 0,
        "segments_started": 0,
        "segments_completed": 0,
        "homography_failures": 0,
        "gap_resets": 0,
        "track_observations_used": 0,
        "auto_mask_shape": [int(h), int(w)],
        "auto_mask_nonzero_ratio": float(np.count_nonzero(auto_mask) / auto_mask.size),
        "algorithm": "star-homography-consistency-selfcal",
    }

    try:
        duration_ms_est = _estimate_video_duration_ms(cap)
        end_ms = min(duration_sec * 1000.0, duration_ms_est) if duration_ms_est else (duration_sec * 1000.0)
        target_times = np.arange(0.0, max(1.0, end_ms), sample_interval_sec * 1000.0, dtype=np.float64)

        prev_pre = None
        prev_ms = None
        active_tracks: List[Dict] = []
        next_track_id = 0
        C_prev = np.eye(3, dtype=np.float64)  # previous frame -> segment reference frame
        segment_active = False
        frames_in_segment = 0

        max_track_gap_ms = max(6000.0, sample_interval_sec * 3000.0)
        target_active_tracks = 220
        min_tracks_for_h = 12
        min_seed_tracks = 30
        reseed_every_frames = 5

        _emit(progress_callback, f"Night self-calibration: sampling up to {len(target_times)} timestamps over {end_ms/1000.0:.1f}s...")

        for idx, t_ms in enumerate(target_times):
            if progress_callback and idx % 25 == 0:
                _emit(progress_callback, f"Night self-calibration: sampled {idx}/{len(target_times)} timestamps...")

            frame, actual_ms = _read_frame_at_ms(cap, float(t_ms))
            if frame is None:
                stats["frames_sampled_failed"] = int(stats["frames_sampled_failed"]) + 1
                continue
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            stats["frames_sampled_success"] = int(stats["frames_sampled_success"]) + 1

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
                    base_mask=auto_mask,
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
                if auto_mask[yi, xi] == 0:
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
                    base_mask=auto_mask,
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
        cap.release()

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
) -> Dict[str, object]:
    video_path = _find_first_video_path(sources)
    if not video_path:
        raise FileNotFoundError("No video file found in sources.")
    _emit(progress_callback, f"Night self-calibration: selected video -> {video_path}")
    return estimate_distortion_map_from_night_video(
        video_path=str(video_path),
        map_x_path=map_x_path,
        map_y_path=map_y_path,
        duration_minutes=duration_minutes,
        sample_interval_sec=sample_interval_sec,
        progress_callback=progress_callback,
        auto_mask_output_path=auto_mask_output_path,
        metadata_output_path=metadata_output_path,
        strength=strength,
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
