"""Build a fixed-camera astrometric model from tracked stellar trajectories.

The ordinary camera-model builder starts from one plate-solved image.  This
module adds a complementary path: stars are tracked through many video frames
and the camera projection is fitted so each track follows sidereal rotation.
An existing model supplies only the absolute sky orientation.  Trajectory
observations, including observations outside that model's trusted mask, drive
the detector-space fit and its independent holdout validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import cv2
import numpy as np
from scipy.optimize import least_squares

import camera_model_builder as static_builder
from camera_model_builder import _feature_matrix, _project, _write_json_atomic
from camera_plate_model import FixedCameraPlateModel, MODEL_TYPE
import local_wideangle_astrometry as local_astrometry


SIDEREAL_DAY_SECONDS = local_astrometry.SIDEREAL_DAY_SECONDS


@dataclass
class TrajectoryBuildRequest:
    source: str
    initial_model_path: str
    start: str = ""
    end: str = ""
    cache_root: Optional[str] = None
    sample_interval_seconds: float = 10.0
    maximum_frames: int = 420
    maximum_features: int = 1200
    minimum_track_observations: int = 18
    minimum_track_span_seconds: float = 150.0
    polynomial_degree: int = 5
    support_columns: int = 24
    support_rows: int = 14
    minimum_cell_tracks: int = 3


@dataclass
class TrajectoryBuildResult:
    success: bool
    model_path: str = ""
    report_path: str = ""
    enabled: bool = False
    target_met: bool = False
    support_fraction: float = 0.0
    residual_median_px: float = float("inf")
    residual_p95_px: float = float("inf")
    holdout_p95_px: float = float("inf")
    trajectory_count: int = 0
    observation_count: int = 0
    right_half_trajectory_count: int = 0
    error: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class _Track:
    identifier: int
    frame_indices: list[int] = field(default_factory=list)
    times_seconds: list[float] = field(default_factory=list)
    points: list[list[float]] = field(default_factory=list)


def _emit(callback: Optional[Callable[[str], None]], message: str) -> None:
    if callback:
        callback(str(message))


def _selected_paths(request: TrajectoryBuildRequest) -> list[str]:
    # A trajectory build needs all intervening segments, not the evenly
    # distributed subset used by the single-frame builder.
    return static_builder.select_video_paths(
        request.source, request.start, request.end, maximum=100000,
    )


def _sample_video_frames(
    paths: Sequence[str],
    interval_seconds: float,
    maximum_frames: int,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> tuple[list[np.ndarray], list[datetime], list[str]]:
    frames: list[np.ndarray] = []
    timestamps: list[datetime] = []
    sources: list[str] = []
    interval_seconds = max(0.5, float(interval_seconds))
    for path_index, path in enumerate(paths):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            continue
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0.1 or count <= 0:
            cap.release()
            continue
        duration = count / fps
        base = local_astrometry._capture_datetime(path)
        offsets = np.arange(0.0, max(0.001, duration - 0.001), interval_seconds)
        for offset in offsets:
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(count - 1, round(offset * fps)))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            timestamps.append(base + timedelta(seconds=float(offset)))
            sources.append(path)
        cap.release()
        if path_index % 10 == 0:
            _emit(progress_callback, f"軌跡用フレーム読込: {path_index + 1}/{len(paths)} 動画")
    if len(frames) > max(3, int(maximum_frames)):
        indices = np.linspace(0, len(frames) - 1, int(maximum_frames)).round().astype(int)
        frames = [frames[int(index)] for index in indices]
        timestamps = [timestamps[int(index)] for index in indices]
        sources = [sources[int(index)] for index in indices]
    if len(frames) < 3:
        raise RuntimeError("恒星軌跡に必要な動画フレームが不足しています")
    return frames, timestamps, sources


def _star_candidates(gray: np.ndarray, maximum: int) -> np.ndarray:
    """Return sub-pixel point-source centers without favouring image centre."""
    work = gray.astype(np.float32)
    highpass = work - cv2.GaussianBlur(work, (0, 0), 4.0)
    median = float(np.median(highpass))
    mad = float(np.median(np.abs(highpass - median))) + 1e-3
    normalized = (highpass - median) / (1.4826 * mad)
    response = np.clip(normalized, 0.0, 40.0).astype(np.float32)
    points = cv2.goodFeaturesToTrack(
        response,
        maxCorners=max(20, int(maximum)),
        qualityLevel=0.12,
        minDistance=6.0,
        blockSize=5,
        useHarrisDetector=False,
    )
    if points is None:
        return np.empty((0, 2), dtype=np.float32)
    points = points.reshape(-1, 2).astype(np.float32)
    height, width = gray.shape[:2]
    usable = (
        (points[:, 0] >= 7) & (points[:, 0] < width - 7)
        & (points[:, 1] >= 7) & (points[:, 1] < height - 7)
    )
    return points[usable]


def track_stellar_trajectories(
    frames: Sequence[np.ndarray],
    timestamps: Sequence[datetime],
    *,
    maximum_features: int = 1200,
    minimum_observations: int = 18,
    minimum_span_seconds: float = 150.0,
    reseed_interval: int = 4,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> list[_Track]:
    if len(frames) != len(timestamps) or len(frames) < 3:
        raise ValueError("frames and timestamps must have equal length >= 3")
    origin = timestamps[0]
    tracks: dict[int, _Track] = {}
    active_ids: list[int] = []
    active_points = np.empty((0, 2), dtype=np.float32)
    next_identifier = 0

    def seed(frame_index: int) -> None:
        nonlocal active_points, next_identifier
        available = max(0, int(maximum_features) - len(active_ids))
        if available <= 0:
            return
        candidates = _star_candidates(frames[frame_index], available * 2)
        if len(active_points):
            # Avoid duplicate tracks when a long-lived star is detected again.
            distances = np.linalg.norm(
                candidates[:, np.newaxis, :] - active_points[np.newaxis, :, :], axis=2
            )
            candidates = candidates[np.min(distances, axis=1) >= 7.0]
        for point in candidates[:available]:
            identifier = next_identifier
            next_identifier += 1
            time_seconds = (timestamps[frame_index] - origin).total_seconds()
            tracks[identifier] = _Track(
                identifier, [frame_index], [time_seconds], [[float(point[0]), float(point[1])]],
            )
            active_ids.append(identifier)
            active_points = np.vstack((active_points, point.reshape(1, 2)))

    seed(0)
    lk = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.01))
    for frame_index in range(1, len(frames)):
        previous = frames[frame_index - 1]
        current = frames[frame_index]
        if len(active_points):
            forward, status_f, _error = cv2.calcOpticalFlowPyrLK(
                previous, current, active_points.reshape(-1, 1, 2), None, **lk
            )
            backward, status_b, _error = cv2.calcOpticalFlowPyrLK(
                current, previous, forward, None, **lk
            )
            forward = forward.reshape(-1, 2)
            backward = backward.reshape(-1, 2)
            status = status_f.ravel().astype(bool) & status_b.ravel().astype(bool)
            fb_error = np.linalg.norm(backward - active_points, axis=1)
            step = np.linalg.norm(forward - active_points, axis=1)
            height, width = current.shape[:2]
            status &= np.isfinite(forward).all(axis=1)
            status &= fb_error <= 1.25
            status &= step <= 20.0
            status &= (forward[:, 0] >= 6) & (forward[:, 0] < width - 6)
            status &= (forward[:, 1] >= 6) & (forward[:, 1] < height - 6)
            kept_ids: list[int] = []
            kept_points: list[np.ndarray] = []
            time_seconds = (timestamps[frame_index] - origin).total_seconds()
            for index, identifier in enumerate(active_ids):
                if not status[index]:
                    continue
                point = forward[index]
                tracks[identifier].frame_indices.append(frame_index)
                tracks[identifier].times_seconds.append(time_seconds)
                tracks[identifier].points.append([float(point[0]), float(point[1])])
                kept_ids.append(identifier)
                kept_points.append(point)
            active_ids = kept_ids
            active_points = (
                np.asarray(kept_points, dtype=np.float32).reshape(-1, 2)
                if kept_points else np.empty((0, 2), dtype=np.float32)
            )
        if frame_index % max(1, int(reseed_interval)) == 0:
            seed(frame_index)
        if frame_index % 25 == 0:
            _emit(progress_callback, f"恒星追跡: {frame_index + 1}/{len(frames)} フレーム")

    accepted: list[_Track] = []
    for track in tracks.values():
        if len(track.points) < int(minimum_observations):
            continue
        span = track.times_seconds[-1] - track.times_seconds[0]
        if span < float(minimum_span_seconds):
            continue
        points = np.asarray(track.points, dtype=float)
        times = np.asarray(track.times_seconds, dtype=float)
        displacement = float(np.linalg.norm(points[-1] - points[0]))
        if displacement < max(1.5, 0.003 * span):
            # Fixed-pattern pixels and sensor defects do not follow the sky.
            continue
        scaled = (times - np.mean(times)) / max(1.0, span)
        design = np.column_stack((np.ones(len(times)), scaled, scaled * scaled))
        fitted = design @ np.linalg.lstsq(design, points, rcond=None)[0]
        smooth_error = np.linalg.norm(fitted - points, axis=1)
        if float(np.percentile(smooth_error, 95)) > 1.8:
            continue
        accepted.append(track)
    return accepted


def _grid_mask(payload: dict[str, Any], width: int, height: int) -> np.ndarray:
    grid = payload.get("support_grid")
    if not grid:
        return np.full((height, width), 255, dtype=np.uint8)
    values = np.asarray(grid, dtype=np.uint8)
    return cv2.resize(values, (width, height), interpolation=cv2.INTER_NEAREST)


def filter_tracks_by_sidereal_consistency(
    tracks: Sequence[_Track],
    timestamps: Sequence[datetime],
    initial_payload: dict[str, Any],
    *,
    maximum_p95_px: float = 6.0,
) -> tuple[list[_Track], np.ndarray]:
    """Reject smooth non-stellar tracks using sidereal reprojection.

    Cloud edges can pass the optical-flow smoothness check.  A real fixed-sky
    source must additionally map back to one constant ICRS coordinate after
    removing sidereal rotation.  The seed model is used for this temporal
    consistency test, not as evidence that an untrusted detector region has
    correct absolute coordinates.
    """
    model = FixedCameraPlateModel(initial_payload)
    reference = datetime.fromisoformat(
        str(initial_payload["reference_datetime"]).replace("Z", "+00:00")
    )
    accepted: list[_Track] = []
    residual_p95: list[float] = []
    for track in tracks:
        points = np.asarray(track.points, dtype=float)
        inverse_ra, inverse_dec = model.pixel_to_world_values(points[:, 0], points[:, 1])
        physical_ra = []
        for value, frame_index in zip(np.asarray(inverse_ra), track.frame_indices):
            target = timestamps[int(frame_index)]
            if reference.tzinfo is not None and target.tzinfo is None:
                target = target.replace(tzinfo=reference.tzinfo)
            elif reference.tzinfo is None and target.tzinfo is not None:
                target = target.replace(tzinfo=None)
            delta = (target - reference).total_seconds() / SIDEREAL_DAY_SECONDS * 360.0
            physical_ra.append((float(value) + delta) % 360.0)
        sky_ra = _circular_mean_degrees(np.asarray(physical_ra))
        sky_dec = float(np.nanmedian(inverse_dec))
        prediction = []
        for frame_index in track.frame_indices:
            target = timestamps[int(frame_index)]
            if reference.tzinfo is not None and target.tzinfo is None:
                target = target.replace(tzinfo=reference.tzinfo)
            elif reference.tzinfo is None and target.tzinfo is not None:
                target = target.replace(tzinfo=None)
            delta = (target - reference).total_seconds() / SIDEREAL_DAY_SECONDS * 360.0
            x, y = model.world_to_pixel_values((sky_ra - delta) % 360.0, sky_dec)
            prediction.append([float(x), float(y)])
        errors = np.linalg.norm(np.asarray(prediction) - points, axis=1)
        p95 = float(np.percentile(errors, 95))
        if np.isfinite(p95) and p95 <= float(maximum_p95_px):
            accepted.append(track)
            residual_p95.append(p95)
    return accepted, np.asarray(residual_p95, dtype=float)


def _parameters_for_degree(payload: dict[str, Any], degree: int) -> np.ndarray:
    terms = (degree + 1) * (degree + 2) // 2
    result = np.zeros(7 + terms * 2, dtype=float)
    result[:7] = np.asarray(payload["stg_parameters"], dtype=float)
    old = np.asarray(payload["correction_coefficients"], dtype=float).reshape(-1, 2)
    result[7:7 + min(terms, len(old)) * 2] = old[:terms].ravel()
    return result


def _circular_mean_degrees(values: np.ndarray) -> float:
    radians = np.deg2rad(values)
    return float(np.rad2deg(np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians)))) % 360.0)


def fit_trajectory_projection(
    tracks: Sequence[_Track],
    timestamps: Sequence[datetime],
    initial_payload: dict[str, Any],
    *,
    degree: int = 3,
    iterations: int = 4,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not tracks:
        raise RuntimeError("適合に使える恒星軌跡がありません")
    width = int(initial_payload["width"])
    height = int(initial_payload["height"])
    initial_model = FixedCameraPlateModel(initial_payload)
    initial_reference = datetime.fromisoformat(
        str(initial_payload["reference_datetime"]).replace("Z", "+00:00")
    )
    reference = timestamps[len(timestamps) // 2]
    if initial_reference.tzinfo is not None and reference.tzinfo is None:
        reference = reference.replace(tzinfo=initial_reference.tzinfo)
    elif initial_reference.tzinfo is None and reference.tzinfo is not None:
        reference = reference.replace(tzinfo=None)
    support_mask = _grid_mask(initial_payload, width, height)

    track_sky = np.zeros((len(tracks), 2), dtype=float)
    anchored = np.zeros(len(tracks), dtype=bool)
    for track_index, track in enumerate(tracks):
        points = np.asarray(track.points, dtype=float)
        frame_indices = np.asarray(track.frame_indices, dtype=int)
        inverse_ra, inverse_dec = initial_model.pixel_to_world_values(points[:, 0], points[:, 1])
        physical_ra = []
        for ra_value, frame_index in zip(np.asarray(inverse_ra), frame_indices):
            target = timestamps[int(frame_index)]
            if initial_reference.tzinfo is not None and target.tzinfo is None:
                target = target.replace(tzinfo=initial_reference.tzinfo)
            elif initial_reference.tzinfo is None and target.tzinfo is not None:
                target = target.replace(tzinfo=None)
            delta = (target - initial_reference).total_seconds() / SIDEREAL_DAY_SECONDS * 360.0
            physical_ra.append((float(ra_value) + delta) % 360.0)
        track_sky[track_index] = [
            _circular_mean_degrees(np.asarray(physical_ra)), float(np.nanmedian(inverse_dec)),
        ]
        rounded = np.rint(points).astype(int)
        rounded[:, 0] = np.clip(rounded[:, 0], 0, width - 1)
        rounded[:, 1] = np.clip(rounded[:, 1], 0, height - 1)
        anchored[track_index] = float(np.mean(support_mask[rounded[:, 1], rounded[:, 0]] > 0)) >= 0.70

    parameters = _parameters_for_degree(initial_payload, degree)
    initial_parameters = parameters.copy()
    training_rows: list[tuple[int, int, float, float, float]] = []
    holdout_rows: list[tuple[int, int, float, float, float]] = []
    for track_index, track in enumerate(tracks):
        for local_index, (frame_index, time_seconds, point) in enumerate(zip(
            track.frame_indices, track.times_seconds, track.points
        )):
            row = (track_index, int(frame_index), float(time_seconds), float(point[0]), float(point[1]))
            (holdout_rows if local_index % 5 == 4 else training_rows).append(row)
    if len(training_rows) < 100:
        raise RuntimeError("軌跡モデルの学習点が不足しています")

    def arrays(rows: Sequence[tuple[int, int, float, float, float]]):
        track_indices = np.asarray([row[0] for row in rows], dtype=int)
        frame_indices = np.asarray([row[1] for row in rows], dtype=int)
        target = np.asarray([[row[3], row[4]] for row in rows], dtype=float)
        delta = np.asarray([
            (timestamps[int(index)] - reference).total_seconds() / SIDEREAL_DAY_SECONDS * 360.0
            for index in frame_indices
        ])
        return track_indices, frame_indices, target, delta

    train_tracks, _train_frames, train_target, train_delta = arrays(training_rows)
    hold_tracks, _hold_frames, hold_target, hold_delta = arrays(holdout_rows)

    for _iteration in range(max(1, int(iterations))):
        train_ra = (track_sky[train_tracks, 0] - train_delta) % 360.0
        train_dec = track_sky[train_tracks, 1]
        world = static_builder._unit_vectors(train_ra, train_dec)
        # Degree >= 3 can still produce an attractive low residual while
        # exploding outside the tracks.  A small parameter prior suppresses
        # that unconstrained edge behaviour without freezing the lens model.
        scale = np.ones_like(parameters)
        scale[:3] = 0.02
        scale[3:5] = 0.04
        scale[5:7] = 20.0
        scale[7:] = 30.0

        def residual(candidate: np.ndarray) -> np.ndarray:
            image = (_project(candidate, world, width, height, degree) - train_target).ravel()
            prior = (candidate - initial_parameters) / scale
            return np.concatenate((image, prior * 0.12))

        fitted = least_squares(
            residual, parameters, loss="soft_l1", f_scale=1.25,
            x_scale="jac", max_nfev=1200,
        )
        parameters = fitted.x
        model_payload = {
            "model_type": MODEL_TYPE, "width": width, "height": height,
            "polynomial_degree": degree,
            "stg_parameters": parameters[:7].tolist(),
            "correction_coefficients": parameters[7:].reshape(-1, 2).tolist(),
            "reference_datetime": reference.isoformat(),
        }
        model = FixedCameraPlateModel(model_payload)
        # Re-estimate only unanchored track coordinates. Anchored tracks keep
        # the absolute orientation from the trusted part of the seed model.
        for track_index, track in enumerate(tracks):
            if anchored[track_index]:
                continue
            # Keep every fifth observation genuinely unseen: it must not
            # influence either camera parameters or the latent sky coordinate
            # assigned to this trajectory.
            training_local = np.asarray(
                [index for index in range(len(track.points)) if index % 5 != 4],
                dtype=int,
            )
            points = np.asarray(track.points, dtype=float)[training_local]
            frame_indices = np.asarray(track.frame_indices, dtype=int)[training_local]
            inverse_ra, inverse_dec = model.pixel_to_world_values(points[:, 0], points[:, 1])
            physical_ra = []
            for value, frame_index in zip(np.asarray(inverse_ra), frame_indices):
                delta = (
                    timestamps[int(frame_index)] - reference
                ).total_seconds() / SIDEREAL_DAY_SECONDS * 360.0
                physical_ra.append((float(value) + delta) % 360.0)
            track_sky[track_index] = [
                _circular_mean_degrees(np.asarray(physical_ra)), float(np.nanmedian(inverse_dec)),
            ]

    payload = {
        "model_type": MODEL_TYPE, "width": width, "height": height,
        "polynomial_degree": degree,
        "stg_parameters": parameters[:7].tolist(),
        "correction_coefficients": parameters[7:].reshape(-1, 2).tolist(),
        "residual_grid": None,
        "reference_datetime": reference.isoformat(),
    }
    train_world = static_builder._unit_vectors(
        (track_sky[train_tracks, 0] - train_delta) % 360.0,
        track_sky[train_tracks, 1],
    )
    hold_world = static_builder._unit_vectors(
        (track_sky[hold_tracks, 0] - hold_delta) % 360.0,
        track_sky[hold_tracks, 1],
    )
    train_residual = np.linalg.norm(
        _project(parameters, train_world, width, height, degree) - train_target, axis=1
    )
    holdout_residual = np.linalg.norm(
        _project(parameters, hold_world, width, height, degree) - hold_target, axis=1
    ) if len(hold_target) else np.asarray([], dtype=float)

    per_track_p95 = np.full(len(tracks), np.inf)
    for track_index in range(len(tracks)):
        values = train_residual[train_tracks == track_index]
        if len(values):
            per_track_p95[track_index] = float(np.percentile(values, 95))
    stats = {
        "track_sky": track_sky,
        "anchored": anchored,
        "train_tracks": train_tracks,
        "train_target": train_target,
        "train_residual": train_residual,
        "hold_tracks": hold_tracks,
        "hold_target": hold_target,
        "holdout_residual": holdout_residual,
        "per_track_p95": per_track_p95,
    }
    return payload, stats


def _trajectory_support_grid(
    tracks: Sequence[_Track],
    per_track_p95: np.ndarray,
    width: int,
    height: int,
    columns: int,
    rows: int,
    minimum_tracks: int,
) -> tuple[list[list[int]], float, np.ndarray]:
    memberships = [[set() for _column in range(columns)] for _row in range(rows)]
    for track_index, track in enumerate(tracks):
        if not np.isfinite(per_track_p95[track_index]) or per_track_p95[track_index] > 3.0:
            continue
        for x, y in track.points:
            column = min(columns - 1, max(0, int(float(x) / width * columns)))
            row = min(rows - 1, max(0, int(float(y) / height * rows)))
            memberships[row][column].add(track_index)
    counts = np.asarray(
        [[len(memberships[row][column]) for column in range(columns)] for row in range(rows)],
        dtype=int,
    )
    grid = (counts >= max(1, int(minimum_tracks))).astype(np.uint8)
    return grid.tolist(), float(np.mean(grid)), counts


def build_trajectory_camera_model(
    request: TrajectoryBuildRequest,
    *,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> TrajectoryBuildResult:
    try:
        initial_path = Path(request.initial_model_path).expanduser().resolve()
        initial_payload = json.loads(initial_path.read_text(encoding="utf-8"))
        if initial_payload.get("model_type") != MODEL_TYPE:
            raise ValueError("初期モデルの形式が固定カメラモデルではありません")
        paths = _selected_paths(request)
        _emit(progress_callback, f"軌跡解析対象: {len(paths)} 動画")
        frames, timestamps, frame_sources = _sample_video_frames(
            paths, request.sample_interval_seconds, request.maximum_frames, progress_callback,
        )
        _emit(progress_callback, f"恒星追跡を開始: {len(frames)} フレーム")
        raw_tracks = track_stellar_trajectories(
            frames, timestamps,
            maximum_features=request.maximum_features,
            minimum_observations=request.minimum_track_observations,
            minimum_span_seconds=request.minimum_track_span_seconds,
            progress_callback=progress_callback,
        )
        tracks, seed_consistency = filter_tracks_by_sidereal_consistency(
            raw_tracks, timestamps, initial_payload,
        )
        if len(tracks) < 20:
            raise RuntimeError(f"有効な恒星軌跡が不足しています ({len(tracks)}本)")
        _emit(
            progress_callback,
            f"恒星時運動に一致する軌跡: {len(tracks)}/{len(raw_tracks)}本",
        )
        payload, fit = fit_trajectory_projection(
            tracks, timestamps, initial_payload, degree=request.polynomial_degree,
        )
        width, height = int(payload["width"]), int(payload["height"])
        support, support_fraction, support_counts = _trajectory_support_grid(
            tracks, fit["per_track_p95"], width, height,
            request.support_columns, request.support_rows, request.minimum_cell_tracks,
        )
        train = fit["train_residual"]
        holdout = fit["holdout_residual"]
        median = float(np.median(train))
        p95 = float(np.percentile(train, 95))
        holdout_p95 = float(np.percentile(holdout, 95)) if len(holdout) else float("inf")
        right_tracks = sum(
            1 for track in tracks if np.median(np.asarray(track.points)[:, 0]) >= width / 2
        )
        enabled = bool(
            len(tracks) >= 20 and right_tracks >= 5
            and p95 <= 3.0 and holdout_p95 <= 3.25 and support_fraction >= 0.20
        )
        root = (
            Path(request.cache_root).expanduser().resolve()
            if request.cache_root else local_astrometry._default_cache_root()
        )
        _date, camera_alias = local_astrometry._night_identity(paths[0], width, height)
        model_dir = root / "camera_models" / f"trajectory-{camera_alias}-{_date}-{width}x{height}"
        model_path = model_dir / "camera_model.json"
        report_path = model_dir / "build_report.json"
        payload.update({
            "wcs_path": str(initial_payload.get("wcs_path", "")),
            "camera_aliases": [camera_alias],
            "valid_dates": [_date],
            "enabled": enabled,
            "target_met": bool(enabled and support_fraction >= 0.80),
            "support_grid": support,
            "support_fraction": support_fraction,
            "model_label": "VIDEO STAR-TRAJECTORY CAMERA MODEL" if enabled else "VIDEO STAR-TRAJECTORY MODEL (CANDIDATE)",
            "model_revision": "video-star-trajectories-v1",
            "algorithm_version": "video-star-trajectories-v1",
            # Sidereal trajectories plus held-out observations provide the
            # temporal projection.  The display policy additionally gates
            # each constellation edge on current-frame endpoint detections so
            # clouds do not receive synthetic blue lines.
            "verified_constellation_only": False,
            "constellation_anchor_filter": False,
            "constellation_anchor_tolerance_px": 4.0,
            "constellation_star_alignment": False,
            "constellation_render_policy": "model-supported-detected-endpoints",
            "constellation_support_policy": "display-bridged-internal-holes",
            "constellation_temporal_hold_frames": 3,
            "constellation_projection_guard": "sky-pixel-sky-roundtrip",
            "constellation_max_sky_roundtrip_deg": 0.1,
            "catalog_stars": [],
            "source_videos": sorted(set(frame_sources)),
            "fit_stats": {
                "residual_median_px": median,
                "residual_p95_px": p95,
                "holdout_residual_p95_px": holdout_p95,
                "sample_count": int(len(train)),
                "holdout_sample_count": int(len(holdout)),
                "trajectory_count": int(len(tracks)),
                "right_half_trajectory_count": int(right_tracks),
                "fit_source": "tracked stellar trajectories with sidereal bundle adjustment",
            },
            "trajectory_validation": {
                "frame_count": len(frames),
                "sample_interval_seconds": request.sample_interval_seconds,
                "raw_track_count": len(raw_tracks),
                "track_count": len(tracks),
                "right_half_track_count": right_tracks,
                "seed_consistency_p95_px": {
                    "median": float(np.median(seed_consistency)),
                    "p95": float(np.percentile(seed_consistency, 95)),
                    "maximum": float(np.max(seed_consistency)),
                },
                "support_track_counts": support_counts.tolist(),
                "training_policy": "four observations train, every fifth observation holdout",
                "stationary_features_rejected": True,
            },
            "created_at": datetime.now().astimezone().isoformat(),
        })
        report = {
            "model": payload,
            "request": request.__dict__,
            "tracks": [
                {
                    "id": track.identifier,
                    "observation_count": len(track.points),
                    "span_seconds": track.times_seconds[-1] - track.times_seconds[0],
                    "median_x": float(np.median(np.asarray(track.points)[:, 0])),
                    "median_y": float(np.median(np.asarray(track.points)[:, 1])),
                    "residual_p95_px": float(fit["per_track_p95"][index]),
                }
                for index, track in enumerate(tracks)
            ],
        }
        _write_json_atomic(model_path, payload)
        _write_json_atomic(report_path, report)
        return TrajectoryBuildResult(
            success=True,
            model_path=str(model_path),
            report_path=str(report_path),
            enabled=enabled,
            target_met=bool(enabled and support_fraction >= 0.80),
            support_fraction=support_fraction,
            residual_median_px=median,
            residual_p95_px=p95,
            holdout_p95_px=holdout_p95,
            trajectory_count=len(tracks),
            observation_count=len(train) + len(holdout),
            right_half_trajectory_count=right_tracks,
            diagnostics={
                "frame_count": len(frames), "raw_track_count": len(raw_tracks),
                "support_counts": support_counts.tolist(),
            },
        )
    except Exception as exc:
        return TrajectoryBuildResult(False, error=f"{type(exc).__name__}: {exc}")
