"""Automatic, API-free astrometry for fixed ultra-wide night cameras.

The first solve for a camera/night builds a temporal star image, removes fixed
sensor noise, solves against a compact local Tycho-2 index, and writes a SIP
WCS calibration.  The calibration is reused for the whole night.  Subsequent
frames update the celestial grid by sidereal rotation while retaining the same
lens-distortion model.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import contourpy
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS
from astropy.wcs.utils import fit_wcs_from_points

from camera_plate_model import FixedCameraPlateModel, MODEL_TYPE as CAMERA_MODEL_TYPE


# v2 uses a match-count-aware SIP order.  The previous fixed high-order fit
# could overfit sparse matches and bend constellation lines near the rim.
ALGORITHM_VERSION = "local-wideangle-sip-v2"
SIDEREAL_DAY_SECONDS = 86164.0905
_DATE_DIR = re.compile(r"^(?:19|20)\d{6}$")
_cache_lock = threading.RLock()
_loaded_calibrations: Dict[str, Tuple[float, Dict[str, Any], Any]] = {}
_forward_grid_cache: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
_constellation_lines: Optional[Tuple[np.ndarray, ...]] = None

# Sparse sky stick figures are unsafe to connect directly in an ultra-wide
# projection.  Sample their sky paths densely, then reject gaps caused by the
# calibrated-area boundary or a projection branch.
_CONSTELLATION_MAX_ANGULAR_STEP_DEG = 0.5
_CONSTELLATION_MAX_PIXEL_STEP_FACTOR = 0.20
# A high-order detector polynomial can be locally many-to-one even when its
# residuals at tracked stars are excellent.  In that case sky->pixel lands on
# a different inverse branch and a short constellation edge can cross most of
# the image.  Correct branches round-trip to far below an arcminute here; a
# 0.1-degree ceiling leaves ample numerical margin while rejecting folds.
_CONSTELLATION_MAX_SKY_ROUNDTRIP_DEG = 0.1
_CONSTELLATION_ALIGNMENT_MAX_MAG = 4.8
_CONSTELLATION_ALIGNMENT_MATCH_RADIUS_PX = 8.0
_CONSTELLATION_ALIGNMENT_MIN_MATCHES = 4


class CalibrationNotFoundError(RuntimeError):
    pass


def _emit(callback: Optional[Callable], message: str) -> None:
    if not callback:
        return
    try:
        callback(message)
    except TypeError:
        callback((message, None))


def is_available() -> bool:
    try:
        importlib_metadata.version("astrometry")
        return True
    except importlib_metadata.PackageNotFoundError:
        return False


def _default_cache_root() -> Path:
    configured = os.environ.get("METEOR_ASTROMETRY_CACHE")
    if configured:
        return Path(configured).expanduser().resolve()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "MeteorDetector" / "astrometry"
    return Path.home() / ".cache" / "meteor_detector" / "astrometry"


def _open_video(path: str) -> cv2.VideoCapture:
    thread_property = getattr(cv2, "CAP_PROP_N_THREADS", None)
    if thread_property is not None:
        try:
            cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG, [int(thread_property), 2])
            if cap.isOpened():
                return cap
            cap.release()
        except (cv2.error, TypeError, ValueError):
            pass
    return cv2.VideoCapture(path)


def _capture_datetime(path: str) -> datetime:
    source = Path(path)
    parts = source.parts
    for index, part in enumerate(parts):
        if not _DATE_DIR.fullmatch(part):
            continue
        hour = 0
        minute = 0
        if index + 1 < len(parts) and parts[index + 1].isdigit():
            hour = max(0, min(23, int(parts[index + 1][:2])))
        if source.stem.isdigit():
            minute = max(0, min(59, int(source.stem[:2])))
        try:
            return datetime.strptime(part, "%Y%m%d").replace(hour=hour, minute=minute)
        except ValueError:
            continue
    filename_match = re.search(
        r"((?:19|20)\d{6})[^0-9]?([01]\d|2[0-3])?([0-5]\d)?", source.stem
    )
    if filename_match:
        try:
            value = datetime.strptime(filename_match.group(1), "%Y%m%d")
            return value.replace(
                hour=int(filename_match.group(2) or 0),
                minute=int(filename_match.group(3) or 0),
            )
        except ValueError:
            pass
    try:
        stat = source.stat()
        return datetime.fromtimestamp(getattr(stat, "st_birthtime", stat.st_mtime))
    except OSError:
        return datetime.now()


def _night_identity(path: str, width: int, height: int) -> Tuple[str, str]:
    source = Path(path).expanduser().resolve()
    date = _capture_datetime(str(source)).strftime("%Y%m%d")
    camera_root = source.parent
    for parent in (source.parent, *source.parents):
        if _DATE_DIR.fullmatch(parent.name):
            camera_root = parent.parent
            break
    digest = hashlib.sha256(
        f"{camera_root}|{width}x{height}".encode("utf-8")
    ).hexdigest()[:12]
    return date, digest


def _calibration_paths(
    source_path: str,
    width: int,
    height: int,
    cache_root: Optional[str] = None,
    reference_key: Optional[str] = None,
):
    root = Path(cache_root).expanduser().resolve() if cache_root else _default_cache_root()
    date, camera = _night_identity(source_path, width, height)
    night = root / "calibrations" / camera / date
    night.mkdir(parents=True, exist_ok=True)
    suffix = f"_{reference_key}" if reference_key else ""
    return {
        "root": root,
        "night": night,
        "metadata": night / f"calibration{suffix}.json",
        "wcs": night / f"wideangle_sip{suffix}.wcs",
        "diagnostic": night / f"star_extraction{suffix}.jpg",
        "reference": night / f"reference{suffix}.jpg",
        "validation": night / f"calibration_validation{suffix}.jpg",
        "index_cache": root / "indexes",
    }


def _registered_camera_model(
    source_path: str,
    width: int,
    height: int,
    cache_root: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Return the best reusable model registered for this source root.

    ``valid_dates`` records nights used for validation; it is not an expiry
    date for a physically fixed camera.  Prefer an exact-night model, but
    fall back to the highest-quality enabled model for the same camera and
    resolution when a later night has no exact validation entry.
    """
    root = Path(cache_root).expanduser().resolve() if cache_root else _default_cache_root()
    _date, camera = _night_identity(source_path, width, height)
    candidates = []
    for path in sorted((root / "camera_models").glob("*/camera_model.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("model_type") != CAMERA_MODEL_TYPE:
                continue
            # Older experimental builder output did not record a model
            # revision and could reuse a stale registered orientation.  Keep
            # it out of automatic selection; explicit paths remain supported.
            if not payload.get("algorithm_version") and not payload.get("model_revision"):
                continue
            if not payload.get("enabled", False):
                continue
            if int(payload.get("width", 0)) != width or int(payload.get("height", 0)) != height:
                continue
            if camera not in payload.get("camera_aliases", []):
                continue
            valid_dates = payload.get("valid_dates")
            if isinstance(valid_dates, str):
                valid_dates = [valid_dates]
            normalized_dates = {str(value)[:10] for value in (valid_dates or [])}
            exact_night = not normalized_dates or _date in normalized_dates
            wcs_path = Path(payload["wcs_path"]).expanduser()
            if not wcs_path.exists():
                continue
            reference = datetime.fromisoformat(str(payload["reference_datetime"]).replace("Z", "+00:00"))
            support = payload.get("support_fraction", payload.get("sip_support_fraction"))
            if support is None:
                validation = payload.get("support_grid_validation")
                if isinstance(validation, dict) and validation.get("validated_fraction") is not None:
                    support = validation.get("validated_fraction")
            if support is None:
                grid = payload.get("support_grid")
                if isinstance(grid, dict):
                    grid = grid.get("grid") or grid.get("values")
                try:
                    cells = [bool(cell) for row in grid for cell in row]
                    support = sum(cells) / len(cells) if cells else 0.0
                except (TypeError, ValueError):
                    support = 0.0
            try:
                support = float(support)
                support = support / 100.0 if support > 1.0 else support
            except (TypeError, ValueError):
                support = 0.0
            fit_stats = payload.get("fit_stats") or {}
            quality = payload.get("residual_p95_px", fit_stats.get("residual_p95_px", float("inf")))
            try:
                quality = float(quality)
            except (TypeError, ValueError):
                quality = float("inf")
            candidates.append((
                exact_night,
                max(0.0, min(1.0, support)),
                quality,
                reference,
                path,
                wcs_path,
                payload,
            ))
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
    if not candidates:
        return None

    # Exact-night validation wins.  Within the same group, prefer broader
    # coverage, lower residual error, and the newer reference model.
    candidates.sort(
        key=lambda item: (
            not item[0],
            -item[1],
            item[2],
            -item[3].timestamp(),
        )
    )
    exact_night, _support, _quality, _reference, path, wcs_path, payload = candidates[0]
    return {
        "wcs_file": str(wcs_path),
        "calibration_path": str(path),
        "plate_solve_datetime": _reference,
        "job_id": "local-wideangle-camera-model",
        **payload,
        "_model_date_match": bool(exact_night),
    }


def _video_sample_stack(
    video_path: str,
    max_frames: int = 1500,
    average_prefix: int = 50,
    sample_stride: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    cap = _open_video(video_path)
    if not cap.isOpened():
        raise IOError(f"動画を開けません: {video_path}")
    prefix_sum: Optional[np.ndarray] = None
    prefix_count = 0
    samples: List[np.ndarray] = []
    frame_index = 0
    try:
        while frame_index < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prefix_sum is None:
                prefix_sum = np.zeros_like(gray, dtype=np.float32)
            if frame_index < average_prefix:
                prefix_sum += gray
                prefix_count += 1
            if frame_index % sample_stride == 0:
                samples.append(gray)
            frame_index += 1
    finally:
        cap.release()
    if prefix_sum is None or prefix_count == 0 or not samples:
        raise IOError(f"較正用フレームを読めません: {video_path}")
    average = np.clip(prefix_sum / prefix_count, 0, 255).astype(np.uint8)
    return average, np.stack(samples)


def _video_centered_stack(
    video_path: str,
    center_frame_index: int,
    half_window_seconds: float = 10.0,
    maximum_samples: int = 31,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Build a mean image from a time window centered on one video frame.

    The full window is accumulated without retaining every frame in memory.
    A sparse set of temporal samples is returned as well so the existing
    fixed-pattern-noise suppression can be applied during star extraction.
    """
    half_window_seconds = float(half_window_seconds)
    if half_window_seconds < 0.0:
        raise ValueError("half_window_seconds must be non-negative")
    cap = _open_video(video_path)
    if not cap.isOpened():
        raise IOError(f"動画を開けません: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not (0.1 <= fps <= 240.0):
            raise IOError(f"動画のフレームレートを読めません: {video_path}")
        center = max(0, int(center_frame_index))
        if frame_count > 0:
            center = min(center, frame_count - 1)
        radius = max(0, int(round(fps * half_window_seconds)))
        start = max(0, center - radius)
        end = center + radius
        if frame_count > 0:
            end = min(end, frame_count - 1)
        expected = max(1, end - start + 1)
        stride = max(1, int(np.ceil(expected / max(1, int(maximum_samples)))))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        accumulated: Optional[np.ndarray] = None
        samples: List[np.ndarray] = []
        read_count = 0
        while read_count < expected:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if accumulated is None:
                accumulated = np.zeros_like(gray, dtype=np.float32)
            accumulated += gray
            if read_count % stride == 0 or read_count == expected - 1:
                samples.append(gray.copy())
            read_count += 1
    finally:
        cap.release()
    if accumulated is None or read_count == 0 or not samples:
        raise IOError(f"スタック用フレームを読めません: {video_path}")
    average = np.clip(accumulated / read_count, 0, 255).astype(np.uint8)
    info = {
        "stack_method": "centered_mean",
        "stack_half_window_seconds": half_window_seconds,
        "stack_start_frame": int(start),
        "stack_end_frame": int(start + read_count - 1),
        "stack_frame_count": int(read_count),
        "stack_fps": fps,
        "reference_frame_index": int(center),
    }
    return average, np.stack(samples), info


def _frames_sample_stack(frames: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if not frames:
        raise ValueError("frames must not be empty")
    gray = [frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    prefix = gray[: min(50, len(gray))]
    average = np.mean(np.stack(prefix).astype(np.float32), axis=0).astype(np.uint8)
    stride = max(1, len(gray) // 30)
    return average, np.stack(gray[::stride][:30])


def _extract_stars(
    average: np.ndarray,
    temporal_samples: Optional[np.ndarray] = None,
    maximum_stars: int = 180,
    exclude_lower_region: bool = True,
    build_diagnostic: bool = True,
) -> Tuple[List[List[float]], Optional[np.ndarray], np.ndarray]:
    work = average.astype(np.float32)
    if temporal_samples is not None and len(temporal_samples) >= 3:
        fixed = np.median(temporal_samples, axis=0).astype(np.float32)
        work = work - fixed + float(np.median(fixed))
    highpass = work - cv2.GaussianBlur(work, (0, 0), 5)
    median = float(np.median(highpass))
    mad = float(np.median(np.abs(highpass - median))) + 1e-3
    zscore = (highpass - median) / (1.4826 * mad)
    height, width = average.shape[:2]
    binary = (zscore > 5.0).astype(np.uint8)
    if exclude_lower_region:
        binary[int(height * 0.84):, :] = 0
    binary[:, :5] = 0
    binary[:, max(0, width - 5):] = 0
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    central: List[Tuple[float, List[float]]] = []
    outer: List[Tuple[float, List[float]]] = []
    for label in range(1, count):
        x, y, box_width, box_height, area = stats[label]
        cx, cy = centroids[label]
        aspect = max(box_width, box_height) / max(1, min(box_width, box_height))
        if not (1 <= area <= 45 and max(box_width, box_height) <= 10 and aspect <= 3.0):
            continue
        component = labels[y:y + box_height, x:x + box_width] == label
        flux = float(np.sum(np.maximum(highpass[y:y + box_height, x:x + box_width], 0)[component]))
        record = (flux, [float(cx), float(cy)])
        if width * 0.20 < cx < width * 0.80 and height * 0.15 < cy < height * 0.80:
            central.append(record)
        else:
            outer.append(record)
    central.sort(reverse=True)
    outer.sort(reverse=True)
    # Bright central stars make the first quad robust; outer stars then let the
    # SIP fit learn the edge curvature rather than treating it as noise.
    selected = central[:180] + outer[: max(0, maximum_stars - min(180, len(central)))]
    stars = [position for _flux, position in selected[:maximum_stars]]
    diagnostic_bgr = None
    if build_diagnostic:
        diagnostic = np.clip((zscore - 2.0) * 35.0, 0, 255).astype(np.uint8)
        if exclude_lower_region:
            diagnostic[int(height * 0.84):, :] = 0
        diagnostic_bgr = cv2.cvtColor(diagnostic, cv2.COLOR_GRAY2BGR)
        for index, (x, y) in enumerate(stars):
            color = (0, 255, 0) if index < min(180, len(central)) else (0, 180, 255)
            cv2.circle(diagnostic_bgr, (round(x), round(y)), 4, color, 1, cv2.LINE_AA)
    reference = np.clip(work, 0, 255).astype(np.uint8)
    return stars, diagnostic_bgr, reference


def _solve_stars(
    stars: Sequence[Sequence[float]],
    width: int,
    height: int,
    date_obs: datetime,
    paths: Dict[str, Path],
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    if len(stars) < 20:
        raise RuntimeError(f"ローカル較正に必要な星が不足しています ({len(stars)}個)")
    if not is_available():
        raise RuntimeError("astrometry Python package is not installed")
    _emit(progress_callback, f"ローカル星図照合中（星候補 {len(stars)}個、外部APIなし）...")
    request = {
        "stars": [[float(x), float(y)] for x, y in stars],
        "width": width,
        "height": height,
        "date_obs": date_obs.isoformat(),
        "output_wcs": str(paths["wcs"]),
        "index_cache": str(paths["index_cache"]),
        "index_scales": [15, 16, 17, 18, 19],
        "scale_lower_arcsec": 120.0,
        "scale_upper_arcsec": 260.0,
        "sip_order": 5,
        "sip_inverse_order": 5,
        "positional_noise_pixels": 3.0,
        "distractor_ratio": 0.65,
        "maximum_matches": 2,
        "maximum_quads": 200000,
    }
    if paths["metadata"].exists():
        try:
            previous = json.loads(paths["metadata"].read_text(encoding="utf-8"))
            if "center_ra_deg" in previous and "center_dec_deg" in previous:
                request["position_hint"] = {
                    "ra_deg": float(previous["center_ra_deg"]),
                    "dec_deg": float(previous["center_dec_deg"]),
                    "radius_deg": 12.0,
                }
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    worker = Path(__file__).with_name("local_wideangle_worker.py")
    with tempfile.TemporaryDirectory(prefix="meteor-local-astrometry-") as temporary:
        request_path = Path(temporary) / "request.json"
        response_path = Path(temporary) / "response.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        process = subprocess.run(
            [sys.executable, str(worker), str(request_path), str(response_path)],
            cwd=temporary,
            capture_output=True,
            text=True,
            timeout=360,
        )
        if not response_path.exists():
            raise RuntimeError(
                f"ローカル星図照合ワーカーが結果を返しません: {process.stderr[-500:]}"
            )
        response = json.loads(response_path.read_text(encoding="utf-8"))
    if not response.get("matched"):
        raise RuntimeError(f"ローカル星図照合に失敗しました: {response.get('error', 'unknown')}")
    return response


def _persist_calibration(
    source_path: str,
    date_obs: datetime,
    width: int,
    height: int,
    paths: Dict[str, Path],
    solve_result: Dict[str, Any],
    star_count: int,
    reference_frame_index: Optional[int] = None,
) -> Dict[str, Any]:
    payload = {
        "algorithm_version": ALGORITHM_VERSION,
        "source": str(Path(source_path).expanduser()),
        "reference_datetime": date_obs.isoformat(),
        "width": int(width),
        "height": int(height),
        "star_count": int(star_count),
        "wcs_path": str(paths["wcs"]),
        "calibration_path": str(paths["metadata"]),
        "diagnostic_path": str(paths["diagnostic"]),
        "reference_path": str(paths["reference"]),
        "validation_path": str(paths["validation"]),
        "solver": "local-astrometry-net-tycho2-sip-adaptive",
        "external_api_used": False,
        **{key: solve_result[key] for key in (
            "center_ra_deg", "center_dec_deg", "scale_arcsec_per_pixel", "logodds",
            "sip_refined", "sip_order", "sip_match_count",
            "sip_residual_median_px", "sip_residual_p95_px", "sip_reason",
            "sip_support_hull", "validation_path",
        ) if key in solve_result},
    }
    # Keep the solver's catalog in the calibration metadata.  The automatic
    # fixed-camera builder uses it as an auditable record of the stars that
    # established the initial WCS; older calibration files simply omit it.
    if solve_result.get("catalog_stars"):
        payload["catalog_stars"] = solve_result["catalog_stars"]
    support_hull = solve_result.get("sip_support_hull")
    if support_hull and len(support_hull) >= 3:
        hull = np.asarray(support_hull, dtype=np.float32)
        payload["sip_support_fraction"] = float(
            cv2.contourArea(hull) / max(1.0, float(width * height))
        )
    if reference_frame_index is not None:
        payload["reference_frame_index"] = int(reference_frame_index)
    temporary = paths["metadata"].with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, paths["metadata"])
    return payload


def _refine_sip_wcs(
    paths: Dict[str, Path],
    observed_stars: Sequence[Sequence[float]],
    catalog_stars: Sequence[Dict[str, Any]],
    width: int,
    height: int,
    date_obs: datetime,
) -> Dict[str, Any]:
    """Fit actual SIP coefficients from mutual catalog/image star matches."""
    if len(observed_stars) < 16 or len(catalog_stars) < 16:
        return {"sip_refined": False, "sip_reason": "insufficient correspondences"}
    with fits.open(paths["wcs"]) as hdul:
        base_header = hdul[0].header.copy()
    current = WCS(base_header, relax=True, fix=False)
    observed = np.asarray(observed_stars, dtype=float)
    sky = SkyCoord(
        ra=np.asarray([star["ra_deg"] for star in catalog_stars]) * u.deg,
        dec=np.asarray([star["dec_deg"] for star in catalog_stars]) * u.deg,
        frame="icrs",
    )
    ideal_x, ideal_y = current.world_to_pixel(sky)
    ideal = np.column_stack((ideal_x, ideal_y))

    def polynomial_features(points: np.ndarray, degree: int) -> np.ndarray:
        x = (points[:, 0] - width / 2) / max(1.0, width / 2)
        y = (points[:, 1] - height / 2) / max(1.0, height / 2)
        columns = [np.ones(len(points)), x, y]
        if degree >= 2:
            columns.extend((x * x, x * y, y * y))
        if degree >= 3:
            columns.extend((x ** 3, x * x * y, x * y * y, y ** 3))
        if degree >= 4:
            columns.extend((x ** 4, x ** 3 * y, x * x * y * y, x * y ** 3, y ** 4))
        return np.asarray(columns).T

    def mutual_pairs(predicted: np.ndarray, threshold: float):
        valid = np.isfinite(predicted).all(axis=1)
        valid &= predicted[:, 0] > -width * 0.08
        valid &= predicted[:, 0] < width * 1.08
        valid &= predicted[:, 1] > -height * 0.08
        valid &= predicted[:, 1] < height * 1.08
        catalog_indices = np.flatnonzero(valid)
        if not len(catalog_indices):
            return np.asarray([], dtype=int), np.asarray([], dtype=int)
        distances = np.linalg.norm(
            predicted[catalog_indices, np.newaxis, :] - observed[np.newaxis, :, :], axis=2
        )
        nearest_observed = np.argmin(distances, axis=1)
        nearest_catalog = np.argmin(distances, axis=0)
        catalog_matches: List[int] = []
        observed_matches: List[int] = []
        for local_index, observed_index in enumerate(nearest_observed):
            if nearest_catalog[observed_index] != local_index:
                continue
            if distances[local_index, observed_index] >= threshold:
                continue
            catalog_matches.append(int(catalog_indices[local_index]))
            observed_matches.append(int(observed_index))
        return np.asarray(catalog_matches), np.asarray(observed_matches)

    # Bootstrap a detector-space distortion map from the unambiguous central
    # pairs, then grow the match set. This recovers stars displaced by 50+ px at
    # the wide rim without accepting arbitrary nearby hot pixels.
    predicted = ideal.copy()
    catalog_array, observed_array = mutual_pairs(predicted, 10.0)
    if len(catalog_array) < 12:
        return {"sip_refined": False, "sip_reason": "catalog matching did not converge"}
    for degree, threshold in ((2, 10.0), (2, 10.0), (3, 10.0), (3, 8.0), (4, 6.0)):
        features = polynomial_features(ideal[catalog_array], degree)
        coefficients = np.linalg.lstsq(features, observed[observed_array], rcond=None)[0]
        training_residual = np.linalg.norm(
            features @ coefficients - observed[observed_array], axis=1
        )
        median = float(np.median(training_residual))
        mad = float(np.median(np.abs(training_residual - median))) + 1e-6
        keep = training_residual <= max(2.0, median + 3.0 * 1.4826 * mad)
        if np.count_nonzero(keep) >= len(features.T):
            coefficients = np.linalg.lstsq(
                polynomial_features(ideal[catalog_array[keep]], degree),
                observed[observed_array[keep]], rcond=None,
            )[0]
        predicted = polynomial_features(ideal, degree) @ coefficients
        expanded_catalog, expanded_observed = mutual_pairs(predicted, threshold)
        if len(expanded_catalog) >= len(catalog_array):
            catalog_array, observed_array = expanded_catalog, expanded_observed

    if len(catalog_array) < 18:
        return {"sip_refined": False, "sip_reason": "too few verified wide-field pairs"}
    # A SIP5 model has enough freedom to fit sparse/noisy matches perfectly in
    # the centre while diverging at the image rim.  Use higher orders only
    # when the verified matches support them across the field.
    match_count = len(catalog_array)
    if match_count >= 80:
        sip_degree = 5
    elif match_count >= 45:
        sip_degree = 4
    else:
        sip_degree = 3
    current = fit_wcs_from_points(
        (observed[observed_array, 0] + 1.0, observed[observed_array, 1] + 1.0),
        sky[catalog_array], projection=current, sip_degree=sip_degree,
    )
    fit_x, fit_y = current.world_to_pixel(sky[catalog_array])
    residuals = np.hypot(
        fit_x - observed[observed_array, 0], fit_y - observed[observed_array, 1]
    )
    header = current.to_header(relax=True)
    hdu = fits.PrimaryHDU()
    for card in header.cards:
        hdu.header.append(card, end=True)
    hdu.header["IMAGEW"] = width
    hdu.header["IMAGEH"] = height
    hdu.header["DATE-OBS"] = date_obs.isoformat()
    hdu.header["CALTYPE"] = ("LOCAL-SIP", "Local wide-angle calibration")
    hdu.header.add_history("SIP fitted from mutual image/catalog star matches")
    hdu.writeto(paths["wcs"], overwrite=True, output_verify="silentfix")
    validation = cv2.imread(str(paths["reference"]), cv2.IMREAD_GRAYSCALE)
    if validation is not None:
        validation = cv2.cvtColor(validation, cv2.COLOR_GRAY2BGR)
        for px, py, ox, oy in zip(
            fit_x, fit_y, observed[observed_array, 0], observed[observed_array, 1]
        ):
            predicted_point = (round(float(px)), round(float(py)))
            observed_point = (round(float(ox)), round(float(oy)))
            cv2.line(validation, observed_point, predicted_point, (255, 120, 255), 1, cv2.LINE_AA)
            cv2.circle(validation, observed_point, 4, (80, 255, 80), 1, cv2.LINE_AA)
            cv2.drawMarker(
                validation, predicted_point, (255, 80, 255), cv2.MARKER_CROSS, 7, 1, cv2.LINE_AA
            )
        cv2.putText(
            validation,
            f"green=observed magenta=SIP{sip_degree} predicted  median={np.median(residuals):.2f}px p95={np.percentile(residuals, 95):.2f}px",
            (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.imwrite(str(paths["validation"]), validation, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return {
        "sip_refined": True,
        "sip_order": int(current.sip.a_order),
        "sip_match_count": int(len(catalog_array)),
        "sip_residual_median_px": float(np.median(residuals)),
        "sip_residual_p95_px": float(np.percentile(residuals, 95)),
        "sip_support_hull": [
            [float(x), float(y)]
            for x, y in cv2.convexHull(
                observed[observed_array].astype(np.float32)
            ).reshape(-1, 2)
        ],
        "validation_path": str(paths["validation"]),
    }


def _solve_samples(
    average: np.ndarray,
    samples: Optional[np.ndarray],
    source_path: str,
    date_obs: datetime,
    cache_root: Optional[str],
    progress_callback: Optional[Callable],
    reference_key: Optional[str] = None,
    reference_frame_index: Optional[int] = None,
) -> Dict[str, Any]:
    # All star-extraction operations require one channel.  The selected-frame
    # path supplies a BGR temporal-mean frame while the legacy paths already
    # use grayscale, so normalize both here.
    if average.ndim == 3:
        average = cv2.cvtColor(average, cv2.COLOR_BGR2GRAY)
    if average.ndim != 2:
        raise ValueError("較正用画像はグレースケールまたはBGR画像である必要があります")
    if samples is not None and samples.ndim == 4:
        samples = np.stack(
            [cv2.cvtColor(item, cv2.COLOR_BGR2GRAY) for item in samples]
        )
    height, width = average.shape[:2]
    paths = _calibration_paths(source_path, width, height, cache_root, reference_key)
    stars, diagnostic, reference = _extract_stars(
        average, samples, maximum_stars=180
    )
    fit_stars, _fit_diagnostic, _fit_reference = _extract_stars(
        average, samples, maximum_stars=500
    )
    cv2.imwrite(str(paths["diagnostic"]), diagnostic, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(paths["reference"]), reference, [cv2.IMWRITE_JPEG_QUALITY, 92])
    solved = _solve_stars(stars, width, height, date_obs, paths, progress_callback)
    sip_stats = _refine_sip_wcs(
        paths, fit_stars, solved.get("catalog_stars", []), width, height, date_obs
    )
    if not sip_stats.get("sip_refined"):
        raise RuntimeError(
            f"広角歪み較正が収束しません: {sip_stats.get('sip_reason', 'unknown')}"
        )
    solved.update(sip_stats)
    payload = _persist_calibration(
        source_path, date_obs, width, height, paths, solved, len(stars), reference_frame_index
    )
    payload["detected_star_count"] = int(len(stars))
    payload["fit_star_count"] = int(len(fit_stars))
    _emit(
        progress_callback,
        f"ローカル広角較正成功: SIP{payload.get('sip_order', '?')} / HFOV約"
        f"{payload.get('scale_arcsec_per_pixel', 0) * width / 3600:.1f}°",
    )
    return {
        "wcs_file": str(paths["wcs"]),
        "calibration_path": str(paths["metadata"]),
        "plate_solve_datetime": date_obs,
        "job_id": "local-wideangle-sip-adaptive",
        **payload,
    }


def solve_video_local(
    video_path: str,
    *,
    cache_root: Optional[str] = None,
    force: bool = False,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    probe = _open_video(video_path)
    if not probe.isOpened():
        raise IOError(f"動画を開けません: {video_path}")
    try:
        width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        probe.release()
    if width <= 0 or height <= 0:
        raise IOError(f"動画の解像度を読めません: {video_path}")
    if not force:
        camera_model = _registered_camera_model(video_path, width, height, cache_root)
        if camera_model is not None:
            model_label = str(camera_model.get("model_label") or "登録済み補正データ")
            support = camera_model.get("support_fraction", camera_model.get("sip_support_fraction"))
            try:
                support_text = f" / 被覆率 {float(support) * 100:.0f}%" if support is not None else ""
            except (TypeError, ValueError):
                support_text = ""
            if camera_model.get("_model_date_match", True):
                _emit(
                    progress_callback,
                    f"このカメラ専用の広角補正データを再利用します: {model_label}{support_text}",
                )
            else:
                _emit(
                    progress_callback,
                    f"同じカメラ・解像度の登録済み補正データを再利用します: {model_label}{support_text} "
                    "（基準夜が異なるため、新規星図照合は行いません）",
                )
            return camera_model
    paths = _calibration_paths(video_path, width, height, cache_root)
    if not force and paths["metadata"].exists() and paths["wcs"].exists():
        payload = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        if payload.get("algorithm_version") == ALGORITHM_VERSION:
            reference = datetime.fromisoformat(payload["reference_datetime"])
            return {
                "wcs_file": str(paths["wcs"]),
                "calibration_path": str(paths["metadata"]),
                "plate_solve_datetime": reference,
                "job_id": "local-wideangle-cache",
                **payload,
            }
    average, samples = _video_sample_stack(video_path)
    return _solve_samples(
        average, samples, video_path, _capture_datetime(video_path), cache_root, progress_callback
    )


def solve_video_frame_local(
    video_path: str,
    frame_index: int,
    *,
    cache_root: Optional[str] = None,
    force: bool = False,
    progress_callback: Optional[Callable] = None,
    stack_half_window_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Create or reuse a calibration from one user-selected video frame."""
    if stack_half_window_seconds is not None:
        return solve_video_frame_stack_local(
            video_path,
            frame_index,
            half_window_seconds=stack_half_window_seconds,
            cache_root=cache_root,
            force=force,
            progress_callback=progress_callback,
        )
    frame_index = max(0, int(frame_index))
    cap = _open_video(video_path)
    if not cap.isOpened():
        raise IOError(f"動画を開けません: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    if not ok or frame is None:
        raise IOError(f"基準フレームを読めません: {video_path} #{frame_index}")
    height, width = frame.shape[:2]
    source_digest = hashlib.sha256(
        str(Path(video_path).expanduser().resolve()).encode("utf-8")
    ).hexdigest()[:8]
    reference_key = f"frame_{frame_index}_{source_digest}"
    paths = _calibration_paths(video_path, width, height, cache_root, reference_key)
    if not force and paths["metadata"].exists() and paths["wcs"].exists():
        payload = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        if payload.get("algorithm_version") == ALGORITHM_VERSION:
            reference = datetime.fromisoformat(payload["reference_datetime"])
            return {
                "wcs_file": str(paths["wcs"]),
                "calibration_path": str(paths["metadata"]),
                "plate_solve_datetime": reference,
                "job_id": "local-wideangle-cache",
                **payload,
            }
    timestamp = _capture_datetime(video_path)
    if 0.1 <= fps <= 240.0:
        timestamp += timedelta(seconds=frame_index / fps)
    return _solve_samples(
        frame,
        None,
        video_path,
        timestamp,
        cache_root,
        progress_callback,
        reference_key=reference_key,
        reference_frame_index=frame_index,
    )


def solve_video_frame_stack_local(
    video_path: str,
    frame_index: int,
    *,
    half_window_seconds: float = 10.0,
    cache_root: Optional[str] = None,
    force: bool = False,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Solve WCS from a centered temporal stack around a selected frame."""
    frame_index = max(0, int(frame_index))
    half_window_seconds = float(half_window_seconds)
    if half_window_seconds < 0.0:
        raise ValueError("half_window_seconds must be non-negative")

    probe = _open_video(video_path)
    if not probe.isOpened():
        raise IOError(f"動画を開けません: {video_path}")
    try:
        width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        probe.release()
    if width <= 0 or height <= 0:
        raise IOError(f"動画の解像度を読めません: {video_path}")

    source_digest = hashlib.sha256(
        str(Path(video_path).expanduser().resolve()).encode("utf-8")
    ).hexdigest()[:8]
    window_label = str(int(round(half_window_seconds))).replace("-", "m")
    reference_key = f"stack_pm{window_label}s_frame_{frame_index}_{source_digest}"
    paths = _calibration_paths(video_path, width, height, cache_root, reference_key)
    if not force and paths["metadata"].exists() and paths["wcs"].exists():
        payload = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        if payload.get("algorithm_version") == ALGORITHM_VERSION:
            reference = datetime.fromisoformat(payload["reference_datetime"])
            return {
                "wcs_file": str(paths["wcs"]),
                "calibration_path": str(paths["metadata"]),
                "plate_solve_datetime": reference,
                "job_id": "local-wideangle-stack-cache",
                **payload,
            }

    average, samples, stack_info = _video_centered_stack(
        video_path, frame_index, half_window_seconds=half_window_seconds
    )
    timestamp = _capture_datetime(video_path)
    fps = float(stack_info.get("stack_fps") or 0.0)
    if 0.1 <= fps <= 240.0:
        timestamp += timedelta(seconds=int(stack_info["reference_frame_index"]) / fps)
    _emit(
        progress_callback,
        f"基準フレーム前後±{half_window_seconds:g}秒をスタックしてWCSを計算します "
        f"（{stack_info['stack_frame_count']}フレーム）",
    )
    result = _solve_samples(
        average,
        # For a short centered window, the temporal median still contains
        # the slowly moving stars and would subtract much of the S/N gained
        # by the mean stack.  Use the stack directly for WCS extraction.
        None,
        video_path,
        timestamp,
        cache_root,
        progress_callback,
        reference_key=reference_key,
        reference_frame_index=int(stack_info["reference_frame_index"]),
    )
    result.update(stack_info)
    result["stack_sample_count"] = int(len(samples))
    metadata_path = Path(result["calibration_path"])
    try:
        persisted = json.loads(metadata_path.read_text(encoding="utf-8"))
        persisted.update({
            key: value for key, value in result.items()
            if key.startswith("stack_") or key in {
                "reference_frame_index", "detected_star_count", "fit_star_count"
            }
        })
        temporary = metadata_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(persisted, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, metadata_path)
        result.update(persisted)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    return result


def solve_reference_frame_local(
    frame: np.ndarray,
    *,
    source_identity: str,
    observation_datetime: datetime,
    reference_frame_index: int,
    cache_root: Optional[str] = None,
    force: bool = False,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Solve one already-averaged frame selected from timelapse output."""
    height, width = frame.shape[:2]
    source_digest = hashlib.sha256(
        str(Path(source_identity).expanduser().resolve()).encode("utf-8")
    ).hexdigest()[:8]
    reference_key = f"sample_{int(reference_frame_index)}_{source_digest}"
    paths = _calibration_paths(source_identity, width, height, cache_root, reference_key)
    if not force and paths["metadata"].exists() and paths["wcs"].exists():
        payload = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        if payload.get("algorithm_version") == ALGORITHM_VERSION:
            return {"calibration_path": str(paths["metadata"]), **payload}
    return _solve_samples(
        frame,
        None,
        source_identity,
        observation_datetime,
        cache_root,
        progress_callback,
        reference_key=reference_key,
        reference_frame_index=reference_frame_index,
    )


def solve_frames_local(
    frames: Sequence[np.ndarray],
    *,
    source_identity: str,
    observation_datetime: Optional[datetime] = None,
    cache_root: Optional[str] = None,
    force: bool = False,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    average, samples = _frames_sample_stack(frames)
    height, width = average.shape[:2]
    paths = _calibration_paths(source_identity, width, height, cache_root)
    if not force and paths["metadata"].exists() and paths["wcs"].exists():
        payload = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        reference = datetime.fromisoformat(payload["reference_datetime"])
        return {"wcs_file": str(paths["wcs"]), "calibration_path": str(paths["metadata"]),
                "plate_solve_datetime": reference, "job_id": "local-wideangle-cache", **payload}
    date_obs = observation_datetime or datetime.now()
    return _solve_samples(
        average, samples, source_identity, date_obs, cache_root, progress_callback
    )


def solve_image_local(
    image_path: str,
    *,
    source_path: Optional[str] = None,
    cache_root: Optional[str] = None,
    force: bool = False,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise IOError(f"画像を読み込めません: {image_path}")
    identity = source_path or image_path
    if not force:
        camera_model = _registered_camera_model(
            identity, image.shape[1], image.shape[0], cache_root
        )
        if camera_model is not None:
            _emit(progress_callback, "このカメラ専用の広角モデルを再利用します")
            return camera_model
    return _solve_samples(
        image, None, identity, _capture_datetime(identity), cache_root, progress_callback
    )


def get_or_create_night_calibration(
    source_path: str,
    *,
    calibration_path: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
) -> str:
    if calibration_path:
        resolved = Path(calibration_path).expanduser().resolve()
        if resolved.exists():
            return str(resolved)
        raise CalibrationNotFoundError(f"較正ファイルがありません: {resolved}")
    result = solve_video_local(source_path, progress_callback=progress_callback)
    return str(result["calibration_path"])


def _resolve_calibration_path(calibration_path: Optional[str]) -> Path:
    value = calibration_path or os.environ.get("METEOR_WIDEANGLE_CALIBRATION")
    if not value:
        raise CalibrationNotFoundError(
            "当晩の広角較正がまだありません。先に動画からローカル較正を実行してください。"
        )
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise CalibrationNotFoundError(f"較正ファイルがありません: {path}")
    return path


def _load_calibration(calibration_path: Optional[str]) -> Tuple[Dict[str, Any], Any]:
    path = _resolve_calibration_path(calibration_path)
    stamp = path.stat().st_mtime
    key = str(path)
    with _cache_lock:
        cached = _loaded_calibrations.get(key)
        if cached and cached[0] == stamp:
            return cached[1], cached[2]
        if path.suffix.lower() in {".wcs", ".fits", ".fit"}:
            wcs_path = path
            metadata = {"wcs_path": str(path)}
            # A bare WCS does not contain the convex hull of verified detector
            # matches.  Recover the sibling metadata when possible so callers
            # cannot accidentally extrapolate a wide-angle SIP polynomial over
            # uncalibrated image edges.
            candidates: List[Path] = []
            if path.stem.startswith("wideangle_sip"):
                suffix = path.stem[len("wideangle_sip"):]
                candidates.append(path.with_name(f"calibration{suffix}.json"))
            candidates.extend(sorted(path.parent.glob("calibration*.json")))
            for candidate in dict.fromkeys(candidates):
                if not candidate.exists():
                    continue
                try:
                    candidate_metadata = json.loads(candidate.read_text(encoding="utf-8"))
                    candidate_wcs = Path(candidate_metadata.get("wcs_path", "")).expanduser()
                    if not candidate_wcs.is_absolute():
                        candidate_wcs = candidate.parent / candidate_wcs
                    if candidate_wcs.resolve() == path.resolve():
                        metadata = candidate_metadata
                        metadata.setdefault("calibration_path", str(candidate))
                        break
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    continue
        else:
            metadata = json.loads(path.read_text(encoding="utf-8"))
            if metadata.get("model_type") == CAMERA_MODEL_TYPE:
                # Fixed-camera models created before the catalog-star handoff
                # contain the fitted projection but not the catalog used to
                # establish it.  Recover that auditable catalog from the
                # sibling calibration JSON so per-frame alignment can still
                # correct the small residual left by temporal averaging.
                model_wcs = Path(str(metadata.get("wcs_path", ""))).expanduser()
                if not model_wcs.is_absolute():
                    model_wcs = path.parent / model_wcs
                candidates: List[Path] = []
                if model_wcs.name.startswith("wideangle_sip"):
                    suffix = model_wcs.stem[len("wideangle_sip"):]
                    candidates.append(model_wcs.with_name(f"calibration{suffix}.json"))
                candidates.extend(sorted(model_wcs.parent.glob("calibration*.json")))
                for candidate in dict.fromkeys(candidates):
                    if not candidate.exists():
                        continue
                    try:
                        candidate_metadata = json.loads(candidate.read_text(encoding="utf-8"))
                        candidate_wcs = Path(str(candidate_metadata.get("wcs_path", ""))).expanduser()
                        if not candidate_wcs.is_absolute():
                            candidate_wcs = candidate.parent / candidate_wcs
                        if candidate_wcs.resolve() != model_wcs.resolve():
                            continue
                        for key in (
                            "catalog_stars", "sip_residual_median_px", "sip_residual_p95_px",
                            "sip_match_count", "center_ra_deg", "center_dec_deg",
                        ):
                            if key not in metadata and key in candidate_metadata:
                                metadata[key] = candidate_metadata[key]
                        break
                    except (OSError, ValueError, TypeError, json.JSONDecodeError):
                        continue
                # The model's reference epoch is part of its fitted camera
                # orientation.  Do not replace it with the WCS DATE-OBS here:
                # validated fixed-camera models may intentionally encode a
                # different epoch after applying a sidereal rotation.
                model = FixedCameraPlateModel(metadata)
                metadata.setdefault("calibration_path", str(path))
                _loaded_calibrations[key] = (stamp, metadata, model)
                return metadata, model
            wcs_path = Path(metadata["wcs_path"]).expanduser()
            if not wcs_path.is_absolute():
                wcs_path = path.parent / wcs_path
        with fits.open(wcs_path) as hdul:
            header = hdul[0].header.copy()
        wcs = WCS(header, relax=True, fix=False)
        if not wcs.is_celestial:
            raise ValueError(f"有効な天球WCSではありません: {wcs_path}")
        metadata.setdefault("reference_datetime", header.get("DATE-OBS"))
        metadata.setdefault("width", int(header.get("IMAGEW", 1920)))
        metadata.setdefault("height", int(header.get("IMAGEH", 1080)))
        if wcs.sip is not None:
            metadata.setdefault("sip_order", int(wcs.sip.a_order))
        _loaded_calibrations[key] = (stamp, metadata, wcs)
        return metadata, wcs


def _bridge_support_grid_for_display(values: np.ndarray) -> np.ndarray:
    """Fill only small interior holes used to clip visual grid contours.

    The original support grid remains authoritative for constellation safety
    and astrometric validation.  This display copy joins a one-cell gap only
    when validated cells surround it, so the outer unvalidated boundary is
    never grown merely to make the overlay look continuous.
    """
    strict = (np.asarray(values) > 0).astype(np.uint8)
    if strict.ndim != 2 or min(strict.shape) < 3:
        return strict
    display = strict.copy()
    source = strict.copy()
    rows, columns = source.shape
    for row in range(1, rows - 1):
        for column in range(1, columns - 1):
            if source[row, column]:
                continue
            neighborhood = source[row - 1:row + 2, column - 1:column + 2]
            neighbors = int(np.sum(neighborhood))
            horizontal = bool(source[row, column - 1] and source[row, column + 1])
            vertical = bool(source[row - 1, column] and source[row + 1, column])
            if neighbors >= 5 or ((horizontal or vertical) and neighbors >= 3):
                display[row, column] = 1
    return display


def _add_short_polyline_bridges(
    visibility_mask: np.ndarray,
    line: np.ndarray,
    support_mask: np.ndarray,
    maximum_gap_px: float,
    thickness: int = 5,
) -> None:
    """Expose a contour only across short unsupported runs bounded at both ends."""
    vertices = np.asarray(line, dtype=float).reshape(-1, 2)
    if len(vertices) < 2 or maximum_gap_px <= 0:
        return
    samples: list[np.ndarray] = [vertices[0]]
    distances: list[float] = [0.0]
    distance = 0.0
    for start, end in zip(vertices[:-1], vertices[1:]):
        length = float(np.linalg.norm(end - start))
        count = max(1, int(np.ceil(length / 2.0)))
        for index in range(1, count + 1):
            point = start + (end - start) * (index / count)
            previous = samples[-1]
            distance += float(np.linalg.norm(point - previous))
            samples.append(point)
            distances.append(distance)
    sampled = np.asarray(samples, dtype=float)
    rounded = np.rint(sampled).astype(int)
    height, width = support_mask.shape[:2]
    inside = (
        (rounded[:, 0] >= 0) & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0) & (rounded[:, 1] < height)
    )
    supported = np.zeros(len(rounded), dtype=bool)
    supported[inside] = support_mask[
        rounded[inside, 1], rounded[inside, 0]
    ] > 0
    index = 0
    while index < len(supported):
        if supported[index]:
            index += 1
            continue
        start = index
        while index < len(supported) and not supported[index]:
            index += 1
        end = index - 1
        left = start - 1
        right = end + 1
        if left < 0 or right >= len(supported):
            continue
        if not (supported[left] and supported[right]):
            continue
        if distances[right] - distances[left] > float(maximum_gap_px):
            continue
        bridge = np.rint(sampled[left:right + 1]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            visibility_mask, [bridge], False, 255, max(1, int(thickness)), cv2.LINE_8,
        )


def _forward_grid_model(wcs: WCS, metadata: Dict[str, Any], width: int, height: int):
    """Cache stable pixel->sky contour generators (no inverse SIP iteration)."""
    key = (id(wcs), width, height)
    with _cache_lock:
        cached = _forward_grid_cache.get(key)
        if cached is not None:
            return cached
    calibration_width = max(1, int(metadata.get("width", width)))
    calibration_height = max(1, int(metadata.get("height", height)))
    columns = max(65, min(161, calibration_width // 12 + 1))
    rows = max(41, min(101, calibration_height // 12 + 1))
    x_cal = np.linspace(0.0, calibration_width - 1.0, columns)
    y_cal = np.linspace(0.0, calibration_height - 1.0, rows)
    mesh_x, mesh_y = np.meshgrid(x_cal, y_cal)
    ra, dec = wcs.pixel_to_world_values(mesh_x, mesh_y)
    center_ra = float(metadata.get("center_ra_deg", np.nanmedian(ra)))
    unwrapped_ra = center_ra + ((np.asarray(ra) - center_ra + 180.0) % 360.0 - 180.0)
    x_output = x_cal * width / calibration_width
    y_output = y_cal * height / calibration_height
    support_hull = metadata.get("sip_support_hull")
    support_grid = metadata.get("support_grid")
    support_output = np.full((height, width), 255, dtype=np.uint8)
    display_support_output = support_output
    maximum_gap = 0
    range_ra = np.asarray(unwrapped_ra)
    range_dec = np.asarray(dec)
    if support_grid:
        grid_values = np.asarray(support_grid, dtype=np.uint8)
        if grid_values.ndim != 2 or not grid_values.size:
            raise ValueError("support_grid must be a non-empty 2D array")
        support = np.zeros((calibration_height, calibration_width), dtype=np.uint8)
        grid_rows, grid_columns = grid_values.shape
        for row in range(grid_rows):
            for column in range(grid_columns):
                if not grid_values[row, column]:
                    continue
                left = round(column * calibration_width / grid_columns)
                right = round((column + 1) * calibration_width / grid_columns)
                top = round(row * calibration_height / grid_rows)
                bottom = round((row + 1) * calibration_height / grid_rows)
                support[top:bottom, left:right] = 255
        sample_x = np.clip(np.rint(mesh_x).astype(int), 0, calibration_width - 1)
        sample_y = np.clip(np.rint(mesh_y).astype(int), 0, calibration_height - 1)
        unsupported = support[sample_y, sample_x] == 0
        range_ra = np.ma.array(unwrapped_ra, mask=unsupported).compressed()
        range_dec = np.ma.array(dec, mask=unsupported).compressed()
        support_output = cv2.resize(
            support, (width, height), interpolation=cv2.INTER_NEAREST
        )
        display_grid = _bridge_support_grid_for_display(grid_values)
        display_support = np.zeros((calibration_height, calibration_width), dtype=np.uint8)
        for row in range(grid_rows):
            for column in range(grid_columns):
                if not display_grid[row, column]:
                    continue
                left = round(column * calibration_width / grid_columns)
                right = round((column + 1) * calibration_width / grid_columns)
                top = round(row * calibration_height / grid_rows)
                bottom = round((row + 1) * calibration_height / grid_rows)
                display_support[top:bottom, left:right] = 255
        display_support_output = cv2.resize(
            display_support, (width, height), interpolation=cv2.INTER_NEAREST
        )
        cell_width = width / max(1, grid_columns)
        cell_height = height / max(1, grid_rows)
        # Two support cells correspond to the short visual breaks seen when a
        # contour crosses a sparse cell near the validated boundary.
        maximum_gap_cells = float(metadata.get("display_grid_max_gap_cells", 2.1))
        maximum_gap = round(max(cell_width, cell_height) * maximum_gap_cells)
    elif support_hull and len(support_hull) >= 3:
        support = np.zeros((calibration_height, calibration_width), dtype=np.uint8)
        hull = np.rint(np.asarray(support_hull, dtype=float)).astype(np.int32)
        cv2.fillConvexPoly(support, hull, 255)
        # A small margin avoids clipping the line exactly at the outer matched
        # stars while still hiding unconstrained polynomial extrapolation.
        margin = max(9, round(min(calibration_width, calibration_height) * 0.025))
        support = cv2.dilate(
            support,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin | 1, margin | 1)),
        )
        sample_x = np.clip(np.rint(mesh_x).astype(int), 0, calibration_width - 1)
        sample_y = np.clip(np.rint(mesh_y).astype(int), 0, calibration_height - 1)
        unsupported = support[sample_y, sample_x] == 0
        range_ra = np.ma.array(unwrapped_ra, mask=unsupported).compressed()
        range_dec = np.ma.array(dec, mask=unsupported).compressed()
        scaled_hull = np.rint(
            np.asarray(support_hull, dtype=float)
            * np.asarray([width / calibration_width, height / calibration_height])
        ).astype(np.int32)
        support_output.fill(0)
        cv2.fillConvexPoly(support_output, scaled_hull, 255)
        output_margin = max(7, round(min(width, height) * 0.02)) | 1
        support_output = cv2.dilate(
            support_output,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (output_margin, output_margin)),
        )
        display_support_output = support_output
    ra_values = np.asarray(range_ra).ravel()
    dec_values = np.asarray(range_dec).ravel()
    model = {
        "center_ra": center_ra,
        "ra_min": float(np.nanpercentile(ra_values, 1)),
        "ra_max": float(np.nanpercentile(ra_values, 99)),
        "dec_min": float(np.nanpercentile(dec_values, 1)),
        "dec_max": float(np.nanpercentile(dec_values, 99)),
        "support_mask": support_output,
        "display_support_mask": display_support_output,
        "display_grid_max_gap_px": maximum_gap,
        "ra_contours": contourpy.contour_generator(
            x=x_output, y=y_output, z=unwrapped_ra, corner_mask=True
        ),
        "dec_contours": contourpy.contour_generator(
            x=x_output, y=y_output, z=np.asarray(dec), corner_mask=True
        ),
    }
    with _cache_lock:
        _forward_grid_cache[key] = model
    return model


def _draw_contour_lines(
    output: np.ndarray,
    lines: Sequence[np.ndarray],
    color: Tuple[int, int, int],
    label: Optional[str] = None,
    visibility_mask: Optional[np.ndarray] = None,
    support_mask: Optional[np.ndarray] = None,
    maximum_gap_px: float = 0.0,
) -> None:
    usable = [line for line in lines if isinstance(line, np.ndarray) and len(line) >= 3]
    for line in usable:
        if visibility_mask is not None and support_mask is not None:
            _add_short_polyline_bridges(
                visibility_mask, line, support_mask, maximum_gap_px,
            )
        points = np.rint(line).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(output, [points], False, color, 1, cv2.LINE_AA)
    if label and usable:
        longest = max(usable, key=len)
        point = np.rint(longest[len(longest) // 2]).astype(int)
        height, width = output.shape[:2]
        if 5 <= point[0] < width - 70 and 16 <= point[1] < height - 5:
            cv2.putText(
                output, label, tuple(point), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (170, 240, 255), 1, cv2.LINE_AA,
            )


def _load_constellation_lines() -> Tuple[np.ndarray, ...]:
    """Load vendored J2000 constellation stick figures once per process."""
    global _constellation_lines
    with _cache_lock:
        if _constellation_lines is not None:
            return _constellation_lines
        asset = Path(__file__).with_name("assets") / "constellations.lines.json"
        try:
            payload = json.loads(asset.read_text(encoding="utf-8"))
            lines = []
            for feature in payload.get("features", []):
                geometry = feature.get("geometry", {})
                for line in geometry.get("coordinates", []):
                    points = np.asarray(line, dtype=float)
                    if points.ndim == 2 and points.shape[0] >= 2 and points.shape[1] >= 2:
                        lines.append(points[:, :2])
            _constellation_lines = tuple(lines)
        except (OSError, ValueError, TypeError):
            # Missing optional display data must never prevent grid annotation.
            _constellation_lines = ()
        return _constellation_lines


def _project_sky_with_forward_wcs(
    wcs: Any,
    ra: np.ndarray,
    dec: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Invert pixel->sky numerically, including forward-only SIP distortion.

    The local SIP fit intentionally stores the reliable forward distortion
    model.  ``WCS.world_to_pixel_values`` can use an approximate inverse SIP
    solution and visibly displace constellation endpoints in a wide-angle
    image.  It remains a useful starting estimate; Newton refinement below
    evaluates only the calibrated forward transform.
    """
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    if isinstance(wcs, FixedCameraPlateModel):
        # The fixed-camera model is itself the authoritative forward
        # projection.  Its inverse is only an approximate numerical helper
        # near the ultra-wide rim, so sending these points through the SIP
        # Newton refinement below can move them onto a different branch.
        try:
            x, y = wcs.world_to_pixel_values(ra, dec)
        except Exception:
            return np.zeros_like(ra), np.zeros_like(dec), np.zeros_like(ra, dtype=bool)
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        return x, y, np.isfinite(x) & np.isfinite(y)
    try:
        # Astropy's inverse SIP iteration is only an initial estimate here.  A
        # forward-only wide-angle fit can legitimately report NoConvergence at
        # the rim; the Newton loop below re-evaluates the calibrated forward
        # transform and decides which points are actually usable.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r".*WCS\.all_world2pix.*failed to converge.*"
            )
            x, y = wcs.world_to_pixel_values(ra, dec)
    except Exception:
        return np.zeros_like(ra), np.zeros_like(dec), np.zeros_like(ra, dtype=bool)
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    step = 0.5
    for _ in range(8):
        active = np.flatnonzero(valid)
        if not len(active):
            break
        try:
            current_ra, current_dec = wcs.pixel_to_world_values(x[active], y[active])
            x_ra, x_dec = wcs.pixel_to_world_values(x[active] + step, y[active])
            y_ra, y_dec = wcs.pixel_to_world_values(x[active], y[active] + step)
        except Exception:
            valid[active] = False
            continue
        scale = np.cos(np.deg2rad(dec[active]))
        # Work in locally tangent RA/Dec coordinates to avoid the RA=0 wrap.
        error_ra = ((np.asarray(current_ra) - ra[active] + 180.0) % 360.0 - 180.0) * scale
        error_dec = np.asarray(current_dec) - dec[active]
        j00 = ((np.asarray(x_ra) - np.asarray(current_ra) + 180.0) % 360.0 - 180.0) * scale / step
        j01 = ((np.asarray(y_ra) - np.asarray(current_ra) + 180.0) % 360.0 - 180.0) * scale / step
        j10 = (np.asarray(x_dec) - np.asarray(current_dec)) / step
        j11 = (np.asarray(y_dec) - np.asarray(current_dec)) / step
        determinant = j00 * j11 - j01 * j10
        usable = np.isfinite(determinant) & (np.abs(determinant) > 1e-10)
        failed = active[~usable]
        valid[failed] = False
        active = active[usable]
        if not len(active):
            continue
        dx = (j11[usable] * error_ra[usable] - j01[usable] * error_dec[usable]) / determinant[usable]
        dy = (-j10[usable] * error_ra[usable] + j00[usable] * error_dec[usable]) / determinant[usable]
        # Limit a pathological initial estimate rather than jumping across the
        # full image and converging to a different branch of the distortion.
        x[active] -= np.clip(dx, -80.0, 80.0)
        y[active] -= np.clip(dy, -80.0, 80.0)
        if len(dx) and float(np.max(np.maximum(np.abs(dx), np.abs(dy)))) < 0.02:
            break
    return x, y, valid & np.isfinite(x) & np.isfinite(y)


def _estimate_constellation_pixel_offset(
    detected_stars: Sequence[Sequence[float]],
    metadata: Dict[str, Any],
    wcs: WCS,
    delta_ra: float,
    support_mask: np.ndarray,
    width: int,
    height: int,
) -> Tuple[float, float]:
    """Estimate the current image-to-WCS residual from bright catalog stars.

    The SIP model is fixed for the night, but a wide-angle fit can retain a
    small, position-dependent residual.  Matching only bright catalog stars
    in the current frame lets the constellation sticks follow the actual
    stellar image without allowing arbitrary faint detections to move them.
    The result is deliberately a translation: higher-order changes belong in
    the calibration model, not in a per-frame display adjustment.
    """
    catalog = metadata.get("catalog_stars") or []
    if len(catalog) < _CONSTELLATION_ALIGNMENT_MIN_MATCHES:
        return 0.0, 0.0
    observed = np.asarray(detected_stars, dtype=float).reshape(-1, 2)
    if len(observed) < _CONSTELLATION_ALIGNMENT_MIN_MATCHES:
        return 0.0, 0.0
    observed = observed[np.isfinite(observed).all(axis=1)]
    if len(observed) < _CONSTELLATION_ALIGNMENT_MIN_MATCHES:
        return 0.0, 0.0

    selected = []
    for star in catalog:
        try:
            magnitude = float((star.get("metadata") or {}).get("MAG", np.inf))
            ra_value = float(star["ra_deg"])
            dec_value = float(star["dec_deg"])
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        if np.isfinite(magnitude) and magnitude <= _CONSTELLATION_ALIGNMENT_MAX_MAG:
            selected.append((ra_value, dec_value))
    if len(selected) < _CONSTELLATION_ALIGNMENT_MIN_MATCHES:
        return 0.0, 0.0

    sky_ra = np.asarray([(ra - delta_ra) % 360.0 for ra, _dec in selected])
    sky_dec = np.asarray([dec for _ra, dec in selected])
    predicted_x, predicted_y, valid = _project_sky_with_forward_wcs(
        wcs, sky_ra, sky_dec
    )
    predicted = np.column_stack((predicted_x, predicted_y))
    valid &= np.isfinite(predicted).all(axis=1)
    valid &= (predicted[:, 0] >= 0) & (predicted[:, 0] < width)
    valid &= (predicted[:, 1] >= 0) & (predicted[:, 1] < height)
    safe_predicted = np.nan_to_num(predicted, nan=0.0, posinf=0.0, neginf=0.0)
    rounded = np.rint(np.clip(safe_predicted, 0, [width - 1, height - 1])).astype(int)
    valid &= support_mask[rounded[:, 1], rounded[:, 0]] > 0
    if np.count_nonzero(valid) < _CONSTELLATION_ALIGNMENT_MIN_MATCHES:
        return 0.0, 0.0
    predicted = predicted[valid]

    distances = np.linalg.norm(
        predicted[:, np.newaxis, :] - observed[np.newaxis, :, :], axis=2
    )
    nearest_observed = np.argmin(distances, axis=1)
    nearest_predicted = np.argmin(distances, axis=0)
    residuals = []
    for predicted_index, observed_index in enumerate(nearest_observed):
        if nearest_predicted[observed_index] != predicted_index:
            continue
        if distances[predicted_index, observed_index] > _CONSTELLATION_ALIGNMENT_MATCH_RADIUS_PX:
            continue
        residuals.append(observed[observed_index] - predicted[predicted_index])
    residuals = np.asarray(residuals, dtype=float).reshape(-1, 2)
    if len(residuals) < _CONSTELLATION_ALIGNMENT_MIN_MATCHES:
        return 0.0, 0.0

    center = np.median(residuals, axis=0)
    for _ in range(2):
        distances_from_center = np.linalg.norm(residuals - center, axis=1)
        mad = float(np.median(np.abs(distances_from_center - np.median(distances_from_center))))
        keep = distances_from_center <= max(2.5, float(np.median(distances_from_center)) + 3.0 * 1.4826 * mad)
        if np.count_nonzero(keep) < _CONSTELLATION_ALIGNMENT_MIN_MATCHES:
            break
        center = np.median(residuals[keep], axis=0)
    center = np.clip(center, -6.0, 6.0)
    return float(center[0]), float(center[1])


def _sample_constellation_line(
    line: np.ndarray,
    max_angular_step_deg: float = _CONSTELLATION_MAX_ANGULAR_STEP_DEG,
) -> np.ndarray:
    """Densely interpolate a sky polyline along great-circle segments."""
    coordinates = np.asarray(line, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[0] < 2:
        return coordinates.reshape(-1, 2)
    ra = np.deg2rad(coordinates[:, 0])
    dec = np.deg2rad(coordinates[:, 1])
    vectors = np.column_stack((
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec),
    ))
    sampled: List[np.ndarray] = [coordinates[0]]
    step = max(float(max_angular_step_deg), 1e-3)
    for start_vector, end_vector in zip(vectors[:-1], vectors[1:]):
        omega = float(np.arccos(np.clip(np.dot(start_vector, end_vector), -1.0, 1.0)))
        count = max(1, int(np.ceil(np.rad2deg(omega) / step)))
        if omega < 1e-10:
            segment_vectors = np.repeat(start_vector[np.newaxis, :], count + 1, axis=0)
        else:
            fractions = np.linspace(0.0, 1.0, count + 1)[:, np.newaxis]
            denominator = np.sin(omega)
            if abs(denominator) < 1e-8:
                # Constellation edges are not antipodal, but this fallback
                # keeps malformed optional data from producing NaNs.
                segment_vectors = (1.0 - fractions) * start_vector + fractions * end_vector
                norms = np.linalg.norm(segment_vectors, axis=1, keepdims=True)
                segment_vectors = np.divide(
                    segment_vectors, norms, out=segment_vectors,
                    where=norms > 1e-12,
                )
            else:
                segment_vectors = (
                    np.sin((1.0 - fractions) * omega) / denominator * start_vector
                    + np.sin(fractions * omega) / denominator * end_vector
                )
        segment_ra = np.rad2deg(np.arctan2(segment_vectors[:, 1], segment_vectors[:, 0])) % 360.0
        segment_dec = np.rad2deg(np.arcsin(np.clip(segment_vectors[:, 2], -1.0, 1.0)))
        sampled.extend(np.column_stack((segment_ra, segment_dec))[1:])
    return np.asarray(sampled, dtype=float)


def _constellation_polyline_runs(
    points: np.ndarray,
    usable: np.ndarray,
    max_pixel_step: float,
) -> Iterable[np.ndarray]:
    """Yield only continuous projected runs that are safe to draw."""
    points = np.asarray(points, dtype=float)
    usable = np.asarray(usable, dtype=bool)
    if len(points) < 2:
        return
    gaps = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
    start = 0
    while start < len(points) - 1:
        while start < len(points) - 1 and not (
            usable[start] and usable[start + 1] and gaps[start] <= max_pixel_step
        ):
            start += 1
        if start >= len(points) - 1:
            break
        end = start + 1
        while end < len(points) and usable[end] and gaps[end - 1] <= max_pixel_step:
            end += 1
        if end - start >= 2:
            yield points[start:end]
        start = end


def _project_constellation_samples(
    wcs: Any,
    ra: np.ndarray,
    dec: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project dense safety samples without broadening the endpoint mask."""
    if isinstance(wcs, FixedCameraPlateModel):
        try:
            x, y = wcs.world_to_pixel_values(ra, dec)
        except Exception:
            return np.zeros_like(ra), np.zeros_like(dec), np.zeros_like(ra, dtype=bool)
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        usable = np.isfinite(x) & np.isfinite(y)
        # Validate the direction that matters for an overlay: requested sky
        # coordinate -> detector -> recovered sky coordinate.  Pixel->sky->
        # pixel alone cannot detect a polynomial fold because it starts on a
        # single already-selected detector branch.
        indices = np.flatnonzero(usable)
        if len(indices):
            try:
                inverse_ra, inverse_dec = wcs.pixel_to_world_values(
                    x[indices], y[indices]
                )
                target_ra = np.deg2rad(np.asarray(ra, dtype=float)[indices])
                target_dec = np.deg2rad(np.asarray(dec, dtype=float)[indices])
                recovered_ra = np.deg2rad(np.asarray(inverse_ra, dtype=float))
                recovered_dec = np.deg2rad(np.asarray(inverse_dec, dtype=float))
                cosine = (
                    np.sin(target_dec) * np.sin(recovered_dec)
                    + np.cos(target_dec) * np.cos(recovered_dec)
                    * np.cos(recovered_ra - target_ra)
                )
                roundtrip_error = np.rad2deg(
                    np.arccos(np.clip(cosine, -1.0, 1.0))
                )
                usable[indices] &= (
                    np.isfinite(roundtrip_error)
                    & (roundtrip_error <= _CONSTELLATION_MAX_SKY_ROUNDTRIP_DEG)
                )
            except Exception:
                usable[indices] = False
        return x, y, usable
    return _project_sky_with_forward_wcs(wcs, ra, dec)


def _constellation_segment_is_safe(
    wcs: Any,
    sky_segment: np.ndarray,
    delta_ra: float,
    support_mask: np.ndarray,
    width: int,
    height: int,
    max_pixel_step: float,
    pixel_offset: Tuple[float, float] = (0.0, 0.0),
) -> bool:
    """Check a segment's interior without drawing extra sampled strokes."""
    sampled = _sample_constellation_line(sky_segment)
    segment_ra = (sampled[:, 0] - delta_ra) % 360.0
    x, y, usable = _project_constellation_samples(wcs, segment_ra, sampled[:, 1])
    points = np.column_stack((x, y))
    points += np.asarray(pixel_offset, dtype=float)
    usable &= np.isfinite(points).all(axis=1)
    usable &= (points[:, 0] >= 0) & (points[:, 0] < width)
    usable &= (points[:, 1] >= 0) & (points[:, 1] < height)
    rounded = np.zeros(points.shape, dtype=np.int32)
    indices = np.flatnonzero(usable)
    if len(indices):
        rounded[indices] = np.rint(points[indices]).astype(np.int32)
    usable &= (
        (rounded[:, 0] >= 0) & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0) & (rounded[:, 1] < height)
    )
    indices = np.flatnonzero(usable)
    if len(indices):
        usable[indices] = support_mask[
            rounded[indices, 1], rounded[indices, 0]
        ] > 0
    if not np.all(usable):
        return False
    if len(points) > 1:
        gaps = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
        if np.any(~np.isfinite(gaps)) or np.any(gaps > max_pixel_step):
            return False
    return True


def _draw_constellation_lines(
    output: np.ndarray,
    wcs: Any,
    delta_ra: float,
    support_mask: np.ndarray,
    anchor_points: Optional[np.ndarray] = None,
    anchor_tolerance_px: float = 5.0,
    pixel_offset: Tuple[float, float] = (0.0, 0.0),
    allow_partial_segments: bool = False,
    edge_admission: Optional[np.ndarray] = None,
) -> None:
    """Project visible constellation line segments into the current frame."""
    height, width = output.shape[:2]
    # Keep the constellation sticks visually distinct from the yellow sky
    # grid.  A dark halo also keeps them readable over clouds and bright stars.
    color = (255, 150, 60)
    lines = _load_constellation_lines()
    if not lines:
        return
    # A constellation edge is a connection between two catalog stars.  Keep
    # endpoint projection separate from the visibility test: a star can be
    # outside the frame while the connecting edge is still partly visible.
    lengths = [len(line) for line in lines]
    coordinates = np.concatenate(lines, axis=0)
    ra = (coordinates[:, 0] - delta_ra) % 360.0
    x, y, usable = _project_sky_with_forward_wcs(wcs, ra, coordinates[:, 1])
    vertex_points = np.column_stack((x, y))
    pixel_shift = np.asarray(pixel_offset, dtype=float)
    if pixel_shift.shape != (2,) or not np.isfinite(pixel_shift).all():
        pixel_shift = np.zeros(2, dtype=float)
    draw_points = vertex_points + pixel_shift
    usable &= np.isfinite(vertex_points).all(axis=1)
    usable &= np.isfinite(draw_points).all(axis=1)
    endpoint_visible = usable.copy()
    endpoint_visible &= (
        (draw_points[:, 0] >= 0) & (draw_points[:, 0] < width)
        & (draw_points[:, 1] >= 0) & (draw_points[:, 1] < height)
    )
    rounded = np.zeros(draw_points.shape, dtype=np.int32)
    visible_indices = np.flatnonzero(endpoint_visible)
    if len(visible_indices):
        rounded[visible_indices] = np.rint(
            draw_points[visible_indices]
        ).astype(np.int32)
    # A coordinate such as y=1079.6 is mathematically inside a 1080px frame,
    # but rounds to 1080 and must not index the support mask.
    endpoint_visible &= (
        (rounded[:, 0] >= 0) & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0) & (rounded[:, 1] < height)
    )
    visible_indices = np.flatnonzero(endpoint_visible)
    if len(visible_indices):
        endpoint_visible[visible_indices] = support_mask[
            rounded[visible_indices, 1], rounded[visible_indices, 0]
        ] > 0
    # Keep finite projected endpoints for optional anchor matching.  A line
    # is admitted only when both catalog endpoints are in the calibrated
    # image area; otherwise a curved projection can create a floating fragment
    # from an unrelated constellation outside the camera's support.
    anchor_usable = usable.copy()
    if anchor_points is not None and len(anchor_points):
        anchors = np.asarray(anchor_points, dtype=float).reshape(-1, 2)
        finite_anchors = anchors[np.isfinite(anchors).all(axis=1)]
        if len(finite_anchors):
            distances = np.linalg.norm(
                draw_points[:, np.newaxis, :] - finite_anchors[np.newaxis, :, :], axis=2
            )
            anchor_usable &= np.min(distances, axis=1) <= float(anchor_tolerance_px)
        else:
            anchor_usable[:] = False
    temporal_edge_admission = None
    if edge_admission is not None:
        temporal_edge_admission = np.asarray(edge_admission, dtype=bool).reshape(-1)
    max_pixel_step = max(
        96.0, min(width, height) * _CONSTELLATION_MAX_PIXEL_STEP_FACTOR
    )
    offset = 0
    segment_offset = 0
    thickness = max(1, round(min(width, height) / 720.0))
    for line, length in zip(lines, lengths):
        for index in range(length - 1):
            current_segment_offset = segment_offset
            segment_offset += 1
            # Legacy/single-frame calibrations still require both endpoint
            # stars to be visible and detected.  A trajectory+Gaia model has
            # already established a stable detector-to-sky projection across
            # time, so it may safely draw the supported part of an edge even
            # when one endpoint falls outside the mask.  This prevents valid
            # in-frame stars from losing their connecting line merely because
            # the other endpoint is just outside a support-cell boundary.
            current_endpoints_admitted = (
                endpoint_visible[offset + index]
                and endpoint_visible[offset + index + 1]
                and anchor_usable[offset + index]
                and anchor_usable[offset + index + 1]
            )
            held_edge_admitted = bool(
                temporal_edge_admission is not None
                and current_segment_offset < len(temporal_edge_admission)
                and temporal_edge_admission[current_segment_offset]
                and endpoint_visible[offset + index]
                and endpoint_visible[offset + index + 1]
            )
            endpoints_admitted = current_endpoints_admitted or held_edge_admitted
            if not allow_partial_segments and not endpoints_admitted:
                continue
            if allow_partial_segments and anchor_points is not None and not endpoints_admitted:
                continue
            sampled = _sample_constellation_line(line[index:index + 2])
            segment_ra = (sampled[:, 0] - delta_ra) % 360.0
            sample_x, sample_y, sample_usable = _project_constellation_samples(
                wcs, segment_ra, sampled[:, 1]
            )
            sample_points = np.column_stack((sample_x, sample_y))
            sample_points += pixel_shift
            sample_usable &= np.isfinite(sample_points).all(axis=1)
            sample_usable &= (
                (sample_points[:, 0] >= 0) & (sample_points[:, 0] < width)
                & (sample_points[:, 1] >= 0) & (sample_points[:, 1] < height)
            )
            rounded_samples = np.zeros(sample_points.shape, dtype=np.int32)
            visible_indices = np.flatnonzero(sample_usable)
            if len(visible_indices):
                rounded_samples[visible_indices] = np.rint(
                    sample_points[visible_indices]
                ).astype(np.int32)
            # A coordinate such as y=1079.6 is mathematically inside a
            # 1080px frame, but rounds to 1080 and must not index the mask.
            sample_usable &= (
                (rounded_samples[:, 0] >= 0) & (rounded_samples[:, 0] < width)
                & (rounded_samples[:, 1] >= 0) & (rounded_samples[:, 1] < height)
            )
            visible_indices = np.flatnonzero(sample_usable)
            if len(visible_indices):
                sample_usable[visible_indices] = support_mask[
                    rounded_samples[visible_indices, 1],
                    rounded_samples[visible_indices, 0],
                ] > 0
            runs = _constellation_polyline_runs(
                sample_points, sample_usable, max_pixel_step
            )
            for run in runs:
                polyline = np.rint(run).astype(np.int32).reshape(-1, 1, 2)
                if len(polyline) < 2:
                    continue
                cv2.polylines(
                    output, [polyline], False, (0, 0, 0), thickness + 2, cv2.LINE_AA,
                )
                cv2.polylines(
                    output, [polyline], False, color, thickness, cv2.LINE_AA,
                )
        offset += length


def _constellation_segment_detection_mask(
    wcs: Any,
    delta_ra: float,
    support_mask: np.ndarray,
    anchor_points: Optional[np.ndarray],
    anchor_tolerance_px: float = 5.0,
    pixel_offset: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Return current-frame endpoint detection status for every constellation edge."""
    lines = _load_constellation_lines()
    segment_count = sum(max(0, len(line) - 1) for line in lines)
    admitted = np.zeros(segment_count, dtype=bool)
    if not lines or anchor_points is None:
        return admitted
    anchors = np.asarray(anchor_points, dtype=float).reshape(-1, 2)
    finite_anchors = anchors[np.isfinite(anchors).all(axis=1)]
    if not len(finite_anchors):
        return admitted
    coordinates = np.concatenate(lines, axis=0)
    ra = (coordinates[:, 0] - delta_ra) % 360.0
    x, y, usable = _project_sky_with_forward_wcs(wcs, ra, coordinates[:, 1])
    vertex_points = np.column_stack((x, y))
    pixel_shift = np.asarray(pixel_offset, dtype=float)
    if pixel_shift.shape != (2,) or not np.isfinite(pixel_shift).all():
        pixel_shift = np.zeros(2, dtype=float)
    draw_points = vertex_points + pixel_shift
    usable &= np.isfinite(draw_points).all(axis=1)
    height, width = support_mask.shape[:2]
    endpoint_visible = usable.copy()
    endpoint_visible &= (
        (draw_points[:, 0] >= 0) & (draw_points[:, 0] < width)
        & (draw_points[:, 1] >= 0) & (draw_points[:, 1] < height)
    )
    rounded = np.zeros(draw_points.shape, dtype=np.int32)
    visible_indices = np.flatnonzero(endpoint_visible)
    if len(visible_indices):
        rounded[visible_indices] = np.rint(draw_points[visible_indices]).astype(np.int32)
    endpoint_visible &= (
        (rounded[:, 0] >= 0) & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0) & (rounded[:, 1] < height)
    )
    visible_indices = np.flatnonzero(endpoint_visible)
    if len(visible_indices):
        endpoint_visible[visible_indices] = support_mask[
            rounded[visible_indices, 1], rounded[visible_indices, 0]
        ] > 0
    distances = np.linalg.norm(
        draw_points[:, np.newaxis, :] - finite_anchors[np.newaxis, :, :], axis=2
    )
    anchor_usable = usable & (
        np.min(distances, axis=1) <= float(anchor_tolerance_px)
    )
    offset = 0
    segment_offset = 0
    for line in lines:
        for index in range(len(line) - 1):
            admitted[segment_offset] = bool(
                endpoint_visible[offset + index]
                and endpoint_visible[offset + index + 1]
                and anchor_usable[offset + index]
                and anchor_usable[offset + index + 1]
            )
            segment_offset += 1
        offset += len(line)
    return admitted


def _bridge_boolean_gaps(values: np.ndarray, max_gap_frames: int) -> np.ndarray:
    """Fill only bounded false runs no longer than ``max_gap_frames``."""
    result = np.asarray(values, dtype=bool).copy()
    if result.ndim != 2 or result.size == 0 or max_gap_frames <= 0:
        return result
    for edge_index in range(result.shape[0]):
        edge = result[edge_index]
        index = 0
        while index < len(edge):
            if edge[index]:
                index += 1
                continue
            start = index
            while index < len(edge) and not edge[index]:
                index += 1
            end = index
            if (
                start > 0 and end < len(edge)
                and end - start <= max_gap_frames
                and edge[start - 1] and edge[end]
            ):
                edge[start:end] = True
    return result


def annotate_frame(
    frame_bgr: np.ndarray,
    frame_datetime: datetime,
    calibration_path: Optional[str] = None,
    draw_constellations: bool = False,
    draw_grid: bool = True,
    draw_detected_stars: bool = False,
    constellation_edge_admission: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Draw selected local, distortion-aware celestial annotations."""
    metadata, wcs = _load_calibration(calibration_path)
    output = np.ascontiguousarray(frame_bgr.copy())
    if output.ndim == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    height, width = output.shape[:2]
    reference_value = metadata.get("reference_datetime")
    reference = datetime.fromisoformat(str(reference_value).replace("Z", "+00:00"))
    target = frame_datetime
    if reference.tzinfo is not None and target.tzinfo is None:
        target = target.replace(tzinfo=reference.tzinfo)
    elif reference.tzinfo is None and target.tzinfo is not None:
        target = target.replace(tzinfo=None)
    delta_ra = ((target - reference).total_seconds() / SIDEREAL_DAY_SECONDS) * 360.0
    detected_stars: List[List[float]] = []
    # Validated fixed-camera models store this policy as
    # ``verified_constellation_only``; newer builder output calls it
    # ``constellation_anchor_filter``.  Honor both names so an older validated
    # 80%-support model does not silently fall back to drawing every catalog
    # line in the sky.
    verify_constellations = bool(
        draw_constellations and metadata.get(
            "constellation_anchor_filter",
            metadata.get("verified_constellation_only", False),
        )
    )
    constellation_render_policy = str(
        metadata.get("constellation_render_policy", "") or ""
    ).strip().lower()
    detected_endpoint_constellations = bool(
        draw_constellations
        and constellation_render_policy == "model-supported-detected-endpoints"
    )
    continuous_constellations = bool(
        draw_constellations
        and constellation_render_policy == "model-supported-continuous"
    )
    if continuous_constellations:
        # The trajectory+Gaia solution is the temporal evidence.  Requiring
        # fresh endpoint detections in every timelapse frame makes a correct
        # line blink whenever a thin cloud hides one star for a few frames.
        verify_constellations = False
    elif detected_endpoint_constellations:
        # This policy is deliberately cloud-visible: a line is drawn only if
        # both of its endpoint stars are detected in this frame.  The
        # trajectory model still supplies the sky position and branch guard;
        # detections decide whether the line is currently observable.
        verify_constellations = True
    align_constellations = bool(
        draw_constellations
        and metadata.get(
            "constellation_star_alignment",
            metadata.get("verified_constellation_only", False),
        )
        and metadata.get("catalog_stars")
    )
    if draw_detected_stars or verify_constellations or align_constellations:
        detection_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        detected_stars, _diagnostic, _reference = _extract_stars(
            detection_gray, maximum_stars=1000, exclude_lower_region=False,
            build_diagnostic=False,
        )
    grid_color = (90, 210, 255)
    grid = None
    grid_visibility = None
    grid_layer = output.copy()
    if draw_grid or draw_constellations:
        grid = _forward_grid_model(wcs, metadata, width, height)
        if grid is not None:
            grid_visibility = grid.get("display_support_mask", grid["support_mask"]).copy()
    if draw_grid and grid is not None:
        for ra_value in range(0, 360, 15):
            reference_level = (ra_value - delta_ra) % 360.0
            reference_level = grid["center_ra"] + (
                (reference_level - grid["center_ra"] + 180.0) % 360.0 - 180.0
            )
            if grid["ra_min"] <= reference_level <= grid["ra_max"]:
                _draw_contour_lines(
                    grid_layer, grid["ra_contours"].lines(reference_level), grid_color,
                    label=f"RA {ra_value // 15:02d}h",
                    visibility_mask=grid_visibility,
                    support_mask=grid.get("display_support_mask", grid["support_mask"]),
                    maximum_gap_px=grid.get("display_grid_max_gap_px", 0.0),
                )
        for dec_value in range(-80, 81, 10):
            if grid["dec_min"] <= dec_value <= grid["dec_max"]:
                _draw_contour_lines(
                    grid_layer, grid["dec_contours"].lines(float(dec_value)), grid_color,
                    label=f"Dec {dec_value:+d}°",
                    visibility_mask=grid_visibility,
                    support_mask=grid.get("display_support_mask", grid["support_mask"]),
                    maximum_gap_px=grid.get("display_grid_max_gap_px", 0.0),
                )
    if draw_constellations and grid is not None:
        constellation_offset = (0.0, 0.0)
        if align_constellations:
            constellation_offset = _estimate_constellation_pixel_offset(
                detected_stars, metadata, wcs, delta_ra, grid["support_mask"],
                width, height,
            )
        constellation_support = (
            grid.get("display_support_mask", grid["support_mask"])
            if continuous_constellations or detected_endpoint_constellations
            else grid["support_mask"]
        )
        _draw_constellation_lines(
            grid_layer, wcs, delta_ra, constellation_support,
            anchor_points=np.asarray(detected_stars, dtype=float) if verify_constellations else None,
            anchor_tolerance_px=float(metadata.get("constellation_anchor_tolerance_px", 5.0)),
            pixel_offset=constellation_offset,
            allow_partial_segments=continuous_constellations,
            edge_admission=constellation_edge_admission,
        )
    if grid is not None:
        support = (grid_visibility if grid_visibility is not None else grid["support_mask"]) > 0
        output[support] = grid_layer[support]
    if draw_detected_stars:
        marker_radius = max(4, round(min(width, height) * 0.0045))
        marker_thickness = max(1, round(marker_radius / 3))
        for x, y in detected_stars:
            center = (int(round(x)), int(round(y)))
            if 0 <= center[0] < width and 0 <= center[1] < height:
                cv2.circle(
                    output, center, marker_radius, (80, 255, 120),
                    marker_thickness, cv2.LINE_AA,
                )
    sip_order = metadata.get("sip_order")
    model_name = metadata.get("model_label")
    if not model_name:
        model_name = f"LOCAL SIP{sip_order}" if sip_order else "LOCAL SIP"
    status = f"{model_name}  {target:%Y-%m-%d %H:%M:%S}"
    cv2.rectangle(output, (8, 8), (8 + max(270, len(status) * 9), 34), (0, 0, 0), -1)
    cv2.putText(output, status, (14, 27), cv2.FONT_HERSHEY_SIMPLEX,
                0.50, (235, 245, 255), 1, cv2.LINE_AA)
    return output


annotate_frame_local = annotate_frame
