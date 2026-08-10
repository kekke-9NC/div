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
    """Return a reusable fixed-camera model registered for this source root."""
    root = Path(cache_root).expanduser().resolve() if cache_root else _default_cache_root()
    _date, camera = _night_identity(source_path, width, height)
    for path in sorted((root / "camera_models").glob("*/camera_model.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("model_type") != CAMERA_MODEL_TYPE:
                continue
            if not payload.get("enabled", False):
                continue
            if int(payload.get("width", 0)) != width or int(payload.get("height", 0)) != height:
                continue
            if camera not in payload.get("camera_aliases", []):
                continue
            valid_dates = payload.get("valid_dates")
            if valid_dates and _date not in valid_dates:
                continue
            wcs_path = Path(payload["wcs_path"]).expanduser()
            if not wcs_path.exists():
                continue
            reference = datetime.fromisoformat(str(payload["reference_datetime"]).replace("Z", "+00:00"))
            return {
                "wcs_file": str(wcs_path),
                "calibration_path": str(path),
                "plate_solve_datetime": reference,
                "job_id": "local-wideangle-camera-model",
                **payload,
            }
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
    return None


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
            _emit(progress_callback, "このカメラ専用の広角モデルを再利用します")
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
) -> Dict[str, Any]:
    """Create or reuse a calibration from one user-selected video frame."""
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
            return {"calibration_path": str(paths["metadata"]), **payload}
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
    ra_values = np.asarray(range_ra).ravel()
    dec_values = np.asarray(range_dec).ravel()
    model = {
        "center_ra": center_ra,
        "ra_min": float(np.nanpercentile(ra_values, 1)),
        "ra_max": float(np.nanpercentile(ra_values, 99)),
        "dec_min": float(np.nanpercentile(dec_values, 1)),
        "dec_max": float(np.nanpercentile(dec_values, 99)),
        "support_mask": support_output,
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
) -> None:
    usable = [line for line in lines if isinstance(line, np.ndarray) and len(line) >= 3]
    for line in usable:
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
    wcs: WCS,
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


def _draw_constellation_lines(
    output: np.ndarray,
    wcs: Any,
    delta_ra: float,
    support_mask: np.ndarray,
    anchor_points: Optional[np.ndarray] = None,
    anchor_tolerance_px: float = 5.0,
) -> None:
    """Project visible constellation line segments into the current frame."""
    height, width = output.shape[:2]
    color = (255, 175, 80)
    lines = _load_constellation_lines()
    if not lines:
        return
    # Project all constellation vertices together.  A SIP forward evaluation
    # is expensive, while one vectorized Newton solve covers every line.
    lengths = [len(line) for line in lines]
    coordinates = np.concatenate(lines, axis=0)
    ra = (coordinates[:, 0] - delta_ra) % 360.0
    x, y, usable = _project_sky_with_forward_wcs(wcs, ra, coordinates[:, 1])
    points = np.column_stack((x, y))
    finite_points = np.isfinite(points).all(axis=1)
    usable &= finite_points
    usable &= (points[:, 0] >= 0) & (points[:, 0] < width)
    usable &= (points[:, 1] >= 0) & (points[:, 1] < height)
    rounded = np.zeros(points.shape, dtype=np.int32)
    visible_indices = np.flatnonzero(usable)
    if len(visible_indices):
        rounded[visible_indices] = np.rint(points[visible_indices]).astype(np.int32)
    # A coordinate such as y=1079.6 is mathematically inside a 1080px frame,
    # but rounds to 1080 and must not index the support mask.
    usable &= (rounded[:, 0] >= 0) & (rounded[:, 0] < width)
    usable &= (rounded[:, 1] >= 0) & (rounded[:, 1] < height)
    in_support = np.zeros(len(rounded), dtype=bool)
    valid_indices = np.flatnonzero(usable)
    if len(valid_indices):
        in_support[valid_indices] = (
            support_mask[rounded[valid_indices, 1], rounded[valid_indices, 0]] > 0
        )
    usable &= in_support
    if anchor_points is not None and len(anchor_points):
        anchors = np.asarray(anchor_points, dtype=float).reshape(-1, 2)
        finite_anchors = anchors[np.isfinite(anchors).all(axis=1)]
        if len(finite_anchors):
            distances = np.linalg.norm(
                points[:, np.newaxis, :] - finite_anchors[np.newaxis, :, :], axis=2
            )
            usable &= np.min(distances, axis=1) <= float(anchor_tolerance_px)
        else:
            usable[:] = False
    offset = 0
    for length in lengths:
        line_points = rounded[offset:offset + length]
        line_usable = usable[offset:offset + length]
        offset += length
        start = 0
        while start < len(line_points):
            while start < len(line_points) and not line_usable[start]:
                start += 1
            end = start
            while end < len(line_points) and line_usable[end]:
                end += 1
            if end - start >= 2:
                cv2.polylines(
                    output, [line_points[start:end].reshape(-1, 1, 2)],
                    False, color, 1, cv2.LINE_AA,
                )
            start = end + 1


def annotate_frame(
    frame_bgr: np.ndarray,
    frame_datetime: datetime,
    calibration_path: Optional[str] = None,
    draw_constellations: bool = False,
    draw_grid: bool = True,
    draw_detected_stars: bool = False,
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
    verify_constellations = bool(
        draw_constellations and metadata.get("verified_constellation_only", False)
    )
    if draw_detected_stars or verify_constellations:
        detection_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        detected_stars, _diagnostic, _reference = _extract_stars(
            detection_gray, maximum_stars=1000, exclude_lower_region=False,
            build_diagnostic=False,
        )
    grid_color = (90, 210, 255)
    grid = None
    grid_layer = output.copy()
    if draw_grid or draw_constellations:
        grid = _forward_grid_model(wcs, metadata, width, height)
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
                )
        for dec_value in range(-80, 81, 10):
            if grid["dec_min"] <= dec_value <= grid["dec_max"]:
                _draw_contour_lines(
                    grid_layer, grid["dec_contours"].lines(float(dec_value)), grid_color,
                    label=f"Dec {dec_value:+d}°",
                )
    if draw_constellations and grid is not None:
        _draw_constellation_lines(
            grid_layer, wcs, delta_ra, grid["support_mask"],
            anchor_points=np.asarray(detected_stars, dtype=float) if verify_constellations else None,
            anchor_tolerance_px=float(metadata.get("constellation_anchor_tolerance_px", 5.0)),
        )
    if grid is not None:
        support = grid["support_mask"] > 0
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
