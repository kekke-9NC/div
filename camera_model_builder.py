"""Build and register a fixed-camera wide-angle astrometric model.

This service is intentionally independent of Tk.  The GUI, the RTSP monitor,
and command-line/tests all call the same code path.  A local SIP WCS provides
the initial astrometric solution; the service then fits the reusable
stereographic camera model and a residual grid to that WCS, validates its
coverage, and atomically registers the result in the normal astrometry cache.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Iterable, Optional, Sequence

import cv2
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

import local_wideangle_astrometry as local_astrometry
from camera_plate_model import FixedCameraPlateModel, MODEL_TYPE
from cloud_coverage import CloudClassification, classify_cloud_fraction
from usage_metrics import record_usage


VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".ts"}
DEFAULT_TARGET_SUPPORT_FRACTION = 0.80
DEFAULT_MINIMUM_SUPPORT_FRACTION = 0.70


@dataclass
class CameraModelBuildRequest:
    source: str
    start: str = ""
    end: str = ""
    cache_root: Optional[str] = None
    cloud_threshold: float = 0.10
    use_cloud_filter: bool = True
    backend: str = "lmstudio_qwen3_5_2b"
    lm_studio_url: str = "http://localhost:1234/v1"
    lm_studio_model_id: str = "qwen/qwen3-vl-4b"
    lm_studio_api_key: str = ""
    minimum_support_fraction: float = DEFAULT_MINIMUM_SUPPORT_FRACTION
    target_support_fraction: float = DEFAULT_TARGET_SUPPORT_FRACTION
    maximum_videos: int = 12
    force_initial_solve: bool = False


@dataclass
class CameraModelBuildResult:
    success: bool
    model_path: str = ""
    calibration_path: str = ""
    report_path: str = ""
    enabled: bool = False
    target_met: bool = False
    support_fraction: float = 0.0
    residual_median_px: float = float("inf")
    residual_p95_px: float = float("inf")
    selected_videos: list[str] = field(default_factory=list)
    cloud_reports: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    elapsed_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "model_path": self.model_path,
            "calibration_path": self.calibration_path,
            "report_path": self.report_path,
            "enabled": self.enabled,
            "target_met": self.target_met,
            "support_fraction": self.support_fraction,
            "residual_median_px": self.residual_median_px,
            "residual_p95_px": self.residual_p95_px,
            "selected_videos": self.selected_videos,
            "cloud_reports": self.cloud_reports,
            "error": self.error,
            "elapsed_seconds": self.elapsed_seconds,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


def _emit(callback: Optional[Callable[[str], None]], message: str) -> None:
    if callback:
        try:
            callback(message)
        except TypeError:
            callback((message, None))


def discover_video_paths(source: str | os.PathLike[str]) -> list[str]:
    path = Path(source).expanduser().resolve()
    if path.is_file():
        return [str(path)] if path.suffix.lower() in VIDEO_EXTENSIONS else []
    if not path.is_dir():
        return []
    return sorted(
        str(item) for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS
    )


def _parse_datetime(value: str, date_hint: Optional[datetime] = None) -> Optional[datetime]:
    value = (value or "").strip()
    if not value:
        return None
    for candidate in (value, value.replace(" ", "T")):
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            pass
    match = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", value)
    if match and date_hint is not None:
        return date_hint.replace(
            hour=int(match.group(1)), minute=int(match.group(2)), second=int(match.group(3) or 0), microsecond=0
        )
    return None


def select_video_paths(source: str, start: str = "", end: str = "", maximum: int = 12) -> list[str]:
    paths = discover_video_paths(source)
    if not paths:
        raise FileNotFoundError(f"動画が見つかりません: {source}")
    if len(paths) == 1 and Path(source).is_file():
        return paths
    date_hint = local_astrometry._capture_datetime(paths[0])
    start_dt = _parse_datetime(start, date_hint)
    end_dt = _parse_datetime(end, date_hint)
    if start_dt is None and start.strip():
        raise ValueError("開始時刻は YYYY-MM-DD HH:MM:SS または HH:MM で指定してください")
    if end_dt is None and end.strip():
        raise ValueError("終了時刻は YYYY-MM-DD HH:MM:SS または HH:MM で指定してください")
    if start_dt is None and end_dt is not None:
        start_dt = end_dt - timedelta(days=1)
    if start_dt is not None and end_dt is not None and end_dt < start_dt and ":" in (end or ""):
        end_dt += timedelta(days=1)
    selected: list[str] = []
    for path in paths:
        stamp = local_astrometry._capture_datetime(path)
        if start_dt is not None and stamp < start_dt:
            continue
        if end_dt is not None and stamp > end_dt:
            continue
        selected.append(path)
    if not selected:
        raise ValueError("指定した時間範囲に動画がありません")
    # Evenly spread the samples over a long range so one minute of clouds does
    # not dominate a multi-hour calibration request.
    maximum = max(1, int(maximum))
    if len(selected) > maximum:
        indices = np.linspace(0, len(selected) - 1, maximum).round().astype(int)
        selected = [selected[int(index)] for index in dict.fromkeys(indices)]
    return selected


def _read_probe_frame(video_path: str) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        frame = None
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 2:
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_count - 1, max(0, frame_count // 3)))
        for _ in range(5):
            ok, candidate = cap.read()
            if ok and candidate is not None:
                frame = candidate
                break
        return frame
    finally:
        cap.release()


def _numeric_seconds(value: str) -> Optional[float]:
    value = (value or "").strip()
    if not value or not re.fullmatch(r"\d+(?:\.\d+)?", value):
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _duration_seconds(value: str) -> Optional[float]:
    numeric = _numeric_seconds(value)
    if numeric is not None:
        return numeric
    parts = (value or "").strip().split(":")
    if len(parts) not in (2, 3) or not all(part.isdigit() for part in parts):
        return None
    try:
        if len(parts) == 2:
            minutes, seconds = (int(part) for part in parts)
            return float(minutes * 60 + seconds)
        hours, minutes, seconds = (int(part) for part in parts)
        return float(hours * 3600 + minutes * 60 + seconds)
    except ValueError:
        return None
def _feature_matrix(points: np.ndarray, width: int, height: int, degree: int) -> np.ndarray:
    x = (points[:, 0] - width / 2.0) / max(1.0, width / 2.0)
    y = (points[:, 1] - height / 2.0) / max(1.0, height / 2.0)
    columns = []
    for total in range(degree + 1):
        for x_power in range(total, -1, -1):
            columns.append((x ** x_power) * (y ** (total - x_power)))
    return np.column_stack(columns)


def _unit_vectors(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    return np.column_stack((np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)))


def _project(parameters: np.ndarray, world: np.ndarray, width: int, height: int, degree: int) -> np.ndarray:
    rotation = Rotation.from_rotvec(parameters[:3]).as_matrix()
    fx, fy = np.exp(parameters[3:5])
    cx, cy = parameters[5:7]
    camera = world @ rotation.T
    rho = np.hypot(camera[:, 0], camera[:, 1])
    theta = np.arctan2(rho, camera[:, 2])
    radial = 2.0 * np.tan(theta / 2.0)
    dx = np.divide(camera[:, 0], rho, out=np.zeros_like(rho), where=rho > 1e-12)
    dy = np.divide(camera[:, 1], rho, out=np.zeros_like(rho), where=rho > 1e-12)
    base = np.column_stack((cx + fx * radial * dx, cy + fy * radial * dy))
    coefficients = parameters[7:].reshape(-1, 2)
    return base + _feature_matrix(base, width, height, degree) @ coefficients


def _initial_parameters(wcs: WCS, width: int, height: int, degree: int) -> np.ndarray:
    center = np.asarray([width / 2.0, height / 2.0])
    points = np.asarray([center, center + [100.0, 0.0], center + [0.0, 100.0]])
    ra, dec = wcs.pixel_to_world_values(points[:, 0], points[:, 1])
    vectors = _unit_vectors(np.asarray(ra), np.asarray(dec))
    ez = vectors[0] / np.linalg.norm(vectors[0])
    ex = vectors[1] - np.dot(vectors[1], ez) * ez
    ex /= max(np.linalg.norm(ex), 1e-12)
    ey = vectors[2] - np.dot(vectors[2], ez) * ez - np.dot(vectors[2], ex) * ex
    ey /= max(np.linalg.norm(ey), 1e-12)
    rotation = np.vstack((ex, ey, ez))
    if np.linalg.det(rotation) < 0:
        ey *= -1.0
        rotation = np.vstack((ex, ey, ez))
    rotation_vec = Rotation.from_matrix(rotation).as_rotvec()
    angular = np.arccos(np.clip(np.dot(ez, vectors[1]), -1.0, 1.0))
    angular_y = np.arccos(np.clip(np.dot(ez, vectors[2]), -1.0, 1.0))
    fx = 100.0 / max(2.0 * np.tan(angular / 2.0), 1e-5)
    fy = 100.0 / max(2.0 * np.tan(angular_y / 2.0), 1e-5)
    terms = (degree + 1) * (degree + 2) // 2
    return np.concatenate((rotation_vec, [np.log(fx), np.log(fy), center[0], center[1]], np.zeros(terms * 2)))


def _fit_model_from_wcs(wcs: WCS, width: int, height: int, degree: int = 4) -> tuple[dict[str, Any], dict[str, float]]:
    columns = max(13, min(31, width // 80))
    rows = max(9, min(23, height // 80))
    x = np.linspace(0.0, width - 1.0, columns)
    y = np.linspace(0.0, height - 1.0, rows)
    mesh_x, mesh_y = np.meshgrid(x, y)
    ra, dec = wcs.pixel_to_world_values(mesh_x, mesh_y)
    world = _unit_vectors(np.asarray(ra).ravel(), np.asarray(dec).ravel())
    target = np.column_stack((mesh_x.ravel(), mesh_y.ravel()))
    usable = np.isfinite(world).all(axis=1) & np.isfinite(target).all(axis=1)
    world = world[usable]
    target = target[usable]
    if len(world) < 60:
        raise RuntimeError("WCSから十分なフィールドサンプルを作れません")
    initial = _initial_parameters(wcs, width, height, degree)

    def residual(parameters: np.ndarray) -> np.ndarray:
        predicted = _project(parameters, world, width, height, degree)
        return (predicted - target).ravel()

    fit = least_squares(
        residual, initial, loss="soft_l1", f_scale=1.5, x_scale="jac", max_nfev=1800,
    )
    if not fit.success and np.linalg.norm(fit.fun) > 20.0 * np.sqrt(len(fit.fun)):
        raise RuntimeError(f"固定カメラ投影の最適化に失敗しました: {fit.message}")
    fitted = _project(fit.x, world, width, height, degree)
    residuals = np.linalg.norm(fitted - target, axis=1)
    x_grid = np.linspace(0.0, width - 1.0, 31)
    y_grid = np.linspace(0.0, height - 1.0, 21)
    gx, gy = np.meshgrid(x_grid, y_grid)
    gra, gdec = wcs.pixel_to_world_values(gx, gy)
    g_world = _unit_vectors(np.asarray(gra).ravel(), np.asarray(gdec).ravel())
    g_target = np.column_stack((gx.ravel(), gy.ravel()))
    g_pred = _project(fit.x, g_world, width, height, degree)
    grid_residual = (g_target - g_pred).reshape(len(y_grid), len(x_grid), 2)
    payload = {
        "model_type": MODEL_TYPE,
        "width": int(width), "height": int(height),
        "polynomial_degree": int(degree),
        "stg_parameters": [float(value) for value in fit.x[:7]],
        "correction_coefficients": fit.x[7:].reshape(-1, 2).tolist(),
        "residual_grid_x": x_grid.tolist(), "residual_grid_y": y_grid.tolist(),
        "residual_grid": grid_residual.tolist(),
    }
    stats = {
        "residual_median_px": float(np.median(residuals)),
        "residual_p95_px": float(np.percentile(residuals, 95)),
        "residual_max_px": float(np.max(residuals)),
        "sample_count": int(len(residuals)),
    }
    return payload, stats


def _support_grid(width: int, height: int, support_hull: Any, columns: int = 20, rows: int = 12) -> tuple[list[list[int]], float]:
    values = np.ones((rows, columns), dtype=np.uint8)
    if support_hull and len(support_hull) >= 3:
        values.fill(0)
        hull = np.asarray(support_hull, dtype=np.float32).reshape(-1, 1, 2)
        for row in range(rows):
            for column in range(columns):
                point = ((column + 0.5) * width / columns, (row + 0.5) * height / rows)
                if cv2.pointPolygonTest(hull, point, False) >= 0:
                    values[row, column] = 1
    fraction = float(np.mean(values))
    return values.tolist(), fraction


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def build_camera_model(
    request: CameraModelBuildRequest,
    *,
    progress_callback: Optional[Callable[[str], None]] = None,
    classifier: Optional[Callable[..., CloudClassification]] = None,
    solver: Optional[Callable[..., dict[str, Any]]] = None,
) -> CameraModelBuildResult:
    """Build a candidate model and register it only after validation."""
    started = time.perf_counter()
    videos: list[str] = []
    clear_videos: list[str] = []
    cloud_reports: list[dict[str, Any]] = []
    result: Optional[CameraModelBuildResult] = None
    try:
        videos = select_video_paths(request.source, request.start, request.end, request.maximum_videos)
        _emit(progress_callback, f"モデル作成対象を {len(videos)} 本選択しました")
        classifier = classifier or classify_cloud_fraction
        for path in videos:
            frame = _read_probe_frame(path)
            if frame is None or not request.use_cloud_filter:
                clear_videos.append(path)
                continue
            result = classifier(
                frame, backend=request.backend, lm_studio_url=request.lm_studio_url,
                lm_studio_model_id=request.lm_studio_model_id, lm_studio_api_key=request.lm_studio_api_key,
            )
            report = {"path": path, **result.as_dict()}
            cloud_reports.append(report)
            _emit(progress_callback, f"雲量 {Path(path).name}: {result.cloud_fraction * 100:.1f}% ({result.source})")
            if result.cloud_fraction < float(request.cloud_threshold):
                clear_videos.append(path)
        if not clear_videos:
            raise RuntimeError("雲量条件を満たす動画がありません")
        seed = clear_videos[len(clear_videos) // 2]
        _emit(progress_callback, f"初期ローカルプレートソルブ: {Path(seed).name}")
        solver = solver or local_astrometry.solve_video_local
        start_seconds = _duration_seconds(request.start)
        end_seconds = _duration_seconds(request.end)
        # Building a camera model is an explicit recalibration request.  Do
        # not let solve_video_local silently return a previously registered
        # model from another night/source; that can carry an old DATE-OBS and
        # rotate every constellation by many degrees.
        force_initial_solve = True
        if len(videos) == 1 and start_seconds is not None:
            cap = cv2.VideoCapture(seed)
            fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else 0.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
            cap.release()
            if fps > 0.1:
                chosen_seconds = start_seconds if end_seconds is None else (start_seconds + max(start_seconds, end_seconds)) / 2.0
                frame_index = round(chosen_seconds * fps)
                if frame_count > 0:
                    frame_index = min(max(0, frame_index), frame_count - 1)
                _emit(progress_callback, f"指定秒数の基準フレームでソルブ: {chosen_seconds:.2f}s")
                calibration = local_astrometry.solve_video_frame_local(
                    seed, frame_index, cache_root=request.cache_root, force=force_initial_solve,
                    progress_callback=progress_callback,
                )
            else:
                calibration = solver(seed, cache_root=request.cache_root, force=force_initial_solve, progress_callback=progress_callback)
        else:
            calibration = solver(seed, cache_root=request.cache_root, force=force_initial_solve, progress_callback=progress_callback)
        wcs_path = Path(calibration["wcs_file"]).expanduser().resolve()
        with fits.open(wcs_path) as hdul:
            header = hdul[0].header.copy()
            wcs = WCS(header, relax=True, fix=False)
            width = int(calibration.get("width") or header.get("IMAGEW"))
            height = int(calibration.get("height") or header.get("IMAGEH"))
            wcs_reference_datetime = header.get("DATE-OBS")
        if width <= 0 or height <= 0:
            raise RuntimeError("WCSの画像サイズが不正です")
        _emit(progress_callback, "WCSを固定カメラ投影へ変換中...")
        # A SIP4 WCS sampled from a clear 2am segment benefits from one extra
        # camera-model term when the solve has enough verified stars.  Keep
        # the conservative degree-4 fallback for sparse/legacy calibrations;
        # the extra degree is only used for the high-precision path and does
        # not alter the WCS itself.
        model_degree = 5 if int(calibration.get("sip_match_count", 0) or 0) >= 45 else 4
        payload, fit_stats = _fit_model_from_wcs(wcs, width, height, degree=model_degree)
        support_source = calibration.get("sip_support_hull")
        if not support_source and calibration.get("support_grid"):
            support = calibration["support_grid"]
            support_array = np.asarray(support, dtype=np.uint8)
            support_fraction = float(np.mean(support_array > 0)) if support_array.size else 0.0
        else:
            support, support_fraction = _support_grid(width, height, support_source)
        # DATE-OBS belongs to the WCS that was actually used for this model.
        # Prefer it over metadata returned by a reused/legacy solver result.
        reference_datetime = wcs_reference_datetime or calibration.get("reference_datetime")
        if isinstance(reference_datetime, datetime):
            reference_datetime = reference_datetime.isoformat()
        if not reference_datetime:
            reference_datetime = local_astrometry._capture_datetime(seed).isoformat()
        aliases = []
        for path in clear_videos:
            try:
                _date, alias = local_astrometry._night_identity(path, width, height)
                if alias not in aliases:
                    aliases.append(alias)
            except Exception:
                continue
        if not aliases:
            aliases = [hashlib.sha256(str(Path(seed).parent).encode()).hexdigest()[:12]]
        camera_alias = aliases[0]
        root = Path(request.cache_root).expanduser().resolve() if request.cache_root else local_astrometry._default_cache_root()
        model_dir = root / "camera_models" / f"auto-{camera_alias}-{width}x{height}"
        model_path = model_dir / "camera_model.json"
        report_path = model_dir / "build_report.json"
        enabled = support_fraction >= float(request.minimum_support_fraction) and fit_stats["residual_p95_px"] <= 3.0
        target_met = support_fraction >= float(request.target_support_fraction) and fit_stats["residual_p95_px"] <= 2.0
        payload.update({
            "reference_datetime": str(reference_datetime),
            "wcs_path": str(wcs_path),
            "camera_aliases": aliases,
            "valid_dates": [],
            "enabled": bool(enabled),
            "target_met": bool(target_met),
            "support_fraction": support_fraction,
            "support_grid": support,
            "model_label": "AUTO FIXED CAMERA" if enabled else "AUTO FIXED CAMERA (CANDIDATE)",
            "model_revision": "camera-model-builder-v2",
            "reference_datetime_source": "wcs:DATE-OBS" if wcs_reference_datetime else "solver-or-source",
            "verified_constellation_only": bool(enabled),
            "constellation_anchor_tolerance_px": 5.0,
            # Keep the stars that established the WCS with the reusable model.
            # Annotated temporal-mean frames can shift a few pixels, so the
            # renderer uses these catalog anchors to estimate a small per-frame
            # translation before validating constellation endpoints.
            "catalog_stars": calibration.get("catalog_stars", []),
            "constellation_star_alignment": bool(enabled),
            "source_videos": clear_videos,
            "cloud_threshold": float(request.cloud_threshold),
            "fit_stats": fit_stats,
            "created_at": datetime.now().astimezone().isoformat(),
        })
        report = {
            "model": payload,
            "calibration": calibration,
            "selected_videos": videos,
            "clear_videos": clear_videos,
            "cloud_reports": cloud_reports,
            "quality": {**fit_stats, "support_fraction": support_fraction, "enabled": enabled, "target_met": target_met},
        }
        _write_json_atomic(model_path, payload)
        _write_json_atomic(report_path, report)
        _emit(progress_callback, f"カメラ補正データを登録しました: {model_path}")
        result = CameraModelBuildResult(
            True, str(model_path), str(calibration.get("calibration_path", "")), str(report_path),
            bool(enabled), bool(target_met), support_fraction,
            fit_stats["residual_median_px"], fit_stats["residual_p95_px"],
            videos, cloud_reports,
        )
        return result
    except Exception as exc:
        _emit(progress_callback, f"高精度モデル作成失敗: {exc}")
        result = CameraModelBuildResult(False, selected_videos=[], error=f"{type(exc).__name__}: {exc}")
        return result
    finally:
        elapsed = time.perf_counter() - started
        input_tokens = sum(max(0, int(report.get("input_tokens", 0) or 0)) for report in cloud_reports)
        output_tokens = sum(max(0, int(report.get("output_tokens", 0) or 0)) for report in cloud_reports)
        total_tokens = sum(max(0, int(report.get("total_tokens", 0) or 0)) for report in cloud_reports)
        if result is not None:
            result.elapsed_seconds = elapsed
            result.input_tokens = input_tokens
            result.output_tokens = output_tokens
            result.total_tokens = total_tokens or input_tokens + output_tokens
        record_usage(
            "camera_model_build",
            elapsed_seconds=elapsed,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens or input_tokens + output_tokens,
            status=("success" if result is not None and result.success else "error"),
            source=request.source,
            model=request.lm_studio_model_id,
            error=(result.error if result is not None else ""),
            metadata={
                "selected_videos": len(videos),
                "clear_videos": len(clear_videos),
                "backend": request.backend,
                "report_path": result.report_path if result is not None else "",
            },
        )
