"""High-precision meteor radiant analysis for the analysis tab.

The detector stores meteor lines in camera pixels.  This module keeps the
analysis independent from Tk: it restores those lines, checks them against a
registered camera model's validated support area, converts them to ICRS
coordinates at the detection time, matches active meteor showers, and can
render the result on a 3-D celestial sphere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


SIDEREAL_DAY_SECONDS = 86164.0905
POINT_PATTERN = re.compile(
    r"\(\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\)"
)


@dataclass(frozen=True)
class MeteorShower:
    code: str
    name: str
    active_start: Tuple[int, int]
    peak: Tuple[int, int]
    active_end: Tuple[int, int]
    radiant_ra_deg: float
    radiant_dec_deg: float
    match_limit_deg: float = 12.0

    def is_active(self, moment: datetime) -> bool:
        """Return whether the approximate activity window contains moment."""
        day = moment.timetuple().tm_yday
        start = datetime(moment.year, *self.active_start).timetuple().tm_yday
        end = datetime(moment.year, *self.active_end).timetuple().tm_yday
        if start <= end:
            return start <= day <= end
        return day >= start or day <= end

    def radiant_vector(self) -> np.ndarray:
        return radec_to_unit_vector(self.radiant_ra_deg, self.radiant_dec_deg)


# Approximate geocentric radiants and activity windows for the principal
# annual showers.  The UI reports angular residuals so the result is not
# presented as a definitive orbit solution.  Values are intentionally kept in
# one auditable table so a future IAU/IMO catalogue can replace them cleanly.
METEOR_SHOWERS: Tuple[MeteorShower, ...] = (
    MeteorShower("QUA", "しぶんぎ座流星群", (1, 1), (1, 4), (1, 12), 230.0, 49.0),
    MeteorShower("LYR", "こと座流星群", (4, 14), (4, 22), (4, 30), 271.0, 34.0),
    MeteorShower("ETA", "みずがめ座η流星群", (4, 19), (5, 6), (5, 28), 337.0, -1.0),
    MeteorShower("SDA", "みずがめ座δ南流星群", (7, 12), (7, 30), (8, 23), 340.0, -16.0),
    MeteorShower("CAP", "やぎ座α流星群", (7, 3), (7, 30), (8, 15), 307.0, -10.0),
    MeteorShower("PER", "ペルセウス座流星群", (7, 17), (8, 12), (8, 24), 48.0, 58.0),
    MeteorShower("KCG", "はくちょう座κ流星群", (8, 3), (8, 17), (8, 25), 286.0, 59.0),
    MeteorShower("AUR", "ぎょしゃ座α流星群", (8, 25), (9, 1), (9, 10), 91.0, 39.0),
    MeteorShower("SPE", "9月ペルセウス座ε流星群", (9, 5), (9, 9), (9, 21), 48.0, 40.0),
    MeteorShower("DRA", "りゅう座流星群", (10, 6), (10, 8), (10, 10), 262.0, 54.0),
    MeteorShower("ORI", "オリオン座流星群", (10, 2), (10, 21), (11, 7), 95.0, 16.0),
    MeteorShower("STA", "おうし座南流星群", (10, 10), (11, 5), (11, 20), 52.0, 13.0),
    MeteorShower("LEO", "しし座流星群", (11, 6), (11, 17), (11, 30), 152.0, 22.0),
    MeteorShower("GEM", "ふたご座流星群", (12, 4), (12, 14), (12, 20), 112.0, 33.0),
    MeteorShower("URS", "こぐま座流星群", (12, 17), (12, 22), (12, 26), 217.0, 76.0),
)


@dataclass
class RadiantCandidate:
    code: str
    name: str
    radiant_ra_deg: float
    radiant_dec_deg: float
    angular_distance_deg: float
    active: bool
    side: str
    score: float


@dataclass
class RadiantResult:
    info_path: str
    source: str
    detection_time: Optional[datetime]
    start_pixel: Tuple[float, float]
    end_pixel: Tuple[float, float]
    start_radec: Tuple[float, float]
    end_radec: Tuple[float, float]
    line_source: str
    support_fraction: float
    fully_supported: bool
    model_label: str
    shower_code: str = "SPO"
    shower_name: str = "未分類（散在流星または判定保留）"
    radiant_radec: Optional[Tuple[float, float]] = None
    radiant_distance_deg: Optional[float] = None
    radiant_side: str = "unknown"
    confidence: str = "判定保留"
    candidates: List[RadiantCandidate] = field(default_factory=list)
    note: str = ""


@dataclass
class RadiantReport:
    model_path: str
    model_label: str
    results: List[RadiantResult]
    skipped: List[Tuple[str, str]]

    @property
    def supported_results(self) -> List[RadiantResult]:
        return [result for result in self.results if result.fully_supported]


def radec_to_unit_vector(ra_deg: float, dec_deg: float) -> np.ndarray:
    ra = math.radians(float(ra_deg))
    dec = math.radians(float(dec_deg))
    cos_dec = math.cos(dec)
    return np.asarray((cos_dec * math.cos(ra), cos_dec * math.sin(ra), math.sin(dec)), dtype=float)


def unit_vector_to_radec(vector: Sequence[float]) -> Tuple[float, float]:
    vector = _normalize(np.asarray(vector, dtype=float))
    return (
        math.degrees(math.atan2(vector[1], vector[0])) % 360.0,
        math.degrees(math.asin(float(np.clip(vector[2], -1.0, 1.0)))),
    )


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("ゼロ長の天球ベクトルです")
    return vector / norm


def angular_distance_deg(first: Sequence[float], second: Sequence[float]) -> float:
    dot = float(np.clip(np.dot(_normalize(np.asarray(first)), _normalize(np.asarray(second))), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def slerp_vectors(first: Sequence[float], second: Sequence[float], count: int = 32) -> np.ndarray:
    first = _normalize(np.asarray(first, dtype=float))
    second = _normalize(np.asarray(second, dtype=float))
    count = max(2, int(count))
    dot = float(np.clip(np.dot(first, second), -1.0, 1.0))
    t = np.linspace(0.0, 1.0, count)
    if abs(dot) > 0.9995:
        points = first[None, :] + t[:, None] * (second - first)[None, :]
        return points / np.linalg.norm(points, axis=1, keepdims=True)
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    return (
        np.sin((1.0 - t)[:, None] * theta) / sin_theta * first[None, :]
        + np.sin(t[:, None] * theta) / sin_theta * second[None, :]
    )


def _parse_point(value: Optional[str]) -> Optional[Tuple[float, float]]:
    if not value:
        return None
    match = POINT_PATTERN.search(str(value))
    if not match:
        return None
    try:
        return float(match.group(1)), float(match.group(2))
    except ValueError:
        return None


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value or str(value).strip().upper() in {"N/A", "NONE", "UNKNOWN"}:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1]
        if not re.search(r"[+-]\d{2}:\d{2}$", text):
            text += "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _line_from_artifact(data: Dict[str, str], center: Tuple[float, float]) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], str]]:
    """Recover legacy line endpoints from a saved composite image."""
    candidates: List[Tuple[float, float, float, float]] = []
    for key in ("Saved Composite Path", "Saved Full Diff Path", "Saved Annotated Path", "Saved Full Path"):
        artifact = str(data.get(key, "")).strip()
        if not artifact or not os.path.isfile(artifact):
            continue
        image = cv2.imread(artifact, cv2.IMREAD_COLOR)
        if image is None:
            continue
        try:
            from image_processing import detect_lines

            minimum = max(20, round(min(image.shape[:2]) * 0.012))
            lines = detect_lines(image, min_length=minimum)
        except Exception:
            lines = []
        if not lines:
            continue
        cx, cy = center
        for first, second in lines:
            x1, y1 = float(first[0]), float(first[1])
            x2, y2 = float(second[0]), float(second[1])
            length = math.hypot(x2 - x1, y2 - y1)
            if length < minimum:
                continue
            # Distance from the recorded center to the finite segment.
            dx, dy = x2 - x1, y2 - y1
            denominator = dx * dx + dy * dy
            t = 0.0 if denominator <= 1e-9 else np.clip(((cx - x1) * dx + (cy - y1) * dy) / denominator, 0.0, 1.0)
            nearest = (x1 + t * dx, y1 + t * dy)
            distance = math.hypot(nearest[0] - cx, nearest[1] - cy)
            candidates.append((distance, -length, x1, y1, x2, y2))
        if candidates:
            best = min(candidates, key=lambda item: (item[0], item[1]))
            return (best[2], best[3]), (best[4], best[5]), f"artifact:{key}"
    return None


def parse_pixel_line(data: Dict[str, str]) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], str]]:
    start = _parse_point(data.get("Detected Line Start (px)"))
    end = _parse_point(data.get("Detected Line End (px)"))
    if start is not None and end is not None:
        return start, end, "info.txt"
    center = _parse_point(data.get("Detected Line Center (px)"))
    if center is None:
        return None
    return _line_from_artifact(data, center)


def _capture_datetime(source: str) -> Optional[datetime]:
    try:
        import local_wideangle_astrometry

        return local_wideangle_astrometry._capture_datetime(source)
    except Exception:
        return None


def _same_time_zone(first: datetime, second: datetime) -> Tuple[datetime, datetime]:
    if first.tzinfo is not None and second.tzinfo is None:
        second = second.replace(tzinfo=first.tzinfo)
    elif first.tzinfo is None and second.tzinfo is not None:
        second = second.replace(tzinfo=None)
    return first, second


def _sidereal_delta_ra(reference: datetime, target: datetime) -> float:
    reference, target = _same_time_zone(reference, target)
    return (target - reference).total_seconds() / SIDEREAL_DAY_SECONDS * 360.0


def _probe_size(source: str) -> Optional[Tuple[int, int]]:
    if not source or not os.path.isfile(source):
        return None
    capture = cv2.VideoCapture(source)
    try:
        if not capture.isOpened():
            return None
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height) if width > 0 and height > 0 else None
    finally:
        capture.release()


def _data_source(data: Dict[str, str]) -> str:
    return str(data.get("Source", "")).strip()


def _load_data(path: str) -> Dict[str, str]:
    import meteor_sky_viewer as viewer

    return viewer.parse_info_file(path)


def resolve_model_path(info_paths: Sequence[str], requested_path: Optional[str] = None) -> Optional[str]:
    """Resolve an explicit model or the best registered model for the first source."""
    if requested_path and os.path.isfile(os.path.expanduser(requested_path)):
        return str(Path(requested_path).expanduser().resolve())
    import local_wideangle_astrometry

    for path in info_paths:
        data = _load_data(path)
        source = _data_source(data)
        size = _probe_size(source)
        if size is None:
            continue
        selected = local_wideangle_astrometry._registered_camera_model(source, *size)
        if selected and selected.get("calibration_path"):
            return str(selected["calibration_path"])
    return None


def _match_shower(
    start_vector: np.ndarray,
    end_vector: np.ndarray,
    moment: Optional[datetime],
) -> Tuple[Optional[MeteorShower], Optional[np.ndarray], Optional[float], str, List[RadiantCandidate]]:
    normal_raw = np.cross(start_vector, end_vector)
    normal_norm = float(np.linalg.norm(normal_raw))
    if normal_norm < 1e-8:
        return None, None, None, "unknown", []
    normal = normal_raw / normal_norm
    tangent = _normalize(end_vector - np.dot(start_vector, end_vector) * start_vector)
    theta = math.acos(float(np.clip(np.dot(start_vector, end_vector), -1.0, 1.0)))
    candidates: List[RadiantCandidate] = []
    for shower in METEOR_SHOWERS:
        radiant = shower.radiant_vector()
        distance = math.degrees(math.asin(min(1.0, abs(float(np.dot(normal, radiant))))))
        q = math.atan2(float(np.dot(radiant, tangent)), float(np.dot(radiant, start_vector)))
        if q < -1e-5:
            side = "start"
        elif q > theta + 1e-5:
            side = "end"
        else:
            side = "segment"
        active = bool(moment and shower.is_active(moment))
        # An annual shower must not be reported as the origin merely because
        # its radiant happens to be geometrically close outside its activity
        # season.  Keep inactive candidates in the audit trail, but only let
        # active candidates become the classification.
        temporal_penalty = 0.0 if active else 100.0
        side_penalty = 0.0 if side in {"start", "end"} else 15.0
        score = distance + temporal_penalty + side_penalty
        candidates.append(RadiantCandidate(
            shower.code, shower.name, shower.radiant_ra_deg, shower.radiant_dec_deg,
            distance, active, side, score,
        ))
    candidates.sort(key=lambda item: item.score)
    best = candidates[0] if candidates else None
    shower = next((item for item in METEOR_SHOWERS if item.code == best.code), None) if best else None
    if best is None or not best.active:
        return None, None, best.angular_distance_deg if best else None, best.side if best else "unknown", candidates
    if best.angular_distance_deg > (shower.match_limit_deg if shower else 12.0) or best.side == "segment":
        return None, None, best.angular_distance_deg, best.side, candidates
    return shower, shower.radiant_vector(), best.angular_distance_deg, best.side, candidates


def _read_radec(data: Dict[str, str], prefix: str) -> Optional[Tuple[float, float]]:
    try:
        ra = float(data.get(f"RA {prefix} (deg)", "N/A"))
        dec = float(data.get(f"Dec {prefix} (deg)", "N/A"))
        if not (math.isfinite(ra) and math.isfinite(dec)):
            return None
        return ra % 360.0, dec
    except (TypeError, ValueError):
        return None


def analyze_info_files(
    info_paths: Sequence[str],
    model_path: Optional[str] = None,
    progress_callback: Optional[Any] = None,
) -> RadiantReport:
    """Analyze dropped info files against a fixed camera plate model."""
    if not info_paths:
        raise ValueError("解析対象のinfo.txtがありません")
    resolved_model = resolve_model_path(info_paths, model_path)
    if not resolved_model:
        raise ValueError("同じカメラ・解像度の高精度カメラ補正データを選択してください")
    import local_wideangle_astrometry

    metadata, wcs = local_wideangle_astrometry._load_calibration(resolved_model)
    model_label = str(metadata.get("model_label") or Path(resolved_model).parent.name)
    results: List[RadiantResult] = []
    skipped: List[Tuple[str, str]] = []
    size_cache: Dict[str, Optional[Tuple[int, int]]] = {}
    grid_cache: Dict[Tuple[int, int], Any] = {}
    for index, info_path in enumerate(info_paths, start=1):
        if progress_callback:
            progress_callback(f"放射点解析中: {index}/{len(info_paths)}")
        try:
            data = _load_data(info_path)
            source = _data_source(data)
            line = parse_pixel_line(data)
            if line is None:
                skipped.append((info_path, "流星の始点・終点を取得できませんでした"))
                continue
            start_pixel, end_pixel, line_source = line
            if source not in size_cache:
                size_cache[source] = _probe_size(source)
            size = size_cache[source]
            if size is None:
                size = None
                for key in ("Saved Composite Path", "Saved Full Diff Path", "Saved Annotated Path"):
                    candidate = data.get(key)
                    if candidate and os.path.isfile(candidate):
                        image = cv2.imread(candidate)
                        if image is not None:
                            size = (image.shape[1], image.shape[0])
                            break
                if size is None:
                    skipped.append((info_path, "元動画または描画画像の解像度を取得できませんでした"))
                    continue
            width, height = size
            metadata_width = int(metadata.get("width", width))
            metadata_height = int(metadata.get("height", height))
            if (width, height) != (metadata_width, metadata_height):
                scale_x = width / metadata_width
                scale_y = height / metadata_height
                start_pixel = (start_pixel[0] * scale_x, start_pixel[1] * scale_y)
                end_pixel = (end_pixel[0] * scale_x, end_pixel[1] * scale_y)
            frame_time = _parse_iso_datetime(data.get("Detection Time (UTC)")) or _capture_datetime(source)
            if frame_time is None:
                skipped.append((info_path, "検出時刻を取得できませんでした"))
                continue
            reference_value = metadata.get("reference_datetime")
            reference = _parse_iso_datetime(str(reference_value)) if reference_value else None
            if reference is None:
                skipped.append((info_path, "補正データの基準時刻がありません"))
                continue
            delta_ra = _sidereal_delta_ra(reference, frame_time)
            grid_key = (width, height)
            if grid_key not in grid_cache:
                grid_cache[grid_key] = local_wideangle_astrometry._forward_grid_model(wcs, metadata, width, height)
            grid = grid_cache[grid_key]
            support_mask = grid["support_mask"]
            samples = np.linspace(np.asarray(start_pixel), np.asarray(end_pixel), 41)
            rounded = np.rint(samples).astype(int)
            valid = (
                (rounded[:, 0] >= 0) & (rounded[:, 0] < width)
                & (rounded[:, 1] >= 0) & (rounded[:, 1] < height)
            )
            valid_indices = np.flatnonzero(valid)
            if len(valid_indices):
                valid[valid_indices] &= support_mask[rounded[valid_indices, 1], rounded[valid_indices, 0]] > 0
            support_fraction = float(np.mean(valid)) if len(valid) else 0.0
            fully_supported = bool(len(valid) and np.all(valid))
            if not fully_supported:
                skipped.append((info_path, f"プレートソルブ有効領域外または一部外（有効率 {support_fraction * 100:.0f}%）"))
                continue
            start_ra, start_dec = wcs.pixel_to_world_values(*start_pixel)
            end_ra, end_dec = wcs.pixel_to_world_values(*end_pixel)
            start_radec = ((float(start_ra) + delta_ra) % 360.0, float(start_dec))
            end_radec = ((float(end_ra) + delta_ra) % 360.0, float(end_dec))
            start_vector = radec_to_unit_vector(*start_radec)
            end_vector = radec_to_unit_vector(*end_radec)
            shower, radiant, radiant_distance, side, candidates = _match_shower(start_vector, end_vector, frame_time)
            if shower is None:
                shower_code = "SPO"
                shower_name = "未分類（散在流星または判定保留）"
                confidence = "判定保留"
                inactive = next((candidate for candidate in candidates if not candidate.active), None)
                if inactive and inactive.angular_distance_deg <= 12.0:
                    note = (
                        f"幾何的に近い候補は{inactive.name}ですが、活動期間外のため採用しません"
                    )
                else:
                    note = "流星線分と活動中流星群の放射点が許容角内で一致しません"
            else:
                shower_code = shower.code
                shower_name = shower.name
                confidence = "高" if radiant_distance is not None and radiant_distance <= 6.0 else "候補"
                note = (
                    f"放射点との角距離 {radiant_distance:.1f}° / {side}側延長。"
                    "検出線の端点順は保存されていないため、運動方向は未確定"
                )
            results.append(RadiantResult(
                info_path=info_path,
                source=source,
                detection_time=frame_time,
                start_pixel=start_pixel,
                end_pixel=end_pixel,
                start_radec=start_radec,
                end_radec=end_radec,
                line_source=line_source,
                support_fraction=support_fraction,
                fully_supported=fully_supported,
                model_label=model_label,
                shower_code=shower_code,
                shower_name=shower_name,
                radiant_radec=unit_vector_to_radec(radiant) if radiant is not None else None,
                radiant_distance_deg=radiant_distance,
                radiant_side=side,
                confidence=confidence,
                candidates=candidates,
                note=note,
            ))
        except Exception as exc:
            skipped.append((info_path, str(exc)))
    return RadiantReport(resolved_model, model_label, results, skipped)


def _extension_vectors(result: RadiantResult) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    start = radec_to_unit_vector(*result.start_radec)
    end = radec_to_unit_vector(*result.end_radec)
    radiant = radec_to_unit_vector(*result.radiant_radec) if result.radiant_radec else None
    return start, end, radiant


def _japanese_font_properties():
    """Return a Japanese-capable font when the host provides one.

    Matplotlib's default DejaVu font cannot render the Japanese UI labels and
    silently replaces them with tofu boxes.  The app runs primarily on macOS,
    but the fallback list keeps headless/Linux installs usable as well.
    """
    try:
        from matplotlib import font_manager

        candidates = (
            "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
            "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc",
            "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
            "/System/Library/Fonts/ヒラギノ丸ゴ ProN W4.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        )
        for candidate in candidates:
            if os.path.isfile(candidate):
                return font_manager.FontProperties(fname=candidate)
    except Exception:
        pass
    return None


def draw_radiant_sphere(report: RadiantReport, figure=None):
    """Draw a 3-D celestial sphere; returns the Matplotlib figure and axes."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    japanese_font = _japanese_font_properties()
    if figure is None:
        figure = plt.figure(figsize=(10, 8), facecolor="#0B0F18")
    figure.clear()
    axis = figure.add_subplot(111, projection="3d", facecolor="#0B0F18")
    # Matplotlib's default 3-D panes are opaque light gray and overwhelm the
    # dark application theme.  The sphere and celestial grid should carry the
    # visual hierarchy, with only a restrained coordinate frame behind them.
    axis.grid(False)
    for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
        pane.set_facecolor("#0B0F18")
        pane.set_edgecolor("#243650")
        pane.set_alpha(0.28)
    u = np.linspace(0.0, 2.0 * np.pi, 72)
    v = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 36)
    sphere_x = np.outer(np.cos(u), np.cos(v))
    sphere_y = np.outer(np.sin(u), np.cos(v))
    sphere_z = np.outer(np.ones_like(u), np.sin(v))
    axis.plot_surface(sphere_x, sphere_y, sphere_z, color="#17243A", alpha=0.18, linewidth=0, shade=False)
    for dec in range(-60, 61, 30):
        points = np.asarray([radec_to_unit_vector(ra, dec) for ra in np.linspace(0, 360, 181)])
        axis.plot(points[:, 0], points[:, 1], points[:, 2], color="#53657E", alpha=0.35, linewidth=0.55)
    for ra in range(0, 360, 30):
        points = np.asarray([radec_to_unit_vector(ra, dec) for dec in np.linspace(-90, 90, 91)])
        axis.plot(points[:, 0], points[:, 1], points[:, 2], color="#53657E", alpha=0.3, linewidth=0.55)

    colors = {
        "PER": "#70A7FF", "SDA": "#F5C76B", "CAP": "#FF9B71", "KCG": "#66D9EF",
        "ORI": "#C58BFF", "GEM": "#5DE2A5", "SPO": "#C5CFDD",
    }
    plotted_labels = set()
    shower_handles = []
    for result in report.supported_results:
        start, end, radiant = _extension_vectors(result)
        color = colors.get(result.shower_code, "#C5CFDD")
        path = slerp_vectors(start, end, 40)
        axis.plot(path[:, 0], path[:, 1], path[:, 2], color=color, linewidth=2.4, alpha=0.95)
        if radiant is not None:
            if result.radiant_side == "start":
                extension = slerp_vectors(radiant, start, 36)
            else:
                extension = slerp_vectors(end, radiant, 36)
            axis.plot(extension[:, 0], extension[:, 1], extension[:, 2], color=color, linewidth=1.4, linestyle="--", alpha=0.85)
            axis.scatter([radiant[0]], [radiant[1]], [radiant[2]], color=color, s=48, depthshade=False)
            if result.shower_code not in plotted_labels:
                axis.text(radiant[0] * 1.08, radiant[1] * 1.08, radiant[2] * 1.08, result.shower_code, color=color, fontsize=9)
                shower_handles.append(
                    Line2D(
                        [0], [0], marker="o", linestyle="None", color=color,
                        markerfacecolor=color, markeredgecolor=color,
                        label=f"{result.shower_code} {result.shower_name}",
                    )
                )
                plotted_labels.add(result.shower_code)
        else:
            axis.scatter([start[0]], [start[1]], [start[2]], color="#C5CFDD", s=20, depthshade=False)

    total_inputs = len(report.results) + len(report.skipped)
    axis.set_title(
        f"放射点解析 | 有効流星 {len(report.supported_results)} / 読込 {total_inputs} | {report.model_label}",
        color="#F4F7FC", pad=18, fontproperties=japanese_font,
    )
    axis.set_xlabel("X", color="#A8B3C5")
    axis.set_ylabel("Y", color="#A8B3C5")
    axis.set_zlabel("Z", color="#A8B3C5")
    axis.tick_params(colors="#A8B3C5", labelsize=8)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xlim(-1.08, 1.08)
    axis.set_ylim(-1.08, 1.08)
    axis.set_zlim(-1.08, 1.08)
    axis.view_init(elev=22, azim=-58)
    handles = [Line2D([0], [0], color="#F4F7FC", linewidth=2.4, label="実際の流星経路")]
    handles.append(Line2D([0], [0], color="#F4F7FC", linewidth=1.4, linestyle="--", label="放射点までの延長"))
    handles.extend(shower_handles)
    axis.legend(
        handles=handles,
        loc="upper left",
        facecolor="#141C2A",
        edgecolor="#415875",
        labelcolor="#F4F7FC",
        prop=japanese_font,
    )
    figure.tight_layout()
    return figure, axis


def save_radiant_report_plot(report: RadiantReport, output_path: str) -> str:
    import matplotlib.pyplot as plt

    figure, _axis = draw_radiant_sphere(report, plt.figure(figsize=(10, 8), facecolor="#0B0F18"))
    figure.savefig(output_path, dpi=150, facecolor=figure.get_facecolor())
    plt.close(figure)
    return output_path
