"""Alternative visualizations for a high-precision radiant analysis report.

The 3-D sphere is useful for explaining the geometry, but it is not the most
efficient view for every question.  This module produces the complementary
views used by the analysis workflow:

* Aitoff all-sky map
* full great-circle convergence map
* camera-plane overlay
* rectangular RA/Dec map
* radiant-density heatmap
* RA/Dec polar plot
* cumulative time animation

All functions accept the same :class:`RadiantReport` as the existing sphere
renderer so the support-mask filtering and shower classification cannot drift
between visualizations.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import meteor_radiant_analysis as analysis


DEFAULT_LATITUDE_DEG = 35.0
DEFAULT_LONGITUDE_DEG = 135.0
COLORS = {
    "PER": "#70A7FF",
    "SDA": "#F5C76B",
    "CAP": "#FF9B71",
    "KCG": "#66D9EF",
    "ORI": "#C58BFF",
    "GEM": "#5DE2A5",
    "SPO": "#C5CFDD",
}
BACKGROUND = "#0B0F18"
PANEL = "#141C2A"
TEXT = "#F4F7FC"
MUTED = "#A8B3C5"


def _font():
    return analysis._japanese_font_properties()


def _color(result: analysis.RadiantResult) -> str:
    return COLORS.get(result.shower_code, COLORS["SPO"])


def _lon_deg(ra_deg: float) -> float:
    """Convert RA to a continuous [-180, 180] longitude for sky projections."""
    return ((float(ra_deg) + 180.0) % 360.0) - 180.0


def _radec_from_vectors(vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vectors = np.asarray(vectors, dtype=float)
    ra = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0])) % 360.0
    dec = np.degrees(np.arcsin(np.clip(vectors[:, 2], -1.0, 1.0)))
    return ra, dec


def _path_radec(result: analysis.RadiantResult, count: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    points = analysis.slerp_vectors(
        analysis.radec_to_unit_vector(*result.start_radec),
        analysis.radec_to_unit_vector(*result.end_radec),
        count,
    )
    return _radec_from_vectors(points)


def _extension_radec(result: analysis.RadiantResult, count: int = 64) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if result.radiant_radec is None:
        return None
    radiant = analysis.radec_to_unit_vector(*result.radiant_radec)
    start = analysis.radec_to_unit_vector(*result.start_radec)
    end = analysis.radec_to_unit_vector(*result.end_radec)
    if result.radiant_side == "start":
        points = analysis.slerp_vectors(radiant, start, count)
    else:
        points = analysis.slerp_vectors(end, radiant, count)
    return _radec_from_vectors(points)


def _split_wrap(lon_deg: np.ndarray, lat_deg: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split a path when it crosses the RA=0 seam."""
    lon = np.asarray(lon_deg, dtype=float)
    lat = np.asarray(lat_deg, dtype=float)
    if len(lon) < 2:
        return [(lon, lat)]
    breaks = np.flatnonzero(np.abs(np.diff(lon)) > 180.0)
    chunks: List[Tuple[np.ndarray, np.ndarray]] = []
    start = 0
    for end in breaks + 1:
        if end - start >= 2:
            chunks.append((lon[start:end], lat[start:end]))
        start = int(end)
    if len(lon) - start >= 2:
        chunks.append((lon[start:], lat[start:]))
    return chunks or [(lon, lat)]


def _configure_dark_figure(figure, title: str):
    figure.clear()
    figure.set_facecolor(BACKGROUND)
    font = _font()
    axis = figure.add_subplot(111, facecolor=BACKGROUND)
    axis.set_title(title, color=TEXT, pad=18, fontproperties=font)
    axis.tick_params(colors=MUTED)
    for spine in axis.spines.values():
        spine.set_color("#304866")
    return axis, font


def _aitoff_axis(figure, title: str):
    figure.clear()
    figure.set_facecolor(BACKGROUND)
    font = _font()
    axis = figure.add_subplot(111, projection="aitoff", facecolor=BACKGROUND)
    axis.set_title(title, color=TEXT, pad=18, fontproperties=font)
    axis.grid(True, color="#53657E", alpha=0.35, linewidth=0.55)
    ticks = np.deg2rad(np.arange(-150, 151, 30))
    labels = ["14h", "16h", "18h", "20h", "22h", "0h", "2h", "4h", "6h", "8h", "10h"]
    axis.set_xticks(ticks)
    axis.set_xticklabels(labels, color=MUTED, fontsize=8)
    axis.tick_params(axis="y", colors=MUTED, labelsize=8)
    return axis, font


def _legend_handles(results: Iterable[analysis.RadiantResult], line_labels: bool = True):
    from matplotlib.lines import Line2D

    handles = []
    if line_labels:
        handles.extend([
            Line2D([0], [0], color=TEXT, linewidth=2.0, label="実際の流星経路"),
            Line2D([0], [0], color=TEXT, linewidth=1.2, linestyle="--", label="放射点までの天球投影"),
        ])
    seen = set()
    for result in results:
        if result.shower_code in seen or result.radiant_radec is None:
            continue
        handles.append(Line2D(
            [0], [0], marker="o", linestyle="None", color=_color(result),
            markerfacecolor=_color(result), label=f"{result.shower_code} {result.shower_name}",
        ))
        seen.add(result.shower_code)
    return handles


def _plot_aitoff_result(axis, result: analysis.RadiantResult, linewidth: float = 1.8, alpha: float = 0.9):
    color = _color(result)
    ra, dec = _path_radec(result)
    for lon, lat in _split_wrap(np.asarray([_lon_deg(value) for value in ra]), dec):
        axis.plot(np.deg2rad(lon), np.deg2rad(lat), color=color, linewidth=linewidth, alpha=alpha)
    extension = _extension_radec(result)
    if extension is not None:
        ext_ra, ext_dec = extension
        for lon, lat in _split_wrap(np.asarray([_lon_deg(value) for value in ext_ra]), ext_dec):
            axis.plot(np.deg2rad(lon), np.deg2rad(lat), color=color, linewidth=max(0.8, linewidth * 0.65), linestyle="--", alpha=alpha * 0.9)
        radiant_ra, radiant_dec = result.radiant_radec
        axis.scatter(np.deg2rad(_lon_deg(radiant_ra)), np.deg2rad(radiant_dec), color=color, s=28, zorder=5)


def draw_aitoff_map(report: analysis.RadiantReport, figure=None):
    """Draw an all-sky Aitoff map with meteor paths and radiants."""
    from matplotlib.figure import Figure

    if figure is None:
        figure = Figure(figsize=(12, 7), facecolor=BACKGROUND)
    axis, font = _aitoff_axis(
        figure,
        f"全天Aitoff図 | 有効流星 {len(report.supported_results)} / 読込 {len(report.results) + len(report.skipped)} | {report.model_label}",
    )
    for result in report.supported_results:
        _plot_aitoff_result(axis, result)
    axis.legend(handles=_legend_handles(report.supported_results), loc="upper right", facecolor=PANEL, edgecolor="#415875", labelcolor=TEXT, prop=font, fontsize=8)
    figure.tight_layout()
    return figure, axis


def _great_circle_radec(result: analysis.RadiantResult, count: int = 360) -> Tuple[np.ndarray, np.ndarray]:
    start = analysis.radec_to_unit_vector(*result.start_radec)
    end = analysis.radec_to_unit_vector(*result.end_radec)
    normal = np.cross(start, end)
    normal = normal / np.linalg.norm(normal)
    tangent = np.cross(normal, start)
    tangent = tangent / np.linalg.norm(tangent)
    angles = np.linspace(0.0, 2.0 * np.pi, count)
    points = np.cos(angles)[:, None] * start[None, :] + np.sin(angles)[:, None] * tangent[None, :]
    return _radec_from_vectors(points)


def draw_convergence_map(report: analysis.RadiantReport, figure=None):
    """Draw full great circles so their radiant-side convergence is visible."""
    from matplotlib.figure import Figure

    if figure is None:
        figure = Figure(figsize=(12, 7), facecolor=BACKGROUND)
    axis, font = _aitoff_axis(figure, f"放射点大円収束図 | {len(report.supported_results)}件")
    for result in report.supported_results:
        ra, dec = _great_circle_radec(result)
        color = _color(result)
        for lon, lat in _split_wrap(np.asarray([_lon_deg(value) for value in ra]), dec):
            axis.plot(np.deg2rad(lon), np.deg2rad(lat), color=color, linewidth=0.45, alpha=0.18)
        _plot_aitoff_result(axis, result, linewidth=1.0, alpha=0.72)
    axis.legend(handles=_legend_handles(report.supported_results), loc="upper right", facecolor=PANEL, edgecolor="#415875", labelcolor=TEXT, prop=font, fontsize=8)
    figure.tight_layout()
    return figure, axis


def _load_first_background(info_paths: Sequence[str]) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]], Dict[str, str]]:
    for info_path in info_paths:
        data = analysis._load_data(info_path)
        for key in ("Saved Composite Path", "Saved Annotated Path", "Saved Full Diff Path"):
            path = str(data.get(key, "")).strip()
            if not path or not os.path.isfile(path):
                continue
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is not None:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Lift the very dark night image without destroying the stars.
                rgb = np.clip(rgb.astype(np.float32) * 1.35 + 3.0, 0, 255).astype(np.uint8)
                return rgb, (image.shape[1], image.shape[0]), data
    return None, None, {}


def _scale_pixel(point: Tuple[float, float], from_size: Tuple[int, int], to_size: Tuple[int, int]) -> Tuple[float, float]:
    return point[0] * to_size[0] / from_size[0], point[1] * to_size[1] / from_size[1]


def _camera_line_from_info(info_path: str, target_size: Tuple[int, int]) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    data = analysis._load_data(info_path)
    line = analysis.parse_pixel_line(data)
    if line is None:
        return None
    start, end, _source = line
    image = None
    for key in ("Saved Annotated Path", "Saved Composite Path", "Saved Full Diff Path"):
        path = str(data.get(key, "")).strip()
        if path and os.path.isfile(path):
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                break
    if image is None:
        return start, end
    original_size = (image.shape[1], image.shape[0])
    return _scale_pixel(start, original_size, target_size), _scale_pixel(end, original_size, target_size)


def _radiant_pixel(result: analysis.RadiantResult, metadata: Dict[str, Any], wcs: Any) -> Optional[Tuple[float, float]]:
    if result.radiant_radec is None or result.detection_time is None:
        return None
    reference_value = metadata.get("reference_datetime")
    reference = analysis._parse_iso_datetime(str(reference_value)) if reference_value else None
    if reference is None:
        return None
    delta_ra = analysis._sidereal_delta_ra(reference, result.detection_time)
    model_ra = (result.radiant_radec[0] - delta_ra) % 360.0
    try:
        x, y = wcs.world_to_pixel_values(model_ra, result.radiant_radec[1])
        return float(x), float(y)
    except Exception:
        return None


def _camera_extension_pixel_chunks(
    result: analysis.RadiantResult,
    metadata: Dict[str, Any],
    wcs: Any,
    count: int = 64,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Project the radiant-side great-circle extension into camera pixels.

    The fixed-camera model is defined at a reference sidereal time, while the
    radiant and detected endpoints are expressed at the event time.  Move all
    sky coordinates into that reference frame first, interpolate on the unit
    sphere, and only then apply the WCS.  This preserves the camera's
    wide-angle distortion instead of replacing the projected path with one
    straight pixel chord.

    Matplotlib can clip finite points that are outside the image, but a broken
    projection (or a non-finite WCS result) must not connect two unrelated
    pieces.  Such gaps are returned as separate chunks.
    """
    if result.radiant_radec is None or result.detection_time is None:
        return []
    reference_value = metadata.get("reference_datetime")
    reference = analysis._parse_iso_datetime(str(reference_value)) if reference_value else None
    if reference is None:
        return []

    delta_ra = analysis._sidereal_delta_ra(reference, result.detection_time)

    def model_radec(radec: Tuple[float, float]) -> Tuple[float, float]:
        return ((float(radec[0]) - delta_ra) % 360.0, float(radec[1]))

    radiant_model = model_radec(result.radiant_radec)
    endpoint_radec = result.start_radec if result.radiant_side == "start" else result.end_radec
    endpoint_model = model_radec(endpoint_radec)
    radiant_vector = analysis.radec_to_unit_vector(*radiant_model)
    endpoint_vector = analysis.radec_to_unit_vector(*endpoint_model)
    first_vector, second_vector = (
        (radiant_vector, endpoint_vector)
        if result.radiant_side == "start"
        else (endpoint_vector, radiant_vector)
    )
    points = analysis.slerp_vectors(
        first_vector,
        second_vector,
        count,
    )
    ra, dec = _radec_from_vectors(points)
    try:
        x, y = wcs.world_to_pixel_values(ra, dec)
    except Exception:
        return []
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return []

    # Split only at invalid samples or a projection jump.  Points outside the
    # image are retained so Matplotlib can correctly clip a curve as it enters
    # or leaves the camera frame.
    chunks: List[Tuple[np.ndarray, np.ndarray]] = []
    start = None
    previous = None
    width = float(metadata.get("width", getattr(wcs, "width", 1920)))
    height = float(metadata.get("height", getattr(wcs, "height", 1080)))
    jump_limit = 2.0 * max(width, height)
    for index, valid in enumerate(finite):
        if not valid:
            if start is not None and index - start >= 2:
                chunks.append((x[start:index], y[start:index]))
            start = None
            previous = None
            continue
        current = np.array([x[index], y[index]], dtype=float)
        if start is None:
            start = index
        elif previous is not None and float(np.linalg.norm(current - previous)) > jump_limit:
            if index - start >= 2:
                chunks.append((x[start:index], y[start:index]))
            start = index
        previous = current
    if start is not None and len(x) - start >= 2:
        chunks.append((x[start:], y[start:]))
    return chunks


def draw_camera_overlay(
    report: analysis.RadiantReport,
    info_paths: Sequence[str],
    figure=None,
):
    """Draw all detected lines and radiant extensions in camera pixels."""
    from matplotlib.figure import Figure
    import local_wideangle_astrometry

    background, image_size, _data = _load_first_background(info_paths)
    metadata, wcs = local_wideangle_astrometry._load_calibration(report.model_path)
    if image_size is None:
        image_size = (int(metadata.get("width", 1920)), int(metadata.get("height", 1080)))
    width, height = image_size
    if figure is None:
        figure = Figure(figsize=(12, 7), facecolor=BACKGROUND)
    axis, font = _configure_dark_figure(
        figure,
        f"カメラ画像投影 | 有効 {len(report.results)}件 / 除外 {len(report.skipped)}件 | {report.model_label}",
    )
    if background is not None:
        axis.imshow(background, extent=(0, width, height, 0), interpolation="nearest")
    else:
        axis.set_facecolor("#111827")
    grid = local_wideangle_astrometry._forward_grid_model(wcs, metadata, width, height)
    support = np.asarray(grid["support_mask"], dtype=np.uint8)
    if support.shape == (height, width):
        axis.contour(support, levels=[0.5], colors="#5DE2A5", linewidths=1.2, alpha=0.8, origin="upper", extent=(0, width, height, 0))
    result_by_path = {str(Path(result.info_path).resolve()): result for result in report.results}
    for info_path in info_paths:
        line = _camera_line_from_info(info_path, image_size)
        if line is None:
            continue
        start, end = line
        result = result_by_path.get(str(Path(info_path).resolve()))
        if result is None:
            axis.plot([start[0], end[0]], [start[1], end[1]], color="#E86B7A", linewidth=0.8, alpha=0.35, linestyle=":")
            continue
        color = _color(result)
        axis.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=1.7, alpha=0.88)
        radiant = _radiant_pixel(result, metadata, wcs)
        if radiant is not None:
            for extension_x, extension_y in _camera_extension_pixel_chunks(result, metadata, wcs):
                axis.plot(
                    extension_x,
                    extension_y,
                    color=color,
                    linewidth=1.0,
                    linestyle="--",
                    alpha=0.85,
                    clip_on=True,
                )
            axis.scatter([radiant[0]], [radiant[1]], color=color, s=30, edgecolors="#FFFFFF", linewidths=0.4, zorder=5)
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    axis.set_xlabel("pixel X", color=MUTED)
    axis.set_ylabel("pixel Y", color=MUTED)
    axis.legend(handles=_legend_handles(report.results) + [], loc="upper right", facecolor=PANEL, edgecolor="#415875", labelcolor=TEXT, prop=font, fontsize=8)
    figure.tight_layout()
    return figure, axis


def draw_radec_map(report: analysis.RadiantReport, figure=None):
    """Draw a rectangular RA/Dec plot for numeric inspection."""
    from matplotlib.figure import Figure

    if figure is None:
        figure = Figure(figsize=(12, 7), facecolor=BACKGROUND)
    axis, font = _configure_dark_figure(figure, f"RA-Dec分布 | 有効流星 {len(report.results)}件")
    for result in report.supported_results:
        color = _color(result)
        ra, dec = _path_radec(result)
        axis.plot(ra, dec, color=color, linewidth=1.2, alpha=0.72)
        extension = _extension_radec(result)
        if extension is not None:
            ext_ra, ext_dec = extension
            axis.plot(ext_ra, ext_dec, color=color, linewidth=0.9, linestyle="--", alpha=0.75)
            axis.scatter([result.radiant_radec[0]], [result.radiant_radec[1]], color=color, s=28)
    axis.set_xlim(360, 0)
    axis.set_ylim(-90, 90)
    axis.set_xlabel("Right Ascension (deg)", color=MUTED)
    axis.set_ylabel("Declination (deg)", color=MUTED)
    axis.set_xticks(np.arange(0, 361, 30))
    axis.set_yticks(np.arange(-90, 91, 15))
    axis.grid(True, color="#53657E", alpha=0.28)
    axis.legend(handles=_legend_handles(report.supported_results), loc="upper right", facecolor=PANEL, edgecolor="#415875", labelcolor=TEXT, prop=font, fontsize=8)
    figure.tight_layout()
    return figure, axis


def draw_density_heatmap(report: analysis.RadiantReport, figure=None):
    """Show where the classified shower radiants concentrate."""
    from matplotlib.figure import Figure

    if figure is None:
        figure = Figure(figsize=(12, 7), facecolor=BACKGROUND)
    axis, font = _configure_dark_figure(figure, f"放射点密度ヒートマップ | 有効流星 {len(report.results)}件")
    points = [result.radiant_radec for result in report.supported_results if result.radiant_radec is not None]
    if points:
        ra = np.asarray([point[0] for point in points])
        dec = np.asarray([point[1] for point in points])
        hist, ra_edges, dec_edges = np.histogram2d(ra, dec, bins=(72, 36), range=((0, 360), (-90, 90)))
        image = axis.imshow(
            hist.T,
            extent=(0, 360, -90, 90),
            origin="lower",
            aspect="auto",
            cmap="magma",
            alpha=0.9,
        )
        figure.colorbar(image, ax=axis, pad=0.02, label="radiant count")
        for result in report.supported_results:
            if result.radiant_radec is None:
                continue
            axis.scatter([result.radiant_radec[0]], [result.radiant_radec[1]], color=_color(result), s=12, alpha=0.45)
    axis.set_xlim(360, 0)
    axis.set_ylim(-90, 90)
    axis.set_xlabel("Right Ascension (deg)", color=MUTED)
    axis.set_ylabel("Declination (deg)", color=MUTED)
    axis.grid(True, color="#FFFFFF", alpha=0.14)
    axis.legend(handles=_legend_handles(report.supported_results, line_labels=False), loc="upper right", facecolor=PANEL, edgecolor="#415875", labelcolor=TEXT, prop=font, fontsize=8)
    figure.tight_layout()
    return figure, axis


def _radec_polar_points(ra_deg: Sequence[float], dec_deg: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert fixed equatorial coordinates to an NCP-centered polar plot.

    The radial coordinate is co-declination (90° - Dec): the north celestial
    pole is at the center and the south celestial pole is at the outer rim.
    RA is unwrapped along each sampled path so a path crossing 0h remains a
    short continuous curve instead of being joined across the whole plot.
    """
    ra = np.asarray(ra_deg, dtype=float).reshape(-1)
    dec = np.asarray(dec_deg, dtype=float).reshape(-1)
    return np.unwrap(np.deg2rad(ra)), 90.0 - dec


def _radec_polar_result_path(result: analysis.RadiantResult, count: int = 96) -> Tuple[np.ndarray, np.ndarray]:
    ra, dec = _path_radec(result, count=count)
    return _radec_polar_points(ra, dec)


def draw_radec_polar(report: analysis.RadiantReport, figure=None):
    """Draw fixed RA/Dec on a polar chart centered on the north celestial pole."""
    from matplotlib.figure import Figure

    if figure is None:
        figure = Figure(figsize=(9, 9), facecolor=BACKGROUND)
    figure.clear()
    font = _font()
    axis = figure.add_subplot(111, projection="polar", facecolor=BACKGROUND)
    axis.set_title(
        "RA-Dec極座標図 | 中心=北天極 / 外周=南天極 | 時刻によらない天球基準",
        color=TEXT,
        pad=24,
        fontproperties=font,
    )
    axis.set_theta_zero_location("N")
    # Astronomical charts conventionally increase RA toward the left.
    axis.set_theta_direction(1)
    axis.set_rlim(0, 180)
    declinations = np.arange(60, -91, -30)
    radii = 90.0 - declinations
    axis.set_yticks(radii)
    axis.set_yticklabels([f"{value:+.0f}°" for value in declinations])
    ra_ticks = np.arange(0, 360, 30)
    axis.set_xticks(np.deg2rad(ra_ticks))
    axis.set_xticklabels([f"{int(value / 15):d}h" for value in ra_ticks])
    axis.set_rlabel_position(225)
    axis.tick_params(colors=MUTED, labelsize=8)
    axis.grid(color="#53657E", alpha=0.35)
    for result in report.supported_results:
        color = _color(result)
        path_theta, path_radius = _radec_polar_result_path(result, count=96)
        axis.plot(path_theta, path_radius, color=color, linewidth=1.5, alpha=0.8)

        extension = _extension_radec(result, count=128)
        if extension is not None:
            ext_ra, ext_dec = extension
            ext_theta, ext_radius = _radec_polar_points(ext_ra, ext_dec)
            axis.plot(ext_theta, ext_radius, color=color, linewidth=1.0, linestyle="--", alpha=0.8)
            radiant_theta, radiant_radius = _radec_polar_points(
                [result.radiant_radec[0]], [result.radiant_radec[1]]
            )
            axis.scatter(radiant_theta, radiant_radius, color=color, s=22)
    axis.legend(handles=_legend_handles(report.supported_results), loc="lower right", bbox_to_anchor=(0.98, 0.02), facecolor=PANEL, edgecolor="#415875", labelcolor=TEXT, prop=font, fontsize=8)
    figure.subplots_adjust(top=0.86, bottom=0.07, left=0.08, right=0.92)
    return figure, axis


def draw_horizon_polar(
    report: analysis.RadiantReport,
    figure=None,
    latitude_deg: float = DEFAULT_LATITUDE_DEG,
    longitude_deg: float = DEFAULT_LONGITUDE_DEG,
):
    """Backward-compatible name for the fixed RA/Dec polar visualization."""
    del latitude_deg, longitude_deg
    return draw_radec_polar(report, figure=figure)


def save_sphere_rotation_gif(
    report: analysis.RadiantReport,
    output_path: str,
    fps: int = 12,
    frames: int = 73,
) -> str:
    """Save one complete azimuth rotation of the 3-D radiant sphere."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    # Include both 0° and 360° so the saved GIF contains a complete rotation
    # and loops without a visible angular jump.
    frame_count = max(2, int(frames))
    frame_rate = max(1, int(fps))
    figure = Figure(figsize=(10, 8), facecolor=BACKGROUND)
    FigureCanvasAgg(figure)
    _figure, axis = analysis.draw_radiant_sphere(report, figure=figure)
    initial_azimuth = -58.0

    def update(frame_index: int):
        azimuth = initial_azimuth + 360.0 * float(frame_index) / (frame_count - 1)
        axis.view_init(elev=22, azim=azimuth)
        return (axis,)

    animation = FuncAnimation(
        figure,
        update,
        frames=frame_count,
        interval=1000.0 / frame_rate,
        blit=False,
        repeat=False,
    )
    animation.save(
        output_path,
        writer=PillowWriter(fps=frame_rate),
        dpi=100,
    )
    figure.clear()
    return output_path


def save_figure(drawer, report: analysis.RadiantReport, output_path: str, *args, **kwargs) -> str:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=kwargs.pop("figsize", (12, 7)), facecolor=BACKGROUND)
    drawer(report, figure=figure, *args, **kwargs)
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    canvas.print_figure(output_path, dpi=150, facecolor=figure.get_facecolor())
    figure.clear()
    return output_path


def save_timeline_animation(
    report: analysis.RadiantReport,
    output_path: str,
    fps: int = 6,
) -> str:
    """Save a cumulative Aitoff animation ordered by detection time."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.animation import FFMpegWriter, FuncAnimation
    from matplotlib.figure import Figure

    results = sorted(
        report.supported_results,
        key=lambda result: result.detection_time or datetime.min,
    )
    figure = Figure(figsize=(12, 7), facecolor=BACKGROUND)
    FigureCanvasAgg(figure)
    axis, font = _aitoff_axis(figure, "")
    lines = []
    extensions = []
    markers = []
    for result in results:
        color = _color(result)
        ra, dec = _path_radec(result)
        data = [(np.deg2rad(lon), np.deg2rad(lat)) for lon, lat in _split_wrap(np.asarray([_lon_deg(value) for value in ra]), dec)]
        line_artists = [axis.plot(x, y, color=color, linewidth=1.7, alpha=0.9, visible=False)[0] for x, y in data]
        lines.append(line_artists)
        ext_artists = []
        extension = _extension_radec(result)
        if extension is not None:
            ext_ra, ext_dec = extension
            ext_data = [(np.deg2rad(lon), np.deg2rad(lat)) for lon, lat in _split_wrap(np.asarray([_lon_deg(value) for value in ext_ra]), ext_dec)]
            ext_artists = [axis.plot(x, y, color=color, linewidth=1.0, linestyle="--", alpha=0.8, visible=False)[0] for x, y in ext_data]
        extensions.append(ext_artists)
        if result.radiant_radec is not None:
            marker = axis.scatter([], [], color=color, s=24, visible=False)
        else:
            marker = None
        markers.append(marker)
    title = axis.set_title("", color=TEXT, pad=18, fontproperties=font)
    axis.legend(handles=_legend_handles(results), loc="upper right", facecolor=PANEL, edgecolor="#415875", labelcolor=TEXT, prop=font, fontsize=8)
    figure.tight_layout()

    def update(frame_index):
        for index, result in enumerate(results):
            visible = index <= frame_index
            for artist in lines[index] + extensions[index]:
                artist.set_visible(visible)
            marker = markers[index]
            if marker is not None:
                marker.set_visible(visible)
                if visible:
                    marker.set_offsets([[np.deg2rad(_lon_deg(result.radiant_radec[0])), np.deg2rad(result.radiant_radec[1])]])
        current = results[frame_index] if results else None
        if current is not None and current.detection_time is not None:
            title.set_text(f"放射点時系列 | {current.detection_time.isoformat()} | {frame_index + 1}/{len(results)}件")
        return [artist for group in lines + extensions for artist in group] + [marker for marker in markers if marker is not None] + [title]

    animation = FuncAnimation(figure, update, frames=max(1, len(results)), interval=1000 / max(1, fps), blit=False, repeat=False)
    writer = FFMpegWriter(fps=fps, metadata={"title": "Meteor radiant timeline"}, bitrate=1800)
    animation.save(output_path, writer=writer, dpi=120)
    figure.clear()
    return output_path


def save_visualization_bundle(
    report: analysis.RadiantReport,
    info_paths: Sequence[str],
    output_dir: str,
    prefix: str = "radiant_analysis",
    latitude_deg: float = DEFAULT_LATITUDE_DEG,
    longitude_deg: float = DEFAULT_LONGITUDE_DEG,
) -> Dict[str, str]:
    """Save every alternative visualization and return their paths."""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "sphere": str(output / f"{prefix}_sphere.png"),
        "sphere_rotation_gif": str(output / f"{prefix}_sphere_rotation.gif"),
        "aitoff": str(output / f"{prefix}_aitoff.png"),
        "convergence": str(output / f"{prefix}_convergence.png"),
        "camera_overlay": str(output / f"{prefix}_camera_overlay.png"),
        "radec": str(output / f"{prefix}_radec.png"),
        "density": str(output / f"{prefix}_density.png"),
        "horizon_polar": str(output / f"{prefix}_horizon_polar.png"),
        "timeline": str(output / f"{prefix}_timeline.mp4"),
    }
    save_figure(analysis.draw_radiant_sphere, report, paths["sphere"], figsize=(10, 8))
    save_sphere_rotation_gif(report, paths["sphere_rotation_gif"])
    save_figure(draw_aitoff_map, report, paths["aitoff"])
    save_figure(draw_convergence_map, report, paths["convergence"])
    save_figure(draw_camera_overlay, report, paths["camera_overlay"], info_paths)
    save_figure(draw_radec_map, report, paths["radec"])
    save_figure(draw_density_heatmap, report, paths["density"])
    save_figure(draw_horizon_polar, report, paths["horizon_polar"], latitude_deg=latitude_deg, longitude_deg=longitude_deg, figsize=(9, 9))
    save_timeline_animation(report, paths["timeline"])
    return paths
