"""Validate constellation endpoints against stars used in the final timelapse."""

from datetime import timedelta
import json
from pathlib import Path

import cv2
import numpy as np

import local_wideangle_astrometry as astrometry
import timelapse_creator


ROOT = Path(__file__).resolve().parent
MODEL = Path(
    "/Users/keisukematsumoto/Library/Caches/MeteorDetector/astrometry/"
    "camera_models/trajectory-68ebd9385bc5-20260813-1920x1080/"
    "camera_model.json"
)
REPORT = Path(
    "/Users/keisukematsumoto/Downloads/"
    "20260812-20260813_constellation_alignment_report.json"
)


def _mutual_matches(projected: np.ndarray, observed: np.ndarray, radius: float):
    if not len(projected) or not len(observed):
        return np.zeros(len(projected), dtype=bool), np.asarray([], dtype=float)
    distances = np.linalg.norm(
        projected[:, np.newaxis, :] - observed[np.newaxis, :, :], axis=2,
    )
    nearest_observed = np.argmin(distances, axis=1)
    nearest_projected = np.argmin(distances, axis=0)
    indices = np.arange(len(projected))
    values = distances[indices, nearest_observed]
    matched = (
        (values <= radius)
        & (nearest_projected[nearest_observed] == indices)
    )
    return matched, values[matched]


def _project_vertices(wcs, sky_vertices, timestamp, reference, support_mask):
    delta_ra = (timestamp - reference).total_seconds() / astrometry.SIDEREAL_DAY_SECONDS * 360.0
    x, y, usable = astrometry._project_sky_with_forward_wcs(
        wcs, (sky_vertices[:, 0] - delta_ra) % 360.0, sky_vertices[:, 1],
    )
    points = np.column_stack((x, y))
    height, width = support_mask.shape
    usable &= np.isfinite(points).all(axis=1)
    usable &= (
        (points[:, 0] >= 0) & (points[:, 0] < width)
        & (points[:, 1] >= 0) & (points[:, 1] < height)
    )
    rounded = np.zeros_like(points, dtype=int)
    rounded[usable] = np.rint(points[usable]).astype(int)
    usable &= (
        (rounded[:, 0] >= 0) & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0) & (rounded[:, 1] < height)
    )
    valid = np.flatnonzero(usable)
    usable[valid] &= support_mask[rounded[valid, 1], rounded[valid, 0]] > 0
    return points[usable], usable


def main() -> None:
    timelapse_creator.OUTPUT_FPS = 15
    images, videos = [], []
    for source in (ROOT / "rtsp" / "20260812", ROOT / "rtsp" / "20260813"):
        found_images, found_videos = timelapse_creator.get_files_from_path(str(source))
        images.extend(found_images)
        videos.extend(found_videos)
    total, sources = timelapse_creator.count_total_frames(sorted(images), sorted(videos))
    sample_indices = timelapse_creator.calculate_sample_indices(total, 30)
    loader = timelapse_creator.FrameLoader(sources)
    metadata, wcs = astrometry._load_calibration(str(MODEL))
    reference = astrometry.datetime.fromisoformat(
        str(metadata["reference_datetime"]).replace("Z", "+00:00")
    )
    grid = astrometry._forward_grid_model(wcs, metadata, 1920, 1080)
    support_mask = grid["support_mask"]
    constellation_lines = astrometry._load_constellation_lines()
    lengths = [len(line) for line in constellation_lines]
    sky_vertices = np.concatenate(constellation_lines, axis=0)

    selected_output_frames = list(range(0, 450, 30)) + [449]
    rows = []
    for output_frame in selected_output_frames:
        global_index = sample_indices[output_frame]
        timestamp = loader.timestamp_for_index(global_index)
        mean = loader.load_temporal_mean_frame(
            global_index, (1920, 1080), radius=50,
        )
        if mean is None:
            continue
        gray = cv2.cvtColor(mean, cv2.COLOR_BGR2GRAY)
        stars, _diagnostic, _reference = astrometry._extract_stars(
            gray, maximum_stars=1000, exclude_lower_region=False,
            build_diagnostic=False,
        )
        observed = np.asarray(stars, dtype=float).reshape(-1, 2)
        projected, usable = _project_vertices(
            wcs, sky_vertices, timestamp, reference, support_mask,
        )
        correct_matches, residuals = _mutual_matches(projected, observed, 4.0)
        control_projected, _control_usable = _project_vertices(
            wcs, sky_vertices, timestamp + timedelta(minutes=5), reference, support_mask,
        )
        control_matches, _control_residuals = _mutual_matches(
            control_projected, observed, 4.0,
        )

        matched_original = np.zeros(len(sky_vertices), dtype=bool)
        matched_original[np.flatnonzero(usable)] = correct_matches
        drawable_edges = 0
        offset = 0
        for length in lengths:
            drawable_edges += sum(
                bool(matched_original[offset + index] and matched_original[offset + index + 1])
                for index in range(length - 1)
            )
            offset += length
        rows.append({
            "output_frame": output_frame,
            "timestamp": timestamp.isoformat(),
            "detected_star_count": int(len(observed)),
            "projected_constellation_vertex_count": int(len(projected)),
            "correct_mutual_match_count": int(np.sum(correct_matches)),
            "five_minute_control_match_count": int(np.sum(control_matches)),
            "matched_endpoint_residual_median_px": (
                float(np.median(residuals)) if len(residuals) else None
            ),
            "matched_endpoint_residual_p95_px": (
                float(np.percentile(residuals, 95)) if len(residuals) else None
            ),
            "drawable_constellation_edge_count": int(drawable_edges),
        })
        print(rows[-1], flush=True)
    loader.cleanup()

    useful = [row for row in rows if row["correct_mutual_match_count"] >= 2]
    all_residual_p95 = [
        row["matched_endpoint_residual_p95_px"] for row in useful
        if row["matched_endpoint_residual_p95_px"] is not None
    ]
    report = {
        "video": str(
            Path("/Users/keisukematsumoto/Downloads/")
            / "20260812-20260813_star-trajectory_constellations_timelapse.mp4"
        ),
        "model": str(MODEL),
        "checked_output_frame_count": len(rows),
        "frames_with_two_or_more_endpoint_matches": len(useful),
        "correct_match_total": int(sum(row["correct_mutual_match_count"] for row in rows)),
        "five_minute_control_match_total": int(sum(row["five_minute_control_match_count"] for row in rows)),
        "worst_useful_frame_endpoint_p95_px": max(all_residual_p95) if all_residual_p95 else None,
        "rows": rows,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
