"""Compare candidate camera models against raw-night stars and geometry."""

from datetime import datetime, timedelta
import json
from pathlib import Path

import cv2
import numpy as np

import local_wideangle_astrometry as astrometry
import timelapse_creator


ROOT = Path(__file__).resolve().parent
MODELS = {
    "trajectory_degree5": Path(
        "/Users/keisukematsumoto/Library/Caches/MeteorDetector/astrometry/"
        "camera_models/trajectory-68ebd9385bc5-20260813-1920x1080/camera_model.json"
    ),
    "stable_mosaic": Path(
        "/Users/keisukematsumoto/Library/Caches/MeteorDetector/astrometry/"
        "camera_models/current-tripod-mosaic-stable-20260813/camera_model.json"
    ),
}
REPORT = Path(
    "/Users/keisukematsumoto/Downloads/"
    "20260812-20260813_constellation_model_comparison.json"
)


def _mutual_matches(projected: np.ndarray, observed: np.ndarray, radius: float):
    if not len(projected) or not len(observed):
        return np.zeros(len(projected), dtype=bool), np.asarray([], dtype=float)
    distances = np.linalg.norm(
        projected[:, np.newaxis, :] - observed[np.newaxis, :, :], axis=2
    )
    nearest_observed = np.argmin(distances, axis=1)
    nearest_projected = np.argmin(distances, axis=0)
    indices = np.arange(len(projected))
    values = distances[indices, nearest_observed]
    matched = (values <= radius) & (nearest_projected[nearest_observed] == indices)
    return matched, values[matched]


def _project(model, vertices, timestamp, reference, support):
    delta = (timestamp - reference).total_seconds() / astrometry.SIDEREAL_DAY_SECONDS * 360.0
    x, y, usable = astrometry._project_sky_with_forward_wcs(
        model, (vertices[:, 0] - delta) % 360.0, vertices[:, 1]
    )
    points = np.column_stack((x, y))
    height, width = support.shape
    usable &= np.isfinite(points).all(axis=1)
    usable &= (
        (points[:, 0] >= 0) & (points[:, 0] < width)
        & (points[:, 1] >= 0) & (points[:, 1] < height)
    )
    rounded = np.zeros_like(points, dtype=int)
    valid = np.flatnonzero(usable)
    rounded[valid] = np.rint(points[valid]).astype(int)
    usable &= (
        (rounded[:, 0] >= 0) & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0) & (rounded[:, 1] < height)
    )
    valid = np.flatnonzero(usable)
    usable[valid] &= support[rounded[valid, 1], rounded[valid, 0]] > 0
    return points[usable]


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
    selected = list(range(0, 450, 30)) + [449]
    observations = []
    for output_frame in selected:
        global_index = sample_indices[output_frame]
        timestamp = loader.timestamp_for_index(global_index)
        mean = loader.load_temporal_mean_frame(global_index, (1920, 1080), radius=50)
        gray = cv2.cvtColor(mean, cv2.COLOR_BGR2GRAY)
        stars, _diagnostic, _reference = astrometry._extract_stars(
            gray, maximum_stars=1000, exclude_lower_region=False,
            build_diagnostic=False,
        )
        observations.append((output_frame, timestamp, np.asarray(stars, dtype=float)))
    loader.cleanup()

    vertices = np.concatenate(astrometry._load_constellation_lines(), axis=0)
    report = {}
    for label, path in MODELS.items():
        metadata, model = astrometry._load_calibration(str(path))
        reference = datetime.fromisoformat(
            str(metadata["reference_datetime"]).replace("Z", "+00:00")
        )
        grid = astrometry._forward_grid_model(model, metadata, 1920, 1080)
        support = grid["support_mask"]
        rows = []
        for output_frame, timestamp, observed in observations:
            projected = _project(model, vertices, timestamp, reference, support)
            correct, residuals = _mutual_matches(projected, observed, 4.0)
            control = _project(
                model, vertices, timestamp + timedelta(minutes=5), reference, support
            )
            control_matches, _ = _mutual_matches(control, observed, 4.0)
            rows.append({
                "output_frame": output_frame,
                "timestamp": timestamp.isoformat(),
                "projected_count": len(projected),
                "correct_matches": int(np.count_nonzero(correct)),
                "control_matches": int(np.count_nonzero(control_matches)),
                "residual_median_px": float(np.median(residuals)) if len(residuals) else None,
                "residual_p95_px": float(np.percentile(residuals, 95)) if len(residuals) else None,
            })
        useful_residuals = [
            row["residual_p95_px"] for row in rows
            if row["correct_matches"] >= 2 and row["residual_p95_px"] is not None
        ]
        report[label] = {
            "path": str(path),
            "support_fraction": float(np.mean(support > 0)),
            "correct_match_total": sum(row["correct_matches"] for row in rows),
            "control_match_total": sum(row["control_matches"] for row in rows),
            "worst_useful_p95_px": max(useful_residuals) if useful_residuals else None,
            "rows": rows,
        }
        print(label, json.dumps({k: v for k, v in report[label].items() if k != "rows"}), flush=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
