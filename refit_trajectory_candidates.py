"""Refit lower-order trajectory candidates and audit projection regularity."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import cv2
import numpy as np

from camera_plate_model import FixedCameraPlateModel
import local_wideangle_astrometry as astrometry
import trajectory_camera_model as trajectory


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "rtsp" / "20260813"
SEED = Path(
    "/Users/keisukematsumoto/Library/Caches/MeteorDetector/astrometry/"
    "camera_models/current-tripod-mosaic-stable-20260813/camera_model.json"
)
OUTPUT = Path("/Users/keisukematsumoto/Downloads/trajectory-refit-candidates")


def _pixel_roundtrip(model: FixedCameraPlateModel, support_grid: list[list[int]]) -> dict:
    support = cv2.resize(
        np.asarray(support_grid, dtype=np.uint8),
        (model.width, model.height),
        interpolation=cv2.INTER_NEAREST,
    )
    xs = np.linspace(10, model.width - 11, 96)
    ys = np.linspace(10, model.height - 11, 54)
    mesh_x, mesh_y = np.meshgrid(xs, ys)
    points = np.column_stack((mesh_x.ravel(), mesh_y.ravel()))
    rounded = np.rint(points).astype(int)
    usable = support[rounded[:, 1], rounded[:, 0]] > 0
    points = points[usable]
    if not len(points):
        return {
            "sample_count": 0,
            "median_px": None,
            "p95_px": None,
            "maximum_px": None,
            "over_5px_fraction": None,
            "over_20px_fraction": None,
        }
    ra, dec = model.pixel_to_world_values(points[:, 0], points[:, 1])
    back_x, back_y = model.world_to_pixel_values(ra, dec)
    errors = np.linalg.norm(np.column_stack((back_x, back_y)) - points, axis=1)
    finite = errors[np.isfinite(errors)]
    return {
        "sample_count": int(len(finite)),
        "median_px": float(np.median(finite)),
        "p95_px": float(np.percentile(finite, 95)),
        "maximum_px": float(np.max(finite)),
        "over_5px_fraction": float(np.mean(finite > 5.0)),
        "over_20px_fraction": float(np.mean(finite > 20.0)),
    }


def _constellation_scale(model: FixedCameraPlateModel, reference: datetime) -> dict:
    asset = json.loads(
        (ROOT / "assets" / "constellations.lines.json").read_text(encoding="utf-8")
    )
    rows = []
    for minute in (0, 45, 90, 180, 240):
        delta_ra = minute * 60.0 / astrometry.SIDEREAL_DAY_SECONDS * 360.0
        for feature in asset.get("features", []):
            constellation = str(feature.get("id", ""))
            for line_index, line in enumerate(feature.get("geometry", {}).get("coordinates", [])):
                coordinates = np.asarray(line, dtype=float)
                for segment_index in range(len(coordinates) - 1):
                    sky = coordinates[segment_index:segment_index + 2]
                    ra = (sky[:, 0] - delta_ra) % 360.0
                    x, y = model.world_to_pixel_values(ra, sky[:, 1])
                    pixels = np.column_stack((x, y))
                    if not np.isfinite(pixels).all():
                        continue
                    if not np.all(
                        (pixels[:, 0] >= 0) & (pixels[:, 0] < model.width)
                        & (pixels[:, 1] >= 0) & (pixels[:, 1] < model.height)
                    ):
                        continue
                    ra_rad = np.deg2rad(sky[:, 0])
                    dec_rad = np.deg2rad(sky[:, 1])
                    dot = (
                        np.sin(dec_rad[0]) * np.sin(dec_rad[1])
                        + np.cos(dec_rad[0]) * np.cos(dec_rad[1])
                        * np.cos(ra_rad[1] - ra_rad[0])
                    )
                    angular = float(np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0))))
                    pixel = float(np.linalg.norm(pixels[1] - pixels[0]))
                    if angular <= 1e-6:
                        continue
                    rows.append({
                        "minute": minute,
                        "constellation": constellation,
                        "line_index": line_index,
                        "segment_index": segment_index,
                        "angular_deg": angular,
                        "pixel_length": pixel,
                        "pixels_per_degree": pixel / angular,
                        "start": pixels[0].tolist(),
                        "end": pixels[1].tolist(),
                    })
    scales = np.asarray([row["pixels_per_degree"] for row in rows], dtype=float)
    longest = sorted(rows, key=lambda row: row["pixel_length"], reverse=True)[:20]
    return {
        "segment_sample_count": len(rows),
        "pixels_per_degree_median": float(np.median(scales)),
        "pixels_per_degree_p95": float(np.percentile(scales, 95)),
        "pixels_per_degree_maximum": float(np.max(scales)),
        "over_50_pixels_per_degree_count": int(np.count_nonzero(scales > 50.0)),
        "longest_projected_segments": longest,
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    seed_payload = json.loads(SEED.read_text(encoding="utf-8"))
    request = trajectory.TrajectoryBuildRequest(
        source=str(SOURCE),
        initial_model_path=str(SEED),
        start="00:00",
        end="00:45",
        sample_interval_seconds=10.0,
        maximum_frames=360,
        maximum_features=1200,
        minimum_track_observations=18,
        minimum_track_span_seconds=150.0,
        support_columns=24,
        support_rows=14,
        minimum_cell_tracks=2,
    )
    paths = trajectory._selected_paths(request)
    frames, timestamps, sources = trajectory._sample_video_frames(
        paths, request.sample_interval_seconds, request.maximum_frames, print,
    )
    raw_tracks = trajectory.track_stellar_trajectories(
        frames,
        timestamps,
        maximum_features=request.maximum_features,
        minimum_observations=request.minimum_track_observations,
        minimum_span_seconds=request.minimum_track_span_seconds,
        progress_callback=print,
    )
    tracks, seed_consistency = trajectory.filter_tracks_by_sidereal_consistency(
        raw_tracks, timestamps, seed_payload,
    )
    print(f"accepted tracks: {len(tracks)}/{len(raw_tracks)}", flush=True)
    summaries = []
    for degree in (2, 3, 4, 5):
        print(f"fitting degree {degree}", flush=True)
        payload, fit = trajectory.fit_trajectory_projection(
            tracks, timestamps, seed_payload, degree=degree,
        )
        support, support_fraction, support_counts = trajectory._trajectory_support_grid(
            tracks,
            fit["per_track_p95"],
            int(payload["width"]),
            int(payload["height"]),
            request.support_columns,
            request.support_rows,
            request.minimum_cell_tracks,
        )
        train = fit["train_residual"]
        holdout = fit["holdout_residual"]
        payload.update({
            "support_grid": support,
            "support_fraction": support_fraction,
            "model_label": f"TRAJECTORY REFIT DEGREE {degree} CANDIDATE",
            "enabled": False,
            "target_met": False,
            "constellation_render_policy": "model-verified-continuous",
            "constellation_support_policy": "strict-trajectory-support",
            "fit_stats": {
                "residual_median_px": float(np.median(train)),
                "residual_p95_px": float(np.percentile(train, 95)),
                "holdout_residual_p95_px": float(np.percentile(holdout, 95)),
                "sample_count": int(len(train)),
                "holdout_sample_count": int(len(holdout)),
                "trajectory_count": len(tracks),
            },
            "source_videos": sorted(set(sources)),
            "trajectory_validation": {
                "raw_track_count": len(raw_tracks),
                "track_count": len(tracks),
                "seed_consistency_p95_px": float(np.percentile(seed_consistency, 95)),
                "support_track_counts": support_counts.tolist(),
            },
        })
        model = FixedCameraPlateModel(payload)
        candidate_path = OUTPUT / f"camera_model_degree{degree}.json"
        candidate_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary = {
            "degree": degree,
            "fit_stats": payload["fit_stats"],
            "support_fraction": support_fraction,
            "pixel_roundtrip": _pixel_roundtrip(model, support),
            "constellation_projection": _constellation_scale(
                model, datetime.fromisoformat(payload["reference_datetime"])
            ),
        }
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    report = {
        "source": str(SOURCE),
        "seed": str(SEED),
        "summaries": summaries,
    }
    (OUTPUT / "candidate_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
