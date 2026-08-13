from datetime import datetime, timedelta

import cv2
import numpy as np

from camera_plate_model import FixedCameraPlateModel
from trajectory_camera_model import (
    _Track,
    _trajectory_support_grid,
    fit_trajectory_projection,
    track_stellar_trajectories,
)


def test_tracker_keeps_moving_stars_and_rejects_fixed_pixels():
    frames = []
    times = []
    origin = datetime(2026, 8, 13, 0, 0, 0)
    moving = np.asarray([[30.0, 25.0], [75.0, 45.0], [120.0, 70.0], [155.0, 35.0]])
    for index in range(30):
        image = np.zeros((100, 200), dtype=np.uint8)
        for x, y in moving:
            cv2.circle(image, (round(x + index * 0.35), round(y + index * 0.08)), 2, 240, -1)
        # A hot pixel remains fixed and must not become a stellar trajectory.
        cv2.circle(image, (185, 90), 1, 255, -1)
        frames.append(image)
        times.append(origin + timedelta(seconds=index * 10))
    tracks = track_stellar_trajectories(
        frames, times, maximum_features=20, minimum_observations=20,
        minimum_span_seconds=150, reseed_interval=6,
    )
    assert len(tracks) >= 3
    assert all(np.linalg.norm(np.asarray(track.points)[-1] - np.asarray(track.points)[0]) > 1.5 for track in tracks)


def test_support_grid_requires_distinct_good_tracks():
    class Track:
        def __init__(self, points):
            self.points = points

    tracks = [
        Track([[10, 10], [20, 10]]),
        Track([[11, 11], [21, 11]]),
        Track([[12, 12], [22, 12]]),
        Track([[80, 80], [90, 80]]),
    ]
    grid, fraction, counts = _trajectory_support_grid(
        tracks, np.asarray([1.0, 1.5, 2.0, 1.0]), 100, 100, 2, 2, 3,
    )
    assert grid == [[1, 0], [0, 0]]
    assert fraction == 0.25
    assert counts[0, 0] == 3


def test_bundle_fit_predicts_unseen_trajectory_observations():
    width, height = 320, 180
    payload = {
        "model_type": "fixed-camera-stg-poly",
        "width": width,
        "height": height,
        "polynomial_degree": 2,
        "stg_parameters": [0.15, -0.22, 0.08, np.log(150), np.log(148), 160, 90],
        "correction_coefficients": np.zeros((6, 2)).tolist(),
        "reference_datetime": "2026-08-13T00:00:00",
        "support_grid": [[1] * 8 for _ in range(5)],
    }
    model = FixedCameraPlateModel(payload)
    origin = datetime(2026, 8, 13, 0, 0, 0)
    times = [origin + timedelta(seconds=index * 30) for index in range(20)]
    rng = np.random.default_rng(8)
    tracks = []
    for identifier, (ra, dec) in enumerate(
        (pair for pair in zip(np.linspace(5, 345, 24), np.linspace(-55, 60, 24)))
    ):
        points = []
        for timestamp in times:
            delta = (timestamp - origin).total_seconds() / 86164.0905 * 360
            x, y = model.world_to_pixel_values((ra - delta) % 360, dec)
            points.append([float(x + rng.normal(0, 0.05)), float(y + rng.normal(0, 0.05))])
        tracks.append(_Track(identifier, list(range(len(times))), [index * 30 for index in range(len(times))], points))
    _fitted, stats = fit_trajectory_projection(tracks, times, payload, degree=2, iterations=2)
    assert np.percentile(stats["train_residual"], 95) < 0.25
    assert np.percentile(stats["holdout_residual"], 95) < 0.25
