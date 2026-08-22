import json
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from camera_model_builder import (
    CameraModelBuildRequest,
    _duration_seconds,
    _fit_model_from_wcs,
    _support_grid,
    automatic_model_source,
    build_camera_model,
    select_auto_video_paths,
    select_video_paths,
)
from camera_plate_model import FixedCameraPlateModel
from cloud_coverage import CloudClassification


def _simple_wcs(path: Path, width: int = 320, height: int = 180) -> str:
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [width / 2, height / 2]
    wcs.wcs.crval = [120.0, 35.0]
    wcs.wcs.cd = np.array([[-0.08, 0.012], [0.009, 0.08]])
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    header = wcs.to_header(relax=True)
    header["IMAGEW"] = width
    header["IMAGEH"] = height
    fits.PrimaryHDU(header=header).writeto(path, overwrite=True)
    return str(path)


def test_duration_seconds_and_support_grid():
    assert _duration_seconds("12") == 12
    assert _duration_seconds("01:02") == 62
    assert _duration_seconds("01:02:03") == 3723
    grid, fraction = _support_grid(100, 100, [[10, 10], [89, 10], [89, 89], [10, 89]], 10, 10)
    assert len(grid) == 10
    assert 0.6 < fraction < 0.9


def test_wcs_conversion_is_subpixel_on_sampled_field(tmp_path):
    wcs_path = _simple_wcs(tmp_path / "seed.wcs")
    with fits.open(wcs_path) as hdul:
        wcs = WCS(hdul[0].header, relax=True, fix=False)
    payload, stats = _fit_model_from_wcs(wcs, 320, 180)
    model = FixedCameraPlateModel(payload)
    x = np.array([20.0, 120.0, 250.0, 300.0])
    y = np.array([20.0, 90.0, 150.0, 40.0])
    ra, dec = wcs.pixel_to_world_values(x, y)
    px, py = model.world_to_pixel_values(ra, dec)
    assert np.max(np.hypot(px - x, py - y)) < 1.5
    assert stats["residual_p95_px"] < 1.5


def test_build_registers_model_and_filters_cloudy_source(tmp_path):
    video = tmp_path / "20260809_040000.mp4"
    video.write_bytes(b"placeholder")
    wcs_path = _simple_wcs(tmp_path / "seed.wcs")
    seen = {}

    def fake_solver(path, **kwargs):
        seen["path"] = path
        seen["kwargs"] = kwargs
        return {
            "wcs_file": wcs_path,
            "calibration_path": "fake-calibration.json",
            "reference_datetime": "2026-08-09T04:00:00",
            "width": 320,
            "height": 180,
            "sip_support_hull": [[0, 0], [319, 0], [319, 179], [0, 179]],
            "catalog_stars": [{"ra_deg": 120.0, "dec_deg": 35.0}],
        }

    classifier = lambda frame, **kwargs: CloudClassification(0.05, "test", 1.0)
    request = CameraModelBuildRequest(
        source=str(video), start="00:01", end="00:03", auto_select=True,
        cache_root=str(tmp_path / "cache"),
    )
    with mock.patch("camera_model_builder._read_probe_frame", return_value=np.zeros((180, 320, 3), np.uint8)):
        result = build_camera_model(request, classifier=classifier, solver=fake_solver)
    assert result.success
    assert result.enabled
    assert seen["path"] == str(video.resolve())
    payload = json.loads(Path(result.model_path).read_text())
    assert payload["model_type"] == "fixed-camera-stg-poly"
    assert payload["support_fraction"] == 1.0
    assert payload["catalog_stars"] == [{"ra_deg": 120.0, "dec_deg": 35.0}]
    assert payload["constellation_star_alignment"] is True
    assert payload["selection_summary"]["mode"] == "automatic"
    assert seen["kwargs"]["force"] is True


def test_build_prefers_wcs_observation_time_over_stale_solver_metadata(tmp_path):
    video = tmp_path / "20260809_040000.mp4"
    video.write_bytes(b"placeholder")
    wcs_path = tmp_path / "seed.wcs"
    _simple_wcs(wcs_path)
    with fits.open(wcs_path, mode="update") as hdul:
        hdul[0].header["DATE-OBS"] = "2026-08-09T04:00:00"

    def fake_solver(path, **kwargs):
        return {
            "wcs_file": str(wcs_path),
            "calibration_path": "fake-calibration.json",
            "reference_datetime": "2026-08-07T19:59:15.469000",
            "width": 320,
            "height": 180,
            "sip_support_hull": [[0, 0], [319, 0], [319, 179], [0, 179]],
        }

    classifier = lambda frame, **kwargs: CloudClassification(0.05, "test", 1.0)
    request = CameraModelBuildRequest(
        source=str(video), cache_root=str(tmp_path / "cache"),
    )
    with mock.patch("camera_model_builder._read_probe_frame", return_value=np.zeros((180, 320, 3), np.uint8)):
        result = build_camera_model(request, classifier=classifier, solver=fake_solver)

    assert result.success
    payload = json.loads(Path(result.model_path).read_text())
    assert payload["reference_datetime"] == "2026-08-09T04:00:00"


def test_select_video_paths_supports_clock_ranges(tmp_path):
    root = tmp_path / "20260809"
    hour = root / "04"
    hour.mkdir(parents=True)
    for minute in (0, 10, 20):
        (hour / f"{minute:02d}.mp4").write_bytes(b"")
    selected = select_video_paths(str(tmp_path), "04:05", "04:15")
    assert [Path(path).name for path in selected] == ["10.mp4"]


def test_automatic_model_source_expands_recorder_clip_to_date_folder(tmp_path):
    clip = tmp_path / "rtsp" / "20260809" / "04" / "05.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"")
    assert automatic_model_source(str(clip)) == str(clip.parents[1].resolve())


def test_select_auto_video_paths_prefers_night_candidates(tmp_path, monkeypatch):
    root = tmp_path / "20260809"
    root.mkdir()
    for hour in (12, 18, 20, 22):
        hour_dir = root / f"{hour:02d}"
        hour_dir.mkdir()
        (hour_dir / "00.mp4").write_bytes(b"")

    monkeypatch.setattr(
        "camera_model_builder._is_star_visibility_time",
        lambda stamp, _lat, _lon: stamp.hour >= 20,
    )
    selected = select_auto_video_paths(str(root), maximum=2)
    assert [Path(path).parent.name for path in selected] == ["20", "22"]
