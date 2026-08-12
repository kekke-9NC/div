from datetime import datetime, timezone

import matplotlib
import numpy as np
from pathlib import Path

matplotlib.use("Agg")
from matplotlib.figure import Figure

import meteor_radiant_analysis as analysis
import meteor_radiant_visualizations as visualizations


def _report():
    result = analysis.RadiantResult(
        info_path="meteor/test_info.txt",
        source="test.mp4",
        detection_time=datetime(2026, 8, 11, 2, 0, tzinfo=timezone.utc),
        start_pixel=(100.0, 100.0),
        end_pixel=(150.0, 120.0),
        start_radec=(35.0, 50.0),
        end_radec=(25.0, 55.0),
        line_source="test",
        support_fraction=1.0,
        fully_supported=True,
        model_label="TEST MODEL",
        shower_code="PER",
        shower_name="ペルセウス座流星群",
        radiant_radec=(48.0, 58.0),
        radiant_distance_deg=1.2,
        radiant_side="start",
        confidence="高",
    )
    return analysis.RadiantReport("model.json", "TEST MODEL", [result], [])


def test_static_visualizations_render():
    report = _report()
    drawers = (
        visualizations.draw_aitoff_map,
        visualizations.draw_convergence_map,
        visualizations.draw_radec_map,
        visualizations.draw_density_heatmap,
        visualizations.draw_horizon_polar,
    )
    for drawer in drawers:
        figure = Figure(figsize=(8, 6))
        drawer(report, figure=figure)
        assert len(figure.axes) >= 1


def test_split_wrap_breaks_ra_seam():
    chunks = visualizations._split_wrap(
        [170.0, 179.0, -179.0, -170.0],
        [10.0, 11.0, 12.0, 13.0],
    )

    assert len(chunks) == 2
    assert all(len(lon) == 2 for lon, _lat in chunks)


def test_camera_extension_projects_a_sampled_sky_curve():
    class CurvedWcs:
        width = 1920
        height = 1080

        @staticmethod
        def world_to_pixel_values(ra, dec):
            ra = np.asarray(ra, dtype=float)
            dec = np.asarray(dec, dtype=float)
            return 8.0 * ra + 0.02 * dec ** 2, 6.0 * dec + 0.015 * ra ** 2

    result = _report().results[0]
    chunks = visualizations._camera_extension_pixel_chunks(
        result,
        {"reference_datetime": result.detection_time.isoformat()},
        CurvedWcs(),
        count=32,
    )

    assert len(chunks) == 1
    x, y = chunks[0]
    assert len(x) == 32
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(y))
    slopes = np.diff(y) / np.maximum(np.abs(np.diff(x)), 1e-9)
    assert float(np.ptp(slopes)) > 1e-4


def test_radec_polar_points_use_north_pole_center():
    theta, radius = visualizations._radec_polar_points(
        [0.0, 90.0, 180.0, 270.0],
        [90.0, 60.0, 0.0, -30.0],
    )

    assert np.allclose(radius, [0.0, 30.0, 90.0, 120.0])
    assert np.allclose(theta, np.deg2rad([0.0, 90.0, 180.0, 270.0]))


def test_radec_polar_plot_has_north_pole_at_center():
    figure = Figure(figsize=(8, 8))
    _figure, axis = visualizations.draw_horizon_polar(_report(), figure=figure)

    assert axis.get_rmax() == 180.0
    assert axis.get_rmin() == 0.0


def test_sphere_rotation_gif_is_written(tmp_path):
    output = Path(tmp_path) / "sphere_rotation.gif"
    visualizations.save_sphere_rotation_gif(_report(), str(output), fps=4, frames=3)

    assert output.is_file()
    assert output.read_bytes()[:6] == b"GIF89a"


def test_configurable_sphere_png_is_written(tmp_path):
    output = Path(tmp_path) / "sphere.png"
    options = analysis.SphereRenderOptions(
        show_x_axis=False,
        show_y_axis=False,
        show_z_axis=True,
        show_coordinate_grid=False,
        legend_showers=("PER",),
    )
    visualizations.save_sphere_png(
        _report(), str(output), width_px=320, height_px=240, dpi=60, options=options
    )

    from PIL import Image

    assert output.is_file()
    with Image.open(output) as image:
        assert image.size == (320, 240)
