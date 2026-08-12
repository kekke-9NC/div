from datetime import datetime, timezone

import matplotlib

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
