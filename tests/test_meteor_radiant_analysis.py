from datetime import datetime

import numpy as np

import meteor_radiant_analysis as radiant
import meteor_shower_catalog as catalogue


def test_parse_pixel_line_prefers_saved_endpoints():
    data = {
        "Detected Line Center (px)": "(500.00, 400.00)",
        "Detected Line Start (px)": "(480.00, 380.00)",
        "Detected Line End (px)": "(520.00, 420.00)",
    }

    assert radiant.parse_pixel_line(data) == ((480.0, 380.0), (520.0, 420.0), "info.txt")


def test_parse_pixel_line_prefers_time_ordered_motion_endpoints():
    data = {
        "Detected Line Start (px)": "(520.00, 420.00)",
        "Detected Line End (px)": "(480.00, 380.00)",
        "Motion Direction Status": "high",
        "Motion Start (px)": "(480.00, 380.00)",
        "Motion End (px)": "(520.00, 420.00)",
    }

    assert radiant.parse_pixel_line(data) == (
        (480.0, 380.0),
        (520.0, 420.0),
        "info.txt:motion",
    )


def test_resolve_analysis_video_uses_event_full_video_for_minimal_info(tmp_path):
    info_path = tmp_path / "20260811_031824560_meteor_1_prob1.00_info.txt"
    full_video = tmp_path / "20260811_031824560_meteor_1_prob1.00_full.mp4"
    full_video.write_bytes(b"placeholder")

    resolved = radiant._resolve_analysis_video(
        str(info_path),
        {"Source": str(tmp_path / "archived" / "03.mp4")},
    )

    assert resolved == str(full_video.resolve())


def test_match_shower_accepts_active_perseids_geometry():
    radiant_vector = radiant.radec_to_unit_vector(48.0, 58.0)
    tangent = np.cross(radiant_vector, np.asarray((0.0, 0.0, 1.0)))
    tangent = tangent / np.linalg.norm(tangent)

    def offset(angle):
        angle = np.deg2rad(angle)
        return radiant._normalize(np.cos(angle) * radiant_vector + np.sin(angle) * tangent)

    shower, _vector, distance, side, candidates = radiant._match_shower(
        offset(18.0), offset(38.0), datetime(2026, 8, 11)
    )

    assert shower is not None
    assert shower.code == "PER"
    assert distance < 0.01
    assert side == "start"
    assert any(candidate.code == "PER" and candidate.active for candidate in candidates)


def test_direction_required_rejects_radiant_on_late_endpoint_side():
    radiant_vector = radiant.radec_to_unit_vector(48.0, 58.0)
    tangent = np.cross(radiant_vector, np.asarray((0.0, 0.0, 1.0)))
    tangent = tangent / np.linalg.norm(tangent)

    def offset(angle):
        angle = np.deg2rad(angle)
        return radiant._normalize(np.cos(angle) * radiant_vector + np.sin(angle) * tangent)

    early, late = offset(18.0), offset(38.0)
    forward = radiant._match_shower(
        early,
        late,
        datetime(2026, 8, 11),
        require_radiant_at_start=True,
    )
    reversed_line = radiant._match_shower(
        late,
        early,
        datetime(2026, 8, 11),
        require_radiant_at_start=True,
    )

    assert forward[0] is not None and forward[0].code == "PER"
    assert reversed_line[0] is None or reversed_line[0].code != "PER"


def test_analyze_info_files_excludes_line_outside_support(monkeypatch, tmp_path):
    model_path = tmp_path / "camera_model.json"
    model_path.write_text("{}", encoding="utf-8")
    info_path = tmp_path / "meteor_info.txt"
    info_path.write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr(radiant, "resolve_model_path", lambda _paths, _requested=None: str(model_path))
    monkeypatch.setattr(
        radiant,
        "_load_data",
        lambda _path: {
            "Source": "synthetic.mp4",
            "Detection Time (UTC)": "2026-08-11T02:00:00+00:00",
            "Detected Line Start (px)": "(10, 10)",
            "Detected Line End (px)": "(90, 90)",
        },
    )
    monkeypatch.setattr(radiant, "_probe_size", lambda _source: (100, 100))

    class FakeWcs:
        def pixel_to_world_values(self, x, y):
            return float(x), float(y - 45.0)

    import local_wideangle_astrometry

    monkeypatch.setattr(
        local_wideangle_astrometry,
        "_load_calibration",
        lambda _path: (
            {
                "model_label": "TEST",
                "width": 100,
                "height": 100,
                "reference_datetime": "2026-08-11T00:00:00+00:00",
            },
            FakeWcs(),
        ),
    )
    support = np.ones((100, 100), dtype=np.uint8)
    support[50:, :] = 0
    monkeypatch.setattr(
        local_wideangle_astrometry,
        "_forward_grid_model",
        lambda _wcs, _metadata, _width, _height: {"support_mask": support},
    )

    report = radiant.analyze_info_files([str(info_path)])

    assert report.results == []
    assert len(report.skipped) == 1
    assert "有効領域外" in report.skipped[0][1]


def test_activity_window_blocks_inactive_radiant_candidate():
    shower = next(item for item in radiant.METEOR_SHOWERS if item.code == "DRA")
    assert shower.is_active(datetime(2026, 10, 8))
    assert not shower.is_active(datetime(2026, 8, 11))


def test_shower_catalogue_round_trip(tmp_path):
    entries = catalogue.default_catalogue()
    codes = {item.code for item in entries}
    assert len(entries) >= 30
    assert {"PER", "KCG", "GUM", "JBO", "COM"} <= codes
    assert {item.category for item in entries} == {"大流星群", "小流星群"}
    path = tmp_path / "catalogue.json"
    catalogue.save_catalogue(entries, str(path))
    loaded = catalogue.load_catalogue(str(path))

    assert len(loaded) == len(entries)
    assert {item.code for item in loaded} == {item.code for item in entries}
    perseid = next(item for item in loaded if item.code == "PER")
    assert perseid.category == "大流星群"
    assert perseid.radiant_ra_deg == 48.0


def test_shower_catalogue_rejects_invalid_coordinates_and_dates():
    base = {
        "code": "TST",
        "name": "テスト流星群",
        "active_start": [1, 1],
        "peak": [1, 5],
        "active_end": [1, 10],
        "radiant_ra_deg": 20,
        "radiant_dec_deg": 10,
        "match_limit_deg": 12,
    }
    invalid_dec = dict(base, radiant_dec_deg=91)
    invalid_date = dict(base, peak=[2, 30])
    invalid_ra = dict(base, radiant_ra_deg=float("inf"))

    import pytest

    with pytest.raises(ValueError):
        catalogue.record_to_shower(invalid_dec)
    with pytest.raises(ValueError):
        catalogue.record_to_shower(invalid_date)
    with pytest.raises(ValueError):
        catalogue.record_to_shower(invalid_ra)


def test_custom_catalogue_is_used_for_matching():
    custom = catalogue.record_to_shower({
        "code": "CUS",
        "name": "カスタム流星群",
        "active_start": [1, 1],
        "peak": [1, 5],
        "active_end": [1, 10],
        "radiant_ra_deg": 48,
        "radiant_dec_deg": 58,
        "match_limit_deg": 12,
    })
    radiant_vector = radiant.radec_to_unit_vector(48.0, 58.0)
    tangent = np.cross(radiant_vector, np.asarray((0.0, 0.0, 1.0)))
    tangent = tangent / np.linalg.norm(tangent)
    start = radiant._normalize(np.cos(np.deg2rad(18)) * radiant_vector + np.sin(np.deg2rad(18)) * tangent)
    end = radiant._normalize(np.cos(np.deg2rad(38)) * radiant_vector + np.sin(np.deg2rad(38)) * tangent)

    shower, _vector, _distance, _side, _candidates = radiant._match_shower(
        start, end, datetime(2026, 1, 5), showers=[custom]
    )
    assert shower is not None
    assert shower.code == "CUS"


def test_radiant_sphere_is_labeled_in_equatorial_coordinates():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = radiant.RadiantReport(model_path="", model_label="TEST", results=[], skipped=[])
    figure, axis = radiant.draw_radiant_sphere(report)

    assert "RA/Dec基準" in axis.get_title()
    labels = {text.get_text() for text in axis.texts}
    assert "RA 0h" in labels
    assert "Dec +0°" in labels
    plt.close(figure)


def test_report_exposes_supported_shower_counts():
    result = radiant.RadiantResult(
        info_path="a.txt",
        source="test.mp4",
        detection_time=datetime(2026, 8, 11),
        start_pixel=(0.0, 0.0),
        end_pixel=(1.0, 1.0),
        start_radec=(0.0, 0.0),
        end_radec=(1.0, 1.0),
        line_source="test",
        support_fraction=1.0,
        fully_supported=True,
        model_label="TEST",
        shower_code="PER",
    )
    report = radiant.RadiantReport(model_path="", model_label="TEST", results=[result, result], skipped=[])
    assert report.shower_counts == {"PER": 2}
