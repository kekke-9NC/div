import json

from camera_model_catalog import discover_camera_models, format_model_details


def _write_model(path, **overrides):
    payload = {
        "model_type": "fixed-camera-stg-poly",
        "model_label": "TEST V4",
        "width": 1920,
        "height": 1080,
        "support_fraction": 0.8,
        "reference_datetime": "2026-08-09T01:50:00",
        "valid_dates": ["20260809"],
        "enabled": True,
        "wcs_path": str(path.parent / "wideangle_sip.wcs"),
    }
    payload.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    (path.parent / "wideangle_sip.wcs").write_bytes(b"wcs")


def test_discover_models_exposes_coverage_and_reference_night(tmp_path):
    model_path = tmp_path / "camera_models" / "v4" / "camera_model.json"
    _write_model(model_path)

    models = discover_camera_models(str(tmp_path))

    assert len(models) == 1
    model = models[0]
    assert model["support_percent"] == 80.0
    assert model["reference_night"] == "2026/08/09"
    assert "被覆率 80%" in model["display_name"]
    assert "基準夜 2026/08/09" in format_model_details(model)


def test_discover_models_skips_stale_wcs(tmp_path):
    model_path = tmp_path / "camera_models" / "stale" / "camera_model.json"
    _write_model(model_path, wcs_path=str(tmp_path / "missing.wcs"))

    assert discover_camera_models(str(tmp_path)) == []
