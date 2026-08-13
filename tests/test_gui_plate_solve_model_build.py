import json

from camera_model_builder import CameraModelBuildRequest, CameraModelBuildResult
from trajectory_camera_model import TrajectoryBuildResult
import gui_plate_solve


def _seed_payload():
    return {
        "model_type": "fixed-camera-stg-poly",
        "stg_parameters": [0.0] * 7,
        "correction_coefficients": [],
        "reference_datetime": "2026-08-13T00:00:00",
    }


def test_app_trajectory_mode_uses_video_tracker_with_selected_seed(tmp_path, monkeypatch):
    seed = tmp_path / "camera_model.json"
    seed.write_text(json.dumps(_seed_payload()), encoding="utf-8")
    captured = {}

    def trajectory_builder(request, progress_callback=None):
        captured["request"] = request
        return TrajectoryBuildResult(success=True, model_path="trajectory.json")

    def unexpected_static_builder(*args, **kwargs):
        raise AssertionError("an existing fixed-camera seed must be reused")

    monkeypatch.setattr(gui_plate_solve, "build_trajectory_camera_model", trajectory_builder)
    monkeypatch.setattr(gui_plate_solve, "build_camera_model", unexpected_static_builder)
    result = gui_plate_solve._build_camera_model_for_app(
        CameraModelBuildRequest(source="night-folder", start="00:00", end="01:00"),
        use_trajectory=True,
        initial_model_path=str(seed),
    )
    assert result.model_path == "trajectory.json"
    assert captured["request"].source == "night-folder"
    assert captured["request"].initial_model_path == str(seed)


def test_app_trajectory_mode_creates_seed_when_none_is_selected(tmp_path, monkeypatch):
    seed = tmp_path / "generated-camera-model.json"
    seed.write_text(json.dumps(_seed_payload()), encoding="utf-8")
    calls = []

    def static_builder(request, progress_callback=None):
        calls.append("static")
        return CameraModelBuildResult(success=True, model_path=str(seed))

    def trajectory_builder(request, progress_callback=None):
        calls.append("trajectory")
        assert request.initial_model_path == str(seed)
        return TrajectoryBuildResult(success=True, model_path="trajectory.json")

    monkeypatch.setattr(gui_plate_solve, "build_camera_model", static_builder)
    monkeypatch.setattr(gui_plate_solve, "build_trajectory_camera_model", trajectory_builder)
    gui_plate_solve._build_camera_model_for_app(
        CameraModelBuildRequest(source="night-folder"),
        use_trajectory=True,
    )
    assert calls == ["static", "trajectory"]
