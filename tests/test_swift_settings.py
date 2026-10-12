"""Verify the production Swift workspace restores and saves every legacy key."""

import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def workspace_validation(tmp_path_factory):
    if sys.platform != "darwin" or not shutil.which("swiftc"):
        pytest.skip("The macOS Swift compiler is required")
    build = tmp_path_factory.mktemp("swift-settings-build")
    core = ROOT / "swiftui/Sources/MeteorDetectorCore"
    app = ROOT / "swiftui/Sources/MeteorDetectorApp"
    subprocess.run(
        ["swiftc", "-emit-library", "-emit-module", "-module-name", "MeteorDetectorCore",
         str(core / "Models.swift"), str(core / "LegacySettingsDocument.swift"),
         "-o", str(build / "libMeteorDetectorCore.dylib")],
        check=True, capture_output=True, text=True, timeout=120,
    )
    executable = build / "validate"
    subprocess.run(
        ["swiftc", "-parse-as-library", str(app / "PythonBridge.swift"),
         str(app / "WorkspaceStore.swift"), str(app / "MaskPreviewFrame.swift"),
         str(ROOT / "tests/swift/WorkspaceSettingsValidation.swift"),
         "-I", str(build), "-L", str(build), "-lMeteorDetectorCore",
         "-Xlinker", "-rpath", "-Xlinker", str(build), "-o", str(executable)],
        check=True, capture_output=True, text=True, timeout=120,
    )
    return executable


def legacy_saved_keys():
    tree = ast.parse((ROOT / "gui_settings.py").read_text())
    save = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "save_settings")
    assignment = next(node for node in save.body if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict))
    return {key.value for key in assignment.value.keys} | {"global_wcs_info"}


@pytest.mark.parametrize("invalid,calibration", [(False, False), (True, False), (False, True)])
def test_workspace_keeps_all_legacy_keys(workspace_validation, tmp_path, monkeypatch, invalid, calibration):
    monkeypatch.setenv("METEOR_PYTHON", sys.executable)
    monkeypatch.setenv("METEOR_LEGACY_SETTINGS_HOME", str(tmp_path / "test-home"))
    settings = {key: f"retained:{key}" for key in legacy_saved_keys()}
    settings.update({
        "folder_paths": [], "rtsp_urls": [], "processing_source_priority": ["rtsp", "folder", "periodic"],
        "concurrency": "4", "interval": "2.5", "duration": "60", "periodic_scan_interval": "1",
        "rtsp_fps": "25", "rtsp_preset": "clear", "rtsp_fixed_pattern_samples": 180,
        "temporal_mean_frames": 3, "noise_twin_enabled": False,
        "use_plate_solve": True, "plate_solve_wcs_path": "/missing/calibration.wcs",
        "plate_solve_mode": "local",
        "plate_solve_model_path": "/missing/model.json", "selected_model_path": "/missing/selected.pth",
        "apply_rtsp_dark": True, "rtsp_fixed_pattern_path": "/missing/dark.npz", "apply_mask": False,
        "camera_model_method": "静止画プレートソルブ", "auto_time_updater_enabled": False,
        "global_wcs_info": {"model_path": "/missing/model.json", "future": {"keep": True}},
        "advanced_settings": {"meteor_prob": "0.7", "future_threshold": 123},
        "full_video_timestamp": {"enabled": False, "position": "左上", "size_percent": "3", "future": 9},
        "processed_video_encoding": {"codec": "MPEG-4 (互換)", "quality": "カスタム", "bitrate_mbps": "30", "future": 8},
        "video_concat_settings": {"bitrate": "Auto", "codec": "h265", "fps": "25", "future": 7},
        "save_options": {"video": True, "future_output": False}, "custom_model_paths": ["/missing/custom.pth"],
        "summary_video_config": [
            {"name": "Full Size Video", "enabled": True, "future": {"keep": 1}},
            {"name": "Future Output", "enabled": True, "future": {"keep": 2}},
        ],
        "future_settings": {"nested": [1, 2, 3]},
    })
    if calibration:
        calibration_path = tmp_path / "legacy-camera.wcs"
        calibration_path.write_text("placeholder", encoding="utf-8")
        settings["global_wcs_info"]["calibration_path"] = str(calibration_path)
    (tmp_path / "fixture.json").write_text(json.dumps(settings), encoding="utf-8")
    # A live protocol fixture isolates Swift state restoration from processing.
    (tmp_path / "swift_backend.py").write_text(
        "import json, pathlib, sys\n"
        "root = pathlib.Path(__file__).parent\n"
        "settings = json.loads((root / 'fixture.json').read_text())\n"
        f"invalid = {invalid!r}\n"
        "for line in sys.stdin:\n"
        "    req = json.loads(line); command = req['command']\n"
        "    payload = {}\n"
        "    if command == 'ping': payload = {'name': 'Meteor Detector'}\n"
        "    if command == 'load_settings': payload = {'settings': settings, 'unsupportedFeatures': []}\n"
        "    if command == 'save_settings':\n"
        "        (root / 'saved.json').write_text(json.dumps(req['payload']['settings']))\n"
        "    print(json.dumps({'type': 'response', 'id': req['id'], 'ok': not (invalid and command == 'load_settings'), 'error': 'invalid settings', 'payload': payload}), flush=True)\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [str(workspace_validation), str(tmp_path), "invalid" if invalid else "valid"],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    if invalid:
        assert not (tmp_path / "saved.json").exists()
        return
    saved = json.loads((tmp_path / "saved.json").read_text())
    assert legacy_saved_keys() <= saved.keys()
    for key in ("full_video_timestamp", "processed_video_encoding", "custom_model_paths",
                "global_wcs_info", "future_settings", "camera_model_method"):
        assert saved[key] == settings[key]
    assert saved["advanced_settings"] == {"meteor_prob": "0.8", "future_threshold": 123}
    assert saved["video_concat_settings"]["future"] == 7
    assert saved["save_options"]["future_output"] is False
    assert saved["summary_video_config"][0]["future"] == {"keep": 1}
    assert saved["summary_video_config"][1] == settings["summary_video_config"][1]
