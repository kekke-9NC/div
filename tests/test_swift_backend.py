import json
import io
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run_bridge(tmp_path: Path, *requests):
    command = [PYTHON, str(ROOT / "swift_backend.py"), "--stdio", "--root", str(tmp_path)]
    payload = "".join(json.dumps(request, ensure_ascii=False) + "\n" for request in requests)
    environment = os.environ.copy()
    environment["METEOR_DETECTOR_ROOT"] = str(ROOT)
    result = subprocess.run(
        command,
        input=payload,
        text=True,
        capture_output=True,
        cwd=ROOT,
        env=environment,
        check=True,
        timeout=20,
    )
    return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]


def test_ping_and_empty_settings_are_json_lines(tmp_path):
    messages = run_bridge(
        tmp_path,
        {"id": "ping", "command": "ping", "payload": {}},
        {"id": "settings", "command": "load_settings", "payload": {}},
    )

    responses = {message["id"]: message for message in messages if message["type"] == "response"}
    assert responses["ping"]["ok"] is True
    assert responses["ping"]["payload"]["bridgeVersion"] == 1
    assert responses["settings"]["payload"]["settings"] == {}


def test_summary_config_fills_legacy_duration_defaults():
    from swift_backend import Bridge

    assert Bridge._validated_summary_config(
        [{"name": "Composite Image", "enabled": True}]
    ) == [{"name": "Composite Image", "enabled": True, "duration": 1.0}]


def test_noise_twin_options_validate_temporal_mean_without_loading_a_model():
    from swift_backend import Bridge

    assert Bridge(Path("/tmp"))._validated_noise_twin_options(
        {"temporalMeanFrames": 3, "saveTemporalMeanVideo": False}
    ) == {
        "enabled": False,
        "model_path": "",
        "require_validated": True,
        "temporal_mean_frames": 3,
        "save_temporal_mean_video": False,
    }

    try:
        Bridge(Path("/tmp"))._validated_noise_twin_options(
            {"temporalMeanFrames": 4}
        )
    except ValueError as exc:
        assert "0, 3, or 5" in str(exc)
    else:
        raise AssertionError("unsupported temporal mean window should fail")


def test_selected_model_path_requires_an_existing_file(tmp_path):
    from swift_backend import Bridge

    model_path = tmp_path / "custom_detector.pth"
    model_path.write_bytes(b"placeholder")
    bridge = Bridge(tmp_path)

    assert bridge._validated_model_path(str(model_path)) == str(model_path.resolve())
    assert bridge._validated_model_path("") == ""

    try:
        bridge._validated_model_path(str(tmp_path / "missing.pth"))
    except ValueError as exc:
        assert "検出モデルが見つかりません" in str(exc)
    else:
        raise AssertionError("missing selected model should fail")


def test_selected_model_is_loaded_with_metadata_before_processing(tmp_path):
    from swift_backend import Bridge

    model_path = tmp_path / "custom_detector.pth"
    model_path.write_bytes(b"placeholder")
    calls = {}
    fake_model = SimpleNamespace(
        reload_model=lambda **kwargs: calls.update(reload=kwargs) or (True, "ok")
    )
    fake_catalog = SimpleNamespace(
        load_model_metadata=lambda path: calls.update(metadata_path=path) or {"architecture": "test"}
    )

    bridge = Bridge(tmp_path, protocol_stdout=io.StringIO())
    with mock.patch.dict(
        sys.modules,
        {"model": fake_model, "model_catalog": fake_catalog},
    ):
        bridge._load_selected_model({"modelPath": str(model_path)})

    assert calls["metadata_path"] == str(model_path.resolve())
    assert calls["reload"]["model_path"] == str(model_path.resolve())
    assert calls["reload"]["metadata"] == {"architecture": "test"}


def test_existing_wcs_path_is_validated_for_coordinate_annotations(tmp_path):
    from swift_backend import Bridge

    wcs_path = tmp_path / "camera.wcs"
    wcs_path.write_text("placeholder", encoding="utf-8")
    bridge = Bridge(tmp_path)

    assert bridge._validated_wcs_info(str(wcs_path)) == {
        "wcs_file": str(wcs_path.resolve()),
        "job_id": "manual-wcs",
    }
    assert bridge._validated_wcs_info("") is None

    invalid_path = tmp_path / "camera.txt"
    invalid_path.write_text("placeholder", encoding="utf-8")
    try:
        bridge._validated_wcs_info(str(invalid_path))
    except ValueError as exc:
        assert "json" in str(exc)
    else:
        raise AssertionError("unsupported WCS extension should fail")


def test_camera_model_payload_validates_source_and_bounds(tmp_path):
    from swift_backend import Bridge

    source = tmp_path / "sample.mp4"
    source.write_bytes(b"placeholder")
    bridge = Bridge(tmp_path)
    payload = bridge._validated_camera_model_payload(
        {
            "source": str(source),
            "autoSelect": False,
            "cloudThreshold": 0.2,
            "useCloudFilter": True,
            "maximumVideos": 4,
            "observationLatitude": 35.5,
            "observationLongitude": 139.7,
        }
    )
    assert payload["source"] == str(source.resolve())
    assert payload["autoSelect"] is False
    assert payload["maximumVideos"] == 4
    assert payload["observationLatitude"] == 35.5
    assert payload["observationLongitude"] == 139.7

    try:
        bridge._validated_camera_model_payload(
            {"source": str(source), "observationLatitude": 91}
        )
    except ValueError as exc:
        assert "observationLatitude" in str(exc)
    else:
        raise AssertionError("invalid camera-model coordinates should fail")


def test_camera_model_cancellation_emits_a_terminal_result(tmp_path):
    from swift_backend import Bridge

    source = tmp_path / "sample.mp4"
    source.write_bytes(b"placeholder")
    fake_module = ModuleType("camera_model_builder")

    class FakeRequest:
        def __init__(self, **_kwargs):
            pass

    def fake_build(_request, progress_callback=None):
        try:
            progress_callback("処理中")
        except RuntimeError:
            # Match the production builder's failure notification path.
            progress_callback("高精度モデル作成失敗")
        return SimpleNamespace(
            success=False,
            model_path="",
            error="停止要求",
            as_dict=lambda: {"success": False, "error": "停止要求"},
        )

    fake_module.CameraModelBuildRequest = FakeRequest
    fake_module.build_camera_model = fake_build
    protocol = io.StringIO()
    bridge = Bridge(tmp_path, protocol_stdout=protocol)
    cancel_event = threading.Event()
    cancel_event.set()
    payload = bridge._validated_camera_model_payload({"source": str(source)})
    with mock.patch.dict(sys.modules, {"camera_model_builder": fake_module}):
        bridge._build_camera_model_worker("camera-model", payload, cancel_event)

    messages = [json.loads(line) for line in protocol.getvalue().splitlines()]
    result_events = [message for message in messages if message.get("event") == "camera_model_result"]
    assert len(result_events) == 1
    assert result_events[0]["payload"]["cancelled"] is True
    assert any(
        message.get("event") == "run_state" and message["payload"].get("state") == "cancelled"
        for message in messages
    )


def test_detection_mask_loads_legacy_npz(tmp_path):
    import numpy as np

    from swift_backend import Bridge

    mask_path = tmp_path / "app_masks.npz"
    np.savez(mask_path, mask_image=np.array([[0, 300], [True, False]]))
    mask = Bridge(tmp_path)._load_detection_mask(
        {"applyMask": True, "maskPath": str(mask_path)}
    )

    assert mask.dtype == np.uint8
    assert mask.tolist() == [[0, 255], [1, 0]]


def test_detection_mask_rejects_missing_payload_file(tmp_path):
    from swift_backend import Bridge

    try:
        Bridge(tmp_path)._load_detection_mask(
            {"applyMask": True, "maskPath": str(tmp_path / "missing.npz")}
        )
    except ValueError as exc:
        assert "見つかりません" in str(exc)
    else:
        raise AssertionError("missing detection mask should fail")


def test_fixed_pattern_validation_accepts_current_and_legacy_npz(tmp_path):
    import numpy as np

    from swift_backend import Bridge

    current_path = tmp_path / "current.npz"
    np.savez(current_path, fixed_correction=np.zeros((5, 7), dtype=np.int16))
    current = Bridge(tmp_path)._validate_fixed_pattern_payload({"path": str(current_path)})
    assert current["valid"] is True
    assert current["width"] == 7
    assert current["height"] == 5
    assert current["method"] == "fixed_correction"

    legacy_path = tmp_path / "legacy.npz"
    np.savez(legacy_path, dark_frame=np.zeros((3, 4), dtype=np.uint8))
    legacy = Bridge(tmp_path)._validate_fixed_pattern_payload({"path": str(legacy_path)})
    assert legacy["method"] == "dark_frame"
    assert legacy["width"] == 4
    assert legacy["height"] == 3
    correction, key = Bridge(tmp_path)._read_fixed_pattern_correction(str(legacy_path))
    assert key == "dark_frame"
    assert correction.dtype == np.uint8


def test_fixed_pattern_validation_rejects_missing_array(tmp_path):
    import numpy as np

    from swift_backend import Bridge

    path = tmp_path / "invalid.npz"
    np.savez(path, unrelated=np.zeros((2, 2), dtype=np.uint8))
    try:
        Bridge(tmp_path)._validate_fixed_pattern_payload({"path": str(path)})
    except ValueError as exc:
        assert "fixed_correction" in str(exc)
    else:
        raise AssertionError("invalid fixed-pattern archive should fail")


def test_fixed_pattern_build_payload_validates_source_modes(tmp_path):
    from swift_backend import Bridge

    video = tmp_path / "dark.mp4"
    video.write_bytes(b"placeholder")
    bridge = Bridge(tmp_path)
    payload = bridge._validated_fixed_pattern_build_payload(
        {"mode": "video", "source": str(video), "samples": 90}
    )
    assert payload["mode"] == "video"
    assert payload["source"] == str(video.resolve())
    assert payload["samples"] == 90
    assert payload["output"].endswith("rtsp_dark_frame.npz")

    try:
        bridge._validated_fixed_pattern_build_payload(
            {"mode": "url", "source": "https://example.invalid/live"}
        )
    except ValueError as exc:
        assert "RTSP" in str(exc)
    else:
        raise AssertionError("non-RTSP calibration source should fail")


def test_save_mask_from_strokes_writes_legacy_npz(tmp_path):
    import numpy as np

    from swift_backend import Bridge

    mask_path = tmp_path / "app_masks.npz"
    plate_mask = np.full((12, 20), 200, dtype=np.uint8)
    companion = np.array([1, 2, 3], dtype=np.int16)
    np.savez(mask_path, mask_image=plate_mask, plate_solve_mask_image=plate_mask, companion=companion)
    saved = Bridge(tmp_path)._save_mask_from_strokes(
        {
            "maskPath": str(mask_path),
            "width": 20,
            "height": 12,
            "brushSize": 0.2,
            "strokes": [
                {"mode": "exclude", "points": [[0.1, 0.5], [0.9, 0.5]]},
                {"mode": "restore", "points": [[0.5, 0.5]]},
            ],
        }
    )

    assert saved == str(mask_path.resolve())
    with np.load(mask_path, allow_pickle=False) as archive:
        mask = archive["mask_image"]
        assert np.array_equal(archive["plate_solve_mask_image"], plate_mask)
        assert np.array_equal(archive["companion"], companion)
    assert mask.shape == (12, 20)
    assert mask.dtype == np.uint8
    assert int(mask.min()) == 0
    assert int(mask.max()) == 255


def test_save_mask_from_strokes_rejects_non_npz_path(tmp_path):
    from swift_backend import Bridge

    try:
        Bridge(tmp_path)._save_mask_from_strokes(
            {"maskPath": str(tmp_path / "mask.png"), "width": 4, "height": 4}
        )
    except ValueError as exc:
        assert ".npz" in str(exc)
    else:
        raise AssertionError("mask editor should only write NPZ masks")


def test_validate_mask_command_reports_shape_and_errors(tmp_path):
    import numpy as np

    mask_path = tmp_path / "app_masks.npz"
    np.savez(mask_path, mask_image=np.zeros((4, 6), dtype=np.uint8))
    messages = run_bridge(
        tmp_path,
        {
            "id": "valid",
            "command": "validate_mask",
            "payload": {"maskPath": str(mask_path)},
        },
        {
            "id": "missing",
            "command": "validate_mask",
            "payload": {"maskPath": str(tmp_path / "missing.npz")},
        },
    )
    responses = {message["id"]: message for message in messages if message["type"] == "response"}

    assert responses["valid"]["ok"] is True
    assert responses["valid"]["payload"]["width"] == 6
    assert responses["valid"]["payload"]["height"] == 4
    assert responses["missing"]["ok"] is False


def test_detection_mask_is_passed_to_all_run_modes(tmp_path):
    import numpy as np

    from swift_backend import Bridge

    mask_path = tmp_path / "app_masks.npz"
    np.savez(mask_path, mask_image=np.full((2, 2), 255, dtype=np.uint8))
    fixed_pattern_path = tmp_path / "rtsp_dark_frame.npz"
    np.savez(fixed_pattern_path, fixed_correction=np.ones((2, 2), dtype=np.int16))
    calls = {}

    def fake_pipeline(**kwargs):
        calls["local"] = kwargs

    def fake_monitor(**kwargs):
        calls["periodic"] = kwargs

    def fake_rtsp(**kwargs):
        calls["rtsp"] = kwargs

    fake_config = SimpleNamespace(MIN_LINE_LENGTH=1, RTSP_SEGMENT_DURATION=1)
    fake_download_pipeline = SimpleNamespace(run_pipeline=fake_pipeline)
    fake_file_utils = SimpleNamespace(
        monitor_directory=fake_monitor,
        rtsp_save_and_process_thread_target=fake_rtsp,
    )
    payload = {
        "applyMask": True,
        "maskPath": str(mask_path),
        "applyFixedPattern": True,
        "fixedPatternPath": str(fixed_pattern_path),
        "saveOptions": Bridge._default_save_options(),
        "summaryConfig": Bridge._default_summary_config(),
        "meteorSavePath": str(tmp_path / "meteor"),
        "notMeteorSavePath": str(tmp_path / "not_meteor"),
        "interval": 1.0,
        "duration": 1.0,
    }

    bridge = Bridge(tmp_path, protocol_stdout=io.StringIO())
    with mock.patch.dict(
        sys.modules,
        {
            "config": fake_config,
            "download_pipeline": fake_download_pipeline,
            "file_utils": fake_file_utils,
        },
    ):
        bridge._run_pipeline(
            {**payload, "sources": [{"path": str(tmp_path / "sample.mp4")}]},
            threading.Event(),
        )
        bridge._run_periodic(
            {**payload, "directory": str(tmp_path), "scanInterval": 5},
            threading.Event(),
        )
        bridge._run_rtsp(
            {**payload, "url": "rtsp://camera/live"},
            threading.Event(),
        )

    for call in calls.values():
        assert call["mask"].tolist() == [[255, 255], [255, 255]]
        correction = call.get("fixed_pattern_correction", call.get("dark_frame"))
        assert correction.tolist() == [[1, 1], [1, 1]]


def test_rtsp_preset_and_fps_are_applied_temporarily():
    from swift_backend import Bridge

    config = SimpleNamespace(
        RTSP_PRESET_CLEAR_SKY={
            "min_line_length": 20,
            "hough_threshold": 25,
            "canny_thresh1": 75,
            "canny_thresh2": 180,
        },
        RTSP_PRESET_CLOUDY={
            "min_line_length": 25,
            "hough_threshold": 35,
            "canny_thresh1": 100,
            "canny_thresh2": 240,
        },
        RTSP_MIN_LINE_LENGTH=99,
        RTSP_HOUGH_THRESHOLD=99,
        RTSP_CANNY_THRESH1=99,
        RTSP_CANNY_THRESH2=99,
        RTSP_FPS=25,
    )

    with Bridge._temporary_rtsp_config(config, {"rtspPreset": "clear", "rtspFps": 30}):
        assert config.RTSP_MIN_LINE_LENGTH == 20
        assert config.RTSP_HOUGH_THRESHOLD == 25
        assert config.RTSP_CANNY_THRESH1 == 75
        assert config.RTSP_CANNY_THRESH2 == 180
        assert config.RTSP_FPS == 30

    assert config.RTSP_MIN_LINE_LENGTH == 99
    assert config.RTSP_HOUGH_THRESHOLD == 99
    assert config.RTSP_CANNY_THRESH1 == 99
    assert config.RTSP_CANNY_THRESH2 == 99
    assert config.RTSP_FPS == 25


def test_settings_are_written_atomically_and_unknown_commands_fail(tmp_path):
    messages = run_bridge(
        tmp_path,
        {
            "id": "save",
            "command": "save_settings",
            "payload": {
                "settings": {
                    "folder_paths": ["/tmp/日本語"],
                    "custom": 42,
                    "rtsp_time_limit_enabled": True,
                    "rtsp_start_hour": 18,
                    "rtsp_end_hour": 6,
                    "rtsp_notification_sound": False,
                    "rtsp_preset": "clear",
                    "rtsp_fps": "30",
                    "selected_model_path": "/tmp/custom_detector.pth",
                    "processing_source_priority": ["folder", "rtsp", "periodic"],
                    "summary_video_config": [
                        {"name": "Composite Image", "enabled": False, "duration": 3.5}
                    ],
                }
            },
        },
        {"id": "load", "command": "load_settings", "payload": {}},
        {"id": "bad", "command": "not-a-command", "payload": {}},
    )

    responses = {message["id"]: message for message in messages if message["type"] == "response"}
    assert responses["save"]["ok"] is True
    settings = responses["load"]["payload"]["settings"]
    assert settings["folder_paths"] == ["/tmp/日本語"]
    assert settings["custom"] == 42
    assert settings["rtsp_time_limit_enabled"] is True
    assert settings["rtsp_start_hour"] == 18
    assert settings["rtsp_end_hour"] == 6
    assert settings["rtsp_notification_sound"] is False
    assert settings["rtsp_preset"] == "clear"
    assert settings["rtsp_fps"] == "30"
    assert settings["selected_model_path"] == "/tmp/custom_detector.pth"
    assert settings["processing_source_priority"] == ["folder", "rtsp", "periodic"]
    assert settings["summary_video_config"] == [
        {"name": "Composite Image", "enabled": False, "duration": 3.5}
    ]
    assert responses["bad"]["ok"] is False
    assert (tmp_path / "app_settings.json").read_text(encoding="utf-8").endswith("\n")


def test_invalid_run_requests_return_errors_without_starting_processing(tmp_path):
    messages = run_bridge(
        tmp_path,
        {"id": "run", "command": "run_detection", "payload": {"sources": []}},
        {"id": "periodic", "command": "run_periodic", "payload": {"directory": str(tmp_path / "missing")}},
        {
            "id": "bad_periodic",
            "command": "run_periodic",
            "payload": {"directory": str(tmp_path), "saveOptions": []},
        },
        {"id": "rtsp", "command": "run_rtsp", "payload": {"url": "not-a-url"}},
        {
            "id": "bad_rtsp_options",
            "command": "run_rtsp",
            "payload": {
                "url": "rtsp://camera/live",
                "notifyOnDetection": "false",
                "rtspPreset": "invalid",
            },
        },
        {
            "id": "bad_periodic_window",
            "command": "run_periodic",
            "payload": {
                "directory": str(tmp_path),
                "timeLimitEnabled": True,
                "startHour": 18,
                "startMinute": 30,
                "endHour": 18,
                "endMinute": 30,
            },
        },
        {
            "id": "bad_summary",
            "command": "run_detection",
            "payload": {
                "sources": [{"path": str(tmp_path / "sample.mp4")}],
                "summaryConfig": [{"name": "Composite Image", "enabled": "yes"}],
            },
        },
        {
            "id": "bad_summary_name",
            "command": "run_detection",
            "payload": {
                "sources": [{"path": str(tmp_path / "sample.mp4")}],
                "summaryConfig": [{"name": "Unknown Output", "enabled": True}],
            },
        },
        {
            "id": "bad_summary_empty",
            "command": "run_detection",
            "payload": {
                "sources": [{"path": str(tmp_path / "sample.mp4")}],
                "summaryConfig": [
                    {"name": "Composite Image", "enabled": False},
                    {"name": "Full Size Video", "enabled": False},
                ],
            },
        },
        {
            "id": "bad_mask_type",
            "command": "run_detection",
            "payload": {
                "sources": [{"path": str(tmp_path / "sample.mp4")}],
                "applyMask": "true",
            },
        },
        {
            "id": "bad_rtsp_window",
            "command": "run_rtsp",
            "payload": {
                "url": "rtsp://camera/live",
                "timeLimitEnabled": True,
                "startHour": 18,
                "startMinute": 30,
                "endHour": 18,
                "endMinute": 30,
            },
        },
    )

    responses = {message["id"]: message for message in messages if message["type"] == "response"}
    assert responses["run"]["ok"] is False
    assert responses["periodic"]["ok"] is False
    assert responses["bad_periodic"]["ok"] is False
    assert responses["rtsp"]["ok"] is False
    assert responses["bad_rtsp_options"]["ok"] is False
    assert responses["bad_periodic_window"]["ok"] is False
    assert responses["bad_summary"]["ok"] is False
    assert responses["bad_summary_name"]["ok"] is False
    assert responses["bad_summary_empty"]["ok"] is False
    assert responses["bad_mask_type"]["ok"] is False
    assert responses["bad_rtsp_window"]["ok"] is False


def test_corrupt_settings_are_not_overwritten(tmp_path):
    settings_path = tmp_path / "app_settings.json"
    settings_path.write_text("{not-json", encoding="utf-8")

    messages = run_bridge(
        tmp_path,
        {"id": "load", "command": "load_settings", "payload": {}},
        {"id": "save", "command": "save_settings", "payload": {"settings": {"new": True}}},
    )

    responses = {message["id"]: message for message in messages if message["type"] == "response"}
    assert responses["load"]["ok"] is False
    assert responses["save"]["ok"] is False
    assert settings_path.read_text(encoding="utf-8") == "{not-json"


def test_non_object_settings_are_not_overwritten(tmp_path):
    settings_path = tmp_path / "app_settings.json"
    settings_path.write_text("[1, 2, 3]", encoding="utf-8")

    messages = run_bridge(
        tmp_path,
        {"id": "load", "command": "load_settings", "payload": {}},
        {"id": "save", "command": "save_settings", "payload": {"settings": {"new": True}}},
    )

    responses = {message["id"]: message for message in messages if message["type"] == "response"}
    assert responses["load"]["ok"] is False
    assert responses["save"]["ok"] is False
    assert settings_path.read_text(encoding="utf-8") == "[1, 2, 3]"


def test_legacy_feature_settings_are_reported_to_the_swiftui_frontend(tmp_path):
    settings_path = tmp_path / "app_settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "summary_video_config": [{"name": "Composite Image", "enabled": False}],
                "apply_rtsp_dark": True,
                "rtsp_time_limit_enabled": True,
                "rtsp_notification_sound": False,
                "rtsp_preset": "clear",
                "rtsp_fps": "30",
                "noise_twin_enabled": False,
                "temporal_mean_frames": 3,
                "rtsp_save_temporal_mean": True,
                "video_concat_settings": {"codec": "h265"},
                "periodic_scan_enabled": True,
                "periodic_scan_directory": str(tmp_path),
                "periodic_time_limit_enabled": True,
            }
        ),
        encoding="utf-8",
    )

    messages = run_bridge(
        tmp_path,
        {"id": "load", "command": "load_settings", "payload": {}},
    )

    response = next(message for message in messages if message.get("id") == "load")
    unsupported = response["payload"]["unsupportedFeatures"]
    assert "サマリー構成" not in unsupported
    assert "動画連結設定" not in unsupported
    assert "定期スキャン" not in unsupported
    assert "RTSP時間制限" not in unsupported
    assert "RTSP固定パターン補正" not in unsupported
    assert "RTSP検出通知音" not in unsupported
    assert "RTSPプリセット" not in unsupported
    assert "RTSPフレームレート" not in unsupported
    assert "NoiseTwin / 時間平均" not in unsupported
    assert "RTSP時間平均保存" not in unsupported


def test_periodic_scan_accepts_a_directory_and_cancel_request(tmp_path):
    messages = run_bridge(
        tmp_path,
        {"id": "periodic", "command": "run_periodic", "payload": {"directory": str(tmp_path)}},
        {"id": "cancel", "command": "cancel", "payload": {}},
    )

    responses = {message["id"]: message for message in messages if message["type"] == "response"}
    assert responses["periodic"]["ok"] is True
    assert responses["cancel"]["ok"] is True


def test_periodic_scan_emits_cancelled_state(tmp_path):
    from swift_backend import Bridge

    output = io.StringIO()
    bridge = Bridge(tmp_path, protocol_stdout=output)
    bridge.handle(
        {
            "id": "periodic",
            "command": "run_periodic",
            "payload": {"directory": str(tmp_path), "scanInterval": 5},
        }
    )
    bridge.handle({"id": "cancel", "command": "cancel", "payload": {}})
    worker = bridge._run_thread
    assert worker is not None
    worker.join(timeout=10)
    assert not worker.is_alive()
    messages = [json.loads(line) for line in output.getvalue().splitlines() if line.strip()]
    assert any(
        message.get("type") == "event"
        and message.get("event") == "run_state"
        and message.get("payload", {}).get("state") == "cancelled"
        for message in messages
    )


def test_rtsp_payload_is_normalized_before_worker_starts(tmp_path):
    import numpy as np

    from swift_backend import Bridge

    output = io.StringIO()
    bridge = Bridge(tmp_path, protocol_stdout=output)
    captured = {}
    fixed_pattern_path = tmp_path / "rtsp_dark_frame.npz"
    np.savez(fixed_pattern_path, fixed_correction=np.zeros((2, 2), dtype=np.int16))

    def fake_run(payload, cancel_event):
        captured.update(payload)

    bridge._run_rtsp = fake_run
    bridge.handle(
        {
            "id": "rtsp",
            "command": "run_rtsp",
            "payload": {
                "url": "rtsp://camera/live",
                "timeLimitEnabled": True,
                "startHour": 18,
                "startMinute": 5,
                "endHour": 6,
                "endMinute": 40,
                "notifyOnDetection": False,
                "applyFixedPattern": True,
                "fixedPatternPath": str(fixed_pattern_path),
                "summaryConfig": [
                    {"name": "Composite Image", "enabled": False, "duration": 3.5},
                    {"name": "Full Size Video", "enabled": True},
                ],
                "rtspPreset": "clear",
                "rtspFps": 30,
            },
        }
    )

    worker = bridge._run_thread
    assert worker is not None
    worker.join(timeout=3)
    assert not worker.is_alive()
    messages = [json.loads(line) for line in output.getvalue().splitlines() if line.strip()]
    response = next(message for message in messages if message.get("id") == "rtsp")
    assert response["ok"] is True
    assert captured["url"] == "rtsp://camera/live"
    assert captured["timeLimitEnabled"] is True
    assert captured["startHour"] == 18
    assert captured["startMinute"] == 5
    assert captured["endHour"] == 6
    assert captured["endMinute"] == 40
    assert captured["notifyOnDetection"] is False
    assert captured["rtspPreset"] == "clear"
    assert captured["rtspFps"] == 30
    assert captured["applyFixedPattern"] is True
    assert captured["fixedPatternPath"] == str(fixed_pattern_path.resolve())
    assert captured["summaryConfig"] == [
        {"name": "Composite Image", "enabled": False, "duration": 3.5},
        {"name": "Full Size Video", "enabled": True},
    ]


def test_local_payload_is_normalized_before_worker_starts(tmp_path):
    from swift_backend import Bridge

    output = io.StringIO()
    bridge = Bridge(tmp_path, protocol_stdout=output)
    captured = {}
    model_path = tmp_path / "custom_detector.pth"
    model_path.write_bytes(b"placeholder")
    wcs_path = tmp_path / "camera.wcs"
    wcs_path.write_text("placeholder", encoding="utf-8")

    def fake_run(payload, cancel_event):
        captured.update(payload)

    bridge._run_pipeline = fake_run
    bridge.handle(
        {
            "id": "local",
            "command": "run_detection",
            "payload": {
                "sources": [{"path": str(tmp_path / "sample.mp4")}],
                "modelPath": str(model_path),
                "plateSolveWCSPath": str(wcs_path),
                "summaryConfig": [
                    {"name": "Zoom Sequence", "enabled": True, "duration": 4.0}
                ],
                "noiseTwinOptions": {
                    "temporalMeanFrames": 3,
                    "saveTemporalMeanVideo": False,
                },
            },
        }
    )

    worker = bridge._run_thread
    assert worker is not None
    worker.join(timeout=3)
    assert not worker.is_alive()
    messages = [json.loads(line) for line in output.getvalue().splitlines() if line.strip()]
    response = next(message for message in messages if message.get("id") == "local")
    assert response["ok"] is True
    assert captured["modelPath"] == str(model_path.resolve())
    assert captured["globalWCSInfo"] == {
        "wcs_file": str(wcs_path.resolve()),
        "job_id": "manual-wcs",
    }
    assert captured["summaryConfig"] == [
        {"name": "Zoom Sequence", "enabled": True, "duration": 4.0}
    ]
    assert captured["noiseTwinOptions"] == {
        "enabled": False,
        "model_path": "",
        "require_validated": True,
        "temporal_mean_frames": 3,
        "save_temporal_mean_video": False,
    }


def test_video_concat_payload_validates_ordered_files_and_options(tmp_path):
    from swift_backend import Bridge

    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mov"
    first.write_bytes(b"placeholder")
    second.write_bytes(b"placeholder")
    output = tmp_path / "joined.mp4"
    bridge = Bridge(tmp_path)

    normalized = bridge._validated_video_concat_payload(
        {
            "inputFiles": [str(first), str(second)],
            "outputPath": str(output),
            "bitrate": "4000k",
            "codec": "h265",
            "fps": "30",
            "safeMode": False,
            "timestampSettings": {
                "enabled": True,
                "position": "左上",
                "size_percent": "2.2",
                "offset_seconds": "-1.5",
            },
        }
    )

    assert normalized["inputFiles"] == [str(first.resolve()), str(second.resolve())]
    assert normalized["outputPath"] == str(output.resolve())
    assert normalized["codec"] == "h265"
    assert normalized["fps"] == 30.0
    assert normalized["safeMode"] is False
    assert normalized["timestampSettings"]["position"] == "左上"
    assert normalized["timestampSettings"]["offset_seconds"] == -1.5

    try:
        bridge._validated_video_concat_payload(
            {
                "inputFiles": [str(first), str(first)],
                "outputPath": str(output),
            }
        )
    except ValueError as exc:
        assert "重複" in str(exc)
    else:
        raise AssertionError("duplicate video inputs should fail")

    try:
        bridge._validated_video_concat_payload(
            {
                "inputFiles": [str(first), str(second)],
                "outputPath": str(tmp_path / "joined.txt"),
            }
        )
    except ValueError as exc:
        assert "動画形式" in str(exc)
    else:
        raise AssertionError("non-video output should fail")


def test_video_concat_worker_emits_progress_and_terminal_result(tmp_path):
    from swift_backend import Bridge

    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.write_bytes(b"placeholder")
    second.write_bytes(b"placeholder")
    payload = Bridge(tmp_path)._validated_video_concat_payload(
        {
            "inputFiles": [str(first), str(second)],
            "outputPath": str(tmp_path / "joined.mp4"),
        }
    )
    fake_module = ModuleType("video_processor")
    captured = {}

    def fake_concatenate_videos(**kwargs):
        captured.update(kwargs)
        kwargs["progress_callback"](0.4, "検証中")
        kwargs["progress_callback"](1.0, "完了")
        Path(kwargs["output_path"]).write_bytes(b"fake-video")
        return True, "fake output"

    fake_module.concatenate_videos = fake_concatenate_videos
    protocol = io.StringIO()
    bridge = Bridge(tmp_path, protocol_stdout=protocol)
    with mock.patch.dict(sys.modules, {"video_processor": fake_module}):
        bridge._run_video_concat(payload, threading.Event())

    messages = [json.loads(line) for line in protocol.getvalue().splitlines() if line.strip()]
    progress = [message for message in messages if message.get("event") == "video_concat_progress"]
    result = [message for message in messages if message.get("event") == "video_concat_result"]
    assert [item["payload"]["fraction"] for item in progress] == [0.4, 1.0]
    assert result[0]["payload"]["success"] is True
    assert result[0]["payload"]["outputPath"] == str((tmp_path / "joined.mp4").resolve())
    assert captured["safe_mode"] is True
    assert captured["timestamp_settings"]["enabled"] is True


def test_video_concat_cancellation_does_not_delete_existing_output(tmp_path):
    from swift_backend import Bridge

    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    output_path = tmp_path / "joined.mp4"
    first.write_bytes(b"placeholder")
    second.write_bytes(b"placeholder")
    output_path.write_bytes(b"existing-user-output")
    payload = Bridge(tmp_path)._validated_video_concat_payload(
        {
            "inputFiles": [str(first), str(second)],
            "outputPath": str(output_path),
        }
    )
    fake_module = ModuleType("video_processor")

    def fake_concatenate_videos(**kwargs):
        assert kwargs["output_path"] != str(output_path.resolve())
        return False, "処理がキャンセルされました"

    fake_module.concatenate_videos = fake_concatenate_videos
    cancel_event = threading.Event()
    cancel_event.set()
    protocol = io.StringIO()
    bridge = Bridge(tmp_path, protocol_stdout=protocol)
    with mock.patch.dict(sys.modules, {"video_processor": fake_module}):
        bridge._run_video_concat(payload, cancel_event)

    assert output_path.read_bytes() == b"existing-user-output"
    messages = [json.loads(line) for line in protocol.getvalue().splitlines() if line.strip()]
    result = next(message for message in messages if message.get("event") == "video_concat_result")
    assert result["payload"]["cancelled"] is True
