"""Legacy settings must reach the same processing options in the Swift backend."""

import io
import json
import os
import sys
import threading
from types import ModuleType
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from settings_runtime import RuntimeSettings, from_legacy_settings
from swift_backend import Bridge


def test_legacy_runtime_settings_preserve_processing_controls(tmp_path):
    settings = {
        "advanced_settings": {"min_line_length": "48", "meteor_prob": "0.75"},
        "full_video_timestamp": {"enabled": False, "position": "左上", "size_percent": "2.4"},
        "ml_training_export_enabled": True,
        "ml_training_data_root": str(tmp_path / "training"),
        "auto_video_mask_enabled": False,
        "processed_video_encoding": {
            "codec": "H.264 / AVC", "quality": "高品質", "bitrate_mbps": "70"
        },
        "ai_vlm_backend": "lmstudio_qwen3_5_2b",
        "lm_studio_vlm_url": "http://localhost:1234/v1",
        "lm_studio_vlm_model_id": "private-model",
        "lm_studio_vlm_api_key": "secret-for-test",
        "astrometry_api_key": "astrometry-secret",
        "use_plate_solve": True,
        "plate_solve_mode": "local",
    }
    runtime = from_legacy_settings(settings, tmp_path)
    assert runtime.advanced_config["MIN_LINE_LENGTH"] == 48
    assert runtime.advanced_config["METEOR_PROBABILITY_THRESHOLD"] == 0.75
    assert runtime.encoding == {"codec": "h264", "quality": "high", "bitrate_mbps": 70}
    assert runtime.save_option_extras["full_video_timestamp_position"] == "左上"
    assert runtime.save_option_extras["ml_training_export_enabled"] is True
    assert runtime.save_option_extras["auto_video_mask_enabled"] is False
    assert runtime.lm_studio_api_key == "secret-for-test"
    fake_config = SimpleNamespace(ASTROMETRY_API_KEY="old")
    with runtime.applied_config(fake_config):
        assert fake_config.ASTROMETRY_API_KEY == "astrometry-secret"
        assert fake_config.MIN_LINE_LENGTH == 48
    assert fake_config.ASTROMETRY_API_KEY == "old"
    assert not hasattr(fake_config, "MIN_LINE_LENGTH")
    assert runtime.use_plate_solve is True


@pytest.mark.parametrize("settings,field", [
    ({"advanced_settings": {"min_line_length": "NaN"}}, "advanced_settings.min_line_length"),
    ({"full_video_timestamp": {"size_percent": "999"}}, "full_video_timestamp.size_percent"),
    ({"processed_video_encoding": {"codec": "bad"}}, "processed_video_encoding.codec"),
    ({"ml_training_export_enabled": True, "ml_training_data_root": ""}, "保存先"),
])
def test_invalid_legacy_settings_report_the_field_without_echoing_values(tmp_path, settings, field):
    settings["unknown_secret"] = "do-not-log-me"
    with pytest.raises(ValueError, match=field) as error:
        from_legacy_settings(settings, tmp_path)
    assert "do-not-log-me" not in str(error.value)


def test_old_worker_count_is_not_silently_clamped_to_six(tmp_path):
    output = io.StringIO()
    bridge = Bridge(tmp_path, protocol_stdout=output)
    received = {}
    bridge._run_pipeline = lambda payload, _cancel: received.update(payload)
    with mock.patch("swift_backend.os.cpu_count", return_value=12):
        bridge._start_run("run", {
            "sources": [{"path": str(tmp_path / "video.mp4")}],
            "maxWorkers": 8,
            "interval": 60.0,
            "duration": 60.0,
            "legacySettings": {"advanced_settings": {"min_line_length": "35"}},
        })
    bridge._run_thread.join(timeout=2)
    assert received["maxWorkers"] == 8
    assert received["interval"] == 60.0
    assert received["duration"] == 60.0
    assert received["_runtimeSettings"].advanced_config["MIN_LINE_LENGTH"] == 35
    assert json.loads(output.getvalue().splitlines()[-1])["ok"] is True


def test_runtime_options_reach_all_three_pipeline_modes(tmp_path, monkeypatch):
    import numpy as np

    np.savez(tmp_path / "app_masks.npz", plate_solve_mask_image=np.ones((2, 2), dtype=np.uint8))
    settings = {
        "advanced_settings": {"min_line_length": "39"},
        "has_plate_solve_mask_image": True,
        "detection_mask_path": str(tmp_path / "app_masks.npz"),
        "full_video_timestamp": {"enabled": False, "position": "右上", "size_percent": "2.0"},
        "auto_video_mask_enabled": True,
        "processed_video_encoding": {
            "codec": "H.265 / HEVC (推奨)", "quality": "標準", "bitrate_mbps": "40"
        },
    }
    captured = {}
    fake_config = SimpleNamespace(MIN_LINE_LENGTH=25, RTSP_SEGMENT_DURATION=60)
    fake_download = SimpleNamespace(run_pipeline=lambda **kwargs: captured.update(local=kwargs))
    fake_file_utils = SimpleNamespace(
        monitor_directory=lambda **kwargs: captured.update(periodic=kwargs),
        rtsp_save_and_process_thread_target=lambda **kwargs: captured.update(rtsp=kwargs),
    )
    monkeypatch.setattr(RuntimeSettings, "configure_ai_backend", lambda _self: None)
    bridge = Bridge(tmp_path, protocol_stdout=io.StringIO())
    common = {
        "legacySettings": settings,
        "saveOptions": Bridge._default_save_options(),
        "summaryConfig": Bridge._default_summary_config(),
        "noiseTwinOptions": {"enabled": False},
        "meteorSavePath": str(tmp_path / "meteor"),
        "notMeteorSavePath": str(tmp_path / "not_meteor"),
    }
    with mock.patch.dict(sys.modules, {
        "config": fake_config, "download_pipeline": fake_download, "file_utils": fake_file_utils,
    }):
        for mode, extra in (
            ("local", {"sources": [{"path": str(tmp_path / "sample.mp4")}] }),
            ("periodic", {"directory": str(tmp_path), "scanInterval": 1}),
            ("rtsp", {"url": "rtsp://example.test/stream"}),
        ):
            payload = {**common, **extra}
            bridge._prepare_runtime_payload(payload)
            getattr(bridge, {
                "local": "_run_pipeline", "periodic": "_run_periodic", "rtsp": "_run_rtsp"
            }[mode])(payload, threading.Event())
            arguments = captured[mode]
            assert arguments["save_options"]["full_video_timestamp_enabled"] is False
            assert arguments["save_options"]["auto_video_mask_enabled"] is True
            assert arguments["noise_twin_options"]["encoding"]["quality"] == "standard"
            assert arguments["plate_solve_mask"].shape == (2, 2)
            assert fake_config.MIN_LINE_LENGTH == 25
    assert captured["periodic"]["min_length"] == 39
    assert captured["rtsp"]["min_length"] == 39


def test_auto_time_update_persists_periodic_window_and_emits_event(tmp_path):
    output = io.StringIO()
    bridge = Bridge(tmp_path, protocol_stdout=output)
    updates = []
    bridge._save_settings = lambda values: updates.append(values)
    bridge._on_auto_time_update(18, 5, 6, 40)
    expected = {
        "periodic_start_hour": "18", "periodic_start_minute": "05",
        "periodic_end_hour": "06", "periodic_end_minute": "40",
    }
    assert updates == [expected]
    event = json.loads(output.getvalue())
    assert event["event"] == "settings_updated"
    assert event["payload"]["settings"] == expected


@pytest.mark.parametrize("path_key", ["wcs_file", "model_path", "calibration_path"])
def test_disabled_plate_solve_ignores_stale_path_and_saved_info_is_restored(tmp_path, path_key):
    bridge = Bridge(tmp_path, protocol_stdout=io.StringIO())
    stale = str(tmp_path / "missing.wcs")
    assert bridge._wcs_path_from_payload({
        "plateSolveWCSPath": stale,
        "legacySettings": {"use_plate_solve": False, "plate_solve_wcs_path": stale},
    }) is None
    path = tmp_path / "camera.wcs"
    path.write_text("placeholder", encoding="utf-8")
    assert bridge._wcs_path_from_payload({
        "plateSolveWCSPath": stale,
        "legacySettings": {"global_wcs_info": {path_key: str(path)}},
    }) == str(path)
    payload = {
        "globalWCSInfo": bridge._validated_wcs_info(str(path)),
        "saveOptions": Bridge._default_save_options(),
        "noiseTwinOptions": {"enabled": False},
        "legacySettings": {
            "use_plate_solve": True,
            "global_wcs_info": {
                path_key: str(path), "job_id": "local-wideangle-cache",
                "plate_solve_datetime": "2026-10-09T21:00:00",
                "calibration_note": "previously measured",
            },
        },
    }
    bridge._prepare_runtime_payload(payload)
    assert payload["globalWCSInfo"]["job_id"] == "local-wideangle-cache"
    assert payload["globalWCSInfo"]["plate_solve_datetime"].year == 2026
    assert payload["globalWCSInfo"]["calibration_note"] == "previously measured"


def test_camera_model_trajectory_method_uses_saved_seed(tmp_path):
    import camera_plate_model

    seed = tmp_path / "seed.json"
    seed.write_text(json.dumps({
        "model_type": camera_plate_model.MODEL_TYPE,
        "stg_parameters": [1], "correction_coefficients": [],
        "reference_datetime": "2026-10-09T21:00:00",
    }), encoding="utf-8")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"placeholder")
    fake_builder = ModuleType("camera_model_builder")
    fake_builder.CameraModelBuildRequest = lambda **kwargs: SimpleNamespace(**kwargs)
    fake_builder.build_camera_model = lambda *_args, **_kwargs: pytest.fail("seed must be reused")
    fake_builder.automatic_model_source = lambda value: value
    calls = []
    fake_trajectory = ModuleType("trajectory_camera_model")
    fake_trajectory.TrajectoryBuildRequest = lambda **kwargs: SimpleNamespace(**kwargs)
    fake_trajectory.build_trajectory_camera_model = lambda request, **_kwargs: (
        calls.append(request) or SimpleNamespace(
            success=True, model_path=str(tmp_path / "result.json"),
            as_dict=lambda: {"success": True, "model_path": str(tmp_path / "result.json")},
        )
    )
    bridge = Bridge(tmp_path, protocol_stdout=io.StringIO())
    payload = bridge._validated_camera_model_payload({
        "source": str(source), "method": "動画の星の動き（推奨）",
        "initialModelPath": str(seed), "interval": "47",
    })
    assert payload["monitorInterval"] == 47
    with mock.patch.dict(sys.modules, {
        "camera_model_builder": fake_builder,
        "trajectory_camera_model": fake_trajectory,
    }):
        bridge._build_camera_model_worker("build", payload, threading.Event())
    assert len(calls) == 1
    assert calls[0].initial_model_path == str(seed)
    assert calls[0].source == str(source)


def test_rtsp_camera_model_monitor_uses_legacy_interval_and_ai(tmp_path):
    recorded = {}
    fake_monitor = ModuleType("camera_model_monitor")

    class Monitor:
        def __init__(self, *_args, **kwargs):
            recorded.update(kwargs)
        def start(self):
            recorded["started"] = True
        def stop(self):
            recorded["stopped"] = True

    fake_monitor.RTSPCameraModelMonitor = Monitor
    bridge = Bridge(tmp_path, protocol_stdout=io.StringIO())
    with mock.patch.dict(sys.modules, {"camera_model_monitor": fake_monitor}):
        bridge._start_camera_model_monitor("monitor", {
            "url": "rtsp://camera.example/live",
            "legacySettings": {
                "camera_model_interval": "37",
                "camera_model_cloud_threshold": "0.08",
                "ai_vlm_backend": "local_qwen3_vl_4b",
            },
        })
    bridge._stop_camera_model_monitor("stop")
    assert recorded["started"] is True
    assert recorded["stopped"] is True
    assert recorded["interval_seconds"] == 37
    assert recorded["cloud_threshold"] == 0.08
    assert recorded["backend"] == "local_qwen3_vl_4b"


def test_camera_model_failure_residuals_remain_valid_json(tmp_path):
    bridge = Bridge(tmp_path, protocol_stdout=io.StringIO())
    cleaned = bridge._json_safe({
        "success": False, "residual_p95_px": float("inf"),
        "diagnostics": [float("nan")],
    })
    bridge.event("camera_model_result", cleaned)
    wire = bridge._protocol_stdout.getvalue()
    assert "Infinity" not in wire and "NaN" not in wire
    assert json.loads(wire)["payload"]["residual_p95_px"] is None
