import json
import io
import os
import subprocess
import sys
from pathlib import Path


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
            "payload": {"url": "rtsp://camera/live", "notifyOnDetection": "false"},
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
                "rtsp_time_limit_enabled": True,
                "rtsp_notification_sound": False,
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
    assert "サマリー構成" in unsupported
    assert "動画連結設定" in unsupported
    assert "定期スキャン" not in unsupported
    assert "RTSP時間制限" not in unsupported
    assert "RTSP検出通知音" not in unsupported


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
    worker.join(timeout=3)
    assert not worker.is_alive()
    messages = [json.loads(line) for line in output.getvalue().splitlines() if line.strip()]
    assert any(
        message.get("type") == "event"
        and message.get("event") == "run_state"
        and message.get("payload", {}).get("state") == "cancelled"
        for message in messages
    )


def test_rtsp_payload_is_normalized_before_worker_starts(tmp_path):
    from swift_backend import Bridge

    output = io.StringIO()
    bridge = Bridge(tmp_path, protocol_stdout=output)
    captured = {}

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
