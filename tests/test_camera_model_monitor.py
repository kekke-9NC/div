from pathlib import Path

from camera_model_monitor import RTSPCameraModelMonitor
from camera_model_builder import CameraModelBuildResult
from cloud_coverage import CloudClassification


def test_monitor_builds_after_clear_probe_and_saved_segments(tmp_path):
    for minute in (0, 1, 2):
        path = tmp_path / f"20260809_04{minute:02d}.mp4"
        path.write_bytes(b"")
    calls = []

    def classifier(frame, **kwargs):
        return CloudClassification(0.04, "test", 1.0)

    def builder(request, **kwargs):
        calls.append(request)
        return CameraModelBuildResult(True, model_path="model.json", target_met=True)

    monitor = RTSPCameraModelMonitor(
        "rtsp://example", save_root=str(tmp_path), minimum_clear_segments=3,
        classifier=classifier, builder=builder,
    )
    monitor._probe_frame = lambda: object()
    result = monitor.run_once()
    assert result is not None
    assert len(calls) == 1
    assert calls[0].source == str(tmp_path)
    assert monitor.last_classification.cloud_fraction == 0.04
