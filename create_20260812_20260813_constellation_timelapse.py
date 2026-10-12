"""Reproduce the 2026-08-12/13 RTSP constellation timelapse."""

from pathlib import Path

import timelapse_creator


ROOT = Path(__file__).resolve().parent
OUTPUT = Path(
    "/Users/keisukematsumoto/Downloads/"
    "20260812-20260813_star-trajectory_constellations_timelapse.mp4"
)
MODEL = Path(
    "/Users/keisukematsumoto/Library/Caches/MeteorDetector/astrometry/"
    "camera_models/trajectory-68ebd9385bc5-20260813-1920x1080/"
    "camera_model.json"
)


def progress(message):
    print(str(message), flush=True)


def main() -> None:
    # 450 evenly spaced output frames cover the available 8.5-hour night.
    timelapse_creator.OUTPUT_FPS = 15
    successful = timelapse_creator.create_timelapse(
        [str(ROOT / "rtsp" / "20260812"), str(ROOT / "rtsp" / "20260813")],
        str(OUTPUT),
        target_duration_seconds=30,
        progress_callback=progress,
        timestamp_settings={
            "enabled": True,
            "position": "bottom_right",
            "size_percent": 1.8,
        },
        temporal_mean_radius_frames=50,
        annotation_settings={
            "enabled": True,
            "calibration_path": str(MODEL),
            "draw_grid": True,
            "draw_constellations": True,
            "draw_detected_stars": False,
            # Keep a line through only a short, bounded detection dropout;
            # longer cloudy intervals remain suppressed.
            "constellation_temporal_hold_frames": 3,
        },
        meteor_insert_settings={"enabled": False},
    )
    if not successful:
        raise SystemExit("timelapse creation failed")
    print(f"RESULT {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
