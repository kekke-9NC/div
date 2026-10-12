"""Create the 30-second full-night constellation/grid timelapse."""

from pathlib import Path

import timelapse_creator
from create_20260820_constellation_timelapse import (
    CALIBRATION,
    ROOT,
    _select_longest_segment_per_minute,
)


OUTPUT = ROOT / "analysis_results" / "20260820-20260821_full_night_grid_constellations_30s.mp4"


def progress(message):
    print(str(message), flush=True)


def main() -> None:
    hour_paths = [
        *[ROOT / "rtsp" / "20260820" / f"{hour:02d}" for hour in range(18, 24)],
        *[ROOT / "rtsp" / "20260821" / f"{hour:02d}" for hour in range(0, 5)],
    ]
    inputs = [
        path
        for hour_path in hour_paths
        for path in _select_longest_segment_per_minute(hour_path)
    ]
    if not inputs:
        raise SystemExit("no valid full-night recordings found")

    timelapse_creator.OUTPUT_FPS = 15
    successful = timelapse_creator.create_timelapse(
        [str(path) for path in inputs],
        str(OUTPUT),
        target_duration_seconds=30,
        output_fps=15,
        progress_callback=progress,
        timestamp_settings={
            "enabled": True,
            "position": "bottom_right",
            "size_percent": 1.8,
        },
        temporal_mean_radius_frames=50,
        annotation_settings={
            "enabled": True,
            "calibration_path": str(CALIBRATION),
            "draw_grid": True,
            "draw_constellations": True,
            "draw_detected_stars": False,
            "constellation_temporal_hold_frames": 3,
        },
        meteor_insert_settings={"enabled": False},
    )
    if not successful:
        raise SystemExit("timelapse creation failed")
    print(f"RESULT {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
