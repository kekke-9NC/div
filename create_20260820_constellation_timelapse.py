"""Create a full-night 15-second constellation/grid timelapse."""

from pathlib import Path
from typing import Dict, List

import timelapse_creator


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "analysis_results" / "20260820-20260821_full_night_grid_constellations_15s.mp4"
CALIBRATION = Path(
    "/Users/keisukematsumoto/Library/Caches/MeteorDetector/astrometry/"
    "camera_models/trajectory-68ebd9385bc5-20260813-1920x1080/camera_model.json"
)


def progress(message):
    print(str(message), flush=True)


def _select_longest_segment_per_minute(hour_path: Path) -> List[Path]:
    """Keep one valid recorder segment for each minute, avoiding reconnect duplicates."""
    candidates: Dict[str, tuple[int, int, Path]] = {}
    for path in sorted(hour_path.glob("*.mp4")):
        minute = path.stem.split("_", 1)[0]
        if not minute.isdigit():
            continue
        frame_count = timelapse_creator.get_video_frame_count(str(path))
        if frame_count <= 0:
            continue
        # Prefer the unsuffixed name when lengths are identical; otherwise use
        # the longest valid segment for that recorder minute.
        preference = 1 if "_" not in path.stem else 0
        candidate = (frame_count, preference, path)
        previous = candidates.get(minute)
        if previous is None or candidate[:2] > previous[:2]:
            candidates[minute] = candidate
    return [candidates[key][2] for key in sorted(candidates)]


def main() -> None:
    # One continuous observation night: 2026-08-20 18:52 through the
    # available 2026-08-21 04:37.  The archive has reconnect duplicates in
    # the last hour, so choose one longest valid segment per recorder minute.
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
    progress(f"一晩分の採用セグメント: {len(inputs)}本")

    timelapse_creator.OUTPUT_FPS = 15
    successful = timelapse_creator.create_timelapse(
        [str(path) for path in inputs],
        str(OUTPUT),
        target_duration_seconds=15,
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
            # The local annotator clips grid contours to the calibrated,
            # well-supported region from the camera model.
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
