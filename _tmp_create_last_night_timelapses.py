from pathlib import Path

import timelapse_creator


# The app's native timelapse default is 60 fps.  Fifteen fps is sufficient for
# this long-night astronomy timelapse and keeps local WCS annotation practical.
timelapse_creator.OUTPUT_FPS = 15


ROOT = Path(__file__).resolve().parent
DOWNLOADS = Path("/Users/keisukematsumoto/Downloads")
INPUTS = [ROOT / "rtsp" / "20260810", ROOT / "rtsp" / "20260811"]
CALIBRATION = (
    "/Users/keisukematsumoto/Library/Caches/MeteorDetector/astrometry/"
    "calibrations/68ebd9385bc5/20260811/calibration.json"
)


def progress(message):
    print(str(message), flush=True)


def create(output_name, draw_constellations):
    output_path = DOWNLOADS / output_name
    settings = {
        "enabled": True,
        "calibration_path": CALIBRATION,
        "draw_grid": True,
        "draw_constellations": draw_constellations,
        "draw_detected_stars": False,
    }
    timestamp_settings = {
        "enabled": True,
        "position": "bottom_right",
        "size_percent": 1.8,
    }
    ok = timelapse_creator.create_timelapse(
        [str(path) for path in INPUTS],
        str(output_path),
        target_duration_seconds=30,
        progress_callback=progress,
        timestamp_settings=timestamp_settings,
        temporal_mean_radius_frames=50,
        annotation_settings=settings,
        meteor_insert_settings={"enabled": False},
    )
    print(f"RESULT {output_path} {ok}", flush=True)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    create("20260810-20260811_timelapse_grid.mp4", False)
    create("20260810-20260811_timelapse_grid_constellations.mp4", True)
