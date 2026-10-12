"""Render selected raw-night previews after constellation projection fixes."""

from pathlib import Path

import cv2

import local_wideangle_astrometry as astrometry
import timelapse_creator


ROOT = Path(__file__).resolve().parent
MODEL = Path(
    "/Users/keisukematsumoto/Library/Caches/MeteorDetector/astrometry/"
    "camera_models/trajectory-68ebd9385bc5-20260813-1920x1080/camera_model.json"
)
OUTPUT_ROOT = Path("/Users/keisukematsumoto/Downloads")


def main() -> None:
    timelapse_creator.OUTPUT_FPS = 15
    images, videos = [], []
    for source in (ROOT / "rtsp" / "20260812", ROOT / "rtsp" / "20260813"):
        found_images, found_videos = timelapse_creator.get_files_from_path(str(source))
        images.extend(found_images)
        videos.extend(found_videos)
    total, sources = timelapse_creator.count_total_frames(sorted(images), sorted(videos))
    sample_indices = timelapse_creator.calculate_sample_indices(total, 30)
    loader = timelapse_creator.FrameLoader(sources)
    previews = []
    for output_frame in (210, 240, 278, 390, 449):
        global_index = sample_indices[output_frame]
        timestamp = loader.timestamp_for_index(global_index)
        mean = loader.load_temporal_mean_frame(global_index, (1920, 1080), radius=50)
        annotated = astrometry.annotate_frame(
            mean,
            timestamp,
            calibration_path=str(MODEL),
            draw_grid=True,
            draw_constellations=True,
            draw_detected_stars=False,
        )
        cv2.putText(
            annotated,
            f"output frame {output_frame}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        path = OUTPUT_ROOT / f"constellation-roundtrip-fix-{output_frame:03d}.png"
        cv2.imwrite(str(path), annotated)
        previews.append(cv2.resize(annotated, (768, 432), interpolation=cv2.INTER_AREA))
        print(path, flush=True)
    sheet = cv2.vconcat((cv2.hconcat(previews[:3]), cv2.hconcat(previews[3:] + [previews[-1] * 0])))
    sheet_path = OUTPUT_ROOT / "20260812-20260813_constellation-roundtrip-fix-preview-sheet.jpg"
    cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94])
    print(sheet_path, flush=True)
    loader.cleanup()


if __name__ == "__main__":
    main()
