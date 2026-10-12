"""Audit every constellation overlay frame against the trajectory model."""

from datetime import datetime
import json
from pathlib import Path

import cv2
import numpy as np

import local_wideangle_astrometry as astrometry
import timelapse_creator


ROOT = Path(__file__).resolve().parent
VIDEO = Path(
    "/Users/keisukematsumoto/Downloads/"
    "20260812-20260813_star-trajectory_constellations_timelapse.mp4"
)
OLD_VIDEO = Path(
    "/Users/keisukematsumoto/Downloads/"
    "20260812-20260813_star-trajectory_constellations_timelapse_pre-stability.mp4"
)
MODEL = Path(
    "/Users/keisukematsumoto/Library/Caches/MeteorDetector/astrometry/"
    "camera_models/trajectory-68ebd9385bc5-20260813-1920x1080/"
    "camera_model.json"
)
REPORT = Path(
    "/Users/keisukematsumoto/Downloads/"
    "20260812-20260813_constellation_render_audit.json"
)


def _blue_mask(frame: np.ndarray) -> np.ndarray:
    blue, green, red = cv2.split(frame)
    blue_i = blue.astype(np.int16)
    green_i = green.astype(np.int16)
    red_i = red.astype(np.int16)
    return (
        (blue_i > 150)
        & (blue_i > green_i + 25)
        & (green_i > red_i + 20)
    ).astype(np.uint8)


def _continuity_metrics(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    counts = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        counts.append(int(np.count_nonzero(_blue_mask(frame))))
    capture.release()
    present = np.asarray(counts, dtype=int) >= 30
    gaps = []
    index = 0
    while index < len(present):
        if present[index]:
            index += 1
            continue
        start = index
        while index < len(present) and not present[index]:
            index += 1
        gaps.append([start, index - 1, index - start])
    short_bounded_gaps = [
        gap for gap in gaps
        if gap[0] > 0 and gap[1] < len(present) - 1 and gap[2] <= 8
    ]
    severe_single_frame_drops = []
    for index in range(1, len(counts) - 1):
        neighbor_minimum = min(counts[index - 1], counts[index + 1])
        if neighbor_minimum >= 100 and counts[index] < 0.25 * neighbor_minimum:
            severe_single_frame_drops.append(index)
    return {
        "decoded_frame_count": len(counts),
        "frames_with_blue_lines": int(np.count_nonzero(present)),
        "minimum_blue_pixel_count": int(min(counts)) if counts else None,
        "median_blue_pixel_count": float(np.median(counts)) if counts else None,
        "zero_blue_runs": gaps,
        "short_bounded_zero_gaps_up_to_8_frames": short_bounded_gaps,
        "severe_single_frame_drop_indices": severe_single_frame_drops,
    }


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

    metadata, wcs = astrometry._load_calibration(str(MODEL))
    reference = datetime.fromisoformat(
        str(metadata["reference_datetime"]).replace("Z", "+00:00")
    )
    grid = astrometry._forward_grid_model(wcs, metadata, 1920, 1080)
    display_support = grid.get("display_support_mask", grid["support_mask"])
    detection_gated = str(
        metadata.get("constellation_render_policy", "") or ""
    ).strip().lower() == "model-supported-detected-endpoints"

    temporal_hold_frames = int(
        metadata.get("constellation_temporal_hold_frames", 0) or 0
    ) if detection_gated else 0
    detected_stars_by_frame = []
    edge_admissions_by_frame = []
    raw_edge_presence = []
    if detection_gated:
        for output_index, global_index in enumerate(sample_indices):
            mean_frame = loader.load_temporal_mean_frame(
                global_index, (1920, 1080), radius=50,
            )
            if mean_frame is None:
                raise RuntimeError(f"時間平均を作成できません: {output_index}")
            gray = cv2.cvtColor(mean_frame, cv2.COLOR_BGR2GRAY)
            detected_stars, _diagnostic, _reference = astrometry._extract_stars(
                gray,
                maximum_stars=1000,
                exclude_lower_region=False,
                build_diagnostic=False,
            )
            detected_stars_by_frame.append(np.asarray(detected_stars, dtype=float))
            timestamp = loader.timestamp_for_index(global_index)
            delta_ra = (
                (timestamp - reference).total_seconds()
                / astrometry.SIDEREAL_DAY_SECONDS
                * 360.0
            )
            raw_edge_presence.append(
                astrometry._constellation_segment_detection_mask(
                    wcs,
                    delta_ra,
                    display_support,
                    detected_stars,
                    anchor_tolerance_px=float(
                        metadata.get("constellation_anchor_tolerance_px", 6.0)
                    ),
                )
            )
        raw_edge_presence = np.asarray(raw_edge_presence, dtype=bool)
        bridged_edge_presence = astrometry._bridge_boolean_gaps(
            raw_edge_presence.T, temporal_hold_frames
        ).T
        edge_admissions_by_frame = [
            bridged_edge_presence[index].copy()
            for index in range(len(sample_indices))
        ]

    capture = cv2.VideoCapture(str(VIDEO))
    expected_recalls = []
    component_recalls = []
    missing_components = []
    expected_pixel_counts = []
    actual_pixel_counts = []
    output_frame = 0
    dilation_kernel = np.ones((5, 5), dtype=np.uint8)
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if output_frame >= len(sample_indices):
            raise RuntimeError("動画フレーム数がサンプル時刻数を超えました")
        timestamp = loader.timestamp_for_index(sample_indices[output_frame])
        detected_stars = (
            detected_stars_by_frame[output_frame]
            if detection_gated else None
        )
        delta_ra = (
            (timestamp - reference).total_seconds()
            / astrometry.SIDEREAL_DAY_SECONDS
            * 360.0
        )
        expected_layer = np.zeros_like(frame)
        astrometry._draw_constellation_lines(
            expected_layer,
            wcs,
            delta_ra,
            display_support,
            anchor_points=(
                np.asarray(detected_stars, dtype=float)
                if detected_stars is not None else None
            ),
            anchor_tolerance_px=float(metadata.get("constellation_anchor_tolerance_px", 6.0)),
            pixel_offset=(0.0, 0.0),
            allow_partial_segments=not detection_gated,
            edge_admission=(
                edge_admissions_by_frame[output_frame]
                if detection_gated else None
            ),
        )
        expected = _blue_mask(expected_layer)
        expected[display_support == 0] = 0
        # The finished timelapse intentionally places opaque captions above
        # the astrometry layer.  They are not missing constellation strokes,
        # so exclude only those two fixed UI regions from render recall.
        expected[:48, :760] = 0
        expected[1035:, 1500:] = 0
        actual = _blue_mask(frame)
        actual_dilated = cv2.dilate(actual, dilation_kernel)
        expected_count = int(np.count_nonzero(expected))
        actual_count = int(np.count_nonzero(actual))
        expected_pixel_counts.append(expected_count)
        actual_pixel_counts.append(actual_count)
        recall = (
            float(np.count_nonzero(expected & actual_dilated)) / expected_count
            if expected_count else 1.0
        )
        expected_recalls.append(recall)

        component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            expected, connectivity=8
        )
        for component in range(1, component_count):
            area = int(stats[component, cv2.CC_STAT_AREA])
            if area < 30:
                continue
            component_mask = labels == component
            component_recall = float(
                np.count_nonzero(component_mask & (actual_dilated > 0)) / area
            )
            component_recalls.append(component_recall)
            if component_recall < 0.80:
                missing_components.append({
                    "output_frame": output_frame,
                    "timestamp": timestamp.isoformat(),
                    "area_px": area,
                    "recall": component_recall,
                })
        output_frame += 1
    capture.release()
    loader.cleanup()

    report = {
        "video": str(VIDEO),
        "model": str(MODEL),
        "render_policy": metadata.get("constellation_render_policy"),
        "support_policy": metadata.get("constellation_support_policy"),
        "detection_gated": detection_gated,
        "temporal_hold_frames": temporal_hold_frames,
        "temporal_hold_added_edge_frames": int(
            np.count_nonzero(
                np.asarray(edge_admissions_by_frame, dtype=bool)
                & ~raw_edge_presence
            )
        ) if detection_gated and len(raw_edge_presence) else 0,
        "new_video_continuity": _continuity_metrics(VIDEO),
        "old_video_continuity": _continuity_metrics(OLD_VIDEO),
        "model_render_comparison": {
            "checked_frame_count": output_frame,
            "minimum_expected_blue_pixel_count": int(min(expected_pixel_counts)),
            "minimum_frame_pixel_recall": float(min(expected_recalls)),
            "median_frame_pixel_recall": float(np.median(expected_recalls)),
            "checked_component_count": len(component_recalls),
            "minimum_component_recall": float(min(component_recalls)),
            "median_component_recall": float(np.median(component_recalls)),
            "components_below_80_percent_recall": missing_components,
        },
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
