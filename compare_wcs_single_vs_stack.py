"""Compare single-frame and centered-stack WCS calibration for one reference."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import warnings

import cv2
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS

import local_wideangle_astrometry as astrometry


SOURCE = Path(
    "/Users/keisukematsumoto/Desktop/other/div/rtsp/20260813/00/30.mp4"
)
CENTER_FRAME = 750
HALF_WINDOW_SECONDS = 10.0
OUTPUT = Path(__file__).resolve().parent / "analysis_results" / "wcs_stack_experiment"


def _read_frame(path: Path, frame_index: int):
    cap = astrometry._open_video(str(path))
    if not cap.isOpened():
        raise IOError(f"動画を開けません: {path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise IOError(f"フレームを読めません: {path} #{frame_index}")
    return frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _star_counts(image, samples=None):
    detected, _diagnostic, _reference = astrometry._extract_stars(
        image, samples, maximum_stars=500, build_diagnostic=False
    )
    solve_set, _diagnostic, _reference = astrometry._extract_stars(
        image, samples, maximum_stars=180, build_diagnostic=False
    )
    return {
        "detected_star_count_raw": len(detected),
        "solve_star_count_capped_180": len(solve_set),
    }


def _copy_if_present(source: str | Path, destination: Path) -> None:
    source_path = Path(source).expanduser()
    if source_path.exists():
        shutil.copy2(source_path, destination)


def _make_comparison_image(left_path: Path, right_path: Path, output_path: Path) -> None:
    left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
    right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
    if left is None or right is None:
        return
    height = min(left.shape[0], right.shape[0])
    width = min(left.shape[1], right.shape[1])
    left = cv2.resize(left, (width, height), interpolation=cv2.INTER_AREA)
    right = cv2.resize(right, (width, height), interpolation=cv2.INTER_AREA)
    cv2.putText(left, "single frame", (24, 42), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(right, "centered +/-10s stack", (24, 42), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 255), 2, cv2.LINE_AA)
    comparison = cv2.hconcat([left, right])
    scale = min(1.0, 1920.0 / comparison.shape[1])
    if scale < 1.0:
        comparison = cv2.resize(
            comparison, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    cv2.imwrite(str(output_path), comparison, [cv2.IMWRITE_JPEG_QUALITY, 95])


def _independent_center_frame_check(
    calibration: dict, observed: np.ndarray, image_shape: tuple[int, int]
) -> dict:
    """Check catalog-to-pixel predictions against the untouched center frame."""
    catalog = calibration.get("catalog_stars") or []
    if not catalog:
        return {"catalog_count": 0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with fits.open(calibration["wcs_path"]) as hdul:
            wcs = WCS(hdul[0].header, relax=True, fix=False)
        sky = SkyCoord(
            ra=np.asarray([star["ra_deg"] for star in catalog]) * u.deg,
            dec=np.asarray([star["dec_deg"] for star in catalog]) * u.deg,
            frame="icrs",
        )
        predicted_x, predicted_y = wcs.world_to_pixel(sky)
    predicted = np.column_stack((predicted_x, predicted_y))
    height, width = image_shape
    inside = (
        np.isfinite(predicted).all(axis=1)
        & (predicted[:, 0] >= 0)
        & (predicted[:, 0] < width)
        & (predicted[:, 1] >= 0)
        & (predicted[:, 1] < height)
    )
    predicted = predicted[inside]
    if len(predicted) == 0 or len(observed) == 0:
        return {
            "catalog_count": len(catalog),
            "predicted_in_frame_count": len(predicted),
        }
    distances = np.sqrt(
        ((predicted[:, None, :] - observed[None, :, :]) ** 2).sum(axis=2)
    ).min(axis=1)
    matched = distances[distances <= 10.0]
    return {
        "catalog_count": len(catalog),
        "predicted_in_frame_count": int(len(predicted)),
        "within_10px_count": int(np.sum(distances <= 10.0)),
        "nearest_median_px": float(np.median(distances)),
        "nearest_p95_px": float(np.percentile(distances, 95)),
        "within_10px_median_px": float(np.median(matched)) if len(matched) else None,
        "within_10px_p95_px": float(np.percentile(matched, 95)) if len(matched) else None,
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    baseline_path = Path(
        "/Users/keisukematsumoto/Library/Caches/MeteorDetector/astrometry/calibrations/"
        "68ebd9385bc5/20260813/calibration_frame_750_9b51e2f2.json"
    )
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    single_frame = _read_frame(SOURCE, CENTER_FRAME)
    baseline.update(_star_counts(single_frame))

    stacked_result = astrometry.solve_video_frame_stack_local(
        str(SOURCE),
        CENTER_FRAME,
        half_window_seconds=HALF_WINDOW_SECONDS,
        force=False,
        progress_callback=lambda message: print(message),
    )
    average, samples, stack_info = astrometry._video_centered_stack(
        str(SOURCE), CENTER_FRAME, half_window_seconds=HALF_WINDOW_SECONDS
    )
    # The WCS path uses the mean stack directly.  Passing the short-window
    # temporal median here would suppress slowly moving stars.
    stacked_counts = _star_counts(average, None)

    single_validation = OUTPUT / "single_validation.jpg"
    stack_validation = OUTPUT / "stack_validation.jpg"
    single_reference = OUTPUT / "single_reference.jpg"
    stack_reference = OUTPUT / "stack_reference.jpg"
    _copy_if_present(baseline["validation_path"], single_validation)
    _copy_if_present(stacked_result["validation_path"], stack_validation)
    _copy_if_present(baseline["reference_path"], single_reference)
    _copy_if_present(stacked_result["reference_path"], stack_reference)
    _copy_if_present(stacked_result["wcs_file"], OUTPUT / "stack_wcs.wcs")
    _copy_if_present(stacked_result["calibration_path"], OUTPUT / "stack_calibration.json")
    _make_comparison_image(single_validation, stack_validation, OUTPUT / "validation_comparison.jpg")
    _make_comparison_image(single_reference, stack_reference, OUTPUT / "reference_comparison.jpg")

    metrics = (
        "star_count",
        "detected_star_count_raw",
        "solve_star_count_capped_180",
        "sip_match_count",
        "sip_residual_median_px",
        "sip_residual_p95_px",
        "sip_support_fraction",
        "center_ra_deg",
        "center_dec_deg",
        "scale_arcsec_per_pixel",
    )
    single = {key: baseline.get(key) for key in metrics}
    stack = {key: stacked_result.get(key) for key in metrics}
    stack.update(stacked_counts)
    stack.update(stack_info)
    delta = {}
    for key in metrics:
        if isinstance(single.get(key), (int, float)) and isinstance(stack.get(key), (int, float)):
            delta[key] = stack[key] - single[key]
    report = {
        "source": str(SOURCE),
        "center_frame_index": CENTER_FRAME,
        "center_datetime": stacked_result.get("reference_datetime"),
        "single_frame": single,
        "centered_stack": stack,
        "delta_stack_minus_single": delta,
        "independent_center_frame_check": {
            "single_frame_wcs": _independent_center_frame_check(
                baseline,
                np.asarray(astrometry._extract_stars(
                    single_frame, None, maximum_stars=500, build_diagnostic=False
                )[0], dtype=float),
                single_frame.shape,
            ),
            "centered_stack_wcs": _independent_center_frame_check(
                stacked_result,
                np.asarray(astrometry._extract_stars(
                    single_frame, None, maximum_stars=500, build_diagnostic=False
                )[0], dtype=float),
                single_frame.shape,
            ),
        },
        "files": {
            "single_validation": str(single_validation),
            "stack_validation": str(stack_validation),
            "validation_comparison": str(OUTPUT / "validation_comparison.jpg"),
            "single_reference": str(single_reference),
            "stack_reference": str(stack_reference),
            "reference_comparison": str(OUTPUT / "reference_comparison.jpg"),
            "stack_wcs": str(OUTPUT / "stack_wcs.wcs"),
            "stack_calibration": str(OUTPUT / "stack_calibration.json"),
        },
    }
    (OUTPUT / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
