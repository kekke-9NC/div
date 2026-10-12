"""Promote the validated trajectory model to stable constellation rendering."""

from datetime import datetime
import json
from pathlib import Path
import shutil

import numpy as np

from camera_model_builder import _write_json_atomic
from local_wideangle_astrometry import _bridge_support_grid_for_display


MODEL_DIR = Path(
    "/Users/keisukematsumoto/Library/Caches/MeteorDetector/astrometry/"
    "camera_models/trajectory-68ebd9385bc5-20260813-1920x1080"
)


def _upgrade(payload):
    strict = np.asarray(payload.get("support_grid") or [], dtype=np.uint8)
    display = _bridge_support_grid_for_display(strict) if strict.size else strict
    payload.update({
        "model_label": "VIDEO STAR-TRAJECTORY + GAIA / STABLE CONSTELLATIONS",
        "model_revision": "video-star-trajectories-v3-stable-constellations",
        "verified_constellation_only": False,
        "constellation_anchor_filter": False,
        "constellation_star_alignment": False,
        "constellation_render_policy": "model-supported-continuous",
        "constellation_support_policy": "display-bridged-internal-holes",
        "constellation_rendering_review": {
            "reviewed_at": datetime.now().astimezone().isoformat(),
            "reason": "remove frame-local cloud flicker and supported partial-edge omissions",
            "projection_evidence": {
                "trajectory_train_p95_px": 1.2808927719893943,
                "trajectory_holdout_p95_px": 1.2125605961641537,
                "independent_gaia_p95_px": 2.6620044196711854,
                "gaia_correct_match_total": 1589,
                "gaia_five_minute_control_total": 256,
            },
            "rejected_model": {
                "label": "CURRENT TRIPOD 2026-08-13 00:30 / 80% VERIFIED",
                "reason": "retired-extrapolation-failure at image edges",
            },
            "original_timelapse_flicker": {
                "short_bounded_zero_runs_le_8_frames": 22,
                "single_frame_severe_drops": 10,
                "output_fps": 15,
            },
            "strict_support_fraction": float(np.mean(strict > 0)) if strict.size else 0.0,
            "display_bridged_support_fraction": float(np.mean(display > 0)) if display.size else 0.0,
            "policy": (
                "project every catalog constellation edge at its frame timestamp; "
                "clip only to the internally bridged model support; allow supported "
                "partial runs when the other endpoint lies outside support"
            ),
        },
    })
    return payload


def main():
    model_path = MODEL_DIR / "camera_model.json"
    model_backup = MODEL_DIR / "camera_model.pre-stable-constellations.json"
    if not model_backup.exists():
        shutil.copy2(model_path, model_backup)
    model = _upgrade(json.loads(model_path.read_text(encoding="utf-8")))
    _write_json_atomic(model_path, model)

    report_path = MODEL_DIR / "build_report.json"
    if report_path.exists():
        report_backup = MODEL_DIR / "build_report.pre-stable-constellations.json"
        if not report_backup.exists():
            shutil.copy2(report_path, report_backup)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if isinstance(report.get("model"), dict):
            report["model"] = _upgrade(report["model"])
        report["constellation_rendering_review"] = model["constellation_rendering_review"]
        _write_json_atomic(report_path, report)

    print(model_path)
    print(json.dumps(model["constellation_rendering_review"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
