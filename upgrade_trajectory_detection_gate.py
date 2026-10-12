"""Switch the selected trajectory model to current-star-gated rendering."""

from datetime import datetime
import json
from pathlib import Path
import shutil


MODEL_DIR = Path(
    "/Users/keisukematsumoto/Library/Caches/MeteorDetector/astrometry/"
    "camera_models/trajectory-68ebd9385bc5-20260813-1920x1080"
)


def _write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    model_path = MODEL_DIR / "camera_model.json"
    report_path = MODEL_DIR / "build_report.json"
    model_backup = MODEL_DIR / "camera_model.pre-detection-gate.json"
    report_backup = MODEL_DIR / "build_report.pre-detection-gate.json"
    if not model_backup.exists():
        shutil.copy2(model_path, model_backup)
    if report_path.exists() and not report_backup.exists():
        shutil.copy2(report_path, report_backup)

    payload = json.loads(model_path.read_text(encoding="utf-8"))
    payload.update({
        "model_label": "VIDEO STAR-TRAJECTORY + GAIA / DETECTION-GATED CONSTELLATIONS",
        "model_revision": "video-star-trajectories-v5-detection-gated",
        "constellation_render_policy": "model-supported-detected-endpoints",
        "constellation_support_policy": "display-bridged-internal-holes",
        "constellation_anchor_filter": True,
        "constellation_anchor_tolerance_px": 6.0,
        "constellation_star_alignment": False,
        "constellation_temporal_hold_frames": 3,
    })
    review = dict(payload.get("constellation_rendering_review") or {})
    review["detection_gate_review"] = {
        "reviewed_at": datetime.now().astimezone().isoformat(),
        "policy": (
            "run star detection on each temporal-mean output frame; draw a line only "
            "when both projected endpoints match detected stars within 6 px"
        ),
        "cloud_behavior": "long cloud gaps suppress an edge; bounded gaps up to 3 output frames are held",
        "partial_segments": "disabled when detection gating is active",
        "temporal_hold_frames": 3,
    }
    payload["constellation_rendering_review"] = review
    _write(model_path, payload)
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["model"] = payload
        _write(report_path, report)
    print(model_path)


if __name__ == "__main__":
    main()
