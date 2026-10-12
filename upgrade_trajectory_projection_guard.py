"""Record the projection-branch guard in the selected trajectory model."""

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
    model_backup = MODEL_DIR / "camera_model.pre-roundtrip-guard.json"
    report_backup = MODEL_DIR / "build_report.pre-roundtrip-guard.json"
    if not model_backup.exists():
        shutil.copy2(model_path, model_backup)
    if report_path.exists() and not report_backup.exists():
        shutil.copy2(report_path, report_backup)

    payload = json.loads(model_path.read_text(encoding="utf-8"))
    payload.update({
        "model_label": "VIDEO STAR-TRAJECTORY + GAIA / BRANCH-GUARDED CONSTELLATIONS",
        "model_revision": "video-star-trajectories-v4-branch-guard",
        "constellation_projection_guard": "sky-pixel-sky-roundtrip",
        "constellation_max_sky_roundtrip_deg": 0.1,
    })
    review = dict(payload.get("constellation_rendering_review") or {})
    review["projection_branch_review"] = {
        "reviewed_at": datetime.now().astimezone().isoformat(),
        "failure_mode": (
            "a locally many-to-one degree-5 detector polynomial mapped short sky edges "
            "onto a wrong image branch"
        ),
        "examples_before_guard": [
            {
                "constellation": "Leo",
                "angular_length_deg": 2.7279509057069324,
                "drawn_length_px": 457.7873943402369,
                "timestamp": "2026-08-13T01:18:30.520000",
            },
            {
                "constellation": "UMa",
                "angular_length_deg": 15.578231125096407,
                "drawn_supported_run_px": 741.8018571460896,
                "timestamp": "2026-08-13T03:20:16.800000",
            },
        ],
        "guard": "reject every dense line sample whose sky->pixel->sky error exceeds 0.1 degree",
    }
    payload["constellation_rendering_review"] = review
    _write(model_path, payload)

    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["model"] = payload
        report["projection_branch_review"] = review["projection_branch_review"]
        _write(report_path, report)
    print(model_path)


if __name__ == "__main__":
    main()
