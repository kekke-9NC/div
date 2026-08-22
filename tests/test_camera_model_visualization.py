import json

import cv2
import numpy as np

from camera_model_visualization import render_camera_model_visualization


def test_render_camera_model_visualization_writes_composite_png(tmp_path):
    model_path = tmp_path / "camera_models" / "test" / "camera_model.json"
    model_path.parent.mkdir(parents=True)
    payload = {
        "model_type": "fixed-camera-stg-poly",
        "model_label": "VIDEO STAR-TRAJECTORY CAMERA MODEL",
        "width": 320,
        "height": 180,
        "polynomial_degree": 2,
        "stg_parameters": [0.0, 0.0, 0.0, np.log(130.0), np.log(100.0), 160.0, 90.0],
        "correction_coefficients": np.zeros((6, 2)).tolist(),
        "reference_datetime": "2026-08-21T19:08:40",
        "support_grid": [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]],
        "support_fraction": 4 / 12,
        "enabled": True,
        "target_met": False,
        "fit_stats": {"residual_p95_px": 1.5, "holdout_residual_p95_px": 1.7},
        "trajectory_validation": {"track_count": 12, "raw_track_count": 15},
    }
    model_path.write_text(json.dumps(payload), encoding="utf-8")
    frame_path = tmp_path / "reference.png"
    frame = np.full((180, 320, 3), 35, dtype=np.uint8)
    cv2.circle(frame, (160, 90), 4, (255, 255, 255), -1)
    assert cv2.imwrite(str(frame_path), frame)

    output_path = tmp_path / "camera_model_visualization.png"
    result = render_camera_model_visualization(
        str(model_path), output_path=str(output_path), frame_path=str(frame_path)
    )

    assert result == str(output_path.resolve())
    rendered = cv2.imread(result, cv2.IMREAD_COLOR)
    assert rendered is not None
    assert rendered.shape[:2] == (1060, 1800)
    assert int(rendered.max()) > 0
