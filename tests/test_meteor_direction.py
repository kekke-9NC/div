import cv2
import numpy as np

from meteor_direction import estimate_motion


def _moving_dot_frames(reverse: bool = False):
    frames = []
    for index in range(20):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(frame, (20 + index * 3, 50), 3, (255, 255, 255), -1)
        frames.append(frame)
    line = ((77, 50), (20, 50)) if reverse else ((20, 50), (77, 50))
    return frames, line


def test_estimate_motion_orders_endpoints_by_time():
    frames, line = _moving_dot_frames()

    estimate = estimate_motion(frames, line, frame_start=100, frame_rate=25.0)

    assert estimate.status == "high"
    assert estimate.start_frame < estimate.end_frame
    assert estimate.start_pixel[0] < estimate.end_pixel[0]
    assert estimate.vector_px[0] > 0
    assert estimate.direction_angle_deg == 0.0
    assert estimate.fit_r2 > 0.9


def test_estimate_motion_is_independent_of_hough_endpoint_order():
    frames, forward_line = _moving_dot_frames()
    _frames, reverse_line = _moving_dot_frames(reverse=True)

    forward = estimate_motion(frames, forward_line, frame_start=100, frame_rate=25.0)
    reverse = estimate_motion(frames, reverse_line, frame_start=100, frame_rate=25.0)

    assert forward.status == reverse.status == "high"
    assert np.allclose(forward.start_pixel, reverse.start_pixel, atol=1.0)
    assert np.allclose(forward.end_pixel, reverse.end_pixel, atol=1.0)
    assert np.allclose(forward.vector_px, reverse.vector_px, atol=1.0)
    assert abs(forward.direction_angle_deg - reverse.direction_angle_deg) < 1e-6


def test_estimate_motion_rejects_static_signal():
    frames = []
    for _index in range(12):
        frame = np.zeros((80, 80, 3), dtype=np.uint8)
        cv2.circle(frame, (40, 40), 3, (255, 255, 255), -1)
        frames.append(frame)

    estimate = estimate_motion(frames, ((20, 40), (60, 40)), frame_start=0, frame_rate=25.0)

    assert estimate.status == "unknown"
    assert estimate.vector_px is None

