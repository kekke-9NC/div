import unittest

import cv2
import numpy as np
from unittest.mock import patch

from moving_point_detector import detect_moving_point_tracks, _frame_detections


class MovingPointDetectorTests(unittest.TestCase):
    def test_parallel_frame_components_preserve_all_track_fields_and_input_pixels(self):
        rng = np.random.default_rng(253)
        for length in (8,25):
            frames = list(rng.integers(0,90,(length,120,160,3),dtype=np.uint8))
            for index,frame in enumerate(frames):
                cv2.circle(frame,(35+index*3,60-index),2,(255,255,255),-1)
            original = [frame.copy() for frame in frames]
            for scale in (.5,1.):
                options = dict(scale=scale,threshold=8.,border=5)
                self.assertEqual(detect_moving_point_tracks(frames,**options),
                                 detect_moving_point_tracks(frames,_component_workers=4,**options))
            for a,b in zip(frames,original):
                np.testing.assert_array_equal(a,b)

    def test_empty_threshold_image_needs_no_components(self):
        with patch('cv2.connectedComponentsWithStats', side_effect=AssertionError('empty image')):
            self.assertEqual(_frame_detections(np.zeros((120, 160), np.float32), 8, 5, 150), [])
    def test_tracks_a_bright_point_without_a_line(self):
        frames = []
        for index in range(12):
            frame = np.zeros((120, 160, 3), dtype=np.uint8)
            cv2.circle(frame, (35 + index * 3, 60 - index), 2, (255, 255, 255), -1)
            frames.append(frame)

        tracks = detect_moving_point_tracks(
            frames,
            scale=1.0,
            threshold=8.0,
            border=5,
        )

        self.assertTrue(tracks)
        track = tracks[0]
        self.assertGreater(track.displacement_px, 20.0)
        self.assertGreater(track.linearity, 0.9)
        self.assertEqual(track.line[0], (35, 60))

    def test_rejects_a_static_bright_point(self):
        frames = []
        for _ in range(12):
            frame = np.zeros((120, 160, 3), dtype=np.uint8)
            cv2.circle(frame, (80, 60), 2, (255, 255, 255), -1)
            frames.append(frame)

        tracks = detect_moving_point_tracks(
            frames,
            scale=1.0,
            threshold=8.0,
            border=5,
        )

        self.assertEqual(tracks, [])


if __name__ == "__main__":
    unittest.main()
