import unittest

import cv2
import numpy as np

from moving_point_detector import detect_moving_point_tracks


class MovingPointDetectorTests(unittest.TestCase):
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
