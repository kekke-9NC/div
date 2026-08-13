import unittest
from unittest import mock

import numpy as np

import config
import image_processing
import video_processing


class MovingPointPipelineTests(unittest.TestCase):
    def test_diff_generator_can_return_the_source_window(self):
        class Capture:
            def __init__(self):
                self.frames = [
                    np.full((8, 10, 3), value, dtype=np.uint8)
                    for value in (10, 20, 30)
                ]

            def isOpened(self):
                return True

            def get(self, prop):
                if prop == image_processing.cv2.CAP_PROP_FPS:
                    return 2.0
                if prop == image_processing.cv2.CAP_PROP_FRAME_COUNT:
                    return float(len(self.frames))
                return 0.0

            def read(self):
                if not self.frames:
                    return False, None
                return True, self.frames.pop(0)

        rows = list(
            image_processing.create_diff_images(
                Capture(), interval=1.0, duration=1.0, include_frame_window=True
            )
        )

        self.assertTrue(rows)
        self.assertEqual(len(rows[0]), 5)
        self.assertEqual(len(rows[0][4]), 2)

    def test_moving_candidate_detector_is_called_for_rtsp_recordings(self):
        with mock.patch.object(
            video_processing.moving_point_detector,
            "detect_moving_point_tracks",
            return_value=[],
        ) as detector:
            tracks = video_processing.moving_point_detector.detect_moving_point_tracks([])
            detector.assert_called_once_with([])
            self.assertEqual(tracks, [])
            self.assertTrue(config.MOVING_POINT_DETECT_ENABLED)

    def test_moving_point_line_wins_duplicate_hough_line(self):
        moving_line = ((700, 330), (680, 350))
        hough_line = ((705, 335), (684, 354))
        unique = video_processing._deduplicate_candidate_lines(
            [moving_line, hough_line]
        )
        self.assertEqual(unique, [moving_line])

    def test_universal_classifier_gets_focused_activity_window(self):
        frames = [np.full((2, 2), index, dtype=np.uint8) for index in range(60)]
        with mock.patch.object(
            video_processing.model,
            "get_active_model_info",
            return_value={"architecture": video_processing.model.UNIVERSAL_ARCHITECTURE_NAME},
        ):
            selected = video_processing._select_model_event_frames(
                frames,
                clip_start_frame=677,
                detection_start_frame=701,
                detection_end_frame=711,
                frame_rate=25.0,
            )
        self.assertEqual(len(selected), 35)
        self.assertEqual(int(selected[0][0, 0]), 12)
        self.assertEqual(int(selected[-1][0, 0]), 46)


if __name__ == "__main__":
    unittest.main()
