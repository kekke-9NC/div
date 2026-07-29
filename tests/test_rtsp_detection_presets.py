import unittest

import cv2
import numpy as np

import config
import image_processing
import temporal_mean


class RtspDetectionPresetTests(unittest.TestCase):
    def test_cloudy_preset_detects_single_frame_meteor_after_three_frame_mean(self):
        frames = []
        for index in range(3):
            frame = np.full((256, 256, 3), 20, dtype=np.uint8)
            if index == 1:
                cv2.line(
                    frame,
                    (60, 210),
                    (170, 50),
                    (255, 255, 255),
                    3,
                    cv2.LINE_AA,
                )
            frames.append(frame)

        averaged = temporal_mean.mean_frame(frames)
        difference = cv2.absdiff(averaged, np.full_like(averaged, 20))
        preset = config.RTSP_PRESET_CLOUDY

        lines = image_processing.detect_lines(
            difference,
            min_length=preset["min_line_length"],
            canny_thresh1=preset["canny_thresh1"],
            canny_thresh2=preset["canny_thresh2"],
            hough_threshold=preset["hough_threshold"],
        )

        self.assertTrue(lines)


if __name__ == "__main__":
    unittest.main()
