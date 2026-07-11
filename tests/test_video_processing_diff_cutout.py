import unittest

import numpy as np

import video_processing


class VideoProcessingDiffCutoutTests(unittest.TestCase):
    def test_saved_and_classification_cutouts_can_keep_different_pixels(self):
        raw_detector_diff = np.arange(36, dtype=np.uint8).reshape(6, 6)
        enhanced_save_diff = np.full((6, 6), 200, dtype=np.uint8)
        rect = (1, 2, 5, 6)

        classification_cutout = video_processing._extract_diff_cutout(
            raw_detector_diff, rect, 4
        )
        saved_cutout = video_processing._extract_diff_cutout(enhanced_save_diff, rect, 4)

        np.testing.assert_array_equal(classification_cutout, raw_detector_diff[2:6, 1:5])
        np.testing.assert_array_equal(saved_cutout, np.full((4, 4), 200, dtype=np.uint8))

    def test_diff_cutout_is_resized_to_requested_size(self):
        diff_image = np.full((5, 5), 120, dtype=np.uint8)

        cutout = video_processing._extract_diff_cutout(diff_image, (1, 1, 4, 4), 8)

        self.assertEqual(cutout.shape, (8, 8))
        self.assertTrue(np.all(cutout == 120))


if __name__ == "__main__":
    unittest.main()
