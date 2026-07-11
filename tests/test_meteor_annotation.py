import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import astrometry


class MeteorAnnotationTests(unittest.TestCase):
    def test_marker_draws_only_a_rectangle(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)

        marked = astrometry._draw_meteor_marker(image, ((80, 80), (120, 120)))

        # The detected trajectory remains unobscured; the marker is only the border.
        np.testing.assert_array_equal(marked[100, 100], np.zeros(3, dtype=np.uint8))
        self.assertTrue(np.any(marked[44, 44] > 0))

    def test_annotation_preserves_source_orientation_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "orientation.png"
            source = np.zeros((12, 16, 3), dtype=np.uint8)
            source[0, 0] = (10, 20, 30)
            source[-1, -1] = (200, 180, 160)
            self.assertTrue(cv2.imwrite(str(source_path), source))

            annotated_path = astrometry.annotate_image_with_wcs(
                str(source_path), {}, timestamp=None
            )

            self.assertIsNotNone(annotated_path)
            annotated = cv2.imread(str(annotated_path))
            np.testing.assert_array_equal(annotated, source)


if __name__ == "__main__":
    unittest.main()
