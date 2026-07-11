import unittest

import numpy as np

import automatic_video_mask


class AutomaticVideoMaskTests(unittest.TestCase):
    def test_bottom_connected_skyline_is_excluded(self):
        height, width = 180, 320
        image = np.full((height, width), 55, dtype=np.uint8)
        image[145:, :] = 12
        image[120:, :70] = 220
        image[132:, 240:] = 185
        # Add building windows so the lower structures have stable texture and edges.
        image[130:170:8, 10:65:12] = 245
        image[142:170:7, 250:315:10] = 240

        mask, stats = automatic_video_mask.build_mask_from_median(image)
        self.assertEqual(mask.shape, image.shape)
        self.assertEqual(mask[20, 160], 255)
        self.assertEqual(mask[170, 160], 0)
        self.assertEqual(mask[150, 30], 0)
        self.assertGreater(stats["sky_fraction"], 0.55)
        self.assertLess(stats["sky_fraction"], 0.95)

    def test_manual_and_automatic_masks_are_intersected(self):
        automatic = np.full((20, 20), 255, dtype=np.uint8)
        automatic[15:] = 0
        manual = np.full((20, 20), 255, dtype=np.uint8)
        manual[:, :5] = 0
        combined = automatic_video_mask.combine_masks(manual, automatic)
        self.assertEqual(combined[5, 2], 0)
        self.assertEqual(combined[18, 10], 0)
        self.assertEqual(combined[5, 10], 255)
