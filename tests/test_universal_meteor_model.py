import unittest

import cv2
import numpy as np
import torch

from universal_meteor_model import (
    FEATURE_COUNT,
    IMAGE_SIZE,
    KYMO_HEIGHT,
    KYMO_WIDTH,
    MeteorFusionUniversal,
    build_universal_inputs,
)


class UniversalMeteorModelTests(unittest.TestCase):
    def test_preprocess_shapes_and_finite_values(self):
        frames = []
        for index in range(20):
            frame = np.full((256, 256), 12, dtype=np.uint8)
            cv2.line(frame, (40 + index * 3, 120), (55 + index * 3, 120), 90, 2)
            frames.append(frame)
        image, kymograph, features = build_universal_inputs(
            frames,
            detected_line=((40, 120), (112, 120)),
            frame_rate=20.0,
        )
        self.assertEqual(image.shape, (5, IMAGE_SIZE, IMAGE_SIZE))
        self.assertEqual(kymograph.shape, (3, KYMO_HEIGHT, KYMO_WIDTH))
        self.assertEqual(features.shape, (FEATURE_COUNT,))
        self.assertTrue(np.isfinite(image).all())
        self.assertTrue(np.isfinite(kymograph).all())
        self.assertTrue(np.isfinite(features).all())
        self.assertGreater(float(image[0].max()), 0.0)

    def test_network_forward(self):
        network = MeteorFusionUniversal()
        logits = network(
            torch.zeros(2, 5, IMAGE_SIZE, IMAGE_SIZE),
            torch.zeros(2, 3, KYMO_HEIGHT, KYMO_WIDTH),
            torch.zeros(2, FEATURE_COUNT),
        )
        self.assertEqual(tuple(logits.shape), (2,))


if __name__ == "__main__":
    unittest.main()
