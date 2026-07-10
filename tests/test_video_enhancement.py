import unittest

import cv2
import numpy as np

import video_enhancement


class VideoEnhancementTests(unittest.TestCase):
    def test_adaptive_strength_recovers_known_pattern_amplitude(self):
        rng = np.random.default_rng(20260711)
        height, width = 72, 96
        correction = rng.normal(0, 3, (height, width)).astype(np.float32)
        correction = correction - cv2.GaussianBlur(correction, (0, 0), 8)
        correction = np.rint(correction).astype(np.int16)
        expected = 0.78
        frames = []
        for _ in range(21):
            noise = rng.normal(0, 0.7, (height, width))
            gray = np.clip(45 + correction * expected + noise, 0, 255).astype(np.uint8)
            frames.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        estimated = video_enhancement.estimate_correction_strength(frames, correction)
        self.assertAlmostEqual(estimated, expected, delta=0.12)

    def test_enhancement_uses_twenty_one_frame_mean(self):
        rng = np.random.default_rng(42)
        correction = np.zeros((48, 64), dtype=np.int16)
        correction[10:38:4, 8:56:5] = 3
        frames = [
            np.clip(50 + rng.normal(0, 10, (48, 64, 3)), 0, 255).astype(np.uint8)
            for _ in range(25)
        ]
        result = video_enhancement.enhance_frames(frames, correction)
        self.assertEqual(len(result.frames), 25)
        self.assertGreater(result.noise_reduction_percent, 50.0)


if __name__ == "__main__":
    unittest.main()
