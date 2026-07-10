import unittest

import cv2
import numpy as np

import video_denoising


class VideoDenoisingTests(unittest.TestCase):
    def _noisy_frames(self):
        rng = np.random.default_rng(1234)
        frames = []
        for _ in range(9):
            noise = rng.normal(0, 11, (72, 96, 3))
            frames.append(np.clip(45 + noise, 0, 255).astype(np.uint8))
        cv2.line(frames[4], (24, 38), (72, 34), (170, 170, 170), 2, cv2.LINE_AA)
        return frames

    def test_temporal_noise_is_reduced(self):
        frames = self._noisy_frames()
        denoised = list(
            video_denoising.iter_denoised_frames(
                frames,
                detected_line=((24, 38), (72, 34)),
            )
        )
        before = video_denoising.estimate_temporal_noise(frames)
        after = video_denoising.estimate_temporal_noise(denoised)
        self.assertLess(after, before * 0.7)

    def test_bright_meteor_is_restored_near_detected_line(self):
        frames = self._noisy_frames()
        protected = video_denoising.denoise_frame(
            frames,
            4,
            detected_line=((24, 38), (72, 34)),
        )
        unprotected = video_denoising.denoise_frame(frames, 4, detected_line=None)
        protected_gray = cv2.cvtColor(protected, cv2.COLOR_BGR2GRAY)
        unprotected_gray = cv2.cvtColor(unprotected, cv2.COLOR_BGR2GRAY)
        corridor = np.zeros(protected_gray.shape, dtype=np.uint8)
        cv2.line(corridor, (24, 38), (72, 34), 255, 4)
        self.assertGreater(
            float(np.percentile(protected_gray[corridor > 0], 95)),
            float(np.percentile(unprotected_gray[corridor > 0], 95)) + 40,
        )

    def test_twenty_one_frame_mean_reduces_noise(self):
        rng = np.random.default_rng(5678)
        frames = [
            np.clip(50 + rng.normal(0, 12, (48, 64, 3)), 0, 255).astype(np.uint8)
            for _ in range(25)
        ]
        denoised = list(
            video_denoising.iter_denoised_frames(
                frames,
                temporal_radius=10,
                temporal_method="mean",
            )
        )
        self.assertLess(
            video_denoising.estimate_temporal_noise(denoised),
            video_denoising.estimate_temporal_noise(frames) * 0.5,
        )


if __name__ == "__main__":
    unittest.main()
