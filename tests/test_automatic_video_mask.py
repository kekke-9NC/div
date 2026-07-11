import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

import automatic_video_mask


class AutomaticVideoMaskTests(unittest.TestCase):
    @staticmethod
    def _moving_cloud_frames(height=180, width=320, count=21):
        yy, xx = np.mgrid[:height, :width]
        frames = []
        for index in range(count):
            clouds = (
                42 * np.sin((xx + index * 27) / 30.0)
                + 25 * np.cos((yy - index * 19) / 22.0)
                + 15 * np.sin((xx + yy + index * 31) / 50.0)
            )
            frames.append(np.clip(65 + clouds, 0, 255).astype(np.uint8))
        return np.stack(frames)

    def test_moving_clouds_are_kept_as_sky(self):
        frames = self._moving_cloud_frames()
        mask, stats = automatic_video_mask.build_mask_from_samples(frames)
        self.assertGreater(stats["sky_fraction"], 0.99)
        self.assertEqual(mask[40, 40], 255)
        self.assertEqual(mask[150, 260], 255)

    def test_static_obstacles_anywhere_are_excluded(self):
        frames = self._moving_cloud_frames()
        for frame in frames:
            cv2.rectangle(frame, (235, 15), (292, 58), 210, thickness=-1)
            cv2.circle(frame, (25, 92), 16, 5, thickness=-1)
            cv2.line(frame, (60, 100), (200, 100), 230, thickness=3)

        mask, stats = automatic_video_mask.build_mask_from_samples(frames)
        self.assertEqual(mask[35, 260], 0)  # top obstruction, not bottom-connected
        self.assertEqual(mask[92, 25], 0)  # side obstruction
        self.assertEqual(mask[100, 100], 0)  # thin wire
        self.assertEqual(mask[150, 160], 255)  # moving cloud remains usable
        self.assertGreater(stats["sky_fraction"], 0.80)
        self.assertLess(stats["sky_fraction"], 0.95)

    def test_single_image_fallback_is_conservative(self):
        height, width = 180, 320
        image = np.full((height, width), 55, dtype=np.uint8)
        cv2.rectangle(image, (20, 120), (70, 175), 220, thickness=-1)

        mask, stats = automatic_video_mask.build_mask_from_median(image)
        self.assertEqual(mask.shape, image.shape)
        self.assertEqual(mask[20, 160], 255)
        self.assertEqual(mask[150, 30], 0)
        self.assertEqual(mask[170, 160], 255)
        self.assertGreater(stats["sky_fraction"], 0.90)

    def test_one_mask_is_generated_and_reused_for_an_hour(self):
        frames = self._moving_cloud_frames(height=90, width=160)
        for frame in frames:
            cv2.rectangle(frame, (5, 5), (25, 25), 220, thickness=-1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hour = root / "20221022" / "21"
            hour.mkdir(parents=True)
            first = hour / "00.mp4"
            second = hour / "59.mp4"
            first.touch()
            second.touch()
            cache = root / "cache"
            sampled = (frames, (320, 180), [str(first), str(second)])
            with mock.patch.object(
                automatic_video_mask, "sample_hour_frames", return_value=sampled
            ) as sampler:
                first_mask, first_preview, _ = automatic_video_mask.create_auto_mask(
                    str(first), str(cache)
                )
                second_mask, second_preview, _ = automatic_video_mask.create_auto_mask(
                    str(second), str(cache)
                )

            self.assertEqual(sampler.call_count, 1)
            np.testing.assert_array_equal(first_mask, second_mask)
            self.assertEqual(first_preview, second_preview)
            metadata_files = list(cache.glob("*.json"))
            self.assertEqual(len(metadata_files), 1)
            metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
            self.assertEqual(metadata["hour_scope"], "20221022/21")
            self.assertEqual(
                metadata["algorithm_version"], automatic_video_mask.ALGORITHM_VERSION
            )

    def test_manual_and_automatic_masks_are_intersected(self):
        automatic = np.full((20, 20), 255, dtype=np.uint8)
        automatic[15:] = 0
        manual = np.full((20, 20), 255, dtype=np.uint8)
        manual[:, :5] = 0
        combined = automatic_video_mask.combine_masks(manual, automatic)
        self.assertEqual(combined[5, 2], 0)
        self.assertEqual(combined[18, 10], 0)
        self.assertEqual(combined[5, 10], 255)
