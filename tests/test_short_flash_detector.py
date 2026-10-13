import unittest

import cv2
import numpy as np
from unittest.mock import patch

from short_flash_detector import ShortFlashDetector, candidate_from_points
import video_processing


class ShortFlashTests(unittest.TestCase):
    def test_empty_foreground_still_advances_background_and_flushes_tracks(self):
        detector = ShortFlashDetector(25, 5)
        with patch('cv2.connectedComponentsWithStats', side_effect=AssertionError('empty image')):
            self.assertEqual(detector.consume([np.full((120, 160, 3), 17, np.uint8)]*8, 0, final=True), [])
        self.assertEqual(detector.last_frame, 7)
        self.assertFalse(detector.active)
        np.testing.assert_array_equal(detector.background, np.full((120, 160), 17, np.float32))
    def test_two_frame_motion_is_kept(self):
        points = [[10, 90, 80, 200, 5, 16, 40],
                  [11, 92, 96, 250, 5, 16, 50]]
        candidate = candidate_from_points(points)
        self.assertIsNotNone(candidate)
        self.assertEqual((candidate.start_frame, candidate.end_frame), (10, 11))

    def test_fading_trail_does_not_extend_luminous_time(self):
        points = [[10, 90, 80, 200, 5, 16, 40],
                  [11, 92, 96, 250, 5, 16, 50]]
        points += [[i, 92, 96, 30, 4, 4, 6] for i in range(12, 30)]
        self.assertEqual(candidate_from_points(points).end_frame, 11)

    def test_stationary_flicker_and_slow_points_are_rejected(self):
        flicker = [[i, 50 + (i % 2), 70, 150, 10, 10, 30] for i in range(10)]
        slow = [[i, 50 + i * .2, 70, 150, 10, 10, 30] for i in range(30)]
        self.assertIsNone(candidate_from_points(flicker))
        self.assertIsNone(candidate_from_points(slow))

    def test_sparse_coincidences_and_long_lived_tracks_are_rejected(self):
        sparse = [[i, 50 + i * 2, 70, 150, 10, 10, 30] for i in (0, 8, 20)]
        satellite = [[i, 50 + i, 70, 150, 10, 10, 30] for i in range(90)]
        self.assertIsNone(candidate_from_points(sparse))
        self.assertIsNone(candidate_from_points(satellite))

    def frames(self):
        result = [np.full((180, 220, 3), 20, np.uint8) for _ in range(50)]
        for index, y in [(24, 75), (25, 94)]:
            cv2.line(result[index], (100, y - 8), (100, y + 8), (180, 180, 180), 2)
        return result

    def test_window_boundary_retains_native_pixels_and_motion(self):
        frames = self.frames()
        detector = ShortFlashDetector(border=5)
        first = detector.consume(frames[:25], 0)
        second = detector.consume(frames[25:], 25, final=True)
        self.assertEqual(first, [])
        self.assertTrue(second)
        self.assertEqual(second[0].start_frame, 24)
        self.assertEqual(second[0].end_frame, 25)

    def test_mask_and_roi_apply_to_new_proposals(self):
        frames = self.frames()
        mask = np.full((180, 220), 255, np.uint8)
        mask[:, 90:110] = 0
        self.assertEqual(ShortFlashDetector(border=5).consume(frames, 0, mask=mask, final=True), [])
        self.assertEqual(ShortFlashDetector(border=5).consume(frames, 0, roi=(120, 0, 220, 180), final=True), [])

    def test_discontinuous_samples_do_not_establish_a_track(self):
        frames = self.frames()
        detector = ShortFlashDetector(border=5)
        detector.consume(frames[:25], 0)
        self.assertEqual(detector.consume(frames[25:], 27, final=True), [])

    def test_finer_stage_preserves_verified_two_frame_bounds(self):
        frames = self.frames()
        indices, _ = video_processing._process_finer_detection_worker(
            frames, 100, 85, 180, 220, 0, ((100, 75), (100, 94)),
            5, 100, 5, 15, temporal_bounds=(24, 25),
        )
        self.assertEqual(indices, [24, 25])

    def test_later_event_at_same_position_is_not_a_duplicate(self):
        saved = [dict(center=(50, 60), label='meteor',
                      detection_start_frame=10, detection_end_frame=13)]
        line = ((50, 55), (50, 65))
        self.assertFalse(video_processing._candidate_already_saved(line, (100, 103), saved, 25))
        self.assertTrue(video_processing._candidate_already_saved(line, (11, 14), saved, 25))

    def test_rejected_coarse_crop_does_not_block_temporal_evidence(self):
        saved = [dict(center=(50, 60), label='not_meteor',
                      detection_start_frame=10, detection_end_frame=13)]
        self.assertFalse(video_processing._candidate_already_saved(
            ((50, 55), (50, 65)), (11, 14), saved, 25, temporal=True))

    def test_long_meteor_in_adjacent_crops_is_one_event(self):
        saved = [dict(center=(325, 561), label='meteor',
                      detection_start_frame=631, detection_end_frame=656,
                      detection_line=((257, 457), (394, 665)))]
        self.assertTrue(video_processing._candidate_already_saved(
            ((400, 675), (426, 710)), (626, 650), saved, 25))

    def test_separate_temporal_flashes_are_not_merged_by_padding(self):
        saved = [dict(center=(50, 60), label='meteor',
                      detection_start_frame=10, detection_end_frame=13)]
        self.assertFalse(video_processing._candidate_already_saved(
            ((50, 55), (50, 65)), (16, 18), saved, 25, temporal=True))


if __name__ == '__main__':
    unittest.main()
