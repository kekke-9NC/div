"""Exact-pixel regressions for optimizations that must not alter inference."""
import tempfile
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import meteor_direction
import video_processing
from video_io import open_video_capture
from universal_meteor_model import _crop_square, _gray, _robust_residual_cube


class DetectionOptimizationTests(unittest.TestCase):
    def test_live_mask_reuse_is_pixel_exact_bounded_and_invalidates(self):
        rng = np.random.default_rng(629)
        frames = list(rng.integers(0, 256, (12, 30, 40, 3), dtype=np.uint8))
        cache = video_processing._VideoFrameRangeCache(max_bytes=frames[0].nbytes * 8)
        mask = rng.integers(0, 2, (15, 20), dtype=np.uint8) * 255
        cache.retain_live_frames(frames[:6])
        first = cache.mask_frames(frames[2:6], mask)
        repeated = cache.mask_frames(frames[3:6], mask)
        self.assertTrue(all(a is b for a, b in zip(first[1:], repeated)))
        cache.retain_live_frames(frames[3:9])
        self.assertEqual(len(cache.frames), 4)
        self.assertLessEqual(sum(f.nbytes for f in cache.frames) + sum(f.nbytes for f in cache.masked.values()), cache.max_bytes)
        mask[:, :8] = 0
        resized = cv2.resize(mask, (40, 30)).astype(np.uint8)
        actual = cache.mask_frames(frames[3:9], mask)
        for raw, got in zip(frames[3:9], actual):
            np.testing.assert_array_equal(got, cv2.bitwise_and(raw, raw, mask=resized))
        self.assertTrue(set(cache.masked).issubset({id(f) for f in cache.frames}))

    def test_run_comparison_requires_exact_probabilities(self):
        from tools.compare_detection_runs import compare
        with tempfile.TemporaryDirectory() as directory:
            before, after = Path(directory) / 'before', Path(directory) / 'after'
            before.mkdir()
            after.mkdir()
            item = {'center': [10, 20], 'probability': .95, 'label': 'meteor',
                    'timestamp': '2026-10-11 02:25:00', 'start_frame': 0,
                    'end_frame': 30, 'detection_start_frame': 12,
                    'detection_end_frame': 14, 'detection_line': [[10, 15], [10, 25]],
                    'saved_paths': {}}
            (before / 'detections.json').write_text(json.dumps([item]))
            (after / 'detections.json').write_text(json.dumps([item]))
            self.assertTrue(compare(before, after)['exact_detections'])
            item['probability'] = np.nextafter(.95, 1.)
            (after / 'detections.json').write_text(json.dumps([item]))
            self.assertFalse(compare(before, after)['exact_detections'])

    def test_uint8_medians_and_residual_cube_are_bit_exact(self):
        rng = np.random.default_rng(414)
        for length in (1, 2, 7, 8, 25, 26):
            stack = rng.integers(0, 256, (length, 32, 40), dtype=np.uint8)
            values = stack.astype(np.float32)
            original_background = np.median(values, axis=0)
            background = np.median(stack, axis=0).astype(np.float32)
            np.testing.assert_array_equal(original_background, background)
            deviation = np.abs(values - original_background)
            temporal_sigma = 1.4826 * np.median(deviation, axis=0)
            global_sigma = (float(np.median(np.abs(np.diff(values, axis=0)))) / .954
                            if length > 1 else float(np.median(deviation)) * 1.4826)
            global_sigma = max(1., global_sigma)
            expected = np.clip(np.maximum(values - original_background, 0.)
                               / np.maximum(temporal_sigma, global_sigma)[None], 0., 30.).astype(np.float32)
            actual, sigma = _robust_residual_cube(stack)
            self.assertEqual(sigma, global_sigma)
            np.testing.assert_array_equal(expected, actual)

    def test_integer_threshold_comparison_is_exact_for_all_pixel_values(self):
        contrast = np.arange(256, dtype=np.uint8)
        thresholds = np.minimum(25., 3.5 + contrast.astype(np.float32) * .18)
        pixels = np.arange(256, dtype=np.uint8)[:, None]
        np.testing.assert_array_equal(
            pixels >= thresholds, pixels >= np.ceil(thresholds).astype(np.uint8)
        )

    def test_crop_before_grayscale_is_exact_including_edges_and_strides(self):
        rng = np.random.default_rng(812)
        frame = rng.integers(0, 256, (221, 319, 3), dtype=np.uint8)
        for source in (frame, frame[:, ::2], frame[:, :, 0]):
            for rect in (None, (11, 7, 170, 179), (-5, -6, 80, 79),
                         (315, 210, 400, 300), (400, 300, 500, 400)):
                for size in (128, 160, 256):
                    gray = _gray(source)
                    if rect is None:
                        expected = gray
                    else:
                        x1, y1, x2, y2 = rect
                        expected = gray[max(0, y1):min(gray.shape[0], y2),
                                        max(0, x1):min(gray.shape[1], x2)]
                    if expected.size == 0:
                        expected = np.zeros((size, size), np.uint8)
                    elif expected.shape != (size, size):
                        expected = cv2.resize(expected, (size, size), interpolation=cv2.INTER_AREA)
                    np.testing.assert_array_equal(expected, _crop_square(source, rect, size))

    def test_corridor_sampling_equals_full_frame_conversion(self):
        rng = np.random.default_rng(921)
        frame = rng.integers(0, 256, (95, 130, 3), dtype=np.uint8)
        optimized = meteor_direction._sample_line_corridor([frame], ((-5, 4), (125, 96)), 64, 7)
        grayscale = meteor_direction._grayscale(frame)
        expected = meteor_direction._sample_line_corridor([grayscale], ((-5, 4), (125, 96)), 64, 7)
        np.testing.assert_array_equal(expected, optimized)

    def test_component_statistics_preserve_exact_float32_summation(self):
        rng = np.random.default_rng(612)
        residual = np.zeros((240, 320), np.float32)
        for _ in range(120):
            x, y = rng.integers(20, 300), rng.integers(20, 220)
            residual[y:y + 3, x:x + 3] += rng.uniform(20, 90)
        smoothed = cv2.GaussianBlur(residual, (3, 3), 0)
        _, labels, stats, _ = cv2.connectedComponentsWithStats((smoothed >= 6.).astype(np.uint8), 8)
        for index in range(1, len(stats)):
            x, y, w, h, _ = stats[index]
            original = smoothed[labels == index]
            optimized = smoothed[y:y + h, x:x + w][labels[y:y + h, x:x + w] == index]
            np.testing.assert_array_equal(original, optimized)
            self.assertEqual(original.sum(), optimized.sum())

    def test_batched_tree_queries_preserve_distances_indices_and_ties(self):
        from scipy.spatial import cKDTree
        rng = np.random.default_rng(441)
        points = rng.normal(size=(400, 2))
        points[:4] = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        positions = np.vstack((rng.normal(size=(900, 2)), [[0, 0], points[0]]))
        tree = cKDTree(points)
        original = [tree.query(position) for position in positions]
        distances, indices = tree.query(positions)
        np.testing.assert_array_equal(distances, [item[0] for item in original])
        np.testing.assert_array_equal(indices, [item[1] for item in original])

    def test_cached_decoding_matches_fresh_seeks_for_overlapping_windows(self):
        rng = np.random.default_rng(48)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'sample.mp4')
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (96, 64))
            self.assertTrue(writer.isOpened())
            for _ in range(70):
                writer.write(rng.integers(0, 256, (64, 96, 3), dtype=np.uint8))
            writer.release()
            # Summary readers now use the same bounded decoder; thread count
            # must not change any decoded pixel relative to OpenCV defaults.
            original = cv2.VideoCapture(path)
            bounded = open_video_capture(path)
            try:
                while True:
                    ok, expected = original.read()
                    got, actual = bounded.read()
                    self.assertEqual(ok, got)
                    if not ok:
                        break
                    np.testing.assert_array_equal(expected, actual)
            finally:
                original.release()
                bounded.release()
            cache = video_processing._VideoFrameRangeCache()
            for start, end in ((0, 30), (10, 20), (15, 45), (22, 50), (0, 10), (65, 90)):
                expected = video_processing._read_video_frame_range(path, start, end)
                actual = cache.read(path, start, end)
                self.assertEqual(len(expected), len(actual))
                for a, b in zip(expected, actual):
                    np.testing.assert_array_equal(a, b)

    def test_mask_cache_is_exact_and_invalidates_after_mask_change(self):
        frames = [np.full((30, 40, 3), 90, np.uint8) for _ in range(3)]
        cache = video_processing._VideoFrameRangeCache()
        with patch.object(video_processing, '_read_video_frame_range', return_value=frames):
            cache.read('sample', 0, 2)
        mask = np.full((15, 20), 255, np.uint8)
        mask[:, :8] = 0
        for change in (False, True):
            if change:
                mask[:, 10:] = 0
            resized = cv2.resize(mask, (40, 30)).astype(np.uint8)
            actual = cache.mask_frames(frames, mask)
            repeated = cache.mask_frames(frames[1:], mask)
            for index, frame in enumerate(frames):
                np.testing.assert_array_equal(actual[index], cv2.bitwise_and(frame, frame, mask=resized))
                np.testing.assert_array_equal(frame, np.full_like(frame, 90))
            self.assertIs(actual[1], repeated[0])

    def test_frame_cache_byte_limit_and_decoder_key(self):
        frames = [np.zeros((10, 10, 3), np.uint8)]
        for limit, expected_calls in ((1, 2), (1024, 1)):
            cache = video_processing._VideoFrameRangeCache(limit)
            with patch.object(video_processing, '_read_video_frame_range', return_value=frames) as read:
                cache.read('source', 0, 0)
                cache.read('source', 0, 0)
                self.assertEqual(read.call_count, expected_calls)
                cache.read('source', 0, 0, decoder_threads=1)
                self.assertEqual(read.call_count, expected_calls + 1)

    def test_replacing_entries_for_enhancement_does_not_modify_raw_cache(self):
        original = np.full((10, 10, 3), 73, np.uint8)
        cache = video_processing._VideoFrameRangeCache()
        with patch.object(video_processing, '_read_video_frame_range', return_value=[original]):
            frames = cache.read('source', 0, 0)
            frames[0] = np.zeros_like(original)
            self.assertIs(cache.read('source', 0, 0)[0], original)

    def test_summary_jobs_finish_and_failures_are_not_reported_as_saved(self):
        paths = [{}, {}, {}]
        def encode(**kwargs):
            if kwargs['output_video_path'] == 'fail.mp4':
                raise RuntimeError('encoder failure')
            return kwargs['output_video_path'] != 'empty.mp4'
        with patch.object(video_processing.video_creation, 'create_summary_video', side_effect=encode):
            writer = video_processing._SummaryVideoWriter()
            for saved, name in zip(paths, ('ok.mp4', 'fail.mp4', 'empty.mp4')):
                writer.submit(saved, output_video_path=name)
                self.assertLessEqual(len(writer.pending), 2)
            writer.finish()
            self.assertEqual(paths, [{'summary': 'ok.mp4'}, {}, {}])
            self.assertFalse(writer.pending)


if __name__ == '__main__':
    unittest.main()
