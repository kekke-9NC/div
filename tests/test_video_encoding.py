import tempfile
import unittest
import threading
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

import video_encoding


class VideoEncodingTests(unittest.TestCase):
    def test_repeated_image_encoder_preserves_order_pixels_count_and_fractional_fps(self):
        rng = np.random.default_rng(671)
        unique = list(rng.integers(0, 256, (4, 48, 64, 3), dtype=np.uint8))
        frames = [unique[0]]*3 + [unique[1]] + [unique[2]]*7 + [unique[3]]*2
        for fps in (6., 7.5, 29.97, 60.):
            with self.subTest(fps=fps), tempfile.TemporaryDirectory() as directory:
                reference, actual = Path(directory)/"reference.mp4", Path(directory)/"actual.mp4"
                settings = {"codec": "h264", "quality": "lossless"}
                writer = video_encoding.FFmpegFrameWriter(reference, fps, (64, 48), settings)
                for frame in frames:
                    writer.write(frame)
                writer.release()
                count = video_encoding.write_repeated_images(actual, iter(frames), fps,
                                                            threading.Event(), settings)
                self.assertEqual(count, len(frames))
                left, right = cv2.VideoCapture(str(reference)), cv2.VideoCapture(str(actual))
                self.assertEqual(int(right.get(cv2.CAP_PROP_FRAME_COUNT)), len(frames))
                self.assertAlmostEqual(right.get(cv2.CAP_PROP_FPS), fps, places=3)
                for _ in frames:
                    expected_ok, expected = left.read()
                    actual_ok, got = right.read()
                    self.assertTrue(expected_ok and actual_ok)
                    np.testing.assert_array_equal(got, expected)
                self.assertFalse(right.read()[0])
                left.release()
                right.release()

    def test_repeated_image_encoder_cancel_retains_existing_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"actual.mp4"
            cancel = threading.Event()
            cancel.set()
            with self.assertRaises(InterruptedError):
                video_encoding.write_repeated_images(output,
                    [np.zeros((48, 64, 3), np.uint8)], 60., cancel)
            self.assertFalse(output.exists())
            self.assertFalse(list(Path(directory).glob(".timelapse_frames_*")))

    def test_source_quality_matches_measured_hevc_bitrate(self):
        settings = video_encoding.EncodingSettings.from_value(
            {"codec": "hevc", "quality": "source"}
        )
        with mock.patch("video_encoding.source_bitrate_mbps", return_value=44.6):
            resolved = video_encoding.resolve_for_source(settings, "source.mp4")
        self.assertEqual(resolved.codec, "hevc")
        self.assertEqual(resolved.bitrate_mbps, 45)
        self.assertEqual(resolved.quality, "custom")

    def test_h264_source_quality_has_small_generation_allowance(self):
        with mock.patch("video_encoding.source_bitrate_mbps", return_value=44.6):
            resolved = video_encoding.resolve_for_source(
                {"codec": "h264", "quality": "source"}, "source.mp4"
            )
        self.assertEqual(resolved.bitrate_mbps, 45)

    def test_hevc_writer_produces_readable_video_with_exact_frame_count(self):
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory) / "encoded.mp4")
            writer = video_encoding.FFmpegFrameWriter(
                output, 25.0, (64, 48),
                {"codec": "hevc", "quality": "custom", "bitrate_mbps": 10},
            )
            self.assertTrue(writer.isOpened())
            for index in range(7):
                writer.write(np.full((48, 64, 3), index * 20, dtype=np.uint8))
            writer.release()
            capture = cv2.VideoCapture(output)
            self.assertTrue(capture.isOpened())
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 7)
            capture.release()

    def test_estimate_is_unknown_for_source_and_lossless_modes(self):
        self.assertIsNone(video_encoding.estimated_megabytes_per_minute({"quality": "source"}))
        self.assertIsNone(video_encoding.estimated_megabytes_per_minute({"quality": "lossless"}))
        self.assertEqual(
            video_encoding.estimated_megabytes_per_minute(
                {"quality": "custom", "bitrate_mbps": 40}
            ),
            300.0,
        )


if __name__ == "__main__":
    unittest.main()
