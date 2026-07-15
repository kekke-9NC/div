import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

import video_encoding


class VideoEncodingTests(unittest.TestCase):
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
