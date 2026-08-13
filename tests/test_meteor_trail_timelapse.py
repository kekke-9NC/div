import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import numpy as np

import meteor_trail_timelapse as trail


class MeteorTrailTimelapseTests(unittest.TestCase):
    def test_discovers_rtsp_files_in_capture_order_and_skips_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rtsp"
            for path in (
                root / "20260813" / "01" / "02.mp4",
                root / "20260813" / "00" / "59.mp4",
                root / "20260813" / "01" / "01.mp4",
                root / "20260813" / "01" / "03_temp_1.mp4",
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"video")

            files = trail.discover_video_files([root])

        self.assertEqual([Path(path).name for path in files], ["59.mp4", "01.mp4", "02.mp4"])

    def test_settings_reject_odd_output_dimensions(self):
        with self.assertRaises(ValueError):
            trail.TrailTimelapseSettings(output_size=(641, 360)).validate()

    def test_default_decay_is_long_enough_for_visible_star_trails(self):
        settings = trail.TrailTimelapseSettings().validate()
        self.assertAlmostEqual(settings.trail_decay, 0.985)
        self.assertGreater(settings.trail_decay, 0.95)

    def test_settings_reject_zero_decay(self):
        with self.assertRaises(ValueError):
            trail.TrailTimelapseSettings(trail_decay=0.0).validate()

    def test_timestamp_is_drawn_in_the_bottom_right_corner(self):
        frame = np.zeros((120, 240, 3), dtype=np.uint8)
        rendered = trail._draw_timestamp(
            frame,
            datetime(2026, 8, 13, 0, 12, 34, 500000),
            "bottom_right",
            1.8,
        )
        self.assertGreater(int(rendered[90:, 120:].max()), 0)
        self.assertEqual(int(rendered[:50, :100].max()), 0)

    def test_source_gap_flushes_window_and_does_not_carry_old_trail(self):
        frames = [np.zeros((4, 6, 3), dtype=np.uint8) for _ in range(3)]
        frames[0][1, 2] = 220
        timestamps = [
            datetime(2026, 8, 13, 0, 0, 0),
            datetime(2026, 8, 13, 0, 0, 0, 500000),
            datetime(2026, 8, 13, 0, 5, 0),
        ]

        class FakeWriter:
            instances = []

            def __init__(self, *_args, **_kwargs):
                self.frames = []
                FakeWriter.instances.append(self)

            def isOpened(self):
                return True

            def write(self, frame):
                self.frames.append(frame.copy())

            def release(self):
                pass

        with (
            mock.patch.object(trail, "discover_video_files", return_value=["one.mp4"]),
            mock.patch.object(trail, "_count_frames", return_value=3),
            mock.patch.object(
                trail,
                "_iter_frames",
                return_value=iter((frame, 2.0, timestamp) for frame, timestamp in zip(frames, timestamps)),
            ),
            mock.patch.object(trail.video_encoding, "FFmpegFrameWriter", FakeWriter),
            mock.patch.object(trail, "_tone_lut", return_value=np.arange(256, dtype=np.uint8)),
        ):
            self.assertTrue(
                trail.create_meteor_trail_timelapse(
                    ["one.mp4"],
                    str(Path(tempfile.gettempdir()) / "trail-gap-test.mp4"),
                    settings=trail.TrailTimelapseSettings(
                        source_seconds_per_output_frame=10.0,
                        output_size=(16, 16),
                        timestamp_enabled=False,
                    ),
                )
            )

        self.assertEqual(len(FakeWriter.instances[0].frames), 2)
        self.assertGreater(int(FakeWriter.instances[0].frames[0].max()), 0)
        self.assertEqual(int(FakeWriter.instances[0].frames[1].max()), 0)

    def test_lut_lifts_dark_night_frames_without_exceeding_uint8(self):
        settings = trail.TrailTimelapseSettings(gamma=1.5, brightness=1.2).validate()
        lut = trail._tone_lut(settings)
        self.assertEqual(lut.dtype, np.uint8)
        self.assertGreater(int(lut[40]), 40)
        self.assertEqual(int(lut[255]), 255)

    def test_composites_source_windows_and_reports_speed(self):
        frames = [
            np.zeros((4, 6, 3), dtype=np.uint8),
            np.zeros((4, 6, 3), dtype=np.uint8),
            np.zeros((4, 6, 3), dtype=np.uint8),
            np.zeros((4, 6, 3), dtype=np.uint8),
        ]
        frames[1][1, 2] = 220
        frames[3][2, 4] = 180

        class FakeWriter:
            def __init__(self, *_args, **_kwargs):
                self.frames = []

            def isOpened(self):
                return True

            def write(self, frame):
                self.frames.append(frame.copy())

            def release(self):
                pass

        callbacks = []
        with (
            mock.patch.object(trail, "discover_video_files", return_value=["one.mp4"]),
            mock.patch.object(trail, "_count_frames", return_value=4),
            mock.patch.object(
                trail,
                "_iter_frames",
                return_value=iter(
                    (frame, 2.0, datetime(2026, 8, 13, 0, 0, index))
                    for index, frame in enumerate(frames)
                ),
            ),
            mock.patch.object(trail.video_encoding, "FFmpegFrameWriter", FakeWriter),
            mock.patch.object(trail, "_tone_lut", return_value=np.arange(256, dtype=np.uint8)),
        ):
            result = trail.create_meteor_trail_timelapse(
                ["one.mp4"],
                str(Path(tempfile.gettempdir()) / "trail-test.mp4"),
                settings=trail.TrailTimelapseSettings(
                    source_seconds_per_output_frame=1.0,
                    output_fps=5.0,
                    output_size=(16, 16),
                    gamma=1.0,
                    contrast=1.0,
                    brightness=1.0,
                ),
                progress_callback=callbacks.append,
            )

        self.assertTrue(result)
        self.assertTrue(callbacks)
        self.assertGreaterEqual(len(callbacks), 1)


if __name__ == "__main__":
    unittest.main()
