import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

import noise_twin_pipeline


class NoiseTwinPipelineTests(unittest.TestCase):
    def make_pipeline(self, root):
        return noise_twin_pipeline.RtspNoiseTwinPipeline(
            rtsp_url="rtsp://camera/stream",
            save_root=root,
            model_path="model.pth",
            analyze_callback=lambda _video, _evidence: True,
            cancel_event=threading.Event(),
            require_validated=False,
        )

    def test_resource_plan_reserves_ui_and_limits_noise_twin_cpu(self):
        with mock.patch("noise_twin_pipeline.os.cpu_count", return_value=18):
            plan = noise_twin_pipeline.PipelineResourcePlan.for_host()
        self.assertEqual(plan.noise_twin_workers, 1)
        self.assertEqual(plan.noise_twin_cpu_threads, 8)
        self.assertEqual(plan.analysis_workers, 1)
        self.assertEqual(plan.ui_reserved_cores, 2)

    def test_capture_uses_stream_copy_and_minute_segments(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = self.make_pipeline(directory)
            with mock.patch("noise_twin_pipeline.shutil.which", return_value="/usr/bin/ffmpeg"):
                command = pipeline._ffmpeg_command()
                output_pattern = Path(command[-1])
                output_parent_exists = output_pattern.parent.is_dir()
        self.assertIn("copy", command)
        self.assertNotIn("libx264", command)
        self.assertEqual(command[command.index("-segment_time") + 1], "60")
        self.assertIn("-segment_atclocktime", command)
        self.assertNotIn("-strftime_mkdir", command)
        self.assertTrue(output_parent_exists)
        self.assertEqual(output_pattern.name, "%M_%S.mp4")

    def test_spool_only_releases_completed_segment(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = self.make_pipeline(directory)
            hour = pipeline.spool_root / "20260716" / "01"
            hour.mkdir(parents=True)
            old = hour / "00_00.mp4"
            newest = hour / "01_00.mp4"
            old.write_bytes(b"0" * 20_000)
            newest.write_bytes(b"1" * 20_000)
            stale = time.time() - 10
            os.utime(old, (stale - 10, stale - 10))
            os.utime(newest, (stale, stale))
            pipeline.capture_running.set()
            completed = pipeline._completed_raw_segments()
            self.assertEqual(completed, [old])
            pipeline.capture_running.clear()
            self.assertEqual(pipeline._completed_raw_segments(), [old, newest])

    def test_final_paths_preserve_date_hour_structure(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = self.make_pipeline(directory)
            raw = pipeline.spool_root / "20260716" / "01" / "23_00.mp4"
            video, evidence = pipeline._final_paths(raw)
            self.assertEqual(video, Path(directory) / "20260716" / "01" / "23_00.mp4")
            self.assertEqual(
                evidence,
                Path(directory) / "20260716" / "01" / "23_00_innovation.mp4",
            )


if __name__ == "__main__":
    unittest.main()
