import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

import noise_twin_pipeline


class NoiseTwinPipelineTests(unittest.TestCase):
    def make_pipeline(self, root, **overrides):
        options = {
            "rtsp_url": "rtsp://camera/stream",
            "save_root": root,
            "model_path": "model.pth",
            "analyze_callback": lambda _video, _evidence: True,
            "cancel_event": threading.Event(),
            "require_validated": False,
            "segment_seconds": 5,
        }
        options.update(overrides)
        return noise_twin_pipeline.RtspNoiseTwinPipeline(
            **options,
        )

    def test_resource_plan_reserves_ui_and_limits_noise_twin_cpu(self):
        with mock.patch("noise_twin_pipeline.os.cpu_count", return_value=18):
            plan = noise_twin_pipeline.PipelineResourcePlan.for_host()
        self.assertEqual(plan.noise_twin_workers, 1)
        self.assertEqual(plan.noise_twin_cpu_threads, 8)
        self.assertEqual(plan.analysis_workers, 1)
        self.assertEqual(plan.ui_reserved_cores, 2)

    def test_capture_forces_keyframes_for_five_second_segments(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = self.make_pipeline(directory)
            with mock.patch("noise_twin_pipeline.shutil.which", return_value="/usr/bin/ffmpeg"):
                command = pipeline._ffmpeg_command()
                output_pattern = Path(command[-1])
                output_parent_exists = pipeline._capture_directory().is_dir()
        self.assertIn("libx264", command)
        self.assertIn("expr:gte(t,n_forced*5)", command)
        self.assertEqual(command[command.index("-segment_time") + 1], "5")
        self.assertNotIn("-segment_atclocktime", command)
        self.assertNotIn("-strftime_mkdir", command)
        self.assertTrue(output_parent_exists)
        self.assertRegex(output_pattern.name, r"^%M_%S_[0-9a-f]{12}\.mp4$")

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

    def test_spool_ignores_segment_removed_during_scan(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = self.make_pipeline(directory)
            raw = pipeline.spool_root / "20260719" / "02" / "36_00.mp4"
            raw.parent.mkdir(parents=True)
            raw.write_bytes(b"0" * 20_000)
            original_stat = Path.stat

            def remove_before_stat(path, *args, **kwargs):
                if path == raw:
                    raw.unlink()
                return original_stat(path, *args, **kwargs)

            with mock.patch.object(Path, "stat", remove_before_stat):
                completed = pipeline._completed_raw_segments()

            self.assertEqual(completed, [])

    def test_spool_does_not_stat_enqueued_segment(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = self.make_pipeline(directory)
            raw = pipeline.spool_root / "20260719" / "02" / "36_00.mp4"
            raw.parent.mkdir(parents=True)
            raw.write_bytes(b"0" * 20_000)
            pipeline.enqueued.add(str(raw))
            original_stat = Path.stat

            def reject_enqueued_stat(path, *args, **kwargs):
                if path == raw:
                    raise AssertionError("enqueued segment was stat-ed")
                return original_stat(path, *args, **kwargs)

            with mock.patch.object(Path, "stat", reject_enqueued_stat):
                completed = pipeline._completed_raw_segments()

            self.assertEqual(completed, [])

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

    def test_temporal_mean_can_preserve_raw_saved_segment(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = self.make_pipeline(
                directory,
                model_path="",
                temporal_mean_frames=3,
                save_temporal_mean_video=False,
            )
            raw = pipeline.spool_root / "20260716" / "01" / "23_00.mp4"
            raw.parent.mkdir(parents=True)
            original = b"raw-rtsp-segment"
            raw.write_bytes(original)
            final_video, _ = pipeline._final_paths(raw)

            with mock.patch(
                "noise_twin_pipeline.temporal_mean.prepare_video"
            ) as prepare_video:
                worker = threading.Thread(target=pipeline._denoise_loop)
                worker.start()
                pipeline.raw_queue.put(str(raw))
                pipeline.raw_queue.join()
                pipeline.cancel_event.set()
                worker.join(timeout=2)

            prepare_video.assert_not_called()
            self.assertEqual(final_video.read_bytes(), original)
            self.assertFalse(
                Path(str(final_video) + ".preprocessed.json").exists()
            )
            pending_marker = Path(str(final_video) + ".analysis_pending.json")
            self.assertTrue(pending_marker.exists())
            self.assertEqual(
                pipeline.analysis_queue.get_nowait(),
                (str(final_video), ""),
            )

    def test_start_restores_pending_raw_video_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = self.make_pipeline(
                directory,
                model_path="",
                temporal_mean_frames=3,
                save_temporal_mean_video=False,
            )
            video = Path(directory) / "20260716" / "01" / "23_00.mp4"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"raw")
            pipeline._analysis_pending_marker(video).write_text(
                '{"method":"temporal_mean_analysis","frames":3}',
                encoding="utf-8",
            )

            with mock.patch.object(threading.Thread, "start"):
                pipeline.start()

            self.assertEqual(
                pipeline.analysis_queue.get_nowait(),
                (str(video), ""),
            )

    def test_evidence_is_retained_until_successor_finishes(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = self.make_pipeline(directory)
            hour = Path(directory) / "20261011" / "20"
            hour.mkdir(parents=True)
            videos = [hour / f"00_{second:02d}_aabbccddeeff.mp4" for second in (0, 5)]
            evidence = [video.with_name(video.stem + "_innovation.mp4") for video in videos]
            for video, sidecar in zip(videos, evidence):
                video.touch()
                sidecar.touch()
                noise_twin_pipeline.noise_twin.processing_marker_path(video).write_text(
                    '{"processed": true}', encoding="utf-8")
            observed = []
            def analyze(video, _evidence):
                observed.append(evidence[0].exists())
                return True
            pipeline.analyze_callback = analyze
            worker = threading.Thread(target=pipeline._analysis_loop)
            worker.start()
            pipeline.analysis_queue.put((str(videos[0]), str(evidence[0])))
            pipeline.analysis_queue.join()
            self.assertTrue(evidence[0].exists())
            pipeline.analysis_queue.put((str(videos[1]), str(evidence[1])))
            pipeline.analysis_queue.join()
            pipeline.cancel_event.set()
            worker.join(timeout=2)
            self.assertEqual(observed, [True, True])
            self.assertFalse(evidence[0].exists())
            self.assertTrue(evidence[1].exists())
            marker = noise_twin_pipeline.noise_twin.load_processing_marker(str(videos[1]))
            self.assertTrue(marker["analyzed"])
            with mock.patch.object(threading.Thread, "start"):
                restarted = self.make_pipeline(directory)
                restarted.start()
            self.assertTrue(restarted.analysis_queue.empty())


if __name__ == "__main__":
    unittest.main()
