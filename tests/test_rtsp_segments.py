import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta
from unittest import mock

import cv2
import numpy as np

import astrometry
import file_utils
import media_time
import rtsp_segments
from short_flash_detector import ShortFlashDetector
import timelapse_creator
import video_processor
import video_processing
import config


class RtspSegmentTests(unittest.TestCase):
    def make_segments(self, root, session="aabbccddeeff", gap=0, first=None):
        start = first or datetime(2026, 10, 11, 20, 0)
        paths = []
        for index in range(3):
            timestamp = start + timedelta(seconds=index * (5 + gap))
            path = Path(root) / timestamp.strftime("%Y%m%d/%H") / (timestamp.strftime("%M_%S") + f"_{session}.mp4")
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (220, 180))
            self.assertTrue(writer.isOpened())
            for frame_index in range(125):
                frame = np.full((180, 220, 3), 10, np.uint8)
                absolute_index = index * 125 + frame_index
                if absolute_index in (124, 125):
                    y = 75 if absolute_index == 124 else 94
                    cv2.line(frame, (100, y - 6), (100, y + 6), (250, 250, 250), 2)
                writer.write(frame)
            writer.release()
            rtsp_segments.write_capture_time(path, timestamp)
            stale = time.time() - 30
            os.utime(path, (stale, stale))
            paths.append(str(path))
        return paths

    def read(self, path):
        capture = cv2.VideoCapture(path)
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        capture.release()
        return frames

    def test_boundary_flash_survives_and_has_one_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_segments(directory)
            owned = []
            for path in paths[:2]:
                with rtsp_segments.analysis_window(path, paths) as context:
                    frames = self.read(context["analysis_source"])
                    candidates = ShortFlashDetector(border=5).consume(frames, 0, final=True)
                    self.assertTrue(candidates, "two-frame flash was lost across the file boundary")
                    for event in candidates:
                        if rtsp_segments.owns_event({"rtsp_analysis_context": context}, event.start_frame):
                            owned.append(context["base_datetime"] + timedelta(seconds=event.start_frame / 25))
                    temporary = context["analysis_source"]
                self.assertFalse(os.path.exists(temporary))
            self.assertEqual(owned, [datetime(2026, 10, 11, 20, 0, 4, 960000)])
            self.assertEqual([len(self.read(path)) for path in paths], [125, 125, 125])

    def test_midnight_boundary_and_captured_time_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_segments(directory, first=datetime(2026, 10, 11, 23, 59, 55))
            with rtsp_segments.analysis_window(paths[1]) as context:
                self.assertEqual(context["core_start_frame"], 125)
                self.assertEqual(context["base_datetime"], datetime(2026, 10, 11, 23, 59, 55))
            self.assertEqual(astrometry.extract_datetime_from_video_path(paths[1]), datetime(2026, 10, 12))
            self.assertEqual(timelapse_creator._source_created_datetime(paths[1]), datetime(2026, 10, 12))

    def test_gap_is_not_joined(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_segments(directory, gap=4)
            with rtsp_segments.analysis_window(paths[1], paths) as context:
                self.assertEqual(context["core_start_frame"], 0)
                self.assertEqual(len(self.read(context["analysis_source"])), 125)

    def test_detector_saves_boundary_event_once_with_full_clip(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_segments(directory)
            results = []
            with (
                mock.patch.object(video_processing.model, "predict_noise_gate", return_value=None),
                mock.patch.object(video_processing.model, "predict_meteor_event", return_value=0.99),
                mock.patch.object(video_processing.model, "get_meteor_decision_threshold", return_value=0.5),
                mock.patch.object(video_processing.image_processing, "detect_lines", return_value=[]),
                mock.patch.object(config, "CUTOUT_SIZE", 64),
                mock.patch.object(config, "FINER_CUTOUT_SIZE", 100),
            ):
                for path in paths[:2]:
                    with rtsp_segments.analysis_window(path, paths) as context:
                        results.extend(video_processing.create_line_video_clips(
                            path, meteor_save_path=directory, not_meteor_save_path=directory,
                            save_options={"video": False, "cutout": False, "full": False,
                                          "composite": False, "info": False, "summary": False,
                                          "full_video": True, "ml_training_export_enabled": False,
                                          "rtsp_analysis_context": context},
                        ))
            meteors = [item for item in results if item["label"] == "meteor"]
            self.assertEqual(len(meteors), 1)
            self.assertEqual((meteors[0]["detection_start_frame"], meteors[0]["detection_end_frame"]), (124, 125))
            full = self.read(meteors[0]["saved_paths"]["full_video"])
            self.assertGreater(len(full), 40, "clip must retain padding across the recording boundary")

    def test_reconnection_is_not_joined(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_segments(directory)
            renamed = paths[0].replace("aabbccddeeff", "001122334455")
            Path(paths[0]).rename(renamed)
            Path(paths[0] + ".capture.json").rename(renamed + ".capture.json")
            paths[0] = renamed
            with rtsp_segments.analysis_window(paths[1], paths) as context:
                self.assertEqual(context["core_start_frame"], 0)

    def test_latest_waits_then_flushes_at_recording_pause(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_segments(directory)
            os.utime(paths[-1], None)
            with self.assertRaises(rtsp_segments.SegmentNotReady):
                with rtsp_segments.analysis_window(paths[-1], paths):
                    pass
            with rtsp_segments.analysis_window(paths[-1], paths, next_wait_seconds=0) as context:
                self.assertEqual(len(self.read(context["analysis_source"])), 250)

    def test_scanner_processes_finished_segments_in_the_current_minute(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_segments(directory)
            os.utime(paths[-1], None)
            processed = set()
            contexts = []
            def analyze(*args, **kwargs):
                contexts.append(args[8]["rtsp_analysis_context"])
                return True
            with mock.patch.object(file_utils, "process_video_file_periodic", side_effect=analyze):
                file_utils.process_new_rtsp_files(directory, processed, max_workers=2)
            self.assertEqual(processed, set(paths[:2]))
            self.assertEqual(len(contexts), 2)

    def test_archive_collection_ignores_spool_and_innovation(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_segments(directory)
            hidden = Path(directory) / ".noise_twin_spool" / "20261011" / "20" / "00_00_aabbccddeeff.mp4"
            hidden.parent.mkdir(parents=True)
            hidden.write_bytes(b"unfinished")
            evidence = Path(paths[0]).with_name(Path(paths[0]).stem + "_innovation.mp4")
            evidence.write_bytes(b"evidence")
            _, collected = timelapse_creator.get_files_from_path(directory)
            self.assertEqual(collected, paths)
            self.assertFalse(video_processor.is_video_file(str(evidence)))
            schedule = video_processor._build_concat_schedule(collected, 0)
            self.assertAlmostEqual(schedule[-1]["end"], 15)
            self.assertEqual(schedule[1]["source_start_time"], datetime(2026, 10, 11, 20, 0, 5))

    def test_capture_sidecar_survives_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "copied.mp4"
            path.touch()
            timestamp = datetime(2026, 10, 11, 20, 0, 5, 123000)
            rtsp_segments.write_capture_time(path, timestamp)
            self.assertEqual(media_time.get_media_start_time(str(path)), (timestamp, "RTSP録画開始時刻"))

    def test_opencv_capture_uses_camera_fps_for_five_seconds(self):
        cancel = threading.Event()
        class Camera:
            remaining = 75
            opened = True
            def isOpened(self):
                return self.opened
            def get(self, key):
                return {cv2.CAP_PROP_FPS: 15, cv2.CAP_PROP_FRAME_WIDTH: 220,
                        cv2.CAP_PROP_FRAME_HEIGHT: 180}.get(key, 0)
            def read(self):
                if not self.remaining:
                    cancel.set()
                    return False, None
                self.remaining -= 1
                rng = np.random.default_rng(self.remaining)
                return True, rng.integers(0, 256, (180, 220, 3), np.uint8)
            def release(self):
                self.opened = False
        with tempfile.TemporaryDirectory() as directory:
            with (mock.patch.object(config, "RTSP_USE_NVIDIA_HWACCEL", False),
                  mock.patch.object(file_utils.utils, "create_rtsp_capture", return_value=Camera())):
                file_utils.save_rtsp_video_segments("rtsp://synthetic", directory,
                                                   segment_duration=5, cancel_flag=cancel, requested_fps=25)
            saved = list(Path(directory).rglob("*.mp4"))
            self.assertEqual(len(saved), 1)
            fps, count, _ = rtsp_segments._properties(saved[0])
            self.assertEqual((fps, count), (15, 75))
            self.assertTrue(Path(str(saved[0]) + ".capture.json").is_file())

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg required")
    def test_batched_concat_keeps_every_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_segments(directory)
            batches = video_processor._normalize_concat_batches(paths, directory, fps=25, batch_size=2)
            self.assertEqual(len(batches), 2)
            self.assertEqual(sum(len(self.read(path)) for path in batches), 375)


if __name__ == "__main__":
    unittest.main()
