import json
from datetime import datetime, timedelta
from pathlib import Path
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from unittest import mock

import cv2
import numpy as np

import config
import file_utils
import image_processing
import media_time
import noise_twin_pipeline
import noise_twin
import rtsp_live
import rtsp_segments
from short_flash_detector import ShortFlashDetector
import temporal_mean
import timelapse_creator
import video_processing
import video_processor


class RtspLiveTests(unittest.TestCase):
    def test_pipeline_natural_eof_preserves_final_ack_and_evidence_pixels(self):
        from concurrent.futures import ThreadPoolExecutor
        with tempfile.TemporaryDirectory() as root:
            paths = self.chunks(root, count=10)
            expected_cap, _ = self.capture(paths)
            expected = list(image_processing.create_diff_images(expected_cap, interval=.25,
                duration=1, buffer_duration=4, is_rtsp=True, include_frame_window=True))
            expected_cap.release()
            cap, consumed = self.capture(paths)
            acknowledgements = []
            acknowledge = cap.acknowledge
            def observe(index, final=False):
                acknowledgements.append((index, final))
                acknowledge(index, final=final)
            cap.acknowledge = observe
            generator = image_processing.create_diff_images(cap, interval=.25,
                duration=1, buffer_duration=4, is_rtsp=True, include_frame_window=True)
            with ThreadPoolExecutor(max_workers=3) as executor, mock.patch.object(
                    video_processing, '_coarse_proposals', return_value=([], [])):
                pipeline = video_processing._LiveProposalPipeline(generator, cap, None, None, cap.cancel, executor)
                actual = list(pipeline)
                pipeline.close()
            self.assertEqual(len(actual), len(expected))
            for packet, original in zip(actual, expected):
                self.assertEqual(packet[1], original[1])
                for index in (0, 2, 3):
                    np.testing.assert_array_equal(packet[index], original[index])
                for frame, reference in zip(packet[-1], original[-1]):
                    np.testing.assert_array_equal(frame, reference)
            self.assertTrue(acknowledgements[-1][1])
            self.assertEqual(acknowledgements[-1][0], sum(rtsp_live.timing(p)['frame_count'] for p in paths)-1)
            self.assertEqual(consumed, paths)
            self.assertFalse(cap.defer_analysis_ack)
            cap.release()

    def test_pipeline_cancel_never_acknowledges_prefetched_unprocessed_windows(self):
        from concurrent.futures import ThreadPoolExecutor
        with tempfile.TemporaryDirectory() as root:
            paths = self.chunks(root, count=25)
            cap, consumed = self.capture(paths)
            generator = image_processing.create_diff_images(cap, interval=.25,
                duration=1, buffer_duration=4, is_rtsp=True, include_frame_window=True)
            with ThreadPoolExecutor(max_workers=3) as executor, mock.patch.object(
                    video_processing, '_coarse_proposals', return_value=([], [])):
                pipeline = video_processing._LiveProposalPipeline(generator, cap, None, None, cap.cancel, executor)
                stream = iter(pipeline)
                for _ in range(25):
                    latest = next(stream)
                cap.cancel.set()
                pipeline.close()
            acknowledged_frames = sum(rtsp_live.timing(path)['frame_count'] for path in consumed)
            retain = round(25*(config.FINER_DETECT_WINDOW_SECONDS+config.FINER_DETECT_PADDING_SECONDS))
            self.assertLessEqual(acknowledged_frames, max(0, latest[1][1]-6+1-retain))
            self.assertLess(len(consumed), len(paths))
            self.assertFalse(cap.defer_analysis_ack)
            cap.release()

    def test_overlapping_extrema_preserve_all_pixels_and_irregular_tail(self):
        rng = np.random.default_rng(344)
        frames = list(rng.integers(0, 256, (110, 30, 40, 3), dtype=np.uint8))
        for step in (1, 6, 25):
            cache = image_processing._WindowExtrema(step)
            for start, end in [(i, i+25) for i in range(0, 80, step)] + [(82, 103), (91, 110)]:
                actual = cache.reduce(frames[start:end], start)
                expected = image_processing._temporal_min_max(frames[start:end])
                for a, b in zip(actual, expected):
                    np.testing.assert_array_equal(a, b)
                self.assertLessEqual(sum(len(v[0]) for v in cache.blocks.values()), end-start)

    def test_missing_archive_is_republished_before_consumed_spool_is_removed(self):
        with tempfile.TemporaryDirectory() as root:
            paths = self.chunks(root, count=4)
            archiver = rtsp_live.MinuteArchiver(root, seconds=2, status=lambda _message: None)
            archiver.compact(flush=True)
            archive = Path(root) / json.loads(Path(str(paths[0])+'.archived.json').read_text())['archive']
            expected = rtsp_segments._properties(archive)
            archive.unlink()
            for path in paths:
                rtsp_live.write_json(str(path)+'.analysis_consumed.json', {'consumed': True})
            archiver.compact(flush=True)
            self.assertTrue(archive.exists())
            self.assertEqual(rtsp_segments._properties(archive), expected)
            self.assertTrue(all(not path.exists() for path in paths))

    def chunks(self, root, count=10, session="aabbccddeeff", gap=0):
        directory = Path(root) / ".rtsp_live/20261012/20" / f"00_00_{session}"
        directory.mkdir(parents=True, exist_ok=True)
        started = datetime(2026, 10, 12, 20)
        paths, total = [], 0
        for index in range(count):
            length = round((index + 1) * 12.5) - round(index * 12.5)
            path = directory / f"chunk_{index:09d}.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (220, 180))
            self.assertTrue(writer.isOpened())
            for local_index in range(length):
                absolute = total + local_index
                frame = np.full((180, 220, 3), 10, np.uint8)
                if absolute in (24, 25):
                    y = 75 if absolute == 24 else 94
                    cv2.line(frame, (100, y - 6), (100, y + 6), (250, 250, 250), 2)
                writer.write(frame)
            writer.release()
            rtsp_segments.write_capture_time(path, started + timedelta(seconds=total / 25 + index * gap))
            rtsp_live.timing(path)
            paths.append(path)
            total += length
        return paths

    def capture(self, paths, **options):
        incoming = queue.Queue()
        for path in paths[1:]:
            incoming.put(path)
        consumed = []
        cap = rtsp_live.SegmentCapture(paths[0], incoming, threading.Event(), consumed.append,
                                      idle_seconds=.01, **options)
        return cap, consumed

    def test_one_continuous_short_flash_across_subsecond_boundary(self):
        with tempfile.TemporaryDirectory() as root:
            paths = self.chunks(root)
            cap, consumed = self.capture(paths)
            detector = ShortFlashDetector(25, 5)
            events, observed = [], []
            for diff in image_processing.create_diff_images(
                cap, interval=.25, duration=1, buffer_duration=4,
                is_rtsp=True, include_frame_window=True,
            ):
                for candidate in detector.consume(diff[-1], diff[1][0], final=cap.analysis_final_frame):
                    events.append((candidate.start_frame, candidate.end_frame))
                    observed.append(diff[-2])
            cap.release()
            self.assertEqual(events, [(24, 25)])
            self.assertLessEqual(observed[0], 60, "boundary detection waited for a minute or buffer warmup")
            self.assertEqual(consumed, paths)
            self.assertFalse(cap.new_chunks)

    def test_live_detector_saves_one_complete_boundary_clip(self):
        with tempfile.TemporaryDirectory() as root:
            paths = self.chunks(root)
            cap, _ = self.capture(paths)
            with (
                mock.patch.object(video_processing.model, "predict_noise_gate", return_value=None),
                mock.patch.object(video_processing.model, "predict_meteor_event", return_value=.99),
                mock.patch.object(video_processing.model, "get_meteor_decision_threshold", return_value=.5),
                mock.patch.object(video_processing.image_processing, "detect_lines", return_value=[]),
                mock.patch.object(config, "CUTOUT_SIZE", 64),
                mock.patch.object(config, "FINER_CUTOUT_SIZE", 100),
            ):
                results = video_processing.create_line_video_clips(
                    source=str(paths[0]), is_rtsp=True, interval=.25, duration=1,
                    buffer_duration=4, meteor_save_path=root, not_meteor_save_path=root,
                    save_options={"video": False, "cutout": False, "full": False,
                        "composite": False, "info": False, "summary": False, "full_video": True,
                        "ml_training_export_enabled": False, "_rtsp_live_capture": cap,
                        "rtsp_analysis_context": {"base_datetime": cap.started_at}},
                )
            meteors = [event for event in results if event["label"] == "meteor"]
            self.assertEqual(len(meteors), 1)
            event = meteors[0]
            self.assertEqual((event["detection_start_frame"], event["detection_end_frame"]), (24, 25))
            saved = event["saved_paths"]["full_video"]
            self.assertGreater(rtsp_segments._properties(saved)[1], 40)
            self.assertEqual(event["timestamp"].date(), datetime(2026, 10, 12).date())

    def test_schedule_crops_receive_meteor_with_individual_outputs_off(self):
        import rtsp_schedule_products
        with tempfile.TemporaryDirectory() as root:
            paths = self.chunks(root)
            cap, _ = self.capture(paths)
            service = rtsp_schedule_products.ScheduleProducts(root,
                {"blend_image": True, "constellations": False}, (20, 0, 20, 1),
                threading.Event(), status=lambda _message: None,
                box_detector=lambda _image, _log: [])
            with (
                mock.patch.object(video_processing.model, "predict_noise_gate", return_value=None),
                mock.patch.object(video_processing.model, "predict_meteor_event", return_value=.99),
                mock.patch.object(video_processing.model, "get_meteor_decision_threshold", return_value=.5),
                mock.patch.object(video_processing.image_processing, "detect_lines", return_value=[]),
                mock.patch.object(config, "CUTOUT_SIZE", 64),
                mock.patch.object(config, "FINER_CUTOUT_SIZE", 100),
            ):
                video_processing.create_line_video_clips(
                    source=str(paths[0]), is_rtsp=True, interval=.25, duration=1,
                    buffer_duration=4, meteor_save_path=root, not_meteor_save_path=root,
                    save_options={"video": False, "cutout": False, "full": False,
                        "composite": False, "info": False, "summary": False,
                        "full_video": False, "ml_training_export_enabled": False,
                        "_rtsp_live_capture": cap, "_schedule_event_callback": service.capture_event,
                        "rtsp_analysis_context": {"base_datetime": cap.started_at}},
                )
            job = next(iter(service.jobs.values()))
            self.assertEqual(len(job["events"]), 1)
            event = job["events"][0]
            self.assertGreater(event["count"], 40)
            frames = [service._event_frame(job, event, i) for i in range(event["count"])]
            self.assertGreater(max(int(frame.max()) for frame in frames), 150)
            output = Path(root)/"products"
            output.mkdir()
            result = service._blends(job, output)
            self.assertTrue(Path(result["blend_image"]).exists())
            self.assertEqual(event["box_source"], "detected_line")

    def test_session_or_clock_gap_resets_the_stream(self):
        with tempfile.TemporaryDirectory() as root:
            first = self.chunks(root, count=1)
            other = self.chunks(root, count=1, session="001122334455")
            cap, _ = self.capture(first + other)
            frames = 0
            while cap.read()[0]:
                frames += 1
            self.assertEqual(frames, 12)
            self.assertEqual(cap.pending, other[0])
            cap.release()
        with tempfile.TemporaryDirectory() as root:
            paths = self.chunks(root, count=2, gap=1)
            cap, _ = self.capture(paths)
            while cap.read()[0]:
                pass
            self.assertEqual(cap.pending, paths[1])
            cap.release()

    def test_mean_analysis_keeps_its_state_across_files(self):
        with tempfile.TemporaryDirectory() as root:
            paths = self.chunks(root, count=3)
            raw, _ = self.capture(paths)
            reference = temporal_mean.TemporalMeanStream(5)
            expected = []
            while True:
                ok, frame = raw.read()
                if not ok:
                    break
                expected.extend(reference.push(frame))
            expected.extend(reference.flush())
            raw.release()
            averaged, _ = self.capture(paths, mean_frames=5)
            actual = []
            while True:
                ok, frame = averaged.read()
                if not ok:
                    break
                actual.append(frame)
            averaged.release()
            self.assertEqual(len(actual), 38)
            self.assertTrue(all(np.array_equal(a, b) for a, b in zip(actual, expected)))

    def test_preprocessing_pipeline_accepts_subsecond_duration(self):
        with tempfile.TemporaryDirectory() as root:
            pipeline = noise_twin_pipeline.RtspNoiseTwinPipeline(
                "rtsp://synthetic", root, "model", lambda *_: True,
                threading.Event(), continuous_analysis=True,
            )
            command = pipeline._ffmpeg_command()
            self.assertEqual(command[command.index("-segment_time") + 1], "0.5")
            self.assertEqual(command[command.index("-strftime") + 1], "0")
            self.assertIn("-segment_list", command)
            self.assertTrue(command[-1].endswith("chunk_%09d.mp4"))

    def test_opencv_subsecond_capture_has_unique_names_and_exact_frame_clock(self):
        cancel = threading.Event()
        class Camera:
            remaining = 50
            def isOpened(self):
                return True
            def get(self, property_id):
                return {cv2.CAP_PROP_FPS: 25, cv2.CAP_PROP_FRAME_WIDTH: 220,
                        cv2.CAP_PROP_FRAME_HEIGHT: 180}.get(property_id, 0)
            def read(self):
                if not self.remaining:
                    cancel.set()
                    return False, None
                self.remaining -= 1
                return True, np.full((180, 220, 3), 10 + self.remaining, np.uint8)
            def release(self):
                pass
        with tempfile.TemporaryDirectory() as root:
            with (mock.patch.object(config, "RTSP_USE_NVIDIA_HWACCEL", False),
                  mock.patch.object(file_utils.utils, "create_rtsp_capture", return_value=Camera())):
                file_utils.save_rtsp_video_segments("rtsp://synthetic", root, .5, cancel)
            paths = rtsp_live.ChunkCatalog(root).paths()
            self.assertEqual([rtsp_live.timing(p)["frame_count"] for p in paths], [12, 13, 12, 13])
            starts = [media_time.get_media_start_time(str(p))[0] for p in paths]
            self.assertEqual([(t - starts[0]).total_seconds() for t in starts], [0, .48, 1, 1.48])

    def test_cancel_retains_the_not_yet_confirmed_tail(self):
        with tempfile.TemporaryDirectory() as root:
            paths = self.chunks(root, count=8)
            cap, consumed = self.capture(paths)
            generator = image_processing.create_diff_images(
                cap, interval=.25, duration=1, buffer_duration=4, is_rtsp=True)
            next(generator)
            cap.cancel.set()
            self.assertEqual(sum(1 for _ in generator), 0)
            self.assertFalse(consumed)
            cap.release()

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg required")
    def test_minute_archive_keeps_all_frames_and_deletes_only_consumed_sources(self):
        with tempfile.TemporaryDirectory() as root:
            directory = rtsp_live.capture_directory(root)
            command = [shutil.which("ffmpeg"), "-nostdin", "-v", "error", "-y",
                "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=25:duration=61"]
            command.extend(rtsp_live.capture_options(directory))
            command.extend(file_utils._rtsp_ffmpeg_segment_output_options(
                str(directory / "chunk_%09d.mp4"), .5, 25))
            subprocess.run(command, check=True, capture_output=True, timeout=30)
            catalog = rtsp_live.ChunkCatalog(root)
            paths = catalog.paths()
            self.assertGreaterEqual(len(paths), 120)
            start = media_time.get_media_start_time(str(paths[0]))[0]
            archive = rtsp_live.MinuteArchiver(root, status=lambda _text: None)
            archive.compact()
            outputs = list(Path(root).glob("[0-9]*/[0-9]*/*_minute_*.mp4"))
            self.assertEqual(len(outputs), 1)
            self.assertEqual(rtsp_segments._properties(outputs[0])[1], 1500)
            self.assertEqual(media_time.get_media_start_time(str(outputs[0]))[0], start)
            self.assertTrue(all(path.exists() for path in paths), "decoder still needs original files")
            for path in paths:
                rtsp_live.write_json(str(path) + ".analysis_consumed.json", {"consumed": True})
            # Simulate a crash between publishing the archive and marking all members.
            Path(str(paths[0]) + ".archived.json").unlink()
            restarted = rtsp_live.MinuteArchiver(root, status=lambda _text: None)
            restarted.compact()
            self.assertFalse(paths[0].exists())
            remaining = catalog.paths()
            self.assertEqual(sum(rtsp_live.timing(p)["frame_count"] for p in remaining), 25)
            _, videos = timelapse_creator.get_files_from_path(root)
            self.assertEqual(videos, [str(outputs[0])])
            self.assertTrue(video_processor.is_video_file(str(outputs[0])))
            restarted.compact(flush=True)
            outputs = list(Path(root).glob("[0-9]*/[0-9]*/*_minute_*.mp4"))
            self.assertEqual(sum(rtsp_segments._properties(p)[1] for p in outputs), 1525)
            self.assertFalse(catalog.paths())

    def test_failed_concat_retains_recordings(self):
        with tempfile.TemporaryDirectory() as root:
            paths = self.chunks(root, count=4)
            archiver = rtsp_live.MinuteArchiver(root, seconds=2)
            with mock.patch("rtsp_live.subprocess.run", return_value=subprocess.CompletedProcess(
                    [], 1, "", "synthetic failure")):
                with self.assertRaisesRegex(IOError, "synthetic failure"):
                    archiver.compact()
            self.assertTrue(all(path.is_file() for path in paths))
            self.assertFalse(list(Path(root).rglob("*.archived.json")))

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg required")
    def test_saved_preprocessing_has_continuous_temporal_context(self):
        for mean_frames in (0, 5):
            with self.subTest(mean_frames=mean_frames), tempfile.TemporaryDirectory() as root:
                pipeline = noise_twin_pipeline.RtspNoiseTwinPipeline(
                    "rtsp://synthetic", root, "synthetic-model", lambda *_: True,
                    threading.Event(), require_validated=False, continuous_analysis=True,
                    temporal_mean_frames=mean_frames,
                    encoding_options={"codec": "h264", "quality": "lossless"},
                )
                paths = self.chunks(pipeline.spool_root, count=3)
                cap, _ = self.capture(paths)
                originals = []
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    originals.append(frame)
                cap.release()
                engine = mock.Mock()
                engine.metadata.model_id = "synthetic"
                engine.metadata.format_version = 1
                engine.infer.side_effect = lambda frames: noise_twin.NoiseTwinResult(
                    frames[noise_twin.TEMPORAL_RADIUS], frames[noise_twin.TEMPORAL_RADIUS], 0, 0, 0, 1)
                with mock.patch("noise_twin_pipeline.noise_twin.NoiseTwinEngine", return_value=engine) as factory:
                    worker = threading.Thread(target=pipeline._denoise_loop)
                    worker.start()
                    for path in paths:
                        pipeline.raw_queue.put(str(path))
                    deadline = time.monotonic() + 8
                    while pipeline.raw_queue.unfinished_tasks and time.monotonic() < deadline:
                        time.sleep(.01)
                    pipeline.cancel_event.set()
                    worker.join(timeout=3)
                self.assertFalse(worker.is_alive())
                self.assertEqual(pipeline.raw_queue.unfinished_tasks, 0)
                self.assertTrue(pipeline.analysis_queue.empty())
                transformed = rtsp_live.ChunkCatalog(root).paths()
                self.assertEqual(len(transformed), 3)
                self.assertFalse(any(path.exists() for path in paths))
                if not mean_frames:
                    factory.assert_called_once()
                    self.assertEqual(engine.infer.call_count, len(originals))
                    window = engine.infer.call_args_list[24].args[0]
                    self.assertTrue(np.array_equal(window[noise_twin.TEMPORAL_RADIUS + 1], originals[25]))
                else:
                    factory.assert_not_called()
                    reference = temporal_mean.TemporalMeanStream(5)
                    expected = []
                    for frame in originals:
                        expected.extend(reference.push(frame))
                    expected.extend(reference.flush())
                    actual, _ = self.capture(transformed)
                    decoded = []
                    while True:
                        ok, frame = actual.read()
                        if not ok:
                            break
                        decoded.append(frame)
                    actual.release()
                    self.assertEqual(len(decoded), len(expected))
                    self.assertLess(max(np.abs(a.astype(float) - b.astype(float)).mean()
                                        for a, b in zip(decoded, expected)), 1.5)


if __name__ == "__main__":
    unittest.main()
