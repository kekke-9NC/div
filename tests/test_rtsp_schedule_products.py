import json
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

import cv2
import numpy as np

import rtsp_live
import rtsp_schedule_products as products
import rtsp_segments


class ScheduleProductTests(unittest.TestCase):
    def tearDown(self):
        products.stop_background_products()

    def service(self, root, **options):
        return products.ScheduleProducts(root, dict(constellations=False, timestamp=False,
            duration=1, fps=6, **options), (20, 0, 20, 1), threading.Event(),
            status=lambda _message: None, box_detector=lambda _image, _log: [])

    def archive(self, root, start, count=30, fps=10, name="00_00_aabbccddeeff_minute_000000000.mp4"):
        directory = Path(root)/start.strftime("%Y%m%d/%H")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory/name
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (96, 64))
        self.assertTrue(writer.isOpened())
        for index in range(count):
            frame = np.full((64, 96, 3), index%160+10, np.uint8)
            cv2.line(frame, (25, 20), (35, 30), (200, 200, 200), 2)
            writer.write(frame)
        writer.release()
        rtsp_segments.write_capture_time(path, start)
        rtsp_live.timing(path)
        return path

    def test_midnight_window_and_exclusive_end(self):
        clock = (17, 0, 7, 0)
        self.assertEqual(products.schedule_window(datetime(2026, 10, 13, 3), *clock),
            (datetime(2026, 10, 12, 17), datetime(2026, 10, 13, 7)))
        self.assertIsNone(products.schedule_window(datetime(2026, 10, 13, 7), *clock))
        self.assertIsNone(products.schedule_window(datetime(2026, 10, 13, 12), *clock))
        self.assertIsNone(products.schedule_window(datetime(2026, 10, 13, 12), 12, 0, 12, 0))

    def test_default_options_and_numeric_validation(self):
        values = products.normalize_options(None)
        self.assertEqual((values["duration"], values["fps"]), (15, 60))
        self.assertTrue(values["constellations"])
        self.assertFalse(values["ai_crop"])
        self.assertEqual(values["crop_margin"], 10)
        self.assertFalse(products.enabled(values))
        for values in ({"duration": float("nan")}, {"fps": 0}, {"crop_margin": -1}, {"duration": 300, "fps": 120}):
            with self.assertRaises(ValueError):
                products.normalize_options(values)

    def test_disabled_outputs_ignore_invalid_optional_settings(self):
        values = {"duration": "invalid", "fps": float("nan"), "crop_margin": -1,
                  "timestamp_position": "invalid"}
        with mock.patch.object(products, "normalize_options", side_effect=AssertionError("validated disabled output")):
            self.assertEqual(products.processing_options(values), products.DEFAULTS)
        with self.assertRaises(ValueError):
            products.processing_options(dict(values, timelapse=True))

    def test_rtsp_entry_point_bypasses_disabled_product_validation(self):
        import file_utils
        with mock.patch.object(file_utils, "_rtsp_save_and_process_thread_target", autospec=True,
                               return_value="ordinary_rtsp") as recorder, \
             mock.patch.object(products, "normalize_options", side_effect=AssertionError("validated")):
            result = file_utils.rtsp_save_and_process_thread_target("rtsp://camera", time_limit_enabled=True,
                schedule_product_options={"duration": "bad", "fps": 0})
        self.assertEqual(result, "ordinary_rtsp")
        recorder.assert_called_once_with("rtsp://camera", time_limit_enabled=True)

    def failed_capture(self, service, start):
        import video_processing
        frames = [np.zeros((64, 96, 3), np.uint8) for _ in range(3)]
        video_processing._cache_scheduled_event({
            "_schedule_event_callback": service.capture_event,
            "_schedule_event_error_callback": service.capture_event_failed},
            frames, 10, start, ((25, 20), (35, 30)), True)

    def test_cache_write_failure_blocks_completion_and_deletion_until_recovery(self):
        with tempfile.TemporaryDirectory() as root:
            service = self.service(root, blend_image=True, concatenate=True)
            start = datetime(2026, 10, 12, 20)
            with mock.patch.object(service.cache, "put", side_effect=OSError("disk full")):
                self.failed_capture(service, start)
            job = next(iter(service.jobs.values()))
            self.assertTrue(job["events"][0]["capture_pending"])
            with self.assertRaises(IOError):
                service.tick(start+timedelta(minutes=1))
            with self.assertRaises(IOError):
                service._finish(job)
            with self.assertRaises(IOError):
                service._concatenate(job, Path(root)/"recording.mp4")
            self.assertNotEqual(job["state"], "complete")
            # Original archives become available after the failed cache write.
            source = self.archive(root, start)
            service.tick(start+timedelta(minutes=1))
            self.assertFalse(job["events"][0]["capture_pending"])
            self.assertEqual(job["state"], "complete")
            self.assertTrue(Path(job["outputs"]["blend_image"]).exists())
            self.assertFalse(source.exists(), "replacement did not resume after recovery")

    def test_manifest_failure_retries_metadata_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as root:
            service = self.service(root, blend_image=True, concatenate=True)
            start = datetime(2026, 10, 12, 20)
            job = service._job(start)
            source = self.archive(root, start)
            with mock.patch.object(service, "_save", side_effect=OSError("manifest write failed")):
                self.failed_capture(service, start)
                with self.assertRaises(IOError):
                    service._finish(job)
            self.assertTrue(source.exists())
            self.assertTrue(job["events"][0]["capture_pending"])
            self.assertTrue(service._capture_guard(job))
            service.tick(start+timedelta(minutes=1))
            self.assertEqual(job["state"], "complete")
            self.assertEqual(len(job["events"]), 1)
            self.assertTrue(Path(job["outputs"]["blend_image"]).exists())
            self.assertFalse(source.exists())

    def test_restart_with_unregistered_capture_blocks_empty_blend_success(self):
        with tempfile.TemporaryDirectory() as root:
            service = self.service(root, blend_image=True, concatenate=True)
            start = datetime(2026, 10, 12, 20)
            service._job(start)
            source = self.archive(root, start)
            with mock.patch.object(service, "_save", side_effect=OSError("manifest write failed")):
                self.failed_capture(service, start)
            restarted = self.service(root, blend_image=True, concatenate=True)
            job = next(iter(restarted.jobs.values()))
            self.assertEqual(job["events"], [], "simulated disk manifest unexpectedly contained the event")
            self.assertTrue(job["capture_unknown"])
            with self.assertRaises(IOError):
                restarted.tick(start+timedelta(minutes=1))
            self.assertNotEqual(job["state"], "complete")
            self.assertTrue(source.exists())
            self.assertFalse(job.get("blends_done", False))

    def test_restart_recovers_registered_capture_after_cache_failure(self):
        with tempfile.TemporaryDirectory() as root:
            service = self.service(root, blend_image=True, concatenate=True)
            start = datetime(2026, 10, 12, 20)
            source = self.archive(root, start)
            with mock.patch.object(service.cache, "put", side_effect=OSError("cache write failed")):
                self.failed_capture(service, start)
            restarted = self.service(root, blend_image=True, concatenate=True)
            job = next(iter(restarted.jobs.values()))
            self.assertTrue(job["events"][0]["capture_pending"])
            restarted.tick(start+timedelta(minutes=1))
            self.assertEqual(job["state"], "complete")
            image = cv2.imread(job["outputs"]["blend_image"])
            self.assertGreater(int(image[25, 30].max()), 150)
            self.assertFalse(source.exists())

    def test_cache_quota_refusal_is_recovered_from_original(self):
        with tempfile.TemporaryDirectory() as root:
            service = self.service(root, blend_image=True, concatenate=True)
            start = datetime(2026, 10, 12, 20)
            with mock.patch.object(service.cache, "put", return_value=False):
                self.failed_capture(service, start)
            job = next(iter(service.jobs.values()))
            self.assertTrue(job["events"][0]["capture_pending"])
            with self.assertRaises(IOError):
                service._finish(job)
            source = self.archive(root, start)
            service.tick(start+timedelta(minutes=1))
            self.assertEqual(job["state"], "complete")
            self.assertFalse(source.exists())

    def test_recovery_manifest_failure_keeps_capture_pending(self):
        with tempfile.TemporaryDirectory() as root:
            service = self.service(root, blend_image=True, concatenate=True)
            start = datetime(2026, 10, 12, 20)
            source = self.archive(root, start)
            with mock.patch.object(service.cache, "put", side_effect=OSError("cache write failed")):
                self.failed_capture(service, start)
            job = next(iter(service.jobs.values()))
            service._archives(job)
            save = service._save
            def fail_recovery_commit(current):
                if current["events"] and not current["events"][0]["capture_pending"]:
                    raise OSError("recovery commit failed")
                save(current)
            with mock.patch.object(service, "_save", side_effect=fail_recovery_commit):
                with self.assertRaises(OSError):
                    service._recover_captures(job)
            self.assertTrue(job["events"][0]["capture_pending"])
            with self.assertRaises(IOError):
                service._finish(job)
            self.assertTrue(source.exists())

    def test_quota_evicts_oldest_and_manual_selective_clear(self):
        with tempfile.TemporaryDirectory() as root:
            cache = products.ProductCache(root, 9)
            self.assertTrue(cache.put("old", {"a": b"1234"}))
            self.assertTrue(cache.put("next", {"a": b"5678"}))
            self.assertTrue(cache.put("new", {"a": b"abcd"}))
            self.assertIsNone(cache.read("old", "a"))
            self.assertEqual(cache.size(), 8)
            self.assertFalse(cache.put("oversize", {"a": b"0"*10}))
            cache.clear(["next"])
            self.assertEqual(cache.read("new", "a"), b"abcd")
            cache.clear()
            self.assertEqual(cache.size(), 0)

    def test_incomplete_cache_entry_is_removed_on_restart(self):
        with tempfile.TemporaryDirectory() as root:
            partial = Path(root)/".entry_interrupted"
            partial.mkdir()
            (partial/"frame.png").write_bytes(b"0"*20)
            cache = products.ProductCache(root, 9)
            self.assertEqual(cache.size(), 0)
            self.assertTrue(cache.put("new", {"a": b"1234"}))
            self.assertLessEqual(cache.size(), 9)

    def test_video_build_reservation_does_not_block_delivery_or_get_removed(self):
        with tempfile.TemporaryDirectory() as root:
            cache = products.ProductCache(root, 20)
            entered, release = threading.Event(), threading.Event()
            def writer(directory):
                (directory/"recording.mp4").write_bytes(b"1234")
                entered.set()
                self.assertTrue(release.wait(3))
                self.assertTrue(directory.exists())
            result = []
            worker = threading.Thread(target=lambda: result.append(cache.build("video", 16, writer)))
            worker.start()
            try:
                self.assertTrue(entered.wait(1))
                # Opening cache management must not delete a running writer.
                manager = products.ProductCache(root, 20)
                self.assertTrue(manager.put("event", {"frame.png": b"abcd"}))
                self.assertFalse(manager.put("full", {"frame.png": b"x"*5}))
            finally:
                release.set()
                worker.join(4)
            self.assertEqual(result, [True])
            self.assertEqual(cache.read("video", "recording.mp4"), b"1234")

    def test_manual_clear_defers_active_inputs_and_discards_active_build(self):
        with tempfile.TemporaryDirectory() as root:
            cache = products.ProductCache(root, 10)
            cache.put("input", {"recording.mp4": b"12345678"})
            with cache.pin(["input"]) as available:
                cache.clear()
                self.assertTrue(available["input"].is_dir())
                self.assertFalse(cache.put("other", {"frame.png": b"123"}))
            self.assertEqual(cache.size(), 0)
            def writer(directory):
                (directory/"recording.mp4").write_bytes(b"1234")
                cache.clear()
            self.assertFalse(cache.build("video", 8, writer))
            self.assertEqual(cache.size(), 0)
            self.assertFalse(list(Path(root).glob(".entry_*")))

    def test_failed_video_cache_build_is_not_published(self):
        with tempfile.TemporaryDirectory() as root:
            cache = products.ProductCache(root, 20)
            def writer(directory):
                (directory/"recording.mp4").write_bytes(b"incomplete")
                raise InterruptedError("app closed")
            with self.assertRaises(InterruptedError):
                cache.build("video", 20, writer)
            self.assertEqual(cache.size(), 0)
            self.assertIsNone(cache.read("video", "recording.mp4"))

    def test_cutouts_are_prepared_before_end_and_shared_without_masking_again(self):
        with tempfile.TemporaryDirectory() as root:
            start = datetime(2026, 10, 12, 20)
            service = self.service(root, blend_image=True, blend_video=True, ai_crop=True)
            self.archive(root, start)
            frames = [np.full((64, 96, 3), 200, np.uint8) for _ in range(3)]
            service.capture_event(frames, 10, start, ((25, 20), (35, 30)))
            service.box_detector = mock.Mock(return_value=[(400, 400, 600, 600)])
            service.tick(start+timedelta(seconds=10))
            job = next(iter(service.jobs.values()))
            event = job["events"][0]
            self.assertEqual(job["state"], "recording")
            self.assertTrue(event["cutouts_prepared"])
            cached = products.decode_png(service.cache.read(event["cutout_key"], "000000.png"))
            self.assertEqual(int(cached[0, 0].max()), 0)
            self.assertEqual(int(cached[cached.shape[0]//2, cached.shape[1]//2].max()), 200)
            with mock.patch.object(service, "_masked_crop", side_effect=AssertionError("cut twice")):
                output = Path(root)/"products"
                output.mkdir()
                results = service._blends(job, output)
            self.assertEqual(set(results), {"blend_image", "blend_video"})
            service.box_detector.assert_called_once()

    def test_restart_rebuilds_missing_and_corrupt_caches_without_repeating_ai(self):
        with tempfile.TemporaryDirectory() as root:
            start = datetime(2026, 10, 12, 20)
            service = self.service(root, timelapse=True, blend_image=True, concatenate=True)
            source = self.archive(root, start, 600)
            service.capture_event([np.ones((64, 96, 3), np.uint8)*200]*3, 10,
                start, ((25, 20), (35, 30)))
            service.tick(start+timedelta(seconds=30))
            job = next(iter(service.jobs.values()))
            event = job["events"][0]
            sample_key = job["id"]+"_sample_000000"
            cutout_key = event["cutout_key"]
            concat_key = job["concat_parts"][0]["key"]
            service.cache.clear([sample_key, cutout_key])
            (service.cache.root/concat_key/"recording.mp4").write_bytes(b"interrupted")
            restarted = self.service(root, timelapse=True, blend_image=True, concatenate=True)
            restarted.existing_only = True
            restarted.box_detector = mock.Mock(side_effect=AssertionError("AI repeated"))
            restarted.tick(start+timedelta(seconds=40))
            recovered = next(iter(restarted.jobs.values()))
            self.assertEqual(recovered["state"], "recording")
            self.assertIsNotNone(restarted.cache.read(sample_key, "frame.png"))
            self.assertTrue(restarted._cutouts_valid(event, cutout_key))
            restarted._validate_concat(restarted.cache.root/concat_key/"recording.mp4",
                recovered["concat_parts"][0]["expected"])
            restarted.box_detector.assert_not_called()
            self.assertTrue(source.exists())
            persisted = json.loads((restarted.job_root/job["id"]/"job.json").read_text())
            self.assertTrue(persisted["events"][0]["cutouts_prepared"])
            self.assertTrue(persisted["concat_parts"][0]["prepared"])

    def test_cached_concat_groups_are_reused_at_end_and_keep_originals_until_then(self):
        with tempfile.TemporaryDirectory() as root:
            service = self.service(root, concatenate=True)
            start = datetime(2026, 10, 12, 20)
            sources = [self.archive(root, start+timedelta(seconds=i*3),
                name=f"00_{i*3:02d}_aabbccddeeff_minute_{i:09d}.mp4") for i in range(11)]
            service.tick(start+timedelta(seconds=35))
            service.tick(start+timedelta(seconds=40))
            job = next(iter(service.jobs.values()))
            self.assertEqual([len(part["sources"]) for part in job["concat_parts"]], [10, 1])
            self.assertTrue(all(part["prepared"] for part in job["concat_parts"]))
            self.assertTrue(all(path.exists() for path in sources))
            with mock.patch.object(service, "_render_concat", wraps=service._render_concat) as render:
                service.tick(start+timedelta(minutes=1))
            self.assertEqual(render.call_count, 1)
            self.assertTrue(all(Path(path).is_relative_to(service.cache.root)
                for path in render.call_args.args[0]))
            self.assertEqual(job["state"], "complete")
            self.assertEqual(rtsp_segments._properties(job["outputs"]["concatenate"])[1], 330)
            self.assertFalse(any(path.exists() for path in sources))

    def test_restart_worker_starts_without_rtsp_and_recording_reuses_it(self):
        with tempfile.TemporaryDirectory() as root:
            start = datetime(2026, 10, 12, 20)
            saved = self.service(root, concatenate=True)
            saved._job(start)
            with mock.patch.object(products.ScheduleProducts, "start") as launch:
                resumed = products.resume_pending_products(root, lambda _message: None)
                self.assertTrue(resumed.existing_only)
                self.assertEqual(len(resumed.jobs), 1)
                self.assertIs(products.resume_pending_products(root), resumed)
                recorded = products.recording_products(root, saved.options, saved.clock, lambda _message: None)
                self.assertIs(recorded, resumed)
                self.assertFalse(recorded.existing_only)
                # Registry reuse plus the start method's live-thread guard.
                self.assertEqual(launch.call_count, 2)
            products.stop_background_products()
            self.assertTrue(resumed.cancel.is_set())

    def test_rtsp_stop_leaves_background_cache_worker_running(self):
        import file_utils
        with tempfile.TemporaryDirectory() as root:
            recorder_cancel = threading.Event()
            with mock.patch.object(products.ScheduleProducts, "start"), \
                 mock.patch.object(file_utils, "_rtsp_save_and_process_thread_target", autospec=True,
                    return_value="stopped") as record:
                result = file_utils.rtsp_save_and_process_thread_target("rtsp://camera", save_root=root,
                    time_limit_enabled=True, cancel_flag=recorder_cancel,
                    schedule_product_options={"concatenate": True})
            service = record.call_args.kwargs["save_options"]["_schedule_products"]
            self.assertEqual(result, "stopped")
            recorder_cancel.set()
            self.assertFalse(service.cancel.is_set())
            products.stop_background_products()
            self.assertTrue(service.cancel.is_set())

    def test_existing_cache_restart_reuses_frames_and_launches_only_one_thread(self):
        with tempfile.TemporaryDirectory() as root:
            start = datetime(2026, 10, 12, 20)
            saved = self.service(root, timelapse=True, blend_image=True, concatenate=True)
            self.archive(root, start, 600)
            saved.capture_event([np.ones((64, 96, 3), np.uint8)*200]*3, 10,
                start, ((25, 20), (35, 30)))
            saved.tick(start+timedelta(seconds=30))
            entered, release = threading.Event(), threading.Event()
            original_tick = products.ScheduleProducts.tick
            def frozen_tick(service):
                # Call the original below to avoid recursing through the patch.
                with mock.patch.object(service, "_source_frame", side_effect=AssertionError("decoded again")), \
                     mock.patch.object(service, "_masked_crop", side_effect=AssertionError("cut again")), \
                     mock.patch.object(service, "_render_concat", side_effect=AssertionError("joined again")):
                    original_tick(service, start+timedelta(seconds=40))
                entered.set()
                release.wait(3)
            try:
                with mock.patch.object(products.ScheduleProducts, "tick", frozen_tick):
                    resumed = products.resume_pending_products(root, lambda _message: None)
                    self.assertTrue(entered.wait(2), "startup cache check failed")
                    first_thread = resumed.thread
                    recorded = products.recording_products(root, saved.options, saved.clock, lambda _message: None)
                    self.assertIs(recorded.thread, first_thread)
                    self.assertTrue(first_thread.is_alive())
            finally:
                release.set()
                products.stop_background_products()

    def test_startup_finishes_an_ended_job_without_starting_a_new_schedule(self):
        with tempfile.TemporaryDirectory() as root:
            start = datetime(2026, 10, 11, 20)
            saved = self.service(root, concatenate=True)
            source = self.archive(root, start)
            job = saved._job(start)
            # Older manifests did not contain a capture guard.
            (saved.job_root/job["id"]/"capture.guard").unlink()
            with mock.patch.object(products.ScheduleProducts, "start"):
                resumed = products.resume_pending_products(root, lambda _message: None)
            resumed.tick(start+timedelta(days=1, seconds=20))
            self.assertEqual(len(resumed.jobs), 1)
            self.assertEqual(resumed.jobs[job["id"]]["state"], "complete")
            self.assertFalse(source.exists())

    def test_resumed_job_keeps_capturing_if_unsaved_ui_options_reset(self):
        import file_utils
        with tempfile.TemporaryDirectory() as root:
            start = datetime(2026, 10, 12, 20)
            saved = self.service(root, blend_image=True)
            saved._job(start)
            with mock.patch.object(products.ScheduleProducts, "start"):
                resumed = products.resume_pending_products(root, lambda _message: None)
            with mock.patch.object(file_utils, "_rtsp_save_and_process_thread_target", autospec=True) as record:
                file_utils.rtsp_save_and_process_thread_target("rtsp://camera", save_root=root,
                    time_limit_enabled=False, schedule_product_options={"duration": "invalid"})
            callback = record.call_args.kwargs["save_options"]["_schedule_event_callback"]
            callback([np.zeros((64, 96, 3), np.uint8)], 10, start, ((25, 20), (35, 30)))
            self.assertEqual(len(resumed.jobs[saved._job(start)["id"]]["events"]), 1)
            self.assertTrue(resumed.existing_only)
            callback([np.zeros((64, 96, 3), np.uint8)], 10, start+timedelta(days=1), ((25, 20), (35, 30)))
            self.assertEqual(len(resumed.jobs), 1, "disabled outputs created a new job")

    def test_cancelled_ffmpeg_concat_is_reaped(self):
        with tempfile.TemporaryDirectory() as root:
            service = self.service(root, concatenate=True)
            process = mock.Mock()
            process.poll.return_value = None
            service.cancel.set()
            with mock.patch.object(products.subprocess, "Popen", return_value=process):
                with self.assertRaises(InterruptedError):
                    service._run_concat_copy([Path(root)/"input.mp4"], Path(root)/"output.mp4")
            process.terminate.assert_called_once()
            process.wait.assert_called_once_with(timeout=3)

    def test_shared_crops_one_ai_call_for_image_and_video(self):
        with tempfile.TemporaryDirectory() as root:
            service = self.service(root, blend_image=True, blend_video=True, ai_crop=True)
            frames = [np.zeros((64, 96, 3), np.uint8) for _ in range(3)]
            frames[1][20:25, 30:35] = 255
            service.capture_event(frames, 10, datetime(2026, 10, 12, 20),
                ((np.int32(30), np.int32(20)), (np.int32(35), np.int32(25))))
            job = next(iter(service.jobs.values()))
            detector = mock.Mock(return_value=[(0, 0, 1000, 1000)])
            service.box_detector = detector
            output = Path(root)/"out"
            output.mkdir()
            results = service._blends(job, output)
            self.assertEqual(detector.call_count, 1)
            self.assertEqual(job["events"][0]["box_source"], "local_ai")
            image = cv2.imread(results["blend_image"])
            self.assertEqual(int(image[22, 32, 0]), 255)
            self.assertEqual(int(image[0, 0, 0]), 0)
            self.assertEqual(rtsp_segments._properties(results["blend_video"])[1], 3)
            # Both outputs consume the exact same persisted event frames.
            self.assertEqual(len(list((service.cache.root/job["events"][0]["key"]).glob("*.png"))), 3)

    def test_ai_failure_uses_detection_rect_and_survives_restart(self):
        with tempfile.TemporaryDirectory() as root:
            service = self.service(root, blend_image=True, ai_crop=True)
            service.box_detector = mock.Mock(side_effect=RuntimeError("offline"))
            service.capture_event([np.ones((64, 96, 3), np.uint8)*50], 10,
                datetime(2026, 10, 12, 20), ((30, 20), (35, 25)))
            job = next(iter(service.jobs.values()))
            service._prepare_event(job, job["events"][0])
            restarted = self.service(root, blend_image=True, ai_crop=True)
            recovered = next(iter(restarted.jobs.values()))
            event = recovered["events"][0]
            self.assertEqual(event["box_source"], "detected_line")
            self.assertTrue(event["ai_done"])
            self.assertTrue((restarted.cache.root/event["key"]).exists())

    def test_default_micro_meteor_uses_native_coordinate_crop_without_ai(self):
        with tempfile.TemporaryDirectory() as root:
            service = self.service(root, blend_image=True, blend_video=True, crop_margin=4)
            start = datetime(2026, 10, 12, 20)
            frames = [np.zeros((64, 96, 3), np.uint8) for _ in range(3)]
            frames[1][20:24, 30:33] = 255
            service.box_detector = mock.Mock(side_effect=AssertionError("default invoked AI"))
            service.capture_event(frames, 10, start, ((30, 20), (32, 23)))
            job = next(iter(service.jobs.values()))
            event = job["events"][0]
            service._prepare_event(job, event)
            self.assertEqual(event["detected_line"], [[30, 20], [32, 23]])
            crop = service._cutout_frame(job, event, 1)
            self.assertEqual(crop.shape, (12, 11, 3))
            self.assertEqual(int(crop[4:8, 4:7].min()), 255)
            self.assertEqual(event["crop_method"], "coordinates")
            self.assertFalse(event["ai_done"])
            output = Path(root)/"products"
            output.mkdir()
            results = service._blends(job, output)
            image = cv2.imread(results["blend_image"])
            self.assertEqual(int(image[20:24, 30:33].min()), 255)
            service.box_detector.assert_not_called()

    def test_coordinate_crop_clips_margin_to_each_image_edge(self):
        with tempfile.TemporaryDirectory() as root:
            service = self.service(root, blend_image=True, crop_margin=4)
            start = datetime(2026, 10, 12, 20)
            cases = [(((0, 0), (2, 3)), (8, 7)), (((93, 60), (95, 63)), (8, 7)),
                     (((95, 0), (93, 3)), (8, 7)), (((2, 60), (0, 63)), (8, 7))]
            for index, (line, shape) in enumerate(cases):
                frame = np.zeros((64, 96, 3), np.uint8)
                cv2.line(frame, line[0], line[1], (255, 255, 255), 1)
                service.capture_event([frame], 10, start+timedelta(seconds=index), line)
                job = next(iter(service.jobs.values()))
                event = job["events"][-1]
                service._prepare_event(job, event)
                crop = service._cutout_frame(job, event, 0)
                self.assertEqual(crop.shape[:2], shape)
                self.assertEqual(np.count_nonzero(crop), np.count_nonzero(frame))

    def test_restart_migrates_old_ai_cache_to_coordinates_using_saved_rect(self):
        with tempfile.TemporaryDirectory() as root:
            start = datetime(2026, 10, 12, 20)
            old = self.service(root, blend_image=True, ai_crop=True)
            frames = [np.ones((64, 96, 3), np.uint8)*200]
            old.capture_event(frames, 10, start, ((25, 20), (35, 30)))
            job = next(iter(old.jobs.values()))
            old.box_detector = lambda _image, _log: [(400, 400, 600, 600)]
            event = job["events"][0]
            old._prepare_event(job, event)
            ai_key = event["cutout_key"]
            self.assertEqual(int(old._cutout_frame(job, event, 0)[0, 0].max()), 0)
            # Original releases lacked a method choice and explicit endpoints.
            job["options"].pop("ai_crop")
            event.pop("detected_line")
            event.pop("crop_method")
            event.pop("coordinate_rect_ready")
            job["blends_done"] = True
            old._save(job)
            resumed = self.service(root, blend_image=True)
            restored = next(iter(resumed.jobs.values()))
            resumed.box_detector = mock.Mock(side_effect=AssertionError("AI invoked"))
            resumed._source_frame = mock.Mock(return_value=frames[0])
            resumed.tick(start+timedelta(seconds=10))
            recovered = restored["events"][0]
            self.assertNotEqual(recovered["cutout_key"], ai_key)
            self.assertEqual(recovered["crop_method"], "coordinates")
            self.assertFalse(restored["blends_done"])
            self.assertEqual(int(resumed._cutout_frame(restored, recovered, 0)[0, 0].max()), 200)
            resumed.box_detector.assert_not_called()

    def test_short_seed_follows_whole_temporal_trail_and_ignores_static_stars(self):
        frames = [np.zeros((96, 120, 3), np.uint8) for _ in range(5)]
        for frame in frames:
            frame[25, 40] = 255  # Static star beyond the actual meteor.
        cv2.line(frames[1], (46, 40), (52, 46), (220, 220, 220), 2)
        cv2.line(frames[2], (52, 46), (60, 54), (220, 220, 220), 2)
        cv2.line(frames[3], (60, 54), (67, 61), (220, 220, 220), 2)
        line = ((52, 46), (54, 48))
        a, b, c, d = products.coordinate_rect(frames, line, 4)
        self.assertLessEqual(a, 45-4)
        self.assertLessEqual(b, 39-4)
        self.assertGreaterEqual(c, 69+4)
        self.assertGreaterEqual(d, 63+4)
        self.assertGreater(b, 25)
        self.assertLess(c-a, 60)
        self.assertLess(d-b, 60)

    def test_legacy_short_crop_recovers_whole_trail_from_original_frames(self):
        with tempfile.TemporaryDirectory() as root:
            service = self.service(root, blend_image=True, crop_margin=4)
            start = datetime(2026, 10, 12, 20)
            frames = [np.zeros((64, 96, 3), np.uint8) for _ in range(3)]
            cv2.line(frames[1], (20, 20), (50, 40), (255, 255, 255), 2)
            service.capture_event(frames, 10, start, ((30, 27), (32, 28)))
            job = next(iter(service.jobs.values()))
            event = job["events"][0]
            # Recreate the crop/metadata saved by an older release.
            event["rect"] = products.line_rect(((30, 27), (32, 28)), (96, 64), 4)
            event.pop("coordinate_rect_ready")
            event.pop("detected_line")
            service._source_frame = mock.Mock(side_effect=frames)
            service._prepare_event(job, event)
            a, b, c, d = event["rect"]
            self.assertLessEqual(a, 19-4)
            self.assertGreaterEqual(c, 52+4)
            self.assertGreaterEqual(d, 42+4)
            self.assertTrue(event["coordinate_rect_ready"])
            recovered = service._cutout_frame(job, event, 1)
            self.assertEqual(np.count_nonzero(recovered), np.count_nonzero(frames[1]))

    def test_deleted_event_cache_reconstructs_from_archive(self):
        with tempfile.TemporaryDirectory() as root:
            service = self.service(root, blend_image=True)
            start = datetime(2026, 10, 12, 20)
            source = self.archive(root, start)
            service.capture_event([np.zeros((64, 96, 3), np.uint8) for _ in range(3)], 10,
                start, ((25, 20), (35, 30)))
            job = next(iter(service.jobs.values()))
            service._archives(job)
            service.cache.clear()
            frame = service._event_frame(job, job["events"][0], 1)
            self.assertGreater(int(frame.max()), 150)
            service._prepare_event(job, job["events"][0])
            self.assertEqual(job["events"][0]["box_source"], "detected_line")
            self.assertTrue(source.exists())

    def test_full_schedule_timelapse_mask_and_cache_recovery(self):
        with tempfile.TemporaryDirectory() as root:
            service = self.service(root, timelapse=True)
            start = datetime(2026, 10, 12, 20)
            source = self.archive(root, start, 600)
            mask = np.full((64, 96), 255, np.uint8)
            mask[:, :16] = 0
            service.mask = mask
            service.tick(start+timedelta(seconds=30))
            job = next(iter(service.jobs.values()))
            self.assertEqual(len(job["samples"]), 6)
            original = mock.Mock(wraps=service._source_frame)
            service._source_frame = original
            service.tick(start+timedelta(seconds=40))
            self.assertEqual(original.call_count, 0, "prepared samples were decoded again")
            service.cache.clear()
            output = Path(root)/"timelapse.mp4"
            service._timelapse(job, output)
            fps, count, _ = rtsp_segments._properties(output)
            self.assertEqual((fps, count), (6, 6))
            capture = cv2.VideoCapture(str(output))
            ok, frame = capture.read()
            capture.release()
            self.assertTrue(ok)
            self.assertLess(float(frame[:, :12].mean()), 3)
            self.assertTrue(source.exists())

    def test_timelapse_repeated_slots_decode_once_with_exact_frames(self):
        with tempfile.TemporaryDirectory() as root:
            service = self.service(root, timelapse=True)
            job = service._job(datetime(2026, 10, 12, 20))
            originals = [np.full((64, 96, 3), value, np.uint8) for value in (17, 73)]
            job["samples"] = {"0": job["start"], "5": job["end"]}
            for index, frame in zip((0, 5), originals):
                service.cache.put(job["id"] + f"_sample_{index:06d}",
                                  {"frame.png": products.png_bytes(frame)})
            emitted = []
            with mock.patch.object(products, "decode_png", wraps=products.decode_png) as decode, \
                 mock.patch("video_encoding.write_repeated_images", side_effect=lambda _path, frames, *_args:
                            (emitted.extend(frame.copy() for frame in frames), 6)[1]), \
                 mock.patch.object(rtsp_segments, "_properties", return_value=(6, 6, (96, 64))):
                service._timelapse(job, Path(root)/"timelapse.mp4")
            self.assertEqual(decode.call_count, 2)
            self.assertEqual(len(emitted), 6)
            for actual, expected in zip(emitted, [originals[0]]*3 + [originals[1]]*3):
                np.testing.assert_array_equal(actual, expected)

    def test_missing_sample_slots_retry_when_archives_change(self):
        with tempfile.TemporaryDirectory() as root:
            service = self.service(root, timelapse=True)
            job = service._job(datetime(2026, 10, 12, 20))
            with mock.patch.object(service, "_source_frame", return_value=None) as read:
                service._prepare_samples(job)
                self.assertEqual(read.call_count, 6)
                service._prepare_samples(job)
                self.assertEqual(read.call_count, 6)
                job["archives"].append("new-minute.mp4")
                service._prepare_samples(job)
                self.assertEqual(read.call_count, 12)

    def test_source_clock_index_refreshes_when_metadata_is_replaced(self):
        with tempfile.TemporaryDirectory() as root:
            service = self.service(root, timelapse=True)
            start = datetime(2026, 10, 12, 20)
            source = self.archive(root, start)
            job = service._job(start)
            job["archives"] = [str(source)]
            self.assertIsNotNone(service._source_frame(job, start))
            data = rtsp_live.timing(source)
            data["start_time"] = (start+timedelta(seconds=10)).isoformat()
            rtsp_live.write_json(str(source)+".capture.json", data)
            self.assertIsNone(service._source_frame(job, start))
            self.assertIsNotNone(service._source_frame(job, start+timedelta(seconds=10)))

    def test_timelapse_preparation_overlaps_backlog_without_early_publication(self):
        with tempfile.TemporaryDirectory() as root:
            service = self.service(root, timelapse=True)
            start = datetime(2026, 10, 12, 20)
            source = self.archive(root, start, count=600)
            service._job(start)
            with mock.patch.object(service, "_timelapse", wraps=service._timelapse) as render:
                with mock.patch.object(service, "_drained", return_value=False):
                    service.tick(start+timedelta(seconds=61))
                    self.assertEqual(render.call_count, 0)
                    service.tick(start+timedelta(seconds=62))
                job = next(iter(service.jobs.values()))
                self.assertEqual(render.call_count, 1)
                self.assertEqual(job["state"], "recording")
                self.assertFalse(job["outputs"])
                self.assertTrue(source.exists())
                self.assertTrue((service.cache.root/service._timelapse_key(job)/"timelapse.mp4").exists())
                service.tick(start+timedelta(seconds=63))
                self.assertEqual(render.call_count, 1)
                self.assertEqual(job["state"], "complete")
                self.assertTrue(Path(job["outputs"]["timelapse"]).exists())

    def test_prepared_timelapse_key_changes_when_sample_pixels_are_replaced(self):
        with tempfile.TemporaryDirectory() as root:
            service = self.service(root, timelapse=True)
            job = service._job(datetime(2026, 10, 12, 20))
            job["samples"] = {"0": job["start"]}
            key = job["id"] + "_sample_000000"
            service.cache.put(key, {"frame.png": products.png_bytes(np.zeros((64, 96, 3), np.uint8))})
            before = service._timelapse_key(job)
            (service.cache.root/key/"frame.png").write_bytes(products.png_bytes(np.full((64, 96, 3), 80, np.uint8)))
            self.assertNotEqual(service._timelapse_key(job), before)

    def test_replaced_archive_rebuilds_existing_sample_pngs(self):
        with tempfile.TemporaryDirectory() as root:
            service = self.service(root, timelapse=True)
            start = datetime(2026, 10, 12, 20)
            source = self.archive(root, start, count=600)
            job = service._job(start)
            service._archives(job)
            service._prepare_samples(job)
            old_key = service._timelapse_key(job)
            metadata = rtsp_live.timing(source)
            # A published replacement can change pixels while keeping timing.
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (96, 64))
            for _ in range(600):
                writer.write(np.full((64, 96, 3), 210, np.uint8))
            writer.release()
            rtsp_live.write_json(str(source)+".capture.json", metadata)
            service._prepare_samples(job)
            self.assertNotEqual(service._timelapse_key(job), old_key)
            for index in job["samples"]:
                key = job["id"] + f"_sample_{int(index):06d}"
                frame = products.decode_png(service.cache.read(key, "frame.png"))
                self.assertGreater(frame.mean(), 200)

    def test_concat_validates_then_replaces_and_records_capture_time(self):
        with tempfile.TemporaryDirectory() as root:
            service = self.service(root, concatenate=True)
            start = datetime(2026, 10, 12, 20)
            first = self.archive(root, start)
            second = self.archive(root, start+timedelta(seconds=3), name="00_03_aabbccddeeff_minute_000000001.mp4")
            job = service._job(start)
            service.tick(start+timedelta(minutes=1))
            self.assertEqual(job["state"], "complete")
            output = Path(job["outputs"]["concatenate"])
            self.assertEqual(rtsp_segments._properties(output)[1], 60)
            self.assertEqual(rtsp_live.timing(output)["start_time"], start.isoformat())
            self.assertFalse(first.exists())
            self.assertFalse(second.exists())
            self.assertFalse(Path(str(first)+".capture.json").exists())

    def test_failed_concat_preserves_all_originals(self):
        with tempfile.TemporaryDirectory() as root:
            service = self.service(root, concatenate=True)
            start = datetime(2026, 10, 12, 20)
            source = self.archive(root, start)
            job = service._job(start)
            service._archives(job)
            with mock.patch.object(service, "_validate_concat", side_effect=IOError("invalid")):
                with self.assertRaises(IOError):
                    service._concatenate(job, Path(root)/"recording.mp4")
            self.assertTrue(source.exists())
            self.assertTrue(Path(str(source)+".capture.json").exists())
            self.assertNotEqual(job["state"], "replacing")

    def test_concat_uses_actual_recording_length_instead_of_stale_metadata(self):
        with tempfile.TemporaryDirectory() as root:
            service = self.service(root, concatenate=True)
            start = datetime(2026, 10, 12, 20)
            source = self.archive(root, start)
            marker = Path(str(source)+".capture.json")
            metadata = json.loads(marker.read_text())
            metadata["frame_count"] = 10
            rtsp_live.write_json(marker, metadata)
            job = service._job(start)
            service.tick(start+timedelta(minutes=1))
            self.assertEqual(rtsp_segments._properties(job["outputs"]["concatenate"])[1], 30)

    def test_boundary_is_frame_accurate_and_keeps_outside_footage(self):
        with tempfile.TemporaryDirectory() as root:
            service = self.service(root, concatenate=True)
            start = datetime(2026, 10, 12, 20)
            source = self.archive(root, start-timedelta(seconds=1), 30)
            job = service._job(start)
            service._archives(job)
            output = Path(root)/"recording.mp4"
            service._concatenate(job, output)
            self.assertEqual(rtsp_segments._properties(output)[1], 20)
            self.assertEqual(rtsp_live.timing(output)["start_time"], start.isoformat())
            self.assertTrue(source.exists(), "out-of-schedule footage was deleted")

    def test_interrupted_deletion_resumes_without_reencoding(self):
        with tempfile.TemporaryDirectory() as root:
            service = self.service(root, concatenate=True)
            start = datetime(2026, 10, 12, 20)
            source = self.archive(root, start)
            job = service._job(start)
            service._archives(job)
            output = Path(root)/"recording.mp4"
            original = Path.unlink
            def fail_source(path, *args, **kwargs):
                if path.resolve() == source.resolve():
                    raise PermissionError("simulated interruption")
                return original(path, *args, **kwargs)
            with mock.patch.object(Path, "unlink", fail_source):
                with self.assertRaises(PermissionError):
                    service._concatenate(job, output)
            self.assertEqual(job["state"], "replacing")
            self.assertTrue(output.exists())
            restarted = self.service(root, concatenate=True)
            with mock.patch.object(products.subprocess, "run", side_effect=AssertionError("reencoded")):
                restarted.tick(start+timedelta(minutes=1))
            self.assertFalse(source.exists())
            self.assertEqual(next(iter(restarted.jobs.values()))["state"], "complete")

    def test_finish_waits_for_analysis_and_archive_tail(self):
        with tempfile.TemporaryDirectory() as root:
            service = self.service(root, concatenate=True)
            start = datetime(2026, 10, 12, 20)
            source = self.archive(root, start)
            session = Path(root)/".rtsp_live/20261012/20/00_00_aabbccddeeff"
            session.mkdir(parents=True)
            rtsp_live.write_json(session/".session.json", {"start_time": start.isoformat()})
            chunk = session/"chunk_000000000.mp4"
            chunk.write_bytes(source.read_bytes())
            rtsp_segments.write_capture_time(chunk, start)
            job = service._job(start)
            service.flush_archives = mock.Mock()
            service.tick(start+timedelta(minutes=1))
            self.assertEqual(job["state"], "recording")
            self.assertTrue(source.exists())
            service.flush_archives.assert_called_once()
            for suffix in (".analysis_consumed.json", ".archived.json"):
                rtsp_live.write_json(str(chunk)+suffix, {})
            service.tick(start+timedelta(minutes=1))
            self.assertEqual(job["state"], "recording")
            chunk.unlink()
            service.tick(start+timedelta(minutes=1))
            self.assertEqual(job["state"], "complete")

    def test_no_meteors_is_successful_without_empty_outputs(self):
        with tempfile.TemporaryDirectory() as root:
            service = self.service(root, blend_image=True, blend_video=True)
            start = datetime(2026, 10, 12, 20)
            self.archive(root, start)
            job = service._job(start)
            service.tick(start+timedelta(minutes=1))
            self.assertEqual(job["state"], "complete")
            self.assertTrue(job["blends_done"])
            self.assertEqual(job["outputs"], {})

    def test_constellations_use_shared_obs_renderer_and_snapshot(self):
        from obs_meteor_overlay import sky_overlay
        with tempfile.TemporaryDirectory() as root:
            service = self.service(root)
            start = datetime(2026, 10, 12, 20)
            job = service._job(start)
            job["options"]["constellations"] = True
            payload = {"width": 96, "height": 64}
            layer = np.zeros((64, 96, 4), np.uint8)
            layer[20, 30] = [255, 0, 0, 255]
            with mock.patch.object(sky_overlay, "approved_model", return_value=(
                    {"line_opacity": .6}, payload, "model-sha")) as approved, \
                 mock.patch.object(sky_overlay, "render_layer", return_value=layer) as render:
                original = np.zeros((64, 96, 3), np.uint8)
                frame = service._annotate(job, original, start)
                service._annotate(job, original, start+timedelta(seconds=2))
            self.assertEqual(approved.call_count, 1)
            self.assertEqual(render.call_count, 2)
            self.assertEqual(frame[20, 30].tolist(), [0, 0, 255])
            self.assertEqual(job["sky_revision"], "model-sha")
            self.assertEqual(render.call_args.args[0], payload)
            self.assertEqual(render.call_args.args[3], .6)

    def test_slow_ai_does_not_hold_detector_delivery_lock(self):
        with tempfile.TemporaryDirectory() as root:
            service = self.service(root, blend_image=True, ai_crop=True)
            start = datetime(2026, 10, 12, 20)
            frames = [np.zeros((64, 96, 3), np.uint8)]
            service.capture_event(frames, 10, start, ((30, 20), (35, 25)))
            entered, release = threading.Event(), threading.Event()
            def slow_ai(_image, _log):
                entered.set()
                release.wait(3)
                return []
            service.box_detector = slow_ai
            worker = threading.Thread(target=service.tick, args=(start,))
            worker.start()
            self.assertTrue(entered.wait(1))
            delivered = threading.Event()
            def deliver():
                service.capture_event(frames, 10, start+timedelta(seconds=1), ((40, 20), (45, 25)))
                delivered.set()
            detector = threading.Thread(target=deliver)
            detector.start()
            try:
                self.assertTrue(delivered.wait(1), "AI blocked continuous detection")
            finally:
                release.set()
                worker.join(4)
                detector.join(4)


if __name__ == "__main__":
    unittest.main()
