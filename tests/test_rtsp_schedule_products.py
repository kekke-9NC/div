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

    def test_shared_crops_one_ai_call_for_image_and_video(self):
        with tempfile.TemporaryDirectory() as root:
            service = self.service(root, blend_image=True, blend_video=True)
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
            service = self.service(root, blend_image=True)
            service.box_detector = mock.Mock(side_effect=RuntimeError("offline"))
            service.capture_event([np.ones((64, 96, 3), np.uint8)*50], 10,
                datetime(2026, 10, 12, 20), ((30, 20), (35, 25)))
            job = next(iter(service.jobs.values()))
            service._prepare_event(job, job["events"][0])
            restarted = self.service(root, blend_image=True)
            recovered = next(iter(restarted.jobs.values()))
            event = recovered["events"][0]
            self.assertEqual(event["box_source"], "detected_line")
            self.assertTrue(event["ai_done"])
            self.assertTrue((restarted.cache.root/event["key"]).exists())

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
            service = self.service(root, blend_image=True)
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
