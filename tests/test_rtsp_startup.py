import json
from datetime import datetime, timedelta
from pathlib import Path
import queue
import tempfile
import threading
import unittest
from unittest import mock

import gui_processing
import noise_twin_pipeline
import rtsp_live
import rtsp_schedule_products as products
import rtsp_startup


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.start = datetime(2026, 10, 11, 20)
        self.service = products.ScheduleProducts(self.root, {"blend_image": True},
            (20, 0, 21, 0), threading.Event(), status=lambda _message: None)
        self.job = self.service._job(self.start)

    def tearDown(self):
        products.stop_background_products()
        rtsp_startup.clear_choices()
        self.temporary.cleanup()

    def app(self, choice):
        app = object.__new__(gui_processing.ProcessingMixin)
        app._startup_analysis_ready = False
        app.progress_queue = queue.Queue()
        app.append_log = mock.Mock()
        app._prompt_startup_analysis = mock.Mock(return_value=choice)
        return app

    def prepare(self, app):
        with mock.patch.object(gui_processing.config, "RTSP_SAVE_ROOT", str(self.root)):
            return app._prepare_startup_analysis()

    def test_resume_requires_choice_and_reuses_saved_settings(self):
        app = self.app("resume")
        with mock.patch.object(products.ScheduleProducts, "start") as start:
            self.assertTrue(self.prepare(app))
        app._prompt_startup_analysis.assert_called_once()
        start.assert_called_once()
        resumed = products.pending_recording_products(self.root, lambda _message: None)
        self.assertEqual(resumed.jobs[self.job["id"]]["options"], self.job["options"])
        # Starting later must not ask again or create a second worker.
        self.assertTrue(self.prepare(app))
        app._prompt_startup_analysis.assert_called_once()

    def test_fresh_preserves_previous_manifest_and_excludes_it_when_recording_starts(self):
        marker = self.service.job_root / self.job["id"] / "job.json"
        before = marker.read_bytes()
        app = self.app("fresh")
        with mock.patch.object(products, "resume_pending_products") as resume:
            self.assertTrue(self.prepare(app))
        resume.assert_not_called()
        with mock.patch.object(products.ScheduleProducts, "start"):
            recorded = products.recording_products(self.root, self.service.options,
                self.service.clock, lambda _message: None)
        self.assertNotIn(self.job["id"], recorded.jobs)
        self.assertEqual(marker.read_bytes(), before)
        # A later app launch can still resume the preserved job.
        self.assertEqual(rtsp_startup.pending_jobs(self.root)[0]["id"], self.job["id"])

    def test_fresh_in_same_window_creates_separate_job_from_choice_time(self):
        when = self.start + timedelta(minutes=10)
        rtsp_startup.defer_previous(self.root, [self.job], when)
        fresh = products.ScheduleProducts(self.root, {"timelapse": True},
            self.service.clock, threading.Event(), status=lambda _message: None)
        self.assertIsNone(fresh._job(self.start))
        new = fresh._job(when)
        self.assertNotEqual(new["id"], self.job["id"])
        self.assertEqual(new["start"], when.isoformat())
        self.assertTrue(new["options"]["timelapse"])
        self.assertEqual(new["events"], [])
        self.assertEqual(fresh._job(when + timedelta(minutes=1))["id"], new["id"])

    def test_later_or_closed_dialog_blocks_manual_and_scheduled_start(self):
        app = self.app(None)
        app.after = mock.Mock()
        app._current_rtsp_schedule_window_key = mock.Mock()
        app.apply_advanced_settings_to_config = mock.Mock()
        with mock.patch.object(products, "resume_pending_products") as resume, \
             mock.patch.object(gui_processing.config, "RTSP_SAVE_ROOT", str(self.root)):
            self.assertFalse(app._prepare_startup_analysis())
            app._check_rtsp_schedule_autostart()
            app.start_processing()
        resume.assert_not_called()
        app._current_rtsp_schedule_window_key.assert_not_called()
        app.apply_advanced_settings_to_config.assert_not_called()
        self.assertFalse(app._startup_analysis_ready)
        app.after.assert_called_once()

    def test_empty_startup_needs_no_dialog_and_starts_no_worker(self):
        self.job["state"] = "complete"
        self.service._save(self.job)
        app = self.app("resume")
        with mock.patch.object(products, "resume_pending_products") as resume:
            self.assertTrue(self.prepare(app))
        app._prompt_startup_analysis.assert_not_called()
        resume.assert_not_called()

    def test_unreadable_manifest_keeps_start_blocked(self):
        (self.service.job_root / self.job["id"] / "job.json").write_text("broken")
        app = self.app("resume")
        with mock.patch.object(products, "resume_pending_products") as resume:
            self.assertFalse(self.prepare(app))
        resume.assert_not_called()
        self.assertFalse(app._startup_analysis_ready)

    def test_deferred_chunks_are_excluded_but_new_chunks_are_detected(self):
        directory = self.root / ".rtsp_live"
        directory.mkdir()
        old = directory / "chunk_old.mp4"
        old.touch()
        rtsp_startup.defer_previous(self.root, [self.job], self.start)
        new = directory / "chunk_new.mp4"
        new.touch()
        with mock.patch.object(rtsp_live.rtsp_segments, "is_finalized_mp4", return_value=True), \
             mock.patch.object(rtsp_live, "timing", return_value={"start_time": self.start.isoformat()}):
            self.assertEqual(rtsp_live.ChunkCatalog(self.root).paths(), [new])
        self.assertTrue(old.exists())
        fresh = products.ScheduleProducts(self.root, {"timelapse": True}, self.service.clock,
            threading.Event(), status=lambda _message: None)
        job = fresh._job(self.start + timedelta(minutes=1))
        new.unlink()
        self.assertTrue(fresh._drained(job), "deferred chunks blocked the fresh job")

    def test_deferred_spool_and_analysis_stay_saved_and_new_work_is_queued(self):
        spool = self.root / ".noise_twin_spool"
        spool.mkdir()
        old_raw = spool / "old.mp4"
        old_raw.write_bytes(b"x" * 11000)
        old_video = self.root / "old.mp4"
        old_video.touch()
        old_marker = Path(str(old_video) + ".preprocessed.json")
        old_marker.write_text(json.dumps({"method": "temporal_mean", "analyzed": False}))
        self.assertEqual(set(rtsp_startup.pending_recordings(self.root)), {old_video, old_raw})
        rtsp_startup.defer_previous(self.root, [self.job], self.start)
        new_raw = spool / "new.mp4"
        new_raw.write_bytes(b"x" * 11000)
        new_video = self.root / "new.mp4"
        new_video.touch()
        Path(str(new_video) + ".preprocessed.json").write_text(old_marker.read_text())
        pipeline = noise_twin_pipeline.RtspNoiseTwinPipeline("rtsp://camera", str(self.root), "",
            lambda *_args: True, threading.Event(), temporal_mean_frames=3, segment_seconds=5)
        with mock.patch.object(noise_twin_pipeline.time, "time", return_value=10**12):
            self.assertEqual(pipeline._completed_raw_segments(), [new_raw])
        pipeline._restore_analysis()
        self.assertEqual(pipeline.analysis_queue.get_nowait(), (str(new_video), ""))
        self.assertTrue(pipeline.analysis_queue.empty())
        self.assertTrue(old_raw.exists())
        self.assertTrue(old_marker.exists())


if __name__ == "__main__":
    unittest.main()
