import unittest
import threading
from unittest import mock

import download_pipeline


class ProgressLimiterTests(unittest.TestCase):
    def test_start_and_zero_detection_completion_are_visible(self):
        messages = []
        limiter = download_pipeline._ProgressLimiter(messages.append, 10)

        limiter.message(('"video.mp4" の処理を開始...', None))
        limiter.message(("完了: video.mp4 検出: 0件", None))

        self.assertEqual(len(messages), 2)

    def test_heartbeat_is_rate_limited(self):
        messages = []
        limiter = download_pipeline._ProgressLimiter(messages.append, 100)

        with mock.patch.object(
            download_pipeline.time, "monotonic", side_effect=[10.0, 11.0, 21.0]
        ):
            limiter.heartbeat(0, 4, 8, 88)
            limiter.heartbeat(0, 4, 8, 88)
            limiter.heartbeat(4, 4, 8, 84)

        self.assertEqual(len(messages), 2)
        self.assertIn("完了 4/100", messages[-1][0])


class DownloadPipelineScalingTests(unittest.TestCase):
    def test_worker_plan_reserves_two_ui_cores(self):
        plan = download_pipeline.compute_worker_plan(20, logical_cpus=18)
        self.assertEqual(plan["reserved_for_ui"], 2)
        self.assertLessEqual(plan["video_workers"], 6)
        self.assertEqual(plan["finer_workers"], 1)
        self.assertEqual(plan["decoder_threads"], 2)
        self.assertLessEqual(
            plan["video_workers"] * plan["opencv_threads"] + plan["reserved_for_ui"],
            18,
        )

    def test_single_video_can_use_multiple_finer_workers(self):
        plan = download_pipeline.compute_worker_plan(1, logical_cpus=18)
        self.assertEqual(plan["video_workers"], 1)
        self.assertEqual(plan["finer_workers"], 2)
        self.assertEqual(plan["decoder_threads"], 4)

    def test_progress_limiter_suppresses_per_file_noise(self):
        received = []
        limiter = download_pipeline._ProgressLimiter(received.append, total=10_000)
        for index in range(1, 10_001):
            limiter.message((f'"clip{index}.mp4" の処理を開始...', None))
            limiter.message((f"ダウンロード完了: clip{index}.mp4", None))
            limiter.message((f"完了: clip{index}.mp4 検出: 0件", None))
            limiter.progress(index)
        progress_events = [item for item in received if item[0] is None]
        self.assertLessEqual(len(progress_events), 2)
        self.assertEqual(progress_events[-1][1], (10_000, 10_000))

    def test_detection_and_errors_are_not_suppressed(self):
        received = []
        limiter = download_pipeline._ProgressLimiter(received.append, total=2)
        limiter.message(("検出 1: meteor", None))
        limiter.message(("処理エラー (clip.mp4): broken", None))
        self.assertEqual(len(received), 2)

    def test_non_meteor_noise_is_summarized(self):
        received = []
        limiter = download_pipeline._ProgressLimiter(received.append, total=100)
        for index in range(100):
            limiter.message((f"検出 {index}: not_meteor (Prob: 0.01)", None))
            limiter.message((("  -> Not Meteor: Probability 0.01"), None))
            limiter.message((("  -> Video: clip.mp4"), None))
        limiter.finish()

        self.assertEqual(received, [("非流星候補: 100件（個別ログは省略）", None)])

    def test_large_fast_pipeline_processes_every_source(self):
        sources = [{"path": f"clip-{index}.mp4"} for index in range(1_000)]
        received = []
        processed = []
        with mock.patch.object(
            download_pipeline.network_copy, "ensure_local_copy",
            side_effect=lambda path, **_kwargs: (path, None),
        ), mock.patch.object(
            download_pipeline.video_processing, "create_line_video_clips",
            side_effect=lambda source, **_kwargs: processed.append(source),
        ), mock.patch.object(
            download_pipeline.network_copy, "cleanup_tempdir",
        ):
            download_pipeline.run_pipeline(
                sources=sources, max_workers=4, interval=1.0, duration=1.0,
                mask=None, global_wcs_info=None, plate_solve_mask=None,
                meteor_save_path="meteor", not_meteor_save_path="not_meteor",
                cancel_flag=threading.Event(), progress_callback=received.append,
            )
        self.assertEqual(len(processed), len(sources))
        self.assertEqual(len(set(processed)), len(sources))
        # One worker-plan message plus a heavily throttled progress stream.
        self.assertLess(len(received), 25)

    def test_cancel_drains_queued_items_without_hanging(self):
        sources = [{"path": f"clip-{index}.mp4"} for index in range(100)]
        cancel = threading.Event()
        started = threading.Event()

        def process_one(**_kwargs):
            started.set()
            cancel.set()

        with mock.patch.object(
            download_pipeline.network_copy, "ensure_local_copy",
            side_effect=lambda path, **_kwargs: (path, None),
        ), mock.patch.object(
            download_pipeline.video_processing, "create_line_video_clips",
            side_effect=process_one,
        ), mock.patch.object(
            download_pipeline.network_copy, "cleanup_tempdir",
        ):
            runner = threading.Thread(
                target=download_pipeline.run_pipeline,
                kwargs=dict(
                    sources=sources, max_workers=4, interval=1.0, duration=1.0,
                    mask=None, global_wcs_info=None, plate_solve_mask=None,
                    meteor_save_path="meteor", not_meteor_save_path="not_meteor",
                    cancel_flag=cancel, progress_callback=lambda _item: None,
                ),
            )
            runner.start()
            self.assertTrue(started.wait(timeout=2.0))
            runner.join(timeout=5.0)

        self.assertFalse(runner.is_alive(), "cancel left queue.join() blocked")
