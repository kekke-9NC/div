import unittest
from unittest import mock
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from types import SimpleNamespace
from pathlib import Path
import json
import os

import timelapse_creator
from astropy.io import fits
from astropy.wcs import WCS


class TimelapseCreatorTests(unittest.TestCase):
    def test_ffprobe_rejects_video_with_h264_packet_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory, "broken.mp4")
            video.write_bytes(b"broken")
            probe = SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"streams": [{"nb_frames": "1500"}]}),
                stderr="[h264] Invalid NAL unit size",
            )
            with mock.patch.object(timelapse_creator.subprocess, "run", return_value=probe):
                count = timelapse_creator.get_video_frame_count(str(video))
        self.assertEqual(count, 0)
        self.assertIn(
            "Invalid NAL",
            timelapse_creator._video_probe_errors[os.path.abspath(str(video))],
        )

    def test_ffprobe_frame_count_is_used_for_valid_video(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory, "valid.mp4")
            video.write_bytes(b"video")
            probe = SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"streams": [{"nb_frames": "1500"}]}),
                stderr="",
            )
            with mock.patch.object(timelapse_creator.subprocess, "run", return_value=probe):
                count = timelapse_creator.get_video_frame_count(str(video))
        self.assertEqual(count, 1500)

    def test_folder_scan_excludes_recorder_temp_video(self):
        with tempfile.TemporaryDirectory() as directory:
            complete = Path(directory, "09.mp4")
            unfinished = Path(directory, "09_temp_1783620548963395000.mp4")
            complete.touch()
            unfinished.touch()

            _images, videos = timelapse_creator.get_files_from_path(directory)

        self.assertEqual(videos, [str(complete)])

    def test_explicit_recorder_temp_video_is_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            unfinished = Path(directory, "09_temp_1783620548963395000.mp4")
            unfinished.touch()

            _images, videos = timelapse_creator.get_files_from_path(str(unfinished))

        self.assertEqual(videos, [])

    def test_progress_event_remains_string_compatible(self):
        event = timelapse_creator.TimelapseProgress("working", 0.25, 12.5)

        self.assertEqual(event, "working")
        self.assertEqual(event.fraction, 0.25)
        self.assertEqual(event.eta_seconds, 12.5)

    def test_fast_filter_without_temporal_mean_samples_all_frames(self):
        indices = timelapse_creator.calculate_sample_indices(9000, 15)

        graph = timelapse_creator._build_fast_filter_graph(
            9000, indices, 0, (1920, 1080), None, None
        )

        self.assertIn("selected_n*9000/900", graph)
        self.assertNotIn("tmix=", graph)
        self.assertIn("[out]", graph)

    def test_fast_filter_uses_small_reverse_tail_for_centered_mean(self):
        indices = timelapse_creator.calculate_sample_indices(9000, 15)

        graph = timelapse_creator._build_fast_filter_graph(
            9000, indices, 50, (1920, 1080), None, None
        )

        self.assertIn("tmix=frames=101", graph)
        self.assertIn("trim=start_frame=8900", graph)
        self.assertIn("lt(selected_n\\,895)", graph)
        self.assertIn("eq(n\\,59)", graph)
        self.assertIn("eq(n\\,99)", graph)

    def test_fast_filter_does_not_require_an_unused_reverse_tail(self):
        indices = timelapse_creator.calculate_sample_indices(90000, 15)

        graph = timelapse_creator._build_fast_filter_graph(
            90000, indices, 50, (1920, 1080), None, None
        )

        self.assertIsNotNone(graph)
        self.assertIn("tmix=frames=101", graph)
        self.assertIn("lt(selected_n\\,900)", graph)
        self.assertNotIn("split=2", graph)
        self.assertNotIn("reverse", graph)

    def test_fast_filter_rejects_too_short_temporal_window(self):
        indices = timelapse_creator.calculate_sample_indices(80, 15)

        graph = timelapse_creator._build_fast_filter_graph(
            80, indices, 50, (1920, 1080), None, None
        )

        self.assertIsNone(graph)

    def test_runtime_fast_failure_does_not_silently_use_slow_path(self):
        loader = mock.MagicMock()
        loader.load_frame.return_value = timelapse_creator.np.zeros(
            (16, 16, 3), dtype=timelapse_creator.np.uint8
        )
        messages = []

        with (
            mock.patch.object(
                timelapse_creator, "get_files_from_path",
                return_value=([], ["input.mp4"]),
            ),
            mock.patch.object(
                timelapse_creator, "count_total_frames",
                return_value=(1000, [("input.mp4", 0, 1000)]),
            ),
            mock.patch.object(timelapse_creator, "FrameLoader", return_value=loader),
            mock.patch.object(
                timelapse_creator, "_create_video_timelapse_fast", return_value=False
            ),
            mock.patch.object(timelapse_creator, "TemporalMeanFrameCache") as slow_cache,
        ):
            result = timelapse_creator.create_timelapse(
                ["input.mp4"], "output.mp4", progress_callback=messages.append
            )

        self.assertFalse(result)
        slow_cache.assert_not_called()
        loader.cleanup.assert_called_once()
        self.assertTrue(any("自動切り替えしません" in str(item) for item in messages))

    def test_invalid_probed_video_is_not_sent_to_fast_concat(self):
        loader = mock.MagicMock()
        loader.load_frame.return_value = timelapse_creator.np.zeros(
            (16, 16, 3), dtype=timelapse_creator.np.uint8
        )

        with (
            mock.patch.object(
                timelapse_creator, "get_files_from_path",
                return_value=([], ["complete.mp4", "broken.mp4"]),
            ),
            mock.patch.object(
                timelapse_creator, "count_total_frames",
                return_value=(1000, [("complete.mp4", 0, 1000)]),
            ),
            mock.patch.object(timelapse_creator, "FrameLoader", return_value=loader),
            mock.patch.object(
                timelapse_creator, "_create_video_timelapse_fast", return_value=True
            ) as fast_path,
        ):
            result = timelapse_creator.create_timelapse(
                ["folder"], "output.mp4", progress_callback=lambda _message: None
            )

        self.assertTrue(result)
        self.assertEqual(fast_path.call_args.args[0], ["complete.mp4"])

    def test_local_annotation_loader_accepts_fallback_name(self):
        expected = mock.Mock()
        module = SimpleNamespace(annotate_frame_local=expected)

        with mock.patch.object(
            timelapse_creator.importlib, "import_module", return_value=module
        ) as import_module:
            result = timelapse_creator._load_local_annotation_callable()

        self.assertIs(result, expected)
        import_module.assert_called_once_with("local_wideangle_astrometry")

    def test_local_annotation_loader_has_clear_missing_module_error(self):
        with mock.patch.object(
            timelapse_creator.importlib,
            "import_module",
            side_effect=ModuleNotFoundError("missing"),
        ):
            with self.assertRaisesRegex(RuntimeError, "local_wideangle_astrometry"):
                timelapse_creator._load_local_annotation_callable()

    def test_local_annotation_accepts_tuple_and_preserves_encoder_shape(self):
        source = timelapse_creator.np.zeros((12, 20, 3), dtype=timelapse_creator.np.uint8)
        annotated = timelapse_creator.np.full((6, 10), 123, dtype=timelapse_creator.np.uint8)
        annotator = mock.Mock(return_value=(annotated, {"calibrated": True}))
        frame_time = datetime(2026, 7, 10, 1, 2, 3)

        result = timelapse_creator._apply_local_annotation(
            annotator,
            source,
            frame_time,
            {"calibration_path": "/tmp/night-calibration.json"},
        )

        self.assertEqual(result.shape, source.shape)
        self.assertEqual(result.dtype, timelapse_creator.np.uint8)
        annotator.assert_called_once_with(
            source,
            frame_time,
            calibration_path="/tmp/night-calibration.json",
        )

    def test_local_annotation_passes_independent_overlay_options(self):
        source = timelapse_creator.np.zeros((12, 20, 3), dtype=timelapse_creator.np.uint8)
        annotator = mock.Mock(return_value=source.copy())
        frame_time = datetime(2026, 7, 10, 1, 2, 3)

        timelapse_creator._apply_local_annotation(
            annotator, source, frame_time,
            {
                "calibration_path": "/tmp/calibration.json",
                "draw_grid": False,
                "draw_constellations": True,
                "draw_detected_stars": True,
            },
        )

        annotator.assert_called_once_with(
            source, frame_time,
            calibration_path="/tmp/calibration.json",
            draw_grid=False,
            draw_constellations=True,
            draw_detected_stars=True,
        )

    def test_annotation_settings_default_to_grid_without_other_overlays(self):
        settings = timelapse_creator._normalize_annotation_settings({"enabled": True})
        self.assertTrue(settings["draw_grid"])
        self.assertFalse(settings["draw_constellations"])
        self.assertFalse(settings["draw_detected_stars"])

    def test_annotation_enabled_bypasses_fast_ffmpeg_and_annotates_each_frame(self):
        frame = timelapse_creator.np.zeros((16, 16, 3), dtype=timelapse_creator.np.uint8)
        frame_time = datetime(2026, 7, 10, 1, 0, 0)
        loader = mock.MagicMock()
        loader.load_frame.return_value = frame.copy()
        loader.timestamp_for_index.return_value = frame_time

        mean_cache = mock.MagicMock()
        mean_cache.full_preload_enabled = False
        mean_cache.enabled = True
        mean_cache._retain_all_frames = False
        mean_cache.mean_for_index.side_effect = [frame.copy(), frame.copy()]

        stdin = mock.MagicMock()
        stdin.closed = False
        proc = mock.MagicMock(stdin=stdin)
        proc.poll.return_value = 0
        annotator = mock.Mock(side_effect=lambda image, _time, calibration_path=None: image + 1)

        with (
            mock.patch.object(
                timelapse_creator, "get_files_from_path", return_value=([], ["input.mp4"])
            ),
            mock.patch.object(
                timelapse_creator,
                "count_total_frames",
                return_value=(2, [("input.mp4", 0, 2)]),
            ),
            mock.patch.object(
                timelapse_creator, "calculate_sample_indices", return_value=[0, 1]
            ),
            mock.patch.object(timelapse_creator, "FrameLoader", return_value=loader),
            mock.patch.object(
                timelapse_creator, "TemporalMeanFrameCache", return_value=mean_cache
            ),
            mock.patch.object(
                timelapse_creator, "_load_local_annotation_callable", return_value=annotator
            ),
            mock.patch.object(
                timelapse_creator, "_prepare_local_annotation_calibration"
            ) as prepare_calibration,
            mock.patch.object(
                timelapse_creator, "_create_video_timelapse_fast"
            ) as fast_path,
            mock.patch.object(timelapse_creator, "_run_annotate_pipeline") as annotation_pipeline,
            mock.patch.object(
                timelapse_creator, "_select_h264_encoder", return_value=("test", [], "test")
            ),
            mock.patch.object(
                timelapse_creator.subprocess, "Popen", return_value=proc
            ) as ffmpeg_popen,
            mock.patch.object(
                timelapse_creator, "_finish_ffmpeg_process", return_value=(0, b"")
            ),
        ):
            result = timelapse_creator.create_timelapse(
                ["input.mp4"],
                "output.mp4",
                timestamp_settings={"enabled": False},
                temporal_mean_radius_frames=0,
                annotation_settings={"enabled": True},
            )

        self.assertTrue(result)
        fast_path.assert_not_called()
        prepare_calibration.assert_called_once()
        self.assertEqual(annotator.call_count, 1)
        annotation_pipeline.assert_called_once()
        self.assertEqual(stdin.write.call_count, 1)
        ffmpeg_command = ffmpeg_popen.call_args.args[0]
        self.assertIn("-nostats", ffmpeg_command)
        self.assertIsNot(
            ffmpeg_popen.call_args.kwargs["stderr"],
            timelapse_creator.subprocess.PIPE,
        )

    def test_annotate_pipeline_writes_completed_frames_in_sample_order(self):
        indices = [10, 20, 30, 40, 50]
        frames = [
            timelapse_creator.np.full((4, 4, 3), value, dtype=timelapse_creator.np.uint8)
            for value in range(1, len(indices) + 1)
        ]
        cache = mock.MagicMock()
        cache.mean_for_index.side_effect = frames
        loader = mock.MagicMock()
        loader.timestamp_for_index.side_effect = [datetime(2026, 7, 10, 1, 0, i) for i in range(5)]
        stdin = mock.MagicMock()

        def annotate(frame, _timestamp):
            # Ensure later jobs finish first; output must nevertheless remain ordered.
            if int(frame[0, 0, 0]) == 1:
                time.sleep(0.03)
            return frame + 10

        class ThreadedPool:
            def __init__(self, max_workers, initializer=None, initargs=()):
                self.executor = ThreadPoolExecutor(max_workers=max_workers)
                self.initializer = initializer
                self.initargs = initargs

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.executor.shutdown(wait=True)

            def submit(self, function, *args):
                return self.executor.submit(function, *args)

        with (
            mock.patch.object(timelapse_creator, "ProcessPoolExecutor", ThreadedPool),
            mock.patch.object(timelapse_creator, "_annotation_worker", side_effect=annotate),
        ):
            timelapse_creator._run_annotate_pipeline(
                indices,
                total_output=6,
                temporal_mean_cache=cache,
                loader=loader,
                annotation_settings={"calibration_path": "/tmp/calibration.json"},
                resized_mask=None,
                timestamp_settings={"enabled": False},
                ffmpeg_stdin=stdin,
                progress_callback=None,
            )

        self.assertEqual(cache.mean_for_index.call_args_list, [mock.call(index) for index in indices])
        self.assertEqual(
            [timelapse_creator.np.frombuffer(call.args[0], dtype=timelapse_creator.np.uint8)[0]
             for call in stdin.write.call_args_list],
            [11, 12, 13, 14, 15],
        )

    def test_annotate_pipeline_uses_real_process_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wcs = WCS(naxis=2)
            wcs.wcs.crpix = [8, 8]
            wcs.wcs.cdelt = [-0.1, 0.1]
            wcs.wcs.crval = [230.0, 52.0]
            wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
            wcs_path = root / "wideangle.wcs"
            header = wcs.to_header(relax=True)
            header["IMAGEW"] = 16
            header["IMAGEH"] = 16
            header["DATE-OBS"] = "2026-07-10T01:00:00"
            fits.PrimaryHDU(header=header).writeto(wcs_path)
            calibration = root / "calibration.json"
            calibration.write_text(json.dumps({
                "wcs_path": str(wcs_path), "reference_datetime": "2026-07-10T01:00:00",
                "width": 16, "height": 16, "center_ra_deg": 230.0,
            }), encoding="utf-8")
            cache = mock.MagicMock()
            cache.mean_for_index.side_effect = [
                timelapse_creator.np.zeros((16, 16, 3), dtype=timelapse_creator.np.uint8),
                timelapse_creator.np.zeros((16, 16, 3), dtype=timelapse_creator.np.uint8),
            ]
            loader = mock.MagicMock()
            loader.timestamp_for_index.side_effect = [
                datetime(2026, 7, 10, 1, 0, 0), datetime(2026, 7, 10, 1, 0, 1),
            ]
            stdin = mock.MagicMock()
            timelapse_creator._run_annotate_pipeline(
                [0, 1], 3, cache, loader,
                {"calibration_path": str(calibration), "draw_constellations": False},
                None, {"enabled": False}, stdin, None,
            )
        self.assertEqual(stdin.write.call_count, 2)

    def test_selected_sample_calibration_uses_centered_temporal_mean(self):
        frame = timelapse_creator.np.zeros((12, 20, 3), dtype=timelapse_creator.np.uint8)
        loader = mock.MagicMock()
        loader._get_source_for_index.return_value = ("input.mp4", 0, 100)
        loader.load_temporal_mean_frame.return_value = frame
        timestamp = datetime(2026, 7, 10, 1, 2, 3)
        loader.timestamp_for_index.return_value = timestamp
        solve = mock.Mock(return_value={"calibration_path": "/tmp/selected.json"})
        module = SimpleNamespace(solve_reference_frame_local=solve)
        settings = {
            "calibration_path": None,
            "reference_selected": True,
            "reference_sample_index": 42,
        }

        with mock.patch.object(timelapse_creator.importlib, "import_module", return_value=module):
            timelapse_creator._prepare_local_annotation_calibration(
                ["input.mp4"], settings, None,
                loader=loader, target_size=(20, 12), temporal_mean_radius=50,
            )

        loader.load_temporal_mean_frame.assert_called_once_with(42, (20, 12), radius=50)
        solve.assert_called_once_with(
            frame,
            source_identity="input.mp4",
            observation_datetime=timestamp,
            reference_frame_index=42,
            progress_callback=mock.ANY,
        )
        self.assertEqual(settings["calibration_path"], "/tmp/selected.json")


if __name__ == "__main__":
    unittest.main()
