import unittest
from unittest import mock

import timelapse_creator


class TimelapseCreatorTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
