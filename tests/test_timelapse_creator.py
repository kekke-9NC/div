import unittest

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


if __name__ == "__main__":
    unittest.main()
