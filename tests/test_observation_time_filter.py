import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import observation_time_filter


class ObservationTimeFilterTests(unittest.TestCase):
    def test_only_date_root_activates_filter(self):
        self.assertIsNotNone(observation_time_filter.date_from_root(Path("/archive/20221026")))
        self.assertIsNone(observation_time_filter.date_from_root(Path("/archive/20221026/21")))

    def test_daylight_videos_are_removed_but_unknown_names_are_kept(self):
        root = Path("/archive/20221026")
        videos = [
            root / "03" / "30.mp4",
            root / "12" / "00.mp4",
            root / "21" / "15.mp4",
            root / "misc" / "unknown.mp4",
        ]
        dawn = datetime(2022, 10, 26, 4, 50)
        dusk = datetime(2022, 10, 26, 18, 37)
        with mock.patch.object(
            observation_time_filter, "astronomical_window", return_value=(dawn, dusk)
        ):
            selected, info = observation_time_filter.filter_date_root_videos(
                root, videos, 35.0, 135.0
            )
        self.assertIn(videos[0], selected)
        self.assertNotIn(videos[1], selected)
        self.assertIn(videos[2], selected)
        self.assertIn(videos[3], selected)
        self.assertTrue(info["applied"])
        self.assertEqual(info["unknown_count"], 1)

    def test_legacy_next_day_dawn_is_normalized_to_archive_date(self):
        with mock.patch.object(observation_time_filter.sun_times, "get_sun_times") as get_times:
            get_times.return_value = {
                "astro_dawn": datetime(2022, 10, 27, 4, 50),
                "astro_dusk": datetime(2022, 10, 26, 18, 37),
            }
            dawn, dusk = observation_time_filter.astronomical_window(
                datetime(2022, 10, 26).date(), 35.0, 135.0
            )
        self.assertEqual(dawn, datetime(2022, 10, 26, 4, 50))
        self.assertEqual(dusk, datetime(2022, 10, 26, 18, 37))
