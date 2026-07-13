import os
import unittest
from datetime import datetime
from unittest import mock

import media_time
import timelapse_creator
import video_processor


class MediaTimeTests(unittest.TestCase):
    def test_filesystem_creation_time_has_priority(self):
        expected = datetime(2026, 7, 10, 23, 0, 39)
        with mock.patch.object(media_time, "_filesystem_creation_time", return_value=expected), \
                mock.patch.object(media_time, "_embedded_creation_time") as embedded:
            timestamp, source = media_time.get_media_start_time("/tmp/20260710/23/00.mp4")
        self.assertEqual(expected, timestamp)
        self.assertEqual("ファイル作成時刻", source)
        embedded.assert_not_called()

    def test_path_is_used_when_creation_metadata_is_unavailable(self):
        with mock.patch.object(media_time, "_filesystem_creation_time", return_value=None), \
                mock.patch.object(media_time, "_embedded_creation_time", return_value=None):
            timestamp, source = media_time.get_media_start_time("/archive/20260710/23/04.mp4")
        self.assertEqual(datetime(2026, 7, 10, 23, 4), timestamp)
        self.assertEqual("ファイル階層・名前", source)

    def test_schedule_applies_manual_offset_to_each_source(self):
        starts = {
            "a.mp4": datetime(2026, 7, 10, 23, 0, 39),
            "b.mp4": datetime(2026, 7, 10, 23, 1, 39),
        }
        with mock.patch.object(video_processor, "get_video_duration", return_value=60.0), \
                mock.patch.object(
                    video_processor.media_time,
                    "get_media_start_time",
                    side_effect=lambda path: (starts[path], "ファイル作成時刻"),
                ):
            schedule = video_processor._build_concat_schedule(["a.mp4", "b.mp4"], 30.0)
        self.assertEqual(datetime(2026, 7, 10, 23, 1, 9), schedule[0]["source_start_time"])
        self.assertEqual(datetime(2026, 7, 10, 23, 2, 9), schedule[1]["source_start_time"])
        self.assertEqual(60.0, schedule[1]["start"])

    def test_timelapse_default_name_uses_first_media_start(self):
        expected = datetime(2026, 7, 10, 23, 0, 39)
        with mock.patch.object(timelapse_creator, "get_files_from_path", return_value=([], ["a.mp4"])), \
                mock.patch.object(
                    timelapse_creator.media_time,
                    "first_media_start_time",
                    return_value=(expected, "ファイル作成時刻", "a.mp4"),
                ):
            output = timelapse_creator.get_default_output_path(["input"])
        self.assertEqual("20260710230039.mp4", os.path.basename(output))


if __name__ == "__main__":
    unittest.main()
