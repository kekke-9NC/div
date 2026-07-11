import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import folder_source_discovery


class FolderSourceDiscoveryTests(unittest.TestCase):
    def test_discovers_multiple_roots_without_opening_video_files(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            roots = []
            for day in ("20221022", "20221023", "20221024"):
                root = base / day
                roots.append(root)
                for hour in ("03", "12", "21"):
                    folder = root / hour
                    folder.mkdir(parents=True)
                    for minute in range(5):
                        (folder / f"{minute:02d}.mp4").touch()
            messages = []
            with mock.patch.object(
                folder_source_discovery.observation_time_filter,
                "filter_date_root_videos",
                side_effect=lambda _root, videos, _lat, _lon: (videos, {"applied": False}),
            ):
                sources = folder_source_discovery.discover_sources(
                    [str(root) for root in roots], (".mp4",),
                    twilight_filter_enabled=True, latitude=35.0, longitude=135.0,
                    progress_callback=messages.append,
                )
            self.assertEqual(len(sources), 45)
            self.assertEqual(len(messages), 3)
            self.assertTrue(all(source["is_rtsp"] is False for source in sources))

    def test_duplicate_files_are_returned_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.mp4"
            video.touch()
            sources = folder_source_discovery.discover_sources(
                [str(root), str(video)], (".mp4",), twilight_filter_enabled=False,
                latitude=35.0, longitude=135.0,
            )
            self.assertEqual(len(sources), 1)

    def test_cancelled_scan_stops_between_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = []
            for index in range(3):
                root = Path(directory) / str(index)
                root.mkdir()
                (root / "clip.mp4").touch()
                roots.append(str(root))
            cancel = threading.Event()

            def progress(_message):
                cancel.set()

            sources = folder_source_discovery.discover_sources(
                roots, (".mp4",), twilight_filter_enabled=False,
                latitude=35.0, longitude=135.0, progress_callback=progress,
                cancel_flag=cancel,
            )
            self.assertEqual(sources, [])
