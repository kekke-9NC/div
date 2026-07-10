import unittest
from unittest import mock

import numpy as np

import file_utils
import utils


class NotificationSoundTests(unittest.TestCase):
    def test_macos_uses_system_sound_without_blocking(self):
        with (
            mock.patch.object(utils.platform, "system", return_value="Darwin"),
            mock.patch.object(utils.os.path, "isfile", return_value=True),
            mock.patch.object(utils.subprocess, "Popen") as popen,
        ):
            result = utils.play_notification_sound()

        self.assertTrue(result)
        popen.assert_called_once_with(
            ["/usr/bin/afplay", "/System/Library/Sounds/Glass.aiff"],
            stdout=utils.subprocess.DEVNULL,
            stderr=utils.subprocess.DEVNULL,
        )

    def test_macos_playback_failure_does_not_escape(self):
        with (
            mock.patch.object(utils.platform, "system", return_value="Darwin"),
            mock.patch.object(utils.os.path, "isfile", return_value=True),
            mock.patch.object(utils.subprocess, "Popen", side_effect=OSError("afplay failed")),
        ):
            result = utils.play_notification_sound()

        self.assertFalse(result)

    def test_rtsp_notification_setting_reaches_detection_pipeline(self):
        correction = np.ones((4, 4), dtype=np.int16)
        with mock.patch.object(
            file_utils.video_processing, "create_line_video_clips", return_value=[]
        ) as create_clips:
            result = file_utils.process_video_file_periodic(
                "recorded-rtsp.mp4",
                notify_on_detection=False,
                fixed_pattern_correction=correction,
            )

        self.assertTrue(result)
        self.assertFalse(create_clips.call_args.kwargs["notify_on_detection"])
        self.assertIs(create_clips.call_args.kwargs["fixed_pattern_correction"], correction)


if __name__ == "__main__":
    unittest.main()
