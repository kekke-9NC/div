import threading
import unittest
from unittest import mock

import numpy as np

import file_utils


class RtspFixedPatternModeTests(unittest.TestCase):
    def test_noise_twin_and_temporal_mean_receive_selected_correction(self):
        correction = np.ones((4, 4), dtype=np.int16)
        modes = (
            ("NoiseTwin ON", {"enabled": True, "model_path": "model.pth"}),
            ("3-frame mean ON", {"temporal_mean_frames": 3}),
            ("5-frame mean ON", {"temporal_mean_frames": 5}),
        )

        for mode_name, options in modes:
            for enabled in (False, True):
                selected_correction = correction if enabled else None
                with (
                    self.subTest(mode=mode_name, fixed_pattern_enabled=enabled),
                    mock.patch.object(
                        file_utils.noise_twin_pipeline, "RtspNoiseTwinPipeline"
                    ) as pipeline_class,
                ):
                    file_utils.rtsp_save_and_process_thread_target(
                        "rtsp://camera/stream",
                        cancel_flag=threading.Event(),
                        dark_frame=selected_correction,
                        noise_twin_options=options,
                    )

                    self.assertIs(
                        pipeline_class.call_args.kwargs["correction"],
                        selected_correction,
                    )
                    pipeline_class.return_value.run.assert_called_once_with()

    def test_raw_rtsp_save_applies_temporal_mean_only_during_analysis(self):
        correction = np.ones((4, 4), dtype=np.int16)
        options = {
            "temporal_mean_frames": 3,
            "save_temporal_mean_video": False,
        }
        with (
            mock.patch.object(
                file_utils.noise_twin_pipeline, "RtspNoiseTwinPipeline"
            ) as pipeline_class,
            mock.patch.object(
                file_utils, "process_video_file_periodic", return_value=True
            ) as process_video,
        ):
            file_utils.rtsp_save_and_process_thread_target(
                "rtsp://camera/stream",
                cancel_flag=threading.Event(),
                dark_frame=correction,
                noise_twin_options=options,
            )

            pipeline_kwargs = pipeline_class.call_args.kwargs
            self.assertFalse(pipeline_kwargs["save_temporal_mean_video"])
            pipeline_kwargs["analyze_callback"]("saved-raw.mp4", "")

        analysis_args = process_video.call_args.args
        self.assertIs(analysis_args[-2], correction)
        self.assertFalse(analysis_args[-1]["already_processed"])
        self.assertEqual(analysis_args[-1]["temporal_mean_frames"], 3)


if __name__ == "__main__":
    unittest.main()
