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


if __name__ == "__main__":
    unittest.main()
