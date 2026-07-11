import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import torch
import numpy as np
from PIL import Image

import model
import image_processing
import status_panel
import video_processing


class RuntimeStabilityTests(unittest.TestCase):
    def test_streaming_temporal_extrema_match_stacked_result(self):
        rng = np.random.default_rng(7)
        frames = [
            rng.integers(0, 256, (12, 16, 3), dtype=np.uint8)
            for _ in range(9)
        ]

        maximum, minimum = image_processing._temporal_min_max(frames)

        stacked = np.stack(frames)
        np.testing.assert_array_equal(maximum, stacked.max(axis=0))
        np.testing.assert_array_equal(minimum, stacked.min(axis=0))

    def test_max_composite_allocates_only_requested_crop(self):
        frames = [np.full((20, 30, 3), value, dtype=np.uint8) for value in (4, 90, 12)]
        result = video_processing._max_composite_cutout(frames, (5, 7, 14, 16))

        self.assertEqual(result.shape, (9, 9, 3))
        self.assertTrue(np.all(result == 90))

    def test_capture_limits_ffmpeg_threads_at_open(self):
        capture = mock.Mock()
        capture.isOpened.return_value = True
        with mock.patch.object(video_processing.cv2, "VideoCapture", return_value=capture) as ctor:
            result = video_processing._open_video_capture("sample.mp4", decoder_threads=2)

        self.assertIs(result, capture)
        args = ctor.call_args.args
        self.assertEqual(args[0], "sample.mp4")
        self.assertEqual(args[1], video_processing.cv2.CAP_FFMPEG)
        self.assertEqual(
            args[2],
            [video_processing.cv2.CAP_PROP_N_THREADS, 2],
        )

    def test_status_callback_uses_queue_without_calling_tk(self):
        panel = object.__new__(status_panel.StatusPanel)
        panel.progress_queue = queue.Queue()
        panel.app = mock.Mock()

        panel.get_status_callback()({"download_queue_size": 3})

        self.assertEqual(
            panel.progress_queue.get_nowait(),
            (None, {"pipeline_status": {"download_queue_size": 3}}),
        )
        panel.app.after.assert_not_called()

    def test_tta_is_one_batch_and_concurrent_calls_are_serialized(self):
        active = 0
        max_active = 0
        batch_sizes = []
        state_lock = threading.Lock()

        class SlowFakeModel:
            def __call__(self, batch):
                nonlocal active, max_active
                with state_lock:
                    active += 1
                    max_active = max(max_active, active)
                    batch_sizes.append(int(batch.shape[0]))
                time.sleep(0.02)
                with state_lock:
                    active -= 1
                return torch.tensor([[2.0, 0.0]]).repeat(batch.shape[0], 1)

        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "diff.png"
            Image.new("RGB", (16, 16), "black").save(image_path)
            fake_transform = lambda _image: torch.zeros((3, 8, 8))
            with mock.patch.multiple(
                model,
                model=SlowFakeModel(),
                transform=fake_transform,
                _model_ready=True,
                device=torch.device("cpu"),
                _active_model_info={"meteor_class_index": 0},
            ):
                workers = [
                    threading.Thread(
                        target=model.predict_meteor_probability,
                        args=(str(image_path),),
                    )
                    for _ in range(6)
                ]
                for worker in workers:
                    worker.start()
                for worker in workers:
                    worker.join(timeout=3.0)

        self.assertTrue(all(not worker.is_alive() for worker in workers))
        self.assertEqual(max_active, 1)
        self.assertEqual(batch_sizes, [3] * 6)


if __name__ == "__main__":
    unittest.main()
