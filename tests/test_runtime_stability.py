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
import tkinter_safety


class RuntimeStabilityTests(unittest.TestCase):
    def test_tk_variable_destructor_ignores_dead_interpreter_boolean(self):
        class DeadTk:
            def call(self, *_args):
                return None

            def getboolean(self, value):
                raise tkinter_safety.tk.TclError(
                    f"expected boolean value but got {value!r}"
                )

        variable = object.__new__(tkinter_safety.tk.Variable)
        variable._tk = DeadTk()
        variable._name = "PY_VAR_dead"
        variable._tclCommands = None

        tkinter_safety._safe_variable_delete(variable)

        self.assertIsNone(variable._tk)

    def test_tk_variable_destructor_never_calls_tcl_from_worker_thread(self):
        calls = []

        class ThreadBoundTk:
            def call(self, *_args):
                calls.append(threading.current_thread())
                raise AssertionError("worker thread called Tcl")

            def getboolean(self, _value):
                raise AssertionError("worker thread called Tcl")

        variable = object.__new__(tkinter_safety.tk.Variable)
        variable._tk = ThreadBoundTk()
        variable._name = "PY_VAR_worker"
        variable._tclCommands = None
        worker = threading.Thread(
            target=tkinter_safety._safe_variable_delete, args=(variable,)
        )

        worker.start()
        worker.join(timeout=2)

        self.assertFalse(worker.is_alive())
        self.assertEqual(calls, [])
        self.assertIsNone(variable._tk)

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

    def test_frame_range_reader_uses_a_sidecar_capture(self):
        frame_a = np.zeros((4, 4, 3), dtype=np.uint8)
        frame_b = np.full((4, 4, 3), 7, dtype=np.uint8)
        capture = mock.Mock()
        capture.isOpened.return_value = True
        capture.get.side_effect = [25.0, 11.0, 12.0]
        capture.read.side_effect = [(True, frame_a), (True, frame_b)]

        with mock.patch.object(video_processing, "_open_video_capture", return_value=capture):
            frames = video_processing._read_video_frame_range("sample.mp4", 10, 11)

        self.assertEqual(len(frames), 2)
        capture.set.assert_called_once_with(video_processing.cv2.CAP_PROP_POS_FRAMES, 10)
        capture.release.assert_called_once()

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
