import json
import os
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

import image_processing
import noise_twin


class _FakeEngine:
    def infer(self, frames):
        centre = frames[noise_twin.TEMPORAL_RADIUS]
        value = int(centre[0, 0, 0])
        evidence = np.full_like(centre, value)
        return noise_twin.NoiseTwinResult(
            frame=centre.copy(),
            innovation=evidence,
            noise_sigma=0.0,
            innovation_max=float(value),
            protected_fraction=0.0,
            flux_retention=1.0,
        )


class _FakeCapture:
    def __init__(self, frames, fps=2.0):
        self.frames = [frame.copy() for frame in frames]
        self.index = 0
        self.fps = fps

    def isOpened(self):
        return True

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame.copy()

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return len(self.frames)
        return 0


class NoiseTwinTests(unittest.TestCase):
    def test_stream_processor_preserves_frame_count_and_order(self):
        processor = noise_twin.NoiseTwinStreamProcessor(_FakeEngine())
        results = []
        for value in range(11):
            frame = np.full((8, 8, 3), value, dtype=np.uint8)
            results.extend(processor.push(frame))
        results.extend(processor.flush())

        self.assertEqual(len(results), 11)
        self.assertEqual([int(item.frame[0, 0, 0]) for item in results], list(range(11)))

    def test_stream_processor_handles_clip_shorter_than_window(self):
        processor = noise_twin.NoiseTwinStreamProcessor(_FakeEngine())
        results = []
        for value in (4, 9, 15):
            results.extend(processor.push(np.full((4, 4, 3), value, np.uint8)))
        results.extend(processor.flush())
        self.assertEqual([int(item.frame[0, 0, 0]) for item in results], [4, 9, 15])

    def test_metadata_round_trip_and_validation_default(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = os.path.join(directory, "camera.pth")
            metadata = noise_twin.NoiseTwinMetadata(
                model_id="abc",
                created_at="2026-07-16T00:00:00Z",
                width=1920,
                height=1080,
                fps=25.0,
                source_id="source",
                validation=noise_twin.NoiseTwinValidation(
                    injection_count=10_000,
                    missed_fraction=0.001,
                    flux_retention=0.97,
                    peak_retention=0.94,
                    trajectory_retention=0.99,
                    false_positive_reduction=0.35,
                    realtime_fps=27.0,
                    realtime_test_seconds=1800.0,
                    dropped_frames=0,
                    validated=True,
                ),
            )
            noise_twin.save_metadata(model_path, metadata)
            loaded = noise_twin.load_metadata(model_path)
            self.assertEqual(loaded, metadata)
            self.assertTrue(
                noise_twin.validate_model_for_video(model_path, 1920, 1080, 25, True)
            )

    def test_realtime_validation_rejects_slow_model(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = os.path.join(directory, "camera.pth")
            metadata = noise_twin.NoiseTwinMetadata(
                model_id="slow",
                created_at="now",
                width=1920,
                height=1080,
                fps=25.0,
                source_id="source",
                validation=noise_twin.NoiseTwinValidation(validated=True, realtime_fps=18.0),
            )
            noise_twin.save_metadata(model_path, metadata)
            with self.assertRaises(noise_twin.NoiseTwinError):
                noise_twin.validate_model_for_video(model_path, 1920, 1080, 25, True)

    def test_realtime_validation_requires_thirty_minute_zero_drop_run(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = os.path.join(directory, "camera.pth")
            metadata = noise_twin.NoiseTwinMetadata(
                model_id="not-soaked",
                created_at="now",
                width=1920,
                height=1080,
                fps=25.0,
                source_id="source",
                validation=noise_twin.NoiseTwinValidation(
                    validated=True,
                    realtime_fps=30.0,
                    realtime_test_seconds=1799.0,
                    dropped_frames=0,
                ),
            )
            noise_twin.save_metadata(model_path, metadata)
            with self.assertRaises(noise_twin.NoiseTwinError):
                noise_twin.validate_model_for_video(model_path, 1920, 1080, 25, True)

    def test_processed_video_marker_prevents_double_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            video_path = os.path.join(directory, "segment.mp4")
            metadata = noise_twin.NoiseTwinMetadata(
                model_id="camera-1",
                created_at="now",
                width=1920,
                height=1080,
                fps=25.0,
                source_id="source",
            )
            noise_twin.write_processing_marker(video_path, metadata)
            marker = noise_twin.load_processing_marker(video_path)
            self.assertEqual(marker["model_id"], "camera-1")
            self.assertTrue(marker["processed"])

    def test_innovation_sidecar_drives_coarse_diff(self):
        raw_frames = [np.full((24, 32, 3), 40, np.uint8) for _ in range(3)]
        evidence_frames = [np.zeros((24, 32, 3), np.uint8) for _ in range(3)]
        cv2.line(evidence_frames[1], (3, 12), (28, 12), (220, 220, 220), 2)
        generator = image_processing.create_diff_images(
            _FakeCapture(raw_frames),
            interval=1.0,
            duration=1.0,
            evidence_cap=_FakeCapture(evidence_frames),
        )
        outputs = list(generator)
        self.assertTrue(outputs)
        diff = outputs[0][0]
        self.assertGreater(int(diff.max()), 150)
        self.assertTrue(np.all(outputs[0][2] == 40))

    def test_correction_hash_does_not_expose_frame_data_shape_only(self):
        correction = np.arange(16, dtype=np.int16).reshape(4, 4)
        digest = noise_twin.correction_sha256(correction)
        self.assertEqual(len(digest), 64)
        self.assertNotEqual(digest, noise_twin.correction_sha256(correction + 1))

    def test_network_cannot_create_positive_residual(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        network = noise_twin.build_model().eval()
        parameter_count = sum(parameter.numel() for parameter in network.parameters())
        self.assertGreater(parameter_count, 7_000_000)
        self.assertLess(parameter_count, 8_500_000)
        neighbors = torch.rand(1, 18, 32, 32)
        centre = torch.rand(1, 3, 32, 32)
        with torch.inference_mode():
            clean, background, _gate, _innovation = network(neighbors, centre)
        source_positive = torch.clamp(centre - background, min=0.0)
        output_positive = torch.clamp(clean - background, min=0.0)
        self.assertTrue(torch.all(output_positive <= source_positive + 1e-6))


if __name__ == "__main__":
    unittest.main()
