import os
import tempfile
import unittest

import cv2
import numpy as np

import temporal_mean


class TemporalMeanTests(unittest.TestCase):
    def collect(self, window, values):
        processor = temporal_mean.TemporalMeanStream(window)
        outputs = []
        for value in values:
            outputs.extend(processor.push(np.full((4, 4, 3), value, np.uint8)))
        outputs.extend(processor.flush())
        return [int(frame[0, 0, 0]) for frame in outputs]

    def test_three_frame_centered_mean_preserves_count(self):
        self.assertEqual(self.collect(3, [0, 3, 6, 9]), [1, 3, 6, 8])

    def test_five_frame_short_clip_uses_replicated_edges(self):
        self.assertEqual(self.collect(5, [0, 5, 10]), [3, 5, 7])

    def test_prepare_video_preserves_every_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "source.mp4")
            writer = cv2.VideoWriter(
                source, cv2.VideoWriter_fourcc(*"mp4v"), 25, (32, 24)
            )
            for value in range(9):
                writer.write(np.full((24, 32, 3), value * 10, np.uint8))
            writer.release()
            prepared = temporal_mean.prepare_video(source, 3, temp_dir=directory)
            cap = cv2.VideoCapture(prepared.video_path)
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            self.assertEqual(count, 9)

    def test_processing_marker_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            video = os.path.join(directory, "segment.mp4")
            temporal_mean.write_processing_marker(video, 5, analyzed=True)
            marker = temporal_mean.load_processing_marker(video)
            self.assertEqual(marker["frames"], 5)
            self.assertTrue(marker["analyzed"])


if __name__ == "__main__":
    unittest.main()
