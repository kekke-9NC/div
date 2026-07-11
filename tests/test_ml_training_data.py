import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import ml_training_data


class TrainingDataTests(unittest.TestCase):
    def _export_sample(self, root: Path, event_id: str, when: datetime):
        frames = []
        for index in range(3):
            frame = np.zeros((24, 24, 3), dtype=np.uint8)
            frame[10:13, 4 + index * 4:7 + index * 4] = 200
            frames.append(frame)
        return Path(ml_training_data.export_training_event(
            root_dir=str(root), predicted_label="not_meteor", probability=0.08,
            event_id=event_id, source=f"/{when:%Y%m%d}/22/57.mp4",
            detection_time=when, frames=frames,
            classification_diff=np.max(np.stack(frames), axis=0),
            cutout_rect=(0, 0, 24, 24), cutout_size=24, frame_rate=15.0,
            detected_line=((4, 11), (15, 11)), frame_range=(1, 3),
        ))

    def test_export_review_and_persistent_undo(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            frames = []
            for index in range(9):
                frame = np.zeros((64, 64, 3), dtype=np.uint8)
                cv2.circle(frame, (10 + index * 4, 32), 2, (180, 180, 180), -1)
                frames.append(frame)
            diff = np.max(np.stack(frames), axis=0)

            event_path = ml_training_data.export_training_event(
                root_dir=str(tmp_path), predicted_label="meteor", probability=0.83,
                event_id="sample_event", source="source.mp4",
                detection_time=datetime(2026, 1, 2, 3, 4, 5), frames=frames,
                classification_diff=diff, cutout_rect=(0, 0, 64, 64), cutout_size=64,
                frame_rate=15.0, detected_line=((10, 32), (42, 32)), frame_range=(100, 108),
            )
            event = tmp_path.resolve() / "pending" / "meteor" / "sample_event"
            self.assertEqual(event_path, str(event))
            self.assertTrue(
                {"diff.png", "temporal_rgb.png", "time_of_peak.png", "clip.mp4", "metadata.json"}
                <= {path.name for path in event.iterdir()}
            )
            self.assertEqual(ml_training_data.pending_events(str(tmp_path)), [event])

            record = ml_training_data.review_event(event, str(tmp_path), "not_meteor")
            reviewed = tmp_path / "reviewed" / "not_meteor" / "sample_event"
            self.assertTrue(reviewed.exists())
            with (reviewed / "metadata.json").open(encoding="utf-8") as handle:
                metadata = json.load(handle)
            self.assertTrue(metadata["was_misclassified"])
            self.assertEqual(ml_training_data.pending_events(str(tmp_path)), [])
            self.assertEqual(len(ml_training_data.undoable_reviews(str(tmp_path))), 1)

            restored = ml_training_data.undo_review(record, str(tmp_path))
            self.assertEqual(restored, event)
            self.assertTrue(event.exists())
            self.assertEqual(ml_training_data.undoable_reviews(str(tmp_path)), [])
            with (event / "metadata.json").open(encoding="utf-8") as handle:
                metadata = json.load(handle)
            self.assertEqual(metadata["review_status"], "pending")
            self.assertNotIn("reviewed_label", metadata)

    def test_temporal_representation_preserves_time_order(self):
        frames = []
        for x in (4, 12, 20):
            frame = np.zeros((24, 24), dtype=np.uint8)
            frame[10:13, x:x + 3] = 255
            frames.append(frame)
        _, temporal, peak = ml_training_data.build_temporal_representations(
            frames, (0, 0, 24, 24), 24
        )
        self.assertGreater(temporal[11, 5, 2], 0)
        self.assertGreater(temporal[11, 13, 1], 0)
        self.assertGreater(temporal[11, 21, 0], 0)
        self.assertLess(peak[11, 5], peak[11, 13])
        self.assertLess(peak[11, 13], peak[11, 21])

    def test_skip_is_hidden_persistent_and_reversible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            event = self._export_sample(
                root, "20221028_sample", datetime(2022, 10, 28, 22, 57, 28)
            )
            self.assertEqual(ml_training_data.detection_date(event), "2022-10-28")

            record = ml_training_data.skip_event(event, str(root), "same_date:2022-10-28")
            skipped = root / "skipped" / "not_meteor" / event.name
            self.assertTrue(skipped.exists())
            self.assertEqual(ml_training_data.pending_events(str(root)), [])
            self.assertEqual(ml_training_data.undoable_skips(str(root)), [record])
            with (skipped / "metadata.json").open(encoding="utf-8") as handle:
                metadata = json.load(handle)
            self.assertEqual(metadata["review_status"], "skipped")
            self.assertEqual(metadata["skip_reason"], "same_date:2022-10-28")

            restored = ml_training_data.undo_skip(record, str(root))
            self.assertEqual(restored, event)
            self.assertEqual(ml_training_data.pending_events(str(root)), [event])
            self.assertEqual(ml_training_data.undoable_skips(str(root)), [])
            with (event / "metadata.json").open(encoding="utf-8") as handle:
                metadata = json.load(handle)
            self.assertEqual(metadata["review_status"], "pending")
            self.assertNotIn("skipped_at", metadata)

    def test_detection_date_allows_grouping_same_night_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._export_sample(root, "first", datetime(2022, 10, 28, 18, 0))
            second = self._export_sample(root, "second", datetime(2022, 10, 28, 23, 59))
            third = self._export_sample(root, "third", datetime(2022, 10, 29, 0, 1))
            grouped = [
                event for event in ml_training_data.pending_events(str(root))
                if ml_training_data.detection_date(event) == "2022-10-28"
            ]
            self.assertEqual(grouped, [first, second])
            self.assertNotIn(third, grouped)


if __name__ == "__main__":
    unittest.main()
