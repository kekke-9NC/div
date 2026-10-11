import json
import shutil
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import ml_training_data


class TrainingDataTests(unittest.TestCase):
    def _export_sample(
        self,
        root: Path,
        event_id: str,
        when: datetime,
        predicted_label: str = "not_meteor",
        media_directories=None,
    ):
        frames = []
        for index in range(3):
            frame = np.zeros((24, 24, 3), dtype=np.uint8)
            frame[10:13, 4 + index * 4:7 + index * 4] = 200
            frames.append(frame)
        return Path(ml_training_data.export_training_event(
            root_dir=str(root), predicted_label=predicted_label,
            probability=0.95 if predicted_label == "meteor" else 0.08,
            event_id=event_id, source=f"/{when:%Y%m%d}/22/57.mp4",
            detection_time=when, frames=frames,
            classification_diff=np.max(np.stack(frames), axis=0),
            cutout_rect=(0, 0, 24, 24), cutout_size=24, frame_rate=15.0,
            detected_line=((4, 11), (15, 11)), frame_range=(1, 3),
            media_directories=media_directories,
            minimum_probability=0.0,
        ))

    def test_export_skips_probabilities_below_configured_floor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kwargs = {
                "root_dir": str(root), "predicted_label": "not_meteor",
                "probability": 0.149, "event_id": "below_floor", "source": "source.mp4",
                "detection_time": datetime(2026, 1, 2, 3, 4, 5),
                "frames": [np.zeros((24, 24, 3), dtype=np.uint8)],
                "classification_diff": np.zeros((24, 24), dtype=np.uint8),
                "cutout_rect": (0, 0, 24, 24), "cutout_size": 24,
                "frame_rate": 15.0, "detected_line": ((4, 11), (15, 11)),
                "frame_range": (1, 1),
            }
            self.assertIsNone(ml_training_data.export_training_event(**kwargs))
            self.assertFalse((root / "pending").exists())

            kwargs["probability"] = 0.15
            exported = ml_training_data.export_training_event(**kwargs)
            self.assertEqual(exported, str(root / "pending" / "not_meteor" / "below_floor"))
            with (Path(exported) / "metadata.json").open(encoding="utf-8") as handle:
                metadata = json.load(handle)
            self.assertEqual(metadata["minimum_save_probability"], 0.15)

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

    def test_review_moves_detected_media_and_undo_restores_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meteor_dir = root / "detected_meteor"
            not_meteor_dir = root / "detected_not_meteor"
            media_directories = {
                "meteor": str(meteor_dir),
                "not_meteor": str(not_meteor_dir),
            }
            event = self._export_sample(
                root, "false_positive", datetime(2026, 10, 1, 22, 57),
                predicted_label="meteor", media_directories=media_directories,
            )
            meteor_dir.mkdir()
            related_files = [
                meteor_dir / f"{event.name}.mp4",
                meteor_dir / f"{event.name}_composite.jpg",
                meteor_dir / f"{event.name}_info.txt",
            ]
            for path in related_files:
                path.write_text("sample", encoding="utf-8")
            unrelated = meteor_dir / "another_event.mp4"
            unrelated.write_text("keep", encoding="utf-8")

            record = ml_training_data.review_event(event, str(root), "not_meteor")
            reviewed = root / "reviewed" / "not_meteor" / event.name
            self.assertTrue(reviewed.is_dir())
            self.assertEqual(len(record["media_moves"]), len(related_files))
            for path in related_files:
                self.assertFalse(path.exists())
                self.assertTrue(not_meteor_dir.joinpath(path.name).is_file())
            self.assertTrue(unrelated.is_file())

            restored = ml_training_data.undo_review(record, str(root))
            self.assertEqual(restored, event)
            for path in related_files:
                self.assertTrue(path.is_file())
                self.assertFalse(not_meteor_dir.joinpath(path.name).exists())

    def test_skip_does_not_move_detected_media(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meteor_dir = root / "detected_meteor"
            not_meteor_dir = root / "detected_not_meteor"
            media_directories = {
                "meteor": str(meteor_dir),
                "not_meteor": str(not_meteor_dir),
            }
            event = self._export_sample(
                root, "skipped_event", datetime(2026, 10, 2, 22, 57),
                predicted_label="meteor", media_directories=media_directories,
            )
            meteor_dir.mkdir()
            detected_clip = meteor_dir / f"{event.name}.mp4"
            detected_clip.write_text("keep in predicted folder", encoding="utf-8")

            record = ml_training_data.skip_event(event, str(root))

            self.assertTrue(Path(record["to"]).is_dir())
            self.assertTrue(detected_clip.is_file())
            self.assertFalse(not_meteor_dir.exists())

    def test_sync_moves_media_for_legacy_reviewed_events(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meteor_dir = root / "detected_meteor"
            not_meteor_dir = root / "detected_not_meteor"
            media_directories = {
                "meteor": str(meteor_dir),
                "not_meteor": str(not_meteor_dir),
            }
            event = self._export_sample(
                root, "legacy_false_positive", datetime(2026, 10, 3, 22, 57),
                predicted_label="meteor",
            )
            meteor_dir.mkdir()
            detected_files = [
                meteor_dir / f"{event.name}.mp4",
                meteor_dir / f"{event.name}_cutout_diff.jpg",
            ]
            for path in detected_files:
                path.write_text("legacy output", encoding="utf-8")

            metadata = ml_training_data.load_metadata(event)
            metadata.update({
                "review_status": "reviewed",
                "reviewed_label": "not_meteor",
                "was_misclassified": True,
                "reviewed_at": "2026-10-04T00:00:00+00:00",
            })
            (event / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            reviewed = root / "reviewed" / "not_meteor" / event.name
            reviewed.parent.mkdir(parents=True)
            shutil.move(str(event), str(reviewed))
            record = {
                "action": "review",
                "timestamp": "legacy-review",
                "from": str(event),
                "to": str(reviewed),
            }

            moved_count = ml_training_data.sync_reviewed_media(
                str(root), media_directories
            )

            self.assertEqual(moved_count, len(detected_files))
            for path in detected_files:
                self.assertFalse(path.exists())
                self.assertTrue(not_meteor_dir.joinpath(path.name).is_file())
            self.assertEqual(
                ml_training_data.sync_reviewed_media(str(root), media_directories), 0
            )

            restored = ml_training_data.undo_review(record, str(root))
            self.assertEqual(restored, event)
            for path in detected_files:
                self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()
