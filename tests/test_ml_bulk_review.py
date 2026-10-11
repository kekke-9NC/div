import json
from pathlib import Path
import threading
from unittest import mock

import pytest

import ml_bulk_review as bulk
import ml_training_data as data


def event(root, name, probability=0.1, label="not_meteor", **metadata):
    path = root / "pending" / label / name
    path.mkdir(parents=True)
    contents = dict(event_id=name, meteor_probability=probability, predicted_label=label,
                    review_status="pending", detection_time="2026-10-12T23:00:00", **metadata)
    (path / "metadata.json").write_text(json.dumps(contents))
    (path / "clip.mp4").write_bytes(b"original-media")
    return path


def select(root, **filters):
    return bulk.select_events(str(root), bulk.ReviewFilter(**filters), threading.Event())


@pytest.mark.parametrize("comparison,expected", [("below", ["low", "middle"]), ("above", ["high", "middle"])])
def test_inclusive_threshold_matches_both_predictions(tmp_path, comparison, expected):
    event(tmp_path, "low", 0.05, "meteor")
    event(tmp_path, "middle", 0.1)
    event(tmp_path, "high", 0.8)
    assert [item.path.name for item in select(tmp_path, comparison=comparison).candidates] == expected


@pytest.mark.parametrize("probability", [None, "bad", True, float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_probabilities_are_excluded(tmp_path, probability):
    event(tmp_path, "bad", probability)
    result = select(tmp_path, threshold=1)
    assert result.candidates == []
    assert result.unreadable == 1


def test_missing_or_broken_metadata_and_nonpending_are_excluded(tmp_path):
    a = event(tmp_path, "missing")
    (a / "metadata.json").write_text('{"review_status":"pending"}')
    b = event(tmp_path, "broken")
    (b / "metadata.json").write_text("{")
    c = event(tmp_path, "reviewed")
    meta = data.load_metadata(c); meta["review_status"] = "reviewed"
    (c / "metadata.json").write_text(json.dumps(meta))
    result = select(tmp_path)
    assert result.candidates == []
    assert result.unreadable == 3


def test_day_and_prediction_filters(tmp_path):
    event(tmp_path, "today", label="meteor")
    old = event(tmp_path, "yesterday", label="meteor")
    meta = data.load_metadata(old); meta["detection_time"] = "2026-10-11T23:59:59+09:00"
    (old / "metadata.json").write_text(json.dumps(meta))
    event(tmp_path, "negative")
    assert [item.path.name for item in select(tmp_path, detection_date="2026-10-12", predicted_label="meteor").candidates] == ["today"]


@pytest.mark.parametrize("filters", [{"threshold": "nan"}, {"threshold": 1.1}, {"threshold": -1},
                                     {"comparison": "less"}, {"detection_date": "2026-99-99"}, {"predicted_label": "bad"}])
def test_invalid_filters_fail_before_selection(filters):
    with pytest.raises(ValueError):
        bulk.ReviewFilter(**filters)


@pytest.mark.parametrize("action", ["skip", "meteor", "not_meteor"])
def test_bulk_review_persistent_batch_undo_preserves_all_media(tmp_path, action):
    originals = [event(tmp_path, "a"), event(tmp_path, "b", label="meteor")]
    outside = event(tmp_path, "outside", 0.9)
    result = bulk.apply_selection(str(tmp_path), select(tmp_path), action, threading.Event())
    assert len(result.records) == 2
    assert not result.error
    assert outside.exists()
    assert len({record["batch_id"] for record in result.records}) == 1
    active = data.undoable_skips(str(tmp_path)) + data.undoable_reviews(str(tmp_path))
    records = bulk.latest_batch_records(active)
    assert len(records) == 2
    for record in records:
        moved = Path(record["to"])
        assert (moved / "clip.mp4").read_bytes() == b"original-media"
        metadata = data.load_metadata(moved)
        assert metadata["review_status"] == ("skipped" if action == "skip" else "reviewed")
        if action != "skip": assert metadata["reviewed_label"] == action
    undone = bulk.undo_batch(str(tmp_path), records, threading.Event())
    assert len(undone.records) == 2
    assert not undone.error
    for original in originals:
        meta = data.load_metadata(original)
        assert meta["review_status"] == "pending"
        assert "skip_batch_id" not in meta and "review_batch_id" not in meta
    assert bulk.latest_batch_records(data.undoable_skips(str(tmp_path)) + data.undoable_reviews(str(tmp_path))) == []


def test_snapshot_excludes_new_changed_and_already_reviewed_events(tmp_path):
    changed = event(tmp_path, "changed")
    moved = event(tmp_path, "moved")
    selection = select(tmp_path)
    event(tmp_path, "new")
    meta = data.load_metadata(changed); meta["meteor_probability"] = 0.9
    (changed / "metadata.json").write_text(json.dumps(meta))
    data.review_event(moved, str(tmp_path), "meteor")
    result = bulk.apply_selection(str(tmp_path), selection, "skip", threading.Event())
    assert result.records == [] and result.changed == 2
    assert changed.exists() and (tmp_path / "pending/not_meteor/new").exists()


def test_scan_and_apply_cancellation(tmp_path):
    for name in ("a", "b", "c"): event(tmp_path, name)
    cancel = threading.Event()
    partial = bulk.select_events(str(tmp_path), bulk.ReviewFilter(), cancel, lambda *_: cancel.set())
    assert partial.cancelled
    with pytest.raises(ValueError): bulk.apply_selection(str(tmp_path), partial, "meteor", threading.Event())
    cancel.clear()
    result = bulk.apply_selection(str(tmp_path), select(tmp_path), "meteor", cancel,
                                  progress=lambda *_: cancel.set())
    assert result.cancelled and len(result.records) == 1
    assert len(data.pending_events(str(tmp_path))) == 2
    assert len(bulk.undo_batch(str(tmp_path), result.records, threading.Event()).records) == 1


@pytest.mark.parametrize("action", ["skip", "meteor"])
def test_failed_log_rolls_back_only_failed_item_and_retains_undo(tmp_path, action):
    event(tmp_path, "a")
    failed = event(tmp_path, "b")
    original = (failed / "metadata.json").read_text()
    selection = select(tmp_path)
    append = data._append_jsonl
    def fail_second(path, record):
        if record["event_id"] == "b": raise OSError("disk full")
        append(path, record)
    with mock.patch.object(data, "_append_jsonl", side_effect=fail_second):
        result = bulk.apply_selection(str(tmp_path), selection, action, threading.Event())
    assert len(result.records) == 1 and "disk full" in result.error
    assert failed.exists() and data.load_metadata(failed) == json.loads(original)
    assert len(bulk.undo_batch(str(tmp_path), result.records, threading.Event()).records) == 1


def test_failed_skip_undo_log_can_be_retried(tmp_path):
    pending = event(tmp_path, "a")
    result = bulk.apply_selection(str(tmp_path), select(tmp_path), "skip", threading.Event())
    with mock.patch.object(data, "_append_jsonl", side_effect=OSError("disk full")):
        undone = bulk.undo_batch(str(tmp_path), result.records, threading.Event())
    assert undone.error and not pending.exists()
    assert data.load_metadata(Path(result.records[0]["to"]))["review_status"] == "skipped"
    assert len(data.undoable_skips(str(tmp_path))) == 1
    assert len(bulk.undo_batch(str(tmp_path), result.records, threading.Event()).records) == 1


def test_linked_events_cannot_escape_dataset(tmp_path):
    target = event(tmp_path / "other", "outside")
    pending = tmp_path / "pending/not_meteor"; pending.mkdir(parents=True)
    (pending / "link").symlink_to(target, target_is_directory=True)
    assert select(tmp_path).unreadable == 1
    assert target.exists()


def test_linked_metadata_is_excluded(tmp_path):
    real = event(tmp_path, "real", 0.9)
    linked = event(tmp_path, "linked", 0.1)
    (linked / "metadata.json").unlink()
    (linked / "metadata.json").symlink_to(real / "metadata.json")
    selected = select(tmp_path, threshold=1)
    assert [item.path.name for item in selected.candidates] == ["real"]
    assert selected.unreadable == 1


def test_bulk_meteor_reclassification_moves_and_restores_detection_media(tmp_path):
    meteor = tmp_path / "meteor"; negative = tmp_path / "not_meteor"
    meteor.mkdir(); negative.mkdir()
    source = negative / "a_clip.mp4"; source.write_bytes(b"detection-video")
    event(tmp_path / "dataset", "a")
    result = bulk.apply_selection(str(tmp_path / "dataset"), select(tmp_path / "dataset"), "meteor", threading.Event(),
                                  {"meteor": str(meteor), "not_meteor": str(negative)})
    assert not result.error and not source.exists()
    assert (meteor / source.name).read_bytes() == b"detection-video"
    assert not bulk.undo_batch(str(tmp_path / "dataset"), result.records, threading.Event()).error
    assert source.read_bytes() == b"detection-video"


def test_latest_batch_ignores_later_single_reviews(tmp_path):
    event(tmp_path, "a")
    result = bulk.apply_selection(str(tmp_path), select(tmp_path), "meteor", threading.Event())
    later = event(tmp_path, "later")
    data.review_event(later, str(tmp_path), "meteor")
    assert bulk.latest_batch_records(data.undoable_reviews(str(tmp_path))) == result.records
