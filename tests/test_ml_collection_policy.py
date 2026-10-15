import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

import ml_collection_policy as policy
import ml_training_data


def decide(probability, threshold=0.06657051318251248, when=None, fraction=0.30):
    return policy.collection_decision(
        probability=probability, decision_threshold=threshold,
        boundary_fraction=fraction, source="source.mp4",
        detection_time=when or datetime(2026, 10, 14, 3),
        frame_range=(1, 3), cutout_rect=(0, 0, 24, 24),
    )


def test_model_positive_below_old_floor_and_boundary_are_retained(monkeypatch):
    monkeypatch.setattr(policy, "sample_value", lambda _: 1)
    assert decide(0.08)["reason"] == "predicted_meteor"
    assert decide(0.05)["reason"] == "decision_boundary"
    assert decide(0.04) is None
    # The boundary moves with the selected model's operating threshold.
    assert decide(0.05, threshold=0.17177035153000328) is None
    threshold = 0.06657051318251248
    assert decide(threshold)["reason"] == "predicted_meteor"
    assert decide(threshold * 0.7)["reason"] == "decision_boundary"


def test_exact_sample_rate_and_zero_width(monkeypatch):
    monkeypatch.setattr(policy, "sample_value", lambda _: 0.000999)
    assert decide(0.01)["reason"] == "low_score_sample"
    monkeypatch.setattr(policy, "sample_value", lambda _: 0.001)
    assert decide(0.01) is None
    assert decide(0.05, fraction=0) is None
    assert decide(0, threshold=0)["reason"] == "predicted_meteor"


def test_sampling_is_stable_and_approximately_one_in_a_thousand():
    keys = [str(index) for index in range(100_000)]
    count = sum(policy.sample_value(key) < 0.001 for key in keys)
    assert 70 <= count <= 130
    assert policy.sample_value("same-event") == policy.sample_value("same-event")


def test_model_change_does_not_reroll_low_score_selection(monkeypatch):
    keys = []
    monkeypatch.setattr(policy, "sample_value", lambda key: keys.append(key) or 0)
    old = decide(.01, threshold=.1718)
    new = decide(.01, threshold=.0666)
    assert old["event_key"] == new["event_key"]
    assert keys[0] == keys[1]


def test_observation_night_crosses_midnight_and_resets_at_noon():
    assert decide(1, when=datetime(2026, 10, 13, 20))["observation_night"] == "2026-10-13"
    assert decide(1, when=datetime(2026, 10, 14, 11, 59))["observation_night"] == "2026-10-13"
    assert decide(1, when=datetime(2026, 10, 14, 12))["observation_night"] == "2026-10-14"


@pytest.mark.parametrize("threshold,fraction", [(float("nan"), .3), (-.1, .3), (.5, float("inf")), (.5, 1.1)])
def test_invalid_policy_cannot_silently_discard_data(threshold, fraction):
    with pytest.raises(ValueError):
        decide(.1, threshold=threshold, fraction=fraction)


def test_cap_is_shared_between_workers_and_survives_a_new_process(tmp_path):
    night = "2026-10-13"
    with ThreadPoolExecutor(max_workers=12) as workers:
        results = list(workers.map(lambda index: policy.reserve_low_score(tmp_path, night, str(index)), range(160)))
    assert sum(results) == 100
    code = "from pathlib import Path; import ml_collection_policy as p; import sys; print(p.reserve_low_score(Path(sys.argv[1]), '2026-10-13', 'after-restart'))"
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)],
                            cwd=str(Path(policy.__file__).parent), text=True, capture_output=True, check=True)
    assert result.stdout.strip() == "False"
    assert policy.reserve_low_score(tmp_path, "2026-10-14", "next-night")


def test_same_sample_cannot_consume_multiple_slots(tmp_path):
    assert policy.reserve_low_score(tmp_path, "2026-10-13", "same")
    assert not policy.reserve_low_score(tmp_path, "2026-10-13", "same")
    policy.release_low_score(tmp_path, "2026-10-13", "same")
    assert policy.reserve_low_score(tmp_path, "2026-10-13", "same")


def export(root, probability, *, index=0, model="v6"):
    frame = np.zeros((24, 24, 3), dtype=np.uint8)
    return ml_training_data.export_training_event(
        root_dir=str(root), predicted_label="meteor" if probability >= .0666 else "not_meteor",
        probability=probability, event_id=f"event_{model}_{index}", source="source.mp4",
        detection_time=datetime(2026, 10, 14, 3) + timedelta(seconds=index),
        frames=[frame] * 3, classification_diff=frame,
        cutout_rect=(0, 0, 24, 24), cutout_size=24, frame_rate=15,
        detected_line=((4, 11), (15, 11)), frame_range=(1, 3),
        decision_threshold=.0666, boundary_fraction=.30,
        model_info={"release_id": model, "model_sha256": "a" * 64, "preprocess_version": 2},
    )


def test_export_records_model_and_reason_without_using_old_floor(tmp_path, monkeypatch):
    monkeypatch.setattr(policy, "sample_value", lambda _: 1)
    event = export(tmp_path, .08)
    metadata = ml_training_data.load_metadata(Path(event))
    assert metadata["review_status"] == "pending"
    assert metadata["minimum_save_probability"] is None
    assert metadata["collection"]["reason"] == "predicted_meteor"
    assert metadata["collection"]["boundary_fraction"] == .30
    assert metadata["model"]["release_id"] == "v6"
    assert metadata["model"]["model_sha256"] == "a" * 64
    boundary = export(tmp_path, .05, index=1)
    assert ml_training_data.load_metadata(Path(boundary))["collection"]["reason"] == "decision_boundary"
    assert not (tmp_path / "collection_quota.sqlite3").exists()
    assert export(tmp_path, .01, index=2) is None


def test_full_quota_only_blocks_low_scores_even_after_model_change(tmp_path, monkeypatch):
    monkeypatch.setattr(policy, "sample_value", lambda _: 0)
    event = export(tmp_path, .01)
    assert ml_training_data.load_metadata(Path(event))["collection"]["reason"] == "low_score_sample"
    for index in range(99):
        assert policy.reserve_low_score(tmp_path, "2026-10-13", f"filler-{index}")
    assert export(tmp_path, .01, index=1, model="v7") is None
    assert export(tmp_path, .08, index=2, model="v7")
    assert export(tmp_path, .05, index=3, model="v7")


def test_failed_export_releases_slot_and_removes_partial_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(policy, "sample_value", lambda _: 0)
    with monkeypatch.context() as patch:
        patch.setattr(ml_training_data.cv2, "imwrite", lambda *_: False)
        with pytest.raises(IOError, match="diff.png"):
            export(tmp_path, .01)
    assert not list((tmp_path / "pending" / "not_meteor").iterdir())
    with sqlite3.connect(str(tmp_path / "collection_quota.sqlite3")) as connection:
        assert connection.execute("SELECT COUNT(*) FROM low_score_slots").fetchone()[0] == 0
    assert export(tmp_path, .01)
