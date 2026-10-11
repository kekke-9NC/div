from pathlib import Path
import json

import cv2
import numpy as np
import pytest
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from event_boost_model import (ARCHITECTURE_NAME, EventBoostClassifier,
                               export_boosted_trees, extract_event_features,
                               extract_motion_features)
from universal_meteor_model import build_universal_inputs


def moving_frames():
    frames = []
    for index in range(12):
        frame = np.full((128, 128), 8, dtype=np.uint8)
        cv2.line(frame, (20 + index * 4, 60), (28 + index * 4, 60), 100, 2)
        frames.append(frame)
    return frames


def test_portable_trees_preserve_probabilities_and_safe_checkpoint(tmp_path):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(120, 5))
    y = (x[:, 0] + .4 * x[:, 1] > 0).astype(int)
    with threadpool_limits(limits=1):
        estimators = [HistGradientBoostingClassifier(
            max_iter=12, max_leaf_nodes=5, min_samples_leaf=3,
            early_stopping=False, random_state=seed).fit(x, y) for seed in (1, 2)]
    checkpoint = export_boosted_trees(estimators, x.shape[1])
    path = tmp_path / "test.pth"
    torch.save(checkpoint, path)
    # This catches accidental NumPy/pickle objects entering the deployment file.
    portable = EventBoostClassifier(torch.load(path, weights_only=True))
    expected = np.mean([m.predict_proba(x)[:, 1] for m in estimators], axis=0)
    np.testing.assert_allclose(portable.predict_proba(x), expected, rtol=0, atol=1e-12)
    for index in (0, 5, 119):
        np.testing.assert_allclose(portable.predict_proba(x[index]), expected[index:index + 1], rtol=0, atol=1e-12)
    # Constructing the runtime must not mutate the original export payload.
    EventBoostClassifier(checkpoint)
    assert isinstance(checkpoint["members"][0]["trees"][0]["value"], torch.Tensor)
    with pytest.raises(ValueError, match="feature"):
        portable.predict_proba(np.zeros(4))


@pytest.mark.parametrize("count", [1, 2, 12])
def test_event_features_handle_short_and_blank_clips(count):
    blank = [np.zeros((128, 128), dtype=np.uint8) for _ in range(count)]
    moving = moving_frames()[:count]
    vectors = []
    for frames in (blank, moving):
        image, kymo, physical = build_universal_inputs(frames)
        vector = np.r_[extract_event_features(image, kymo, physical),
                       extract_motion_features(frames)]
        assert np.isfinite(vector).all()
        vectors.append(vector)
    assert vectors[0].shape == vectors[1].shape
    assert vectors[0].ndim == 1


def test_motion_crop_matches_standalone_event():
    cropped = moving_frames()
    full = [np.pad(frame, ((20, 10), (30, 10))) for frame in cropped]
    np.testing.assert_array_equal(
        extract_motion_features(cropped, frame_rate=25),
        extract_motion_features(full, (30, 20, 158, 148), frame_rate=25))


def test_failed_model_switch_preserves_working_classifier(monkeypatch, tmp_path):
    import model
    sentinel = object()
    info = {"architecture": ARCHITECTURE_NAME, "decision_threshold": .37}
    monkeypatch.setattr(model, "model", sentinel)
    monkeypatch.setattr(model, "_model_ready", True)
    monkeypatch.setattr(model, "_active_model_info", info)
    ok, _ = model.reload_model(str(tmp_path / "missing.pth"))
    assert not ok
    assert model.model is sentinel and model._model_ready
    broken = tmp_path / "broken.pth"
    broken.write_bytes(b"invalid model")
    ok, _ = model.reload_model(str(broken))
    assert not ok
    assert model.model is sentinel and model._model_ready
    assert model.get_meteor_decision_threshold() == .37


def test_event_boost_uses_the_focused_activity_window(monkeypatch):
    import video_processing
    monkeypatch.setattr(video_processing.model, "get_active_model_info",
                        lambda: {"architecture": ARCHITECTURE_NAME})
    frames = [np.full((2, 2), index, dtype=np.uint8) for index in range(60)]
    selected = video_processing._select_model_event_frames(frames, 677, 701, 711, 25.)
    assert len(selected) == 35
    assert selected[0][0, 0] == 12 and selected[-1][0, 0] == 46


def test_model_catalog_retains_event_boost_metadata(tmp_path):
    import model_catalog
    path = str(tmp_path / "detector.pth")
    model_catalog.save_model_metadata(path, [0] * 3, [1] * 3,
        class_names=["not_meteor", "meteor"], extra={
            "architecture": ARCHITECTURE_NAME, "decision_threshold": .37,
            "test_metrics": {"precision": .99}})
    loaded = model_catalog.load_model_metadata(path)
    assert loaded["architecture"] == ARCHITECTURE_NAME
    assert loaded["meteor_class_index"] == 1
    assert loaded["decision_threshold"] == .37
    assert loaded["test_metrics"]["precision"] == .99


def test_reviewed_dataset_rejects_conflicting_duplicates_and_unconfirmed_labels(tmp_path):
    from train_event_boost import reviewed_events
    for label in ("meteor", "not_meteor"):
        (tmp_path / label).mkdir()
    def write_event(label, name, clip, status="reviewed"):
        directory = tmp_path / label / name
        directory.mkdir()
        (directory / "clip.mp4").write_bytes(clip)
        (directory / "metadata.json").write_text(json.dumps({
            "review_status": status, "reviewed_label": label,
            "detection_time": "2026-07-01T01:00:00", "frame_rate": 25.}))
        return directory
    write_event("meteor", "a", b"same event")
    write_event("meteor", "b", b"same event")
    write_event("not_meteor", "c", b"different event")
    events, duplicates = reviewed_events(tmp_path)
    assert len(events) == 2 and len(duplicates) == 1
    conflicting = write_event("not_meteor", "d", b"same event")
    with pytest.raises(ValueError, match="Conflicting"):
        reviewed_events(tmp_path)
    (conflicting / "clip.mp4").write_bytes(b"unconfirmed event")
    (conflicting / "metadata.json").write_text(json.dumps({"review_status": "pending"}))
    with pytest.raises(ValueError, match="Unconfirmed"):
        reviewed_events(tmp_path)


def test_frozen_image_encoder_runs_offline_with_embedded_weights(monkeypatch):
    from transformers import Dinov2Config, Dinov2Model
    from event_transformer import EventImageEncoder, encoder_pixels
    config = Dinov2Config(hidden_size=24, num_hidden_layers=1, num_attention_heads=3,
                          mlp_ratio=2, image_size=28, patch_size=14)
    original = Dinov2Model(config)
    checkpoint = {"config": config.to_dict(), "state_dict": original.state_dict()}
    def forbidden(*args, **kwargs):
        raise AssertionError("Runtime must not download weights")
    monkeypatch.setattr(Dinov2Model, "from_pretrained", forbidden)
    encoder = EventImageEncoder(checkpoint)
    image = np.zeros((5, 160, 160), dtype=np.float32)
    assert encoder_pixels(image).shape == (2, 3, 224, 224)
    vectors = encoder.encode([image, image])
    assert vectors.shape == (2, 96)
    assert np.isfinite(vectors).all()
    np.testing.assert_array_equal(vectors[0], vectors[1])


def test_portable_linear_head_matches_sklearn():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    x = np.random.default_rng(42).normal(size=(80, 5))
    y = (x[:, 0] > 0).astype(int)
    estimator = make_pipeline(StandardScaler(), LogisticRegression(C=.1)).fit(x, y)
    scaler, linear = estimator.steps[0][1], estimator.steps[1][1]
    checkpoint = export_boosted_trees([], 5)
    checkpoint["linear_members"] = [{"mean": torch.from_numpy(scaler.mean_),
        "scale": torch.from_numpy(scaler.scale_), "coef": torch.from_numpy(linear.coef_[0]),
        "intercept": float(linear.intercept_[0])}]
    portable = EventBoostClassifier(checkpoint)
    np.testing.assert_allclose(portable.predict_proba(x), estimator.predict_proba(x)[:, 1], atol=1e-12, rtol=0)
