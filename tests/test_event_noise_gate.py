from unittest.mock import Mock, patch

import cv2
import numpy as np

import event_noise_gate
import model
import model_catalog
import short_flash_detector
import video_processing


def test_component_scores_and_stable_ties_match_original_dense_loop():
    rng = np.random.default_rng(47)
    binary = np.zeros((160, 240), np.uint8)
    residual = np.zeros_like(binary)
    for y in range(0, 160, 4):
        for x in range(0, 237, 5):
            binary[y, x:x+3] = 1
            residual[y, x:x+3] = rng.choice([4, 5, 7, 10, 15], size=3)
    _, labels, stats, centers = cv2.connectedComponentsWithStats(binary, 8)
    expected = []
    for k in range(1, len(stats)):
        x, y, w, h, area = map(int, stats[k])
        if area < 3 or area > 1000 or w > 350 or h > 350:
            continue
        values = residual[y:y+h, x:x+w][labels[y:y+h, x:x+w] == k]
        score, peak = int(values.sum()), int(values.max())
        if score >= 20 and peak >= 5:
            expected.append([*centers[k], score, w, h, peak])
    expected.sort(key=lambda d: d[2], reverse=True)
    assert short_flash_detector._component_detections(residual, labels, stats, centers) == expected[:400]


def test_native_gate_crop_and_standalone_crop_have_identical_features():
    rng = np.random.default_rng(23)
    frames = [rng.integers(0, 256, (90, 110, 3), np.uint8) for _ in range(6)]
    rect = (20, 10, 84, 74)
    crops = [f[10:74, 20:84] for f in frames]
    np.testing.assert_array_equal(
        event_noise_gate.extract_noise_gate_features(frames, rect, 25),
        event_noise_gate.extract_noise_gate_features(crops, frame_rate=25),
    )


def test_optional_gate_failure_keeps_full_classifier_path(monkeypatch):
    monkeypatch.setattr(model, '_active_noise_gate', None)
    assert model.predict_noise_gate([], None, 25) is None
    gate = Mock()
    gate.evaluate.side_effect = ValueError('bad optional input')
    monkeypatch.setattr(model, '_active_noise_gate', gate)
    assert model.predict_noise_gate([], None, 25) is None


def test_gate_metadata_survives_catalog_normalization(tmp_path):
    import json
    path = tmp_path / 'main.pth'
    path.write_bytes(b'main')
    metadata = {'architecture': 'meteor_event_transformer_boost_v1',
                'noise_gate_file': 'auxiliary/filter.pth',
                'noise_gate_sha256': 'a' * 64,
                'noise_gate_policy': 'reviewed chronological events'}
    (tmp_path / 'main.pth.meta.json').write_text(json.dumps(metadata))
    actual = model_catalog.load_model_metadata(str(path))
    for key, value in metadata.items():
        assert actual[key] == value


def test_builtin_release_upgrade_preserves_custom_or_acknowledged_choices(tmp_path):
    import json
    target = tmp_path / 'v2.pth'
    target.write_bytes(b'model')
    (tmp_path / 'v2.pth.meta.json').write_text(json.dumps({
        'release_id': '20261011', 'upgrade_from': ['v1.pth'],
    }))
    old = str(tmp_path / 'v1.pth')
    assert model_catalog.upgraded_builtin_model_path(old, str(target)) == str(target)
    assert model_catalog.upgraded_builtin_model_path(old, str(target), '20261011') == old
    custom = str(tmp_path / 'custom' / 'v1.pth')
    assert model_catalog.upgraded_builtin_model_path(custom, str(target)) == custom


def test_frame_cache_preserves_surrounding_resident_window():
    frames = [np.full((4, 4, 3), i, np.uint8) for i in range(30)]
    def read(source, start, end, threads):
        return frames[start:end+1]
    cache = video_processing._VideoFrameRangeCache(48 * 20 * 2)
    with patch.object(video_processing, '_read_video_frame_range', side_effect=read) as decoder:
        cache.read('source', 5, 14)
        cache.read('source', 10, 19)
        actual = cache.read('source', 5, 9)
        assert decoder.call_count == 2
        assert [int(f[0, 0, 0]) for f in actual] == list(range(5, 10))


def test_resident_slice_requires_complete_range_and_keeps_list_private():
    frames = [np.zeros((2, 2), np.uint8) for _ in range(5)]
    assert video_processing._resident_frame_slice(frames, 10, 9, 12) is None
    assert video_processing._resident_frame_slice(frames, 10, 11, 15) is None
    result = video_processing._resident_frame_slice(frames, 10, 11, 13)
    assert result == frames[1:4]
    result.pop()
    assert len(frames) == 5


def test_compact_exports_never_change_meteor_media(monkeypatch):
    monkeypatch.setattr(model, 'get_active_model_info', lambda: {
        'non_meteor_full_media_probability_min': .01,
    })
    monkeypatch.setattr(model, 'get_meteor_decision_threshold', lambda: .44)
    assert video_processing._keep_full_event_media(.8, True)
    assert video_processing._keep_full_event_media(0., True)
    assert not video_processing._keep_full_event_media(.005, False)
    assert video_processing._keep_full_event_media(.01, False)
    assert video_processing._keep_full_event_media(float('nan'), False)


def test_models_without_compact_export_policy_keep_previous_media(monkeypatch):
    monkeypatch.setattr(model, 'get_active_model_info', lambda: {})
    assert video_processing._keep_full_event_media(0., False)


def test_invalid_gate_checksum_preserves_active_classifier(monkeypatch, tmp_path):
    main = tmp_path / 'main.pth'
    main.write_bytes(b'model')
    gate = tmp_path / 'filter.pth'
    gate.write_bytes(b'changed filter')
    old_model, old_gate = object(), object()
    old_info = {'architecture': model.EVENT_BOOST_ARCHITECTURE_NAME,
                'decision_threshold': .37}
    monkeypatch.setattr(model, 'model', old_model)
    monkeypatch.setattr(model, '_active_noise_gate', old_gate)
    monkeypatch.setattr(model, '_active_model_info', old_info)
    monkeypatch.setattr(model, '_model_ready', True)
    monkeypatch.setattr(model, '_load_state_dict', lambda path: {})
    monkeypatch.setattr(model, 'EventBoostClassifier', lambda state: Mock(feature_count=3))
    ok, error = model.reload_model(str(main), metadata={
        'architecture': model.EVENT_BOOST_ARCHITECTURE_NAME,
        'noise_gate_file': gate.name, 'noise_gate_sha256': '0' * 64,
    })
    assert not ok and 'checksum' in error
    assert model.model is old_model and model._active_noise_gate is old_gate
    assert model._active_model_info is old_info and model._model_ready
