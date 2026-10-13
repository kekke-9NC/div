"""Exact-result regressions for dense and long-running legacy analyses."""
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from unittest.mock import patch

import numpy as np

import model
from universal_meteor_model import _robust_residual_cube, _uint8_median
from video_processing import _SavedEventIndex, _candidate_already_saved
from temporal_statistics import temporal_median


def test_signed_temporal_median_matches_numpy_for_all_short_lengths():
    rng = np.random.default_rng(159)
    for count in range(1,65):
        values = rng.integers(-32768,32768,size=(count,32,64),dtype=np.int16)
        values[0,0,:2] = [-32768,32767]
        before = values.copy()
        np.testing.assert_array_equal(temporal_median(values),np.median(values,axis=0))
        np.testing.assert_array_equal(values,before)
        np.testing.assert_array_equal(temporal_median(values[:,::2,::2]),np.median(values[:,::2,::2],axis=0))


def test_native_temporal_median_covers_full_event_crops_exactly():
    rng = np.random.default_rng(45)
    for dtype in (np.uint8,np.int16):
        values = rng.integers(0,256,(64,256,256),dtype=dtype)
        before = values.copy()
        np.testing.assert_array_equal(temporal_median(values),np.median(values,axis=0))
        np.testing.assert_array_equal(values,before)


def test_signed_temporal_median_falls_back_without_new_native_symbol():
    import component_statistics
    values = np.arange(120,dtype=np.int16).reshape(6,4,5)
    component_statistics._load()
    with patch.object(component_statistics,'_median_i16_function',None):
        np.testing.assert_array_equal(temporal_median(values),np.median(values,axis=0))
from short_flash_detector import _track_predictions
from temporal_statistics import half_integer_percentiles


def test_dense_component_ranking_matches_scalar_peaks_including_low_peak_ties():
    import cv2
    from short_flash_detector import _component_detections
    residual = np.zeros((160, 320), np.uint8)
    for index in range(1000):
        y, x = divmod(index, 64)
        residual[y*8:y*8+3, x*5:x*5+3] = 4 + index % 11
    _, labels, stats, centers = cv2.connectedComponentsWithStats((residual > 0).astype(np.uint8), 8)
    expected = []
    for k in range(1, len(stats)):
        x, y, w, h, area = stats[k]
        if not 3 <= area <= 1000 or w > 350 or h > 350:
            continue
        values = residual[y:y+h, x:x+w][labels[y:y+h, x:x+w] == k]
        peak, score = int(values.max()), int(values.sum())
        if peak >= 5 and score >= 20:
            expected.append([*centers[k], score, w, h, peak])
    expected.sort(key=lambda value: value[2], reverse=True)
    assert _component_detections(residual, labels, stats, centers) == expected[:400]


def test_live_gray_cache_preserves_pixels_scale_and_window_eviction():
    import moving_point_detector as moving
    rng = np.random.default_rng(556)
    frames = list(rng.integers(0, 256, (10, 100, 120, 3), dtype=np.uint8))
    cache = moving.GrayFrameCache()
    for scale in (.5, .5, 1.):
        actual = cache.convert(frames[2:8], scale)
        for raw, gray in zip(frames[2:8], actual):
            np.testing.assert_array_equal(gray, moving._gray_small(raw, scale))
    cache.convert(frames[5:10], .5)
    assert set(cache.entries) == {id(frame) for frame in frames[5:10]}


def test_histogram_percentiles_match_numpy_linear_interpolation_exactly():
    rng = np.random.default_rng(614)
    quantiles = [0, 50, 90, 99, 99.9, 100]
    for shape in ((1,), (2,), (3,), (4096,), (50, 64, 64)):
        for data in (rng.integers(0, 511, shape).astype(np.float32) * .5,
                     np.zeros(shape, np.float32), np.full(shape, 255, np.float32)):
            np.testing.assert_array_equal(half_integer_percentiles(data, quantiles), np.percentile(data, quantiles))


def test_gate_feature_optimizations_preserve_the_original_vector():
    import event_noise_gate
    rng = np.random.default_rng(805)
    for length in (1, 2, 25, 50):
        frames = list(rng.integers(0, 256, (length, 64, 64), dtype=np.uint8))
        actual = event_noise_gate.extract_noise_gate_features(frames)
        with patch.object(event_noise_gate, 'temporal_median', side_effect=lambda v: np.median(v, axis=0)), \
             patch.object(event_noise_gate, '_uint8_median', side_effect=lambda v: float(np.median(v))), \
             patch.object(event_noise_gate, 'half_integer_percentiles', side_effect=np.percentile):
            expected = event_noise_gate.extract_noise_gate_features(frames)
        np.testing.assert_array_equal(actual, expected)


def test_spatial_pruning_preserves_greedy_tracks_scores_and_ties():
    import moving_point_detector as moving
    rng = np.random.default_rng(848)
    initial = rng.uniform(20, 600, (150, 2))
    velocity = rng.uniform(-3, 3, (150, 2))
    detections = [[dict(x=float(x), y=float(y), score=float(10 + i % 3))
                   for i, (x, y) in enumerate(initial + frame * velocity)] for frame in range(12)]
    frames = [np.zeros((32, 32), dtype=np.uint8) for _ in detections]
    with patch.object(moving, '_frame_detections', side_effect=detections):
        actual = moving.detect_moving_point_tracks(frames, scale=1., max_candidates=200)
    with patch.object(moving, '_frame_detections', side_effect=detections), \
         patch.object(moving, '_nearby_detections', return_value=None):
        expected = moving.detect_moving_point_tracks(frames, scale=1., max_candidates=200)
    assert actual == expected


def test_track_positions_match_scalar_predictions_and_gate_decisions_at_boundaries():
    rng = np.random.default_rng(836)
    tracks, positions, gates = [], [], []
    for i in range(1000):
        points = []
        for t in range(95, 98 if i % 3 else 96):
            points.append([t, *rng.normal(300, 20, 2), 100, 8, 8, 30])
        tracks.append(points)
        last = points[-1]
        gap = 100 - last[0]
        velocity = np.zeros(2)
        if len(points) >= 2:
            previous = points[-2]
            velocity = (np.asarray(last[1:3]) - previous[1:3]) / max(1, last[0]-previous[0])
        positions.append(np.asarray(last[1:3]) + velocity * gap)
        gates.append(40 if len(points) < 2 else max(12, min(90, np.linalg.norm(velocity) * gap * .7 + 12)))
    actual_positions, provisional = _track_predictions(tracks, 100)
    np.testing.assert_array_equal(actual_positions, positions)
    gates = np.array(gates)
    for distances in (gates, np.nextafter(gates, 0.), np.nextafter(gates, np.inf), gates + 1.):
        corrected = provisional.copy()
        near = np.abs(distances - provisional) <= 1e-10
        corrected[near] = gates[near]
        np.testing.assert_array_equal(distances < corrected, distances < gates)


def test_temporal_median_keeps_chronological_input_and_exact_even_medians():
    rng = np.random.default_rng(85)
    for shape in ((1, 8, 10), (2, 8, 10), (25, 8, 10), (50, 8, 10), (7, 1, 1)):
        for dtype in (np.uint8, np.int16, np.float32):
            values = rng.integers(0, 256, shape, dtype=np.uint8).astype(dtype)
            for source in (values, values[:, ::2, ::2]):
                before = source.copy()
                np.testing.assert_array_equal(temporal_median(source), np.median(source, axis=0))
                np.testing.assert_array_equal(source, before)


def test_large_temporal_partition_matches_numpy_and_does_not_mutate_frames():
    rng = np.random.default_rng(926)
    values = rng.integers(0, 256, (25, 480, 720), dtype=np.uint8)
    before = values.copy()
    np.testing.assert_array_equal(temporal_median(values), np.median(values, axis=0))
    np.testing.assert_array_equal(values, before)


def test_sequential_capture_reuses_lookahead_without_changing_any_frame(tmp_path):
    import cv2
    from video_io import open_video_capture
    from video_processing import _VideoFrameRangeCache, _CachedSequentialCapture
    path = str(tmp_path / 'lookahead.mp4')
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (96, 64))
    rng = np.random.default_rng(52)
    for i in range(180):
        writer.write(rng.integers(0, 256, (64, 96, 3), dtype=np.uint8))
    writer.release()
    original = open_video_capture(path)
    reader = open_video_capture(path)
    cache = _VideoFrameRangeCache()
    actual = _CachedSequentialCapture(reader, cache, path, 2)
    try:
        for i in range(181):
            if i in (25, 80, 130):
                cache.read(path, i - 10, min(179, i + 40))
            ok, expected_frame = original.read()
            got, actual_frame = actual.read()
            assert ok == got
            if ok:
                np.testing.assert_array_equal(actual_frame, expected_frame)
                assert actual.get(cv2.CAP_PROP_POS_FRAMES) == i + 1
    finally:
        original.release()
        actual.release()


def test_training_temporal_images_remain_pixel_exact():
    from ml_training_data import build_temporal_representations
    rng = np.random.default_rng(541)
    for length in (1, 2, 25, 50):
        frames = list(rng.integers(0, 256, (length, 32, 32), dtype=np.uint8))
        crops, actual, actual_peak = build_temporal_representations(frames, (0, 0, 32, 32), 32)
        stack = np.stack(crops).astype(np.int16)
        response = np.abs(stack - np.median(stack, axis=0).astype(np.int16)).astype(np.uint8)
        groups = np.array_split(np.arange(length), 3)
        channels = [response[group].max(axis=0) if len(group) else np.zeros_like(frames[0]) for group in groups]
        np.testing.assert_array_equal(actual, np.stack(channels[::-1], axis=-1))
        peak = np.round(response.argmax(axis=0) * 255. / max(1, length - 1)).astype(np.uint8)
        peak[response.max(axis=0) < 3] = 0
        np.testing.assert_array_equal(actual_peak, peak)


def test_histogram_median_matches_partition_for_extremes_and_strided_arrays():
    assert np.isnan(_uint8_median(np.empty(0, dtype=np.uint8)))
    rng = np.random.default_rng(711)
    for count in (1, 2, 3, 256, 257, 131072):
        for values in (rng.integers(0, 256, count, dtype=np.uint8),
                       np.full(count, 255, dtype=np.uint8)):
            assert _uint8_median(values) == float(np.median(values))
            assert _uint8_median(values[::2]) == float(np.median(values[::2]))


def test_integer_deviations_preserve_entire_cube_and_sigma():
    rng = np.random.default_rng(51)
    for length in (1, 2, 3, 24, 25, 50, 101):
        for dtype in (np.uint8, np.float32):
            stack = rng.integers(0, 256, (length, 24, 32), dtype=np.uint8).astype(dtype)
            if dtype == np.float32:
                stack *= .7
            values = stack.astype(np.float32)
            background = np.median(values, axis=0)
            deviation = np.abs(values - background)
            sigma = 1.4826 * np.median(deviation, axis=0)
            global_sigma = (float(np.median(np.abs(np.diff(values, axis=0)))) / .954
                            if length > 1 else float(np.median(deviation)) * 1.4826)
            global_sigma = max(1., global_sigma)
            expected = np.clip(np.maximum(values - background, 0.) /
                               np.maximum(sigma, global_sigma)[None], 0., 30.).astype(np.float32)
            actual, actual_sigma = _robust_residual_cube(stack)
            assert actual_sigma == global_sigma
            np.testing.assert_array_equal(actual, expected)


def test_motion_percentiles_equal_individual_frame_calculations():
    from event_boost_model import extract_motion_features
    original_percentile = np.percentile
    def per_frame(values, q, axis=None, **kwargs):
        if axis == (1, 2):
            return np.array([original_percentile(frame, q, **kwargs) for frame in values])
        return original_percentile(values, q, axis=axis, **kwargs)
    rng = np.random.default_rng(87)
    for count in (1, 2, 25):
        frames = list(rng.integers(0, 256, (count, 128, 128), dtype=np.uint8))
        actual = extract_motion_features(frames)
        with patch.object(np, 'percentile', side_effect=per_frame):
            expected = extract_motion_features(frames)
        np.testing.assert_array_equal(actual, expected)


def test_index_preserves_duplicates_including_out_of_order_bounds_and_margins():
    rng = np.random.default_rng(617)
    saved = []
    indexed = _SavedEventIndex(25)
    for i in range(3000):
        first = int(rng.integers(-40, 25000))
        center = tuple(rng.integers(0, 1920, 2).tolist())
        item = dict(center=center, label='meteor' if i % 3 == 0 else 'not_meteor',
                    detection_start_frame=first, detection_end_frame=first + int(rng.integers(0, 100)),
                    detection_line=(center, (center[0] + 200, center[1] + 5)))
        saved.append(item)
        indexed.append(item)
    # Unusually long events use the bounded-index fallback; old results with
    # only clip bounds remain supported, including negative default times.
    for item in (dict(center=(40, 40), start_frame=-100000, end_frame=100000),
                 dict(center=(500, 600)),
                 dict(center=(500, 600), start_frame=25, end_frame=30)):
        saved.append(item)
        indexed.append(item)
    queries = [(-1, -1), (19, 19), (24, 25), (30, 31), (36, 36), (37, 37),
               (-100000, 100000)]
    queries += [(int(first), int(first) + 60) for first in rng.integers(0, 25000, 100)]
    for bounds in queries:
        for temporal in (False, True):
            for line in (((500, 590), (500, 610)), ((42, 40), (242, 40))):
                assert _candidate_already_saved(line, bounds, indexed, 25, temporal) == \
                       _candidate_already_saved(line, bounds, saved, 25, temporal)
    assert indexed == saved


def test_long_live_history_queries_only_nearby_buckets_in_original_order():
    indexed = _SavedEventIndex(25)
    for i in range(20000):
        indexed.append(dict(center=(50, 60), label='not_meteor',
                            detection_start_frame=i * 25, detection_end_frame=i * 25 + 2))
    nearby = list(indexed.overlapping((499975, 499977), 6))
    assert nearby == indexed[-2:]
    assert _candidate_already_saved(((50, 55), (50, 65)), (499975, 499977), indexed, 25)
    assert not _candidate_already_saved(((50, 55), (50, 65)), (499975, 499977), indexed, 25, True)


def test_event_preprocessing_overlaps_but_accelerator_calls_are_serialized():
    barrier = threading.Barrier(4)
    active = maximum_active = 0
    state_lock = threading.Lock()

    def prepare(**kwargs):
        barrier.wait(timeout=5)
        return np.zeros((5, 16, 16), np.float32), np.zeros((3, 16, 16), np.float32), np.zeros(12, np.float32)

    class Encoder:
        def encode(self, images):
            nonlocal active, maximum_active
            with state_lock:
                active += 1
                maximum_active = max(active, maximum_active)
            time.sleep(.01)
            with state_lock:
                active -= 1
            return np.zeros((1, 4), np.float32)

    class Classifier:
        encoder = Encoder()
        def predict_proba(self, features):
            return np.array([.75])

    with patch.multiple(model, model=Classifier(), _model_ready=True,
                        _active_model_info={'architecture': model.EVENT_TRANSFORMER_ARCHITECTURE_NAME},
                        build_universal_inputs=prepare,
                        extract_event_features=lambda *args: np.zeros(2),
                        extract_motion_features=lambda *args: np.zeros(2)):
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda _: model.predict_meteor_event([object()], None, None, 25), range(4)))
    assert results == [.75] * 4
    assert maximum_active == 1


def test_model_reload_during_preprocessing_keeps_one_revision_per_event():
    old = type('Classifier', (), {'predict_proba': lambda self, _: np.array([.75])})()
    new = type('Classifier', (), {'predict_proba': lambda self, _: np.array([.25])})()
    def prepare(**kwargs):
        with model._model_lock:
            model.model = new
            model._active_model_info = {'architecture': model.UNIVERSAL_ARCHITECTURE_NAME}
        return np.zeros((5, 16, 16), np.float32), np.zeros((3, 16, 16), np.float32), np.zeros(12, np.float32)
    with patch.multiple(model, model=old, _model_ready=True,
                        _active_model_info={'architecture': model.EVENT_BOOST_ARCHITECTURE_NAME},
                        build_universal_inputs=prepare,
                        extract_event_features=lambda *args: np.zeros(2),
                        extract_motion_features=lambda *args: np.zeros(2)):
        assert model.predict_meteor_event([object()], None, None, 25) == .75


def test_prepared_event_keeps_exact_vectors_and_requires_identical_evidence():
    rng = np.random.default_rng(985)
    frames = list(rng.integers(0,256,(8,80,96,3),dtype=np.uint8))
    rect,line = (3,5,72,75),((25,30),(45,50))
    vectors = []
    class Classifier:
        def predict_proba(self, vector):
            vectors.append(vector.copy())
            return np.array([.75])
    architecture = model.EVENT_BOOST_ARCHITECTURE_NAME
    with patch.multiple(model,model=Classifier(),_model_ready=True,
                        _active_model_info={'architecture':architecture}):
        prepared = model.prepare_meteor_inputs(frames,rect,line,25)
        assert prepared.matches(frames,rect,line,25,architecture)
        assert not prepared.matches([frame.copy() for frame in frames],rect,line,25,architecture)
        assert not prepared.matches(frames,rect,line,24,architecture)
        assert not prepared.matches(frames,rect,((26,30),(45,50)),25,architecture)
        assert not prepared.matches(frames,(4,5,72,75),line,25,architecture)
        assert not prepared.matches(frames,rect,line,25,model.UNIVERSAL_ARCHITECTURE_NAME)
        assert model.predict_meteor_event(frames,rect,line,25) == .75
        with patch.object(model,'_prepare_meteor_inputs',side_effect=AssertionError('recomputed')):
            assert model.predict_meteor_event(frames,rect,line,25,prepared=prepared) == .75
        np.testing.assert_array_equal(vectors[0],vectors[1])
        with patch.object(model,'_prepare_meteor_inputs',wraps=model._prepare_meteor_inputs) as prepare:
            assert model.predict_meteor_event([frame.copy() for frame in frames],rect,line,25,prepared=prepared) == .75
            assert prepare.call_count == 1


def test_prepared_event_model_reload_uses_new_weights_and_rechecks_architecture():
    frames = [np.zeros((40,40,3),np.uint8)]*3
    old = type('Classifier',(),{'predict_proba':lambda self,v:np.array([.75])})()
    new = type('Classifier',(),{'predict_proba':lambda self,v:np.array([.25])})()
    with patch.multiple(model,model=old,_model_ready=True,
                        _active_model_info={'architecture':model.EVENT_BOOST_ARCHITECTURE_NAME}):
        prepared = model.prepare_meteor_inputs(frames,None,None,25)
        with model._model_lock:
            model.model = new
        assert model.predict_meteor_event(frames,None,None,25,prepared=prepared) == .25
        model._active_model_info = {'architecture':model.EVENT_TRANSFORMER_ARCHITECTURE_NAME}
        image,kymo,features = prepared.image,prepared.kymograph,prepared.features
        new.encoder = type('Encoder',(),{'encode':lambda self,images:np.zeros((1,4),np.float32)})()
        with patch.object(model,'_prepare_meteor_inputs',wraps=model._prepare_meteor_inputs) as prepare:
            assert model.predict_meteor_event(frames,None,None,25,prepared=prepared) == .25
            assert prepare.call_count == 1
