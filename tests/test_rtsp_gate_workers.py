import sys
from concurrent.futures import Future
from unittest.mock import Mock, patch

import numpy as np

import event_noise_gate
import model
from rtsp_gate_workers import LiveGateFeatures, _worker_diagnostics


def test_spawned_cpu_features_match_in_process_every_value():
    rng = np.random.default_rng(4141)
    workers = LiveGateFeatures(workers=2)
    try:
        for length in (1, 2, 7, 8, 25):
            frames = list(rng.integers(0, 256, (length, 64, 64), dtype=np.uint8))
            expected = event_noise_gate.extract_noise_gate_features(frames, None, 25)
            np.testing.assert_array_equal(workers.extract(frames, None, 25), expected)
        assert workers.pool is not None
        import moving_point_detector
        for scale in (.5, 1.):
            frames = list(rng.integers(0, 256, (25, 90, 120, 3), dtype=np.uint8))
            options = dict(scale=scale, frame_rate=25., threshold=8., max_gap=1,
                           max_step=14., max_candidates=20)
            expected = moving_point_detector.detect_moving_point_tracks(frames, **options)
            assert workers.moving_tracks(frames, moving_point_detector.GrayFrameCache(), **options) == expected
        name = workers.moving_memory.name
        assert not workers.pool.submit(_worker_diagnostics).result()['torch_loaded']
    finally:
        workers.close()
    assert workers.pool is None
    from multiprocessing import shared_memory
    try:
        leaked = shared_memory.SharedMemory(name=name)
    except FileNotFoundError:
        pass
    else:
        leaked.close()
        raise AssertionError('shared image slot leaked')


def test_broken_cpu_pool_keeps_exact_local_gate_features():
    frames = [np.full((64, 64), i, np.uint8) for i in range(8)]
    failure = Future()
    failure.set_exception(RuntimeError('worker unavailable'))
    pool = Mock()
    pool.submit.return_value = failure
    workers = LiveGateFeatures(workers=2)
    with patch('rtsp_gate_workers.ProcessPoolExecutor', return_value=pool):
        expected = event_noise_gate.extract_noise_gate_features(frames, None, 25)
        np.testing.assert_array_equal(workers.extract(frames, None, 25), expected)
        np.testing.assert_array_equal(workers.extract(frames, None, 25), expected)
    assert pool.submit.call_count == 1
    assert workers.failed
    workers.close()
    pool.shutdown.assert_called_once_with(wait=True, cancel_futures=True)


def test_frozen_launcher_uses_local_features_without_spawning(monkeypatch):
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    workers = LiveGateFeatures(workers=2)
    frames = [np.zeros((64, 64), np.uint8)]
    np.testing.assert_array_equal(workers.extract(frames, None, 25),
                                  event_noise_gate.extract_noise_gate_features(frames, None, 25))
    assert workers.pool is None


def test_closed_workers_cannot_restart_or_allocate_shared_pixels():
    import moving_point_detector
    workers = LiveGateFeatures(workers=2)
    workers.close()
    workers.close()
    frames = [np.zeros((64, 64), np.uint8)]*8
    with patch('rtsp_gate_workers.ProcessPoolExecutor', side_effect=AssertionError('restarted')), \
         patch('rtsp_gate_workers.shared_memory.SharedMemory', side_effect=AssertionError('allocated')):
        np.testing.assert_array_equal(workers.extract(frames, None, 25),
                                      event_noise_gate.extract_noise_gate_features(frames, None, 25))
        assert workers.moving_tracks(frames) == moving_point_detector.detect_moving_point_tracks(frames)
    assert workers.pool is None and workers.moving_memory is None


def test_shared_pixel_allocation_failure_uses_original_moving_detector():
    import moving_point_detector
    workers = LiveGateFeatures(workers=2)
    frames = [np.zeros((64, 64), np.uint8)]*8
    with patch('rtsp_gate_workers.shared_memory.SharedMemory', side_effect=OSError('out of shared memory')):
        assert workers.moving_tracks(frames) == moving_point_detector.detect_moving_point_tracks(frames)
    assert workers.failed
    workers.close()


def test_gate_revision_is_pinned_while_cpu_features_are_pending(monkeypatch):
    gate = event_noise_gate.EventNoiseGate.__new__(event_noise_gate.EventNoiseGate)
    gate.threshold, gate.brightness_min = .37, 60
    monkeypatch.setattr(model, '_active_noise_gate', gate)
    extractor = Mock()
    def replace(*args):
        model._active_noise_gate = None
        return np.zeros(74, np.float32)
    extractor.extract.side_effect = replace
    assert model.predict_noise_gate([np.zeros((64, 64), np.uint8)], None, 25,
                                    feature_workers=extractor) == (1., .37)
