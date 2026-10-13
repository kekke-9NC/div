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
    workers.activate()
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
    workers.activate()
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
    workers.activate()
    frames = [np.zeros((64, 64), np.uint8)]
    np.testing.assert_array_equal(workers.extract(frames, None, 25),
                                  event_noise_gate.extract_noise_gate_features(frames, None, 25))
    assert workers.pool is None


def test_source_gui_spawn_import_does_not_load_gui_or_torch():
    import subprocess
    from pathlib import Path
    result = subprocess.run([sys.executable,'-c',
        "import runpy,sys; runpy.run_path('main_gui.py',run_name='__mp_main__'); "
        "assert 'torch' not in sys.modules and 'tkinter' not in sys.modules"],
        cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr


def test_supported_frozen_gui_allows_cpu_workers(monkeypatch):
    monkeypatch.setattr(sys,'frozen',True,raising=False)
    monkeypatch.setattr(sys,'_meteor_cpu_spawn_ready',True,raising=False)
    workers = LiveGateFeatures(workers=2)
    assert not workers._local()
    workers.close()


def test_activity_switches_placement_and_resets_for_each_connection():
    workers = LiveGateFeatures(workers=2)
    workers.observe(1)
    assert not workers.active
    workers.observe(2)
    assert workers.active
    for _ in range(31):
        workers.observe(0)
    assert workers.active
    workers.observe(0)
    assert not workers.active
    workers.observe(2)
    workers.reset_activity()
    assert not workers.active and workers.quiet_windows == 0
    workers.close()
    workers.observe(20)
    assert workers.closed


def test_closed_workers_cannot_restart_or_allocate_shared_pixels():
    import moving_point_detector
    workers = LiveGateFeatures(workers=2)
    workers.activate()
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
    workers.activate()
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


def test_stateful_flash_process_matches_all_windows_and_recovers_from_failure():
    import cv2
    from short_flash_detector import ShortFlashDetector
    from rtsp_gate_workers import _state_views
    frames = [np.full((90, 120, 3), 17, np.uint8) for _ in range(130)]
    for index in (23, 24, 65, 66):
        cv2.line(frames[index], (35+index%4, 30), (38+index%4, 46), (180, 180, 180), 2)
    mask = np.full((90, 120), 255, np.uint8)
    mask[:, :12] = 0
    workers = LiveGateFeatures(workers=2)
    workers.activate()
    actual, expected = workers.flash_detector(25, 5), ShortFlashDetector(25, 5)
    try:
        for first in range(0, 78, 6):
            window = frames[first:first+25]
            assert actual.consume(window, first, mask=mask) == expected.consume(window, first, mask=mask)
            for observed, reference in zip(_state_views(actual.checkpoint, (90, 120), actual.state['slot']),
                                           (expected.background, expected.threshold, expected.threshold_u8)):
                np.testing.assert_array_equal(observed, reference)
            assert actual.state['active'] == expected.active
            assert actual.state['last_frame'] == expected.last_frame
        failed = Future()
        failed.set_exception(RuntimeError('worker lost after a valid checkpoint'))
        with patch.object(actual.pool, 'submit', return_value=failed):
            assert actual.consume(frames[78:103], 78, mask=mask) == expected.consume(frames[78:103], 78, mask=mask)
        assert actual.local_mode
        assert actual.consume(frames[84:109], 84, mask=mask, final=True) == expected.consume(frames[84:109], 84, mask=mask, final=True)
        np.testing.assert_array_equal(actual.local.background, expected.background)
        names = [actual.raw.name, actual.checkpoint.name]
        actual.close()
        actual.close()
        assert actual.consume(frames[90:115], 90, final=True) == expected.consume(frames[90:115], 90, final=True)
        assert not workers.flash_proposers
    finally:
        workers.close()
    from multiprocessing import shared_memory
    for name in names:
        try:
            memory = shared_memory.SharedMemory(name=name)
        except FileNotFoundError:
            pass
        else:
            memory.close()
            raise AssertionError('flash checkpoint leaked')


def test_flash_activity_transitions_gaps_and_dimensions_keep_serial_state():
    import cv2
    from short_flash_detector import ShortFlashDetector
    from rtsp_gate_workers import _state_views
    workers = LiveGateFeatures(workers=2)
    actual, expected = workers.flash_detector(25, 5), ShortFlashDetector(25, 5)
    try:
        for number, (first, shape, active) in enumerate(((0,(90,120),False), (6,(90,120),True),
                (12,(90,120),False), (18,(90,120),True), (60,(90,120),True), (66,(96,128),True))):
            if active:
                workers.activate()
            else:
                workers.reset_activity()
            frames = [np.full((*shape,3), 17, np.uint8) for _ in range(25)]
            for i in (0,1,21,22):
                cv2.line(frames[i], (40+i%3,30), (42+i%3,47), (180,180,180), 2)
            assert actual.consume(frames,first,final=number==5) == expected.consume(frames,first,final=number==5)
            if actual.state is not None and not actual.local_mode:
                values = _state_views(actual.checkpoint, actual.shape, actual.state['slot'])
                last, tracks = actual.state['last_frame'], actual.state['active']
            else:
                values = (actual.local.background,actual.local.threshold,actual.local.threshold_u8)
                last, tracks = actual.local.last_frame, actual.local.active
            for a,b in zip(values,(expected.background,expected.threshold,expected.threshold_u8)):
                np.testing.assert_array_equal(a,b)
            assert last == expected.last_frame and tracks == expected.active
    finally:
        workers.close()
