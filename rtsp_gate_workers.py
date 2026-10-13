"""Bounded CPU feature workers owned by one continuous RTSP run.

Only grayscale crops and a bounded shared grayscale cube cross the process boundary. Model weights,
accelerator calls, classification decisions and durable acknowledgements stay
with the owner process. Spawn avoids inheriting a live MPS/CUDA context.
"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from multiprocessing import shared_memory
import os
import sys
import threading

import numpy as np


def _initialize_worker():
    import cv2
    cv2.setNumThreads(1)


def _worker_diagnostics():
    return dict(pid=os.getpid(), torch_loaded='torch' in sys.modules)


def _features(crops, frame_rate):
    from event_noise_gate import extract_noise_gate_features
    return extract_noise_gate_features(list(crops), None, frame_rate)


def _moving_tracks(name, shape, dtype, options):
    import moving_point_detector
    memory = shared_memory.SharedMemory(name=name)
    try:
        gray = np.ndarray(shape, dtype=np.dtype(dtype), buffer=memory.buf)
        return moving_point_detector.detect_moving_point_tracks(list(gray), _prepared_gray=gray, **options)
    finally:
        memory.close()


class LiveGateFeatures:
    def __init__(self, workers=None):
        self.workers = workers or max(1, min(6, (os.cpu_count() or 1) // 3))
        self.pool = None
        self.lock = threading.Lock()
        self.failed = False
        self.closed = False
        self.moving_lock = threading.Lock()
        self.moving_memory = None

    def _executor(self):
        with self.lock:
            if self.closed:
                raise RuntimeError('RTSP CPU workers are closed')
            if self.pool is None:
                self.pool = ProcessPoolExecutor(max_workers=self.workers,
                    mp_context=multiprocessing.get_context('spawn'), initializer=_initialize_worker)
            return self.pool

    def _local(self):
        return self.workers == 1 or getattr(sys, 'frozen', False) or self.failed or self.closed

    def extract(self, frames, rect, frame_rate):
        from event_noise_gate import extract_noise_gate_features
        # A frozen launcher without multiprocessing bootstrap support keeps
        # the established in-process path rather than spawning another GUI.
        if self._local():
            return extract_noise_gate_features(frames, rect, frame_rate)
        from event_pixels import _crop_square
        crops = np.stack([_crop_square(frame, rect, 64) for frame in frames])
        try:
            return self._executor().submit(_features, crops, frame_rate).result()
        except Exception as exc:
            self.failed = True
            print(f"[RTSP解析] CPU一次分類を通常経路へ戻します: {exc}")
            return extract_noise_gate_features(frames, rect, frame_rate)

    def moving_tracks(self, frames, gray_cache=None, **options):
        with self.moving_lock:
            return self._moving_tracks(frames, gray_cache, options)

    def _moving_tracks(self, frames, gray_cache, options):
        import moving_point_detector
        if self._local():
            return moving_point_detector.detect_moving_point_tracks(frames, gray_cache=gray_cache, **options)
        # One chronological proposer owns this slot and waits for its result
        # before reusing it. Full-resolution video and detector state remain
        # in the owner; only the exact downsampled grayscale cube is shared.
        scale = options.get('scale', .5)
        gray = np.stack(gray_cache.convert(frames, scale) if gray_cache is not None else
                        [moving_point_detector._gray_small(frame, scale) for frame in frames])
        try:
            if self.moving_memory is None or self.moving_memory.size < gray.nbytes:
                if self.moving_memory is not None:
                    self.moving_memory.close()
                    self.moving_memory.unlink()
                    self.moving_memory = None
                self.moving_memory = shared_memory.SharedMemory(create=True, size=gray.nbytes)
            target = np.ndarray(gray.shape, dtype=gray.dtype, buffer=self.moving_memory.buf)
            np.copyto(target, gray)
            return self._executor().submit(_moving_tracks, self.moving_memory.name,
                                          gray.shape, gray.dtype.str, options).result()
        except Exception as exc:
            self.failed = True
            print(f"[RTSP解析] CPU候補生成を通常経路へ戻します: {exc}")
            return moving_point_detector.detect_moving_point_tracks(frames, gray_cache=gray_cache, **options)

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            pool, self.pool = self.pool, None
        try:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
        finally:
            with self.moving_lock:
                if self.moving_memory is not None:
                    self.moving_memory.close()
                    self.moving_memory.unlink()
                    self.moving_memory = None
