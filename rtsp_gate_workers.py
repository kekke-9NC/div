"""Bounded CPU feature workers owned by one continuous RTSP run.

Grayscale crops, bounded shared image cubes and flash state cross the process boundary. Model weights,
accelerator calls, classification decisions and durable acknowledgements stay
with the owner process. Spawn avoids inheriting a live MPS/CUDA context.
"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from multiprocessing import shared_memory
import os
import queue
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


def _event_inputs(crops, motion_crops, rect, line, frame_rate):
    from event_representation import build_universal_from_crops
    from event_boost_model import extract_event_features, extract_motion_features_from_crops
    image, kymograph, physical = build_universal_from_crops(crops,rect,line,frame_rate)
    image = (np.clip(image,0.,1.)*65535.).astype(np.uint16).astype(np.float32)/65535.
    kymograph = (np.clip(kymograph,0.,1.)*65535.).astype(np.uint16).astype(np.float32)/65535.
    vector = np.concatenate([extract_event_features(image,kymograph,physical),
                             extract_motion_features_from_crops(motion_crops,frame_rate)])
    return image,kymograph,physical,vector


_EVENT_OUTPUT_RESERVE = 2*1024**2
_EVENT_SLOT_LIMIT = 32*1024**2


def _shared_event_inputs(name, crop_shape, motion_shape, rect, line, frame_rate):
    memory = shared_memory.SharedMemory(name=name)
    try:
        crop_bytes,motion_bytes = int(np.prod(crop_shape)),int(np.prod(motion_shape))
        offset = (crop_bytes+motion_bytes+7)//8*8
        limit = offset+_EVENT_OUTPUT_RESERVE
        if limit>memory.size:
            raise ValueError('Incomplete event input slot')
        crops = np.ndarray(crop_shape,np.uint8,buffer=memory.buf)
        motion = np.ndarray(motion_shape,np.uint8,buffer=memory.buf,offset=crop_bytes)
        result = []
        for values in _event_inputs(crops,motion,rect,line,frame_rate):
            if values.dtype != np.float32 or offset+values.nbytes>limit:
                raise ValueError('Event output exceeded the bounded shared slot')
            np.copyto(np.ndarray(values.shape,values.dtype,buffer=memory.buf,offset=offset),values)
            result.append((values.shape,values.dtype.str,offset))
            offset += values.nbytes
        return result
    finally:
        memory.close()


def _moving_tracks(name, shape, dtype, options):
    import moving_point_detector
    memory = shared_memory.SharedMemory(name=name)
    try:
        gray = np.ndarray(shape, dtype=np.dtype(dtype), buffer=memory.buf)
        return moving_point_detector.detect_moving_point_tracks(list(gray), _prepared_gray=gray,
                                                                 _component_workers=4, **options)
    finally:
        memory.close()


def _state_views(memory, shape, slot):
    pixels = int(np.prod(shape))
    start = slot * pixels * 9
    background = np.ndarray(shape, np.float32, buffer=memory.buf, offset=start)
    threshold = np.ndarray(shape, np.float32, buffer=memory.buf, offset=start + pixels * 4)
    threshold_u8 = np.ndarray(shape, np.uint8, buffer=memory.buf, offset=start + pixels * 8)
    return background, threshold, threshold_u8


_flash_cache = None


def _flash_tracks(raw_name, shape, dtype, state_name, state, first, final, mask, roi, fps, border):
    global _flash_cache
    from short_flash_detector import ShortFlashDetector
    raw = shared_memory.SharedMemory(name=raw_name)
    checkpoint = shared_memory.SharedMemory(name=state_name)
    try:
        frames = np.ndarray(shape, np.dtype(dtype), buffer=raw.buf)
        cached, _flash_cache = _flash_cache, None
        revision = None if state is None else (state['last_frame'],state['slot'])
        if cached is not None and cached[:2] == (state_name,revision):
            detector = cached[2]
        else:
            detector = ShortFlashDetector(fps, border, component_workers=4)
        if state is not None and (cached is None or cached[:2] != (state_name,revision)):
            background, threshold, threshold_u8 = _state_views(checkpoint, shape[1:3], state['slot'])
            detector.background = background.copy()
            detector.threshold = threshold.copy()
            detector.threshold_u8 = threshold_u8.copy()
            detector.active = state['active']
            detector.last_frame = state['last_frame']
        result = detector.consume(list(frames), first, mask=mask, roi=roi, final=final)
        slot = 1 - state['slot'] if state is not None else 0
        views = _state_views(checkpoint, shape[1:3], slot)
        for destination, source in zip(views, (detector.background, detector.threshold, detector.threshold_u8)):
            np.copyto(destination, source)
        # Dedicated single-process ownership avoids restoring unchanged
        # arrays each window. Never reuse state after an incomplete task.
        _flash_cache = (state_name,(detector.last_frame,slot),detector)
        return result, dict(slot=slot, active=detector.active, last_frame=detector.last_frame)
    finally:
        raw.close()
        checkpoint.close()


class LiveFlashDetector:
    """Serial state transitions with a double-buffered recovery checkpoint."""
    def __init__(self, owner, fps, border):
        from short_flash_detector import ShortFlashDetector
        self.owner, self.fps, self.border = owner, fps, border
        self.local = ShortFlashDetector(fps, border)
        self.raw = self.checkpoint = self.state = None
        self.shape = None
        self.lock = threading.Lock()
        self.local_mode = self.closed = False
        self.pool = None

    def _restore(self, permanent=True):
        if self.state is not None and not self.local_mode:
            values = _state_views(self.checkpoint, self.shape, self.state['slot'])
            self.local.background, self.local.threshold, self.local.threshold_u8 = [value.copy() for value in values]
            self.local.active = self.state['active']
            self.local.last_frame = self.state['last_frame']
        self.local_mode = permanent

    def consume(self, frames, first_frame, mask=None, roi=None, final=False):
        with self.lock:
            if not frames or self.closed or self.local_mode or self.owner._local():
                self._restore()
                return self.local.consume(frames, first_frame, mask=mask, roi=roi, final=final)
            if not self.owner.active:
                if self.state is not None:
                    self._restore(permanent=False)
                    self.state = None
                return self.local.consume(frames, first_frame, mask=mask, roi=roi, final=final)
            last = self.state['last_frame'] if self.state is not None else self.local.last_frame
            offset = max(0, last - first_frame + 1)
            unseen = frames[offset:]
            try:
                shape = frames[0].shape[:2]
                if self.shape is not None and shape != self.shape:
                    raise ValueError('RTSP flash frame dimensions changed')
                self.shape = shape
                stack = np.stack(unseen) if unseen else np.empty((0, *frames[0].shape), frames[0].dtype)
                if self.raw is None or self.raw.size < stack.nbytes:
                    if self.raw is not None:
                        self.raw.close()
                        self.raw.unlink()
                        self.raw = None
                    self.raw = shared_memory.SharedMemory(create=True, size=max(1, stack.nbytes))
                if self.checkpoint is None:
                    self.checkpoint = shared_memory.SharedMemory(create=True, size=int(np.prod(shape)) * 18)
                if self.state is None and self.local.background is not None:
                    for destination, source in zip(_state_views(self.checkpoint, shape, 0),
                            (self.local.background, self.local.threshold, self.local.threshold_u8)):
                        np.copyto(destination, source)
                    self.state = dict(slot=0, active=self.local.active, last_frame=self.local.last_frame)
                target = np.ndarray(stack.shape, stack.dtype, buffer=self.raw.buf)
                np.copyto(target, stack)
                if self.pool is None:
                    self.pool = ProcessPoolExecutor(max_workers=1,
                        mp_context=multiprocessing.get_context('spawn'),initializer=_initialize_worker)
                result, state = self.pool.submit(_flash_tracks,
                    self.raw.name, stack.shape, stack.dtype.str, self.checkpoint.name,
                    self.state, first_frame + offset, final, mask, roi, self.fps, self.border).result()
                self.state = state  # Publish only a fully written checkpoint.
                return result
            except Exception as exc:
                print(f"[RTSP解析] CPU短時間検出を通常経路へ戻します: {exc}")
                self._restore()
                return self.local.consume(frames, first_frame, mask=mask, roi=roi, final=final)

    def close(self):
        with self.lock:
            if self.closed:
                return
            self._restore()
            self.closed = True
            if self.pool is not None:
                self.pool.shutdown(wait=True,cancel_futures=True)
                self.pool = None
            for memory in (self.raw, self.checkpoint):
                if memory is not None:
                    memory.close()
                    memory.unlink()
            self.raw = self.checkpoint = None
            self.state = None
        self.owner._release_flash(self)


class LiveGateFeatures:
    def __init__(self, workers=None):
        # This pool owns both gate features and the more expensive main-event
        # features. Half the logical cores leaves capacity for decoding, the
        # serial chronological proposers and the annotation process.
        self.workers = workers or max(1,min(8,((os.cpu_count() or 1)-1)//2))
        self.pool = None
        self.lock = threading.Lock()
        self.failed = False
        self.closed = False
        self.moving_lock = threading.Lock()
        self.moving_memory = None
        self.moving_pool = None
        self.flash_proposers = []
        self.active = False
        self.quiet_windows = 0
        self.event_slots = queue.Queue(maxsize=self.workers)
        for _ in range(self.workers):
            self.event_slots.put(None)
        self.event_memories = {}

    def activate(self):
        with self.lock:
            if not self.closed:
                self.active, self.quiet_windows = True, 0

    def reset_activity(self):
        with self.lock:
            self.active, self.quiet_windows = False, 0

    def observe(self, candidates):
        with self.lock:
            if self.closed:
                return
            if candidates >= 2:
                self.active, self.quiet_windows = True, 0
            elif self.active:
                self.quiet_windows += 1
                if self.quiet_windows >= 32:
                    self.active = False

    def flash_detector(self, fps, border):
        detector = LiveFlashDetector(self, fps, border)
        with self.lock:
            if not self.closed:
                self.flash_proposers.append(detector)
        return detector

    def _release_flash(self, detector):
        with self.lock:
            if detector in self.flash_proposers:
                self.flash_proposers.remove(detector)

    def _executor(self):
        with self.lock:
            if self.closed:
                raise RuntimeError('RTSP CPU workers are closed')
            if self.pool is None:
                self.pool = ProcessPoolExecutor(max_workers=self.workers,
                    mp_context=multiprocessing.get_context('spawn'), initializer=_initialize_worker)
            return self.pool

    def _local(self):
        unsupported_launcher = getattr(sys, 'frozen', False) and not getattr(sys, '_meteor_cpu_spawn_ready', False)
        return self.workers == 1 or unsupported_launcher or self.failed or self.closed

    def extract(self, frames, rect, frame_rate):
        from event_noise_gate import extract_noise_gate_features
        # A frozen launcher without multiprocessing bootstrap support keeps
        # the established in-process path rather than spawning another GUI.
        if self._local() or not self.active:
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

    def prepare_meteor(self, frames, rect, line, frame_rate):
        if self._local() or not self.active:
            return None
        from event_pixels import _crop_square
        acquired = False
        slot = None
        try:
            # Borrow before materializing input stacks. At most workers slots
            # and their bounded input temporaries can be live simultaneously.
            required = (len(frames)*(256**2+128**2)+7)//8*8+_EVENT_OUTPUT_RESERVE
            if required>_EVENT_SLOT_LIMIT:
                return None
            while not self.closed:
                try:
                    slot = self.event_slots.get(timeout=.1)
                    acquired = True
                    break
                except queue.Empty:
                    pass
            if not acquired or self.closed:
                return None
            # Only exact grayscale model crops cross the process boundary;
            # frame identity and inference ownership remain in the parent.
            crops = np.stack([_crop_square(frame,rect,256) for frame in frames])
            motion = np.stack([_crop_square(frame,rect,128) for frame in frames])
            with self.lock:
                if self.closed:
                    return None
                if slot is None or slot.size<required:
                    if slot is not None:
                        self.event_memories.pop(slot.name)
                        slot.close()
                        slot.unlink()
                        slot = None
                    slot = shared_memory.SharedMemory(create=True,size=required)
                    self.event_memories[slot.name] = slot
            np.copyto(np.ndarray(crops.shape,np.uint8,buffer=slot.buf),crops)
            np.copyto(np.ndarray(motion.shape,np.uint8,buffer=slot.buf,offset=crops.nbytes),motion)
            metadata = self._executor().submit(_shared_event_inputs,slot.name,crops.shape,motion.shape,rect,line,frame_rate).result()
            start = (crops.nbytes+motion.nbytes+7)//8*8
            arrays = []
            if len(metadata)!=4:
                raise ValueError('Incomplete event outputs')
            for shape,dtype,offset in metadata:
                kind = np.dtype(dtype)
                if (kind!=np.dtype(np.float32) or any(part<=0 for part in shape) or offset!=start or offset%4
                        or offset+int(np.prod(shape))*kind.itemsize>required):
                    raise ValueError('Invalid event output layout')
                arrays.append(np.ndarray(shape,kind,buffer=slot.buf,offset=offset).copy())
                start += arrays[-1].nbytes
            return tuple(arrays)
        except Exception as exc:
            self.failed = True
            print(f"[RTSP解析] CPU主分類前処理を通常経路へ戻します: {exc}")
            return None
        finally:
            if acquired:
                self.event_slots.put(slot)

    def _moving_tracks(self, frames, gray_cache, options):
        import moving_point_detector
        if self._local() or not self.active:
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
            if self.moving_pool is None:
                self.moving_pool = ProcessPoolExecutor(max_workers=1,
                    mp_context=multiprocessing.get_context('spawn'),initializer=_initialize_worker)
            return self.moving_pool.submit(_moving_tracks, self.moving_memory.name,
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
            flash_proposers, self.flash_proposers = self.flash_proposers, []
        try:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
        finally:
            # Pool completion does not imply that parents finished copying.
            # Collect every slot only after all borrowers returned it.
            for _ in range(self.workers):
                self.event_slots.get()
            for memory in self.event_memories.values():
                memory.close()
                memory.unlink()
            self.event_memories.clear()
            for detector in flash_proposers:
                detector.close()
            with self.moving_lock:
                if self.moving_pool is not None:
                    self.moving_pool.shutdown(wait=True,cancel_futures=True)
                    self.moving_pool = None
                if self.moving_memory is not None:
                    self.moving_memory.close()
                    self.moving_memory.unlink()
                    self.moving_memory = None
