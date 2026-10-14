"""Exact scheduled sky rendering outside the detector's Python interpreter."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from multiprocessing import shared_memory
import sys
import threading

import numpy as np


def _initialize():
    import cv2
    cv2.setNumThreads(1)


def _render(name, shape, payload, when, masked, opacity):
    from obs_meteor_overlay import sky_overlay
    memory = shared_memory.SharedMemory(name=name)
    try:
        size = int(np.prod(shape))
        frame = np.ndarray(shape,np.uint8,buffer=memory.buf)
        output = np.ndarray(shape,np.uint8,buffer=memory.buf,offset=size)
        mask = np.ndarray(shape[:2],np.uint8,buffer=memory.buf,offset=2*size) if masked else None
        layer = sky_overlay.render_layer(payload,when,mask,opacity)
        alpha = layer[:,:,3:4].astype(np.float32)/255
        np.copyto(output,np.rint(frame*(1-alpha)+layer[:,:,:3][:,:,::-1]*alpha).astype(np.uint8))
        return 'torch' in sys.modules
    finally:
        memory.close()


class AnnotationWorker:
    def __init__(self):
        self.pool = self.memory = None
        self.lock = threading.Lock()
        self.closed = self.failed = False
        self.torch_loaded = None

    def render(self, frame, payload, when, mask, opacity):
        unsupported = getattr(sys,'frozen',False) and not getattr(sys,'_meteor_cpu_spawn_ready',False)
        with self.lock:
            if (self.closed or self.failed or unsupported or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3
                    or mask is not None and (mask.dtype != np.uint8 or mask.ndim != 2)):
                return None
            try:
                size = frame.nbytes
                required = size*2+frame.shape[0]*frame.shape[1]
                if self.memory is None or self.memory.size < required:
                    self._release_memory()
                    self.memory = shared_memory.SharedMemory(create=True,size=required)
                np.copyto(np.ndarray(frame.shape,np.uint8,buffer=self.memory.buf),frame)
                if mask is not None:
                    import cv2
                    np.copyto(np.ndarray(frame.shape[:2],np.uint8,buffer=self.memory.buf,offset=2*size),
                              cv2.resize(mask,frame.shape[1::-1],interpolation=cv2.INTER_NEAREST))
                if self.pool is None:
                    self.pool = ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn'),initializer=_initialize)
                self.torch_loaded = self.pool.submit(_render,self.memory.name,frame.shape,payload,when,mask is not None,opacity).result()
                return np.ndarray(frame.shape,np.uint8,buffer=self.memory.buf,offset=size).copy()
            except Exception as exc:
                self.failed = True
                print(f'[自動作成] 星座描画を通常経路へ戻します: {exc}')
                return None

    def _release_memory(self):
        if self.memory is not None:
            self.memory.close()
            self.memory.unlink()
            self.memory = None

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            try:
                if self.pool is not None:
                    self.pool.shutdown(wait=True,cancel_futures=True)
                    self.pool = None
            finally:
                self._release_memory()
