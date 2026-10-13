"""Exact CPU pixel preparation shared by event models and RTSP workers."""
from typing import Optional, Sequence
import cv2
import numpy as np
import threading
import weakref

_frame_origins = {}
_origin_lock = threading.RLock()


def register_mask_pixels(masked, source, mask_revision):
    """Certify only an immutable bitwise-mask result; views/copies inherit none."""
    key = id(masked)
    def discard(reference):
        with _origin_lock:
            current = _frame_origins.get(key)
            if current is not None and current[0] is reference:
                _frame_origins.pop(key,None)
    with _origin_lock:
        _frame_origins[key] = (weakref.ref(masked,discard),source,mask_revision)


def same_frame_pixels(before, after):
    if before is after:
        return True
    with _origin_lock:
        left,right = _frame_origins.get(id(before)),_frame_origins.get(id(after))
        return (left is not None and right is not None and left[0]() is before and right[0]() is after
                and left[1] is right[1] and left[2] == right[2])


def _gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame.astype(np.uint8, copy=False)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



def _crop_square(
    frame: np.ndarray,
    rect: Optional[Sequence[int]],
    size: int = 256,
) -> np.ndarray:
    if rect is None:
        crop = frame
    else:
        x1, y1, x2, y2 = (int(v) for v in rect)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((size, size), dtype=np.uint8)
    # BGR-to-gray is pixel-local. Convert only the requested pixels, retaining
    # the original conversion-before-resize order and exact uint8 values.
    crop = _gray(crop)
    if crop.shape != (size, size):
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return crop



def _uint8_median(values: np.ndarray) -> float:
    """Exact scalar median without partitioning an entire residual cube."""
    if not values.size:
        return float('nan')
    # OpenCV counts uint8 bins without NumPy's full int64 pixel conversion.
    # Its float32 bin counts are exact up to 2**24 samples; use integer
    # bincount for larger cubes so no histogram count can lose precision.
    if values.size <= 2**24:
        histogram = cv2.calcHist([values.reshape(1, -1)], [0], None, [256], [0, 256]).ravel()
    else:
        histogram = np.bincount(values.ravel(), minlength=256)
    counts = np.cumsum(histogram, dtype=np.int64)
    middle = np.searchsorted(counts, [(values.size - 1) // 2 + 1, values.size // 2 + 1])
    return float(middle.sum()) * .5

