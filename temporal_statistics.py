"""Exact temporal statistics on contiguous pixel histories."""
import numpy as np
import cv2
import os
import threading
from concurrent.futures import ThreadPoolExecutor

_median_executor = None
_median_executor_lock = threading.Lock()
_median_workers = max(1, min(8, (os.cpu_count() or 1) // 2))


def _executor():
    global _median_executor
    with _median_executor_lock:
        if _median_executor is None:
            _median_executor = ThreadPoolExecutor(max_workers=_median_workers,
                                                  thread_name_prefix='temporal-median')
    return _median_executor


def temporal_median(values):
    """Keep each pixel's time samples adjacent while partitioning a private copy.

    NumPy's axis-0 partition repeatedly copies strided pixel histories. Moving
    time last once reduces that work, with the same middle samples, dtype and
    even-length mean. The caller's chronological cube is never reordered.
    """
    if values.dtype == np.uint8 and 1 <= values.shape[0] <= 32 and values.size <= 1024**2:
        from component_statistics import small_temporal_median
        native = small_temporal_median(values)
        if native is not None:
            return native
    if (values.dtype == np.uint8 and 1 <= values.shape[0] <= 32
            and 8*1024**2 <= values.size <= 64*1024**2 and _median_workers > 1):
        from component_statistics import temporal_median_range
        contiguous = np.ascontiguousarray(values)
        pixels = contiguous[0].size
        step = (pixels+_median_workers-1)//_median_workers
        futures = [_executor().submit(temporal_median_range,contiguous,start,min(pixels,start+step))
                   for start in range(0,pixels,step)]
        native = [future.result() for future in futures]
        if all(value is not None for value in native):
            return np.concatenate(native).reshape(values.shape[1:])
    histories = np.ascontiguousarray(np.moveaxis(values, 0, -1))
    # ascontiguousarray can return the input for one-frame/one-pixel cases.
    if np.shares_memory(histories, values):
        histories = histories.copy()
    if histories.size < 8 * 1024 * 1024 or _median_workers == 1:
        return np.median(histories, axis=-1, overwrite_input=True)
    # Full-resolution coarse windows contain millions of independent pixel
    # histories. Partition disjoint rows on one bounded shared pool; each row
    # keeps exactly the same samples and NumPy median arithmetic. Small model
    # and gate crops stay on their calling worker to avoid nested pool overhead.
    flat = histories.reshape(-1, histories.shape[-1])
    step = (len(flat) + _median_workers - 1) // _median_workers
    futures = [_executor().submit(np.median, flat[start:start+step],
                                      axis=-1, overwrite_input=True)
               for start in range(0, len(flat), step)]
    return np.concatenate([future.result() for future in futures]).reshape(histories.shape[:-1])


def half_integer_percentiles(values, quantiles):
    """Exact linear percentiles for uint8-derived nonnegative half-integers.

    Backgrounds and deviations in the foreground gate lie on this fixed
    [0, 255] grid. Count bins once rather than partitioning large time cubes.
    Float32 histogram counts are exact within this bound on sample count.
    """
    if not values.size or values.size > 2**24:
        return np.percentile(values, quantiles)
    histogram = cv2.calcHist([values.reshape(1, -1)], [0], None, [511], [0., 255.5]).ravel()
    counts = np.cumsum(histogram, dtype=np.int64)
    positions = (values.size - 1) * (np.asarray(quantiles, dtype=np.float64) / 100.)
    ranks = np.floor(positions).astype(np.int64)
    lower = np.searchsorted(counts, ranks + 1) * .5
    upper = np.searchsorted(counts, np.minimum(ranks + 1, values.size - 1) + 1) * .5
    weight = positions - ranks
    difference = upper - lower
    result = lower + difference * weight
    # Match NumPy's linear interpolation arithmetic on both sides of .5.
    np.subtract(upper, difference * (1. - weight), out=result, where=weight >= .5)
    return result
