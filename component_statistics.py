"""Exact optional native aggregation; the NumPy reference remains available."""
import ctypes
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import threading

import numpy as np

_lock = threading.Lock()
_checked = False
_function = None
_area_function = None
_fractional_function = None
_batch_function = None
_median_function = None
_median_range_function = None
_median_i16_function = None


def _library_name():
    return 'component_statistics' + ('.dylib' if sys.platform == 'darwin' else '.dll' if sys.platform == 'win32' else '.so')


def build_library(destination):
    source = Path(__file__).with_name('assets') / 'component_statistics.c'
    compiler = shutil.which('clang') or shutil.which('cc')
    if not compiler:
        raise RuntimeError('C compiler unavailable')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.component_build_', dir=destination.parent) as temporary:
        output = Path(temporary) / _library_name()
        shared = ['-dynamiclib'] if sys.platform == 'darwin' else ['-shared', '-fPIC']
        subprocess.run([compiler, '-O3', '-std=c99', *shared, str(source), '-lm', '-o', str(output)],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        os.replace(output, destination)
    return destination


def _load():
    global _checked, _function, _area_function, _fractional_function, _batch_function, _median_function, _median_range_function, _median_i16_function
    with _lock:
        if _checked:
            return _function
        _checked = True
        try:
            packaged = Path(__file__).with_name('assets') / _library_name()
            if packaged.is_file():
                library = packaged
            elif not getattr(sys, 'frozen', False):
                source = Path(__file__).with_name('assets') / 'component_statistics.c'
                digest = hashlib.sha256(source.read_bytes()).hexdigest()[:24]
                # Cache compiled code separately from all recording/app data.
                directory = Path(tempfile.gettempdir()) / f'meteor_components_{os.getuid() if hasattr(os, "getuid") else "user"}'
                directory.mkdir(mode=0o700, exist_ok=True)
                library = directory / f'{platform.machine()}_{digest}_{_library_name()}'
                if not library.is_file():
                    build_library(library)
            else:
                return None
            loaded = ctypes.CDLL(str(library))
            function = loaded.meteor_component_statistics
            function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                 ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p]
            function.restype = None
            _function = function
            _area_function = loaded.meteor_area_crop_i32
            _area_function.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                      ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                      ctypes.c_int, ctypes.c_void_p]
            _area_function.restype = None
            _fractional_function = loaded.meteor_area_crop_5_2
            _fractional_function.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                             ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
            _fractional_function.restype = None
            _batch_function = loaded.meteor_area_crops
            _batch_function.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
                                       ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
            _batch_function.restype = None
            _median_function = loaded.meteor_temporal_median
            _median_function.argtypes = [ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_void_p]
            _median_function.restype = ctypes.c_int
            _median_range_function = loaded.meteor_temporal_median_range
            _median_range_function.argtypes = [ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t,
                                              ctypes.c_size_t,ctypes.c_size_t,ctypes.c_void_p]
            _median_range_function.restype = ctypes.c_int
            _median_i16_function = getattr(loaded,'meteor_temporal_median_i16',None)
            if _median_i16_function is not None:
                _median_i16_function.argtypes = [ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_void_p]
                _median_i16_function.restype = ctypes.c_int
        except (OSError, AttributeError, RuntimeError, subprocess.SubprocessError):
            # Native acceleration is optional. Never alter component scores
            # or detection availability when compilation/loading is absent.
            _function = None
        return _function


def aggregate(labels, values, eligible, count):
    if values.dtype != np.uint8 or labels.dtype != np.int32 or labels.shape != values.shape:
        return None
    function = _load()
    if function is None:
        return None
    labels, values = np.ascontiguousarray(labels), np.ascontiguousarray(values)
    keep = np.zeros(count, np.uint8)
    keep[eligible] = 1
    sums, peaks = np.zeros(count, np.uint64), np.zeros(count, np.uint8)
    function(labels.ctypes.data, values.ctypes.data, keep.ctypes.data,
             values.size, count, sums.ctypes.data, peaks.ctypes.data)
    return sums, peaks


def area_crop(integral, x, y, sx, sy, size=64):
    _load()
    if _area_function is None or integral.dtype != np.int32 or not integral.flags.c_contiguous:
        return None
    height, width = integral.shape
    if min(x, y) < 0 or min(sx, sy, size) < 1 or x+sx*size >= width or y+sy*size >= height:
        return None
    output = np.empty((size, size), np.uint8)
    _area_function(integral.ctypes.data, width, x, y, sx, sy, size, output.ctypes.data)
    return output


def area_crop_160(integral, x, y):
    _load()
    if _fractional_function is None or integral.dtype != np.int32 or not integral.flags.c_contiguous:
        return None
    height, width = integral.shape
    if min(x, y) < 0 or x+160 >= width or y+160 >= height:
        return None
    output = np.empty((64, 64), np.uint8)
    _fractional_function(integral.ctypes.data, width, x, y, 64, output.ctypes.data)
    return output


def area_crops(integrals, x, y, width, height):
    _load()
    if _batch_function is None or not integrals:
        return None
    shape = integrals[0].shape
    if any(image.dtype != np.int32 or image.shape != shape or not image.flags.c_contiguous for image in integrals):
        return None
    if (min(x,y) < 0 or x+width >= shape[1] or y+height >= shape[0]
            or not (width == height == 160 or min(width,height) >= 64 and width%64 == height%64 == 0)):
        return None
    pointers = (ctypes.c_void_p * len(integrals))(*(image.ctypes.data for image in integrals))
    output = np.empty((len(integrals),64,64),np.uint8)
    _batch_function(pointers,len(integrals),shape[1],x,y,width,height,output.ctypes.data)
    return [image.copy() for image in output]


def small_temporal_median(values):
    if values.dtype not in (np.uint8,np.int16) or not 1 <= values.shape[0] <= 32 or values.size > 1024**2:
        return None
    _load()
    function = _median_function if values.dtype == np.uint8 else _median_i16_function
    if function is None:
        return None
    values = np.ascontiguousarray(values)
    result = np.empty(values.shape[1:],np.float64)
    if not function(values.ctypes.data,len(values),result.size,result.ctypes.data):
        return None
    return result


def temporal_median_range(values,start,end):
    if (values.dtype != np.uint8 or not values.flags.c_contiguous or not 1 <= len(values) <= 32
            or not 0 <= start < end <= values[0].size):
        return None
    _load()
    if _median_range_function is None:
        return None
    result = np.empty(end-start,np.float64)
    if not _median_range_function(values.ctypes.data,len(values),values[0].size,start,end-start,result.ctypes.data):
        return None
    return result
