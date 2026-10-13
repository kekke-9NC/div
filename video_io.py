"""Bounded FFmpeg decoder pools shared by detection and saved-video readers."""
import cv2
from contextlib import contextmanager
from functools import wraps
import threading

_budget_lock = threading.Lock()
_budget_users = 0
_original_threads = None


@contextmanager
def rtsp_opencv_budget():
    """Share a four-thread ceiling across overlapping continuous RTSP runs."""
    global _budget_users, _original_threads
    with _budget_lock:
        if not _budget_users:
            _original_threads = cv2.getNumThreads()
            cv2.setNumThreads(min(4,max(1,_original_threads)))
        _budget_users += 1
    try:
        yield
    finally:
        with _budget_lock:
            _budget_users -= 1
            if not _budget_users:
                cv2.setNumThreads(_original_threads)
                _original_threads = None


def bounded_rtsp_opencv(function):
    @wraps(function)
    def run(*args,**kwargs):
        with rtsp_opencv_budget():
            return function(*args,**kwargs)
    return run


def open_video_capture(source: str, decoder_threads: int = 2) -> cv2.VideoCapture:
    """Open a capture without letting FFmpeg create a CPU-sized pool.

    OpenCV's macOS FFmpeg backend defaults to 16 codec threads for each H.264
    capture on the current machine, plus another helper pool after the first
    read.  Several videos in parallel therefore created around one hundred
    native threads.  ``CAP_PROP_N_THREADS`` is an *open-only* property, so it
    must be supplied to the constructor rather than set afterwards.
    """
    try:
        thread_count = max(1, min(8, int(decoder_threads)))
    except (TypeError, ValueError):
        thread_count = 2

    thread_property = getattr(cv2, "CAP_PROP_N_THREADS", None)
    ffmpeg_backend = getattr(cv2, "CAP_FFMPEG", None)
    if thread_property is not None and ffmpeg_backend is not None:
        try:
            capture = cv2.VideoCapture(
                source,
                ffmpeg_backend,
                [thread_property, thread_count],
            )
            if capture.isOpened():
                return capture
            capture.release()
        except (cv2.error, TypeError, ValueError):
            pass
    return cv2.VideoCapture(source)
