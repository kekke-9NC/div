"""Bounded FFmpeg decoder pools shared by detection and saved-video readers."""
import cv2


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
