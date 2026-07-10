"""Fixed-pattern calibration helpers shared by RTSP capture and processing."""

from __future__ import annotations

import cv2
import numpy as np


def apply_fixed_pattern_correction(frame: np.ndarray, correction: np.ndarray | None) -> np.ndarray:
    """Apply a signed, zero-centred fixed-pattern correction to a BGR frame.

    New calibration files contain an ``int16`` correction.  Older files contain
    an unsigned full dark frame; retain their historic direct-subtraction
    behaviour so existing user calibrations remain usable.
    """
    if frame is None or correction is None:
        return frame
    if correction.shape[:2] != frame.shape[:2]:
        correction = cv2.resize(correction, (frame.shape[1], frame.shape[0]),
                                interpolation=cv2.INTER_LINEAR)
    if correction.dtype == np.uint8:
        return cv2.subtract(frame, correction)
    values = frame.astype(np.int16)
    if correction.ndim == 2:
        values -= correction[..., None].astype(np.int16)
    else:
        values -= correction.astype(np.int16)
    return np.clip(values, 0, 255).astype(np.uint8)
