"""Limit native-operation return latency in the macOS legacy owner process."""
import math
import os
import sys


def configure_native_return_scheduling():
    """Shorten the GIL wait for concurrent native/I/O work, without busy waits.

    A CPU-bound Python thread can hold the GIL for the default five milliseconds
    while each OpenCV/NumPy operation or filesystem call waits to reenter Python.
    The legacy owner has many such threads. Workers and non-macOS launch paths
    keep their existing policy. An already shorter interval is preserved.
    """
    current = sys.getswitchinterval()
    if sys.platform != 'darwin':
        return current
    try:
        requested = float(os.environ.get('METEOR_GIL_SWITCH_INTERVAL', '.001'))
    except ValueError:
        return current
    if not math.isfinite(requested) or not .0005 <= requested <= .005:
        return current
    if requested < current:
        sys.setswitchinterval(requested)
        return requested
    return current
