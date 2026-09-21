"""Small advisory process lock shared by the SwiftUI bridge and legacy GUI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, TextIO

try:
    import fcntl
except ImportError:  # pragma: no cover - macOS uses fcntl
    fcntl = None


class AppProcessLock:
    def __init__(self, root: Path) -> None:
        self.path = Path(root).resolve() / ".meteor_detector.lock"
        self._handle: Optional[TextIO] = None

    def acquire(self) -> bool:
        if self._handle is not None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        if fcntl is None:
            self._handle = handle
            return True
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            return False
        self._handle = handle
        return True

    def release(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def __enter__(self) -> "AppProcessLock":
        if not self.acquire():
            raise RuntimeError("Meteor Detector is already running")
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.release()
