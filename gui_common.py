import os
import hashlib
import sys
import time
import threading
import queue
import json
import io
import contextlib
import shutil
import re
import subprocess
import importlib.util
import cv2
import numpy as np
import tkinter as tk
import tkinter_safety  # installs the CPython 3.11 Tcl-shutdown lifecycle guard
import tkinter.font as tkfont
from tkinter import ttk, filedialog, messagebox, Toplevel, Canvas, PanedWindow
from tkinter.scrolledtext import ScrolledText
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk, ImageDraw
from pathlib import Path
from datetime import datetime
from astropy.io import fits
from astropy.wcs import WCS
import concurrent.futures
from typing import List, Dict, Any, Optional
import status_panel
import ui_state
import network_copy
import download_pipeline
import meteor_sky_viewer as msv
import coordinate_manager as coord_mgr
import config
import file_utils
import video_processing
import astrometry
import image_processing
import model
import model_catalog
import ml_review
import observation_time_filter
import folder_source_discovery
import utils
import location_utils
import sun_times
import auto_time_updater
import long_exposure_map
import distortion_correction
import meteor_angle_analysis
import lighten_blend_video
import lighten_blend_image
import timelapse_creator
import video_processor
from tkinter import simpledialog
import chat_gui

STATUS_CALLBACK = None

class _StderrProgressStream(io.TextIOBase):
    def __init__(self, log_callback, passthrough=None):
        super().__init__()
        self._log_callback = log_callback
        self._passthrough = passthrough
        self._buffer = ""
        self._lock = threading.Lock()
        self._last_line = ""
        self._last_emit_time = 0.0

    def _should_log(self, line: str) -> bool:
        s = line.strip()
        if not s:
            return False
        if "UserWarning" in s:
            return True
        if "hf_xet" in s or "Xet Storage is enabled" in s:
            return True
        if s.startswith("Fetching ") and " files" in s:
            return True
        if s.endswith(".safetensors") or ".safetensors:" in s:
            return True
        if ".bin:" in s or "model-" in s:
            return True
        if "%" in s:
            return True
        speed_tokens = ("B/s", "kB/s", "MB/s", "GB/s")
        if any(t in s for t in speed_tokens):
            return True
        return False

    def _emit(self, line: str, partial: bool = False):
        line = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", line)
        line = " ".join(line.strip().split())
        if not self._should_log(line):
            return

        now = time.time()
        force_emit = ("100%" in line) or ("UserWarning" in line) or ("hf_xet" in line)
        min_interval = 0.3 if partial else 0.15
        if not force_emit and (now - self._last_emit_time) < min_interval:
            return
        if line == self._last_line:
            return

        self._last_line = line
        self._last_emit_time = now
        self._log_callback(line)

    def write(self, s):
        if not s:
            return 0
        if self._passthrough is not None:
            try:
                self._passthrough.write(s)
            except Exception:
                pass

        text = s.replace("\r", "\n")
        with self._lock:
            self._buffer += text
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                self._emit(line, partial=False)
            # tqdm は改行なしで更新するため、末尾バッファも進捗行なら随時表示する
            if self._buffer and self._should_log(self._buffer):
                self._emit(self._buffer, partial=True)
        return len(s)

    def flush(self):
        if self._passthrough is not None:
            try:
                self._passthrough.flush()
            except Exception:
                pass
        with self._lock:
            tail = self._buffer
            self._buffer = ""
        if tail:
            self._emit(tail, partial=False)

def imread_with_japanese_path(path):
    """日本語パスに対応した画像読み込み関数

    cv2.imreadは日本語などのマルチバイト文字を含むパスで
    読み込みに失敗することがあるため、np.fromfileとcv2.imdecodeを使用
    """
    try:
        img_array = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return cv2.imread(path)


__all__ = [name for name in globals() if not name.startswith("__")]

