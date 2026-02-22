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

class App(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        
        if sys.platform == 'win32':
            try:
                import ctypes
                myappid = 'mycompany.meteor_detector.gui.1.0'
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
            except (ImportError, AttributeError) as e:
                print(f"警告: タスクバーアイコンIDの設定に失敗しました: {e}")

        self.title(config.GUI_WINDOW_TITLE)
        self.geometry("1280x800")
        
        self.setup_icon()
        self.setup_style()

        self.folder_paths = []
        self.rtsp_urls = []
        self.mask_image = None
        self.plate_solve_mask_image = None
        self.selfcal_mask_image = None
        self.global_wcs_info = None
        self.summary_video_config = [
            {'name': "Composite Image", 'enabled': True, 'duration': 1.0},
            {'name': "Annotated Image", 'enabled': False, 'duration': 2.0},
            {'name': "Full Size Video", 'enabled': True},
            {'name': "Zoom Sequence", 'enabled': False, 'duration': 2.0},
            {'name': "Cutout Video", 'enabled': True}
        ]
        if getattr(sys, 'frozen', False):
            # exeと同じディレクトリに設定ファイルを置く
            base_path = os.path.dirname(sys.executable)
        else:
            base_path = os.path.dirname(os.path.abspath(__file__))
        self.settings_file = os.path.join(base_path, "app_settings.json")
        self.masks_file = os.path.join(base_path, "app_masks.npz")
        self._migrate_legacy_settings_files()

        self.worker_thread = None
        self.rtsp_thread = None
        self.periodic_scan_thread = None
        self.cancel_flag = threading.Event()
        self.progress_queue = queue.Queue()
        self.start_time_gui = None

        self.setup_variables()
        self.setup_ui()
        self.update_start_button_state()
        self.load_settings()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.after(100, self.update_progress)

    def _migrate_legacy_settings_files(self):
        """Move legacy setting files from CWD to app directory if needed."""
        try:
            legacy_settings = os.path.abspath("app_settings.json")
            legacy_masks = os.path.abspath("app_masks.npz")
            target_settings = os.path.abspath(self.settings_file)
            target_masks = os.path.abspath(self.masks_file)

            if legacy_settings != target_settings and os.path.exists(legacy_settings) and not os.path.exists(target_settings):
                os.makedirs(os.path.dirname(target_settings), exist_ok=True)
                shutil.move(legacy_settings, target_settings)
                print(f"設定ファイルを移動しました: {legacy_settings} -> {target_settings}")

            if legacy_masks != target_masks and os.path.exists(legacy_masks) and not os.path.exists(target_masks):
                os.makedirs(os.path.dirname(target_masks), exist_ok=True)
                shutil.move(legacy_masks, target_masks)
                print(f"マスクファイルを移動しました: {legacy_masks} -> {target_masks}")
        except Exception as e:
            print(f"設定ファイル移動の確認中にエラーが発生しました: {e}")

    def check_admin_password(self):
        """Prompt for admin password and return True if correct."""
        # パスワード確認
        password = simpledialog.askstring("管理者認証", "この操作を実行するには管理者パスワードを入力してください:", show='*')
        if not password:
            return False
            
        # SHA-256でハッシュ化して比較 (パスワード: 141421)
        pw_hash = hashlib.sha256(password.encode()).hexdigest()
        admin_hash = "cfb24c91a9b83d9967f5b6a177037f5803abf3c8a8471a62c4fa48ab076434f0"
        
        if pw_hash != admin_hash:
            messagebox.showerror("アクセス拒否", "パスワードが正しくありません。")
            return False
            
        return True

    def setup_icon(self):
        try:
            if getattr(sys, 'frozen', False):
                base_path = sys._MEIPASS
            else:
                base_path = os.path.dirname(os.path.abspath(__file__))
            
            icon_path = os.path.join(base_path, "icon.ico")

            if os.path.exists(icon_path):
                self.iconbitmap(icon_path)
            else:
                print(f"警告: アイコンファイルが見つかりません: {icon_path}")
        except Exception as e:
            print(f"アイコンの設定中にエラーが発生しました: {e}")

    def setup_style(self):
        style = ttk.Style(self)
        style.theme_use('clam')
        
        BG_COLOR = "#2E3F5B"
        FG_COLOR = "#EAEAEA"
        SELECT_BG = "#4A6A9B"
        FRAME_BG = "#263347"
        
        self.configure(background=BG_COLOR)
        
        style.configure(".", background=BG_COLOR, foreground=FG_COLOR, font=('Segoe UI', 10))
        style.configure("TFrame", background=BG_COLOR)
        style.configure("TLabel", background=BG_COLOR, foreground=FG_COLOR)
        style.configure("TButton", background="#4A6A9B", foreground="white", borderwidth=0)
        style.map("TButton", background=[('active', '#5C7DB8')])
        
        # Admin protected button style (Gray)
        style.configure("Gray.TButton", background="#666666", foreground="white", borderwidth=0)
        style.map("Gray.TButton", background=[('active', '#888888')])
        
        style.configure("TNotebook", background=BG_COLOR, borderwidth=0)
        style.configure("TNotebook.Tab", background=BG_COLOR, foreground=FG_COLOR, padding=[10, 5], font=('Segoe UI', 10))
        style.map("TNotebook.Tab", background=[("selected", SELECT_BG)], foreground=[("selected", "white")])
        
        # Spacer tab style: blend into background, no visible border
        style.configure("Spacer.TNotebook.Tab", background=BG_COLOR, foreground=BG_COLOR, borderwidth=0, padding=[10, 5])
        style.map("Spacer.TNotebook.Tab", background=[("disabled", BG_COLOR)], foreground=[("disabled", BG_COLOR)])
        
        style.configure("TLabelframe", background=FRAME_BG, bordercolor=SELECT_BG, padding=10)
        style.configure("TLabelframe.Label", font=('Segoe UI', 11, 'bold'), background=FRAME_BG, foreground=FG_COLOR)

        style.configure("TEntry", fieldbackground="#3A4D6B", foreground=FG_COLOR, insertcolor=FG_COLOR, bordercolor=SELECT_BG)
        style.configure("TSpinbox", fieldbackground="#3A4D6B", foreground=FG_COLOR, insertcolor=FG_COLOR, bordercolor=SELECT_BG)
        
        style.configure("Vertical.TScrollbar", background=BG_COLOR, troughcolor=FRAME_BG, bordercolor=BG_COLOR, arrowcolor=FG_COLOR)
        style.map("Vertical.TScrollbar", background=[('active', SELECT_BG)])

        style.configure("Horizontal.TProgressbar", background=SELECT_BG)

        # Radiobutton style
        style.configure("TRadiobutton", background=BG_COLOR, foreground=FG_COLOR, font=('Segoe UI', 10))
        style.map("TRadiobutton", background=[('active', BG_COLOR), ('disabled', BG_COLOR)], foreground=[('disabled', '#AAAAAA')])

        # Combobox style matching dark theme with Red text
        style.configure("TCombobox", fieldbackground="#3A4D6B", background=BG_COLOR, foreground="#FF0000", arrowcolor=FG_COLOR, bordercolor=SELECT_BG)
        style.map("TCombobox", fieldbackground=[('readonly', '#3A4D6B')], selectbackground=[('readonly', '#3A4D6B')], 
                  foreground=[('readonly', '#FF0000')], selectforeground=[('readonly', '#FF0000')])

    def setup_variables(self):
        self.rtsp_url_var = tk.StringVar()
        self.periodic_scan_var = tk.BooleanVar(value=False)
        self.periodic_time_limit_var = tk.BooleanVar(value=False)
        self.start_hour_var = tk.StringVar(value="17")
        self.start_min_var = tk.StringVar(value="00")
        self.end_hour_var = tk.StringVar(value="07")
        self.end_min_var = tk.StringVar(value="00")
        self.periodic_dir_var = tk.StringVar()
        self.periodic_interval_var = tk.StringVar(value=str(config.DEFAULT_SCAN_INTERVAL))
        self.save_options_vars = {
            k: tk.BooleanVar(value=v) for k, v in {
                'video': config.DEFAULT_SAVE_VIDEO_CLIP, 'cutout': config.DEFAULT_SAVE_CUTOUT_DIFF,
                'full': config.DEFAULT_SAVE_FULL_DIFF, 'composite': config.DEFAULT_SAVE_COMPOSITE,
                'info': config.DEFAULT_SAVE_DETECTION_INFO, 'summary': True,
                'full_video': config.DEFAULT_SAVE_FULL_VIDEO
            }.items()
        }
        self.plate_solve_wcs_path_var = tk.StringVar()
        self.plate_solve_video_path_var = tk.StringVar()
        self.plate_solve_status_var = tk.StringVar(value="プレートソルブ: 未実行")
        self.use_plate_solve_var = tk.BooleanVar(value=True)
        self.concurrency_var = tk.StringVar(value=str(config.DEFAULT_CONCURRENCY))
        self.interval_var = tk.StringVar(value=str(config.DEFAULT_INTERVAL))
        self.duration_var = tk.StringVar(value=str(config.DEFAULT_DURATION))
        self.mask_path_var = tk.StringVar()
        self.apply_mask_var = tk.BooleanVar(value=False)
        self.meteor_save_path_var = tk.StringVar(value=config.DEFAULT_METEOR_SAVE_PATH)
        self.not_meteor_save_path_var = tk.StringVar(value=config.DEFAULT_NOT_METEOR_SAVE_PATH)
        self.selected_model_path_var = tk.StringVar(value=config.MODEL_PATH)
        self.model_meta_info_var = tk.StringVar(value="")
        self.custom_model_paths = []
        # store last-fetched coordinates (display only for now)
        self.current_lat_var = tk.StringVar(value="--")
        self.current_lon_var = tk.StringVar(value="--")
        # analysis tab: list of meteor info files selected for analysis
        self.analysis_files = []
        # analysis window reference and custom points
        self.analysis_window = None
        self.analysis_canvas = None
        self.analysis_cx = None
        self.analysis_cy = None
        self.analysis_pixel_per_deg = None
        # Coordinate manager for custom points
        self.coord_manager = coord_mgr.CoordinateManager()
        self.coord_manager.set_change_callback(self.on_coordinates_changed)
        self.auto_time_updater_enabled_var = tk.BooleanVar(value=False)
        self.auto_updater = auto_time_updater.AutoTimeUpdater()
        self.auto_updater.set_update_callback(self._on_auto_time_update)
        self.auto_updater.set_log_callback(self.append_log)
        self.rtsp_preset_var = tk.StringVar(value="cloudy")  # "clear" or "cloudy"
        self.rtsp_fps_var = tk.StringVar(value=str(config.RTSP_FPS))
        # RTSP time limit for recording (similar to periodic scan)
        self.rtsp_time_limit_var = tk.BooleanVar(value=False)
        self.rtsp_start_hour_var = tk.StringVar(value="17")
        self.rtsp_start_min_var = tk.StringVar(value="00")
        self.rtsp_end_hour_var = tk.StringVar(value="07")
        self.rtsp_end_min_var = tk.StringVar(value="00")
        self.plate_solve_mode_var = tk.StringVar(value="local")
        self.astrometry_api_key_var = tk.StringVar(value="")
        self.video_concat_files = []
        self.video_concat_bitrate_var = tk.StringVar(value="8000k")
        self.video_concat_codec_var = tk.StringVar(value="h264")
        self.video_concat_fps_var = tk.StringVar(value="Auto")
        self.video_concat_safe_mode_var = tk.BooleanVar(value=True) 

        # Advanced settings variables (config.py values)
        # 検出関連
        self.cfg_min_line_length_var = tk.StringVar(value=str(config.MIN_LINE_LENGTH))
        self.cfg_border_size_var = tk.StringVar(value=str(config.BORDER_SIZE))
        self.cfg_duplicate_thresh_var = tk.StringVar(value=str(config.DUPLICATE_DETECTION_THRESHOLD))
        self.cfg_meteor_prob_var = tk.StringVar(value=str(config.METEOR_PROBABILITY_THRESHOLD))
        
        # 詳細検出関連
        self.cfg_finer_window_sec_var = tk.StringVar(value=str(config.FINER_DETECT_WINDOW_SECONDS))
        self.cfg_finer_comp_step_var = tk.StringVar(value=str(config.FINER_COMPOSITE_STEP))
        self.cfg_finer_min_length_var = tk.StringVar(value=str(config.FINER_DETECT_MIN_LENGTH))
        self.cfg_finer_padding_sec_var = tk.StringVar(value=str(config.FINER_DETECT_PADDING_SECONDS))
        self.cfg_finer_cutout_size_var = tk.StringVar(value=str(config.FINER_CUTOUT_SIZE))
        self.cfg_rtsp_scale_lower_var = tk.StringVar(value=str(config.RTSP_SCALE_LOWER))
        self.cfg_rtsp_scale_upper_var = tk.StringVar(value=str(config.RTSP_SCALE_UPPER))
        
        # 飛行機判定関連
        self.cfg_airplane_dur_thresh_var = tk.StringVar(value=str(config.AIRPLANE_DURATION_THRESHOLD))
        self.cfg_airplane_frame_thresh_var = tk.StringVar(value=str(config.AIRPLANE_FRAME_THRESHOLD))
        self.cfg_tracking_dist_thresh_var = tk.StringVar(value=str(config.TRACKING_DISTANCE_THRESHOLD))
        
        # 動画クリップ関連
        self.cfg_max_clip_dur_var = tk.StringVar(value=str(config.MAX_CLIP_DURATION))
        self.cfg_clip_dur_sec_var = tk.StringVar(value=str(config.CLIP_DURATION_SECONDS))
        self.cfg_cutout_size_var = tk.StringVar(value=str(config.CUTOUT_SIZE))
        self.cfg_rtsp_scale_lower_var.trace_add("write", lambda *_: self._sync_rtsp_scale_from_ui())
        self.cfg_rtsp_scale_upper_var.trace_add("write", lambda *_: self._sync_rtsp_scale_from_ui())

    def setup_ui(self):
        main_pane = PanedWindow(self, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, bg="#2E3F5B")
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = ttk.Frame(main_pane, padding=10)
        
        right_frame = ttk.Frame(main_pane, padding=10)
        self.create_info_panel(right_frame)

        self.notebook = ttk.Notebook(left_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_usage = self.create_usage_tab(self.notebook)
        self.tab_source = self.create_source_tab(self.notebook)

        self.tab_settings = self.create_settings_tab(self.notebook)
        self.tab_analysis = self.create_analysis_tab(self.notebook)
        self.tab_chat = chat_gui.create_tab(self.notebook, app=self)
        self.tab_advanced_settings = self.create_advanced_settings_tab(self.notebook)

        self.notebook.add(self.tab_usage, text="使い方")
        self.notebook.add(self.tab_source, text="ソース選択")

        self.notebook.add(self.tab_settings, text="保存設定")
        self.notebook.add(self.tab_analysis, text="解析")
        self.notebook.add(self.tab_chat, text="Chat")
        self.notebook.add(self.tab_advanced_settings, text="⚙️")
        
        # 右ログ領域を少し狭くし、左の設定領域を広く確保する
        main_pane.add(left_frame, width=860, minsize=720)
        main_pane.add(right_frame, width=360, minsize=300)

        def _set_initial_sash():
            try:
                total = main_pane.winfo_width()
                if total <= 0:
                    return
                desired_right = 360
                sash_x = max(720, total - desired_right - 20)
                main_pane.sash_place(0, sash_x, 0)
            except Exception:
                pass

        self.after(120, _set_initial_sash)

    def _sync_rtsp_scale_from_ui(self):
        try:
            lower = float(self.cfg_rtsp_scale_lower_var.get())
            upper = float(self.cfg_rtsp_scale_upper_var.get())
            if lower > 0 and upper > lower:
                config.RTSP_SCALE_LOWER = lower
                config.RTSP_SCALE_UPPER = upper
        except Exception:
            pass


    def create_usage_tab(self, parent):
        frame = ttk.Frame(parent)

        canvas = tk.Canvas(frame, highlightthickness=0, bg="#2E3F5B")
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        def on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind("<Configure>", on_canvas_configure)

        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", on_mousewheel)

        def _unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")

        canvas.bind("<Enter>", _bind_mousewheel)
        canvas.bind("<Leave>", _unbind_mousewheel)

        pad_x = 10
        pad_y = 6
        wrap_w = 520
        base_bg = "#2E3F5B"
        phase_styles = {
            "detect": {
                "border": "#4F77A8",
                "header_bg": "#2F4E74",
                "header_fg": "#EAF4FF",
                "body_bg": "#314765",
                "text_fg": "#EAEAEA",
            },
            "post": {
                "border": "#4E8C72",
                "header_bg": "#2F5A4B",
                "header_fg": "#ECFFF6",
                "body_bg": "#355346",
                "text_fg": "#EAEAEA",
            },
        }

        ttk.Label(
            scrollable_frame,
            text="✨ 流星検出アプリの使い方",
            font=("Arial", 16, "bold"),
            foreground="#87CEEB",
        ).pack(pady=(15, 8), padx=pad_x, anchor="w")

        ttk.Label(
            scrollable_frame,
            text=(
                "各Stepの説明文の中にあるボタンを押すと、該当タブや操作場所へ直接移動します。"
            ),
            justify=tk.LEFT,
            wraplength=wrap_w,
        ).pack(padx=pad_x, pady=(0, 14), anchor="w")

        def make_inline_button(parent_frame, label, command, text_color="#FFD700", button_bg="#3A4D6B"):
            btn = tk.Button(
                parent_frame,
                text=label,
                command=command,
                bg=button_bg,
                fg=text_color,
                activebackground="#4A6A9B",
                activeforeground=text_color,
                font=("Segoe UI", 10, "bold"),
                relief="raised",
                bd=1,
                padx=6,
                pady=1,
                cursor="hand2",
            )
            btn.bind("<Enter>", lambda e, b=btn: b.configure(bg="#4A6A9B"))
            btn.bind("<Leave>", lambda e, b=btn, bg=button_bg: b.configure(bg=bg))
            return btn

        default_font = tkfont.nametofont("TkDefaultFont")
        button_font = tkfont.Font(family="Segoe UI", size=10, weight="bold")

        def _fit_prefix(text, max_px):
            if not text:
                return ""
            if default_font.measure(text) <= max_px:
                return text
            lo, hi = 1, len(text)
            best = 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if default_font.measure(text[:mid]) <= max_px:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            return text[:best]

        def add_inline_line(parent_frame, segments, section_bg, text_fg):
            max_line_px = wrap_w - 24

            def new_row():
                r = tk.Frame(parent_frame, bg=section_bg)
                r.pack(fill=tk.X, padx=10, pady=(0, 6), anchor="w")
                return r

            row = new_row()
            used_px = 0

            for seg in segments:
                if not seg:
                    continue
                kind = seg[0]

                if kind == "button":
                    label = seg[1]
                    command = seg[2]
                    color = seg[3] if len(seg) >= 4 else "#FFD700"
                    btn = make_inline_button(row, label, command, color, button_bg="#3A4D6B")
                    btn.update_idletasks()
                    req_px = btn.winfo_reqwidth() + 10
                    if used_px > 0 and used_px + req_px > max_line_px:
                        btn.destroy()
                        row = new_row()
                        used_px = 0
                        btn = make_inline_button(row, label, command, color, button_bg="#3A4D6B")
                        btn.update_idletasks()
                        req_px = btn.winfo_reqwidth() + 10
                    btn.pack(side=tk.LEFT, padx=4)
                    used_px += req_px
                    continue

                if kind == "text":
                    text = seg[1]
                    while text:
                        remain_px = max_line_px - used_px
                        if remain_px < 40 and used_px > 0:
                            row = new_row()
                            used_px = 0
                            remain_px = max_line_px

                        if default_font.measure(text) <= remain_px:
                            tk.Label(row, text=text, bg=section_bg, fg=text_fg).pack(side=tk.LEFT)
                            used_px += default_font.measure(text) + 6
                            text = ""
                        else:
                            part = _fit_prefix(text, remain_px)
                            if not part:
                                row = new_row()
                                used_px = 0
                                continue
                            tk.Label(row, text=part, bg=section_bg, fg=text_fg).pack(side=tk.LEFT)
                            used_px += default_font.measure(part) + 6
                            text = text[len(part):]
                            if text:
                                row = new_row()
                                used_px = 0

        def add_phase_separator(text):
            sep = tk.Frame(scrollable_frame, bg=base_bg)
            sep.pack(fill=tk.X, padx=pad_x, pady=(10, 8))
            tk.Frame(sep, bg="#6FB792", height=2).pack(fill=tk.X, pady=(0, 4))
            tk.Label(
                sep,
                text=text,
                bg=base_bg,
                fg="#9DE3BC",
                font=("Segoe UI", 10, "bold"),
            ).pack(anchor="center")
            tk.Frame(sep, bg="#6FB792", height=2).pack(fill=tk.X, pady=(4, 0))

        def add_section(title, lines, phase="detect"):
            style = phase_styles["detect"] if phase not in phase_styles else phase_styles[phase]
            section_outer = tk.Frame(
                scrollable_frame,
                bg=style["border"],
                highlightthickness=1,
                highlightbackground=style["border"],
            )
            section_outer.pack(fill=tk.X, padx=pad_x, pady=pad_y)

            header = tk.Label(
                section_outer,
                text=title,
                anchor="w",
                bg=style["header_bg"],
                fg=style["header_fg"],
                font=("Segoe UI", 11, "bold"),
                padx=8,
                pady=5,
            )
            header.pack(fill=tk.X, padx=1, pady=(1, 0))

            body = tk.Frame(section_outer, bg=style["body_bg"])
            body.pack(fill=tk.X, padx=1, pady=(0, 1))
            for line in lines:
                add_inline_line(body, line, section_bg=style["body_bg"], text_fg=style["text_fg"])

        add_section(
            "Step 0: 最短で動かす手順 🚀",
            [
                [("text", "最初に "), ("button", "ソース選択へ", self.navigate_to_source_drop_area, "#FFD700"), ("text", " で入力データを追加します。")],
                [("text", "次に "), ("button", "保存設定へ", self.navigate_to_settings_tab, "#FFD700"), ("text", " で保存先・保存項目を確認します。")],
                [("text", "準備ができたら "), ("button", "開始ボタン", self.navigate_to_start_button, "#90EE90"), ("text", " を押して処理を開始します。")],
            ],
            phase="detect",
        )

        add_section(
            "Step 1: 入力ソースの準備（動画 / RTSP / 定期スキャン） 📂",
            [
                [("text", "動画を使う場合は "), ("button", "ドロップ領域へ", self.navigate_to_source_drop_area, "#FFD700"), ("text", " から追加します。")],
                [("text", "RTSPを使う場合は "), ("button", "RTSP入力欄", self.navigate_to_rtsp_entry, "#87CEEB"), ("text", " にURLを入力し、"), ("button", "RTSP追加ボタン", self.navigate_to_rtsp_add_button, "#87CEEB"), ("text", " を押します。")],
                [("text", "定期運用する場合は "), ("button", "定期スキャン設定", self.navigate_to_periodic_scan_section, "#FFD700"), ("text", " を有効化し、"), ("button", "監視フォルダ選択", self.navigate_to_periodic_dir_button, "#FFD700"), ("text", " を設定します。")],
            ],
            phase="detect",
        )

        add_section(
            "Step 2: 保存・検出・座標設定（マスク / プレートソルブ / APIキー） ⚙️",
            [
                [("text", "保存設定は "), ("button", "保存設定へ", self.navigate_to_settings_tab, "#FFD700"), ("text", " で行います。")],
                [("text", "誤検出が多い場合は "), ("button", "検出マスク", self.navigate_to_detection_mask_button, "#FFD700"), ("text", " を作成して調整します。")],
                [("text", "プレートソルブを使う場合は "), ("button", "PS動画の選択", self.navigate_to_plate_solve_select_video_button, "#87CEEB"), ("text", " の後に "), ("button", "プレートソルブ実行", self.navigate_to_plate_solve_run_button, "#87CEEB"), ("text", " を押します。")],
                [("text", "プレートソルブの視野角は "), ("button", "プレートソルブ時の視野角設定", self.navigate_to_plate_solve_fov_settings, "#87CEEB"), ("text", " で設定してください。設定した値が正しくない場合、プレートソルブがうまくいかない場合があります。")],
                [("text", "API利用時は "), ("button", "API Key入力欄", self.navigate_to_api_key_entry, "#87CEEB"), ("text", " を設定します。")],
                [("text", "準備ができたら "), ("button", "開始ボタン", self.navigate_to_start_button, "#90EE90"), ("text", " を押して処理を開始します。")],
            ],
            phase="detect",
        )

        add_phase_separator("ここから後処理ステップ（検出後に使う機能）")

        add_section(
            "Step 3: 解析タブ（画像/動画生成） 🧪",
            [
                [("text", "まず "), ("button", "解析タブへ", self.navigate_to_analysis_tab, "#90EE90"), ("text", " を開きます。")],
                [("text", "静止画を作るときは "), ("button", "比較明合成画像", self.navigate_to_blend_image_button, "#FFD700"), ("text", " を使います。")],
                [("text", "動画を作るときは "), ("button", "比較明合成動画", self.navigate_to_blend_video_button, "#FFD700"), ("text", " を使います。")],
                [("text", "時間変化を確認したいときは "), ("button", "タイムラプス", self.navigate_to_timelapse_button, "#FFD700"), ("text", " を使います。")],
            ],
            phase="post",
        )

        add_section(
            "Step 4: 動画連結（複数動画を一本化） 🔗",
            [
                [("text", "連結対象の動画は解析タブ内の動画連結エリアへドラッグ＆ドロップします（必要ならファイル追加も可）。")],
                [("text", "設定後、"), ("button", "連結開始ボタン", self.navigate_to_video_concat_start_button, "#FFD700"), ("text", " で1本に連結します。")],
            ],
            phase="post",
        )

        add_section(
            "Step 5: 困ったときの確認先 💬",
            [
                [("text", "処理状況の確認は "), ("button", "ログ確認", self.navigate_to_log_tab, "#FFD700"), ("text", " と "), ("button", "処理状況", self.navigate_to_processing_status_tab, "#FFD700"), ("text", " を使います。")],
                [("text", "操作手順に迷った場合は "), ("button", "Chatへ", self.navigate_to_chat_tab, "#FFD700"), ("text", " で質問できます。")],
            ],
            phase="post",
        )

        add_section(
            "Step 6: モデル学習と切替 🧠",
            [
                [("text", "推論に使うCNNは "), ("button", "流星分類モデル選択", self.navigate_to_model_selector, "#87CEEB"), ("text", " から切り替えます。")],
                [("text", "独自モデルを作る場合は "), ("button", "機械学習モデル作成", self.navigate_to_model_training_button, "#FFD700"), ("text", " から学習画面を開きます。")],
            ],
            phase="post",
        )

        ttk.Label(
            scrollable_frame,
            text="迷った場合は Step 0 から順に進めてください。",
            justify=tk.LEFT,
            wraplength=wrap_w,
            foreground="#AAAAAA",
        ).pack(padx=pad_x, pady=(8, 14), anchor="w")

        return frame

    def create_source_tab(self, parent):
        frame = ttk.Frame(parent)
        # スクロール可能なキャンバスとスクロールバーを作成
        canvas = tk.Canvas(frame, highlightthickness=0, bg="#2E3F5B")
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        # キャンバスのリサイズ時に内部フレームの幅を合わせる
        def on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind("<Configure>", on_canvas_configure)

        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # マウスホイールでスクロール
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        def _bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        def _unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")

        canvas.bind("<Enter>", _bind_mousewheel)
        canvas.bind("<Leave>", _unbind_mousewheel)
        
        # ===== ここから内部ウィジェット =====
        # Note: pack()の親は scrollable_frame にする
        
        lf_folder = ttk.LabelFrame(scrollable_frame, text="フォルダ / 動画ファイル")
        lf_folder.pack(fill=tk.X, expand=True, pady=5)
        
        self.source_drop_label = ttk.Label(lf_folder, text="ここにフォルダや動画ファイルをドラッグ＆ドロップ", relief=tk.SOLID, padding=20, anchor=tk.CENTER, borderwidth=1)
        self.source_drop_label.pack(fill=tk.X, pady=5)
        self.source_drop_label.drop_target_register(DND_FILES)
        self.source_drop_label.dnd_bind('<<Drop>>', self.drop)

        self.source_drop_label._original_bg = None

        # Folder list (styled)
        list_container = ttk.Frame(lf_folder)
        list_container.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 内側のリスト用のキャンバス（スクロールイベントの競合に注意）
        self.folder_list_canvas = tk.Canvas(list_container, bg="#3A4D6B", highlightthickness=0, height=120)
        self.folder_list_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        inner_scrollbar = ttk.Scrollbar(list_container, orient=tk.VERTICAL, command=self.folder_list_canvas.yview)
        inner_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.folder_list_canvas.configure(yscrollcommand=inner_scrollbar.set)
        
        self.folder_list_frame = tk.Frame(self.folder_list_canvas, bg="#3A4D6B")
        self.folder_list_window = self.folder_list_canvas.create_window((0, 0), window=self.folder_list_frame, anchor="nw")
        
        def on_frame_configure(event):
            self.folder_list_canvas.configure(scrollregion=self.folder_list_canvas.bbox("all"))
        self.folder_list_frame.bind("<Configure>", on_frame_configure)
        
        def on_inner_canvas_configure(event):
            self.folder_list_canvas.itemconfig(self.folder_list_window, width=event.width)
        self.folder_list_canvas.bind("<Configure>", on_inner_canvas_configure)
        
        # 内側スクロール: 親と競合しないようローカルでbindする
        
        def on_inner_mousewheel(event):
            self.folder_list_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            # イベント伝播を止めたいが、Tkinter bindでは return "break" する必要がある
            return "break"

        self.folder_list_canvas.bind("<MouseWheel>", on_inner_mousewheel)
        self.folder_list_frame.bind("<MouseWheel>", on_inner_mousewheel)
        
        # Store item frames for selection
        self.folder_item_frames = []
        self.folder_selected_indices = set()

        btn_frame = ttk.Frame(lf_folder)
        btn_frame.pack(fill=tk.X, pady=(5,0))
        ttk.Button(btn_frame, text="選択項目を削除", command=self.remove_selected_folders).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="すべて削除", command=self.remove_all_folders).pack(side=tk.LEFT, padx=2)

        lf_rtsp = ttk.LabelFrame(scrollable_frame)
        lf_rtsp.pack(fill=tk.X, expand=True, pady=5)
        
        # RTSPストリームのタイトル行にiボタンを追加
        rtsp_title_frame = ttk.Frame(lf_rtsp)
        rtsp_title_frame.pack(fill=tk.X, anchor=tk.W)
        ttk.Label(rtsp_title_frame, text="RTSPストリーム", font=("", 9, "bold")).pack(side=tk.LEFT)
        
        rtsp_info_label = ttk.Label(rtsp_title_frame, text=" ⓘ ", font=("Arial", 9), foreground="#87CEEB", cursor="hand2")
        rtsp_info_label.pack(side=tk.LEFT)
        
        rtsp_info_text = "外部GPUが無い場合はCPUの負荷が高くなり\n映像が乱れることがあります。"
        rtsp_info_label._tooltip = None
        rtsp_info_label._tooltip_hover = False
        
        def show_rtsp_tooltip(event):
            if rtsp_info_label._tooltip is not None:
                return
            tooltip = tk.Toplevel(self)
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root + 10}+{event.y_root + 10}")
            tooltip.configure(bg="#2E3F5B")
            frame_tt = ttk.Frame(tooltip, padding=8)
            frame_tt.pack()
            ttk.Label(frame_tt, text=rtsp_info_text, justify=tk.LEFT).pack()
            
            def on_tooltip_enter(e):
                rtsp_info_label._tooltip_hover = True
            def on_tooltip_leave(e):
                rtsp_info_label._tooltip_hover = False
                self.after(100, check_rtsp_tooltip)
            
            tooltip.bind("<Enter>", on_tooltip_enter)
            tooltip.bind("<Leave>", on_tooltip_leave)
            rtsp_info_label._tooltip = tooltip
        
        def check_rtsp_tooltip():
            if rtsp_info_label._tooltip and not rtsp_info_label._tooltip_hover:
                try:
                    rtsp_info_label._tooltip.destroy()
                except:
                    pass
                rtsp_info_label._tooltip = None
        
        def hide_rtsp_tooltip(event):
            self.after(150, check_rtsp_tooltip)
        
        rtsp_info_label.bind("<Enter>", show_rtsp_tooltip)
        rtsp_info_label.bind("<Leave>", hide_rtsp_tooltip)
        
        entry_frame = ttk.Frame(lf_rtsp)
        entry_frame.pack(fill=tk.X)
        ttk.Label(entry_frame, text="URL:").pack(side=tk.LEFT, padx=(0,5))
        self.rtsp_url_entry = ttk.Entry(entry_frame, textvariable=self.rtsp_url_var)
        self.rtsp_url_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Label(entry_frame, text="FPS:").pack(side=tk.LEFT, padx=(10, 5))
        fps_spin = ttk.Spinbox(entry_frame, from_=1, to=120, increment=1, width=5, textvariable=self.rtsp_fps_var)
        fps_spin.pack(side=tk.LEFT, padx=(0, 5))
        
        self.btn_add_rtsp = ttk.Button(entry_frame, text="追加", command=self.add_rtsp_url)
        self.btn_add_rtsp.pack(side=tk.LEFT, padx=(5,0))
        
        # RTSP list (styled)
        rtsp_list_container = ttk.Frame(lf_rtsp)
        rtsp_list_container.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.rtsp_list_canvas = tk.Canvas(rtsp_list_container, bg="#3A4D6B", highlightthickness=0, height=60)
        self.rtsp_list_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        rtsp_scrollbar = ttk.Scrollbar(rtsp_list_container, orient=tk.VERTICAL, command=self.rtsp_list_canvas.yview)
        rtsp_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.rtsp_list_canvas.configure(yscrollcommand=rtsp_scrollbar.set)
        
        self.rtsp_list_frame = tk.Frame(self.rtsp_list_canvas, bg="#3A4D6B")
        self.rtsp_list_window = self.rtsp_list_canvas.create_window((0, 0), window=self.rtsp_list_frame, anchor="nw")
        
        def on_rtsp_frame_configure(event):
            self.rtsp_list_canvas.configure(scrollregion=self.rtsp_list_canvas.bbox("all"))
        self.rtsp_list_frame.bind("<Configure>", on_rtsp_frame_configure)
        
        def on_rtsp_canvas_configure(event):
            self.rtsp_list_canvas.itemconfig(self.rtsp_list_window, width=event.width)
        self.rtsp_list_canvas.bind("<Configure>", on_rtsp_canvas_configure)
        
        def on_rtsp_inner_mousewheel(event):
            self.rtsp_list_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            return "break"
            
        self.rtsp_list_canvas.bind("<MouseWheel>", on_rtsp_inner_mousewheel)
        self.rtsp_list_frame.bind("<MouseWheel>", on_rtsp_inner_mousewheel)
        
        self.rtsp_item_frames = []
        self.rtsp_selected_indices = set()
        
        rtsp_btn_frame = ttk.Frame(lf_rtsp)
        rtsp_btn_frame.pack(fill=tk.X, pady=(5,0))
        ttk.Button(rtsp_btn_frame, text="選択項目を削除", command=self.remove_selected_rtsp).pack(side=tk.LEFT, padx=2)
        ttk.Button(rtsp_btn_frame, text="すべて削除", command=self.remove_all_rtsp).pack(side=tk.LEFT, padx=2)
        self.btn_rtsp_plate_solve = ttk.Button(rtsp_btn_frame, text="RTSPからプレートソルブ", command=self.start_rtsp_plate_solve)
        self.btn_rtsp_plate_solve.pack(side=tk.LEFT, padx=(10, 2))
        self.btn_rtsp_mask = ttk.Button(rtsp_btn_frame, text="RTSPからマスク作成", command=self.create_rtsp_mask)
        self.btn_rtsp_mask.pack(side=tk.LEFT, padx=2)
        
        rtsp_time_frame = ttk.Frame(lf_rtsp)
        rtsp_time_frame.pack(fill=tk.X, pady=(8,0))
        
        rtsp_time_row1 = ttk.Frame(rtsp_time_frame)
        rtsp_time_row1.pack(fill=tk.X)
        ttk.Checkbutton(rtsp_time_row1, text="録画時間制限を有効にする", variable=self.rtsp_time_limit_var, command=self.toggle_rtsp_time_limit_frame).pack(side=tk.LEFT, anchor=tk.W)
        ttk.Button(rtsp_time_row1, text="自動で設定", command=self.fetch_current_location_rtsp).pack(side=tk.LEFT, padx=(8,0))
        
        self.rtsp_time_limit_detail_frame = ttk.Frame(rtsp_time_frame)
        
        rtsp_start_frame = ttk.Frame(self.rtsp_time_limit_detail_frame)
        rtsp_start_frame.pack(fill=tk.X, pady=2)
        ttk.Label(rtsp_start_frame, text="開始時刻:", width=10).pack(side=tk.LEFT)
        ttk.Spinbox(rtsp_start_frame, from_=0, to=23, width=3, textvariable=self.rtsp_start_hour_var, format="%02.0f").pack(side=tk.LEFT)
        ttk.Label(rtsp_start_frame, text=":").pack(side=tk.LEFT)
        ttk.Spinbox(rtsp_start_frame, from_=0, to=59, width=3, textvariable=self.rtsp_start_min_var, format="%02.0f").pack(side=tk.LEFT)
        
        rtsp_end_frame = ttk.Frame(self.rtsp_time_limit_detail_frame)
        rtsp_end_frame.pack(fill=tk.X, pady=2)
        ttk.Label(rtsp_end_frame, text="終了時刻:", width=10).pack(side=tk.LEFT)
        ttk.Spinbox(rtsp_end_frame, from_=0, to=23, width=3, textvariable=self.rtsp_end_hour_var, format="%02.0f").pack(side=tk.LEFT)
        ttk.Label(rtsp_end_frame, text=":").pack(side=tk.LEFT)
        ttk.Spinbox(rtsp_end_frame, from_=0, to=59, width=3, textvariable=self.rtsp_end_min_var, format="%02.0f").pack(side=tk.LEFT)
        
        # 録画時間外でも解析は継続する旨の説明
        ttk.Label(self.rtsp_time_limit_detail_frame, text="※録画終了後も、保存済み動画の解析は継続します", foreground="#87CEEB").pack(anchor=tk.W, pady=(2,0))
        
        self.toggle_rtsp_time_limit_frame()
        
        # ===== 定期スキャン (移設) =====
        lf_periodic = ttk.LabelFrame(scrollable_frame, text="定期スキャン (監視フォルダ)")
        lf_periodic.pack(fill=tk.X, expand=True, pady=5)

        # Header frame for Checkbutton + Help
        header_frame = ttk.Frame(lf_periodic)
        header_frame.pack(fill=tk.X, anchor=tk.W)
        
        self.chk_periodic_scan = ttk.Checkbutton(header_frame, text="定期スキャンを有効にする", variable=self.periodic_scan_var, command=self.update_start_button_state)
        self.chk_periodic_scan.pack(side=tk.LEFT)
        
        help_label = ttk.Label(header_frame, text=" ? ", font=("Arial", 10, "bold"), foreground="#87CEEB", cursor="hand2")
        help_label.pack(side=tk.LEFT, padx=5)
        
        help_text = """指定した監視フォルダを一定間隔でスキャンし、
新しいファイルを自動的に解析する機能です。

atomcam2で利用する場合は、GitHubで公開されている
「atomcam_tools」を利用してください。
その際、ネットワークフォルダー設定でatomcam2の
データ保存先フォルダを指定する必要があります。"""

        help_label._tooltip = None
        help_label._tooltip_hover = False
        
        def show_periodic_tooltip(event):
            if help_label._tooltip is not None: return
            tooltip = tk.Toplevel(self)
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            tooltip.configure(bg="#2E3F5B")
            f = ttk.Frame(tooltip, padding=8)
            f.pack()
            ttk.Label(f, text=help_text, justify=tk.LEFT, foreground="#EAEAEA", background="#2E3F5B").pack()
            
            def on_enter(e): help_label._tooltip_hover = True
            def on_leave(e): 
                help_label._tooltip_hover = False
                self.after(100, check_periodic_tooltip)
                
            tooltip.bind("<Enter>", on_enter)
            tooltip.bind("<Leave>", on_leave)
            help_label._tooltip = tooltip

        def check_periodic_tooltip():
            if help_label._tooltip and not help_label._tooltip_hover:
                try: help_label._tooltip.destroy()
                except: pass
                help_label._tooltip = None

        def hide_periodic_tooltip(event):
            self.after(150, check_periodic_tooltip)

        help_label.bind("<Enter>", show_periodic_tooltip)
        help_label.bind("<Leave>", hide_periodic_tooltip)
        
        dir_frame = ttk.Frame(lf_periodic)
        dir_frame.pack(fill=tk.X, pady=5)
        ttk.Label(dir_frame, text="監視フォルダ:").pack(side=tk.LEFT, padx=(0,5))
        ttk.Entry(dir_frame, textvariable=self.periodic_dir_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.btn_select_periodic_dir = ttk.Button(dir_frame, text="選択", command=self.select_periodic_dir)
        self.btn_select_periodic_dir.pack(side=tk.LEFT, padx=(5,0))
        
        interval_frame = ttk.Frame(lf_periodic)
        interval_frame.pack(fill=tk.X, pady=5)
        ttk.Label(interval_frame, text="スキャン間隔 (秒):").pack(side=tk.LEFT)
        ttk.Entry(interval_frame, textvariable=self.periodic_interval_var, width=5).pack(side=tk.LEFT)

        lf_time = ttk.LabelFrame(scrollable_frame, text="時間制限 (定期スキャン用)")
        lf_time.pack(fill=tk.X, expand=True, pady=5)
        
        row_frame = ttk.Frame(lf_time)
        row_frame.pack(fill=tk.X)
        self.chk_time_limit = ttk.Checkbutton(row_frame, text="時間制限を有効にする", variable=self.periodic_time_limit_var, command=self.toggle_time_limit_frame)
        self.chk_time_limit.pack(side=tk.LEFT, anchor=tk.W)
        self.btn_periodic_auto_time = ttk.Button(row_frame, text="自動で設定", command=self.fetch_current_location)
        self.btn_periodic_auto_time.pack(side=tk.LEFT, padx=(8,0))
        ttk.Checkbutton(row_frame, text="自動更新を有効にする", variable=self.auto_time_updater_enabled_var, command=self.toggle_auto_time_updater).pack(side=tk.LEFT, padx=(8,0))
        
        self.time_limit_frame = ttk.Frame(lf_time)
        
        start_frame = ttk.Frame(self.time_limit_frame)
        start_frame.pack(fill=tk.X, pady=2)
        ttk.Label(start_frame, text="開始時刻:", width=10).pack(side=tk.LEFT)
        ttk.Spinbox(start_frame, from_=0, to=23, width=3, textvariable=self.start_hour_var, format="%02.0f").pack(side=tk.LEFT)
        ttk.Label(start_frame, text=":").pack(side=tk.LEFT)
        ttk.Spinbox(start_frame, from_=0, to=59, width=3, textvariable=self.start_min_var, format="%02.0f").pack(side=tk.LEFT)
        
        end_frame = ttk.Frame(self.time_limit_frame)
        end_frame.pack(fill=tk.X, pady=2)
        ttk.Label(end_frame, text="終了時刻:", width=10).pack(side=tk.LEFT)
        ttk.Spinbox(end_frame, from_=0, to=23, width=3, textvariable=self.end_hour_var, format="%02.0f").pack(side=tk.LEFT)
        ttk.Label(end_frame, text=":").pack(side=tk.LEFT)
        ttk.Spinbox(end_frame, from_=0, to=59, width=3, textvariable=self.end_min_var, format="%02.0f").pack(side=tk.LEFT)

        self.toggle_time_limit_frame()

        return frame

    def navigate_to_source_drop_area(self):
        """Navigate to Source Selection tab and highlight the drop area for a few seconds."""
        self.notebook.select(self.tab_source)

        style = ttk.Style()
        style.configure("Highlight.TLabel", background="#FFD700", foreground="#000000")
        self.source_drop_label.configure(style="Highlight.TLabel")

        def flash_highlight(count=0):
            if count >= 6:
                self.source_drop_label.configure(style="TLabel")
                return
            if count % 2 == 0:
                style.configure("Highlight.TLabel", background="#FFD700", foreground="#000000")
            else:
                style.configure("Highlight.TLabel", background="#4A6A9B", foreground="#EAEAEA")
            self.after(400, lambda: flash_highlight(count + 1))

        flash_highlight()

    def navigate_to_start_button(self):
        self._flash_button(self.start_button)

    def navigate_to_rtsp_entry(self):
        """Navigate to Source tab and highlight RTSP URL entry."""
        self.notebook.select(self.tab_source)
        self._flash_entry(self.rtsp_url_entry)
        try:
            self.rtsp_url_entry.focus_set()
        except Exception:
            pass

    def navigate_to_rtsp_add_button(self):
        self.notebook.select(self.tab_source)
        if hasattr(self, "btn_add_rtsp"):
            self._flash_button(self.btn_add_rtsp)

    def navigate_to_rtsp_plate_solve_button(self):
        self.notebook.select(self.tab_source)
        if hasattr(self, "btn_rtsp_plate_solve"):
            self._flash_button(self.btn_rtsp_plate_solve)

    def navigate_to_rtsp_mask_button(self):
        self.notebook.select(self.tab_source)
        if hasattr(self, "btn_rtsp_mask"):
            self._flash_button(self.btn_rtsp_mask)

    def navigate_to_periodic_scan_section(self):
        self.notebook.select(self.tab_source)
        if hasattr(self, "btn_select_periodic_dir"):
            self._flash_button(self.btn_select_periodic_dir)

    def navigate_to_periodic_dir_button(self):
        self.notebook.select(self.tab_source)
        if hasattr(self, "btn_select_periodic_dir"):
            self._flash_button(self.btn_select_periodic_dir)

    def navigate_to_periodic_auto_time_button(self):
        self.notebook.select(self.tab_source)
        if hasattr(self, "btn_periodic_auto_time"):
            self._flash_button(self.btn_periodic_auto_time)

    def navigate_to_settings_tab(self):
        self.notebook.select(self.tab_settings)
        if hasattr(self, "btn_detection_mask"):
            self._flash_button(self.btn_detection_mask)

    def navigate_to_mask_download_button(self):
        self.notebook.select(self.tab_settings)
        if hasattr(self, "btn_download_mask"):
            self._flash_button(self.btn_download_mask)

    def navigate_to_plate_solve_select_video_button(self):
        self.notebook.select(self.tab_settings)
        if hasattr(self, "btn_select_plate_solve_video"):
            self._flash_button(self.btn_select_plate_solve_video)

    def navigate_to_plate_solve_run_button(self):
        self.notebook.select(self.tab_settings)
        if hasattr(self, "btn_run_plate_solve"):
            self._flash_button(self.btn_run_plate_solve)

    def navigate_to_api_key_entry(self):
        self.notebook.select(self.tab_settings)
        if hasattr(self, "api_key_entry"):
            self._flash_entry(self.api_key_entry)
            try:
                self.api_key_entry.focus_set()
            except Exception:
                pass

    def navigate_to_summary_settings_button(self):
        self.notebook.select(self.tab_settings)
        if hasattr(self, "btn_summary_settings"):
            self._flash_button(self.btn_summary_settings)

    def navigate_to_analysis_tab(self):
        self.notebook.select(self.tab_analysis)

    def navigate_to_model_training_button(self):
        self.notebook.select(self.tab_analysis)
        if hasattr(self, "btn_model_training"):
            self._flash_button(self.btn_model_training)

    def navigate_to_model_selector(self):
        self.notebook.select(self.tab_settings)
        if hasattr(self, "btn_model_refresh"):
            self._flash_button(self.btn_model_refresh)

    def navigate_to_plate_solve_fov_settings(self):
        self.notebook.select(self.tab_advanced_settings)
        if hasattr(self, "btn_apply_plate_solve_fov"):
            self._flash_button(self.btn_apply_plate_solve_fov)

    def navigate_to_analysis_start_button(self):
        self.notebook.select(self.tab_analysis)
        if hasattr(self, "btn_analysis_start"):
            self._flash_button(self.btn_analysis_start)

    def navigate_to_blend_image_button(self):
        self.notebook.select(self.tab_analysis)
        if hasattr(self, "btn_blend_image"):
            self._flash_button(self.btn_blend_image)

    def navigate_to_blend_video_button(self):
        self.notebook.select(self.tab_analysis)
        if hasattr(self, "btn_blend_video"):
            self._flash_button(self.btn_blend_video)

    def navigate_to_timelapse_button(self):
        self.notebook.select(self.tab_analysis)
        if hasattr(self, "btn_timelapse"):
            self._flash_button(self.btn_timelapse)

    def navigate_to_long_exposure_button(self):
        self.notebook.select(self.tab_analysis)
        if hasattr(self, "btn_long_exposure"):
            self._flash_button(self.btn_long_exposure)

    def navigate_to_distortion_button(self):
        self.notebook.select(self.tab_analysis)
        if hasattr(self, "btn_distortion"):
            self._flash_button(self.btn_distortion)

    def navigate_to_angle_analysis_button(self):
        self.notebook.select(self.tab_analysis)
        if hasattr(self, "btn_angle_analysis"):
            self._flash_button(self.btn_angle_analysis)

    def navigate_to_video_concat_start_button(self):
        self.notebook.select(self.tab_analysis)
        if hasattr(self, "btn_video_concat_start"):
            self._flash_button(self.btn_video_concat_start)

    def navigate_to_chat_tab(self):
        self.notebook.select(self.tab_chat)

    def navigate_to_advanced_tab(self):
        self.notebook.select(self.tab_advanced_settings)
        if hasattr(self, "btn_reset_advanced"):
            self._flash_button(self.btn_reset_advanced)

    def navigate_to_log_tab(self):
        if hasattr(self, "status_panel") and hasattr(self.status_panel, "notebook"):
            try:
                self.status_panel.notebook.select(self.status_panel.log_frame)
            except Exception:
                pass

    def navigate_to_processing_status_tab(self):
        if hasattr(self, "status_panel") and hasattr(self.status_panel, "notebook"):
            try:
                self.status_panel.notebook.select(self.status_panel.status_frame)
            except Exception:
                pass

    def navigate_to_detection_mask_button(self):
        self.notebook.select(self.tab_settings)
        if hasattr(self, "btn_detection_mask"):
            self._scroll_settings_to_widget(self.btn_detection_mask, top_margin=20)
            self.after(160, lambda: self._flash_button(self.btn_detection_mask))

    def navigate_to_ps_mask_button(self):
        self.notebook.select(self.tab_settings)
        if hasattr(self, "btn_ps_mask"):
            self._scroll_settings_to_widget(self.btn_ps_mask, top_margin=20)
            self.after(160, lambda: self._flash_button(self.btn_ps_mask))

    def _scroll_settings_to_widget(self, widget, top_margin=16):
        """Scroll settings tab canvas so target widget becomes visible near top."""
        def do_scroll():
            try:
                if widget is None or not widget.winfo_exists():
                    return
                if not hasattr(self, "settings_canvas") or not hasattr(self, "settings_scrollable_frame"):
                    return
                canvas = self.settings_canvas
                scrollable_frame = self.settings_scrollable_frame
                if not canvas.winfo_exists() or not scrollable_frame.winfo_exists():
                    return

                self.update_idletasks()
                bbox = canvas.bbox("all")
                if not bbox:
                    return

                total_h = max(1, bbox[3] - bbox[1])
                view_h = max(1, canvas.winfo_height())
                max_scroll = max(1, total_h - view_h)

                # y position inside scrollable frame (independent from current scroll)
                y_in_frame = widget.winfo_rooty() - scrollable_frame.winfo_rooty()
                target_y = max(0, y_in_frame - top_margin)
                frac = min(1.0, max(0.0, target_y / max_scroll))
                canvas.yview_moveto(frac)
            except Exception:
                pass

        # Run twice to stabilize position after tab switch/layout refresh.
        self.after(20, do_scroll)
        self.after(140, do_scroll)

    def _flash_entry(self, entry):
        if entry is None:
            return
        try:
            if not entry.winfo_exists():
                return
            base_style = entry.cget("style") or "TEntry"
            style = ttk.Style()
            highlight_style = f"Highlight{entry.winfo_id()}.TEntry"
            style.configure(highlight_style, fieldbackground="#FFD700", foreground="#000000")

            def flash(count=0):
                if not entry.winfo_exists():
                    return
                if count >= 6:
                    entry.configure(style=base_style)
                    return
                entry.configure(style=highlight_style if count % 2 == 0 else base_style)
                self.after(400, lambda: flash(count + 1))

            flash()
        except Exception:
            pass

    def _flash_button(self, button):
        if button is None:
            return
        try:
            if not button.winfo_exists():
                return
            base_style = button.cget("style") or "TButton"
            style = ttk.Style()
            highlight_style = f"Highlight{button.winfo_id()}.TButton"
            style.configure(highlight_style, background="#FFD700", foreground="#000000")

            def flash(count=0):
                if not button.winfo_exists():
                    return
                if count >= 6:
                    button.configure(style=base_style)
                    return
                button.configure(style=highlight_style if count % 2 == 0 else base_style)
                self.after(400, lambda: flash(count + 1))

            flash()
        except Exception:
            pass

    def navigate_to_analysis_actions(self):
        """Navigate to Analysis tab and highlight the blend/timelapse buttons."""
        self.notebook.select(self.tab_analysis)
        self.navigate_to_blend_image_button()
        self.navigate_to_blend_video_button()
        self.navigate_to_timelapse_button()

    def _ensure_date_prefix(self, path: str) -> str:
        """Ensure the filename starts with YYYYMMDD_. If not, prepend today's date.

        Returns possibly-updated full path.
        """
        try:
            if not path:
                return path
            dirpath = os.path.dirname(path)
            basename = os.path.basename(path)
            # If already starts with YYYYMMDD_ then leave as-is
            if len(basename) >= 9 and basename[:9].isdigit() and basename[8] == '_':
                return path
            date_prefix = datetime.now().strftime("%Y%m%d_")
            new_name = date_prefix + basename
            return os.path.join(dirpath, new_name)
        except Exception:
            return path

    def create_analysis_tab(self, parent):
        """Create the '解析' tab where users can drop meteor info .txt files and run batch drawing."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        lf = ttk.LabelFrame(frame, text="流星解析 (info.txt ドロップ)")
        lf.pack(fill=tk.BOTH, expand=True, pady=5)

        drop_label = ttk.Label(lf, text="ここに流星の .txt ファイルをドラッグ＆ドロップ", relief=tk.SOLID, padding=20, anchor=tk.CENTER, borderwidth=1)
        drop_label.pack(fill=tk.X, pady=5)
        drop_label.drop_target_register(DND_FILES)
        drop_label.dnd_bind('<<Drop>>', self.drop_analysis)

        # Analysis list (styled)
        analysis_list_container = ttk.Frame(lf)
        analysis_list_container.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.analysis_list_canvas = tk.Canvas(analysis_list_container, bg="#3A4D6B", highlightthickness=0, height=100)
        self.analysis_list_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(analysis_list_container, orient=tk.VERTICAL, command=self.analysis_list_canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.analysis_list_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.analysis_list_frame = tk.Frame(self.analysis_list_canvas, bg="#3A4D6B")
        self.analysis_list_window = self.analysis_list_canvas.create_window((0, 0), window=self.analysis_list_frame, anchor="nw")
        
        def on_analysis_frame_configure(event):
            self.analysis_list_canvas.configure(scrollregion=self.analysis_list_canvas.bbox("all"))
        self.analysis_list_frame.bind("<Configure>", on_analysis_frame_configure)
        
        def on_analysis_canvas_configure(event):
            self.analysis_list_canvas.itemconfig(self.analysis_list_window, width=event.width)
        self.analysis_list_canvas.bind("<Configure>", on_analysis_canvas_configure)
        
        def on_analysis_mousewheel(event):
            self.analysis_list_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.analysis_list_canvas.bind("<MouseWheel>", on_analysis_mousewheel)
        self.analysis_list_frame.bind("<MouseWheel>", on_analysis_mousewheel)
        
        self.analysis_item_frames = []
        self.analysis_selected_indices = set()

        btn_frame = ttk.Frame(lf)
        btn_frame.pack(fill=tk.X, pady=(5,0))
        ttk.Button(btn_frame, text="選択項目を削除", command=self.remove_selected_analysis).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="すべて削除", command=self.remove_all_analysis).pack(side=tk.LEFT, padx=2)

        action_frame = ttk.Frame(frame)
        action_frame.pack(fill=tk.X, pady=8)
        
        row1 = ttk.Frame(action_frame)
        row1.pack(fill=tk.X, pady=2)
        self.btn_analysis_start = ttk.Button(row1, text="解析開始", command=self.start_analysis, style="Gray.TButton")
        self.btn_analysis_start.pack(side=tk.LEFT, padx=(0,5))
        ttk.Button(row1, text="座標点を追加", command=self.add_custom_point, style="Gray.TButton").pack(side=tk.LEFT, padx=(0,5))
        ttk.Button(row1, text="座標点を管理", command=self.manage_coordinates, style="Gray.TButton").pack(side=tk.LEFT, padx=(0,5))

        row2 = ttk.Frame(action_frame)
        row2.pack(fill=tk.X, pady=2)
        self.btn_long_exposure = ttk.Button(row2, text="長時間輝線マップを作成", command=self.create_long_exposure_map_callback, style="Gray.TButton")
        self.btn_long_exposure.pack(side=tk.LEFT, padx=(0,5))
        self.btn_distortion = ttk.Button(row2, text="ゆがみ補正", command=self.apply_distortion_correction_callback, style="Gray.TButton")
        self.btn_distortion.pack(side=tk.LEFT, padx=(0,5))
        self.btn_distortion_selfcal = ttk.Button(
            row2,
            text="夜間自己校正(20分)",
            command=self.estimate_distortion_map_night_callback,
            style="Gray.TButton"
        )
        self.btn_distortion_selfcal.pack(side=tk.LEFT, padx=(0,5))
        self.btn_distortion_map_view = ttk.Button(
            row2,
            text="ゆがみマップ表示",
            command=self.visualize_distortion_map_callback,
            style="Gray.TButton"
        )
        self.btn_distortion_map_view.pack(side=tk.LEFT, padx=(0,5))
        self.btn_angle_analysis = ttk.Button(row2, text="角度分布分析", command=self.analyze_angles_callback, style="Gray.TButton")
        self.btn_angle_analysis.pack(side=tk.LEFT, padx=(0,5))

        row3 = ttk.Frame(action_frame)
        row3.pack(fill=tk.X, pady=2)
        self.btn_blend_image = ttk.Button(row3, text="比較明合成画像を作成", command=self.create_lighten_blend_image_callback)
        self.btn_blend_image.pack(side=tk.LEFT, padx=(0,5))
        self.btn_blend_video = ttk.Button(row3, text="比較明合成動画を作成", command=self.create_lighten_blend_video_callback)
        self.btn_blend_video.pack(side=tk.LEFT, padx=(0,5))
        self.btn_timelapse = ttk.Button(row3, text="タイムラプス作成", command=self.create_timelapse_callback)
        self.btn_timelapse.pack(side=tk.LEFT, padx=(0,5))

        row4 = ttk.Frame(action_frame)
        row4.pack(fill=tk.X, pady=2)
        self.btn_model_training = ttk.Button(row4, text="機械学習モデル作成", command=self.open_model_training_tool)
        self.btn_model_training.pack(side=tk.LEFT, padx=(0, 5))

        lf_concat = ttk.LabelFrame(frame, text="動画連結")
        lf_concat.pack(fill=tk.BOTH, expand=True, pady=5)

        concat_drop_label = ttk.Label(lf_concat, text="ここに動画ファイルをドラッグ＆ドロップ", relief=tk.SOLID, padding=15, anchor=tk.CENTER, borderwidth=1)
        concat_drop_label.pack(fill=tk.X, pady=5)
        concat_drop_label.drop_target_register(DND_FILES)
        concat_drop_label.dnd_bind('<<Drop>>', self.drop_video_concat)

        concat_list_container = ttk.Frame(lf_concat)
        concat_list_container.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.video_concat_list_canvas = tk.Canvas(concat_list_container, bg="#3A4D6B", highlightthickness=0, height=80)
        self.video_concat_list_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        concat_scrollbar = ttk.Scrollbar(concat_list_container, orient=tk.VERTICAL, command=self.video_concat_list_canvas.yview)
        concat_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.video_concat_list_canvas.configure(yscrollcommand=concat_scrollbar.set)
        
        self.video_concat_list_frame = tk.Frame(self.video_concat_list_canvas, bg="#3A4D6B")
        self.video_concat_list_window = self.video_concat_list_canvas.create_window((0, 0), window=self.video_concat_list_frame, anchor="nw")
        
        def on_concat_frame_configure(event):
            self.video_concat_list_canvas.configure(scrollregion=self.video_concat_list_canvas.bbox("all"))
        self.video_concat_list_frame.bind("<Configure>", on_concat_frame_configure)
        
        def on_concat_canvas_configure(event):
            self.video_concat_list_canvas.itemconfig(self.video_concat_list_window, width=event.width)
        self.video_concat_list_canvas.bind("<Configure>", on_concat_canvas_configure)
        
        def on_concat_mousewheel(event):
            self.video_concat_list_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.video_concat_list_canvas.bind("<MouseWheel>", on_concat_mousewheel)
        self.video_concat_list_frame.bind("<MouseWheel>", on_concat_mousewheel)
        
        self.video_concat_item_frames = []
        self.video_concat_selected_indices = set()

        concat_btn_frame = ttk.Frame(lf_concat)
        concat_btn_frame.pack(fill=tk.X, pady=(5,0))
        ttk.Button(concat_btn_frame, text="ファイル追加", command=self.add_video_concat_files).pack(side=tk.LEFT, padx=2)
        ttk.Button(concat_btn_frame, text="選択削除", command=self.remove_selected_video_concat).pack(side=tk.LEFT, padx=2)
        ttk.Button(concat_btn_frame, text="すべて削除", command=self.remove_all_video_concat).pack(side=tk.LEFT, padx=2)

        concat_settings_frame = ttk.Frame(lf_concat)
        concat_settings_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(concat_settings_frame, text="ビットレート:").pack(side=tk.LEFT, padx=(0,5))
        bitrate_combo = ttk.Combobox(concat_settings_frame, textvariable=self.video_concat_bitrate_var, 
                                      values=["1000k","2000k","4000k", "8000k", "12000k", "16000k", "20000k"], width=8, state="readonly")
        bitrate_combo.pack(side=tk.LEFT, padx=(0,15))
        
        ttk.Label(concat_settings_frame, text="コーデック:").pack(side=tk.LEFT, padx=(0,5))
        ttk.Radiobutton(concat_settings_frame, text="H.264", variable=self.video_concat_codec_var, value="h264").pack(side=tk.LEFT, padx=(0,5))
        ttk.Radiobutton(concat_settings_frame, text="H.265", variable=self.video_concat_codec_var, value="h265").pack(side=tk.LEFT, padx=(0,5))

        ttk.Label(concat_settings_frame, text="FPS:").pack(side=tk.LEFT, padx=(10,5))
        fps_combo = ttk.Combobox(concat_settings_frame, textvariable=self.video_concat_fps_var,
                                 values=["Auto", "15", "24", "25", "30", "60"], width=6, state="readonly")
        fps_combo.pack(side=tk.LEFT, padx=(0,5))

        concat_settings_row2 = ttk.Frame(lf_concat)
        concat_settings_row2.pack(fill=tk.X, pady=(0, 5))
        ttk.Checkbutton(concat_settings_row2, text="セーフモード（タイムスタンプ補正）", variable=self.video_concat_safe_mode_var).pack(side=tk.LEFT, padx=(5,0))
        help_label = tk.Label(concat_settings_row2, text="?", font=("", 9, "bold"), fg="#87CEEB", bg="#2E3F5B", cursor="hand2")
        
        help_label.pack(side=tk.LEFT, padx=(2, 5))
        
        help_text = ("動画連結時に、入力ファイルのタイムスタンプ情報が正しくない場合や、\n"
                     "動画間で不整合がある場合に、このオプションを有効にしてください。\n"
                     "全フレームを再エンコードして一時ファイルを作成するため、\n"
                     "処理に時間がかかりますが、連結の安定性が向上します。")
        self._setup_help_tooltip(help_label, help_text)

        self.btn_video_concat_start = ttk.Button(lf_concat, text="連結開始", command=self.start_video_concat)
        self.btn_video_concat_start.pack(pady=5)

        return frame

    def _ensure_training_tool_dependencies(self):
        required = [
            ("customtkinter", "customtkinter"),
            ("sklearn", "scikit-learn"),
        ]
        missing = [(module_name, pkg_name) for module_name, pkg_name in required if importlib.util.find_spec(module_name) is None]
        if not missing:
            return True

        missing_pkgs = [pkg_name for _, pkg_name in missing]
        self.append_log(f"学習ツール依存が不足しています。自動インストールを開始: {', '.join(missing_pkgs)}")
        cmd = [sys.executable, "-m", "pip", "install", *missing_pkgs]
        try:
            result = subprocess.run(
                cmd,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                err = (result.stderr or result.stdout or "").strip()
                tail = err[-800:] if err else "詳細ログなし"
                messagebox.showerror(
                    "依存関係エラー",
                    f"学習ツール依存の自動インストールに失敗しました。\n"
                    f"実行コマンド: {' '.join(cmd)}\n\n{tail}",
                )
                self.append_log("学習ツール依存の自動インストールに失敗しました。")
                return False
            self.append_log("学習ツール依存の自動インストールが完了しました。")
            return True
        except Exception as e:
            messagebox.showerror("依存関係エラー", f"依存関係インストール中にエラーが発生しました: {e}")
            self.append_log(f"依存関係インストールエラー: {e}")
            return False

    def open_model_training_tool(self):
        trainer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_labeled_backup0826.py")
        if not os.path.exists(trainer_path):
            messagebox.showerror("エラー", f"学習スクリプトが見つかりません: {trainer_path}")
            return

        if not self._ensure_training_tool_dependencies():
            return

        try:
            subprocess.Popen([sys.executable, trainer_path], cwd=os.path.dirname(trainer_path))
            self.append_log("機械学習モデル作成ツールを起動しました。")
        except Exception as e:
            messagebox.showerror("エラー", f"学習ツールの起動に失敗しました: {e}")
            self.append_log(f"学習ツール起動エラー: {e}")

    def _setup_help_tooltip(self, widget, text):
        """ヘルプツールチップを作成（汎用版）"""
        self._help_tooltip = None
        self._hide_scheduled = None
        
        def show_tooltip(event=None):
            if self._hide_scheduled:
                self.after_cancel(self._hide_scheduled)
                self._hide_scheduled = None
            if self._help_tooltip:
                return
            x = widget.winfo_rootx() + 20
            y = widget.winfo_rooty() + 20
            self._help_tooltip = tk.Toplevel(self)
            self._help_tooltip.wm_overrideredirect(True)
            self._help_tooltip.wm_geometry(f"+{x}+{y}")
            
            # ダークテーマっぽい配色を使用
            bg_color = "#2E3F5B"
            fg_color = "#EAEAEA"
            
            frame = tk.Frame(self._help_tooltip, background=bg_color, relief=tk.SOLID, borderwidth=1)
            frame.pack()
            
            # 複数行テキストに対応
            for line in text.split('\n'):
                tk.Label(frame, text=line, font=("", 9), 
                       background=bg_color, foreground=fg_color, anchor=tk.W, justify=tk.LEFT).pack(fill=tk.X, padx=8, pady=1)
            
            # ツールチップ内にマウスが入ったら消えないように
            self._help_tooltip.bind("<Enter>", lambda e: cancel_hide())
            self._help_tooltip.bind("<Leave>", schedule_hide)
        
        def cancel_hide():
            if self._hide_scheduled:
                self.after_cancel(self._hide_scheduled)
                self._hide_scheduled = None
        
        def schedule_hide(event=None):
            if self._hide_scheduled:
                self.after_cancel(self._hide_scheduled)
            self._hide_scheduled = self.after(200, hide_tooltip)
        
        def hide_tooltip():
            if self._help_tooltip:
                self._help_tooltip.destroy()
                self._help_tooltip = None
            self._hide_scheduled = None
        
        widget.bind("<Enter>", show_tooltip)
        widget.bind("<Leave>", schedule_hide)

    def drop_analysis(self, event):
        paths = self.splitlist(event.data)
        added = False
        for p in paths:
            p = p.strip('{}')
            if os.path.isfile(p) and Path(p).suffix.lower() in ['.txt']:
                if p not in self.analysis_files:
                    self.analysis_files.append(p)
                    self._add_analysis_item(p)
                    added = True

        if not added:
            messagebox.showwarning("情報", "有効な .txt ファイルがドロップされませんでしたか、既に追加済みです。")
    
    def _add_analysis_item(self, filepath):
        """Add a styled item to the analysis list with modern badge."""
        index = len(self.analysis_item_frames)
        
        item_frame = tk.Frame(self.analysis_list_frame, bg="#3A4D6B", cursor="hand2")
        item_frame.pack(fill=tk.X, padx=2, pady=1)
        
        badge_canvas = tk.Canvas(item_frame, width=50, height=22, bg="#3A4D6B", highlightthickness=0)
        badge_canvas.pack(side=tk.LEFT, padx=(4, 6), pady=2)
        
        self._draw_rounded_rect(badge_canvas, 2, 2, 48, 20, 8, fill="#E67E22", outline="")
        badge_canvas.create_text(25, 11, text="TXT", fill="white", font=("Segoe UI", 8, "bold"))
        
        path_label = tk.Label(item_frame, text=filepath, bg="#3A4D6B", fg="#EAEAEA", 
                               anchor="w", font=("Segoe UI", 9))
        path_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        
        def on_click(event, idx=index):
            self._toggle_analysis_selection(idx)
        
        item_frame.bind("<Button-1>", on_click)
        badge_canvas.bind("<Button-1>", on_click)
        path_label.bind("<Button-1>", on_click)
        
        def on_mousewheel(event):
            self.analysis_list_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        item_frame.bind("<MouseWheel>", on_mousewheel)
        badge_canvas.bind("<MouseWheel>", on_mousewheel)
        path_label.bind("<MouseWheel>", on_mousewheel)
        
        self.analysis_item_frames.append({
            'frame': item_frame,
            'badge': badge_canvas,
            'label': path_label,
            'selected': False
        })
    
    def _toggle_analysis_selection(self, index):
        """Toggle selection state of analysis item."""
        if index < 0 or index >= len(self.analysis_item_frames):
            return
        
        item = self.analysis_item_frames[index]
        if item['selected']:
            item['frame'].config(bg="#3A4D6B")
            item['label'].config(bg="#3A4D6B")
            item['badge'].config(bg="#3A4D6B")
            item['selected'] = False
            self.analysis_selected_indices.discard(index)
        else:
            item['frame'].config(bg="#5A7D9B")
            item['label'].config(bg="#5A7D9B")
            item['badge'].config(bg="#5A7D9B")
            item['selected'] = True
            self.analysis_selected_indices.add(index)

    def remove_selected_analysis(self):
        if not self.analysis_selected_indices:
            return
        for idx in sorted(self.analysis_selected_indices, reverse=True):
            if 0 <= idx < len(self.analysis_files):
                del self.analysis_files[idx]
                item = self.analysis_item_frames.pop(idx)
                item['frame'].destroy()
        self.analysis_selected_indices.clear()
        
        for i, item in enumerate(self.analysis_item_frames):
            def make_click_handler(idx):
                return lambda e: self._toggle_analysis_selection(idx)
            item['frame'].bind("<Button-1>", make_click_handler(i))
            item['badge'].bind("<Button-1>", make_click_handler(i))
            item['label'].bind("<Button-1>", make_click_handler(i))

    def remove_all_analysis(self):
        if not self.analysis_files: return
        if messagebox.askyesno("確認", "リストからすべての解析ファイルを削除しますか？"):
            self.analysis_files.clear()
            for item in self.analysis_item_frames:
                item['frame'].destroy()
            self.analysis_item_frames.clear()
            self.analysis_selected_indices.clear()

    # ===== 動画連結機能 =====
    
    def drop_video_concat(self, event):
        """動画ファイルのドラッグ＆ドロップ処理"""
        paths = self.splitlist(event.data)
        added = False
        for p in paths:
            p = p.strip('{}')
            if os.path.isdir(p):
                # フォルダの場合は中の動画ファイルを追加
                for root, dirs, files in os.walk(p):
                    for f in sorted(files):
                        filepath = os.path.join(root, f)
                        if video_processor.is_video_file(filepath):
                            if filepath not in self.video_concat_files:
                                self.video_concat_files.append(filepath)
                                self._add_video_concat_item(filepath)
                                added = True
            elif os.path.isfile(p) and video_processor.is_video_file(p):
                if p not in self.video_concat_files:
                    self.video_concat_files.append(p)
                    self._add_video_concat_item(p)
                    added = True
        
        if not added:
            messagebox.showwarning("情報", "有効な動画ファイルが見つからないか、既に追加済みです。")

    def add_video_concat_files(self):
        """ダイアログから動画ファイルを追加"""
        filetypes = [
            ("動画ファイル", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v *.ts *.mts *.m2ts"),
            ("すべてのファイル", "*.*")
        ]
        files = filedialog.askopenfilenames(
            title="動画ファイルを選択",
            filetypes=filetypes
        )
        for f in files:
            if f not in self.video_concat_files:
                self.video_concat_files.append(f)
                self._add_video_concat_item(f)

    def _add_video_concat_item(self, filepath):
        """動画連結リストにアイテムを追加"""
        index = len(self.video_concat_item_frames)
        
        item_frame = tk.Frame(self.video_concat_list_frame, bg="#3A4D6B", cursor="hand2")
        item_frame.pack(fill=tk.X, padx=2, pady=1)
        
        badge_canvas = tk.Canvas(item_frame, width=30, height=22, bg="#3A4D6B", highlightthickness=0)
        badge_canvas.pack(side=tk.LEFT, padx=(4, 6), pady=2)
        
        self._draw_rounded_rect(badge_canvas, 2, 2, 28, 20, 8, fill="#3498DB", outline="")
        badge_canvas.create_text(15, 11, text=str(index + 1), fill="white", font=("Segoe UI", 8, "bold"))
        
        filename = os.path.basename(filepath)
        path_label = tk.Label(item_frame, text=filename, bg="#3A4D6B", fg="#EAEAEA", 
                               anchor="w", font=("Segoe UI", 9))
        path_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        
        def on_click(event, idx=index):
            self._toggle_video_concat_selection(idx)
        
        item_frame.bind("<Button-1>", on_click)
        badge_canvas.bind("<Button-1>", on_click)
        path_label.bind("<Button-1>", on_click)
        
        def on_mousewheel(event):
            self.video_concat_list_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        item_frame.bind("<MouseWheel>", on_mousewheel)
        badge_canvas.bind("<MouseWheel>", on_mousewheel)
        path_label.bind("<MouseWheel>", on_mousewheel)
        
        self.video_concat_item_frames.append({
            'frame': item_frame,
            'badge': badge_canvas,
            'label': path_label,
            'selected': False
        })
        
        self.append_log(f"動画連結リストにアイテムを追加しました: {filepath}")
        
        # 最初のファイルの場合、自動的にFPSを検出して設定する（Auto選択時用）
        if len(self.video_concat_files) == 1 and self.video_concat_fps_var.get() == "Auto":
            try:
                def detect_fps():
                    fps = video_processor.get_video_fps(filepath)
                    if fps > 0:
                        # 整数に近い場合は整数にする (29.97などはそのまま)
                        if abs(fps - round(fps)) < 0.01:
                            fps_val = str(int(round(fps)))
                        else:
                            fps_val = f"{fps:.2f}"
                        self.after(0, lambda: self.append_log(f"自動検出したFPSを設定しました: {fps_val}"))
                        # 必要ならここで変数を更新しても良いが、"Auto"のまま処理側で取得するのが安全
                
                threading.Thread(target=detect_fps, daemon=True).start()
            except:
                pass

    def _toggle_video_concat_selection(self, index):
        """動画連結リストの選択状態をトグル"""
        if index < 0 or index >= len(self.video_concat_item_frames):
            return
        
        item = self.video_concat_item_frames[index]
        if item['selected']:
            item['frame'].config(bg="#3A4D6B")
            item['label'].config(bg="#3A4D6B")
            item['badge'].config(bg="#3A4D6B")
            item['selected'] = False
            self.video_concat_selected_indices.discard(index)
        else:
            item['frame'].config(bg="#5A7D9B")
            item['label'].config(bg="#5A7D9B")
            item['badge'].config(bg="#5A7D9B")
            item['selected'] = True
            self.video_concat_selected_indices.add(index)

    def remove_selected_video_concat(self):
        """選択された動画を連結リストから削除"""
        if not self.video_concat_selected_indices:
            return
        for idx in sorted(self.video_concat_selected_indices, reverse=True):
            if 0 <= idx < len(self.video_concat_files):
                del self.video_concat_files[idx]
                item = self.video_concat_item_frames.pop(idx)
                item['frame'].destroy()
        self.video_concat_selected_indices.clear()
        self._reindex_video_concat_list()

    def remove_all_video_concat(self):
        """すべての動画を連結リストから削除"""
        if not self.video_concat_files:
            return
        if messagebox.askyesno("確認", "連結リストからすべての動画を削除しますか？"):
            self.video_concat_files.clear()
            for item in self.video_concat_item_frames:
                item['frame'].destroy()
            self.video_concat_item_frames.clear()
            self.video_concat_selected_indices.clear()

    def _reindex_video_concat_list(self):
        """動画連結リストの番号を振り直し"""
        for i, item in enumerate(self.video_concat_item_frames):
            item['badge'].delete("all")
            self._draw_rounded_rect(item['badge'], 2, 2, 28, 20, 8, fill="#3498DB", outline="")
            item['badge'].create_text(15, 11, text=str(i + 1), fill="white", font=("Segoe UI", 8, "bold"))
            
            def make_click_handler(idx):
                return lambda e: self._toggle_video_concat_selection(idx)
            item['frame'].bind("<Button-1>", make_click_handler(i))
            item['badge'].bind("<Button-1>", make_click_handler(i))
            item['label'].bind("<Button-1>", make_click_handler(i))

    def start_video_concat(self):
        """動画連結処理を開始"""
        if len(self.video_concat_files) < 2:
            messagebox.showwarning("情報", "連結するには2つ以上の動画ファイルを追加してください。")
            return
        
        # 出力ファイルを選択
        output_path = filedialog.asksaveasfilename(
            title="出力ファイルを保存",
            defaultextension=".mp4",
            filetypes=[("MP4ファイル", "*.mp4"), ("すべてのファイル", "*.*")]
        )
        
        if not output_path:
            return
        
        bitrate = self.video_concat_bitrate_var.get()
        codec = self.video_concat_codec_var.get()
        fps_str = self.video_concat_fps_var.get()
        safe_mode = self.video_concat_safe_mode_var.get()
        files = list(self.video_concat_files)
        
        fps_val = None
        if fps_str != "Auto":
            try:
                fps_val = float(fps_str)
            except ValueError:
                pass
        else:
            try:
                fps_val = video_processor.get_video_fps(files[0])
            except:
                pass
        
        self.append_log(f"動画連結を開始: {len(files)}ファイル")
        self.append_log(f"設定: ビットレート={bitrate}, コーデック={codec}, FPS={fps_str}")
        
        thread = threading.Thread(
            target=self._video_concat_worker,
            args=(files, output_path, bitrate, codec, fps_val, safe_mode),
            daemon=True
        )
        thread.start()

    def _video_concat_worker(self, files, output_path, bitrate, codec, fps, safe_mode):
        """動画連結のバックグラウンド処理"""
        def progress_callback(progress, message):
            self.after(0, lambda: self.append_log(message))
        
        def cancel_check():
            return self.cancel_flag.is_set()
        
        try:
            success, message = video_processor.concatenate_videos(
                input_files=files,
                output_path=output_path,
                bitrate=bitrate,
                codec=codec,
                fps=fps,
                safe_mode=safe_mode,
                progress_callback=progress_callback,
                cancel_check=cancel_check
            )
            
            if success:
                self.after(0, lambda: messagebox.showinfo("完了", message))
                self.after(0, lambda: self.append_log(f"連結完了: {output_path}"))
            else:
                self.after(0, lambda: messagebox.showerror("エラー", message))
                self.after(0, lambda: self.append_log(f"連結エラー: {message}"))
        except Exception as e:
            error_msg = f"予期せぬエラー: {e}"
            self.after(0, lambda: messagebox.showerror("エラー", error_msg))
            self.after(0, lambda: self.append_log(error_msg))

    def start_analysis(self):
        if not self.check_admin_password():
            return
            
        if not self.analysis_files:
            messagebox.showwarning("情報", "解析するファイルを追加してください。")
            return

        # Open a new window with a combined sky plot and draw all meteors
        win = Toplevel(self)
        win.title("流星まとめ表示")
        width, height = 900, 900
        win.geometry(f"{width}x{height}")

        canvas = tk.Canvas(win, width=width, height=height, bg="white")
        canvas.pack(fill=tk.BOTH, expand=True)

        cx, cy = width // 2, height // 2
        radius_px = min(width, height) // 2 - 60
        pixel_per_deg = radius_px / 90.0  # Northern hemisphere only (Dec 0° to +90°)

        # Store references for adding custom points later
        self.analysis_window = win
        self.analysis_canvas = canvas
        self.analysis_cx = cx
        self.analysis_cy = cy
        self.analysis_pixel_per_deg = pixel_per_deg

        # draw sky grid
        try:
            msv.draw_sky(canvas, cx, cy, radius_px)
        except Exception as e:
            messagebox.showerror("描画エラー", f"背景グリッドの描画に失敗しました: {e}")
            win.destroy(); return

        # draw each meteor; use parse_info_file and draw_meteor from meteor_sky_viewer
        failures = []
        for p in self.analysis_files:
            try:
                data = msv.parse_info_file(p)
                msv.draw_meteor(canvas, data, cx, cy, pixel_per_deg)
            except Exception as e:
                failures.append((p, str(e)))

        # draw custom points
        self.draw_custom_points()

        if failures:
            msg = "以下のファイルでプロットに失敗しました:\n" + "\n".join([f"{os.path.basename(f)}: {err}" for f, err in failures])
            messagebox.showwarning("一部失敗", msg)

    def add_custom_point(self):
        """Show dialog to add a custom coordinate point."""
        if not self.check_admin_password():
            return

        def on_add(name: str, ra: float, dec: float):
            self.coord_manager.add_point(name, ra, dec)
        
        dialog = coord_mgr.CoordinateDialog(self, on_add)
        dialog.show()
    
    def manage_coordinates(self):
        """Show dialog to manage coordinate points."""
        if not self.check_admin_password():
            return

        dialog = coord_mgr.CoordinateListDialog(self, self.coord_manager)
        dialog.show()
    
    def on_coordinates_changed(self):
        """Callback when coordinates are added or removed."""
        # Redraw if analysis window is open
        if self.analysis_window and self.analysis_window.winfo_exists():
            self.draw_custom_points()

    def draw_custom_points(self):
        """Draw all custom coordinate points on the analysis canvas."""
        if not self.analysis_canvas or not self.analysis_window or not self.analysis_window.winfo_exists():
            return

        # Delete previous custom point markers
        self.analysis_canvas.delete("custom_point")

        # Draw each custom point from the coordinate manager
        for name, ra, dec in self.coord_manager.get_points():
            try:
                x, y = msv.sky_to_xy(ra, dec, self.analysis_cx, self.analysis_cy, self.analysis_pixel_per_deg)
                
                # Draw a marker (small circle)
                r = 5
                self.analysis_canvas.create_oval(
                    x - r, y - r, x + r, y + r, 
                    fill="blue", outline="darkblue", width=2,
                    tags="custom_point"
                )
                
                # Draw the name label
                self.analysis_canvas.create_text(
                    x + 8, y - 8, 
                    text=name, 
                    anchor="nw", 
                    fill="blue", 
                    font=("Arial", 9, "bold"),
                    tags="custom_point"
                )
            except Exception as e:
                print(f"Failed to draw custom point {name}: {e}")



    def create_settings_tab(self, parent):
        frame = ttk.Frame(parent)

        # スクロール可能なキャンバスとスクロールバーを作成
        canvas = tk.Canvas(frame, highlightthickness=0, bg="#2E3F5B")
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        self.settings_canvas = canvas
        self.settings_scrollable_frame = scrollable_frame

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        # キャンバスのリサイズ時に内部フレームの幅を合わせる
        def on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind("<Configure>", on_canvas_configure)

        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # マウスホイールでスクロール（キャンバス上にカーソルがある間のみ）
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        def _bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        def _unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")

        # Canvasに入った時だけスクロールを割り当て、出たら解除
        # これにより他のタブやウィジェットへの干渉を防ぐ
        canvas.bind("<Enter>", _bind_mousewheel)
        canvas.bind("<Leave>", _unbind_mousewheel)

        # 以下の内容はscrollable_frameに配置
        lf_save = ttk.LabelFrame(scrollable_frame, text="保存アイテム")
        lf_save.pack(fill=tk.X, pady=5)
        save_map = {
            'video': "動画クリップ (cutout.mp4)", 
            'full_video': "フルサイズ動画 (fullsize.mp4)", 
            'cutout': "切り出し差分画像 (cutout_diff.jpg)", 
            'full': "全体差分画像 (full_diff.jpg)",
            'composite': "比較明合成画像 (composite.jpg)", 
            'info': "検出情報 (info.txt)", 
            'summary': "概要動画 (summary.mp4)"
        }
        for key, text in save_map.items():
            f = ttk.Frame(lf_save)
            f.pack(fill=tk.X)
            var = self.save_options_vars[key]
            chk = ttk.Checkbutton(f, text=text, variable=var)
            chk.pack(side=tk.LEFT, anchor=tk.W)
            if key == 'summary':
                # ヘルプの?マーク
                summary_help_label = tk.Label(f, text="?", font=("", 9, "bold"), fg="#87CEEB", bg="#2E3F5B", cursor="hand2")
                summary_help_label.pack(side=tk.LEFT, padx=(2, 0))
                summary_help_label._tooltip = None
                
                def show_summary_tooltip(event, label=summary_help_label):
                    if label._tooltip is not None:
                        return
                    tooltip = tk.Toplevel(self)
                    tooltip.wm_overrideredirect(True)
                    tooltip.wm_geometry(f"+{event.x_root + 15}+{event.y_root + 10}")
                    tooltip.configure(bg="#2E3F5B")
                    content_frame = tk.Frame(tooltip, bg="#2E3F5B", padx=8, pady=5)
                    content_frame.pack()
                    tk.Label(content_frame, text="SNSなどの投稿に便利な編集済みの動画を作成します", 
                             bg="#2E3F5B", fg="#EAEAEA", font=("Yu Gothic UI", 9)).pack()
                    label._tooltip = tooltip
                    
                    def hide_after_delay():
                        if label._tooltip:
                            label._tooltip.destroy()
                            label._tooltip = None
                    
                    tooltip.after(2000, hide_after_delay)
                
                def hide_summary_tooltip(event, label=summary_help_label):
                    if label._tooltip:
                        label._tooltip.after(200, lambda: close_tooltip(label))
                
                def close_tooltip(label):
                    if label._tooltip:
                        label._tooltip.destroy()
                        label._tooltip = None
                
                summary_help_label.bind("<Enter>", show_summary_tooltip)
                summary_help_label.bind("<Leave>", hide_summary_tooltip)
                
                self.btn_summary_settings = ttk.Button(f, text="詳細設定...", command=self.create_summary_settings_window)
                self.btn_summary_settings.pack(side=tk.LEFT, padx=5)
                var.trace_add("write", lambda *args: self.toggle_summary_settings_button())
                self.toggle_summary_settings_button()

        lf_path = ttk.LabelFrame(scrollable_frame, text="保存先")
        lf_path.pack(fill=tk.X, pady=5)
        path_frame1 = ttk.Frame(lf_path); path_frame1.pack(fill=tk.X, pady=2)
        ttk.Label(path_frame1, text="流星:", width=10).pack(side=tk.LEFT)
        ttk.Entry(path_frame1, textvariable=self.meteor_save_path_var, state='readonly').pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(path_frame1, text="選択", command=lambda: self.select_save_path(self.meteor_save_path_var)).pack(side=tk.LEFT, padx=(5,0))
        path_frame2 = ttk.Frame(lf_path); path_frame2.pack(fill=tk.X, pady=2)
        ttk.Label(path_frame2, text="非流星:", width=10).pack(side=tk.LEFT)
        ttk.Entry(path_frame2, textvariable=self.not_meteor_save_path_var, state='readonly').pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(path_frame2, text="選択", command=lambda: self.select_save_path(self.not_meteor_save_path_var)).pack(side=tk.LEFT, padx=(5,0))

        lf_model = ttk.LabelFrame(scrollable_frame, text="流星分類に使用するモデル")
        lf_model.pack(fill=tk.X, pady=5)
        model_row1 = ttk.Frame(lf_model)
        model_row1.pack(fill=tk.X, pady=2)
        ttk.Label(model_row1, text="モデル:", width=10).pack(side=tk.LEFT)
        self.cmb_model_select = ttk.Combobox(
            model_row1,
            textvariable=self.selected_model_path_var,
            state="readonly",
        )
        self.cmb_model_select.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.cmb_model_select.bind("<<ComboboxSelected>>", self.on_model_combobox_selected)
        self.btn_model_refresh = ttk.Button(model_row1, text="更新", command=self.refresh_model_candidates)
        self.btn_model_refresh.pack(side=tk.LEFT, padx=(5, 0))
        self.btn_select_model_file = ttk.Button(model_row1, text="参照", command=self.select_model_file)
        self.btn_select_model_file.pack(side=tk.LEFT, padx=(5, 0))

        model_row2 = ttk.Frame(lf_model)
        model_row2.pack(fill=tk.X, pady=(2, 4))
        ttk.Button(model_row2, text="流星分類モデルとして適用", command=lambda: self.apply_selected_model(show_message=True)).pack(side=tk.LEFT)
        ttk.Label(model_row2, textvariable=self.model_meta_info_var, foreground="#87CEEB").pack(side=tk.LEFT, padx=(10, 0))
        self.refresh_model_candidates()

        lf_astro = ttk.LabelFrame(scrollable_frame, text="プレートソルブ & マスク")
        lf_astro.pack(fill=tk.X, pady=5)
        
        # プレートソルブ説明用ヘルプアイコン
        ps_help_frame = ttk.Frame(lf_astro); ps_help_frame.pack(fill=tk.X, pady=2)
        ps_help_label = ttk.Label(ps_help_frame, text="ⓘ", font=("Arial", 11), foreground="#87CEEB", cursor="question_arrow")
        ps_help_label.pack(side=tk.LEFT, padx=(0,5))
        ttk.Label(ps_help_frame, text="機能の説明を表示", foreground="gray").pack(side=tk.LEFT)
        
        ps_help_text = """【プレートソルブとは】
動画のフレームを解析し、星の位置から撮影方向（赤経・赤緯）を
特定する機能です。これにより、検出した流星の軌跡に
星座やエリア名、座標情報をアノテーション（注釈付け）できます。

【使い方】
1. 「動画から実行」で動画ファイルを選択
2. 「実行」ボタンでプレートソルブを開始
3. 成功すると、以降の流星検出時に座標情報が付与されます

【マスク機能】
特定のエリア（建物、木など）を検出対象から除外できます。

⚠️ 注意事項
広角レンズ使用時は画像の歪みの影響で、
画像端付近のアノテーションが正確でない場合があります。"""
        
        ps_help_label._tooltip = None
        ps_help_label._tooltip_hover = False
        
        def show_ps_tooltip(event):
            if ps_help_label._tooltip is not None:
                return
            
            tooltip = tk.Toplevel(self)
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root + 15}+{event.y_root + 10}")
            tooltip.configure(bg="#333D4D")
            
            content_frame = tk.Frame(tooltip, bg="#333D4D", padx=10, pady=8)
            content_frame.pack()
            
            text_label = tk.Label(content_frame, text=ps_help_text, justify=tk.LEFT, bg="#333D4D", fg="#FFFFFF", font=("Yu Gothic UI", 10))
            text_label.pack()
            
            tooltip._hover = False
            
            def on_tooltip_enter(e):
                tooltip._hover = True
            def on_tooltip_leave(e):
                tooltip._hover = False
                tooltip.after(100, check_tooltip)
            
            def check_tooltip():
                if not ps_help_label._tooltip_hover and not tooltip._hover:
                    tooltip.destroy()
                    ps_help_label._tooltip = None
            
            tooltip.bind("<Enter>", on_tooltip_enter)
            tooltip.bind("<Leave>", on_tooltip_leave)
            
            ps_help_label._tooltip = tooltip
        
        def hide_ps_tooltip(event):
            ps_help_label._tooltip_hover = False
            if ps_help_label._tooltip:
                ps_help_label._tooltip.after(150, lambda: check_ps_close())
        
        def check_ps_close():
            if ps_help_label._tooltip and not ps_help_label._tooltip_hover and not getattr(ps_help_label._tooltip, '_hover', False):
                ps_help_label._tooltip.destroy()
                ps_help_label._tooltip = None
        
        def on_ps_enter(event):
            ps_help_label._tooltip_hover = True
            show_ps_tooltip(event)
        
        ps_help_label.bind("<Enter>", on_ps_enter)
        ps_help_label.bind("<Leave>", hide_ps_tooltip)

        ps_frame = ttk.Frame(lf_astro); ps_frame.pack(fill=tk.X, pady=2)
        ttk.Label(ps_frame, text="動画から実行:").pack(side=tk.LEFT, padx=(0,5))
        ttk.Entry(ps_frame, textvariable=self.plate_solve_video_path_var, state='readonly').pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.btn_select_plate_solve_video = ttk.Button(ps_frame, text="選択", command=self.select_plate_solve_video)
        self.btn_select_plate_solve_video.pack(side=tk.LEFT, padx=(5,0))
        self.btn_run_plate_solve = ttk.Button(ps_frame, text="実行", command=self.start_plate_solve)
        self.btn_run_plate_solve.pack(side=tk.LEFT, padx=(5,0))
        
        ps_wcs_frame = ttk.Frame(lf_astro); ps_wcs_frame.pack(fill=tk.X, pady=2)
        ttk.Label(ps_wcs_frame, text="既存WCSファイル:").pack(side=tk.LEFT, padx=(0,5))
        ttk.Entry(ps_wcs_frame, textvariable=self.plate_solve_wcs_path_var, state='readonly').pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(ps_wcs_frame, text="選択", command=self.select_plate_solve_wcs_file).pack(side=tk.LEFT, padx=(5,0))

        # Plate solve mode selection (local vs API)
        ps_mode_frame = ttk.Frame(lf_astro); ps_mode_frame.pack(fill=tk.X, pady=2)
        ttk.Label(ps_mode_frame, text="ソルバー:").pack(side=tk.LEFT, padx=(0,5))
        ttk.Radiobutton(ps_mode_frame, text="ローカル (coming soon)", variable=self.plate_solve_mode_var, value="local", state="disabled").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(ps_mode_frame, text="API (Astrometry.net)", variable=self.plate_solve_mode_var, value="api").pack(side=tk.LEFT, padx=5)

        # Astrometry.net API key input
        api_key_frame = ttk.Frame(lf_astro); api_key_frame.pack(fill=tk.X, pady=2)
        ttk.Label(api_key_frame, text="API Key").pack(side=tk.LEFT)
        
        # ヘルプボタン（?マーク）とツールチップ
        api_key_url = "https://nova.astrometry.net/"
        api_key_help_text_before = """Astrometry.net APIキーの取得手順

1. 公式サイトにアクセス
   URL: """
        api_key_help_text_after = """

2. サインイン（ログイン）
   画面右上にある [Sign in] をクリックし、
   GoogleアカウントやGitHubアカウントなどを
   使用してログインしてください。

3. ダッシュボードへ移動
   ログインすると、自動的に「Dashboard」ページに
   リダイレクトされます。
   （もし別のページにいる場合は、上部メニューの
   [Dashboard] ➞ [My Profile] をクリック）

4. APIキーの確認
   ダッシュボードページ内の「API Key」という
   項目を探してください。
   Your API key is: XXXXXXXX
   と表示されている英数字の文字列が、
   あなたのAPIキーです。"""
        
        help_label = ttk.Label(api_key_frame, text=" ? ", font=("Arial", 10, "bold"), foreground="#87CEEB", cursor="question_arrow")
        help_label.pack(side=tk.LEFT)
        
        ttk.Label(api_key_frame, text=":").pack(side=tk.LEFT, padx=(0,5))
        
        self.api_key_entry = ttk.Entry(api_key_frame, textvariable=self.astrometry_api_key_var, show="*", width=30)
        self.api_key_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        # API Key変更時にconfigを更新
        self.astrometry_api_key_var.trace_add("write", lambda *args: setattr(config, 'ASTROMETRY_API_KEY', self.astrometry_api_key_var.get()))
        
        # ツールチップの作成
        help_label._tooltip = None
        help_label._tooltip_hover = False
        
        def show_tooltip(event):
            if help_label._tooltip is not None:
                return  # 既に表示中の場合は何もしない
            
            tooltip = tk.Toplevel(self)
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root + 10}+{event.y_root + 10}")
            tooltip.configure(bg="#2E3F5B")
            frame = ttk.Frame(tooltip, padding=10)
            frame.pack()
            
            # Textウィジェットでクリック可能なURLを実装
            text_widget = tk.Text(frame, wrap=tk.WORD, width=45, height=20, bg="#3A4D6B", fg="#EAEAEA", 
                                  relief=tk.FLAT, highlightthickness=0, cursor="arrow", font=("Arial", 9))
            text_widget.pack()
            text_widget.insert(tk.END, api_key_help_text_before)
            
            # URLをハイパーリンクとして挿入
            text_widget.tag_configure("hyperlink", foreground="#00BFFF", underline=True)
            url_start = text_widget.index(tk.END)
            text_widget.insert(tk.END, api_key_url, "hyperlink")
            
            def open_url(e):
                import webbrowser
                webbrowser.open(api_key_url)
            
            text_widget.tag_bind("hyperlink", "<Button-1>", open_url)
            text_widget.tag_bind("hyperlink", "<Enter>", lambda e: text_widget.config(cursor="hand2"))
            text_widget.tag_bind("hyperlink", "<Leave>", lambda e: text_widget.config(cursor="arrow"))
            
            text_widget.insert(tk.END, api_key_help_text_after)
            text_widget.config(state=tk.DISABLED)
            
            # ツールチップにマウスが入った時・出た時のイベント
            def on_tooltip_enter(e):
                help_label._tooltip_hover = True
            
            def on_tooltip_leave(e):
                help_label._tooltip_hover = False
                # 少し遅延してからチェック（マウスが移動中の場合に対応）
                self.after(100, check_and_hide_tooltip)
            
            tooltip.bind("<Enter>", on_tooltip_enter)
            tooltip.bind("<Leave>", on_tooltip_leave)
            
            help_label._tooltip = tooltip
        
        def check_and_hide_tooltip():
            if help_label._tooltip and not help_label._tooltip_hover:
                try:
                    help_label._tooltip.destroy()
                except:
                    pass
                help_label._tooltip = None
            
        def hide_tooltip(event):
            # 少し遅延してからツールチップを閉じる（ツールチップにカーソルが移動する時間を確保）
            self.after(150, check_and_hide_tooltip)
        
        help_label.bind("<Enter>", show_tooltip)
        help_label.bind("<Leave>", hide_tooltip)

        ttk.Label(lf_astro, textvariable=self.plate_solve_status_var, foreground="#87CEEB").pack(pady=2)
        ttk.Checkbutton(lf_astro, text="プレートソルブ結果を利用する", variable=self.use_plate_solve_var).pack(anchor=tk.W)

        ttk.Separator(lf_astro, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        mask_btn_frame = ttk.Frame(lf_astro); mask_btn_frame.pack(fill=tk.X, pady=2)
        self.btn_detection_mask = ttk.Button(mask_btn_frame, text="検出マスク作成", command=lambda: self.create_mask_window(is_plate_solve_mask=False))
        self.btn_detection_mask.pack(side=tk.LEFT)
        self.btn_download_mask = ttk.Button(mask_btn_frame, text="💾", width=3, command=self.download_mask)
        self.btn_download_mask.pack(side=tk.LEFT, padx=2)
        self.btn_ps_mask = ttk.Button(mask_btn_frame, text="プレートソルブ用マスク作成", command=lambda: self.create_mask_window(is_plate_solve_mask=True))
        self.btn_ps_mask.pack(side=tk.LEFT, padx=5)
        
        self.mask_preview_frame = ttk.Frame(lf_astro); self.mask_preview_frame.pack(pady=5)
        self.mask_preview_label = ttk.Label(self.mask_preview_frame, text="検出マスクなし")
        self.mask_preview_label.pack(side=tk.LEFT, padx=10)
        self.ps_mask_preview_label = ttk.Label(self.mask_preview_frame, text="PSマスクなし")
        self.ps_mask_preview_label.pack(side=tk.LEFT, padx=10)

        ttk.Checkbutton(lf_astro, text="検出マスクを適用する", variable=self.apply_mask_var).pack(anchor=tk.W)

        return frame

    def _model_search_dirs(self):
        candidates = [
            os.path.dirname(os.path.abspath(__file__)),
            getattr(config, "EXE_DIR", ""),
        ]
        result = []
        seen = set()
        for d in candidates:
            if not d:
                continue
            abs_d = os.path.abspath(d)
            if abs_d in seen or not os.path.isdir(abs_d):
                continue
            seen.add(abs_d)
            result.append(abs_d)
        return result

    def refresh_model_candidates(self):
        current = self.selected_model_path_var.get().strip()
        discovered = model_catalog.discover_model_paths(
            self._model_search_dirs(),
            self.custom_model_paths,
            recursive=False,
        )
        if current and current not in discovered:
            discovered.append(current)
        self.available_model_paths = discovered
        if hasattr(self, "cmb_model_select"):
            self.cmb_model_select.configure(values=discovered)
        if (not current) and discovered:
            self.selected_model_path_var.set(discovered[0])
        self.update_model_meta_info(self.selected_model_path_var.get())

    def update_model_meta_info(self, model_path):
        model_path = (model_path or "").strip()
        if not model_path or not os.path.exists(model_path):
            self.model_meta_info_var.set("mean/std: --")
            return

        meta = model_catalog.load_model_metadata(model_path)
        mean = np.round(np.array(meta.get("mean", model_catalog.DEFAULT_MEAN), dtype=float), 3).tolist()
        std = np.round(np.array(meta.get("std", model_catalog.DEFAULT_STD), dtype=float), 3).tolist()
        resize = meta.get("input_resize")
        resize_text = "no resize" if resize is None else f"{int(resize[0])}x{int(resize[1])}"
        self.model_meta_info_var.set(f"mean={mean} std={std} resize={resize_text}")

    def on_model_combobox_selected(self, _event=None):
        self.apply_selected_model(show_message=False)

    def select_model_file(self):
        model_path = filedialog.askopenfilename(
            title="モデルファイルを選択",
            filetypes=(("PyTorch Model", "*.pth"), ("All files", "*.*")),
        )
        if not model_path:
            return
        model_path = os.path.abspath(model_path)
        if model_path not in self.custom_model_paths:
            self.custom_model_paths.append(model_path)
        self.selected_model_path_var.set(model_path)
        self.refresh_model_candidates()
        self.apply_selected_model(show_message=True)

    def apply_selected_model(self, show_message=False, silent=False):
        model_path = self.selected_model_path_var.get().strip()
        if not model_path:
            if not silent:
                messagebox.showwarning("警告", "モデルが選択されていません。")
            return False

        model_path = os.path.abspath(model_path)
        if not os.path.exists(model_path):
            if not silent:
                messagebox.showerror("エラー", f"モデルファイルが見つかりません: {model_path}")
            return False

        metadata = model_catalog.load_model_metadata(model_path)
        ok, msg = model.reload_model(model_path=model_path, metadata=metadata)
        if not ok:
            if not silent:
                messagebox.showerror("エラー", f"モデル読み込みに失敗しました: {msg}")
            self.append_log(f"モデル読み込み失敗: {msg}")
            return False

        config.MODEL_PATH = model_path
        self.selected_model_path_var.set(model_path)
        self.update_model_meta_info(model_path)
        self.append_log(f"使用モデルを切り替えました: {os.path.basename(model_path)}")
        if show_message:
            messagebox.showinfo("モデル適用", f"モデルを適用しました:\n{model_path}")
        return True

    def _create_basic_processing_params_section(self, parent):
        """Create basic processing controls moved from 保存設定 to 詳細設定 tab."""
        lf_params = ttk.LabelFrame(parent, text="処理パラメータ")
        lf_params.pack(fill=tk.X, pady=5)

        p_frame1 = ttk.Frame(lf_params)
        p_frame1.pack(fill=tk.X, pady=2)
        ttk.Label(p_frame1, text="同時処理数:", width=20).pack(side=tk.LEFT)
        ttk.Spinbox(p_frame1, from_=1, to=os.cpu_count() or 1, width=5, textvariable=self.concurrency_var).pack(side=tk.LEFT)

        p_frame2 = ttk.Frame(lf_params)
        p_frame2.pack(fill=tk.X, pady=2)
        ttk.Label(p_frame2, text="差分作成間隔 (秒):", width=20).pack(side=tk.LEFT)
        ttk.Spinbox(p_frame2, from_=1, to=60, width=5, textvariable=self.interval_var).pack(side=tk.LEFT)

        p_frame3 = ttk.Frame(lf_params)
        p_frame3.pack(fill=tk.X, pady=2)
        ttk.Label(p_frame3, text="差分作成期間 (秒):", width=20).pack(side=tk.LEFT)
        ttk.Spinbox(p_frame3, from_=1, to=60, width=5, textvariable=self.duration_var).pack(side=tk.LEFT)

        p_frame4 = ttk.Frame(lf_params)
        p_frame4.pack(fill=tk.X, pady=2)
        ttk.Label(p_frame4, text="RTSP検出感度:", width=20).pack(side=tk.LEFT)
        ttk.Radiobutton(p_frame4, text="雲が少ないとき", variable=self.rtsp_preset_var, value="clear").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(p_frame4, text="雲が多いとき（推奨）", variable=self.rtsp_preset_var, value="cloudy").pack(side=tk.LEFT, padx=5)

    def create_advanced_settings_tab(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        canvas = tk.Canvas(frame, highlightthickness=0, bg="#2E3F5B")
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        def on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind("<Configure>", on_canvas_configure)

        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", on_mousewheel)

        # 基本の処理パラメータ（保存設定タブから移設）
        self._create_basic_processing_params_section(scrollable_frame)

        lf_ps_fov = ttk.LabelFrame(scrollable_frame, text="プレートソルブ時の視野角設定")
        lf_ps_fov.pack(fill=tk.X, pady=5)
        ps_fov_row = ttk.Frame(lf_ps_fov)
        ps_fov_row.pack(fill=tk.X, pady=2)
        ttk.Label(ps_fov_row, text="Lower:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Spinbox(ps_fov_row, from_=10, to=180, increment=0.5, width=8, textvariable=self.cfg_rtsp_scale_lower_var).pack(side=tk.LEFT)
        ttk.Label(ps_fov_row, text="Upper:").pack(side=tk.LEFT, padx=(10, 4))
        ttk.Spinbox(ps_fov_row, from_=10, to=180, increment=0.5, width=8, textvariable=self.cfg_rtsp_scale_upper_var).pack(side=tk.LEFT)
        self.btn_apply_plate_solve_fov = ttk.Button(
            ps_fov_row,
            text="視野角を反映",
            command=self.apply_advanced_settings_to_config,
        )
        self.btn_apply_plate_solve_fov.pack(side=tk.LEFT, padx=(12, 0))

        # 検出パラメータ
        lf_detect = ttk.LabelFrame(scrollable_frame, text="検出パラメータ")
        lf_detect.pack(fill=tk.X, pady=5)
        
        def add_param(parent, text, var, from_, to, increment=1):
            f = ttk.Frame(parent)
            f.pack(fill=tk.X, pady=2)
            ttk.Label(f, text=text, width=35).pack(side=tk.LEFT)
            ttk.Spinbox(f, from_=from_, to=to, increment=increment, width=8, textvariable=var).pack(side=tk.LEFT)

        add_param(lf_detect, "最小線長 (MIN_LINE_LENGTH):", self.cfg_min_line_length_var, 1, 100)
        add_param(lf_detect, "枠外無視サイズ (BORDER_SIZE):", self.cfg_border_size_var, 0, 100)
        add_param(lf_detect, "重複検出閾値 (DUPLICATE_THRESH):", self.cfg_duplicate_thresh_var, 10, 500, 10)
        add_param(lf_detect, "流星判定確率 (METEOR_PROB_THRESH):", self.cfg_meteor_prob_var, 0.1, 1.0, 0.05)

        # 詳細検出パラメータ
        lf_finer = ttk.LabelFrame(scrollable_frame, text="詳細検出パラメータ")
        lf_finer.pack(fill=tk.X, pady=5)
        add_param(lf_finer, "詳細検出ウィンドウ秒数:", self.cfg_finer_window_sec_var, 1.0, 10.0, 0.5)
        add_param(lf_finer, "比較明合成ステップ数:", self.cfg_finer_comp_step_var, 1, 20)
        add_param(lf_finer, "最小線長 (詳細検出):", self.cfg_finer_min_length_var, 5, 50)
        add_param(lf_finer, "パディング秒数:", self.cfg_finer_padding_sec_var, 0.1, 2.0, 0.1)
        add_param(lf_finer, "カットアウトサイズ (詳細検出):", self.cfg_finer_cutout_size_var, 128, 1024, 64)

        # 飛行機判定パラメータ
        # 飛行機判定パラメータは設定セクションから非表示化（内部パラメータは維持）

        # 動画クリップパラメータ
        lf_clip = ttk.LabelFrame(scrollable_frame, text="動画クリップパラメータ")
        lf_clip.pack(fill=tk.X, pady=5)
        add_param(lf_clip, "最大クリップ秒数:", self.cfg_max_clip_dur_var, 1, 30)
        add_param(lf_clip, "クリップ秒数目安:", self.cfg_clip_dur_sec_var, 1.0, 30.0, 0.5)
        add_param(lf_clip, "切り出しサイズ (CUTOUT_SIZE):", self.cfg_cutout_size_var, 128, 1023, 64)

        btn_frame = ttk.Frame(scrollable_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        self.btn_reset_advanced = ttk.Button(btn_frame, text="デフォルトに戻す", command=self.reset_advanced_settings)
        self.btn_reset_advanced.pack(side=tk.LEFT, padx=5)

        return frame

    def reset_advanced_settings(self):
        if messagebox.askyesno("確認", "詳細設定を初期値に戻しますか？"):
            self.cfg_min_line_length_var.set("25")
            self.cfg_border_size_var.set("30")
            self.cfg_duplicate_thresh_var.set("100")
            self.cfg_meteor_prob_var.set("0.5")
            self.cfg_finer_window_sec_var.set("4.0")
            self.cfg_finer_comp_step_var.set("3")
            self.cfg_finer_min_length_var.set("15")
            self.cfg_finer_padding_sec_var.set("0.5")
            self.cfg_finer_cutout_size_var.set("384")
            self.cfg_airplane_dur_thresh_var.set("7")
            self.cfg_airplane_frame_thresh_var.set("7")
            self.cfg_tracking_dist_thresh_var.set("200")
            self.cfg_max_clip_dur_var.set("2")
            self.cfg_clip_dur_sec_var.set("3.0")
            self.cfg_cutout_size_var.set("256")
            self.cfg_rtsp_scale_lower_var.set("85")
            self.cfg_rtsp_scale_upper_var.set("100")
            self.append_log("詳細設定をデフォルト値にリセットしました。")

    def create_info_panel(self, parent):
        panel = status_panel.StatusPanel(parent, progress_queue=self.progress_queue, app=self)
        panel.pack(fill=tk.BOTH, expand=True, pady=5)
        self.status_panel = panel

        self.log_text = panel.log_text
        self._init_summary_log_hover_preview()

        status_row = ttk.Frame(parent)
        status_row.pack(fill=tk.X, pady=5)
        self.progress = ttk.Progressbar(status_row, orient=tk.HORIZONTAL, mode='determinate')
        self.progress.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=(0,10))
        self.status_label = ttk.Label(status_row, text="待機中", width=15)
        self.status_label.pack(side=tk.LEFT)

        time_frame = ttk.Frame(parent)
        time_frame.pack(fill=tk.X, pady=5)
        self.eta_label = ttk.Label(time_frame, text="ETA: --:--:--", width=20)
        self.eta_label.pack(side=tk.LEFT)
        self.elapsed_label = ttk.Label(time_frame, text="経過: 00:00:00", width=20)
        self.elapsed_label.pack(side=tk.LEFT)

        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=(10,0))
        self.start_button = ttk.Button(btn_frame, text="開始", command=self.start_processing)
        self.start_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,5))
        self.cancel_button = ttk.Button(btn_frame, text="キャンセル", command=self.cancel_processing, state=tk.DISABLED)
        self.cancel_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5,0))

        # Share status callback with worker-side pipeline.
        try:
            global STATUS_CALLBACK
            STATUS_CALLBACK = panel.get_status_callback()
        except Exception:
            STATUS_CALLBACK = None

    def append_log(self, message: str):
        if threading.current_thread() is not threading.main_thread():
            self.after(0, lambda m=message: self.append_log(m))
            return
        if not self.log_text.winfo_exists():
            return
        self.log_text.config(state='normal')
        try:
            view_top, view_bottom = self.log_text.yview()
        except Exception:
            view_top, view_bottom = (1.0, 1.0)
        follow_tail = view_bottom >= 0.995

        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        summary_ref = self._extract_summary_video_ref(message)
        if summary_ref:
            line_no = self.log_text.index("end-2c").split('.')[0]
            self.log_text.tag_add("summary_hover", f"{line_no}.0", f"{line_no}.end")
            self._summary_log_line_map[line_no] = {
                "summary_ref": summary_ref,
                "resolved_path": self._resolve_summary_video_path(summary_ref),
            }

        if follow_tail:
            self.log_text.see(tk.END)
        else:
            try:
                self.log_text.yview_moveto(view_top)
            except Exception:
                pass
        self.log_text.config(state='disabled')

    def _init_summary_log_hover_preview(self):
        self._summary_log_line_map = {}
        self._active_summary_line = None
        self._summary_preview_window = None
        self._summary_preview_title_label = None
        self._summary_preview_image_label = None
        self._summary_preview_open_button = None
        self._summary_preview_photo = None
        self._summary_preview_capture = None
        self._summary_preview_after_id = None
        self._summary_preview_hide_after_id = None
        self._summary_preview_fps = 12.0

        self.log_text.tag_config("summary_hover", foreground="#87CEEB", underline=True)
        self.log_text.bind("<Motion>", self._on_log_text_motion_for_summary_preview, add="+")
        self.log_text.bind("<Leave>", self._on_log_text_leave_for_summary_preview, add="+")

    def _extract_summary_video_ref(self, message: str) -> Optional[str]:
        if not message:
            return None
        patterns = [
            r"->\s*Summary:\s*(.+?\.mp4)\s*$",
            r"概要動画を保存しました:\s*(.+?\.mp4)\s*$",
        ]
        for pattern in patterns:
            m = re.search(pattern, message, flags=re.IGNORECASE)
            if m:
                return m.group(1).strip().strip('"').strip("'")
        return None

    def _resolve_summary_video_path(self, summary_ref: str) -> Optional[str]:
        if not summary_ref:
            return None

        ref = summary_ref.strip().strip('"').strip("'")
        if not ref:
            return None

        candidates = []

        if os.path.isabs(ref):
            candidates.append(ref)
        else:
            candidates.append(os.path.abspath(ref))

        if hasattr(self, "meteor_save_path_var"):
            try:
                meteor_dir = self.meteor_save_path_var.get()
                if meteor_dir:
                    candidates.append(os.path.join(meteor_dir, ref))
                    candidates.append(os.path.join(meteor_dir, os.path.basename(ref)))
            except Exception:
                pass

        if hasattr(self, "not_meteor_save_path_var"):
            try:
                not_meteor_dir = self.not_meteor_save_path_var.get()
                if not_meteor_dir:
                    candidates.append(os.path.join(not_meteor_dir, ref))
                    candidates.append(os.path.join(not_meteor_dir, os.path.basename(ref)))
            except Exception:
                pass

        try:
            candidates.append(os.path.join(config.DEFAULT_METEOR_SAVE_PATH, os.path.basename(ref)))
            candidates.append(os.path.join(config.DEFAULT_NOT_METEOR_SAVE_PATH, os.path.basename(ref)))
        except Exception:
            pass

        for path in candidates:
            if path and os.path.exists(path):
                return os.path.abspath(path)
        return None

    def _on_log_text_motion_for_summary_preview(self, event):
        if not hasattr(self, "log_text") or not self.log_text.winfo_exists():
            return

        self._cancel_summary_preview_hide()

        try:
            index = self.log_text.index(f"@{event.x},{event.y}")
        except Exception:
            self._hide_summary_preview()
            return

        if "summary_hover" not in self.log_text.tag_names(index):
            self._hide_summary_preview()
            return

        line_no = index.split('.')[0]
        meta = self._summary_log_line_map.get(line_no)
        if not meta:
            self._hide_summary_preview()
            return

        if self._active_summary_line != line_no:
            self._active_summary_line = line_no
            self._show_summary_preview_for_line(meta, event)
        else:
            self._move_summary_preview(event)

    def _on_log_text_leave_for_summary_preview(self, _event):
        self._schedule_summary_preview_hide(260)

    def _cancel_summary_preview_hide(self):
        if self._summary_preview_hide_after_id is not None:
            try:
                self.after_cancel(self._summary_preview_hide_after_id)
            except Exception:
                pass
            self._summary_preview_hide_after_id = None

    def _schedule_summary_preview_hide(self, delay_ms: int = 220):
        self._cancel_summary_preview_hide()
        self._summary_preview_hide_after_id = self.after(delay_ms, self._hide_summary_preview_if_pointer_outside)

    def _hide_summary_preview_if_pointer_outside(self):
        self._summary_preview_hide_after_id = None

        preview_has_pointer = False
        if self._summary_preview_window is not None and self._summary_preview_window.winfo_exists():
            try:
                px = self.winfo_pointerx()
                py = self.winfo_pointery()
                preview_has_pointer = self._summary_preview_window.winfo_containing(px, py) is not None
            except Exception:
                preview_has_pointer = False
        if preview_has_pointer:
            return

        if hasattr(self, "log_text") and self.log_text.winfo_exists():
            try:
                px = self.winfo_pointerx()
                py = self.winfo_pointery()
                x = px - self.log_text.winfo_rootx()
                y = py - self.log_text.winfo_rooty()
                if 0 <= x < self.log_text.winfo_width() and 0 <= y < self.log_text.winfo_height():
                    index = self.log_text.index(f"@{x},{y}")
                    if "summary_hover" in self.log_text.tag_names(index):
                        return
            except Exception:
                pass

        self._hide_summary_preview()

    def _on_summary_preview_enter(self, _event):
        self._cancel_summary_preview_hide()

    def _on_summary_preview_leave(self, _event):
        self._schedule_summary_preview_hide(180)

    def _show_summary_preview_for_line(self, meta: Dict[str, Any], event):
        self._hide_summary_preview()

        summary_path = meta.get("resolved_path")
        if not summary_path:
            summary_path = self._resolve_summary_video_path(meta.get("summary_ref", ""))
            if summary_path:
                meta["resolved_path"] = summary_path

        win = tk.Toplevel(self)
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        win.configure(bg="#0F1724")
        win.bind("<Enter>", self._on_summary_preview_enter, add="+")
        win.bind("<Leave>", self._on_summary_preview_leave, add="+")

        container = tk.Frame(win, bg="#0F1724", bd=1, relief=tk.SOLID)
        container.pack(fill=tk.BOTH, expand=True)
        container.bind("<Enter>", self._on_summary_preview_enter, add="+")
        container.bind("<Leave>", self._on_summary_preview_leave, add="+")
        win.geometry("392x300")

        title = os.path.basename(summary_path) if summary_path else "summary.mp4 が見つかりません"
        if len(title) > 56:
            title = title[:53] + "..."
        self._summary_preview_title_label = tk.Label(
            container,
            text=title,
            bg="#0F1724",
            fg="#D9E5FF",
            anchor="w",
            padx=8,
            pady=4,
            font=("Segoe UI", 9, "bold"),
        )
        self._summary_preview_title_label.pack(fill=tk.X)
        self._summary_preview_title_label.bind("<Enter>", self._on_summary_preview_enter, add="+")
        self._summary_preview_title_label.bind("<Leave>", self._on_summary_preview_leave, add="+")

        preview_area = tk.Frame(container, bg="#000000", width=372, height=214)
        preview_area.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        preview_area.pack_propagate(False)
        preview_area.bind("<Enter>", self._on_summary_preview_enter, add="+")
        preview_area.bind("<Leave>", self._on_summary_preview_leave, add="+")

        self._summary_preview_image_label = tk.Label(
            preview_area,
            bg="#000000",
            fg="#EAEAEA",
            text="プレビューを読み込み中...",
        )
        self._summary_preview_image_label.pack(fill=tk.BOTH, expand=True)
        self._summary_preview_image_label.bind("<Enter>", self._on_summary_preview_enter, add="+")
        self._summary_preview_image_label.bind("<Leave>", self._on_summary_preview_leave, add="+")

        action_frame = tk.Frame(container, bg="#0F1724")
        action_frame.pack(fill=tk.X, padx=6, pady=(0, 6))
        action_frame.bind("<Enter>", self._on_summary_preview_enter, add="+")
        action_frame.bind("<Leave>", self._on_summary_preview_leave, add="+")

        self._summary_preview_open_button = ttk.Button(
            action_frame,
            text="ファイルの場所を開く",
            command=lambda m=meta: self._open_summary_file_location(m),
        )
        self._summary_preview_open_button.pack(side=tk.RIGHT)
        self._summary_preview_open_button.bind("<Enter>", self._on_summary_preview_enter, add="+")
        self._summary_preview_open_button.bind("<Leave>", self._on_summary_preview_leave, add="+")
        if not summary_path:
            self._summary_preview_open_button.configure(state=tk.DISABLED)

        self._summary_preview_window = win
        self._move_summary_preview(event)

        if not summary_path or not os.path.exists(summary_path):
            self._summary_preview_image_label.configure(text="summary.mp4 の場所を特定できません")
            return

        cap = cv2.VideoCapture(summary_path)
        if not cap.isOpened():
            cap.release()
            self._summary_preview_image_label.configure(text="summary.mp4 を開けません")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1 or fps > 120:
            fps = 12.0
        self._summary_preview_fps = float(fps)
        self._summary_preview_capture = cap
        self._update_summary_preview_frame()

    def _open_summary_file_location(self, meta: Dict[str, Any]):
        summary_path = meta.get("resolved_path")
        if not summary_path:
            summary_path = self._resolve_summary_video_path(meta.get("summary_ref", ""))
            if summary_path:
                meta["resolved_path"] = summary_path

        if not summary_path:
            self.append_log("Summary動画の場所を特定できませんでした。")
            return

        target = os.path.abspath(summary_path)
        folder = os.path.dirname(target)
        if not os.path.exists(target):
            self.append_log(f"Summary動画が見つかりません: {target}")
            if folder and os.path.isdir(folder):
                target = folder
            else:
                return

        try:
            if sys.platform.startswith("win"):
                if os.path.isfile(target):
                    select_target = target.replace("/", "\\")
                    subprocess.Popen(["explorer", "/select," + select_target])
                else:
                    os.startfile(target)
            elif sys.platform == "darwin":
                if os.path.isfile(target):
                    subprocess.Popen(["open", "-R", target])
                else:
                    subprocess.Popen(["open", target])
            else:
                open_target = target if os.path.isdir(target) else folder
                subprocess.Popen(["xdg-open", open_target])
        except Exception as e:
            self.append_log(f"ファイルの場所を開けませんでした: {e}")

    def _move_summary_preview(self, event):
        if not self._summary_preview_window or not self._summary_preview_window.winfo_exists():
            return

        self._summary_preview_window.update_idletasks()
        ww = self._summary_preview_window.winfo_width()
        wh = self._summary_preview_window.winfo_height()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()

        if hasattr(self, "log_text") and self.log_text.winfo_exists():
            try:
                lx = self.log_text.winfo_rootx()
                ly = self.log_text.winfo_rooty()
                lw = self.log_text.winfo_width()
                lh = self.log_text.winfo_height()
            except Exception:
                lx = event.x_root
                ly = event.y_root
                lw = 0
                lh = 0
        else:
            lx = event.x_root
            ly = event.y_root
            lw = 0
            lh = 0

        x = lx + lw + 12
        if x + ww > sw - 8:
            x = max(8, lx - ww - 12)

        if ly <= event.y_root <= (ly + lh):
            y = event.y_root - (wh // 3)
        else:
            y = ly + 8
        y = max(8, y)
        if y + wh > sh - 8:
            y = max(8, sh - wh - 8)

        self._summary_preview_window.geometry(f"+{x}+{y}")

    def _update_summary_preview_frame(self):
        cap = self._summary_preview_capture
        label = self._summary_preview_image_label

        if cap is None or label is None or not label.winfo_exists():
            return

        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            if not ok:
                label.configure(text="動画フレームを取得できません")
                return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        max_w, max_h = 360, 202
        scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        if (new_w, new_h) != (w, h):
            frame_rgb = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        photo = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
        self._summary_preview_photo = photo
        label.configure(image=photo, text="")

        delay = int(1000 / max(1.0, self._summary_preview_fps))
        delay = max(30, min(150, delay))
        self._summary_preview_after_id = self.after(delay, self._update_summary_preview_frame)

    def _hide_summary_preview(self):
        self._active_summary_line = None
        self._cancel_summary_preview_hide()

        if self._summary_preview_after_id is not None:
            try:
                self.after_cancel(self._summary_preview_after_id)
            except Exception:
                pass
            self._summary_preview_after_id = None

        if self._summary_preview_capture is not None:
            try:
                self._summary_preview_capture.release()
            except Exception:
                pass
            self._summary_preview_capture = None

        if self._summary_preview_window is not None:
            try:
                if self._summary_preview_window.winfo_exists():
                    self._summary_preview_window.destroy()
            except Exception:
                pass
            self._summary_preview_window = None

        self._summary_preview_title_label = None
        self._summary_preview_image_label = None
        self._summary_preview_open_button = None
        self._summary_preview_photo = None

    def _run_on_main_thread(self, func):
        if threading.current_thread() is threading.main_thread():
            return func()
        result_queue = queue.Queue(maxsize=1)

        def wrapper():
            try:
                result_queue.put((True, func()))
            except Exception as e:
                result_queue.put((False, e))

        self.after(0, wrapper)
        ok, payload = result_queue.get()
        if ok:
            return payload
        raise payload

    @staticmethod
    def _format_size_bytes(size_bytes: int) -> str:
        units = ["B", "KB", "MB", "GB", "TB"]
        size = float(max(0, size_bytes))
        for unit in units:
            if size < 1024 or unit == units[-1]:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size_bytes} B"

    def _estimate_llm_storage_requirements(self, detector_module) -> Dict[str, Any]:
        repo_id = getattr(detector_module, "MODEL_ID", "Qwen/Qwen3-VL-4B-Instruct")
        download_bytes = int(10.0 * (1024 ** 3))
        final_bytes = int(4.5 * (1024 ** 3))
        overhead_bytes = int(1.5 * (1024 ** 3))
        fetched_metadata = False

        try:
            from huggingface_hub import HfApi
            info = HfApi().model_info(repo_id, files_metadata=True)
            file_sizes = [s.size for s in getattr(info, "siblings", []) if getattr(s, "size", None)]
            if file_sizes:
                download_bytes = int(sum(file_sizes))
                final_bytes = max(int(download_bytes * 0.45), int(3.0 * (1024 ** 3)))
                fetched_metadata = True
        except Exception as e:
            self.append_log(f"モデル容量情報の取得に失敗したため既定値で見積もります: {e}")

        temporary_bytes = download_bytes + final_bytes + overhead_bytes
        free_bytes = shutil.disk_usage(os.path.abspath(".")).free

        return {
            "repo_id": repo_id,
            "download_bytes": download_bytes,
            "final_bytes": final_bytes,
            "temporary_bytes": temporary_bytes,
            "free_bytes": free_bytes,
            "fetched_metadata": fetched_metadata,
        }

    def update_start_button_state(self, *args):
        is_running = (self.worker_thread and self.worker_thread.is_alive()) or \
                     (self.rtsp_thread and self.rtsp_thread.is_alive()) or \
                     (self.periodic_scan_thread and self.periodic_scan_thread.is_alive())

        periodic_enabled = self.periodic_scan_var.get()
        periodic_time_limit_enabled = self.periodic_time_limit_var.get()

        try:
            enable = ui_state.should_enable_start(
                is_running=is_running,
                cancel_flag_set=self.cancel_flag.is_set(),
                periodic_enabled=periodic_enabled,
                folder_paths=self.folder_paths,
                rtsp_urls=self.rtsp_urls,
                periodic_time_limit_enabled=periodic_time_limit_enabled,
                start_hour=self.start_hour_var.get(),
                start_min=self.start_min_var.get(),
                end_hour=self.end_hour_var.get(),
                end_min=self.end_min_var.get(),
            )
        except Exception:
            # Fallback conservative behavior
            enable = not is_running and (periodic_enabled or self.folder_paths or self.rtsp_urls)

        self.start_button.config(state=tk.NORMAL if enable else tk.DISABLED)

    def drop(self, event):
        paths = self.splitlist(event.data)
        
        items_to_add = [] # (fps_str, path_str, internal_path)
        
        def get_fps_str(video_path):
            """Get FPS string for a video file."""
            fps_str = "??"
            try:
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    fps_str = f"{fps:.2f}"
                    cap.release()
                else:
                    fps_str = "Error"
            except Exception:
                fps_str = "Error"
            return fps_str
        
        for path in paths:
            if os.path.isdir(path):
                # Scan folder for video files
                video_files = sorted([
                    str(p) for p in Path(path).rglob('*') 
                    if p.suffix.lower() in config.PERIODIC_VIDEO_EXTENSIONS
                ])
                
                if not video_files:
                    continue
                
                # Get FPS for all videos in this folder
                fps_values = []
                for video_path in video_files:
                    fps_values.append(get_fps_str(video_path))
                
                # Check if all FPS values are the same
                unique_fps = set(fps_values)
                if len(unique_fps) == 1 and path not in self.folder_paths:
                    # All same FPS - group as folder
                    fps_str = fps_values[0]
                    path_str = f"{path} ({len(video_files)} files)"
                    items_to_add.append((fps_str, path_str, path))
                else:
                    # Mixed FPS - add individual files
                    for video_path, fps_str in zip(video_files, fps_values):
                        if video_path not in self.folder_paths:
                            items_to_add.append((fps_str, video_path, video_path))
                            
            elif os.path.isfile(path) and Path(path).suffix.lower() in config.PERIODIC_VIDEO_EXTENSIONS:
                if path not in self.folder_paths:
                    fps_str = get_fps_str(path)
                    items_to_add.append((fps_str, path, path))

        if items_to_add:
            for fps_str, path_str, internal_path in items_to_add:
                if internal_path not in self.folder_paths:
                    self.folder_paths.append(internal_path)
                    self._add_folder_item(fps_str, path_str)
            self.update_start_button_state()
        else:
            messagebox.showwarning("情報", "有効なフォルダまたは動画ファイルがドロップされませんでした。")
    
    def _add_folder_item(self, fps_str, path_str):
        """Add a styled item to the folder list with modern FPS badge."""
        index = len(self.folder_item_frames)
        
        item_frame = tk.Frame(self.folder_list_frame, bg="#3A4D6B", cursor="hand2")
        item_frame.pack(fill=tk.X, padx=2, pady=1)
        
        badge_canvas = tk.Canvas(item_frame, width=70, height=22, bg="#3A4D6B", highlightthickness=0)
        badge_canvas.pack(side=tk.LEFT, padx=(4, 6), pady=2)
        
        self._draw_rounded_rect(badge_canvas, 2, 2, 68, 20, 8, fill="#4A90D9", outline="")
        badge_canvas.create_text(35, 11, text=f"{fps_str} fps", fill="white", font=("Segoe UI", 8, "bold"))
        
        # Path label
        path_label = tk.Label(item_frame, text=path_str, bg="#3A4D6B", fg="#EAEAEA", 
                              anchor="w", font=("Segoe UI", 9))
        path_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        
        def on_click(event, idx=index):
            self._toggle_folder_selection(idx)
        
        item_frame.bind("<Button-1>", on_click)
        badge_canvas.bind("<Button-1>", on_click)
        path_label.bind("<Button-1>", on_click)
        
        def on_mousewheel(event):
            self.folder_list_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        item_frame.bind("<MouseWheel>", on_mousewheel)
        badge_canvas.bind("<MouseWheel>", on_mousewheel)
        path_label.bind("<MouseWheel>", on_mousewheel)
        
        self.folder_item_frames.append({
            'frame': item_frame,
            'badge': badge_canvas,
            'label': path_label,
            'selected': False
        })
    
    def _draw_rounded_rect(self, canvas, x1, y1, x2, y2, radius, **kwargs):
        """Draw a rounded rectangle on canvas."""
        points = [
            x1 + radius, y1,
            x2 - radius, y1,
            x2, y1,
            x2, y1 + radius,
            x2, y2 - radius,
            x2, y2,
            x2 - radius, y2,
            x1 + radius, y2,
            x1, y2,
            x1, y2 - radius,
            x1, y1 + radius,
            x1, y1,
            x1 + radius, y1,
        ]
        return canvas.create_polygon(points, smooth=True, **kwargs)
    
    def _toggle_folder_selection(self, index):
        """Toggle selection state of folder item."""
        if index < 0 or index >= len(self.folder_item_frames):
            return
        
        item = self.folder_item_frames[index]
        if item['selected']:
            # Deselect
            item['frame'].config(bg="#3A4D6B")
            item['label'].config(bg="#3A4D6B")
            item['badge'].config(bg="#3A4D6B")
            item['selected'] = False
            self.folder_selected_indices.discard(index)
        else:
            # Select
            item['frame'].config(bg="#5A7D9B")
            item['label'].config(bg="#5A7D9B")
            item['badge'].config(bg="#5A7D9B")
            item['selected'] = True
            self.folder_selected_indices.add(index)

    def remove_selected_folders(self):
        if not self.folder_selected_indices:
            return
        for index in sorted(self.folder_selected_indices, reverse=True):
            if 0 <= index < len(self.folder_paths):
                del self.folder_paths[index]
                item = self.folder_item_frames.pop(index)
                item['frame'].destroy()
        self.folder_selected_indices.clear()
        for i, item in enumerate(self.folder_item_frames):
            def make_click_handler(idx):
                return lambda e: self._toggle_folder_selection(idx)
            item['frame'].bind("<Button-1>", make_click_handler(i))
            item['badge'].bind("<Button-1>", make_click_handler(i))
            item['label'].bind("<Button-1>", make_click_handler(i))
        self.update_start_button_state()

    def remove_all_folders(self):
        if not self.folder_paths: return
        if messagebox.askyesno("確認", "リストからすべてのフォルダを削除しますか？"):
            self.folder_paths.clear()
            for item in self.folder_item_frames:
                item['frame'].destroy()
            self.folder_item_frames.clear()
            self.folder_selected_indices.clear()
            self.update_start_button_state()

    def add_rtsp_url(self):
        url = self.rtsp_url_var.get().strip()
        if url and url not in self.rtsp_urls:
            self.rtsp_urls.append(url)
            self._add_rtsp_item(url)
            self.rtsp_url_var.set("")
            self.update_start_button_state()
        elif not url:
            messagebox.showwarning("入力エラー", "RTSP URLを入力してください。")
    
    def _add_rtsp_item(self, url):
        """Add a styled item to the RTSP list with modern badge."""
        index = len(self.rtsp_item_frames)
        
        item_frame = tk.Frame(self.rtsp_list_frame, bg="#3A4D6B", cursor="hand2")
        item_frame.pack(fill=tk.X, padx=2, pady=1)
        
        badge_canvas = tk.Canvas(item_frame, width=55, height=22, bg="#3A4D6B", highlightthickness=0)
        badge_canvas.pack(side=tk.LEFT, padx=(4, 6), pady=2)
        
        self._draw_rounded_rect(badge_canvas, 2, 2, 53, 20, 8, fill="#2ECC71", outline="")
        badge_canvas.create_text(27, 11, text="RTSP", fill="white", font=("Segoe UI", 8, "bold"))
        
        url_label = tk.Label(item_frame, text=url, bg="#3A4D6B", fg="#EAEAEA", 
                             anchor="w", font=("Segoe UI", 9))
        url_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        
        def on_click(event, idx=index):
            self._toggle_rtsp_selection(idx)
        
        item_frame.bind("<Button-1>", on_click)
        badge_canvas.bind("<Button-1>", on_click)
        url_label.bind("<Button-1>", on_click)
        
        def on_mousewheel(event):
            self.rtsp_list_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        item_frame.bind("<MouseWheel>", on_mousewheel)
        badge_canvas.bind("<MouseWheel>", on_mousewheel)
        url_label.bind("<MouseWheel>", on_mousewheel)
        
        self.rtsp_item_frames.append({
            'frame': item_frame,
            'badge': badge_canvas,
            'label': url_label,
            'selected': False
        })
    
    def _toggle_rtsp_selection(self, index):
        """Toggle selection state of RTSP item."""
        if index < 0 or index >= len(self.rtsp_item_frames):
            return
        
        item = self.rtsp_item_frames[index]
        if item['selected']:
            item['frame'].config(bg="#3A4D6B")
            item['label'].config(bg="#3A4D6B")
            item['badge'].config(bg="#3A4D6B")
            item['selected'] = False
            self.rtsp_selected_indices.discard(index)
        else:
            item['frame'].config(bg="#5A7D9B")
            item['label'].config(bg="#5A7D9B")
            item['badge'].config(bg="#5A7D9B")
            item['selected'] = True
            self.rtsp_selected_indices.add(index)

    def remove_selected_rtsp(self):
        if not self.rtsp_selected_indices:
            return
        for index in sorted(self.rtsp_selected_indices, reverse=True):
            if 0 <= index < len(self.rtsp_urls):
                del self.rtsp_urls[index]
                item = self.rtsp_item_frames.pop(index)
                item['frame'].destroy()
        self.rtsp_selected_indices.clear()
        for i, item in enumerate(self.rtsp_item_frames):
            def make_click_handler(idx):
                return lambda e: self._toggle_rtsp_selection(idx)
            item['frame'].bind("<Button-1>", make_click_handler(i))
            item['badge'].bind("<Button-1>", make_click_handler(i))
            item['label'].bind("<Button-1>", make_click_handler(i))
        self.update_start_button_state()

    def remove_all_rtsp(self):
        if not self.rtsp_urls: return
        if messagebox.askyesno("確認", "すべてのRTSP URLを削除しますか？"):
            self.rtsp_urls.clear()
            for item in self.rtsp_item_frames:
                item['frame'].destroy()
            self.rtsp_item_frames.clear()
            self.rtsp_selected_indices.clear()
            self.update_start_button_state()

    def select_periodic_dir(self):
        dir_selected = filedialog.askdirectory(title="監視するフォルダを選択")
        if dir_selected: self.periodic_dir_var.set(dir_selected)

    def toggle_time_limit_frame(self):
        if self.periodic_time_limit_var.get():
            self.time_limit_frame.pack(fill=tk.X, pady=5, padx=20)
        else:
            self.time_limit_frame.pack_forget()
        self.update_start_button_state()

    def toggle_rtsp_time_limit_frame(self):
        """Toggle visibility of the RTSP time limit detail frame."""
        if self.rtsp_time_limit_var.get():
            self.rtsp_time_limit_detail_frame.pack(fill=tk.X, pady=5, padx=20)
        else:
            self.rtsp_time_limit_detail_frame.pack_forget()

    def fetch_current_location_rtsp(self):
        """Fetch current location and auto-set RTSP recording time based on sunset/sunrise."""
        threading.Thread(target=self._fetch_current_location_rtsp_thread, daemon=True).start()

    def _fetch_current_location_rtsp_thread(self):
        try:
            lat, lon = location_utils.get_current_location()
        except Exception as e:
            print(f"fetch_current_location_rtsp: unexpected error: {e}")
            lat, lon = 35.0, 135.0

        try:
            self.after(0, lambda: self.append_log(f"RTSP時間設定: 位置情報取得 (緯度={lat}, 経度={lon})"))
        except Exception:
            pass

        try:
            period = sun_times.compute_night_period(lat, lon)
            start_dt = period.get('start')
            end_dt = period.get('end')
            if start_dt:
                sh, sm = start_dt.hour, start_dt.minute
                self.after(0, lambda: self.rtsp_start_hour_var.set(f"{sh:02d}"))
                self.after(0, lambda: self.rtsp_start_min_var.set(f"{sm:02d}"))
                self.after(0, lambda: self.append_log(f"RTSP時間設定: 録画開始時刻={sh:02d}:{sm:02d}"))
            if end_dt:
                eh, em = end_dt.hour, end_dt.minute
                self.after(0, lambda: self.rtsp_end_hour_var.set(f"{eh:02d}"))
                self.after(0, lambda: self.rtsp_end_min_var.set(f"{em:02d}"))
                self.after(0, lambda: self.append_log(f"RTSP時間設定: 録画終了時刻={eh:02d}:{em:02d}"))
        except Exception as e:
            print(f"compute_night_period for RTSP failed: {e}")

    def fetch_current_location(self):
        """Start background thread to fetch current location and print the result.

        Uses div/location_utils.py (imported as location_utils). If retrieval fails,
        the helper returns the default coordinates (35.0, 135.0).
        """
        threading.Thread(target=self._fetch_current_location_thread, daemon=True).start()

    def _fetch_current_location_thread(self):
        try:
            lat, lon = location_utils.get_current_location()
        except Exception as e:
            print(f"fetch_current_location: unexpected error: {e}")
            lat, lon = 35.0, 135.0

        try:
            self.after(0, lambda: self.current_lat_var.set(f"{lat:.6f}"))
            self.after(0, lambda: self.current_lon_var.set(f"{lon:.6f}"))
        except Exception:
            pass

        print(f"Current location: lat={lat}, lon={lon}")
        try:
            self.after(0, lambda: self.append_log(f"取得した位置情報: 緯度={lat}, 経度={lon}"))
        except Exception:
            pass

        # compute sunrise/sunset and astronomical twilight for the obtained location
        try:
            times = sun_times.get_sun_times(lat, lon)
            def fmt(dt):
                return dt.strftime('%Y-%m-%d %H:%M:%S') if dt else 'N/A'

            print("Computed sun times:")
            print(f"  Sunrise: {fmt(times.get('sunrise'))}")
            print(f"  Sunset: {fmt(times.get('sunset'))}")
            print(f"  Astronomical dawn (astro start): {fmt(times.get('astro_dawn'))}")
            print(f"  Astronomical dusk (astro end): {fmt(times.get('astro_dusk'))}")

            try:
                self.after(0, lambda: self.append_log(f"計算: 日の出={fmt(times.get('sunrise'))}, 日没={fmt(times.get('sunset'))}"))
                self.after(0, lambda: self.append_log(f"計算: 天文薄明開始={fmt(times.get('astro_dawn'))}, 終了={fmt(times.get('astro_dusk'))}"))
            except Exception:
                pass
        except Exception as e:
            print(f"sun_times calculation failed: {e}")

        # compute suggested nightly start/end (midpoints) using sun_times helper
        try:
            period = sun_times.compute_night_period(lat, lon)
            start_dt = period.get('start')
            end_dt = period.get('end')
            if start_dt:
                sh, sm = start_dt.hour, start_dt.minute
                self.after(0, lambda: self.start_hour_var.set(f"{sh:02d}"))
                self.after(0, lambda: self.start_min_var.set(f"{sm:02d}"))
                print(f"Auto-set start time to {sh:02d}:{sm:02d} (midpoint sunset/astro_dusk)")
                self.after(0, lambda: self.append_log(f"自動設定: 開始時刻={sh:02d}:{sm:02d}"))

            if end_dt:
                eh, em = end_dt.hour, end_dt.minute
                self.after(0, lambda: self.end_hour_var.set(f"{eh:02d}"))
                self.after(0, lambda: self.end_min_var.set(f"{em:02d}"))
                print(f"Auto-set end time to {eh:02d}:{em:02d} (midpoint sunrise/astro_dawn next day)")
                self.after(0, lambda: self.append_log(f"自動設定: 終了時刻={eh:02d}:{em:02d}"))
        except Exception as e:
            print(f"compute_night_period failed: {e}")

    def toggle_auto_time_updater(self):
        """自動更新の有効/無効を切り替え"""
        if self.auto_time_updater_enabled_var.get():
            self.auto_updater.start()
        else:
            self.auto_updater.stop()
    
    def _on_auto_time_update(self, start_hour: int, start_min: int, end_hour: int, end_min: int):
        """
        自動更新時に呼び出されるコールバック
        GUIの時刻設定を更新する
        """
        def update_gui():
            self.start_hour_var.set(f"{start_hour:02d}")
            self.start_min_var.set(f"{start_min:02d}")
            self.end_hour_var.set(f"{end_hour:02d}")
            self.end_min_var.set(f"{end_min:02d}")
        
        # メインスレッドで実行
        self.after(0, update_gui)

    def toggle_summary_settings_button(self, *args):
        if hasattr(self, 'btn_summary_settings'):
            state = tk.NORMAL if self.save_options_vars['summary'].get() else tk.DISABLED
            self.btn_summary_settings.config(state=state)

    def select_save_path(self, path_var):
        directory = filedialog.askdirectory(title="保存先を選択", initialdir=path_var.get())
        if directory: path_var.set(directory)

    def select_plate_solve_video(self):
        file_path = filedialog.askopenfilename(title="プレートソルブ用動画を選択", filetypes=[("動画ファイル", "*.mp4 *.avi *.mov"), ("すべてのファイル", "*.*")])
        if file_path: self.plate_solve_video_path_var.set(file_path)

    def select_plate_solve_wcs_file(self):
        file_path = filedialog.askopenfilename(title="既存のWCSファイルを選択", filetypes=[("WCS/FITSファイル", "*.wcs *.fits"), ("すべてのファイル", "*.*")])
        if file_path:
            try:
                ps_datetime = None
                # まずWCSファイル(FITS)のヘッダーから'DATE-OBS'を読み込もうと試みる
                try:
                    with fits.open(file_path) as hdul:
                        header = hdul[0].header
                        if not WCS(header).is_celestial:
                            raise ValueError("有効な天球WCSではありません。")
                        
                        if 'DATE-OBS' in header:
                            date_obs_str = header['DATE-OBS']
                            ps_datetime = datetime.fromisoformat(date_obs_str)
                            print(f"WCSヘッダーから基準時刻を読み込みました: {ps_datetime}")
                except Exception as fits_e:
                    print(f"FITSヘッダーの読み込みまたは解析に失敗しました: {fits_e}")
                    # FITSとして開けなかった場合や'DATE-OBS'がない場合は、従来の方法に進む
                    pass

                # ヘッダーから時刻が取得できなかった場合、ファイルパスから抽出を試みる
                if ps_datetime is None:
                    print("WCSヘッダーに基準時刻が見つからないため、ファイルパスから推定します。")
                    ps_datetime = astrometry.extract_datetime_from_file_path(file_path)

                # それでも時刻が取得できない場合、最終手段として現在時刻を使用する
                if ps_datetime is None:
                    print("ファイルパスからも基準時刻を推定できませんでした。現在時刻を使用します。")
                    ps_datetime = datetime.now()

                self.global_wcs_info = {'wcs_file': file_path, 'plate_solve_datetime': ps_datetime}
                self.plate_solve_wcs_path_var.set(file_path)
                self.plate_solve_status_var.set(f"プレートソルブ: 成功 (既存WCS) @ {ps_datetime.strftime('%H:%M')}")
                messagebox.showinfo("成功", f"既存WCSファイルをロードしました。\n参照時刻: {ps_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
                self.update_start_button_state()

            except Exception as e:
                self.global_wcs_info = None
                self.plate_solve_wcs_path_var.set("")
                self.plate_solve_status_var.set("プレートソルブ: 失敗")
                messagebox.showerror("エラー", f"WCSファイルのロード/検証に失敗しました:\n{e}")

    def start_plate_solve(self):
        self.apply_advanced_settings_to_config()
        threading.Thread(target=self.execute_plate_solve_thread, daemon=True).start()

    def start_rtsp_plate_solve(self):
        """RTSPストリームからプレートソルブを実行する"""
        self.apply_advanced_settings_to_config()
        # 選択されているRTSP URLを取得、選択がなければ最初のURLを使用
        if self.rtsp_selected_indices:
            selected_index = min(self.rtsp_selected_indices)
            rtsp_url = self.rtsp_urls[selected_index]
        elif self.rtsp_urls:
            rtsp_url = self.rtsp_urls[0]
        else:
            messagebox.showwarning("警告", "RTSPストリームを追加してください。")
            return
        threading.Thread(target=self.execute_rtsp_plate_solve_thread, args=(rtsp_url,), daemon=True).start()

    def execute_rtsp_plate_solve_thread(self, rtsp_url: str):
        """RTSPストリームからフレームを取得してプレートソルブを実行するスレッド"""
        self.plate_solve_status_var.set("プレートソルブ: RTSP接続中...")
        self.progress_queue.put((f"RTSPプレートソルブを実行中: {rtsp_url}", None))
        try:
            cap = utils.create_rtsp_capture(rtsp_url)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
            
            if not cap.isOpened():
                raise IOError(f"RTSPストリームを開けません: {rtsp_url}")
            
            self.plate_solve_status_var.set("プレートソルブ: フレーム取得中...")
            
            # 約10秒分のフレームを取得（25fps前提で250フレーム）
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or fps > 120:
                fps = config.RTSP_FPS  # デフォルトのRTSP FPSを使用
            num_frames = int(fps * 10)
            
            frames = []
            for _ in range(num_frames):
                ret, frame = cap.read()
                if ret and frame is not None:
                    frames.append(frame)
                else:
                    # フレーム取得に失敗した場合、少し待って再試行
                    time.sleep(0.01)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        frames.append(frame)
            cap.release()
            
            if len(frames) < 10:
                raise ValueError(f"RTSPストリームから十分なフレームを取得できませんでした。取得フレーム数: {len(frames)}")
            
            self.plate_solve_status_var.set("プレートソルブ: 合成画像作成中...")
            self.progress_queue.put((f"RTSPから{len(frames)}フレームを取得しました。合成画像を作成中...", None))
            
            composite_image = np.max(np.array(frames), axis=0).astype(np.uint8)
            temp_composite_path = os.path.join(config.TEMP_CLIP_DIR, f"rtsp_composite_{time.time_ns()}.jpg")
            os.makedirs(config.TEMP_CLIP_DIR, exist_ok=True)
            cv2.imwrite(temp_composite_path, composite_image)
            
            self.plate_solve_status_var.set("プレートソルブ: 実行中...")
            self.progress_queue.put(("Astrometry.netにアップロード中...", None))
            
            # RTSPプレートソルブでは検出マスク（RTSPから作成したマスク）を使用
            rtsp_mask = self.mask_image if self.apply_mask_var.get() else None
            use_local_solver = (self.plate_solve_mode_var.get() == "local")
            plate_solve_result = astrometry.plate_solve_image(
                temp_composite_path, mask=rtsp_mask,
                plate_solve_video_path=rtsp_url, cancel_flag=self.cancel_flag,
                scale_lower=config.RTSP_SCALE_LOWER, scale_upper=config.RTSP_SCALE_UPPER,
                use_local=use_local_solver
            )
            if os.path.exists(temp_composite_path):
                os.remove(temp_composite_path)
            
            if plate_solve_result and 'wcs_file' in plate_solve_result:
                self.global_wcs_info = plate_solve_result
                ps_datetime = self.global_wcs_info.get('plate_solve_datetime', datetime.now())
                self.plate_solve_status_var.set(f"プレートソルブ: 成功 (RTSP) @ {ps_datetime.strftime('%H:%M')}")
                self.plate_solve_wcs_path_var.set(self.global_wcs_info['wcs_file'])
                self.progress_queue.put((f"RTSPプレートソルブ成功: {self.global_wcs_info['wcs_file']}", None))
                messagebox.showinfo("成功", f"RTSPからのプレートソルブに成功しました。\n参照時刻: {ps_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
                self.update_start_button_state()
            else:
                self.global_wcs_info = None
                self.plate_solve_status_var.set("プレートソルブ: 失敗")
                self.progress_queue.put(("RTSPプレートソルブ失敗", None))
                messagebox.showerror("失敗", "RTSPからのプレートソルブに失敗しました。\nストリーム内容、ネットワーク、APIキーを確認してください。")
                
        except Exception as e:
            self.global_wcs_info = None
            self.plate_solve_status_var.set("プレートソルブ: エラー")
            error_message = f"RTSPプレートソルブ中にエラーが発生しました: {e}"
            self.progress_queue.put((error_message, None))
            messagebox.showerror("エラー", error_message)

    def execute_plate_solve_thread(self):
        video_file_path = self.plate_solve_video_path_var.get()
        if not video_file_path:
            messagebox.showwarning("警告", "プレートソルブに使用する動画を選択してください。")
            self.plate_solve_status_var.set("プレートソルブ: 未実行")
            return

        self.plate_solve_status_var.set("プレートソルブ: 実行中...")
        self.progress_queue.put(("プレートソルブを実行中...", None))
        try:
            cap = cv2.VideoCapture(video_file_path)
            if not cap.isOpened(): raise IOError("動画ファイルを開けません。")
            fps = cap.get(cv2.CAP_PROP_FPS) or config.DEFAULT_FPS
            num_frames = int(fps * 10)
            frames = [cap.read()[1] for _ in range(num_frames) if cap.isOpened() and cap.read()[0]]
            cap.release()
            if not frames: raise ValueError("動画からフレームを取得できませんでした。")
            
            composite_image = np.max(np.array(frames), axis=0).astype(np.uint8)
            temp_composite_path = os.path.join(config.TEMP_CLIP_DIR, f"temp_composite_{time.time_ns()}.jpg")
            os.makedirs(config.TEMP_CLIP_DIR, exist_ok=True)
            cv2.imwrite(temp_composite_path, composite_image)

            use_local_solver = (self.plate_solve_mode_var.get() == "local")
            plate_solve_result = astrometry.plate_solve_image(
                temp_composite_path, mask=self.plate_solve_mask_image,
                plate_solve_video_path=video_file_path, cancel_flag=self.cancel_flag,
                use_local=use_local_solver
            )
            if os.path.exists(temp_composite_path): os.remove(temp_composite_path)

            if plate_solve_result and 'wcs_file' in plate_solve_result:
                self.global_wcs_info = plate_solve_result
                ps_datetime = self.global_wcs_info.get('plate_solve_datetime', datetime.now())
                self.plate_solve_status_var.set(f"プレートソルブ: 成功 @ {ps_datetime.strftime('%H:%M')}")
                self.plate_solve_wcs_path_var.set(self.global_wcs_info['wcs_file'])
                self.progress_queue.put((f"プレートソルブ成功: {self.global_wcs_info['wcs_file']}", None))
                messagebox.showinfo("成功", f"プレートソルブに成功しました。\n参照時刻: {ps_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
                self.update_start_button_state()
            else:
                self.global_wcs_info = None
                self.plate_solve_status_var.set("プレートソルブ: 失敗")
                self.progress_queue.put(("プレートソルブ失敗", None))
                messagebox.showerror("失敗", "プレートソルブに失敗しました。APIキー、ネットワーク、画像内容を確認してください。")

        except Exception as e:
            self.global_wcs_info = None
            self.plate_solve_status_var.set("プレートソルブ: エラー")
            error_message = f"プレートソルブ中にエラーが発生しました: {e}"
            self.progress_queue.put((error_message, None))
            messagebox.showerror("エラー", error_message)

    def on_closing(self):
        if messagebox.askokcancel("終了", "アプリケーションを終了しますか？"):
            self.append_log("設定を保存しています...")
            self._hide_summary_preview()
            self.save_settings()
            # 自動更新を停止
            if self.auto_updater:
                self.auto_updater.stop()
            self.cancel_flag.set()
            self.destroy()

    def start_processing(self):
        # 詳細設定をconfigに適用
        self.apply_advanced_settings_to_config()
        if not self.apply_selected_model(silent=True):
            messagebox.showerror("設定エラー", "有効な学習モデルを選択してください。")
            return
        if (self.worker_thread and self.worker_thread.is_alive()) or \
           (self.rtsp_thread and self.rtsp_thread.is_alive()) or \
           (self.periodic_scan_thread and self.periodic_scan_thread.is_alive()):
            messagebox.showwarning("情報", "別のプロセスが実行中です。")
            return

        self.cancel_flag.clear()
        self.append_log("処理準備中...")

        try:
            params = {
                'max_workers': int(self.concurrency_var.get()),
                'interval_sec': float(self.interval_var.get()),
                'duration_sec': float(self.duration_var.get()),
                'save_options': {k: v.get() for k, v in self.save_options_vars.items()},
                'meteor_save_path': self.meteor_save_path_var.get(),
                'not_meteor_save_path': self.not_meteor_save_path_var.get(),
                'mask': self.mask_image if self.apply_mask_var.get() else None,
                'global_wcs_info': self.global_wcs_info if self.use_plate_solve_var.get() else None,
                'plate_solve_mask': self.plate_solve_mask_image,
                'summary_config': [item.copy() for item in self.summary_video_config]
            }
            
            if self.rtsp_preset_var.get() == "clear":
                preset = config.RTSP_PRESET_CLEAR_SKY
            else:
                preset = config.RTSP_PRESET_CLOUDY
            config.RTSP_MIN_LINE_LENGTH = preset['min_line_length']
            config.RTSP_HOUGH_THRESHOLD = preset['hough_threshold']
            config.RTSP_CANNY_THRESH1 = preset['canny_thresh1']
            config.RTSP_CANNY_THRESH2 = preset['canny_thresh2']
            
            try:
                config.RTSP_FPS = int(self.rtsp_fps_var.get())
            except ValueError:
                messagebox.showwarning("設定警告", f"FPS値が無効です。デフォルト値({config.RTSP_FPS})を使用します。")
            
            self.append_log(f"RTSP検出プリセット: {preset['name']}, FPS: {config.RTSP_FPS}")
            os.makedirs(params['meteor_save_path'], exist_ok=True)
            os.makedirs(params['not_meteor_save_path'], exist_ok=True)
            # set temp_video dir path on the App instance so GUI can shorten logs
            module_dir = os.path.dirname(os.path.abspath(__file__))
            self.temp_video_dir = os.path.join(module_dir, 'temp_video')
            os.makedirs(self.temp_video_dir, exist_ok=True)
        except (ValueError, Exception) as e:
            messagebox.showerror("設定エラー", f"パラメータ値が無効です: {e}")
            return

        self.start_button.config(state=tk.DISABLED)
        self.cancel_button.config(state=tk.NORMAL)
        self.status_label.config(text="処理中...")
        self.progress['value'] = 0
        self.eta_label.config(text="ETA: 計算中...")
        self.elapsed_label.config(text="経過: 00:00:00")
        self.start_time_gui = time.time()

        is_periodic = self.periodic_scan_var.get()

        if is_periodic:
            periodic_dir = self.periodic_dir_var.get().strip()
            if not periodic_dir or not os.path.isdir(periodic_dir):
                messagebox.showerror("設定エラー", "定期スキャン用の有効な監視フォルダを選択してください。")
                self.cancel_processing(restore_button_state=True)
                return
            
            log_msg = f"定期スキャン開始 (フォルダ: {periodic_dir})"
            monitor_kwargs = {
                'directory': periodic_dir, 'scan_interval': int(self.periodic_interval_var.get()),
                'progress_callback': self.progress_queue.put, 'mask': params['mask'], 
                'global_wcs_info': params['global_wcs_info'], 'plate_solve_mask': params['plate_solve_mask'], 
                'meteor_save_path': params['meteor_save_path'], 'not_meteor_save_path': params['not_meteor_save_path'], 
                'cancel_flag': self.cancel_flag, 'save_options': params['save_options'], 
                'interval': params['interval_sec'], 'duration': params['duration_sec'], 
                'min_length': config.MIN_LINE_LENGTH, 'summary_video_config': params['summary_config'],
                'time_limit_enabled': self.periodic_time_limit_var.get(),
                'start_hour': int(self.start_hour_var.get()), 'start_minute': int(self.start_min_var.get()),
                'end_hour': int(self.end_hour_var.get()), 'end_minute': int(self.end_min_var.get())
            }
            if monitor_kwargs['time_limit_enabled']:
                log_msg += f", 時間制限: {monitor_kwargs['start_hour']:02d}:{monitor_kwargs['start_minute']:02d} - {monitor_kwargs['end_hour']:02d}:{monitor_kwargs['end_minute']:02d}"
            self.append_log(log_msg)

            self.periodic_scan_thread = threading.Thread(target=file_utils.monitor_directory, kwargs=monitor_kwargs, daemon=True)
            self.periodic_scan_thread.start()

        elif self.rtsp_urls:
            url = self.rtsp_urls[0]
            rtsp_time_limit = self.rtsp_time_limit_var.get()
            rtsp_sh = int(self.rtsp_start_hour_var.get())
            rtsp_sm = int(self.rtsp_start_min_var.get())
            rtsp_eh = int(self.rtsp_end_hour_var.get())
            rtsp_em = int(self.rtsp_end_min_var.get())
            
            log_msg = f"RTSP処理開始 (URL: {url}, 並列処理数: {params['max_workers']})"
            if rtsp_time_limit:
                log_msg += f", 録画時間制限: {rtsp_sh:02d}:{rtsp_sm:02d} - {rtsp_eh:02d}:{rtsp_em:02d}"
            self.append_log(log_msg)
            
            rtsp_args = (
                url, config.RTSP_SAVE_ROOT, config.RTSP_SEGMENT_DURATION, 60, self.progress_queue.put,
                params['mask'], params['global_wcs_info'], params['plate_solve_mask'],
                params['meteor_save_path'], params['not_meteor_save_path'], self.cancel_flag,
                params['save_options'], params['interval_sec'], params['duration_sec'],
                config.MIN_LINE_LENGTH, params['summary_config'],
                rtsp_time_limit, rtsp_sh, rtsp_sm, rtsp_eh, rtsp_em,
                params['max_workers']
            )
            self.rtsp_thread = threading.Thread(target=file_utils.rtsp_save_and_process_thread_target, args=rtsp_args, daemon=True)
            self.rtsp_thread.start()

        elif self.folder_paths:
            sources_to_process = []
            self.append_log(f"{len(self.folder_paths)}個の項目を処理します...")
            for path_item in self.folder_paths:
                p = Path(path_item)
                if p.is_dir():
                    found = sorted([p for p in p.rglob('*') if p.suffix.lower() in config.PERIODIC_VIDEO_EXTENSIONS])
                    sources_to_process.extend([{'path': str(fp), 'is_rtsp': False} for fp in found])
                elif p.is_file() and p.suffix.lower() in config.PERIODIC_VIDEO_EXTENSIONS:
                    sources_to_process.append({'path': str(p), 'is_rtsp': False})
            
            if not sources_to_process:
                messagebox.showwarning("情報", "選択されたフォルダに動画ファイルが見つかりませんでした。")
                self.cancel_processing(restore_button_state=True)
                return

            total_videos = len(sources_to_process)
            self.append_log(f"合計 {total_videos} 個の動画ファイルを処理します。")
            self.progress['maximum'] = total_videos
            self.progress_queue.put((f"処理開始 ({total_videos} ファイル)", (0, total_videos)))

            worker_args = (
                self.progress_queue, sources_to_process, params['max_workers'], params['interval_sec'], 
                params['duration_sec'], params['mask'], params['global_wcs_info'], params['plate_solve_mask'],
                params['meteor_save_path'], params['not_meteor_save_path'], self.cancel_flag,
                params['save_options'], params['summary_config']
            )
            self.worker_thread = threading.Thread(target=worker_main_loop, args=worker_args, daemon=True)
            self.worker_thread.start()
        else:
            messagebox.showerror("エラー", "処理対象がありません。")
            self.cancel_processing(restore_button_state=True)

    def cancel_processing(self, restore_button_state=False):
        # リクエストが来たら直ちにキャンセルフラグを立て、UIを更新する
        if not self.cancel_flag.is_set():
            self.append_log("キャンセル要求を受け付けました...")
        else:
            self.append_log("キャンセル要求 (再送) ...")

        # Notify workers and update UI state immediately
        self.cancel_flag.set()
        try:
            self.cancel_button.config(state=tk.DISABLED)
        except Exception:
            pass
        self.status_label.config(text="キャンセル中...")

        self.start_time_gui = None

        # allow Start to be pressed again immediately after cancel requested
        try:
            self.update_start_button_state()
        except Exception:
            pass

        if restore_button_state:
            # restore start button state and label when requested by caller
            self.update_start_button_state()
            self.status_label.config(text="停止")

    def update_progress(self):
        if self.start_time_gui:
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - self.start_time_gui))
            self.elapsed_label.config(text=f"経過: {elapsed_str}")

        try:
            while True:
                message, value = self.progress_queue.get_nowait()
                if isinstance(value, tuple) and len(value) == 2:
                    current, total = value
                    if total > 0:
                        self.progress['maximum'] = total
                        self.progress['value'] = max(0, min(current, total))
                        self.status_label.config(text=f"処理中... ({int(self.progress['value'])}/{int(self.progress['maximum'])})")
                if self.start_time_gui and self.progress['maximum'] > 0 and self.progress['value'] > 0:
                    elapsed = time.time() - self.start_time_gui
                    avg_time = elapsed / self.progress['value']
                    eta_sec = avg_time * (self.progress['maximum'] - self.progress['value'])
                    self.eta_label.config(text=f"ETA: {time.strftime('%H:%M:%S', time.gmtime(eta_sec))}")

                if message:
                    # Shorten messages that reference temporary copied files
                    msg = message
                    try:
                        tmp_root = getattr(self, 'temp_video_dir', None)
                        if tmp_root and isinstance(msg, str):
                            idx = msg.find(tmp_root)
                            if idx != -1:
                                # find bounds of path in the message (try quotes first)
                                start_q = msg.rfind('"', 0, idx)
                                end_q = msg.find('"', idx)
                                if start_q == -1:
                                    start = idx
                                else:
                                    start = start_q + 1
                                if end_q == -1:
                                    # fallback: space or end
                                    space_pos = msg.find(' ', idx)
                                    end = space_pos if (space_pos != -1) else len(msg)
                                else:
                                    end = end_q

                                full_path = msg[idx:end]
                                norm = os.path.normpath(full_path)
                                parts = norm.split(os.sep)
                                # find netcopy_<id> segment
                                net_idx = next((i for i, p in enumerate(parts) if p.startswith('netcopy_')), None)
                                if net_idx is not None and len(parts) > net_idx + 2:
                                    # skip netcopy and drive-name segments
                                    simp_parts = parts[net_idx + 2:]
                                    simp = os.sep.join(simp_parts)
                                else:
                                    simp = os.path.basename(norm)

                                # replace the full path in the message with the simplified path (preserve quotes if present)
                                if start_q != -1 and end_q != -1:
                                    msg = msg[:start_q+1] + simp + msg[end_q:]
                                else:
                                    msg = msg.replace(full_path, simp)
                    except Exception:
                        pass

                    self.append_log(msg)
                    # Consider the run complete only on explicit completion/cancel messages.
                    # Avoid treating transient error words in exception text as 'complete',
                    # because many libraries surface English words like 'failed' in tracebacks
                    # and that would erroneously stop ETA updates.
                    is_complete = (
                        "すべての処理が完了しました" in message or
                        "監視を終了しました" in message or
                        "統合処理終了" in message or
                        "処理はキャンセルされました" in message
                    )
                    if is_complete:
                        self.update_start_button_state()
                        self.cancel_button.config(state=tk.DISABLED)
                        self.status_label.config(text="完了/停止")
                        if "すべての処理が完了しました" in message:
                            self.progress['value'] = self.progress['maximum']
                        self.start_time_gui = None
                        self.cancel_flag.clear()
        except queue.Empty:
            pass

        self.after(100, self.update_progress)

    def create_summary_settings_window(self):
        win = Toplevel(self)
        win.title("概要動画設定")
        win.geometry("500x450")
        win.grab_set()
        win.transient(self)

        temp_config = [item.copy() for item in self.summary_video_config]
        
        ttk.Label(win, text="概要動画に含める項目と順序:").pack(pady=5, padx=10, anchor='w')
        list_frame = ttk.Frame(win); list_frame.pack(fill=tk.BOTH, expand=True, padx=10)
        listbox = tk.Listbox(list_frame, selectmode=tk.SINGLE, exportselection=False,
                             bg="#3A4D6B", fg="#EAEAEA", selectbackground="#5A7AA9", highlightthickness=0)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        check_vars = [tk.BooleanVar(value=item['enabled']) for item in temp_config]
        for i, item in enumerate(temp_config):
            listbox.insert(tk.END, item['name'])

        def move_item(direction):
            idx = listbox.curselection()[0] if listbox.curselection() else -1
            if idx == -1: return
            new_idx = idx + direction
            if 0 <= new_idx < listbox.size():
                item = listbox.get(idx)
                listbox.delete(idx); listbox.insert(new_idx, item)
                listbox.selection_set(new_idx); listbox.activate(new_idx)
                temp_config.insert(new_idx, temp_config.pop(idx))
                check_vars.insert(new_idx, check_vars.pop(idx))

        btn_panel = ttk.Frame(list_frame); btn_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(5,0))
        ttk.Button(btn_panel, text="↑", command=lambda: move_item(-1)).pack(pady=2)
        ttk.Button(btn_panel, text="↓", command=lambda: move_item(1)).pack(pady=2)

        check_frame = ttk.Frame(win); check_frame.pack(fill=tk.X, padx=10, pady=5)
        for i, item in enumerate(temp_config):
            ttk.Checkbutton(check_frame, text=item['name'], variable=check_vars[i]).pack(anchor='w')

        def on_ok():
            for i, var in enumerate(check_vars):
                temp_config[i]['enabled'] = var.get()
            self.summary_video_config = temp_config
            self.append_log("概要動画の設定を更新しました。")
            win.destroy()

        ok_cancel_frame = ttk.Frame(win); ok_cancel_frame.pack(pady=10)
        ttk.Button(ok_cancel_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(ok_cancel_frame, text="キャンセル", command=win.destroy).pack(side=tk.LEFT, padx=5)

    def create_mask_window(self, is_plate_solve_mask: bool):
        base_image_path = None
        is_rtsp_source = False
        if is_plate_solve_mask:
            base_image_path = self.plate_solve_video_path_var.get()
            if not base_image_path:
                messagebox.showwarning("情報", "まずプレートソルブ用の動画を選択してください。")
                return
            window_title = "プレートソルブ用マスク作成"
        else:
            window_title = "検出マスク作成"
            source_folders = self.folder_paths or ([self.periodic_dir_var.get()] if self.periodic_scan_var.get() and self.periodic_dir_var.get() else [])
            for folder in source_folders:
                videos = sorted([p for p in Path(folder).rglob('*') if p.suffix.lower() in config.PERIODIC_VIDEO_EXTENSIONS])
                if videos:
                    base_image_path = str(videos[0])
                    break
            
            # フォルダに動画がなくRTSP URLがある場合はRTSPから取得
            if not base_image_path and self.rtsp_urls:
                base_image_path = self.rtsp_urls[0]
                is_rtsp_source = True
        
        if not base_image_path:
            messagebox.showwarning("情報", "マスク作成の元となる動画ソースが見つかりません。")
            return

        try:
            cap = cv2.VideoCapture(base_image_path)
            if is_rtsp_source:
                # RTSPのタイムアウト設定
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise ValueError("動画またはRTSPストリームからフレームを読み込めませんでした。")
        except Exception as e:
            messagebox.showerror("エラー", f"マスク作成用画像の読み込みに失敗しました:\n{e}")
            return


        win = Toplevel(self)
        win.title(window_title)
        win.geometry("1000x700")
        win.grab_set()

        orig_h, orig_w = frame.shape[:2]
        disp_w, disp_h = 960, 540
        scale = min(disp_w / orig_w, disp_h / orig_h)
        disp_w, disp_h = int(orig_w * scale), int(orig_h * scale)
        
        frame_disp = cv2.resize(frame, (disp_w, disp_h))
        tk_image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB)))

        canvas = Canvas(win, width=disp_w, height=disp_h, cursor="circle")
        canvas.pack(pady=5)
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
        canvas.image = tk_image

        mask_data_disp = Image.new("L", (disp_w, disp_h), 0)
        draw = ImageDraw.Draw(mask_data_disp)
        
        brush_radius = tk.IntVar(value=30)
        def paint(event):
            r = brush_radius.get()
            canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r, fill='white', outline='white', tags="paint")
            draw.ellipse((event.x - r, event.y - r, event.x + r, event.y + r), fill=255)
        canvas.bind("<B1-Motion>", paint)
        canvas.bind("<Button-1>", paint)

        controls = ttk.Frame(win); controls.pack(fill=tk.X, padx=10)
        ttk.Label(controls, text="ブラシサイズ:").pack(side=tk.LEFT)
        ttk.Scale(controls, from_=5, to=100, orient=tk.HORIZONTAL, variable=brush_radius).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(controls, text="クリア", command=lambda: [canvas.delete("paint"), draw.rectangle([0,0,disp_w,disp_h], fill=0)]).pack(side=tk.LEFT)

        def on_ok():
            mask_np_disp = np.array(mask_data_disp)
            final_mask = cv2.bitwise_not(cv2.resize(mask_np_disp, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)) if mask_np_disp.max() > 0 else np.full((orig_h, orig_w), 255, dtype=np.uint8)
            
            if is_plate_solve_mask:
                self.plate_solve_mask_image = final_mask
                self.preview_mask(self.plate_solve_mask_image, self.ps_mask_preview_label, "PSマスク")
            else:
                self.mask_image = final_mask
                self.mask_path_var.set("作成済み (描画)")
                self.apply_mask_var.set(True)
                self.preview_mask(self.mask_image, self.mask_preview_label, "検出マスク")
            win.destroy()
        btn_frame = ttk.Frame(win); btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="キャンセル", command=win.destroy).pack(side=tk.LEFT, padx=5)

    def create_rtsp_mask(self):
        """RTSPストリームからマスクを作成する"""
        # 選択されているRTSP URLを取得、選択がなければ最初のURLを使用
        if self.rtsp_selected_indices:
            selected_index = min(self.rtsp_selected_indices)
            rtsp_url = self.rtsp_urls[selected_index]
        elif self.rtsp_urls:
            rtsp_url = self.rtsp_urls[0]
        else:
            messagebox.showwarning("警告", "RTSPストリームを追加してください。")
            return

        is_plate_solve_mask = self._select_rtsp_mask_type()
        if is_plate_solve_mask is None:
            return
        
        progress_win = Toplevel(self)
        progress_win.title("接続中")
        progress_win.geometry("300x100")
        progress_win.transient(self)
        progress_win.grab_set()
        progress_win.resizable(False, False)
        
        ttk.Label(progress_win, text="RTSPストリームに接続中...\nしばらくお待ちください。").pack(pady=15)
        cancel_flag = threading.Event()
        
        def on_cancel():
            cancel_flag.set()
            progress_win.destroy()
        
        cancel_btn = ttk.Button(progress_win, text="キャンセル", command=on_cancel)
        cancel_btn.pack(pady=5)
        
        progress_win.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - progress_win.winfo_width()) // 2
        y = self.winfo_y() + (self.winfo_height() - progress_win.winfo_height()) // 2
        progress_win.geometry(f"+{x}+{y}")
        
        result_holder = {'frame': None, 'error': None}
        
        def fetch_frame():
            try:
                cap = utils.create_rtsp_capture(rtsp_url)
                if not cap.isOpened():
                    result_holder['error'] = "RTSPストリームを開けませんでした。"
                    return
                ret, frame = cap.read()
                cap.release()
                if cancel_flag.is_set():
                    return
                if not ret or frame is None:
                    result_holder['error'] = "RTSPストリームからフレームを読み込めませんでした。"
                else:
                    result_holder['frame'] = frame
            except Exception as e:
                result_holder['error'] = str(e)
        
        fetch_thread = threading.Thread(target=fetch_frame, daemon=True)
        fetch_thread.start()
        
        def check_thread():
            if cancel_flag.is_set():
                return
            if fetch_thread.is_alive():
                self.after(100, check_thread)
            else:
                try:
                    progress_win.destroy()
                except tk.TclError:
                    pass
                if result_holder['error']:
                    messagebox.showerror("エラー", f"RTSPからのフレーム取得に失敗しました:\n{result_holder['error']}")
                elif result_holder['frame'] is not None:
                    self._open_rtsp_mask_window(result_holder['frame'], is_plate_solve_mask)
        
        self.after(100, check_thread)

    def _select_rtsp_mask_type(self) -> Optional[bool]:
        """RTSPマスク作成時にマスクの用途を選択する"""
        selection = {'is_plate_solve_mask': None}
        style = ttk.Style(self)
        bg_color = self.cget("background") or style.lookup("TFrame", "background") or "#2E3F5B"
        sub_fg_color = "#AFC0DA"

        dialog = Toplevel(self)
        dialog.title("マスク種別の選択")
        dialog.geometry("440x190")
        dialog.transient(self)
        dialog.grab_set()
        dialog.resizable(False, False)
        dialog.configure(background=bg_color)
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)

        container = tk.Frame(dialog, bg=bg_color, padx=18, pady=14)
        container.pack(fill=tk.BOTH, expand=True)

        ttk.Label(container, text="作成するマスクを選択してください。").pack(pady=(4, 8))
        tk.Label(
            container,
            text="RTSPから取得したフレームでマスクを作成します。",
            bg=bg_color,
            fg=sub_fg_color,
            font=("Segoe UI", 10)
        ).pack(pady=(0, 14))

        btn_frame = ttk.Frame(container)
        btn_frame.pack(pady=4)

        def choose_detection_mask():
            selection['is_plate_solve_mask'] = False
            dialog.destroy()

        def choose_plate_solve_mask():
            selection['is_plate_solve_mask'] = True
            dialog.destroy()

        ttk.Button(btn_frame, text="検出用マスク", command=choose_detection_mask).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="プレートソルブ用マスク", command=choose_plate_solve_mask).pack(side=tk.LEFT, padx=6)
        ttk.Button(container, text="キャンセル", command=dialog.destroy).pack(pady=(12, 0))

        dialog.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - dialog.winfo_width()) // 2
        y = self.winfo_y() + (self.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")

        self.wait_window(dialog)
        return selection['is_plate_solve_mask']
    
    def _open_rtsp_mask_window(self, frame, is_plate_solve_mask: bool):
        """RTSPから取得したフレームでマスク作成ウィンドウを開く"""
        
        # マスク作成ウィンドウを開く
        win = Toplevel(self)
        mask_label = "プレートソルブ用マスク" if is_plate_solve_mask else "検出用マスク"
        win.title(f"RTSPから{mask_label}作成")
        win.geometry("1000x700")
        win.grab_set()

        orig_h, orig_w = frame.shape[:2]
        disp_w, disp_h = 960, 540
        scale = min(disp_w / orig_w, disp_h / orig_h)
        disp_w, disp_h = int(orig_w * scale), int(orig_h * scale)
        
        frame_disp = cv2.resize(frame, (disp_w, disp_h))
        tk_image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB)))

        canvas = Canvas(win, width=disp_w, height=disp_h, cursor="circle")
        canvas.pack(pady=5)
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
        canvas.image = tk_image

        # マスクデータをselfに一時保存して確実に参照可能にする
        self._rtsp_mask_data = Image.new("L", (disp_w, disp_h), 0)
        self._rtsp_mask_draw = ImageDraw.Draw(self._rtsp_mask_data)
        self._rtsp_mask_orig_size = (orig_w, orig_h)
        
        brush_radius = tk.IntVar(value=30)
        
        def paint(event):
            r = brush_radius.get()
            canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r, fill='white', outline='white', tags="paint")
            self._rtsp_mask_draw.ellipse((event.x - r, event.y - r, event.x + r, event.y + r), fill=255)
        
        canvas.bind("<B1-Motion>", paint)
        canvas.bind("<Button-1>", paint)

        def clear_mask():
            canvas.delete("paint")
            self._rtsp_mask_draw.rectangle([0, 0, disp_w, disp_h], fill=0)

        controls = ttk.Frame(win)
        controls.pack(fill=tk.X, padx=10)
        ttk.Label(controls, text="ブラシサイズ:").pack(side=tk.LEFT)
        ttk.Scale(controls, from_=5, to=100, orient=tk.HORIZONTAL, variable=brush_radius).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(controls, text="クリア", command=clear_mask).pack(side=tk.LEFT)

        def on_ok():
            try:
                # selfからマスクデータを取得
                mask_np_disp = np.array(self._rtsp_mask_data)
                orig_w, orig_h = self._rtsp_mask_orig_size
                
                print(f"マスクデータ確認: max={mask_np_disp.max()}, min={mask_np_disp.min()}, shape={mask_np_disp.shape}")
                
                # 描画がある場合は反転したマスクを作成、ない場合は全て255（マスクなし）
                if mask_np_disp.max() > 0:
                    # ディスプレイサイズから元のサイズにリサイズ
                    mask_resized = cv2.resize(mask_np_disp, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                    # 白黒反転（描画部分=255をマスク=0に）
                    final_mask = cv2.bitwise_not(mask_resized)
                else:
                    # 描画がなければマスクなし（全て255）
                    final_mask = np.full((orig_h, orig_w), 255, dtype=np.uint8)
                
                # メインアプリのマスクを更新
                if is_plate_solve_mask:
                    self.plate_solve_mask_image = final_mask
                    self.preview_mask(self.plate_solve_mask_image, self.ps_mask_preview_label, "PSマスク")
                else:
                    self.mask_image = final_mask
                    self.mask_path_var.set("作成済み (RTSP)")
                    self.apply_mask_var.set(True)
                    self.preview_mask(self.mask_image, self.mask_preview_label, "検出マスク")
                
                print(f"RTSP{mask_label}作成完了: shape={final_mask.shape}, max={final_mask.max()}, min={final_mask.min()}")
                
                # 一時データを削除
                del self._rtsp_mask_data
                del self._rtsp_mask_draw
                del self._rtsp_mask_orig_size
                
                # ウィンドウを閉じる
                win.destroy()
                
            except Exception as e:
                messagebox.showerror("エラー", f"マスク作成中にエラーが発生しました:\n{e}")
                import traceback
                traceback.print_exc()

        btn_frame = ttk.Frame(win)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="キャンセル", command=win.destroy).pack(side=tk.LEFT, padx=5)

    def preview_mask(self, mask_data, target_label, label_text):
        if mask_data is None:
            target_label.config(image='', text=f"{label_text}なし")
            target_label.image = None
            return
        try:
            preview_data = cv2.bitwise_not(mask_data)
            mask_pil = Image.fromarray(preview_data)
            mask_pil.thumbnail((80, 80))
            mask_photo = ImageTk.PhotoImage(mask_pil)
            target_label.config(image=mask_photo, text=f"{label_text}あり", compound=tk.TOP)
            target_label.image = mask_photo
        except Exception as e:
            print(f"マスクプレビューエラー: {e}")
            target_label.config(image='', text=f"{label_text} (エラー)")

    def download_mask(self):
        """検出マスクをPNG形式（1920x1080）でダウンロードする（マスク部分は透明）"""
        if self.mask_image is None:
            messagebox.showwarning("警告", "検出マスクが作成されていません。\n先にマスクを作成してください。")
            return
        
        # ファイル保存ダイアログを表示
        save_path = filedialog.asksaveasfilename(
            title="マスクを保存",
            defaultextension=".png",
            filetypes=[("PNG画像", "*.png")],
            initialfile="detection_mask.png"
        )
        
        if not save_path:
            return  # キャンセルされた場合
        
        try:
            # マスクを1920x1080にリサイズ（横長）
            target_size = (1920, 1080)  # (width, height)
            resized_mask = cv2.resize(self.mask_image, target_size, interpolation=cv2.INTER_NEAREST)
            
            # RGBAに変換
            # mask_imageでは 255=検出可能領域（非マスク）、0=マスク領域（塗った部分）
            # 出力では: マスク領域（塗った部分）=黒で不透明、非マスク領域=透明
            height, width = resized_mask.shape
            rgba = np.zeros((height, width, 4), dtype=np.uint8)
            
            # アルファチャンネル: マスク領域(0)=不透明(255)、非マスク領域(255)=透明(0)
            rgba[:, :, 3] = 255 - resized_mask  # 反転してマスク部分を不透明に
            # RGB channels stay 0 (black) for mask areas
            
            # PILを使用してPNGとして保存（アルファチャンネル対応）
            pil_image = Image.fromarray(rgba, mode='RGBA')
            pil_image.save(save_path, 'PNG')
            
            messagebox.showinfo("保存完了", f"マスクを保存しました:\n{save_path}")
            self.append_log(f"検出マスクを保存しました: {save_path}")
        except Exception as e:
            messagebox.showerror("エラー", f"マスクの保存に失敗しました:\n{e}")

    def save_settings(self):
        settings = {
            'periodic_scan_enabled': self.periodic_scan_var.get(), 'periodic_scan_directory': self.periodic_dir_var.get(),
            'periodic_scan_interval': self.periodic_interval_var.get(), 'periodic_time_limit_enabled': self.periodic_time_limit_var.get(),
            'periodic_start_hour': self.start_hour_var.get(), 'periodic_start_minute': self.start_min_var.get(),
            'periodic_end_hour': self.end_hour_var.get(), 'periodic_end_minute': self.end_min_var.get(),
            'folder_paths': self.folder_paths, 'rtsp_urls': self.rtsp_urls,
            'save_options': {k: v.get() for k, v in self.save_options_vars.items()},
            'plate_solve_wcs_path': self.plate_solve_wcs_path_var.get(), 'plate_solve_video_path': self.plate_solve_video_path_var.get(),
            'use_plate_solve': self.use_plate_solve_var.get(), 'apply_mask': self.apply_mask_var.get(),
            'mask_path_or_status': self.mask_path_var.get(), 'concurrency': self.concurrency_var.get(),
            'interval': self.interval_var.get(), 'duration': self.duration_var.get(),
            'meteor_save_path': self.meteor_save_path_var.get(), 'not_meteor_save_path': self.not_meteor_save_path_var.get(),
            'selected_model_path': self.selected_model_path_var.get(),
            'custom_model_paths': list(self.custom_model_paths),
            'has_mask_image': self.mask_image is not None, 'has_plate_solve_mask_image': self.plate_solve_mask_image is not None,
            'summary_video_config': self.summary_video_config,
            'auto_time_updater_enabled': self.auto_time_updater_enabled_var.get(),
            'rtsp_preset': self.rtsp_preset_var.get(),
            'rtsp_fps': self.rtsp_fps_var.get(),
            # RTSP time limit settings
            'rtsp_time_limit_enabled': self.rtsp_time_limit_var.get(),
            'rtsp_start_hour': self.rtsp_start_hour_var.get(), 'rtsp_start_minute': self.rtsp_start_min_var.get(),
            # Plate solve mode
            'plate_solve_mode': self.plate_solve_mode_var.get(),
            'rtsp_end_hour': self.rtsp_end_hour_var.get(), 'rtsp_end_minute': self.rtsp_end_min_var.get(),
            # Astrometry.net API key
            'astrometry_api_key': self.astrometry_api_key_var.get(),
            # Advanced settings
            'advanced_settings': {
                'min_line_length': self.cfg_min_line_length_var.get(),
                'border_size': self.cfg_border_size_var.get(),
                'duplicate_thresh': self.cfg_duplicate_thresh_var.get(),
                'meteor_prob': self.cfg_meteor_prob_var.get(),
                'finer_window_sec': self.cfg_finer_window_sec_var.get(),
                'finer_comp_step': self.cfg_finer_comp_step_var.get(),
                'finer_min_length': self.cfg_finer_min_length_var.get(),
                'finer_padding_sec': self.cfg_finer_padding_sec_var.get(),
                'finer_cutout_size': self.cfg_finer_cutout_size_var.get(),
                'airplane_dur_thresh': self.cfg_airplane_dur_thresh_var.get(),
                'airplane_frame_thresh': self.cfg_airplane_frame_thresh_var.get(),
                'tracking_dist_thresh': self.cfg_tracking_dist_thresh_var.get(),
                'max_clip_dur': self.cfg_max_clip_dur_var.get(),
                'clip_dur_sec': self.cfg_clip_dur_sec_var.get(),
                'cutout_size': self.cfg_cutout_size_var.get(),
                'rtsp_scale_lower': self.cfg_rtsp_scale_lower_var.get(),
                'rtsp_scale_upper': self.cfg_rtsp_scale_upper_var.get(),
            }
        }
        if self.global_wcs_info:
            serializable_wcs = self.global_wcs_info.copy()
            if isinstance(serializable_wcs.get('plate_solve_datetime'), datetime):
                serializable_wcs['plate_solve_datetime'] = serializable_wcs['plate_solve_datetime'].isoformat()
            settings['global_wcs_info'] = serializable_wcs

        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f: json.dump(settings, f, indent=4)
            masks_to_save = {}
            if self.mask_image is not None: masks_to_save['mask_image'] = self.mask_image
            if self.plate_solve_mask_image is not None: masks_to_save['plate_solve_mask_image'] = self.plate_solve_mask_image
            if masks_to_save: np.savez(self.masks_file, **masks_to_save)
            elif os.path.exists(self.masks_file): os.remove(self.masks_file)
            print("設定を保存しました。")
        except Exception as e:
            print(f"設定の保存に失敗しました: {e}")

    def load_settings(self):
        if not os.path.exists(self.settings_file): return
        self.append_log(f"設定ファイル読み込み元: {self.settings_file}")
        self.append_log(f"マスクファイル読み込み元: {self.masks_file}")
        if not messagebox.askyesno("設定の復元", "前回の設定を復元しますか？"): return
        
        try:
            with open(self.settings_file, 'r', encoding='utf-8') as f: settings = json.load(f)

            self.periodic_scan_var.set(settings.get('periodic_scan_enabled', False))
            self.periodic_dir_var.set(settings.get('periodic_scan_directory', ''))
            self.periodic_interval_var.set(settings.get('periodic_scan_interval', str(config.DEFAULT_SCAN_INTERVAL)))
            self.periodic_time_limit_var.set(settings.get('periodic_time_limit_enabled', False))
            self.start_hour_var.set(settings.get('periodic_start_hour', '17')); self.start_min_var.set(settings.get('periodic_start_minute', '00'))
            self.end_hour_var.set(settings.get('periodic_end_hour', '07')); self.end_min_var.set(settings.get('periodic_end_minute', '00'))
            self.toggle_time_limit_frame()
            
            # RTSP time limit settings
            self.rtsp_time_limit_var.set(settings.get('rtsp_time_limit_enabled', False))
            self.rtsp_start_hour_var.set(settings.get('rtsp_start_hour', '17')); self.rtsp_start_min_var.set(settings.get('rtsp_start_minute', '00'))
            self.rtsp_end_hour_var.set(settings.get('rtsp_end_hour', '07')); self.rtsp_end_min_var.set(settings.get('rtsp_end_minute', '00'))
            self.toggle_rtsp_time_limit_frame()

            self.folder_paths = settings.get('folder_paths', [])
            # Clear existing items and add restored paths
            for item in self.folder_item_frames:
                item['frame'].destroy()
            self.folder_item_frames.clear()
            self.folder_selected_indices.clear()
            for p in self.folder_paths:
                # For restored paths, show path only (no FPS calculation to avoid delay)
                self._add_folder_item("--", p)
            self.rtsp_urls = settings.get('rtsp_urls', [])
            # Clear and restore RTSP items
            for item in self.rtsp_item_frames:
                item['frame'].destroy()
            self.rtsp_item_frames.clear()
            self.rtsp_selected_indices.clear()
            for url in self.rtsp_urls:
                self._add_rtsp_item(url)

            saved_opts = settings.get('save_options', {})
            for key, var in self.save_options_vars.items(): var.set(saved_opts.get(key, True))

            self.plate_solve_wcs_path_var.set(settings.get('plate_solve_wcs_path', ''))
            self.plate_solve_video_path_var.set(settings.get('plate_solve_video_path', ''))
            self.use_plate_solve_var.set(settings.get('use_plate_solve', True))
            self.plate_solve_mode_var.set(settings.get('plate_solve_mode', 'local'))
            
            if settings.get('global_wcs_info'):
                self.global_wcs_info = settings['global_wcs_info']
                if isinstance(self.global_wcs_info.get('plate_solve_datetime'), str):
                    self.global_wcs_info['plate_solve_datetime'] = datetime.fromisoformat(self.global_wcs_info['plate_solve_datetime'])
                if self.global_wcs_info.get('wcs_file'): self.plate_solve_status_var.set("プレートソルブ: 成功 (復元)")

            self.apply_mask_var.set(settings.get('apply_mask', False))
            self.mask_path_var.set(settings.get('mask_path_or_status', ''))
            self.concurrency_var.set(settings.get('concurrency', str(config.DEFAULT_CONCURRENCY)))
            self.interval_var.set(settings.get('interval', str(config.DEFAULT_INTERVAL)))
            self.duration_var.set(settings.get('duration', str(config.DEFAULT_DURATION)))
            self.meteor_save_path_var.set(settings.get('meteor_save_path', config.DEFAULT_METEOR_SAVE_PATH))
            self.not_meteor_save_path_var.set(settings.get('not_meteor_save_path', config.DEFAULT_NOT_METEOR_SAVE_PATH))
            self.custom_model_paths = [str(p) for p in settings.get('custom_model_paths', []) if isinstance(p, str)]
            self.selected_model_path_var.set(settings.get('selected_model_path', config.MODEL_PATH))
            self.refresh_model_candidates()
            # summary_video_config: 保存された順番を復元し、新項目を末尾に追加
            saved_summary_config = settings.get('summary_video_config', [])
            if saved_summary_config:
                # 保存された設定の名前リスト
                saved_names = [item['name'] for item in saved_summary_config]
                # デフォルト設定の名前リスト
                default_names = [item['name'] for item in self.summary_video_config]
                
                # 保存された順番で新しいリストを構築
                new_config = []
                for saved_item in saved_summary_config:
                    # 保存されたアイテムをそのまま追加（順番を維持）
                    new_config.append(saved_item.copy())
                
                # デフォルトにあって保存設定にない新項目を末尾に追加
                for default_item in self.summary_video_config:
                    if default_item['name'] not in saved_names:
                        new_config.append(default_item.copy())
                
                self.summary_video_config = new_config

            # 自動更新設定を復元
            auto_update_enabled = settings.get('auto_time_updater_enabled', False)
            self.auto_time_updater_enabled_var.set(auto_update_enabled)
            if auto_update_enabled:
                self.auto_updater.start()
            
            # RTSPプリセット設定を復元
            self.rtsp_preset_var.set(settings.get('rtsp_preset', 'cloudy'))
            self.rtsp_fps_var.set(settings.get('rtsp_fps', str(config.RTSP_FPS)))
            config.RTSP_FPS = int(float(self.rtsp_fps_var.get())) # Apply immediately update config

            # Astrometry.net APIキーを復元
            api_key = settings.get('astrometry_api_key', '')
            self.astrometry_api_key_var.set(api_key)
            if api_key:
                config.ASTROMETRY_API_KEY = api_key

            if os.path.exists(self.masks_file):
                loaded_masks = np.load(self.masks_file)
                if settings.get('has_mask_image') and 'mask_image' in loaded_masks: self.mask_image = loaded_masks['mask_image']
                if settings.get('has_plate_solve_mask_image') and 'plate_solve_mask_image' in loaded_masks: self.plate_solve_mask_image = loaded_masks['plate_solve_mask_image']
            
            self.preview_mask(self.mask_image, self.mask_preview_label, "検出マスク")
            self.preview_mask(self.plate_solve_mask_image, self.ps_mask_preview_label, "PSマスク")

            # Advanced settings restore
            adv = settings.get('advanced_settings', {})
            if adv:
                self.cfg_min_line_length_var.set(adv.get('min_line_length', str(config.MIN_LINE_LENGTH)))
                self.cfg_border_size_var.set(adv.get('border_size', str(config.BORDER_SIZE)))
                self.cfg_duplicate_thresh_var.set(adv.get('duplicate_thresh', str(config.DUPLICATE_DETECTION_THRESHOLD)))
                self.cfg_meteor_prob_var.set(adv.get('meteor_prob', str(config.METEOR_PROBABILITY_THRESHOLD)))
                self.cfg_finer_window_sec_var.set(adv.get('finer_window_sec', str(config.FINER_DETECT_WINDOW_SECONDS)))
                self.cfg_finer_comp_step_var.set(adv.get('finer_comp_step', str(config.FINER_COMPOSITE_STEP)))
                self.cfg_finer_min_length_var.set(adv.get('finer_min_length', str(config.FINER_DETECT_MIN_LENGTH)))
                self.cfg_finer_padding_sec_var.set(adv.get('finer_padding_sec', str(config.FINER_DETECT_PADDING_SECONDS)))
                self.cfg_finer_cutout_size_var.set(adv.get('finer_cutout_size', str(config.FINER_CUTOUT_SIZE)))
                self.cfg_airplane_dur_thresh_var.set(adv.get('airplane_dur_thresh', str(config.AIRPLANE_DURATION_THRESHOLD)))
                self.cfg_airplane_frame_thresh_var.set(adv.get('airplane_frame_thresh', str(config.AIRPLANE_FRAME_THRESHOLD)))
                self.cfg_tracking_dist_thresh_var.set(adv.get('tracking_dist_thresh', str(config.TRACKING_DISTANCE_THRESHOLD)))
                self.cfg_max_clip_dur_var.set(adv.get('max_clip_dur', str(config.MAX_CLIP_DURATION)))
                self.cfg_clip_dur_sec_var.set(adv.get('clip_dur_sec', str(config.CLIP_DURATION_SECONDS)))
                self.cfg_cutout_size_var.set(adv.get('cutout_size', str(config.CUTOUT_SIZE)))
                self.cfg_rtsp_scale_lower_var.set(adv.get('rtsp_scale_lower', str(config.RTSP_SCALE_LOWER)))
                self.cfg_rtsp_scale_upper_var.set(adv.get('rtsp_scale_upper', str(config.RTSP_SCALE_UPPER)))
                
                # Apply to config module
                self.apply_advanced_settings_to_config()

            self.apply_selected_model(silent=True)

            self.append_log("前回の設定を復元しました。")
            self.update_start_button_state()
        except Exception as e:
            messagebox.showerror("エラー", f"設定の読み込み中にエラーが発生しました: {e}")

    def apply_advanced_settings_to_config(self):
        """GUIの変数をconfigモジュールに適用する"""
        try:
            config.MIN_LINE_LENGTH = int(float(self.cfg_min_line_length_var.get()))
            config.BORDER_SIZE = int(float(self.cfg_border_size_var.get()))
            config.DUPLICATE_DETECTION_THRESHOLD = int(float(self.cfg_duplicate_thresh_var.get()))
            config.METEOR_PROBABILITY_THRESHOLD = float(self.cfg_meteor_prob_var.get())
            config.FINER_DETECT_WINDOW_SECONDS = float(self.cfg_finer_window_sec_var.get())
            config.FINER_COMPOSITE_STEP = int(float(self.cfg_finer_comp_step_var.get()))
            config.FINER_DETECT_MIN_LENGTH = int(float(self.cfg_finer_min_length_var.get()))
            config.FINER_DETECT_PADDING_SECONDS = float(self.cfg_finer_padding_sec_var.get())
            config.FINER_CUTOUT_SIZE = int(float(self.cfg_finer_cutout_size_var.get()))
            config.AIRPLANE_DURATION_THRESHOLD = int(float(self.cfg_airplane_dur_thresh_var.get()))
            config.AIRPLANE_FRAME_THRESHOLD = int(float(self.cfg_airplane_frame_thresh_var.get()))
            config.TRACKING_DISTANCE_THRESHOLD = int(float(self.cfg_tracking_dist_thresh_var.get()))
            config.MAX_CLIP_DURATION = float(self.cfg_max_clip_dur_var.get())
            config.CLIP_DURATION_SECONDS = float(self.cfg_clip_dur_sec_var.get())
            config.CUTOUT_SIZE = int(float(self.cfg_cutout_size_var.get()))
            config.RTSP_SCALE_LOWER = float(self.cfg_rtsp_scale_lower_var.get())
            config.RTSP_SCALE_UPPER = float(self.cfg_rtsp_scale_upper_var.get())
        except Exception as e:
            self.append_log(f"詳細設定の適用中にエラーが発生しました: {e}")

    def create_long_exposure_map_callback(self):
        """Callback for the 'Create Long Exposure Map' button."""
        if not self.check_admin_password():
            return

        if not self.folder_paths:
            messagebox.showwarning("情報", "ソース選択タブでフォルダまたは動画ファイルを追加してください。")
            return

        output_path = filedialog.asksaveasfilename(
            title="長時間輝線マップの保存先",
            defaultextension=".jpg",
            filetypes=[("JPEG Image", "*.jpg"), ("PNG Image", "*.png"), ("All Files", "*")]
        )
        
        if not output_path:
            return

        def run_task():
            self.append_log("長時間輝線マップの作成を開始します...")
            success = long_exposure_map.create_long_exposure_map(
                self.folder_paths, 
                output_path, 
                progress_callback=self.append_log
            )
            if success:
                messagebox.showinfo("完了", "長時間輝線マップの作成が完了しました。")
                self.append_log("長時間輝線マップの作成が完了しました。")
            else:
                messagebox.showerror("エラー", "長時間輝線マップの作成に失敗しました。ログを確認してください。")
                self.append_log("長時間輝線マップの作成に失敗しました。")

        threading.Thread(target=run_task, daemon=True).start()

    def apply_distortion_correction_callback(self):
        """Callback for the 'Distortion Correction' button."""
        if not self.check_admin_password():
            return

        if not self.folder_paths:
            messagebox.showwarning("情報", "ソース選択タブでフォルダまたは動画ファイルを追加してください。")
            return

        output_path = filedialog.asksaveasfilename(
            title="ゆがみ補正画像の保存先",
            defaultextension=".jpg",
            filetypes=[("JPEG Image", "*.jpg"), ("PNG Image", "*.png"), ("All Files", "*")]
        )
        
        if not output_path:
            return

        # Distortion maps are stored next to this module.
        module_dir = os.path.dirname(os.path.abspath(__file__))
        map_x_path = os.path.join(module_dir, "distortion_map_x.npy")
        map_y_path = os.path.join(module_dir, "distortion_map_y.npy")

        def run_task():
            self.append_log("ゆがみ補正処理を開始します...")
            success = distortion_correction.apply_distortion_correction(
                self.folder_paths, 
                output_path, 
                map_x_path,
                map_y_path,
                progress_callback=self.append_log
            )
            if success:
                messagebox.showinfo("完了", "ゆがみ補正画像の作成が完了しました。")
                self.append_log("ゆがみ補正画像の作成が完了しました。")
            else:
                messagebox.showerror("エラー", "ゆがみ補正画像の作成に失敗しました。ログを確認してください。")
                self.append_log("ゆがみ補正画像の作成に失敗しました。")

        threading.Thread(target=run_task, daemon=True).start()

    def _select_selfcal_mask_mode(self) -> Optional[str]:
        """Select mask mode for night self-calibration."""
        selection = {"mode": None}
        style = ttk.Style(self)
        bg_color = self.cget("background") or style.lookup("TFrame", "background") or "#2E3F5B"

        dialog = Toplevel(self)
        dialog.title("自己校正マスク設定")
        dialog.geometry("520x220")
        dialog.transient(self)
        dialog.grab_set()
        dialog.resizable(False, False)
        dialog.configure(background=bg_color)

        container = tk.Frame(dialog, bg=bg_color, padx=16, pady=14)
        container.pack(fill=tk.BOTH, expand=True)

        ttk.Label(container, text="夜間自己校正で使うマスク方式を選択してください。").pack(anchor=tk.W, pady=(0, 8))
        tk.Label(
            container,
            text="手動マスクは『除外したい領域を塗る』方式です（時刻表示・地上・強いかぶり等）。",
            bg=bg_color,
            fg="#AFC0DA",
            justify=tk.LEFT,
            wraplength=480
        ).pack(anchor=tk.W, pady=(0, 10))

        btns = ttk.Frame(container)
        btns.pack(fill=tk.X, pady=4)

        def choose(mode: str):
            selection["mode"] = mode
            dialog.destroy()

        ttk.Button(btns, text="自動+手動 (推奨)", command=lambda: choose("auto_plus_manual")).pack(fill=tk.X, pady=3)
        ttk.Button(btns, text="自動のみ", command=lambda: choose("auto_only")).pack(fill=tk.X, pady=3)
        ttk.Button(btns, text="手動のみ", command=lambda: choose("manual_only")).pack(fill=tk.X, pady=3)
        ttk.Button(container, text="キャンセル", command=dialog.destroy).pack(pady=(10, 0))

        dialog.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - dialog.winfo_width()) // 2
        y = self.winfo_y() + (self.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")

        self.wait_window(dialog)
        return selection["mode"]

    def _read_frame_for_selfcal_mask(self, video_path: str) -> Optional[np.ndarray]:
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                return None
            return frame
        except Exception:
            return None

    def _draw_manual_exclusion_mask_on_frame(
        self,
        frame: np.ndarray,
        title: str = "自己校正用手動マスク作成",
        existing_mask: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """Open a drawing dialog and return final mask (255=use, 0=exclude)."""
        if frame is None:
            return None

        result = {"mask": None}
        win = Toplevel(self)
        win.title(title)
        win.geometry("1060x760")
        win.transient(self)
        win.grab_set()

        orig_h, orig_w = frame.shape[:2]
        disp_w, disp_h = 980, 600
        scale = min(disp_w / orig_w, disp_h / orig_h)
        disp_w, disp_h = int(orig_w * scale), int(orig_h * scale)

        frame_disp = cv2.resize(frame, (disp_w, disp_h))
        frame_rgb = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB)
        bg_photo = ImageTk.PhotoImage(Image.fromarray(frame_rgb))

        canvas = Canvas(win, width=disp_w, height=disp_h, cursor="circle", bg="black")
        canvas.pack(pady=6)
        canvas.create_image(0, 0, anchor=tk.NW, image=bg_photo)
        canvas.image = bg_photo

        mask_data_disp = Image.new("L", (disp_w, disp_h), 0)
        draw = ImageDraw.Draw(mask_data_disp)

        # If an existing mask is provided, preload it (excluded area -> white paint).
        if existing_mask is not None:
            try:
                m = existing_mask
                if m.ndim == 3:
                    m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
                if m.shape[:2] != (orig_h, orig_w):
                    m = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                m_disp = cv2.resize(m, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
                painted_disp = cv2.bitwise_not(m_disp)
                mask_data_disp = Image.fromarray(painted_disp.astype(np.uint8), mode="L")
                draw = ImageDraw.Draw(mask_data_disp)
            except Exception:
                pass

        overlay_item = None
        overlay_photo_holder = {"photo": None}

        def refresh_overlay():
            nonlocal overlay_item
            mask_np = np.array(mask_data_disp, dtype=np.uint8)
            if mask_np.max() == 0:
                if overlay_item is not None:
                    canvas.delete(overlay_item)
                    overlay_item = None
                overlay_photo_holder["photo"] = None
                return

            rgba = np.zeros((disp_h, disp_w, 4), dtype=np.uint8)
            rgba[..., 0] = 255  # red
            rgba[..., 3] = (mask_np > 0).astype(np.uint8) * 100
            overlay_photo = ImageTk.PhotoImage(Image.fromarray(rgba, mode="RGBA"))
            overlay_photo_holder["photo"] = overlay_photo
            if overlay_item is None:
                overlay_item = canvas.create_image(0, 0, anchor=tk.NW, image=overlay_photo)
            else:
                canvas.itemconfig(overlay_item, image=overlay_photo)

        refresh_overlay()

        brush_radius = tk.IntVar(value=35)
        draw_mode = tk.StringVar(value="paint")  # paint=exclude, erase=restore

        def apply_brush(event):
            r = int(brush_radius.get())
            x0, y0, x1, y1 = event.x - r, event.y - r, event.x + r, event.y + r
            if draw_mode.get() == "paint":
                draw.ellipse((x0, y0, x1, y1), fill=255)
            else:
                draw.ellipse((x0, y0, x1, y1), fill=0)
            refresh_overlay()

        def paint(event):
            draw_mode.set("paint")
            apply_brush(event)

        def erase(event):
            draw_mode.set("erase")
            apply_brush(event)

        canvas.bind("<B1-Motion>", paint)
        canvas.bind("<Button-1>", paint)
        canvas.bind("<B3-Motion>", erase)
        canvas.bind("<Button-3>", erase)

        controls = ttk.Frame(win)
        controls.pack(fill=tk.X, padx=10)
        ttk.Label(controls, text="ブラシサイズ:").pack(side=tk.LEFT)
        ttk.Scale(controls, from_=5, to=150, orient=tk.HORIZONTAL, variable=brush_radius).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        ttk.Label(controls, text="左クリック: 除外 / 右クリック: 復元").pack(side=tk.LEFT, padx=(6, 0))

        sub_controls = ttk.Frame(win)
        sub_controls.pack(fill=tk.X, padx=10, pady=(4, 0))

        def clear_mask():
            draw.rectangle([0, 0, disp_w, disp_h], fill=0)
            refresh_overlay()

        ttk.Button(sub_controls, text="クリア", command=clear_mask).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Label(
            win,
            text="塗った領域は自己校正で除外されます。空以外の地上、時刻表示、強いかぶり、雲の出やすい部分を除外してください。",
            foreground="#87CEEB"
        ).pack(anchor=tk.W, padx=10, pady=(6, 0))

        def on_ok():
            mask_np_disp = np.array(mask_data_disp, dtype=np.uint8)
            if mask_np_disp.max() > 0:
                excluded_resized = cv2.resize(mask_np_disp, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                final_mask = cv2.bitwise_not(excluded_resized)
            else:
                final_mask = np.full((orig_h, orig_w), 255, dtype=np.uint8)
            result["mask"] = final_mask
            win.destroy()

        btn_frame = ttk.Frame(win)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="キャンセル", command=win.destroy).pack(side=tk.LEFT, padx=5)

        self.wait_window(win)
        return result["mask"]

    def estimate_distortion_map_night_callback(self):
        """Generate distortion maps from ~20 minutes of a night-sky video inside the app."""
        if not self.check_admin_password():
            return

        if not self.folder_paths:
            messagebox.showwarning("情報", "ソース選択タブで夜空動画のフォルダまたは動画ファイルを追加してください。")
            return

        initial_dir = None
        try:
            first_source = self.folder_paths[0]
            if os.path.isdir(first_source):
                initial_dir = first_source
            elif os.path.isfile(first_source):
                initial_dir = os.path.dirname(first_source)
        except Exception:
            initial_dir = None
        if not initial_dir:
            initial_dir = os.path.expanduser("~")

        selected_start_video = filedialog.askopenfilename(
            title="夜間自己校正の開始動画を選択 (この動画から後続分割動画を連続利用)",
            initialdir=initial_dir,
            filetypes=[
                ("動画ファイル", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("すべてのファイル", "*.*"),
            ]
        )
        if not selected_start_video:
            return

        mask_mode = self._select_selfcal_mask_mode()
        if not mask_mode:
            return

        use_auto_mask = (mask_mode != "manual_only")
        use_manual_mask = (mask_mode != "auto_only")
        selected_manual_mask = None

        if use_manual_mask:
            reuse_existing = False
            if self.selfcal_mask_image is not None:
                reuse_existing = messagebox.askyesno(
                    "手動自己校正マスク",
                    "既存の自己校正用手動マスクがあります。\n再利用しますか？\n\n"
                    "「いいえ」を選ぶと、開始動画のフレームで描き直します。"
                )

            if reuse_existing:
                selected_manual_mask = self.selfcal_mask_image.copy()
            else:
                frame_for_mask = self._read_frame_for_selfcal_mask(selected_start_video)
                if frame_for_mask is None:
                    messagebox.showerror("エラー", "手動マスク用に開始動画の先頭フレームを読み込めませんでした。")
                    return
                drawn_mask = self._draw_manual_exclusion_mask_on_frame(
                    frame_for_mask,
                    title="自己校正用手動マスク作成",
                    existing_mask=self.selfcal_mask_image
                )
                if drawn_mask is None:
                    return
                self.selfcal_mask_image = drawn_mask
                selected_manual_mask = drawn_mask.copy()

        if not messagebox.askyesno(
            "夜間自己校正マップ生成",
            "選択した開始動画から、ソース内の後続分割動画を連続利用して20分分の自己校正マップを生成します。\n\n"
            f"開始動画:\n{selected_start_video}\n\n"
            f"マスク方式: {'自動+手動' if mask_mode == 'auto_plus_manual' else ('自動のみ' if mask_mode == 'auto_only' else '手動のみ')}\n\n"
            "対策として自動マスク(時刻表示/グロー領域)を作成し、データ欠損区間は自動スキップします。\n"
            "既存の distortion_map_x.npy / distortion_map_y.npy は上書きされます。\n\n"
            "続行しますか？"
        ):
            return

        module_dir = os.path.dirname(os.path.abspath(__file__))
        map_x_path = os.path.join(module_dir, "distortion_map_x.npy")
        map_y_path = os.path.join(module_dir, "distortion_map_y.npy")
        auto_mask_output_path = os.path.join(module_dir, "distortion_selfcal_auto_mask.png")
        manual_mask_output_path = os.path.join(module_dir, "distortion_selfcal_manual_mask.png")
        metadata_output_path = os.path.join(module_dir, "distortion_selfcal_meta.json")

        # Backup current maps if they exist so the user can roll back easily.
        backup_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_map_x = None
        backup_map_y = None
        try:
            if os.path.exists(map_x_path):
                backup_map_x = os.path.join(module_dir, f"distortion_map_x.backup_{backup_ts}.npy")
                shutil.copy2(map_x_path, backup_map_x)
            if os.path.exists(map_y_path):
                backup_map_y = os.path.join(module_dir, f"distortion_map_y.backup_{backup_ts}.npy")
                shutil.copy2(map_y_path, backup_map_y)
        except Exception as e:
            self.append_log(f"既存ゆがみマップのバックアップ作成に失敗しました: {e}")

        def run_task():
            self.append_log("夜間自己校正マップ生成を開始します... (先頭20分, 自動マスク有効)")
            self.append_log(f"開始動画: {selected_start_video}")
            self.append_log("この動画から後続の分割動画を連続利用して20分分を処理します。")
            self.append_log(f"マスク方式: {'自動+手動' if mask_mode == 'auto_plus_manual' else ('自動のみ' if mask_mode == 'auto_only' else '手動のみ')}")
            self.append_log("注意: 固定カメラの夜空動画を前提とします。欠損区間は自動スキップします。")
            try:
                if selected_manual_mask is not None:
                    try:
                        cv2.imwrite(manual_mask_output_path, selected_manual_mask)
                        self.append_log(f"手動自己校正マスクを保存しました: {manual_mask_output_path}")
                    except Exception as e_save_mask:
                        self.append_log(f"手動自己校正マスクの保存に失敗しました: {e_save_mask}")

                result = distortion_correction.estimate_distortion_map_from_night_sources(
                    sources=self.folder_paths,
                    map_x_path=map_x_path,
                    map_y_path=map_y_path,
                    duration_minutes=20.0,
                    sample_interval_sec=2.0,
                    progress_callback=self.append_log,
                    auto_mask_output_path=auto_mask_output_path,
                    metadata_output_path=metadata_output_path,
                    strength=0.5,
                    start_video_path=selected_start_video,
                    manual_mask=selected_manual_mask,
                    use_auto_mask=use_auto_mask,
                )
                stats = result.get("stats", {}) if isinstance(result, dict) else {}
                sample_count = stats.get("residual_samples_before_fit", "N/A")
                used_obs = stats.get("track_observations_used", "N/A")
                sampled_ok = stats.get("frames_sampled_success", "N/A")
                sampled_ng = stats.get("frames_sampled_failed", "N/A")
                p95_resid = stats.get("p95_residual_mag_px", None)
                used_videos = stats.get("videos_touched_count", "N/A")
                start_video_meta = stats.get("video_path_start", selected_start_video)

                summary_lines = [
                    "夜間自己校正マップ生成が完了しました。",
                    f"開始動画: {start_video_meta}",
                    f"使用動画数: {used_videos}",
                    f"マスク方式: {'自動+手動' if mask_mode == 'auto_plus_manual' else ('自動のみ' if mask_mode == 'auto_only' else '手動のみ')}",
                    f"map_x: {map_x_path}",
                    f"map_y: {map_y_path}",
                    f"自動マスク: {auto_mask_output_path}",
                    f"手動マスク: {manual_mask_output_path if selected_manual_mask is not None else '(未使用)'}",
                    f"メタ情報: {metadata_output_path}",
                    f"サンプル成功/失敗: {sampled_ok} / {sampled_ng}",
                    f"残差サンプル数: {sample_count} (観測使用数: {used_obs})",
                ]
                if p95_resid is not None:
                    try:
                        summary_lines.append(f"残差95%値: {float(p95_resid):.3f} px")
                    except Exception:
                        pass
                if backup_map_x or backup_map_y:
                    backup_info = []
                    if backup_map_x:
                        backup_info.append(f"X: {backup_map_x}")
                    if backup_map_y:
                        backup_info.append(f"Y: {backup_map_y}")
                    summary_lines.append("バックアップ作成済み")
                    summary_lines.extend(backup_info)

                self.append_log("夜間自己校正マップ生成が完了しました。")
                self.append_log(f"  map_x: {map_x_path}")
                self.append_log(f"  map_y: {map_y_path}")
                self.append_log(f"  自動マスク: {auto_mask_output_path}")
                self.append_log(f"  メタ情報: {metadata_output_path}")
                messagebox.showinfo("完了", "\n".join(summary_lines))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.append_log(f"夜間自己校正マップ生成に失敗しました: {e}")
                messagebox.showerror(
                    "エラー",
                    "夜間自己校正マップ生成に失敗しました。\n"
                    "ログを確認してください。\n\n"
                    f"詳細: {e}"
                )

        threading.Thread(target=run_task, daemon=True).start()

    def visualize_distortion_map_callback(self):
        """Visualize distortion map (map_x/map_y) on a Tk canvas as heatmap + vector field."""
        if not self.check_admin_password():
            return

        module_dir = os.path.dirname(os.path.abspath(__file__))
        default_map_x = os.path.join(module_dir, "distortion_map_x.npy")
        default_map_y = os.path.join(module_dir, "distortion_map_y.npy")
        default_meta = os.path.join(module_dir, "distortion_selfcal_meta.json")

        map_x_path = default_map_x
        map_y_path = default_map_y

        if not (os.path.exists(map_x_path) and os.path.exists(map_y_path)):
            messagebox.showinfo(
                "情報",
                "既定のゆがみマップが見つからないため、map_x / map_y ファイルを選択してください。"
            )
            map_x_path = filedialog.askopenfilename(
                title="distortion_map_x.npy を選択",
                initialdir=module_dir,
                filetypes=[("NumPy Array", "*.npy"), ("All Files", "*.*")]
            )
            if not map_x_path:
                return

            inferred_map_y = map_x_path.replace("map_x", "map_y")
            if os.path.exists(inferred_map_y):
                map_y_path = inferred_map_y
            else:
                map_y_path = filedialog.askopenfilename(
                    title="distortion_map_y.npy を選択",
                    initialdir=os.path.dirname(map_x_path),
                    filetypes=[("NumPy Array", "*.npy"), ("All Files", "*.*")]
                )
                if not map_y_path:
                    return

        try:
            map_x = np.load(map_x_path).astype(np.float32)
            map_y = np.load(map_y_path).astype(np.float32)
        except Exception as e:
            messagebox.showerror("エラー", f"ゆがみマップの読み込みに失敗しました:\n{e}")
            return

        if map_x.ndim != 2 or map_y.ndim != 2 or map_x.shape != map_y.shape:
            messagebox.showerror(
                "エラー",
                f"map_x / map_y の形状が不正です。\n"
                f"map_x: shape={getattr(map_x, 'shape', None)}\n"
                f"map_y: shape={getattr(map_y, 'shape', None)}"
            )
            return

        h, w = map_x.shape[:2]
        yy, xx = np.indices((h, w), dtype=np.float32)
        dx = map_x - xx
        dy = map_y - yy
        mag = np.hypot(dx, dy)

        finite_mask = np.isfinite(mag)
        if not np.any(finite_mask):
            messagebox.showerror("エラー", "ゆがみマップがすべて非数値です。")
            return

        valid_mag = mag[finite_mask]
        mag_max = float(np.max(valid_mag))
        mag_p50 = float(np.percentile(valid_mag, 50.0))
        mag_p95 = float(np.percentile(valid_mag, 95.0))
        mag_p99 = float(np.percentile(valid_mag, 99.0))
        norm_denom = max(1e-6, mag_p99 if mag_p99 > 0 else mag_max)

        heat_norm = np.clip(mag / norm_denom, 0.0, 1.0)
        heat_u8 = (heat_norm * 255.0).astype(np.uint8)
        heat_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)
        heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

        max_disp_w = 1100
        max_disp_h = 720
        scale = min(max_disp_w / float(w), max_disp_h / float(h), 1.0)
        disp_w = max(1, int(round(w * scale)))
        disp_h = max(1, int(round(h * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_NEAREST
        heat_disp = cv2.resize(heat_rgb, (disp_w, disp_h), interpolation=interp)

        meta_info = {}
        if os.path.exists(default_meta):
            try:
                with open(default_meta, "r", encoding="utf-8") as f:
                    meta_info = json.load(f)
            except Exception:
                meta_info = {}

        win = Toplevel(self)
        win.title("ゆがみマップ可視化")
        win.geometry(f"{min(disp_w + 80, 1280)}x{min(disp_h + 220, 980)}")
        win.transient(self)

        header = ttk.Frame(win, padding=10)
        header.pack(fill=tk.X)

        info_lines = [
            f"map_x: {map_x_path}",
            f"map_y: {map_y_path}",
            f"サイズ: {w} x {h}",
            f"変位量 [px]  p50={mag_p50:.3f}, p95={mag_p95:.3f}, p99={mag_p99:.3f}, max={mag_max:.3f}",
            "表示: 背景=変位量ヒートマップ, 矢印=map_x/map_y の変位ベクトル",
        ]
        if isinstance(meta_info, dict):
            vstart = meta_info.get("video_path_start") or meta_info.get("video_path")
            vcount = meta_info.get("videos_touched_count")
            if vstart:
                info_lines.append(f"自己校正開始動画: {vstart}")
            if vcount is not None:
                info_lines.append(f"自己校正で使用した動画数: {vcount}")
        ttk.Label(header, text="\n".join(info_lines), justify=tk.LEFT).pack(anchor=tk.W)

        canvas_frame = ttk.Frame(win, padding=(10, 0, 10, 10))
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        canvas = Canvas(canvas_frame, bg="#111111", highlightthickness=1, highlightbackground="#444444")
        canvas.pack(fill=tk.BOTH, expand=True)

        tk_img = ImageTk.PhotoImage(Image.fromarray(heat_disp))
        canvas.create_image(0, 0, image=tk_img, anchor="nw")
        canvas.config(scrollregion=(0, 0, disp_w, disp_h))
        win._distortion_map_preview_tk = tk_img  # prevent GC

        # Draw a sparse vector field on top of the heatmap.
        grid_step = max(48, min(120, int(min(w, h) / 10)))
        mag_thresh = max(0.1, mag_p95 * 0.08)
        vec_scale = float(np.clip(24.0 / max(mag_p95, 0.2), 3.0, 35.0))

        def rgb_to_hex(rgb_arr):
            r, g, b = int(rgb_arr[0]), int(rgb_arr[1]), int(rgb_arr[2])
            return f"#{r:02x}{g:02x}{b:02x}"

        for y in range(grid_step // 2, h, grid_step):
            for x in range(grid_step // 2, w, grid_step):
                m = float(mag[y, x])
                if not np.isfinite(m) or m < mag_thresh:
                    continue
                x0 = x * scale
                y0 = y * scale
                vx = float(dx[y, x]) * scale * vec_scale
                vy = float(dy[y, x]) * scale * vec_scale
                x1 = x0 + vx
                y1 = y0 + vy
                color = rgb_to_hex(heat_rgb[y, x])
                canvas.create_line(
                    x0, y0, x1, y1,
                    fill=color,
                    width=2 if m >= mag_p95 else 1,
                    arrow=tk.LAST,
                    arrowshape=(8, 10, 3)
                )
                canvas.create_oval(x0 - 1.5, y0 - 1.5, x0 + 1.5, y0 + 1.5, fill=color, outline="")

        # Corner/center guides
        guide_color = "#FFFFFF"
        for gx, gy, label in (
            (0, 0, "TL"),
            (w - 1, 0, "TR"),
            (0, h - 1, "BL"),
            (w - 1, h - 1, "BR"),
            (w / 2, h / 2, "C"),
        ):
            px = gx * scale
            py = gy * scale
            canvas.create_line(px - 8, py, px + 8, py, fill=guide_color, width=1)
            canvas.create_line(px, py - 8, px, py + 8, fill=guide_color, width=1)
            canvas.create_text(px + 12, py + 12, text=label, fill=guide_color, anchor="nw", font=("Arial", 9, "bold"))

        footer = ttk.Frame(win, padding=(10, 0, 10, 10))
        footer.pack(fill=tk.X)
        ttk.Label(
            footer,
            text="注: この可視化は map_x/map_y の変位を表示するもので、補正の良し悪しは別途プレートソルブ誤差で評価してください。",
            foreground="#87CEEB"
        ).pack(anchor=tk.W)

    def analyze_angles_callback(self):
        """Callback for the 'Angle Distribution Analysis' button."""
        if not self.check_admin_password():
            return

        if not self.analysis_files:
            messagebox.showwarning("情報", "解析するファイルを追加してください。")
            return

        ra_str = simpledialog.askstring("放射点入力", "放射点の赤経 (RA) を度数 (deg) で入力してください:\n(例: 45.0)")
        if ra_str is None: return
        try:
            radiant_ra = float(ra_str)
        except ValueError:
            messagebox.showerror("エラー", "有効な数値を入力してください。")
            return

        dec_str = simpledialog.askstring("放射点入力", "放射点の赤緯 (Dec) を度数 (deg) で入力してください:\n(例: 30.0)")
        if dec_str is None: return
        try:
            radiant_dec = float(dec_str)
        except ValueError:
            messagebox.showerror("エラー", "有効な数値を入力してください。")
            return

        output_path = filedialog.asksaveasfilename(
            title="角度分布グラフの保存先",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("All Files", "*")]
        )
        
        if not output_path:
            return

        def run_task():
            self.append_log("角度分布分析を開始します...")
            success, msg = meteor_angle_analysis.analyze_angles(
                self.analysis_files, 
                radiant_ra, 
                radiant_dec, 
                output_path
            )
            if success:
                messagebox.showinfo("完了", msg)
                self.append_log(f"角度分布分析完了: {msg}")
            else:
                messagebox.showerror("エラー", msg)
                self.append_log(f"角度分布分析失敗: {msg}")

        threading.Thread(target=run_task, daemon=True).start()

    def create_lighten_blend_video_callback(self):
        """Callback for the 'Create Lighten Blend Video' button."""
        initial_dir = self.meteor_save_path_var.get()
        if not initial_dir or not os.path.exists(initial_dir):
            initial_dir = os.path.expanduser("~")
        
        video_files = filedialog.askopenfilenames(
            title="比較明合成する動画ファイルを選択（複数可）",
            initialdir=initial_dir,
            filetypes=[
                ("動画ファイル", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("すべてのファイル", "*.*")
            ]
        )
        
        if not video_files:
            return
        
        if len(video_files) < 2:
            messagebox.showwarning("警告", "比較明合成を行うには2つ以上の動画ファイルを選択してください。")
            return
        
        default_output = lighten_blend_video.get_default_output_path()
        
        output_path = filedialog.asksaveasfilename(
            title="比較明合成動画の保存先",
            initialdir=os.path.dirname(default_output),
            initialfile=os.path.basename(default_output),
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4"), ("AVI Video", "*.avi"), ("All Files", "*")]
        )
        
        if not output_path:
            return
        output_path = self._ensure_date_prefix(output_path)
        
        self.append_log(f"比較明合成動画の作成を開始します... ({len(video_files)}個の動画)")
        
        dialog = ProcessingOptionDialog(self)
        if dialog.result is None:  # キャンセル
            return
            
        mode = dialog.result  # 0:通常, 1:明るいエリアマスク, 2:流星のみ
        # mode=1(明るいエリア)は現状 mode=2(流星のみ)と同じ扱いにする。
        if mode == 0:
            def run_task():
                success = lighten_blend_video.create_lighten_blend_video(
                    list(video_files),
                    output_path,
                    progress_callback=self.append_log
                )
                if success:
                    messagebox.showinfo("完了", f"比較明合成動画の作成が完了しました。\\n保存先: {output_path}")
                    self.append_log(f"比較明合成動画の作成が完了しました: {output_path}")
                else:
                    messagebox.showerror("エラー", "比較明合成動画の作成に失敗しました。ログを確認してください。")
                    self.append_log("比較明合成動画の作成に失敗しました。")
            
            threading.Thread(target=run_task, daemon=True).start()
        else:
            self._create_lighten_blend_video_with_meteor_detection(video_files, output_path)

    def _create_lighten_blend_video_with_meteor_detection(self, video_files, output_path):
        """AI流星検出を使用して比較明合成動画を作成（各動画ごとに検出）"""
        import detection_preview
        import bright_area_detector
        import gc
        
        self.append_log("各動画から比較明合成画像を作成し、流星を検出します...")
        
        # 動画ごとの一時ファイルパスを保存（画像はメモリに保持しない）
        video_composites = {}  # {video_path: {'temp_path': str, 'filename': str, 'shape': (h, w)}}
        
        def run_prep_task():
            if not self._ensure_ai_model_loaded(bright_area_detector):
                return

            # Step 1: 各動画から個別に比較明合成画像を作成
            for i, vp in enumerate(video_files):
                self.append_log(f"動画 {i+1}/{len(video_files)} から合成画像を作成中: {os.path.basename(vp)}")
                
                composite_image = lighten_blend_video.create_composite_from_videos(
                    [vp],  # 1つの動画のみ
                    progress_callback=None,  # 個別のログは抑制
                    sample_interval=1  # 全フレームを使用（流星を見逃さないため）
                )
                
                if composite_image is not None:
                    # 一時ファイルとして保存
                    temp_path = os.path.join(config.TEMP_CLIP_DIR, f"temp_composite_{i}_{os.path.basename(vp)}.png")
                    h, w = composite_image.shape[:2]
                    cv2.imwrite(temp_path, composite_image)
                    
                    video_composites[vp] = {
                        'temp_path': temp_path,
                        'filename': os.path.basename(temp_path),
                        'shape': (h, w)  # サイズ情報のみ保持
                    }
                    
                    # メモリ解放
                    del composite_image
                    gc.collect()
                else:
                    self.append_log(f"警告: 動画から合成画像を作成できませんでした: {os.path.basename(vp)}")
            
            if not video_composites:
                messagebox.showerror("エラー", "有効な合成画像を作成できませんでした。")
                return
            
            self.append_log(f"{len(video_composites)}個の合成画像を作成しました。")
            
            # メインスレッドでプレビューウィンドウを開く
            def open_preview():
                # 合成開始コールバック
                def start_video_synthesis_with_results(results):
                    # 動画ごとの個別マスクを作成（和集合ではなく、各動画に対応するマスクのみ適用）
                    per_video_masks = {}
                    base_shape = None
                    has_detections = False
                    
                    for vp, data in video_composites.items():
                        filename = data['filename']
                        if filename in results:
                            boxes = results[filename]['boxes']
                            if boxes:
                                has_detections = True
                                h, w = data['shape']
                                if base_shape is None:
                                    base_shape = (h, w)
                                
                                mask = bright_area_detector.create_inclusion_mask_from_boxes(
                                    (h, w), boxes, margin=40
                                )
                                
                                # サイズが異なる場合はリサイズ
                                if base_shape != (h, w):
                                    mask = cv2.resize(mask, (base_shape[1], base_shape[0]))
                                
                                # 動画パスをキーとして個別マスクを保存
                                per_video_masks[vp] = mask
                    
                    if not has_detections:
                        if not messagebox.askyesno("確認", "流星が検出されていないか、選択されていません。\\nマスクなしで（通常の比較明合成として）作成しますか？"):
                            # 一時ファイルのクリーンアップ
                            for data in video_composites.values():
                                try:
                                    if os.path.exists(data['temp_path']):
                                        os.remove(data['temp_path'])
                                except:
                                    pass
                            return
                    
                    # 動画ごとのマスクを適用して動画作成
                    def run_video_task():
                        self.append_log("動画ごとのマスクを適用して動画を作成中...")
                        success = lighten_blend_video.create_lighten_blend_video(
                            list(video_files),
                            output_path,
                            progress_callback=self.append_log,
                            per_video_masks=per_video_masks if per_video_masks else None
                        )
                        if success:
                            messagebox.showinfo("完了", f"比較明合成動画の作成が完了しました。\\n保存先: {output_path}")
                            self.append_log(f"比較明合成動画の作成が完了しました: {output_path}")
                        else:
                            messagebox.showerror("エラー", "比較明合成動画の作成に失敗しました。ログを確認してください。")
                            self.append_log("比較明合成動画の作成に失敗しました。")
                        
                        # 一時ファイルのクリーンアップ
                        for data in video_composites.values():
                            try:
                                if os.path.exists(data['temp_path']):
                                    os.remove(data['temp_path'])
                            except:
                                pass
                    
                    threading.Thread(target=run_video_task, daemon=True).start()
                
                # プレビューウィンドウ作成
                preview_window = detection_preview.DetectionPreviewWindow(
                    self, start_video_synthesis_with_results
                )
                
                # 各動画の合成画像に対して検出実行
                def run_detection():
                    total = len(video_composites)
                    self.append_log(f"AIによる流星検出を開始します... ({total}個の画像)")
                    
                    if preview_window.winfo_exists():
                        self.after(0, lambda: preview_window.start_analysis(total))
                    
                    for i, (vp, data) in enumerate(video_composites.items()):
                        if not preview_window.winfo_exists():
                            self.append_log("検出が中断されました。")
                            return
                        
                        self.append_log(f"検出中 ({i+1}/{total}): {os.path.basename(vp)}")
                        
                        # 一時ファイルから画像を読み込み（メモリ節約）
                        composite_img = imread_with_japanese_path(data['temp_path'])
                        if composite_img is None:
                            continue
                        
                        res = bright_area_detector.detect_meteors_with_boxes(
                            composite_img,
                            progress_callback=None  # 個別ログ抑制
                        )
                        boxes = res[1] if res else []
                        
                        # 検出後すぐにメモリ解放
                        del composite_img
                        gc.collect()
                        
                        def reanalyze_wrapper(image):
                            return bright_area_detector.detect_meteors_with_boxes(image)
                        
                        if preview_window.winfo_exists():
                            filename = data['filename']
                            temp_path = data['temp_path']
                            self.after(0, lambda fn=filename, fp=temp_path, b=boxes, cb=reanalyze_wrapper:
                                preview_window.add_item(fn, fp, b, cb))
                    
                    if preview_window.winfo_exists():
                        self.after(0, preview_window.finalize_analysis)
                    self.append_log("全画像の検出が完了しました。結果を確認してください。")
                
                threading.Thread(target=run_detection, daemon=True).start()
            
            self.after(0, open_preview)
        
        threading.Thread(target=run_prep_task, daemon=True).start()

    def create_lighten_blend_image_callback(self):
        """比較明合成画像作成ボタンのコールバック"""
        initial_dir = self.meteor_save_path_var.get()
        if not initial_dir or not os.path.exists(initial_dir):
            initial_dir = os.path.expanduser("~")
        
        # ファイル選択ダイアログで複数の画像/動画ファイルを選択
        file_paths_tuple = filedialog.askopenfilenames(
            title="比較明合成する画像・動画ファイルを選択（複数可）",
            initialdir=initial_dir,
            filetypes=[
                ("画像・動画ファイル", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.mp4 *.avi *.mov *.mkv *.wmv"),
                ("画像ファイル", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("動画ファイル", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("すべてのファイル", "*.*")
            ]
        )
        
        if not file_paths_tuple:
            return
            
        file_paths = set(file_paths_tuple)
        
        if len(file_paths) < 1:
            messagebox.showwarning("警告", "有効なファイルが見つかりません。")
            return

        # 初期ディレクトリ設定
        default_output = "composite.png"
        if len(file_paths) == 1:
            first_path = list(file_paths)[0]
            if os.path.isdir(first_path):
                default_output = os.path.join(os.path.dirname(first_path), f"{os.path.basename(first_path)}_composite.png")
            else:
                 default_output = os.path.join(os.path.dirname(first_path), f"{os.path.splitext(os.path.basename(first_path))[0]}_composite.png")
        
        # ユーザーに保存先を確認
        output_path = filedialog.asksaveasfilename(
            title="比較明合成画像の保存先",
            initialdir=os.path.dirname(default_output),
            initialfile=os.path.basename(default_output),
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("All Files", "*")]
        )
        
        if not output_path:
            return
        output_path = self._ensure_date_prefix(output_path)

        dialog = ProcessingOptionDialog(self)
        if dialog.result is None:  # キャンセル
            return
            
        mode = dialog.result  # 0:通常, 1:明るいエリアマスク, 2:流星のみ
        is_ai_mode = (mode != 0)
        is_meteor_mode = (mode == 2)
        print(f"DEBUG: ProcessingOptionDialog result: mode={mode}, is_ai_mode={is_ai_mode}, is_meteor_mode={is_meteor_mode}")

        if not is_ai_mode:
            def run_normal_task():
                self.append_log(f"比較明合成画像の作成を開始します... ({len(file_paths)}個の要素)")
                success = lighten_blend_image.create_lighten_blend_image(
                    list(file_paths),
                    output_path,
                    progress_callback=self.append_log
                )
                self._handle_synthesis_result(success, output_path)
            
            threading.Thread(target=run_normal_task, daemon=True).start()
            return

        # AI解析モードの場合
        print("DEBUG: AI解析モードに入りました")
        import detection_preview
        import bright_area_detector
        import cv2

        detector_func = bright_area_detector.detect_meteors_with_boxes if is_meteor_mode else bright_area_detector.detect_bright_areas_with_boxes
        print(f"DEBUG: detector_func = {detector_func.__name__}")
        
        def start_synthesis_with_results(results):
            def run_ai_task():
                self.append_log("AI解析結果に基づく合成処理を開始します...")
                # Mask generation must follow the same expanded file order as synthesis.
                
                all_files = []
                image_ext, video_ext = lighten_blend_image.get_supported_extensions()
                for path in file_paths:
                    if os.path.isdir(path):
                        all_files.extend(lighten_blend_image.collect_files_from_folder(path))
                    elif os.path.isfile(path):
                        all_files.append(path)
                
                all_files.sort()  # 名前順で処理されると仮定
                
                # インデックス管理用
                file_index = [0]
                
                def mask_generator(img):
                    if file_index[0] >= len(all_files):
                        return None
                    
                    current_path = all_files[file_index[0]]
                    filename = os.path.basename(current_path)
                    file_index[0] += 1
                    
                    if filename in results:
                        boxes = results[filename]['boxes']
                        h, w = img.shape[:2]
                        if is_meteor_mode:
                            return bright_area_detector.create_inclusion_mask_from_boxes((h, w), boxes)
                        else:
                            return bright_area_detector.create_mask_from_boxes((h, w), boxes)
                    return None

                success = lighten_blend_image.create_lighten_blend_image(
                    all_files,
                    output_path,
                    progress_callback=self.append_log,
                    mask_generator=mask_generator,
                    inclusion_mode=is_meteor_mode
                )
                self._handle_synthesis_result(success, output_path)

            threading.Thread(target=run_ai_task, daemon=True).start()

        preview_window = detection_preview.DetectionPreviewWindow(self, start_synthesis_with_results)
        
        def run_analysis_task():
            try:
                print("DEBUG: run_analysis_task started")
                self.append_log("AIによる画像解析を開始します...")

                if not self._ensure_ai_model_loaded(bright_area_detector):
                    if preview_window.winfo_exists():
                        self.after(0, preview_window.destroy)
                    return
                
                all_files = []
                for path in file_paths:
                    if os.path.isdir(path):
                        all_files.extend(lighten_blend_image.collect_files_from_folder(path))
                    elif os.path.isfile(path):
                        all_files.append(path)
                all_files.sort()
                
                total = len(all_files)
                print(f"DEBUG: Processing {total} files")
                
                if preview_window.winfo_exists():
                    self.after(0, lambda: preview_window.start_analysis(total))
                
                for i, path in enumerate(all_files):
                    if not preview_window.winfo_exists():
                        self.append_log("解析が中断されました。")
                        return
                    
                    filename = os.path.basename(path)
                    self.append_log(f"解析中 ({i+1}/{total}): {filename}")
                    print(f"DEBUG: Analyzing file {i+1}/{total}: {filename}")
                    
                    img = imread_with_japanese_path(path)
                    if img is None:
                        print(f"DEBUG: Failed to read image: {path}")
                        continue
                    
                    def reanalyze_wrapper(image):
                        res = detector_func(image)
                        return res if res else (None, [])
                    
                    print(f"DEBUG: Calling detector_func for {filename}")
                    res = detector_func(img)
                    boxes = res[1] if res else []
                    print(f"DEBUG: Detection result for {filename}: {len(boxes)} boxes found")
                    
                    if preview_window.winfo_exists():
                        self.after(0, lambda fn=filename, fp=path, b=boxes, cb=reanalyze_wrapper: 
                                  preview_window.add_item(fn, fp, b, cb))
                
                self.append_log("全画像の解析が完了しました。検出結果を確認・修正してください。")
                print("DEBUG: Analysis completed")
                
                if preview_window.winfo_exists():
                     # 完了処理（未検出を先頭になど）
                     self.after(0, preview_window.finalize_analysis)
                     messagebox.showinfo("解析完了", "全画像の解析が完了しました。\n未検出の画像が上部に表示されています。\nプレビュー画面で結果を確認し、「修正を確定して合成を開始」ボタンを押してください。")
            except Exception as e:
                print(f"ERROR: run_analysis_task failed with exception: {e}")
                import traceback
                traceback.print_exc()
                self.append_log(f"AI解析中にエラーが発生しました: {e}")

        print("DEBUG: Starting run_analysis_task thread")
        threading.Thread(target=run_analysis_task, daemon=True).start()

    def _handle_synthesis_result(self, success, output_path):
        if success:
            messagebox.showinfo("完了", f"比較明合成画像の作成が完了しました。\n保存先: {output_path}")
            self.append_log(f"比較明合成画像の作成が完了しました: {output_path}")
        else:
            messagebox.showerror("エラー", "比較明合成画像の作成に失敗しました。ログを確認してください。")
            self.append_log("比較明合成画像の作成に失敗しました。")

    def _ensure_ai_model_loaded(self, detector_module) -> bool:
        """AI合成前に内部LLMのロード完了を保証する。"""
        self.append_log("AIモデルのロードを確認中...")

        local_model_dir = getattr(detector_module, "LOCAL_MODEL_DIR", "./quantized_model")
        has_local_model = False
        try:
            has_model_fn = getattr(detector_module, "has_quantized_model", None)
            if callable(has_model_fn):
                has_local_model = bool(has_model_fn())
            else:
                has_local_model = os.path.isdir(local_model_dir)
        except Exception as e:
            self.append_log(f"ローカルモデル状態の確認中にエラー: {e}")
            has_local_model = os.path.isdir(local_model_dir)

        if not has_local_model:
            self.append_log(f"ローカルLLMモデルが見つかりません: {local_model_dir}")
            self.append_log("必要なディスク容量を見積もり中...")
            req = self._estimate_llm_storage_requirements(detector_module)

            info_lines = [
                "ローカルLLMモデルが見つかりませんでした。",
                f"対象モデル: {req['repo_id']}",
                f"一時的に必要な空き容量 (目安): {self._format_size_bytes(req['temporary_bytes'])}",
                f"最終的に必要な容量 (目安): {self._format_size_bytes(req['final_bytes'])}",
                f"現在の空き容量: {self._format_size_bytes(req['free_bytes'])}",
                "",
                "モデルをダウンロードしますか？",
            ]
            if not req["fetched_metadata"]:
                info_lines.insert(4, "※ 容量は取得失敗のため既定値での目安です。")
            if req["free_bytes"] < req["temporary_bytes"]:
                info_lines.insert(5, "※ 警告: 空き容量が一時必要容量を下回っています。")

            should_download = bool(
                self._run_on_main_thread(
                    lambda msg="\n".join(info_lines): messagebox.askyesno("モデルダウンロード確認", msg, parent=self)
                )
            )

            if not should_download:
                self.append_log("ユーザーがモデルダウンロードをキャンセルしました。")
                return False

            self.append_log("モデルダウンロードを開始します。")
            mirror_stream = _StderrProgressStream(self.append_log, passthrough=sys.stderr)
            with contextlib.redirect_stderr(mirror_stream), contextlib.redirect_stdout(mirror_stream):
                connected, err = detector_module.check_vlm_connection(status_callback=self.append_log)
            mirror_stream.flush()
        else:
            connected, err = detector_module.check_vlm_connection(status_callback=self.append_log)

        if connected:
            self.append_log("AIモデルのロードが完了しました。")
            return True

        error_message = f"AIモデルのロードに失敗しました: {err}"
        self.append_log(error_message)
        self.after(0, lambda msg=error_message: messagebox.showerror("エラー", msg, parent=self))
        return False


    def create_timelapse_callback(self):
        """タイムラプス作成ボタンのコールバック。ドラッグ＆ドロップウィンドウを表示する。"""
        TimelapseDragDropWindow(self, self.append_log)


class TimelapseDragDropWindow(Toplevel):
    """タイムラプス作成用のドラッグ＆ドロップウィンドウ"""
    
    def __init__(self, parent, log_callback):
        super().__init__(parent)
        self.parent = parent
        self.log_callback = log_callback
        self.dropped_paths = []
        self.timelapse_mask = None  # タイムラプス用マスク
        
        self.title("タイムラプス作成")
        self.geometry("500x600")
        self.resizable(False, False)
        
        self.setup_ui()
        
        self.transient(parent)
        self.grab_set()
    
    def setup_ui(self):
        main_frame = ttk.Frame(self, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        drop_frame = ttk.LabelFrame(main_frame, text="ファイル / フォルダ", padding=10)
        drop_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.drop_label = ttk.Label(
            drop_frame, 
            text="ここにフォルダや動画ファイルを\nドラッグ＆ドロップしてください",
            relief=tk.SOLID, 
            padding=30, 
            anchor=tk.CENTER,
            borderwidth=2,
            justify=tk.CENTER
        )
        self.drop_label.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.drop_label.drop_target_register(DND_FILES)
        self.drop_label.dnd_bind('<<Drop>>', self.on_drop)
        
        list_frame = ttk.Frame(drop_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.listbox = tk.Listbox(
            list_frame, 
            height=5, 
            bg="#3A4D6B", 
            fg="#EAEAEA",
            relief=tk.FLAT,
            highlightthickness=0
        )
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.config(yscrollcommand=scrollbar.set)
        
        ttk.Button(drop_frame, text="リストをクリア", command=self.clear_list).pack(anchor=tk.E, pady=(5, 0))
        
        duration_frame = ttk.LabelFrame(main_frame, text="動画の長さ", padding=10)
        duration_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.duration_var = tk.IntVar(value=30)
        
        duration_options = ttk.Frame(duration_frame)
        duration_options.pack()
        
        ttk.Radiobutton(duration_options, text="15秒", variable=self.duration_var, value=15).pack(side=tk.LEFT, padx=15)
        ttk.Radiobutton(duration_options, text="30秒", variable=self.duration_var, value=30).pack(side=tk.LEFT, padx=15)
        ttk.Radiobutton(duration_options, text="60秒", variable=self.duration_var, value=60).pack(side=tk.LEFT, padx=15)
        
        mask_frame = ttk.LabelFrame(main_frame, text="マスク設定", padding=10)
        mask_frame.pack(fill=tk.X, pady=(0, 10))
        
        mask_controls = ttk.Frame(mask_frame)
        mask_controls.pack(fill=tk.X)
        
        self.mask_btn = ttk.Button(mask_controls, text="マスク作成", command=self.create_timelapse_mask)
        self.mask_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_mask_btn = ttk.Button(mask_controls, text="マスクをクリア", command=self.clear_timelapse_mask, state=tk.DISABLED)
        self.clear_mask_btn.pack(side=tk.LEFT, padx=5)
        
        self.mask_status_label = ttk.Label(mask_controls, text="マスクなし")
        self.mask_status_label.pack(side=tk.LEFT, padx=10)
        
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X)
        
        ttk.Button(btn_frame, text="作成開始", command=self.start_creation).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="キャンセル", command=self.destroy).pack(side=tk.LEFT, padx=5)
    
    def on_drop(self, event):
        """ドラッグ＆ドロップのイベントハンドラ"""
        # splitlistを使用してパスを分解
        try:
            paths = self.tk.splitlist(event.data)
        except:
            paths = [event.data]
        
        for path in paths:
            path = path.strip('{}')
            if path and path not in self.dropped_paths:
                self.dropped_paths.append(path)
                self.listbox.insert(tk.END, os.path.basename(path))
        
        self.update_drop_label()
    
    def update_drop_label(self):
        """ドロップラベルのテキストを更新"""
        if self.dropped_paths:
            self.drop_label.config(text=f"{len(self.dropped_paths)}個のアイテムが追加されました\n(さらに追加できます)")
        else:
            self.drop_label.config(text="ここにフォルダや動画ファイルを\nドラッグ＆ドロップしてください")
    
    def clear_list(self):
        """リストをクリア"""
        self.dropped_paths.clear()
        self.listbox.delete(0, tk.END)
        self.update_drop_label()
    
    def create_timelapse_mask(self):
        """タイムラプス用マスクを作成"""
        if not self.dropped_paths:
            messagebox.showwarning("警告", "先にファイルまたはフォルダをドロップしてください。")
            return
        
        # 最初のファイルからフレームを取得
        from pathlib import Path
        from PIL import Image, ImageTk, ImageDraw
        
        first_frame = None
        for path in self.dropped_paths:
            if os.path.isfile(path):
                ext = Path(path).suffix.lower()
                if ext in {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v'}:
                    cap = cv2.VideoCapture(path)
                    ret, first_frame = cap.read()
                    cap.release()
                    if ret:
                        break
                elif ext in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}:
                    first_frame = cv2.imread(path)
                    if first_frame is not None:
                        break
            elif os.path.isdir(path):
                # ディレクトリから最初の動画または画像を探す
                for f in sorted(os.listdir(path)):
                    fpath = os.path.join(path, f)
                    if os.path.isfile(fpath):
                        ext = Path(fpath).suffix.lower()
                        if ext in {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v'}:
                            cap = cv2.VideoCapture(fpath)
                            ret, first_frame = cap.read()
                            cap.release()
                            if ret:
                                break
                        elif ext in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}:
                            first_frame = cv2.imread(fpath)
                            if first_frame is not None:
                                break
                if first_frame is not None:
                    break
        
        if first_frame is None:
            messagebox.showerror("エラー", "フレームを取得できませんでした。")
            return
        
        mask_win = Toplevel(self)
        mask_win.title("タイムラプス用マスク作成")
        mask_win.geometry("1000x700")
        mask_win.grab_set()
        mask_win.transient(self)
        
        orig_h, orig_w = first_frame.shape[:2]
        disp_w, disp_h = 960, 540
        scale = min(disp_w / orig_w, disp_h / orig_h)
        disp_w, disp_h = int(orig_w * scale), int(orig_h * scale)
        
        frame_disp = cv2.resize(first_frame, (disp_w, disp_h))
        tk_image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB)))
        
        canvas = Canvas(mask_win, width=disp_w, height=disp_h, cursor="circle")
        canvas.pack(pady=5)
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
        canvas.image = tk_image
        
        mask_data_disp = Image.new("L", (disp_w, disp_h), 0)
        draw = ImageDraw.Draw(mask_data_disp)
        
        brush_radius = tk.IntVar(value=30)
        
        def paint(event):
            r = brush_radius.get()
            canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r, fill='red', outline='red', tags="paint")
            draw.ellipse((event.x - r, event.y - r, event.x + r, event.y + r), fill=255)
        
        canvas.bind("<B1-Motion>", paint)
        canvas.bind("<Button-1>", paint)
        
        controls = ttk.Frame(mask_win)
        controls.pack(fill=tk.X, padx=10)
        ttk.Label(controls, text="ブラシサイズ:").pack(side=tk.LEFT)
        ttk.Scale(controls, from_=5, to=100, orient=tk.HORIZONTAL, variable=brush_radius).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        def clear_drawing():
            canvas.delete("paint")
            draw.rectangle([0, 0, disp_w, disp_h], fill=0)
        
        ttk.Button(controls, text="クリア", command=clear_drawing).pack(side=tk.LEFT)
        
        def on_ok():
            mask_np_disp = np.array(mask_data_disp)
            # 白黒反転（描画部分=マスク=0、非描画部分=表示=255）
            if mask_np_disp.max() > 0:
                mask_resized = cv2.resize(mask_np_disp, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                final_mask = cv2.bitwise_not(mask_resized)
            else:
                final_mask = None  # マスクなし
            
            self.timelapse_mask = final_mask
            if final_mask is not None:
                self.mask_status_label.config(text="マスクあり")
                self.clear_mask_btn.config(state=tk.NORMAL)
            mask_win.destroy()
        
        btn_frame = ttk.Frame(mask_win)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="キャンセル", command=mask_win.destroy).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(mask_win, text="※塗りつぶした領域が黒くマスクされます", foreground="gray").pack()
    
    def clear_timelapse_mask(self):
        """タイムラプス用マスクをクリア"""
        self.timelapse_mask = None
        self.mask_status_label.config(text="マスクなし")
        self.clear_mask_btn.config(state=tk.DISABLED)
    
    def start_creation(self):
        """タイムラプス作成を開始"""
        if not self.dropped_paths:
            messagebox.showwarning("警告", "ファイルまたはフォルダをドロップしてください。")
            return
        
        # 保存先を選択
        default_output = timelapse_creator.get_default_output_path()
        output_path = filedialog.asksaveasfilename(
            title="タイムラプス動画の保存先",
            initialdir=os.path.dirname(default_output),
            initialfile=os.path.basename(default_output),
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4"), ("AVI Video", "*.avi"), ("All Files", "*")]
        )
        
        if not output_path:
            return
        try:
            output_path = self.parent._ensure_date_prefix(output_path)
        except Exception:
            pass
        
        duration = self.duration_var.get()
        paths = list(self.dropped_paths)
        mask = self.timelapse_mask  # マスクを保存
        
        # ウィンドウを閉じる
        self.destroy()
        
        mask_status = "あり" if mask is not None else "なし"
        self.log_callback(f"タイムラプス作成を開始します... (長さ: {duration}秒, {len(paths)}個のアイテム, マスク: {mask_status})")
        
        def run_task():
            success = timelapse_creator.create_timelapse(
                paths,
                output_path,
                target_duration_seconds=duration,
                progress_callback=self.log_callback,
                mask=mask
            )
            if success:
                messagebox.showinfo("完了", f"タイムラプス動画の作成が完了しました。\n保存先: {output_path}")
                self.log_callback(f"タイムラプス動画の作成が完了しました: {output_path}")
            else:
                messagebox.showerror("エラー", "タイムラプス動画の作成に失敗しました。ログを確認してください。")
                self.log_callback("タイムラプス動画の作成に失敗しました。")
        
        threading.Thread(target=run_task, daemon=True).start()



class ProcessingOptionDialog(tk.Toplevel):
    def __init__(self, parent):
        print("DEBUG: ProcessingOptionDialog initialized")
        super().__init__(parent)
        self.title("処理オプション")
        self.result = None
        self.geometry("500x320") 
        self.resizable(False, False)
        
        # メインフレームを作成して全体に配置（テーマの背景色を適用するため）
        self.main_frame = ttk.Frame(self, padding="20 20 20 10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(self.main_frame, text="比較明合成オプション", font=("", 14, "bold")).pack(anchor=tk.W, pady=(0, 15))
        
        mode_frame = ttk.LabelFrame(self.main_frame, text="モード選択", padding=10)
        mode_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.mode_var = tk.IntVar(value=0)
        
        ttk.Radiobutton(mode_frame, text="通常合成 (AIを使用しない)", variable=self.mode_var, value=0).pack(anchor=tk.W, pady=5)
        self.rb_bright = ttk.Radiobutton(mode_frame, text="明るいエリアをマスク (AI検出)", variable=self.mode_var, value=1)
        self.rb_bright.pack(anchor=tk.W, pady=5)
        self.rb_meteor = ttk.Radiobutton(mode_frame, text="流星のみ合成 (AI検出)", variable=self.mode_var, value=2)
        self.rb_meteor.pack(anchor=tk.W, pady=5)
        
        # VRAM warning
        warning_frame = ttk.Frame(self.main_frame)
        warning_frame.pack(fill=tk.X, pady=(5, 10))
        # 黄色 (#FFD700) に変更
        ttk.Label(warning_frame, text="※AI検出を選択時、VRAMが7GB未満の場合は", font=("", 9), foreground="#FFD700").pack(anchor=tk.W)
        ttk.Label(warning_frame, text="  動作が非常に遅くなる可能性があります。", font=("", 9), foreground="#FFD700").pack(anchor=tk.W)
        
        btn_frame = ttk.Frame(self.main_frame)
        btn_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        ttk.Button(btn_frame, text="キャンセル", command=self.destroy).pack(side=tk.RIGHT, padx=5)
        self.next_btn = ttk.Button(btn_frame, text="次へ", command=self.on_ok, state=tk.NORMAL) # 最初から有効化
        self.next_btn.pack(side=tk.RIGHT, padx=5)
        
        self.transient(parent)
        self.grab_set()
        
        self.update_idletasks()
        try:
            x = parent.winfo_x() + (parent.winfo_width() // 2) - (self.winfo_width() // 2)
            y = parent.winfo_y() + (parent.winfo_height() // 2) - (self.winfo_height() // 2)
            self.geometry(f"+{x}+{y}")
        except:
            pass
            
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.wait_window(self)
    
    def on_ok(self):
        self.result = self.mode_var.get()
        self.destroy()
    
    def _setup_help_tooltip(self, widget):
        pass

def worker_main_loop(
    progress_queue: queue.Queue, sources: List[Dict[str, Any]], max_workers: int, interval: float, duration: float,
    mask: Optional[np.ndarray], global_wcs_info: Optional[Dict], plate_solve_mask: Optional[np.ndarray],
    meteor_save_path: str, not_meteor_save_path: str, cancel_flag: threading.Event,
    save_options: Dict[str, bool], summary_video_config: List[Dict[str, Any]]
):
    total_videos = len(sources)
    if total_videos == 0: return

    # Use a two-stage pipeline (downloaders + processors) implemented in download_pipeline
    module_dir = os.path.dirname(os.path.abspath(__file__))
    tmp_root_dir = os.path.join(module_dir, 'temp_video')
    os.makedirs(tmp_root_dir, exist_ok=True)

    try:
        download_pipeline.run_pipeline(
            sources=sources,
            max_workers=max_workers,
            interval=interval,
            duration=duration,
            mask=mask,
            global_wcs_info=global_wcs_info,
            plate_solve_mask=plate_solve_mask,
            meteor_save_path=meteor_save_path,
            not_meteor_save_path=not_meteor_save_path,
            cancel_flag=cancel_flag,
            progress_callback=progress_queue.put,
            save_options=save_options,
            summary_video_config=summary_video_config,
            tmp_root=tmp_root_dir,
            status_callback=STATUS_CALLBACK,
        )

        if not cancel_flag.is_set():
            progress_queue.put(("すべての処理が完了しました。", None))
        else:
            progress_queue.put(("処理はキャンセルされました。", None))

    except Exception as e:
        progress_queue.put((f"パイプライン実行中に例外が発生しました: {e}", None))


if __name__ == "__main__":
    os.makedirs(config.TEMP_CLIP_DIR, exist_ok=True)
    app = App()
    app.mainloop()
