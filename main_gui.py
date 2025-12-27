import os
import sys
import time
import threading
import queue
import json
import cv2
import numpy as np
import tkinter as tk
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

# status callback populated when the UI status panel is created
STATUS_CALLBACK = None
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
from tkinter import simpledialog

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
        self.global_wcs_info = None
        self.summary_video_config = [
            {'name': "Composite Image", 'enabled': True, 'duration': 1.0},
            {'name': "Annotated Image", 'enabled': True, 'duration': 2.0},
            {'name': "Zoom Sequence", 'enabled': True, 'duration': 2.0},
            {'name': "Cutout Video", 'enabled': True},
            {'name': "Full Size Video", 'enabled': False}
        ]
        self.settings_file = "app_settings.json"
        self.masks_file = "app_masks.npz"

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
        
        style.configure("TNotebook", background=BG_COLOR, borderwidth=0)
        style.configure("TNotebook.Tab", background=BG_COLOR, foreground=FG_COLOR, padding=[10, 5], font=('Segoe UI', 10))
        style.map("TNotebook.Tab", background=[("selected", SELECT_BG)], foreground=[("selected", "white")])
        
        style.configure("TLabelframe", background=FRAME_BG, bordercolor=SELECT_BG, padding=10)
        style.configure("TLabelframe.Label", font=('Segoe UI', 11, 'bold'), background=FRAME_BG, foreground=FG_COLOR)

        style.configure("TEntry", fieldbackground="#3A4D6B", foreground=FG_COLOR, insertcolor=FG_COLOR, bordercolor=SELECT_BG)
        style.configure("TSpinbox", fieldbackground="#3A4D6B", foreground=FG_COLOR, insertcolor=FG_COLOR, bordercolor=SELECT_BG)
        
        style.configure("Vertical.TScrollbar", background=BG_COLOR, troughcolor=FRAME_BG, bordercolor=BG_COLOR, arrowcolor=FG_COLOR)
        style.map("Vertical.TScrollbar", background=[('active', SELECT_BG)])

        style.configure("Horizontal.TProgressbar", background=SELECT_BG)

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
        # Auto time updater
        self.auto_time_updater_enabled_var = tk.BooleanVar(value=False)
        self.auto_updater = auto_time_updater.AutoTimeUpdater()
        self.auto_updater.set_update_callback(self._on_auto_time_update)
        self.auto_updater.set_log_callback(self.append_log)
        # RTSP detection preset
        self.rtsp_preset_var = tk.StringVar(value="cloudy")  # "clear" or "cloudy"
        # RTSP time limit for recording (similar to periodic scan)
        self.rtsp_time_limit_var = tk.BooleanVar(value=False)
        self.rtsp_start_hour_var = tk.StringVar(value="17")
        self.rtsp_start_min_var = tk.StringVar(value="00")
        self.rtsp_end_hour_var = tk.StringVar(value="07")
        self.rtsp_end_min_var = tk.StringVar(value="00")
        # Plate solve mode: "local" (WSL solve-field) or "api" (Astrometry.net)
        self.plate_solve_mode_var = tk.StringVar(value="local")
        # Astrometry.net API key
        self.astrometry_api_key_var = tk.StringVar(value="")

    def setup_ui(self):
        main_pane = PanedWindow(self, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, bg="#2E3F5B")
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = ttk.Frame(main_pane, padding=10)
        
        right_frame = ttk.Frame(main_pane, padding=10)
        self.create_info_panel(right_frame)

        notebook = ttk.Notebook(left_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        tab_usage = self.create_usage_tab(notebook)
        tab_source = self.create_source_tab(notebook)
        tab_periodic = self.create_periodic_scan_tab(notebook)
        tab_settings = self.create_settings_tab(notebook)
        tab_analysis = self.create_analysis_tab(notebook)

        notebook.add(tab_usage, text="使い方")
        notebook.add(tab_source, text="ソース選択")
        notebook.add(tab_periodic, text="定期スキャン")
        notebook.add(tab_settings, text="各種設定")
        notebook.add(tab_analysis, text="解析")
        
        main_pane.add(left_frame, width=550)
        main_pane.add(right_frame)


    def create_usage_tab(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # スクロール可能なキャンバスとスクロールバーを作成（各種設定タブと同じ方式）
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
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        # --- コンテンツの追加 ---
        pad_x = 10
        pad_y = 5
        
        # タイトル
        title_lbl = ttk.Label(scrollable_frame, text="✨ 流星検出アプリの使い方", font=("Arial", 16, "bold"), foreground="#87CEEB")
        title_lbl.pack(pady=(15, 10), padx=pad_x, anchor="w")
        
        intro_text = "このアプリは、動画ファイルやRTSPストリームから流星を自動検出し、\n解析・記録するためのツールです。以下の手順に従って操作してください。"
        ttk.Label(scrollable_frame, text=intro_text, justify=tk.LEFT).pack(padx=pad_x, pady=(0, 15), anchor="w")

        # Step 1: ソースの追加
        lf_step1 = ttk.LabelFrame(scrollable_frame, text="Step 1: データの準備 📂")
        lf_step1.pack(fill=tk.X, padx=pad_x, pady=pad_y)
        
        s1_text = """「ソース選択」タブで解析対象を指定します。

1. 動画ファイルの場合:
   - フォルダまたはファイルをリストにドラッグ＆ドロップするか、
     [フォルダ追加] / [ファイル追加] ボタンを使用してください。

2. RTSPストリーム（ライブカメラ）の場合:
   - RTSP URLを入力し、[追加] ボタンを押してください。
   - ※ 外部GPUがない場合、CPU負荷にご注意ください。"""
        ttk.Label(lf_step1, text=s1_text, justify=tk.LEFT).pack(padx=10, pady=10, anchor="w")

        # Step 2: 設定
        lf_step2 = ttk.LabelFrame(scrollable_frame, text="Step 2: 検出設定 ⚙️")
        lf_step2.pack(fill=tk.X, padx=pad_x, pady=pad_y)
        
        s2_text = """「各種設定」タブで検出の感度や保存オプションを設定します。
デフォルト設定のままでも使用可能です。

- API Key: プレートソルブを使用する場合はAstrometry.netのキーを設定してください。
- 保存オプション: 検出時の保存データ（動画、画像、CSV等）を選択します。"""
        ttk.Label(lf_step2, text=s2_text, justify=tk.LEFT).pack(padx=10, pady=10, anchor="w")

        # Step 3: 実行
        lf_step3 = ttk.LabelFrame(scrollable_frame, text="Step 3: 解析開始 ▶️")
        lf_step3.pack(fill=tk.X, padx=pad_x, pady=pad_y)
        
        s3_text = """画面右下の [開始] ボタンをクリックすると解析が始まります。

- 状況バー: 現在の処理キューの状態や進行状況が表示されます。
- キャンセル: 途中で停止したい場合は [キャンセル] ボタンを押してください。"""
        ttk.Label(lf_step3, text=s3_text, justify=tk.LEFT).pack(padx=10, pady=10, anchor="w")

        # Step 4: 結果の確認
        lf_step4 = ttk.LabelFrame(scrollable_frame, text="Step 4: 結果の確認 📊")
        lf_step4.pack(fill=tk.X, padx=pad_x, pady=pad_y)
        
        s4_text = """検出された流星は以下の場所に保存されます。

- 保存先: デフォルトでは `meteor_save_path` に保存されます。
  （設定タブで変更可能）
- ログ: 「処理状況」パネルの [ログ] タブで詳細を確認できます。
  [ログを保存] ボタンでテキストファイルに出力も可能です。"""
        ttk.Label(lf_step4, text=s4_text, justify=tk.LEFT).pack(padx=10, pady=10, anchor="w")

        return frame

    def create_source_tab(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        lf_folder = ttk.LabelFrame(frame, text="フォルダ / 動画ファイル")
        lf_folder.pack(fill=tk.X, expand=True, pady=5)
        
        drop_label = ttk.Label(lf_folder, text="ここにフォルダや動画ファイルをドラッグ＆ドロップ", relief=tk.SOLID, padding=20, anchor=tk.CENTER, borderwidth=1)
        drop_label.pack(fill=tk.X, pady=5)
        drop_label.drop_target_register(DND_FILES)
        drop_label.dnd_bind('<<Drop>>', self.drop)

        list_frame = ttk.Frame(lf_folder)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.listbox_folders = tk.Listbox(list_frame, selectmode=tk.EXTENDED, height=6, bg="#3A4D6B", fg="#EAEAEA", relief=tk.FLAT, highlightthickness=0)
        self.listbox_folders.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.listbox_folders.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox_folders.config(yscrollcommand=scrollbar.set)

        btn_frame = ttk.Frame(lf_folder)
        btn_frame.pack(fill=tk.X, pady=(5,0))
        ttk.Button(btn_frame, text="選択項目を削除", command=self.remove_selected_folders).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="すべて削除", command=self.remove_all_folders).pack(side=tk.LEFT, padx=2)

        lf_rtsp = ttk.LabelFrame(frame)
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
        ttk.Entry(entry_frame, textvariable=self.rtsp_url_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(entry_frame, text="追加", command=self.add_rtsp_url).pack(side=tk.LEFT, padx=(5,0))
        
        rtsp_list_frame = ttk.Frame(lf_rtsp)
        rtsp_list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.listbox_rtsp = tk.Listbox(rtsp_list_frame, selectmode=tk.EXTENDED, height=3, bg="#3A4D6B", fg="#EAEAEA", relief=tk.FLAT, highlightthickness=0)
        self.listbox_rtsp.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        rtsp_scrollbar = ttk.Scrollbar(rtsp_list_frame, orient=tk.VERTICAL, command=self.listbox_rtsp.yview)
        rtsp_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox_rtsp.config(yscrollcommand=rtsp_scrollbar.set)
        
        rtsp_btn_frame = ttk.Frame(lf_rtsp)
        rtsp_btn_frame.pack(fill=tk.X, pady=(5,0))
        ttk.Button(rtsp_btn_frame, text="選択項目を削除", command=self.remove_selected_rtsp).pack(side=tk.LEFT, padx=2)
        ttk.Button(rtsp_btn_frame, text="すべて削除", command=self.remove_all_rtsp).pack(side=tk.LEFT, padx=2)
        ttk.Button(rtsp_btn_frame, text="RTSPからプレートソルブ", command=self.start_rtsp_plate_solve).pack(side=tk.LEFT, padx=(10, 2))
        ttk.Button(rtsp_btn_frame, text="RTSPからマスク作成", command=self.create_rtsp_mask).pack(side=tk.LEFT, padx=2)
        
        # RTSP録画時間制限
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
        
        return frame

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

        list_frame = ttk.Frame(lf)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.listbox_analysis = tk.Listbox(list_frame, selectmode=tk.EXTENDED, height=6, bg="#3A4D6B", fg="#EAEAEA", relief=tk.FLAT, highlightthickness=0)
        self.listbox_analysis.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.listbox_analysis.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox_analysis.config(yscrollcommand=scrollbar.set)

        btn_frame = ttk.Frame(lf)
        btn_frame.pack(fill=tk.X, pady=(5,0))
        ttk.Button(btn_frame, text="選択項目を削除", command=self.remove_selected_analysis).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="すべて削除", command=self.remove_all_analysis).pack(side=tk.LEFT, padx=2)

        # start analysis button
        action_frame = ttk.Frame(frame)
        action_frame.pack(fill=tk.X, pady=8)
        
        # Row 1
        row1 = ttk.Frame(action_frame)
        row1.pack(fill=tk.X, pady=2)
        ttk.Button(row1, text="解析開始", command=self.start_analysis).pack(side=tk.LEFT, padx=(0,5))
        ttk.Button(row1, text="座標点を追加", command=self.add_custom_point).pack(side=tk.LEFT, padx=(0,5))
        ttk.Button(row1, text="座標点を管理", command=self.manage_coordinates).pack(side=tk.LEFT, padx=(0,5))

        # Row 2
        row2 = ttk.Frame(action_frame)
        row2.pack(fill=tk.X, pady=2)
        ttk.Button(row2, text="長時間輝線マップを作成", command=self.create_long_exposure_map_callback).pack(side=tk.LEFT, padx=(0,5))
        ttk.Button(row2, text="ゆがみ補正", command=self.apply_distortion_correction_callback).pack(side=tk.LEFT, padx=(0,5))
        ttk.Button(row2, text="角度分布分析", command=self.analyze_angles_callback).pack(side=tk.LEFT, padx=(0,5))

        # Row 3
        row3 = ttk.Frame(action_frame)
        row3.pack(fill=tk.X, pady=2)
        ttk.Button(row3, text="比較明合成画像を作成", command=self.create_lighten_blend_image_callback).pack(side=tk.LEFT, padx=(0,5))
        ttk.Button(row3, text="比較明合成動画を作成", command=self.create_lighten_blend_video_callback).pack(side=tk.LEFT, padx=(0,5))
        ttk.Button(row3, text="タイムラプス作成", command=self.create_timelapse_callback).pack(side=tk.LEFT, padx=(0,5))

        return frame

    def drop_analysis(self, event):
        paths = self.splitlist(event.data)
        added = False
        for p in paths:
            p = p.strip('{}')
            if os.path.isfile(p) and Path(p).suffix.lower() in ['.txt']:
                if p not in self.analysis_files:
                    self.analysis_files.append(p)
                    self.listbox_analysis.insert(tk.END, p)
                    added = True

        if not added:
            messagebox.showwarning("情報", "有効な .txt ファイルがドロップされませんでしたか、既に追加済みです。")

    def remove_selected_analysis(self):
        selected = self.listbox_analysis.curselection()
        if not selected: return
        for idx in reversed(selected):
            if 0 <= idx < len(self.analysis_files):
                del self.analysis_files[idx]
                self.listbox_analysis.delete(idx)

    def remove_all_analysis(self):
        if not self.analysis_files: return
        if messagebox.askyesno("確認", "リストからすべての解析ファイルを削除しますか？"):
            self.analysis_files.clear()
            self.listbox_analysis.delete(0, tk.END)

    def start_analysis(self):
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
        def on_add(name: str, ra: float, dec: float):
            self.coord_manager.add_point(name, ra, dec)
        
        dialog = coord_mgr.CoordinateDialog(self, on_add)
        dialog.show()
    
    def manage_coordinates(self):
        """Show dialog to manage coordinate points."""
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

    def create_periodic_scan_tab(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        lf = ttk.LabelFrame(frame, text="定期スキャン設定")
        lf.pack(fill=tk.X, expand=True, pady=5)

        ttk.Checkbutton(lf, text="定期スキャンを有効にする", variable=self.periodic_scan_var, command=self.update_start_button_state).pack(anchor=tk.W)
        
        dir_frame = ttk.Frame(lf)
        dir_frame.pack(fill=tk.X, pady=5)
        ttk.Label(dir_frame, text="監視フォルダ:").pack(side=tk.LEFT, padx=(0,5))
        ttk.Entry(dir_frame, textvariable=self.periodic_dir_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(dir_frame, text="選択", command=self.select_periodic_dir).pack(side=tk.LEFT, padx=(5,0))
        
        interval_frame = ttk.Frame(lf)
        interval_frame.pack(fill=tk.X, pady=5)
        ttk.Label(interval_frame, text="スキャン間隔 (秒):").pack(side=tk.LEFT)
        ttk.Entry(interval_frame, textvariable=self.periodic_interval_var, width=5).pack(side=tk.LEFT)

        lf_time = ttk.LabelFrame(frame, text="時間制限")
        lf_time.pack(fill=tk.X, expand=True, pady=5)
        
        row_frame = ttk.Frame(lf_time)
        row_frame.pack(fill=tk.X)
        self.chk_time_limit = ttk.Checkbutton(row_frame, text="時間制限を有効にする", variable=self.periodic_time_limit_var, command=self.toggle_time_limit_frame)
        self.chk_time_limit.pack(side=tk.LEFT, anchor=tk.W)
        ttk.Button(row_frame, text="自動で設定", command=self.fetch_current_location).pack(side=tk.LEFT, padx=(8,0))
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

    def create_settings_tab(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

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
        canvas.bind_all("<MouseWheel>", on_mousewheel)

        # 以下の内容はscrollable_frameに配置
        lf_params = ttk.LabelFrame(scrollable_frame, text="処理パラメータ")
        lf_params.pack(fill=tk.X, pady=5)
        p_frame1 = ttk.Frame(lf_params); p_frame1.pack(fill=tk.X, pady=2)
        ttk.Label(p_frame1, text="同時処理数:", width=20).pack(side=tk.LEFT)
        ttk.Spinbox(p_frame1, from_=1, to=os.cpu_count() or 1, width=5, textvariable=self.concurrency_var).pack(side=tk.LEFT)
        p_frame2 = ttk.Frame(lf_params); p_frame2.pack(fill=tk.X, pady=2)
        ttk.Label(p_frame2, text="差分作成間隔 (秒):", width=20).pack(side=tk.LEFT)
        ttk.Spinbox(p_frame2, from_=1, to=60, width=5, textvariable=self.interval_var).pack(side=tk.LEFT)
        p_frame3 = ttk.Frame(lf_params); p_frame3.pack(fill=tk.X, pady=2)
        ttk.Label(p_frame3, text="差分作成期間 (秒):", width=20).pack(side=tk.LEFT)
        ttk.Spinbox(p_frame3, from_=1, to=60, width=5, textvariable=self.duration_var).pack(side=tk.LEFT)
                # RTSP検出プリセット選択
        p_frame4 = ttk.Frame(lf_params); p_frame4.pack(fill=tk.X, pady=2)
        ttk.Label(p_frame4, text="RTSP検出感度:", width=20).pack(side=tk.LEFT)
        ttk.Radiobutton(p_frame4, text="雲が少ないとき（高感度）", variable=self.rtsp_preset_var, value="clear").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(p_frame4, text="雲が多いとき（ノイズ対策）", variable=self.rtsp_preset_var, value="cloudy").pack(side=tk.LEFT, padx=5)

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

        lf_astro = ttk.LabelFrame(scrollable_frame, text="プレートソルブ & マスク")
        lf_astro.pack(fill=tk.X, pady=5)
        
        ps_frame = ttk.Frame(lf_astro); ps_frame.pack(fill=tk.X, pady=2)
        ttk.Label(ps_frame, text="動画から実行:").pack(side=tk.LEFT, padx=(0,5))
        ttk.Entry(ps_frame, textvariable=self.plate_solve_video_path_var, state='readonly').pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(ps_frame, text="選択", command=self.select_plate_solve_video).pack(side=tk.LEFT, padx=(5,0))
        ttk.Button(ps_frame, text="実行", command=self.start_plate_solve).pack(side=tk.LEFT, padx=(5,0))
        
        ps_wcs_frame = ttk.Frame(lf_astro); ps_wcs_frame.pack(fill=tk.X, pady=2)
        ttk.Label(ps_wcs_frame, text="既存WCSファイル:").pack(side=tk.LEFT, padx=(0,5))
        ttk.Entry(ps_wcs_frame, textvariable=self.plate_solve_wcs_path_var, state='readonly').pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(ps_wcs_frame, text="選択", command=self.select_plate_solve_wcs_file).pack(side=tk.LEFT, padx=(5,0))

        # Plate solve mode selection (local vs API)
        ps_mode_frame = ttk.Frame(lf_astro); ps_mode_frame.pack(fill=tk.X, pady=2)
        ttk.Label(ps_mode_frame, text="ソルバー:").pack(side=tk.LEFT, padx=(0,5))
        ttk.Radiobutton(ps_mode_frame, text="ローカル (WSL)", variable=self.plate_solve_mode_var, value="local").pack(side=tk.LEFT, padx=5)
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
        
        api_key_entry = ttk.Entry(api_key_frame, textvariable=self.astrometry_api_key_var, show="*", width=30)
        api_key_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
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
        ttk.Button(mask_btn_frame, text="検出マスク作成", command=lambda: self.create_mask_window(is_plate_solve_mask=False)).pack(side=tk.LEFT)
        ttk.Button(mask_btn_frame, text="💾", width=3, command=self.download_mask).pack(side=tk.LEFT, padx=2)
        ttk.Button(mask_btn_frame, text="プレートソルブ用マスク作成", command=lambda: self.create_mask_window(is_plate_solve_mask=True)).pack(side=tk.LEFT, padx=5)
        
        self.mask_preview_frame = ttk.Frame(lf_astro); self.mask_preview_frame.pack(pady=5)
        self.mask_preview_label = ttk.Label(self.mask_preview_frame, text="検出マスクなし")
        self.mask_preview_label.pack(side=tk.LEFT, padx=10)
        self.ps_mask_preview_label = ttk.Label(self.mask_preview_frame, text="PSマスクなし")
        self.ps_mask_preview_label.pack(side=tk.LEFT, padx=10)

        ttk.Checkbutton(lf_astro, text="検出マスクを適用する", variable=self.apply_mask_var).pack(anchor=tk.W)
        
        return frame

    def create_info_panel(self, parent):
        # Create a combined Log / Processing Status panel from the status_panel module.
        panel = status_panel.StatusPanel(parent, progress_queue=self.progress_queue, app=self)
        panel.pack(fill=tk.BOTH, expand=True, pady=5)

        # reuse the panel's log_text so existing append_log continues to work
        self.log_text = panel.log_text

        # keep references for existing logic (progress bar, labels, buttons)
        # create a compact status row under the panel (progressbar, ETA, buttons)
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

        # expose the panel's status callback to the module-level worker loop
        try:
            global STATUS_CALLBACK
            STATUS_CALLBACK = panel.get_status_callback()
        except Exception:
            STATUS_CALLBACK = None

    def append_log(self, message: str):
        if not self.log_text.winfo_exists(): return
        self.log_text.config(state='normal')
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

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
        folders_to_add = set()
        for path in paths:
            target_folder = None
            if os.path.isdir(path):
                target_folder = path
            elif os.path.isfile(path) and Path(path).suffix.lower() in config.PERIODIC_VIDEO_EXTENSIONS:
                target_folder = str(Path(path).parent)

            if target_folder and target_folder not in self.folder_paths:
                folders_to_add.add(target_folder)
        
        if folders_to_add:
            for folder in sorted(list(folders_to_add)):
                self.folder_paths.append(folder)
                self.listbox_folders.insert(tk.END, folder)
            self.update_start_button_state()
        else:
            messagebox.showwarning("情報", "有効なフォルダまたは動画ファイルがドロップされませんでした。")

    def remove_selected_folders(self):
        selected_indices = self.listbox_folders.curselection()
        if not selected_indices: return
        for index in reversed(selected_indices):
            if 0 <= index < len(self.folder_paths):
                del self.folder_paths[index]
                self.listbox_folders.delete(index)
        self.update_start_button_state()

    def remove_all_folders(self):
        if not self.folder_paths: return
        if messagebox.askyesno("確認", "リストからすべてのフォルダを削除しますか？"):
            self.folder_paths.clear()
            self.listbox_folders.delete(0, tk.END)
            self.update_start_button_state()

    def add_rtsp_url(self):
        url = self.rtsp_url_var.get().strip()
        if url and url not in self.rtsp_urls:
            self.rtsp_urls.append(url)
            self.listbox_rtsp.insert(tk.END, url)
            self.rtsp_url_var.set("")
            self.update_start_button_state()
        elif not url:
            messagebox.showwarning("入力エラー", "RTSP URLを入力してください。")

    def remove_selected_rtsp(self):
        selected_indices = self.listbox_rtsp.curselection()
        if not selected_indices: return
        for index in reversed(selected_indices):
            if 0 <= index < len(self.rtsp_urls):
                del self.rtsp_urls[index]
                self.listbox_rtsp.delete(index)
        self.update_start_button_state()

    def remove_all_rtsp(self):
        if not self.rtsp_urls: return
        if messagebox.askyesno("確認", "すべてのRTSP URLを削除しますか？"):
            self.rtsp_urls.clear()
            self.listbox_rtsp.delete(0, tk.END)
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

        # update GUI vars on main thread
        try:
            self.after(0, lambda: self.current_lat_var.set(f"{lat:.6f}"))
            self.after(0, lambda: self.current_lon_var.set(f"{lon:.6f}"))
        except Exception:
            pass

        # print result as requested
        print(f"Current location: lat={lat}, lon={lon}")
        # also log into GUI log for convenience
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
                # update GUI spinboxes on main thread
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
        threading.Thread(target=self.execute_plate_solve_thread, daemon=True).start()

    def start_rtsp_plate_solve(self):
        """RTSPストリームからプレートソルブを実行する"""
        # 選択されているRTSP URLを取得、選択がなければ最初のURLを使用
        selected = self.listbox_rtsp.curselection()
        if selected:
            rtsp_url = self.rtsp_urls[selected[0]]
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
            # RTSPストリームのタイムアウト設定
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
            
            # 比較明合成画像を作成
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
            self.save_settings()
            # 自動更新を停止
            if self.auto_updater:
                self.auto_updater.stop()
            self.cancel_flag.set()
            self.destroy()

    def start_processing(self):
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
            
            # RTSP検出プリセットを適用
            if self.rtsp_preset_var.get() == "clear":
                preset = config.RTSP_PRESET_CLEAR_SKY
            else:
                preset = config.RTSP_PRESET_CLOUDY
            config.RTSP_MIN_LINE_LENGTH = preset['min_line_length']
            config.RTSP_HOUGH_THRESHOLD = preset['hough_threshold']
            config.RTSP_CANNY_THRESH1 = preset['canny_thresh1']
            config.RTSP_CANNY_THRESH2 = preset['canny_thresh2']
            self.append_log(f"RTSP検出プリセット: {preset['name']}")
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
            
            log_msg = f"RTSP処理開始 (URL: {url})"
            if rtsp_time_limit:
                log_msg += f", 録画時間制限: {rtsp_sh:02d}:{rtsp_sm:02d} - {rtsp_eh:02d}:{rtsp_em:02d}"
            self.append_log(log_msg)
            
            rtsp_args = (
                url, config.RTSP_SAVE_ROOT, config.RTSP_SEGMENT_DURATION, 60, self.progress_queue.put,
                params['mask'], params['global_wcs_info'], params['plate_solve_mask'],
                params['meteor_save_path'], params['not_meteor_save_path'], self.cancel_flag,
                params['save_options'], params['interval_sec'], params['duration_sec'],
                config.MIN_LINE_LENGTH, params['summary_config'],
                rtsp_time_limit, rtsp_sh, rtsp_sm, rtsp_eh, rtsp_em
            )
            self.rtsp_thread = threading.Thread(target=file_utils.rtsp_save_and_process_thread_target, args=rtsp_args, daemon=True)
            self.rtsp_thread.start()

        elif self.folder_paths:
            sources_to_process = []
            self.append_log(f"{len(self.folder_paths)}個のフォルダを処理します...")
            for folder in self.folder_paths:
                found = sorted([p for p in Path(folder).rglob('*') if p.suffix.lower() in config.PERIODIC_VIDEO_EXTENSIONS])
                sources_to_process.extend([{'path': str(p), 'is_rtsp': False} for p in found])
            
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
        # Immediately set cancel flag so background workers can observe it without
        # relying on a blocking confirmation dialog. This avoids situations where
        # the GUI becomes unresponsive to cancel clicks (confirmation dialogs
        # can cause subtle focus/state issues). We still log the request and
        # disable the cancel button to prevent duplicate clicks.
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

        # stop ETA/timers
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
        listbox = tk.Listbox(list_frame, selectmode=tk.SINGLE, exportselection=False)
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
        """RTSPストリームから検出マスクを作成する"""
        # 選択されているRTSP URLを取得、選択がなければ最初のURLを使用
        selected = self.listbox_rtsp.curselection()
        if selected:
            rtsp_url = self.rtsp_urls[selected[0]]
        elif self.rtsp_urls:
            rtsp_url = self.rtsp_urls[0]
        else:
            messagebox.showwarning("警告", "RTSPストリームを追加してください。")
            return
        
        # 接続中ダイアログを表示
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
        
        # Center the dialog
        progress_win.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - progress_win.winfo_width()) // 2
        y = self.winfo_y() + (self.winfo_height() - progress_win.winfo_height()) // 2
        progress_win.geometry(f"+{x}+{y}")
        
        # バックグラウンドでフレームを取得
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
                    self._open_rtsp_mask_window(result_holder['frame'])
        
        self.after(100, check_thread)
    
    def _open_rtsp_mask_window(self, frame):
        """RTSPから取得したフレームでマスク作成ウィンドウを開く"""
        
        # マスク作成ウィンドウを開く
        win = Toplevel(self)
        win.title("RTSPからマスク作成")
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
                self.mask_image = final_mask
                self.mask_path_var.set("作成済み (RTSP)")
                self.apply_mask_var.set(True)
                
                # プレビューを更新
                self.preview_mask(self.mask_image, self.mask_preview_label, "検出マスク")
                
                print(f"RTSPマスク作成完了: shape={final_mask.shape}, max={final_mask.max()}, min={final_mask.min()}")
                
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
            'has_mask_image': self.mask_image is not None, 'has_plate_solve_mask_image': self.plate_solve_mask_image is not None,
            'summary_video_config': self.summary_video_config,
            'auto_time_updater_enabled': self.auto_time_updater_enabled_var.get(),
            'rtsp_preset': self.rtsp_preset_var.get(),
            # RTSP time limit settings
            'rtsp_time_limit_enabled': self.rtsp_time_limit_var.get(),
            'rtsp_start_hour': self.rtsp_start_hour_var.get(), 'rtsp_start_minute': self.rtsp_start_min_var.get(),
            # Plate solve mode
            'plate_solve_mode': self.plate_solve_mode_var.get(),
            'rtsp_end_hour': self.rtsp_end_hour_var.get(), 'rtsp_end_minute': self.rtsp_end_min_var.get(),
            # Astrometry.net API key
            'astrometry_api_key': self.astrometry_api_key_var.get(),
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
            self.listbox_folders.delete(0, tk.END); [self.listbox_folders.insert(tk.END, p) for p in self.folder_paths]
            self.rtsp_urls = settings.get('rtsp_urls', [])
            self.listbox_rtsp.delete(0, tk.END); [self.listbox_rtsp.insert(tk.END, u) for u in self.rtsp_urls]

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

            self.append_log("前回の設定を復元しました。")
            self.update_start_button_state()
        except Exception as e:
            messagebox.showerror("エラー", f"設定の読み込み中にエラーが発生しました: {e}")

    def create_long_exposure_map_callback(self):
        """Callback for the 'Create Long Exposure Map' button."""
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

        # Define paths for distortion maps (assuming they are in the same directory as the script or specified fixed paths)
        # User specified: C:\Users\kekke\Desktop\my_app\div\distortion_map_x.npy
        # We can construct this relative to the current script location to be safer/more portable if needed,
        # but user gave specific paths. Let's use the div folder path.
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

    def analyze_angles_callback(self):
        """Callback for the 'Angle Distribution Analysis' button."""
        if not self.analysis_files:
            messagebox.showwarning("情報", "解析するファイルを追加してください。")
            return

        # Ask for Radiant Coordinates
        # Using a simple dialog sequence for now. Could be improved with a custom dialog.
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
        # デフォルトで流星の保存先フォルダを開く
        initial_dir = self.meteor_save_path_var.get()
        if not initial_dir or not os.path.exists(initial_dir):
            initial_dir = os.path.expanduser("~")
        
        # ファイル選択ダイアログで複数の動画ファイルを選択
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
        
        # デフォルトの保存パスを取得
        default_output = lighten_blend_video.get_default_output_path()
        
        # ユーザーに保存先を確認
        output_path = filedialog.asksaveasfilename(
            title="比較明合成動画の保存先",
            initialdir=os.path.dirname(default_output),
            initialfile=os.path.basename(default_output),
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4"), ("AVI Video", "*.avi"), ("All Files", "*")]
        )
        
        if not output_path:
            return
        
        self.append_log(f"比較明合成動画の作成を開始します... ({len(video_files)}個の動画)")
        
        def run_task():
            success = lighten_blend_video.create_lighten_blend_video(
                list(video_files),
                output_path,
                progress_callback=self.append_log
            )
            if success:
                messagebox.showinfo("完了", f"比較明合成動画の作成が完了しました。\n保存先: {output_path}")
                self.append_log(f"比較明合成動画の作成が完了しました: {output_path}")
            else:
                messagebox.showerror("エラー", "比較明合成動画の作成に失敗しました。ログを確認してください。")
                self.append_log("比較明合成動画の作成に失敗しました。")
        
        threading.Thread(target=run_task, daemon=True).start()

    def create_lighten_blend_image_callback(self):
        """Callback for the 'Create Lighten Blend Image' button."""
        # デフォルトで流星の保存先フォルダを開く
        initial_dir = self.meteor_save_path_var.get()
        if not initial_dir or not os.path.exists(initial_dir):
            initial_dir = os.path.expanduser("~")
        
        # ファイル選択ダイアログで複数の画像/動画ファイルを選択
        file_paths = filedialog.askopenfilenames(
            title="比較明合成する画像・動画ファイルを選択（複数可）",
            initialdir=initial_dir,
            filetypes=[
                ("画像・動画ファイル", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.mp4 *.avi *.mov *.mkv *.wmv"),
                ("画像ファイル", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("動画ファイル", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("すべてのファイル", "*.*")
            ]
        )
        
        if not file_paths:
            return
        
        if len(file_paths) < 2:
            messagebox.showwarning("警告", "比較明合成を行うには2つ以上のファイルを選択してください。")
            return
        
        # デフォルトの保存パスを取得（流星保存先フォルダを使用）
        default_output = lighten_blend_image.get_default_output_path(initial_dir)
        
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
        
        self.append_log(f"比較明合成画像の作成を開始します... ({len(file_paths)}個のファイル)")
        
        def run_task():
            success = lighten_blend_image.create_lighten_blend_image(
                list(file_paths),
                output_path,
                progress_callback=self.append_log
            )
            if success:
                messagebox.showinfo("完了", f"比較明合成画像の作成が完了しました。\n保存先: {output_path}")
                self.append_log(f"比較明合成画像の作成が完了しました: {output_path}")
            else:
                messagebox.showerror("エラー", "比較明合成画像の作成に失敗しました。ログを確認してください。")
                self.append_log("比較明合成画像の作成に失敗しました。")
        
        threading.Thread(target=run_task, daemon=True).start()

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
        
        # ウィンドウをモーダルに設定
        self.transient(parent)
        self.grab_set()
    
    def setup_ui(self):
        # メインフレーム
        main_frame = ttk.Frame(self, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # ドロップエリア
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
        
        # ドラッグ＆ドロップを登録
        self.drop_label.drop_target_register(DND_FILES)
        self.drop_label.dnd_bind('<<Drop>>', self.on_drop)
        
        # ドロップしたファイルのリスト表示
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
        
        # クリアボタン
        ttk.Button(drop_frame, text="リストをクリア", command=self.clear_list).pack(anchor=tk.E, pady=(5, 0))
        
        # 動画長さ選択
        duration_frame = ttk.LabelFrame(main_frame, text="動画の長さ", padding=10)
        duration_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.duration_var = tk.IntVar(value=30)
        
        duration_options = ttk.Frame(duration_frame)
        duration_options.pack()
        
        ttk.Radiobutton(duration_options, text="15秒", variable=self.duration_var, value=15).pack(side=tk.LEFT, padx=15)
        ttk.Radiobutton(duration_options, text="30秒", variable=self.duration_var, value=30).pack(side=tk.LEFT, padx=15)
        ttk.Radiobutton(duration_options, text="60秒", variable=self.duration_var, value=60).pack(side=tk.LEFT, padx=15)
        
        # マスクフレーム
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
        
        # 開始ボタン
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
        
        # マスク作成ウィンドウを開く
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
