from gui_common import *
from gui_navigation import NavigationMixin
from gui_usage import UsageMixin
from gui_source import SourceMixin
from gui_analysis import AnalysisMixin
from gui_settings import SettingsMixin
from gui_advanced import AdvancedSettingsMixin
from gui_preview import PreviewMixin
from gui_plate_solve import PlateSolveMixin
from gui_processing import ProcessingMixin
from gui_masks import MaskMixin
from gui_tools import ToolsMixin
from gui_synthesis import SynthesisMixin
from gui_live_preview import LivePreviewMixin
from gui_camera_control import CameraControlMixin


class App(
    NavigationMixin, UsageMixin, SourceMixin, AnalysisMixin, SettingsMixin, AdvancedSettingsMixin,
    PreviewMixin, PlateSolveMixin, ProcessingMixin, MaskMixin, ToolsMixin, SynthesisMixin,
    LivePreviewMixin, CameraControlMixin,
    TkinterDnD.Tk,
):
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
        self._init_live_preview_state()
        self._init_camera_control_state()

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
        self.ai_vlm_backend_var = tk.StringVar(value=getattr(config, "DEFAULT_AI_VLM_BACKEND", "local_qwen3_vl_4b"))
        self.lm_studio_vlm_url_var = tk.StringVar(value=getattr(config, "DEFAULT_LM_STUDIO_VLM_URL", "http://localhost:1234/v1"))
        self.lm_studio_vlm_model_var = tk.StringVar(value=getattr(config, "DEFAULT_LM_STUDIO_VLM_MODEL_ID", "qwen3.5-2b"))
        self.lm_studio_vlm_api_key_var = tk.StringVar(value="")
        self.ai_vlm_status_var = tk.StringVar(value="")
        self._last_ai_vlm_backend = self.ai_vlm_backend_var.get()
        self._last_lm_studio_vlm_model = self.lm_studio_vlm_model_var.get()
        self._ai_vlm_operation_lock = threading.Lock()
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
        self.camera_control_base_url_var = tk.StringVar(value="")

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

    def on_closing(self):
        if messagebox.askokcancel("終了", "アプリケーションを終了しますか？"):
            self.append_log("設定を保存しています...")
            self._hide_summary_preview()
            self.close_rtsp_live_preview()
            self.close_camera_control()
            self.save_settings()
            # 自動更新を停止
            if self.auto_updater:
                self.auto_updater.stop()
            self.cancel_flag.set()
            self.destroy()


if __name__ == "__main__":
    os.makedirs(config.TEMP_CLIP_DIR, exist_ok=True)
    app = App()
    app.mainloop()
