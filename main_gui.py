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
        self.geometry("1480x900")
        self.minsize(1180, 720)
        
        self.setup_icon()
        self.setup_style()

        self.folder_paths = []
        self.rtsp_urls = []
        self.mask_image = None
        self.plate_solve_mask_image = None
        self.selfcal_mask_image = None
        self.rtsp_dark_frame = None
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
        self.rtsp_dark_file = os.path.join(base_path, "rtsp_dark_frame.npz")
        self.rtsp_dark_preview_file = os.path.join(base_path, "rtsp_dark_preview.jpg")
        self.noise_twin_model_dir = os.path.join(base_path, "noise_twin_models")
        os.makedirs(self.noise_twin_model_dir, exist_ok=True)
        self._migrate_legacy_settings_files()
        self._clear_lighten_blend_cache()

        self.worker_thread = None
        self.video_concat_thread = None
        self.video_concat_process = None
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

    def _clear_lighten_blend_cache(self):
        """Clear stale lighten-blend composites on app startup."""
        try:
            shutil.rmtree(config.LIGHTEN_BLEND_CACHE_DIR, ignore_errors=True)
            os.makedirs(config.LIGHTEN_BLEND_CACHE_DIR, exist_ok=True)
        except Exception as e:
            print(f"比較明合成キャッシュの初期化中にエラーが発生しました: {e}")

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

            if sys.platform == "darwin":
                jpg_icon_path = os.path.join(base_path, "icon.jpg")
                if os.path.exists(jpg_icon_path):
                    img = Image.open(jpg_icon_path)
                    img.thumbnail((256, 256))
                    self._app_icon_image = ImageTk.PhotoImage(img)
                    self.iconphoto(True, self._app_icon_image)
            elif os.path.exists(icon_path):
                self.iconbitmap(icon_path)
            else:
                print(f"警告: アイコンファイルが見つかりません: {icon_path}")
        except Exception as e:
            print(f"アイコンの設定中にエラーが発生しました: {e}")

    def setup_style(self):
        ui_theme.install_named_fonts(self)
        ui_theme.configure_macos_window(self)

        style = ttk.Style(self)
        style.theme_use('clam')

        c = ui_theme.COLORS
        self.configure(background=c["window"])
        self.option_add("*selectBackground", c["selection"])
        self.option_add("*selectForeground", c["text"])
        self.option_add("*insertBackground", c["text"])

        style.configure(".", background=c["content"], foreground=c["text"])
        style.configure("TFrame", background=c["content_raised"])
        style.configure("Content.TFrame", background=c["content"])
        style.configure("Glass.TFrame", background=c["glass"])
        style.configure("GlassStrong.TFrame", background=c["glass_strong"])
        style.configure("TLabel", background=c["content_raised"], foreground=c["text"])
        style.configure("Glass.TLabel", background=c["glass"], foreground=c["text"])
        style.configure("GlassMuted.TLabel", background=c["glass"], foreground=c["text_secondary"])
        style.configure(
            "Eyebrow.TLabel",
            background=c["content"],
            foreground=c["accent"],
            font=("SF Pro Text", 9, "bold"),
        )
        style.configure(
            "PageTitle.TLabel",
            background=c["content"],
            foreground=c["text"],
            font=("SF Pro Display", 24, "bold"),
        )
        style.configure(
            "PageSubtitle.TLabel",
            background=c["content"],
            foreground=c["text_secondary"],
            font=("SF Pro Text", 11),
        )
        style.configure(
            "DropZone.TLabel",
            background=c["field"],
            foreground=c["text_secondary"],
            bordercolor=c["border_bright"],
            lightcolor=c["border_bright"],
            darkcolor=c["border_bright"],
            borderwidth=1,
            relief=tk.SOLID,
            padding=(18, 18),
            anchor=tk.CENTER,
            font=("SF Pro Text", 11, "bold"),
        )

        button = {
            "foreground": c["text"],
            "borderwidth": 0,
            "padding": (12, 7),
            "font": ("SF Pro Text", 11, "bold"),
        }
        style.configure("TButton", background=c["glass_hover"], **button)
        style.map(
            "TButton",
            background=[
                ("disabled", c["glass_strong"]),
                ("pressed", c["glass_selected"]),
                ("active", c["border_bright"]),
            ],
            foreground=[("disabled", c["text_tertiary"])],
        )
        style.configure("Primary.TButton", background=c["accent_pressed"], **button)
        style.map(
            "Primary.TButton",
            background=[
                ("disabled", c["glass_strong"]),
                ("pressed", c["accent_pressed"]),
                ("active", c["accent_hover"]),
            ],
            foreground=[("disabled", c["text_tertiary"]), ("active", "#07101E")],
        )
        style.configure("Gray.TButton", background=c["glass_hover"], **button)
        style.map("Gray.TButton", background=[("active", c["border_bright"])])
        style.configure("Quiet.TButton", background=c["glass_strong"], **button)
        style.map("Quiet.TButton", background=[("active", c["glass_hover"])])
        style.configure("Toolbar.TButton", background=c["content"], **button)
        style.map("Toolbar.TButton", background=[("active", c["glass_hover"])])
        style.configure("Danger.TButton", background="#713243", **button)
        style.map("Danger.TButton", background=[("active", "#913E52")])

        style.configure("TNotebook", background=c["content"], borderwidth=0, tabmargins=0)
        style.configure(
            "TNotebook.Tab",
            background=c["glass_strong"],
            foreground=c["text_secondary"],
            padding=(14, 8),
            borderwidth=0,
            font=("SF Pro Text", 10, "bold"),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", c["glass_selected"]), ("active", c["glass_hover"])],
            foreground=[("selected", c["text"])],
        )
        style.configure(
            "Content.TNotebook",
            background=c["content"],
            bordercolor=c["content"],
            lightcolor=c["content"],
            darkcolor=c["content"],
            relief=tk.FLAT,
            borderwidth=0,
            tabmargins=0,
        )
        try:
            style.layout("Content.TNotebook.Tab", [])
        except tk.TclError:
            pass

        style.configure(
            "TLabelframe",
            background=c["content_raised"],
            bordercolor=c["border"],
            lightcolor=c["border"],
            darkcolor=c["border"],
            borderwidth=1,
            relief=tk.SOLID,
            padding=14,
        )
        style.configure(
            "TLabelframe.Label",
            background=c["content_raised"],
            foreground=c["text"],
            font=("SF Pro Text", 12, "bold"),
            padding=(2, 0, 8, 4),
        )
        style.configure(
            "Section.TLabelframe",
            background=c["content_raised"],
            bordercolor=c["border"],
            lightcolor=c["border"],
            darkcolor=c["border"],
            borderwidth=1,
            relief=tk.SOLID,
            padding=12,
        )
        style.configure(
            "Section.TLabelframe.Label",
            background=c["content_raised"],
            foreground=c["text"],
            font=("SF Pro Text", 11, "bold"),
        )
        style.configure(
            "Hint.TLabel",
            background=c["content_raised"],
            foreground=c["cyan"],
            font=("SF Pro Text", 10),
        )

        field_options = {
            "fieldbackground": c["field"],
            "foreground": c["text"],
            "insertcolor": c["text"],
            "bordercolor": c["border"],
            "lightcolor": c["border"],
            "darkcolor": c["border"],
            "padding": (8, 6),
        }
        style.configure("TEntry", **field_options)
        style.configure("TSpinbox", **field_options, arrowcolor=c["text_secondary"])
        style.configure(
            "TCombobox",
            **field_options,
            background=c["field"],
            arrowcolor=c["text_secondary"],
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", c["field"]), ("focus", c["field_focus"])],
            selectbackground=[("readonly", c["field"])],
            foreground=[("readonly", c["text"])],
            selectforeground=[("readonly", c["text"])],
        )

        style.configure(
            "TCheckbutton",
            background=c["content_raised"],
            foreground=c["text"],
            padding=(2, 4),
            font=("SF Pro Text", 11),
        )
        style.map(
            "TCheckbutton",
            background=[("active", c["content_raised"])],
            foreground=[("disabled", c["text_tertiary"])],
            indicatorcolor=[("selected", c["accent"]), ("!selected", c["field"])],
        )
        style.configure(
            "TRadiobutton",
            background=c["content_raised"],
            foreground=c["text"],
            padding=(2, 4),
            font=("SF Pro Text", 11),
        )
        style.map(
            "TRadiobutton",
            background=[("active", c["content_raised"])],
            foreground=[("disabled", c["text_tertiary"])],
            indicatorcolor=[("selected", c["accent"]), ("!selected", c["field"])],
        )

        style.configure(
            "Vertical.TScrollbar",
            background=c["glass_hover"],
            troughcolor=c["content"],
            bordercolor=c["content"],
            arrowcolor=c["text_secondary"],
            width=10,
        )
        style.map("Vertical.TScrollbar", background=[("active", c["border_bright"])])
        style.configure(
            "Horizontal.TProgressbar",
            background=c["accent"],
            troughcolor=c["field"],
            bordercolor=c["field"],
            lightcolor=c["accent"],
            darkcolor=c["accent"],
            thickness=7,
        )

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
        self.processing_source_priority = list(ui_state.SOURCE_PRIORITY_DEFAULT)
        self.save_options_vars = {
            k: tk.BooleanVar(value=v) for k, v in {
                'video': config.DEFAULT_SAVE_VIDEO_CLIP, 'cutout': config.DEFAULT_SAVE_CUTOUT_DIFF,
                'full': config.DEFAULT_SAVE_FULL_DIFF, 'composite': config.DEFAULT_SAVE_COMPOSITE,
                'info': config.DEFAULT_SAVE_DETECTION_INFO, 'summary': True,
                'full_video': config.DEFAULT_SAVE_FULL_VIDEO,
            }.items()
        }
        self.full_video_timestamp_enabled_var = tk.BooleanVar(
            value=config.FULL_VIDEO_TIMESTAMP_ENABLED
        )
        self.full_video_timestamp_position_var = tk.StringVar(
            value="右下"
        )
        self.full_video_timestamp_size_var = tk.StringVar(
            value=str(config.FULL_VIDEO_TIMESTAMP_SIZE_PERCENT)
        )
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
        self.ml_training_export_enabled_var = tk.BooleanVar(
            value=config.DEFAULT_ML_TRAINING_EXPORT_ENABLED
        )
        self.ml_training_data_root_var = tk.StringVar(
            value=config.DEFAULT_ML_TRAINING_DATA_ROOT
        )
        self.auto_video_mask_enabled_var = tk.BooleanVar(
            value=config.DEFAULT_AUTO_VIDEO_MASK_ENABLED
        )
        self.date_folder_twilight_filter_enabled_var = tk.BooleanVar(
            value=config.DEFAULT_DATE_FOLDER_TWILIGHT_FILTER_ENABLED
        )
        self.observation_latitude_var = tk.StringVar(
            value=str(config.DEFAULT_OBSERVATION_LATITUDE)
        )
        self.observation_longitude_var = tk.StringVar(
            value=str(config.DEFAULT_OBSERVATION_LONGITUDE)
        )
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
        self.rtsp_notification_sound_var = tk.BooleanVar(value=True)
        # RTSP time limit for recording (similar to periodic scan)
        self.rtsp_time_limit_var = tk.BooleanVar(value=False)
        self.rtsp_start_hour_var = tk.StringVar(value="17")
        self.rtsp_start_min_var = tk.StringVar(value="00")
        self.rtsp_end_hour_var = tk.StringVar(value="07")
        self.rtsp_end_min_var = tk.StringVar(value="00")
        self.apply_rtsp_dark_var = tk.BooleanVar(value=False)
        self.rtsp_dark_status_var = tk.StringVar(value="ダーク: 未撮影")
        self.rtsp_fixed_pattern_samples_var = tk.StringVar(
            value=str(config.RTSP_FIXED_PATTERN_DEFAULT_SAMPLES)
        )
        self.noise_twin_enabled_var = tk.BooleanVar(value=False)
        self.noise_twin_model_path_var = tk.StringVar(value="")
        self.noise_twin_status_var = tk.StringVar(value="NoiseTwin: 未選択")
        self.noise_twin_training_process = None
        self.temporal_mean_frames_var = tk.IntVar(value=0)
        self.processed_video_codec_var = tk.StringVar(value="H.265 / HEVC (推奨)")
        self.processed_video_quality_var = tk.StringVar(value="入力品質基準（推奨）")
        self.processed_video_bitrate_var = tk.StringVar(value="40")
        self.processed_video_encoding_info_var = tk.StringVar(value="RTSPの実ビットレートを測定して自動設定")
        self.plate_solve_mode_var = tk.StringVar(value="local")
        self.astrometry_api_key_var = tk.StringVar(value="")
        self.video_concat_files = []
        self.video_concat_bitrate_var = tk.StringVar(value="Auto")
        self.video_concat_codec_var = tk.StringVar(value="h264")
        self.video_concat_fps_var = tk.StringVar(value="Auto")
        self.video_concat_safe_mode_var = tk.BooleanVar(value=True) 
        self.video_concat_enhancement_var = tk.BooleanVar(value=False)
        self.video_concat_timestamp_enabled_var = tk.BooleanVar(value=config.VIDEO_CONCAT_TIMESTAMP_ENABLED)
        self.video_concat_timestamp_position_var = tk.StringVar(value="右下")
        self.video_concat_timestamp_size_var = tk.StringVar(value=str(config.VIDEO_CONCAT_TIMESTAMP_SIZE_PERCENT))
        self.video_concat_timestamp_offset_var = tk.StringVar(value=str(config.VIDEO_CONCAT_TIMESTAMP_OFFSET_SECONDS))
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
        c = ui_theme.COLORS
        shell = tk.Frame(self, bg=c["window"])
        shell.pack(fill=tk.BOTH, expand=True, padx=12, pady=(8, 12))

        main_pane = PanedWindow(
            shell,
            orient=tk.HORIZONTAL,
            sashrelief=tk.FLAT,
            sashwidth=8,
            showhandle=False,
            borderwidth=0,
            bg=c["window"],
            opaqueresize=True,
        )
        main_pane.pack(fill=tk.BOTH, expand=True)

        sidebar = tk.Frame(
            main_pane,
            bg=c["glass"],
            highlightbackground=c["border"],
            highlightcolor=c["border_bright"],
            highlightthickness=1,
        )
        content = ttk.Frame(main_pane, style="Content.TFrame", padding=(22, 14, 22, 18))
        right_frame = tk.Frame(
            main_pane,
            bg=c["glass"],
            highlightbackground=c["border"],
            highlightcolor=c["border_bright"],
            highlightthickness=1,
            padx=14,
            pady=14,
        )
        # Source-tab construction queries the processing controls to determine
        # whether Start should be enabled, so build this panel first.
        self.create_info_panel(right_frame)

        brand = tk.Frame(sidebar, bg=c["glass"])
        brand.pack(fill=tk.X, padx=18, pady=(20, 22))
        brand_mark = tk.Label(
            brand,
            text="◉",
            bg=c["glass"],
            fg=c["cyan"],
            font=("SF Pro Display", 25, "bold"),
        )
        brand_mark.pack(side=tk.LEFT, padx=(0, 10))
        brand_copy = tk.Frame(brand, bg=c["glass"])
        brand_copy.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(
            brand_copy,
            text="Meteor Detector",
            bg=c["glass"],
            fg=c["text"],
            anchor=tk.W,
            font=("SF Pro Display", 13, "bold"),
        ).pack(fill=tk.X)
        tk.Label(
            brand_copy,
            text="OBSERVATION STUDIO",
            bg=c["glass"],
            fg=c["text_tertiary"],
            anchor=tk.W,
            font=("SF Pro Text", 8, "bold"),
        ).pack(fill=tk.X, pady=(2, 0))

        nav_host = tk.Frame(sidebar, bg=c["glass"])
        nav_host.pack(fill=tk.BOTH, expand=True, padx=10)
        tk.Label(
            nav_host,
            text="ワークスペース",
            bg=c["glass"],
            fg=c["text_tertiary"],
            anchor=tk.W,
            font=("SF Pro Text", 9, "bold"),
        ).pack(fill=tk.X, padx=12, pady=(0, 7))

        header = ttk.Frame(content, style="Content.TFrame")
        header.pack(fill=tk.X, pady=(2, 12))
        header_copy = ttk.Frame(header, style="Content.TFrame")
        header_copy.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.page_eyebrow_var = tk.StringVar()
        self.page_title_var = tk.StringVar()
        self.page_subtitle_var = tk.StringVar()
        ttk.Label(
            header_copy,
            textvariable=self.page_eyebrow_var,
            style="Eyebrow.TLabel",
        ).pack(anchor=tk.W)
        ttk.Label(
            header_copy,
            textvariable=self.page_title_var,
            style="PageTitle.TLabel",
        ).pack(anchor=tk.W, pady=(2, 2))
        ttk.Label(
            header_copy,
            textvariable=self.page_subtitle_var,
            style="PageSubtitle.TLabel",
        ).pack(anchor=tk.W)

        header_actions = ttk.Frame(header, style="Content.TFrame")
        header_actions.pack(side=tk.RIGHT, padx=(12, 0))
        ttk.Button(
            header_actions,
            text="使い方",
            style="Toolbar.TButton",
            command=lambda: self._select_page(self.tab_usage),
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            header_actions,
            text="アクティビティ",
            style="Quiet.TButton",
            command=self._focus_activity,
        ).pack(side=tk.LEFT)

        ttk.Separator(content, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(0, 12))

        self.notebook = ttk.Notebook(content, style="Content.TNotebook")
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_usage = self.create_usage_tab(self.notebook)
        self.tab_source = self.create_source_tab(self.notebook)
        self.tab_settings = self.create_settings_tab(self.notebook)
        self.tab_analysis = self.create_analysis_tab(self.notebook)
        self.tab_chat = chat_gui.create_tab(self.notebook, app=self)
        self.tab_advanced_settings = self.create_advanced_settings_tab(self.notebook)

        tabs = (
            self.tab_usage,
            self.tab_source,
            self.tab_settings,
            self.tab_analysis,
            self.tab_chat,
            self.tab_advanced_settings,
        )
        for tab, meta in zip(tabs, ui_theme.PAGE_META):
            self.notebook.add(tab, text=meta["label"])

        self._page_tabs = tabs
        self._page_meta_by_tab = {
            str(tab): meta for tab, meta in zip(self._page_tabs, ui_theme.PAGE_META)
        }
        self._sidebar_buttons = {}

        for index, (tab, meta) in enumerate(zip(tabs, ui_theme.PAGE_META)):
            if index == 5:
                ttk.Separator(nav_host, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=(15, 14))
                tk.Label(
                    nav_host,
                    text="システム",
                    bg=c["glass"],
                    fg=c["text_tertiary"],
                    anchor=tk.W,
                    font=("SF Pro Text", 9, "bold"),
                ).pack(fill=tk.X, padx=12, pady=(0, 7))
            button = ui_theme.SidebarButton(
                nav_host,
                meta["glyph"],
                meta["label"],
                command=lambda selected_tab=tab: self._select_page(selected_tab),
            )
            button.pack(fill=tk.X, pady=2)
            self._sidebar_buttons[str(tab)] = button

        footer = tk.Frame(sidebar, bg=c["glass_strong"], padx=14, pady=12)
        footer.pack(fill=tk.X, padx=12, pady=12)
        tk.Label(
            footer,
            text="●  準備完了",
            bg=c["glass_strong"],
            fg=c["success"],
            anchor=tk.W,
            font=("SF Pro Text", 10, "bold"),
        ).pack(fill=tk.X)
        tk.Label(
            footer,
            text="入力ソースを選ぶと開始できます",
            bg=c["glass_strong"],
            fg=c["text_tertiary"],
            anchor=tk.W,
            font=("SF Pro Text", 9),
        ).pack(fill=tk.X, pady=(3, 0))

        main_pane.add(sidebar, width=232, minsize=210, stretch="never")
        main_pane.add(content, width=880, minsize=640, stretch="always")
        main_pane.add(right_frame, width=340, minsize=300, stretch="never")

        self.notebook.bind("<<NotebookTabChanged>>", self._on_page_changed, add="+")
        for index, tab in enumerate(tabs, start=1):
            self.bind_all(
                f"<Command-Key-{index}>",
                lambda _event, selected_tab=tab: self._select_page(selected_tab),
                add="+",
            )
            self.bind_all(
                f"<Control-Key-{index}>",
                lambda _event, selected_tab=tab: self._select_page(selected_tab),
                add="+",
            )
        self.bind_all("<Command-Return>", lambda _event: self.start_button.invoke(), add="+")
        self._create_application_menu()

        def _set_initial_sash():
            try:
                total = main_pane.winfo_width()
                if total <= 0:
                    return
                main_pane.sash_place(0, 232, 0)
                main_pane.sash_place(1, max(880, total - 340), 0)
            except Exception:
                pass

        self.after(120, _set_initial_sash)
        self.after_idle(lambda: self._select_page(self.tab_usage))

    def _create_application_menu(self):
        """Expose the main workflow through a conventional macOS menu bar."""
        menu_bar = tk.Menu(self, tearoff=False)

        app_menu = tk.Menu(menu_bar, tearoff=False)
        app_menu.add_command(
            label="Meteor Detectorについて",
            command=lambda: messagebox.showinfo(
                "Meteor Detector",
                f"{config.GUI_WINDOW_TITLE}\n流星観測・解析ワークスペース",
            ),
        )
        app_menu.add_separator()
        app_menu.add_command(
            label="設定…",
            accelerator="⌘,",
            command=lambda: self._select_page(self.tab_advanced_settings),
        )
        app_menu.add_separator()
        app_menu.add_command(label="Meteor Detectorを終了", accelerator="⌘Q", command=self.on_closing)
        menu_bar.add_cascade(label="Meteor Detector", menu=app_menu)

        file_menu = tk.Menu(menu_bar, tearoff=False)
        file_menu.add_command(
            label="動画を追加…",
            accelerator="⌘O",
            command=self.choose_source_videos,
        )
        file_menu.add_command(
            label="フォルダを追加…",
            accelerator="⇧⌘O",
            command=self.choose_source_folder,
        )
        file_menu.add_separator()
        file_menu.add_command(
            label="処理を開始",
            accelerator="⌘↩",
            command=lambda: self.start_button.invoke(),
        )
        file_menu.add_command(label="処理を停止", command=lambda: self.cancel_button.invoke())
        menu_bar.add_cascade(label="ファイル", menu=file_menu)

        view_menu = tk.Menu(menu_bar, tearoff=False)
        for index, (tab, meta) in enumerate(zip(self._page_tabs, ui_theme.PAGE_META), start=1):
            view_menu.add_command(
                label=meta["label"],
                accelerator=f"⌘{index}",
                command=lambda selected_tab=tab: self._select_page(selected_tab),
            )
        view_menu.add_separator()
        view_menu.add_command(label="イベントログ", command=self._focus_activity)
        menu_bar.add_cascade(label="表示", menu=view_menu)

        self.configure(menu=menu_bar)
        self.bind_all("<Command-comma>", lambda _event: self._select_page(self.tab_advanced_settings), add="+")
        self.bind_all("<Command-o>", lambda _event: self.choose_source_videos(), add="+")
        self.bind_all("<Command-Shift-O>", lambda _event: self.choose_source_folder(), add="+")

    def _select_page(self, tab):
        try:
            self.notebook.select(tab)
            self._sync_page_chrome(tab)
        except (AttributeError, tk.TclError):
            pass

    def _on_page_changed(self, _event=None):
        try:
            self._sync_page_chrome(self.notebook.select())
        except (AttributeError, tk.TclError):
            pass

    def _sync_page_chrome(self, tab):
        key = str(tab)
        meta = self._page_meta_by_tab.get(key)
        if not meta:
            return
        self.page_eyebrow_var.set(meta["eyebrow"])
        self.page_title_var.set(meta["title"])
        self.page_subtitle_var.set(meta["subtitle"])
        for tab_key, button in self._sidebar_buttons.items():
            button.set_selected(tab_key == key)

    def _focus_activity(self):
        try:
            self.status_panel.notebook.select(self.status_panel.log_frame)
            self.log_text.focus_set()
        except (AttributeError, tk.TclError):
            pass

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
            # Stop FFmpeg before destroying Tk.  The concat worker is a daemon
            # thread, so destroying the app first could otherwise orphan it.
            self.stop_video_concat()
            if self.noise_twin_training_process is not None:
                try:
                    self.noise_twin_training_process.terminate()
                except Exception:
                    pass
            self.append_log("設定を保存しています...")
            self._hide_summary_preview()
            self.close_rtsp_live_preview()
            self.close_camera_control()
            self.save_settings()
            # 自動更新を停止
            if self.auto_updater:
                self.auto_updater.stop()
            self.destroy()


if __name__ == "__main__":
    if "--noise-twin-worker" in sys.argv:
        worker_index = sys.argv.index("--noise-twin-worker")
        sys.argv = [sys.argv[0]] + sys.argv[worker_index + 1 :]
        import noise_twin_worker

        raise SystemExit(noise_twin_worker.main())
    else:
        os.makedirs(config.TEMP_CLIP_DIR, exist_ok=True)
        app = App()
        app.mainloop()
