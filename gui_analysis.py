from gui_common import *
import media_time
import camera_model_catalog
import meteor_radiant_analysis as mra
import meteor_radiant_visualizations as mrv
from datetime import timedelta

UI_BG = ui_theme.COLORS["content_raised"]
UI_FIELD = ui_theme.COLORS["field"]
UI_SELECTED = ui_theme.COLORS["glass_selected"]
UI_TEXT = ui_theme.COLORS["text"]
UI_CYAN = ui_theme.COLORS["cyan"]
UI_ACCENT = ui_theme.COLORS["accent"]


class AnalysisMixin:
    def create_analysis_tab(self, parent):
        """Create the '解析' tab where users can drop meteor info .txt files and run batch drawing."""
        # The tab is taller than many laptop displays.  Previously only the
        # file lists scrolled, leaving the video-concatenation controls below
        # the visible area with no way to reach them.
        tab_frame = ttk.Frame(parent)
        tab_frame.pack(fill=tk.BOTH, expand=True)
        self.analysis_tab_canvas = tk.Canvas(tab_frame, highlightthickness=0, bg=UI_BG)
        analysis_scrollbar = ttk.Scrollbar(tab_frame, orient=tk.VERTICAL, command=self.analysis_tab_canvas.yview)
        analysis_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.analysis_tab_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.analysis_tab_canvas.configure(yscrollcommand=analysis_scrollbar.set)

        self.analysis_scrollable_frame = ttk.Frame(self.analysis_tab_canvas)
        analysis_window = self.analysis_tab_canvas.create_window(
            (0, 0), window=self.analysis_scrollable_frame, anchor="nw"
        )
        self.analysis_scrollable_frame.bind(
            "<Configure>",
            lambda _event: self.analysis_tab_canvas.configure(scrollregion=self.analysis_tab_canvas.bbox("all")),
        )
        self.analysis_tab_canvas.bind(
            "<Configure>",
            lambda event: self.analysis_tab_canvas.itemconfigure(analysis_window, width=event.width),
        )

        def is_descendant(widget, ancestor):
            while widget is not None:
                if widget == ancestor:
                    return True
                try:
                    widget = widget.master
                except (AttributeError, tk.TclError):
                    return False
            return False

        def on_analysis_tab_mousewheel(event):
            # Keep each file list's own scrollbar independent from the tab.
            if is_descendant(event.widget, self.analysis_list_canvas) or is_descendant(event.widget, self.video_concat_list_canvas):
                return None
            if event.widget != self.analysis_tab_canvas and not is_descendant(event.widget, self.analysis_scrollable_frame):
                return None
            delta = getattr(event, "delta", 0)
            if delta:
                # macOS trackpads may report deltas smaller than 120.
                amount = -delta if abs(delta) < 120 else -int(delta / 120)
            else:
                amount = -1 if getattr(event, "num", None) == 4 else 1
            self.analysis_tab_canvas.yview_scroll(amount, "units")
            return "break"

        # Wheel events are delivered to the child control beneath the pointer.
        self.bind_all("<MouseWheel>", on_analysis_tab_mousewheel, add="+")
        self.bind_all("<Button-4>", on_analysis_tab_mousewheel, add="+")
        self.bind_all("<Button-5>", on_analysis_tab_mousewheel, add="+")

        frame = ttk.Frame(self.analysis_scrollable_frame)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        lf = ttk.LabelFrame(frame, text="流星解析 (info.txt ドロップ)")
        lf.pack(fill=tk.BOTH, expand=True, pady=5)

        drop_label = ttk.Label(
            lf,
            text="流星の info.txt をここにドロップ",
            style="DropZone.TLabel",
        )
        drop_label.pack(fill=tk.X, pady=5)
        drop_label.drop_target_register(DND_FILES)
        drop_label.dnd_bind('<<Drop>>', self.drop_analysis)

        # Analysis list (styled)
        analysis_list_container = ttk.Frame(lf)
        analysis_list_container.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.analysis_list_canvas = tk.Canvas(analysis_list_container, bg=UI_FIELD, highlightthickness=0, height=100)
        self.analysis_list_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(analysis_list_container, orient=tk.VERTICAL, command=self.analysis_list_canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.analysis_list_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.analysis_list_frame = tk.Frame(self.analysis_list_canvas, bg=UI_FIELD)
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

        # The model is selected in the app, not through Finder.  The detail
        # line makes the validated coverage and reference night visible before
        # the user starts the high-precision radiant calculation.
        radiant_model_tools = ttk.LabelFrame(frame, text="高精度・放射点解析")
        radiant_model_tools.pack(fill=tk.X, pady=(2, 8))
        radiant_model_grid = ttk.Frame(radiant_model_tools)
        radiant_model_grid.pack(fill=tk.X, padx=8, pady=8)
        radiant_model_grid.columnconfigure(1, weight=1)
        ttk.Label(radiant_model_grid, text="プレートソルブモデル").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 8), pady=(0, 4)
        )
        self.analysis_radiant_model_var = tk.StringVar(value="自動選択（撮影日の高精度モデル）")
        self.analysis_radiant_model_path_var = tk.StringVar(value="")
        self.analysis_radiant_model_info_var = tk.StringVar(
            value="撮影動画と同じカメラの登録済みモデルを自動選択します"
        )
        self.analysis_radiant_model_choices = {}
        self.analysis_radiant_models = []
        self.analysis_radiant_model_combo = ttk.Combobox(
            radiant_model_grid,
            textvariable=self.analysis_radiant_model_var,
            state="readonly",
        )
        self.analysis_radiant_model_combo.grid(row=0, column=1, sticky=tk.EW, pady=(0, 4))
        self.analysis_radiant_model_combo.bind("<<ComboboxSelected>>", self._on_analysis_radiant_model_selected)
        ttk.Label(
            radiant_model_grid,
            textvariable=self.analysis_radiant_model_info_var,
            style="GlassMuted.TLabel",
            wraplength=600,
            justify=tk.LEFT,
        ).grid(row=1, column=0, columnspan=2, sticky=tk.W)
        self._refresh_analysis_radiant_models()

        action_frame = ttk.Frame(frame)
        action_frame.pack(fill=tk.X, pady=8)
        action_frame.columnconfigure(0, weight=1, uniform="analysis_actions")
        action_frame.columnconfigure(1, weight=1, uniform="analysis_actions")

        trajectory_tools = ttk.LabelFrame(action_frame, text="軌道・校正")
        trajectory_tools.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, 4), pady=(0, 8))
        trajectory_grid = ttk.Frame(trajectory_tools)
        trajectory_grid.pack(fill=tk.X)
        trajectory_grid.columnconfigure(0, weight=1, uniform="trajectory")
        trajectory_grid.columnconfigure(1, weight=1, uniform="trajectory")
        self.btn_analysis_start = ttk.Button(
            trajectory_grid,
            text="旧形式の軌道表示",
            command=self.start_analysis,
            style="Gray.TButton",
        )
        self.btn_radiant_analysis = ttk.Button(
            trajectory_grid,
            text="高精度放射点解析",
            command=self.start_radiant_analysis,
        )
        add_point_button = ttk.Button(
            trajectory_grid,
            text="座標点を追加",
            command=self.add_custom_point,
            style="Gray.TButton",
        )
        manage_point_button = ttk.Button(
            trajectory_grid,
            text="座標点を管理",
            command=self.manage_coordinates,
            style="Gray.TButton",
        )
        self.btn_long_exposure = ttk.Button(
            trajectory_grid,
            text="長時間輝線マップ",
            command=self.create_long_exposure_map_callback,
            style="Gray.TButton",
        )
        self.btn_distortion = ttk.Button(
            trajectory_grid,
            text="ゆがみ補正",
            command=self.apply_distortion_correction_callback,
            style="Gray.TButton",
        )
        self.btn_distortion_selfcal = ttk.Button(
            trajectory_grid,
            text="夜間自己校正",
            command=self.estimate_distortion_map_night_callback,
            style="Gray.TButton"
        )
        self.btn_distortion_map_view = ttk.Button(
            trajectory_grid,
            text="ゆがみマップ表示",
            command=self.visualize_distortion_map_callback,
            style="Gray.TButton"
        )
        self.btn_angle_analysis = ttk.Button(
            trajectory_grid,
            text="角度分布分析",
            command=self.analyze_angles_callback,
            style="Gray.TButton",
        )
        trajectory_buttons = (
            self.btn_analysis_start,
            self.btn_radiant_analysis,
            add_point_button,
            manage_point_button,
            self.btn_long_exposure,
            self.btn_distortion,
            self.btn_distortion_selfcal,
            self.btn_distortion_map_view,
            self.btn_angle_analysis,
        )
        for index, button in enumerate(trajectory_buttons):
            button.grid(
                row=index // 2,
                column=index % 2,
                sticky=tk.EW,
                padx=(0, 4) if index % 2 == 0 else (4, 0),
                pady=3,
            )

        media_tools = ttk.LabelFrame(action_frame, text="メディア作成")
        media_tools.grid(row=0, column=1, sticky=tk.NSEW, padx=(4, 0), pady=(0, 8))
        media_grid = ttk.Frame(media_tools)
        media_grid.pack(fill=tk.X)
        media_grid.columnconfigure(0, weight=1, uniform="media")
        media_grid.columnconfigure(1, weight=1, uniform="media")
        self.btn_blend_image = ttk.Button(media_grid, text="比較明合成画像", command=self.create_lighten_blend_image_callback)
        self.btn_blend_video = ttk.Button(media_grid, text="比較明合成動画", command=self.create_lighten_blend_video_callback)
        self.btn_timelapse = ttk.Button(media_grid, text="タイムラプス", command=self.create_timelapse_callback)
        self.btn_camera_control = ttk.Button(media_grid, text="カメラコントロール", command=self.open_camera_control, style="Gray.TButton")
        self.btn_model_training = ttk.Button(media_grid, text="機械学習モデル作成", command=self.open_model_training_tool)
        media_buttons = (
            self.btn_blend_image,
            self.btn_blend_video,
            self.btn_timelapse,
            self.btn_camera_control,
            self.btn_model_training,
        )
        for index, button in enumerate(media_buttons):
            button.grid(
                row=index // 2,
                column=index % 2,
                sticky=tk.EW,
                padx=(0, 4) if index % 2 == 0 else (4, 0),
                pady=3,
            )

        lf_concat = ttk.LabelFrame(frame, text="動画連結")
        lf_concat.pack(fill=tk.BOTH, expand=True, pady=5)

        concat_drop_label = ttk.Label(
            lf_concat,
            text="連結する動画をここにドロップ",
            style="DropZone.TLabel",
        )
        concat_drop_label.pack(fill=tk.X, pady=5)
        concat_drop_label.drop_target_register(DND_FILES)
        concat_drop_label.dnd_bind('<<Drop>>', self.drop_video_concat)

        concat_list_container = ttk.Frame(lf_concat)
        concat_list_container.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.video_concat_list_canvas = tk.Canvas(concat_list_container, bg=UI_FIELD, highlightthickness=0, height=80)
        self.video_concat_list_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        concat_scrollbar = ttk.Scrollbar(concat_list_container, orient=tk.VERTICAL, command=self.video_concat_list_canvas.yview)
        concat_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.video_concat_list_canvas.configure(yscrollcommand=concat_scrollbar.set)
        
        self.video_concat_list_frame = tk.Frame(self.video_concat_list_canvas, bg=UI_FIELD)
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
                                      values=["Auto", "1000k","2000k","4000k", "8000k", "12000k", "16000k", "20000k"], width=8, state="readonly")
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
        ttk.Checkbutton(
            concat_settings_row2,
            text="適応固定パターン＋21フレーム平均を適用",
            variable=self.video_concat_enhancement_var,
        ).pack(side=tk.LEFT, padx=(12, 0))
        help_label = tk.Label(concat_settings_row2, text="?", font=("", 9, "bold"), fg=UI_CYAN, bg=UI_BG, cursor="hand2")
        
        help_label.pack(side=tk.LEFT, padx=(2, 5))
        
        help_text = ("動画連結時に、入力ファイルのタイムスタンプ情報が正しくない場合や、\n"
                     "動画間で不整合がある場合に、このオプションを有効にしてください。\n"
                     "映像情報を保持した一時ファイルへ高速変換するため、追加の空き容量が必要ですが、\n"
                     "連結の安定性が向上します。")
        self._setup_help_tooltip(help_label, help_text)

        concat_timestamp_row = ttk.Frame(lf_concat)
        concat_timestamp_row.pack(fill=tk.X, pady=(0, 5))
        ttk.Checkbutton(
            concat_timestamp_row,
            text="実時刻を動画に表示",
            variable=self.video_concat_timestamp_enabled_var,
        ).pack(side=tk.LEFT, padx=(5, 8))
        ttk.Label(concat_timestamp_row, text="位置:").pack(side=tk.LEFT)
        ttk.Combobox(
            concat_timestamp_row,
            textvariable=self.video_concat_timestamp_position_var,
            values=["右下", "左下", "右上", "左上"],
            width=5,
            state="readonly",
        ).pack(side=tk.LEFT, padx=(3, 8))
        ttk.Label(concat_timestamp_row, text="文字サイズ:").pack(side=tk.LEFT)
        ttk.Spinbox(
            concat_timestamp_row,
            from_=0.8,
            to=4.0,
            increment=0.1,
            textvariable=self.video_concat_timestamp_size_var,
            width=5,
        ).pack(side=tk.LEFT, padx=(3, 2))
        ttk.Label(concat_timestamp_row, text="%").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(concat_timestamp_row, text="実時刻補正:").pack(side=tk.LEFT)
        ttk.Entry(
            concat_timestamp_row,
            textvariable=self.video_concat_timestamp_offset_var,
            width=7,
        ).pack(side=tk.LEFT, padx=(3, 2))
        ttk.Label(concat_timestamp_row, text="秒（+で後へ）").pack(side=tk.LEFT)

        self.btn_video_concat_start = ttk.Button(lf_concat, text="連結開始", command=self.start_video_concat)
        self.btn_video_concat_start.pack(pady=5)

        # A Notebook tab must be a direct child of the Notebook.  ``frame`` is
        # embedded in the canvas, so return the outer tab container instead.
        return tab_frame

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
            bg_color = UI_BG
            fg_color = UI_TEXT
            
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
        
        item_frame = tk.Frame(self.analysis_list_frame, bg=UI_FIELD, cursor="hand2")
        item_frame.pack(fill=tk.X, padx=2, pady=1)
        
        badge_canvas = tk.Canvas(item_frame, width=50, height=22, bg=UI_FIELD, highlightthickness=0)
        badge_canvas.pack(side=tk.LEFT, padx=(4, 6), pady=2)
        
        self._draw_rounded_rect(badge_canvas, 2, 2, 48, 20, 8, fill="#E67E22", outline="")
        badge_canvas.create_text(25, 11, text="TXT", fill="white", font=("Segoe UI", 8, "bold"))
        
        path_label = tk.Label(item_frame, text=filepath, bg=UI_FIELD, fg=UI_TEXT,
                               anchor="w", font=("SF Pro Text", 9))
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
            item['frame'].config(bg=UI_FIELD)
            item['label'].config(bg=UI_FIELD)
            item['badge'].config(bg=UI_FIELD)
            item['selected'] = False
            self.analysis_selected_indices.discard(index)
        else:
            item['frame'].config(bg=UI_SELECTED)
            item['label'].config(bg=UI_SELECTED)
            item['badge'].config(bg=UI_SELECTED)
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
        
        item_frame = tk.Frame(self.video_concat_list_frame, bg=UI_FIELD, cursor="hand2")
        item_frame.pack(fill=tk.X, padx=2, pady=1)
        
        badge_canvas = tk.Canvas(item_frame, width=30, height=22, bg=UI_FIELD, highlightthickness=0)
        badge_canvas.pack(side=tk.LEFT, padx=(4, 6), pady=2)
        
        self._draw_rounded_rect(badge_canvas, 2, 2, 28, 20, 8, fill=UI_ACCENT, outline="")
        badge_canvas.create_text(15, 11, text=str(index + 1), fill="white", font=("Segoe UI", 8, "bold"))
        
        filename = os.path.basename(filepath)
        path_label = tk.Label(item_frame, text=filename, bg=UI_FIELD, fg=UI_TEXT,
                               anchor="w", font=("SF Pro Text", 9))
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
            item['frame'].config(bg=UI_FIELD)
            item['label'].config(bg=UI_FIELD)
            item['badge'].config(bg=UI_FIELD)
            item['selected'] = False
            self.video_concat_selected_indices.discard(index)
        else:
            item['frame'].config(bg=UI_SELECTED)
            item['label'].config(bg=UI_SELECTED)
            item['badge'].config(bg=UI_SELECTED)
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
            self._draw_rounded_rect(item['badge'], 2, 2, 28, 20, 8, fill=UI_ACCENT, outline="")
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
        
        files = list(self.video_concat_files)
        start_time, time_source, start_file = media_time.first_media_start_time(files)
        try:
            timestamp_offset = float(self.video_concat_timestamp_offset_var.get())
        except (TypeError, ValueError, tk.TclError):
            timestamp_offset = 0.0
        if start_time is not None:
            start_time += timedelta(seconds=timestamp_offset)
        default_name = f"{(start_time or datetime.now()).strftime('%Y%m%d%H%M%S')}.mp4"

        # 出力ファイルを選択
        output_path = filedialog.asksaveasfilename(
            title="出力ファイルを保存",
            initialfile=default_name,
            defaultextension=".mp4",
            filetypes=[("MP4ファイル", "*.mp4"), ("すべてのファイル", "*.*")]
        )
        
        if not output_path:
            return
        
        bitrate = self.video_concat_bitrate_var.get()
        codec = self.video_concat_codec_var.get()
        fps_str = self.video_concat_fps_var.get()
        safe_mode = self.video_concat_safe_mode_var.get()
        apply_enhancement = self.video_concat_enhancement_var.get()
        try:
            timestamp_size = float(self.video_concat_timestamp_size_var.get())
        except (TypeError, ValueError, tk.TclError):
            timestamp_size = config.VIDEO_CONCAT_TIMESTAMP_SIZE_PERCENT
        timestamp_settings = {
            "enabled": self.video_concat_timestamp_enabled_var.get(),
            "position": self.video_concat_timestamp_position_var.get(),
            "size_percent": max(0.8, min(4.0, timestamp_size)),
            "offset_seconds": timestamp_offset,
        }
        
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
        self.append_log(
            f"設定: ビットレート={bitrate}, コーデック={codec}, FPS={fps_str}, "
            f"保存物補正={'ON' if apply_enhancement else 'OFF'}"
        )
        if start_time is not None:
            self.append_log(
                f"開始時刻: {start_time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"({time_source}, 補正={timestamp_offset:+g}秒, {os.path.basename(start_file or '')})"
            )

        # Tk calls from a worker thread can block on macOS before the first
        # ffprobe invocation.  The main thread polls this queue and performs
        # every log/dialog update itself.
        event_queue = queue.Queue()
        self.cancel_flag.clear()
        self.btn_video_concat_start.configure(state=tk.DISABLED)
        thread = threading.Thread(
            target=self._video_concat_worker,
            args=(files, output_path, bitrate, codec, fps_val, safe_mode, apply_enhancement, timestamp_settings, event_queue),
            daemon=True
        )
        self.video_concat_thread = thread
        thread.start()
        self._poll_video_concat_events(event_queue, thread)

    def _poll_video_concat_events(self, event_queue, thread):
        """Apply video-concatenation worker events on Tk's main thread."""
        finished = False
        try:
            while True:
                event_type, payload = event_queue.get_nowait()
                if event_type == "progress":
                    self.append_log(payload)
                elif event_type == "success":
                    message, output_path = payload
                    self.append_log(f"連結完了: {output_path}")
                    messagebox.showinfo("完了", message)
                    finished = True
                elif event_type == "error":
                    self.append_log(f"連結エラー: {payload}")
                    messagebox.showerror("エラー", payload)
                    finished = True
        except queue.Empty:
            pass

        if finished:
            self.video_concat_thread = None
            self.btn_video_concat_start.configure(state=tk.NORMAL)
            return

        if thread.is_alive() or not event_queue.empty():
            self.after(50, self._poll_video_concat_events, event_queue, thread)
        else:
            # A terminal event should always be sent, but do not leave the UI
            # disabled if the worker terminates unexpectedly.
            self.append_log("連結エラー: ワーカースレッドが予期せず終了しました")
            self.video_concat_thread = None
            self.btn_video_concat_start.configure(state=tk.NORMAL)

    def stop_video_concat(self, wait_timeout=5):
        """Cancel concatenation and ensure its FFmpeg child cannot outlive the app."""
        self.cancel_flag.set()

        def terminate_process():
            process = getattr(self, "video_concat_process", None)
            if process is None or process.poll() is not None:
                return
            try:
                process.terminate()
                process.wait(timeout=wait_timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
            except OSError:
                pass

        terminate_process()
        worker = getattr(self, "video_concat_thread", None)
        if worker is not None and worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=wait_timeout)
        # Cover the small race where FFmpeg starts while shutdown is waiting
        # for a validation/remux subprocess to finish.
        terminate_process()

    def _video_concat_worker(self, files, output_path, bitrate, codec, fps, safe_mode, apply_enhancement, timestamp_settings, event_queue):
        """動画連結のバックグラウンド処理"""
        def progress_callback(progress, message):
            event_queue.put(("progress", message))
        
        def cancel_check():
            return self.cancel_flag.is_set()

        def process_callback(process):
            self.video_concat_process = process
        
        try:
            success, message = video_processor.concatenate_videos(
                input_files=files,
                output_path=output_path,
                bitrate=bitrate,
                codec=codec,
                fps=fps,
                safe_mode=safe_mode,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
                process_callback=process_callback,
                apply_enhancement=apply_enhancement,
                fixed_pattern_path=self.rtsp_dark_file,
                timestamp_settings=timestamp_settings,
            )
            
            if success:
                event_queue.put(("success", (message, output_path)))
            else:
                event_queue.put(("error", message))
        except Exception as e:
            error_msg = f"予期せぬエラー: {e}"
            event_queue.put(("error", error_msg))

    def _refresh_analysis_radiant_models(self):
        """Populate the human-readable high-precision model selector."""
        models = camera_model_catalog.discover_camera_models()
        self.analysis_radiant_models = models
        self.analysis_radiant_model_choices = {}
        values = ["自動選択（撮影日の高精度モデル）"]
        for model in models:
            display = model["display_name"]
            if display in self.analysis_radiant_model_choices:
                display = f"{display} [{len(values)}]"
            values.append(display)
            self.analysis_radiant_model_choices[display] = model
        self.analysis_radiant_model_combo.configure(values=values)

        preferred_path = ""
        try:
            preferred_path = str(self.plate_solve_model_path_var.get() or "")
        except (AttributeError, tk.TclError):
            pass
        selected = next((model for model in models if model["path"] == preferred_path), None)
        if selected:
            self.analysis_radiant_model_var.set(selected["display_name"])
            self._on_analysis_radiant_model_selected()
        else:
            self.analysis_radiant_model_var.set(values[0])
            self.analysis_radiant_model_path_var.set("")
            self.analysis_radiant_model_info_var.set(
                "撮影動画と同じカメラの登録済みモデルを、撮影日・解像度・被覆率から自動選択します"
                if models else "登録済みモデルが見つかりません。先に高精度カメラ補正を作成してください"
            )

    def _on_analysis_radiant_model_selected(self, _event=None):
        display = self.analysis_radiant_model_var.get()
        model = self.analysis_radiant_model_choices.get(display)
        if model is None:
            self.analysis_radiant_model_path_var.set("")
            self.analysis_radiant_model_info_var.set(
                "撮影動画と同じカメラの登録済みモデルを、撮影日・解像度・被覆率から自動選択します"
            )
            return
        self.analysis_radiant_model_path_var.set(model["path"])
        self.analysis_radiant_model_info_var.set(camera_model_catalog.format_model_details(model))

    def _radiant_analysis_worker(self, files, model_path, event_queue):
        try:
            report = mra.analyze_info_files(
                files,
                model_path=model_path or None,
                progress_callback=lambda message: event_queue.put(("progress", message)),
            )
            event_queue.put(("success", report))
        except Exception as exc:
            event_queue.put(("error", str(exc)))

    def start_radiant_analysis(self):
        """Run the high-precision support-aware radiant analysis in a worker."""
        if not self.check_admin_password():
            return
        if not self.analysis_files:
            messagebox.showwarning("情報", "解析するinfo.txtを追加してください。")
            return

        files = list(self.analysis_files)
        model_path = str(self.analysis_radiant_model_path_var.get() or "")
        win = Toplevel(self)
        win.title("高精度・流星放射点解析")
        win.geometry("1280x820")
        win.minsize(980, 640)
        self.analysis_radiant_window = win
        self.analysis_radiant_report = None

        header = ttk.Frame(win)
        header.pack(fill=tk.X, padx=14, pady=(12, 6))
        ttk.Label(header, text="流星放射点解析", style="PageTitle.TLabel").pack(side=tk.LEFT)
        status_var = tk.StringVar(value=f"{len(files)}件を準備中…")
        ttk.Label(header, textvariable=status_var, style="GlassMuted.TLabel").pack(side=tk.RIGHT)

        content = ttk.Panedwindow(win, orient=tk.HORIZONTAL)
        content.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 10))
        plot_frame = ttk.Frame(content)
        summary_frame = ttk.Frame(content, padding=(10, 0, 0, 0))
        content.add(plot_frame, weight=4)
        content.add(summary_frame, weight=3)
        ttk.Label(
            plot_frame,
            text="実線 = 検出された流星経路 / 破線 = 判定した放射点方向",
            style="GlassMuted.TLabel",
        ).pack(anchor=tk.W, pady=(0, 4))
        progress_label = ttk.Label(plot_frame, text="高精度モデルで天球座標へ変換しています…")
        progress_label.pack(anchor=tk.W, pady=(0, 4))
        result_host = ttk.Frame(plot_frame)
        result_host.pack(fill=tk.BOTH, expand=True)
        summary_title = ttk.Label(summary_frame, text="解析結果", style="PageTitle.TLabel")
        summary_title.pack(anchor=tk.W, pady=(0, 6))
        summary_text = tk.StringVar(value="解析中…")
        ttk.Label(summary_frame, textvariable=summary_text, style="GlassMuted.TLabel", wraplength=360, justify=tk.LEFT).pack(
            anchor=tk.W, fill=tk.X, pady=(0, 8)
        )
        tree = ttk.Treeview(
            summary_frame,
            columns=("file", "shower", "angle", "confidence"),
            show="headings",
            height=16,
        )
        tree.heading("file", text="ファイル")
        tree.heading("shower", text="放射点候補")
        tree.heading("angle", text="角距離")
        tree.heading("confidence", text="判定")
        tree.column("file", width=175, anchor=tk.W)
        tree.column("shower", width=135, anchor=tk.W)
        tree.column("angle", width=70, anchor=tk.E)
        tree.column("confidence", width=90, anchor=tk.W)
        tree.pack(fill=tk.BOTH, expand=True)
        detail_text = tk.Text(
            summary_frame,
            height=7,
            wrap=tk.WORD,
            background=UI_FIELD,
            foreground=UI_TEXT,
            relief=tk.FLAT,
            borderwidth=0,
        )
        detail_text.pack(fill=tk.X, pady=(8, 0))
        detail_text.configure(state=tk.DISABLED)

        footer = ttk.Frame(win)
        footer.pack(fill=tk.X, padx=14, pady=(0, 12))
        save_button = ttk.Button(footer, text="解析結果をPNG保存", state=tk.DISABLED)
        save_button.pack(side=tk.LEFT)
        save_all_button = ttk.Button(footer, text="全方式の描画を保存", state=tk.DISABLED)
        save_all_button.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(footer, text="閉じる", command=win.destroy).pack(side=tk.RIGHT)

        event_queue = queue.Queue()
        self.btn_radiant_analysis.configure(state=tk.DISABLED)
        worker = threading.Thread(
            target=self._radiant_analysis_worker,
            args=(files, model_path, event_queue),
            daemon=True,
        )
        self.analysis_radiant_thread = worker
        worker.start()

        def show_report(report):
            self.analysis_radiant_report = report
            status_var.set("解析完了")
            progress_label.configure(text="表示内容を確認できます。流星線分は有効領域内のみ描画しています。")
            summary_text.set(
                f"モデル: {report.model_label}\n"
                f"有効領域内: {len(report.supported_results)}件 / 読み込み: {len(report.results) + len(report.skipped)}件\n"
                f"除外: {len(report.skipped)}件"
            )
            for result in report.results:
                angle = f"{result.radiant_distance_deg:.1f}°" if result.radiant_distance_deg is not None else "—"
                tree.insert(
                    "",
                    tk.END,
                    values=(os.path.basename(result.info_path), f"{result.shower_code} {result.shower_name}", angle, result.confidence),
                )
            details = []
            for result in report.results:
                details.append(f"{os.path.basename(result.info_path)}: {result.note}")
            for path, reason in report.skipped:
                details.append(f"除外 {os.path.basename(path)}: {reason}")
            detail_text.configure(state=tk.NORMAL)
            detail_text.delete("1.0", tk.END)
            detail_text.insert("1.0", "\n".join(details) or "詳細はありません")
            detail_text.configure(state=tk.DISABLED)
            try:
                from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
                from matplotlib.figure import Figure

                figure = Figure(figsize=(8, 7), facecolor="#0B0F18")
                mra.draw_radiant_sphere(report, figure=figure)
                figure_canvas = FigureCanvasTkAgg(figure, master=result_host)
                figure_canvas.draw()
                figure_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                self.analysis_radiant_figure = figure
                self.analysis_radiant_canvas = figure_canvas

                def save_plot():
                    output = filedialog.asksaveasfilename(
                        title="放射点解析を保存",
                        initialfile="radiant_analysis.png",
                        defaultextension=".png",
                        filetypes=(("PNG画像", "*.png"), ("すべてのファイル", "*.*")),
                    )
                    if output:
                        mra.save_radiant_report_plot(report, output)
                        self.append_log(f"放射点解析を保存: {output}")

                def save_all_visualizations():
                    directory = filedialog.askdirectory(
                        title="全方式の放射点描画を保存するフォルダを選択",
                        parent=win,
                    )
                    if not directory:
                        return
                    save_all_button.configure(state=tk.DISABLED)
                    progress_label.configure(text="全方式の描画を作成しています…（完了まで数十秒かかる場合があります）")
                    bundle_queue = queue.Queue()

                    def bundle_worker():
                        try:
                            paths = mrv.save_visualization_bundle(
                                report,
                                files,
                                directory,
                                prefix="radiant_analysis",
                            )
                            bundle_queue.put(("success", paths))
                        except Exception as exc:
                            bundle_queue.put(("error", str(exc)))

                    bundle_thread = threading.Thread(target=bundle_worker, daemon=True)
                    bundle_thread.start()

                    def poll_bundle():
                        try:
                            if not win.winfo_exists():
                                return
                        except tk.TclError:
                            return
                        try:
                            event_type, payload = bundle_queue.get_nowait()
                        except queue.Empty:
                            if bundle_thread.is_alive():
                                self.after(100, poll_bundle)
                            else:
                                save_all_button.configure(state=tk.NORMAL)
                                progress_label.configure(text="全方式の描画処理が予期せず終了しました。")
                            return
                        save_all_button.configure(state=tk.NORMAL)
                        if event_type == "success":
                            progress_label.configure(text=f"全方式の描画を保存しました: {directory}")
                            self.append_log(f"放射点解析の全方式描画を保存: {directory}")
                            messagebox.showinfo(
                                "保存完了",
                                "球面・球面一回転GIF・Aitoff・収束図・カメラ投影・RA-Dec・密度・極座標・時系列動画を保存しました。",
                                parent=win,
                            )
                        else:
                            progress_label.configure(text=f"全方式の描画に失敗しました: {payload}")
                            self.append_log(f"全方式の放射点描画エラー: {payload}")
                            messagebox.showerror("保存エラー", str(payload), parent=win)

                    self.after(100, poll_bundle)

                save_button.configure(state=tk.NORMAL, command=save_plot)
                save_all_button.configure(state=tk.NORMAL, command=save_all_visualizations)
            except Exception as exc:
                progress_label.configure(text=f"球面表示に失敗しました: {exc}")
                self.append_log(f"放射点解析の描画エラー: {exc}")

        def poll():
            try:
                if not win.winfo_exists():
                    self.analysis_radiant_thread = None
                    self.btn_radiant_analysis.configure(state=tk.NORMAL)
                    return
            except tk.TclError:
                self.analysis_radiant_thread = None
                self.btn_radiant_analysis.configure(state=tk.NORMAL)
                return
            finished = False
            try:
                while True:
                    event_type, payload = event_queue.get_nowait()
                    if event_type == "progress":
                        progress_label.configure(text=payload)
                    elif event_type == "success":
                        show_report(payload)
                        finished = True
                    elif event_type == "error":
                        status_var.set("解析失敗")
                        progress_label.configure(text=f"解析に失敗しました: {payload}")
                        self.append_log(f"放射点解析エラー: {payload}")
                        messagebox.showerror("放射点解析エラー", payload, parent=win)
                        finished = True
            except queue.Empty:
                pass
            if finished:
                self.analysis_radiant_thread = None
                self.btn_radiant_analysis.configure(state=tk.NORMAL)
            elif worker.is_alive() or not event_queue.empty():
                self.after(80, poll)
            else:
                status_var.set("解析失敗")
                progress_label.configure(text="解析処理が予期せず終了しました。ログを確認してください。")
                self.analysis_radiant_thread = None
                self.btn_radiant_analysis.configure(state=tk.NORMAL)

        self.after(80, poll)

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
