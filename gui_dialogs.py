from gui_common import *
from concurrent.futures import ThreadPoolExecutor, as_completed
import camera_model_catalog


class TimelapseDragDropWindow(Toplevel):
    """タイムラプス作成用のドラッグ＆ドロップウィンドウ"""
    
    def __init__(self, parent, log_callback):
        super().__init__(parent)
        self.parent = parent
        self.log_callback = log_callback
        self.dropped_paths = []
        self.timelapse_mask = None  # タイムラプス用マスク
        self.timelapse_timestamp_enabled_var = tk.BooleanVar(
            value=config.TIMELAPSE_TIMESTAMP_ENABLED
        )
        self.timelapse_timestamp_position_var = tk.StringVar(value="右下")
        self.timelapse_timestamp_size_var = tk.StringVar(
            value=str(config.TIMELAPSE_TIMESTAMP_SIZE_PERCENT)
        )
        self.temporal_mean_radius_var = tk.IntVar(
            value=config.TIMELAPSE_TEMPORAL_MEAN_RADIUS_FRAMES
        )
        self.temporal_mean_summary_var = tk.StringVar()
        self.timelapse_annotation_enabled_var = tk.BooleanVar(
            value=getattr(config, "TIMELAPSE_LOCAL_ANNOTATION_ENABLED", False)
        )
        self.timelapse_annotation_calibration_var = tk.StringVar(
            value=getattr(config, "TIMELAPSE_ANNOTATION_CALIBRATION_PATH", "") or ""
        )
        self.timelapse_annotation_model_var = tk.StringVar(
            value="自動選択（撮影日に合う補正データ）"
        )
        self.timelapse_annotation_model_path_var = tk.StringVar()
        self.timelapse_annotation_model_info_var = tk.StringVar(
            value="未選択（撮影日に合うカメラ補正データを自動選択）"
        )
        self.timelapse_annotation_model_entries = []
        self.timelapse_annotation_model_by_display = {}
        self.timelapse_constellations_var = tk.BooleanVar(
            value=getattr(config, "TIMELAPSE_CONSTELLATIONS_ENABLED", True)
        )
        self.timelapse_grid_var = tk.BooleanVar(value=True)
        self.timelapse_detected_stars_var = tk.BooleanVar(value=False)
        self.timelapse_annotation_reference_sample_var = tk.IntVar(value=0)
        self.timelapse_annotation_reference_selected_var = tk.BooleanVar(value=False)
        self.timelapse_insert_meteors_var = tk.BooleanVar(value=False)
        self.timelapse_fixed_pattern_enabled_var = tk.BooleanVar(
            value=bool(
                getattr(parent, "apply_rtsp_dark_var", None)
                and parent.apply_rtsp_dark_var.get()
            )
        )
        self.timelapse_model_start_var = tk.StringVar()
        self.timelapse_model_end_var = tk.StringVar()
        
        self.title("タイムラプス作成")
        # Keep the model description readable while allowing the content to
        # scroll on smaller displays.
        self.geometry("620x860")
        self.minsize(520, 620)
        self.resizable(True, True)
        
        self.setup_ui()
        
        self.transient(parent)
        self.grab_set()
    
    def setup_ui(self):
        # Keep every control reachable on smaller displays.  The contents are
        # embedded in a canvas rather than making individual sections scroll.
        scroll_host = ttk.Frame(self, style="Content.TFrame")
        scroll_host.pack(fill=tk.BOTH, expand=True)
        self.timelapse_scroll_canvas = tk.Canvas(
            scroll_host,
            background=ui_theme.COLORS["content"],
            highlightthickness=0,
            borderwidth=0,
        )
        scrollbar = ttk.Scrollbar(
            scroll_host, orient=tk.VERTICAL,
            command=self.timelapse_scroll_canvas.yview,
        )
        self.timelapse_scroll_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.timelapse_scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # The scrollable frame must be a canvas child.  Making it a toplevel
        # child causes an incomplete scroll region on macOS.
        main_frame = ttk.Frame(
            self.timelapse_scroll_canvas,
            padding=(18, 16, 18, 22),
            style="Content.TFrame",
        )
        self._timelapse_scroll_window = self.timelapse_scroll_canvas.create_window(
            (0, 0), window=main_frame, anchor=tk.NW
        )
        main_frame.bind("<Configure>", self._update_timelapse_scroll_region)
        self.timelapse_scroll_canvas.bind("<Configure>", self._resize_timelapse_scroll_content)
        self.bind("<MouseWheel>", self._scroll_timelapse_window, add="+")
        self.bind("<Button-4>", self._scroll_timelapse_window, add="+")
        self.bind("<Button-5>", self._scroll_timelapse_window, add="+")

        hero = ttk.Frame(main_frame, style="Glass.TFrame", padding=(16, 13))
        hero.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(
            hero,
            text="観測の流れを、ひとつの動画に",
            style="Glass.TLabel",
            font=("SF Pro Display", 16, "bold"),
        ).pack(anchor=tk.W)
        ttk.Label(
            hero,
            text="素材を追加して、見た目と星空の補正を選ぶだけで作成できます。\n"
                 "下の設定はスクロールできます。2本指スクロールにも対応しています。",
            style="GlassMuted.TLabel",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(4, 0))
        
        drop_frame = ttk.LabelFrame(main_frame, text="ファイル / フォルダ", padding=10)
        drop_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.drop_label = ttk.Label(
            drop_frame,
            text="ここにフォルダや動画ファイルを\nドラッグ＆ドロップしてください",
            style="DropZone.TLabel",
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
            bg=ui_theme.COLORS["field"],
            fg=ui_theme.COLORS["text"],
            selectbackground=ui_theme.COLORS["selection"],
            selectforeground=ui_theme.COLORS["text"],
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

        timestamp_frame = ttk.LabelFrame(main_frame, text="時刻表示", padding=10)
        timestamp_frame.pack(fill=tk.X, pady=(0, 10))

        timestamp_check = ttk.Checkbutton(
            timestamp_frame,
            text="時刻を表示（ファイル作成時刻を基準）",
            variable=self.timelapse_timestamp_enabled_var,
            command=self._toggle_timelapse_timestamp_settings,
        )
        timestamp_check.pack(side=tk.LEFT)
        ttk.Label(timestamp_frame, text="位置:").pack(side=tk.LEFT, padx=(12, 3))
        self.timelapse_timestamp_position_box = ttk.Combobox(
            timestamp_frame,
            textvariable=self.timelapse_timestamp_position_var,
            values=("右下", "左下", "右上", "左上"),
            state="readonly",
            width=6,
        )
        self.timelapse_timestamp_position_box.pack(side=tk.LEFT)
        ttk.Label(timestamp_frame, text="文字サイズ:").pack(side=tk.LEFT, padx=(10, 3))
        self.timelapse_timestamp_size_spin = ttk.Spinbox(
            timestamp_frame,
            from_=0.8,
            to=4.0,
            increment=0.1,
            textvariable=self.timelapse_timestamp_size_var,
            width=4,
        )
        self.timelapse_timestamp_size_spin.pack(side=tk.LEFT)
        ttk.Label(timestamp_frame, text="%（画面高）").pack(side=tk.LEFT, padx=(3, 0))
        self._toggle_timelapse_timestamp_settings()

        fixed_pattern_frame = ttk.LabelFrame(
            main_frame, text="固定パターン補正", padding=10
        )
        fixed_pattern_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Checkbutton(
            fixed_pattern_frame,
            text="RTSP固定パターン補正マップをタイムラプスへ適用",
            variable=self.timelapse_fixed_pattern_enabled_var,
        ).pack(anchor=tk.W)
        fixed_pattern_available = bool(
            getattr(self.parent, "rtsp_dark_frame", None) is not None
            or os.path.exists(getattr(self.parent, "rtsp_dark_file", ""))
        )
        ttk.Label(
            fixed_pattern_frame,
            text=(
                "補正マップを利用できます"
                if fixed_pattern_available
                else "補正マップがありません。先にRTSP設定で作成してください。"
            ),
            foreground="gray",
        ).pack(anchor=tk.W, pady=(2, 0))

        annotation_frame = ttk.LabelFrame(
            main_frame,
            text="星空オーバーレイ（グリッド・星座線）",
            style="Section.TLabelframe",
        )
        annotation_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Checkbutton(
            annotation_frame,
            text="星空オーバーレイを有効にする（外部APIなし）",
            variable=self.timelapse_annotation_enabled_var,
            command=self._toggle_timelapse_annotation_settings,
        ).pack(anchor=tk.W)
        ttk.Label(
            annotation_frame,
            text="このカメラ用の補正データで、レンズの歪みと星の位置を合わせます。\n"
                 "初回の作成は通常より時間がかかります。",
            style="GlassMuted.TLabel",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(2, 4))
        overlay_row = ttk.Frame(annotation_frame)
        overlay_row.pack(fill=tk.X, pady=(0, 4))
        self.timelapse_annotation_overlay_button = ttk.Button(
            overlay_row, text="表示内容を設定…",
            command=self._choose_timelapse_annotation_overlays,
        )
        self.timelapse_annotation_overlay_button.pack(side=tk.LEFT)
        self.timelapse_annotation_overlay_label = ttk.Label(overlay_row, foreground="gray")
        self.timelapse_annotation_overlay_label.pack(side=tk.LEFT, padx=(8, 0))
        self._update_timelapse_annotation_overlay_summary()
        model_row = ttk.Frame(annotation_frame)
        model_row.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(model_row, text="このカメラ用補正データ:").pack(side=tk.LEFT)
        self.timelapse_annotation_model_combo = ttk.Combobox(
            model_row,
            textvariable=self.timelapse_annotation_model_var,
            state="readonly",
            width=42,
        )
        self.timelapse_annotation_model_combo.pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5)
        )
        self.timelapse_annotation_model_combo.bind(
            "<<ComboboxSelected>>", self._on_timelapse_annotation_model_selected
        )
        ttk.Button(
            model_row, text="一覧更新", command=self._refresh_timelapse_annotation_models
        ).pack(side=tk.LEFT)
        ttk.Label(
            annotation_frame,
            textvariable=self.timelapse_annotation_model_info_var,
            style="Hint.TLabel",
            wraplength=520,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(2, 0))
        ttk.Label(
            annotation_frame,
            text="補正データとは、このカメラのレンズの歪み・向き・星の配置を記録したものです。\n"
                 "選択したデータで、動画内の星座線を同じ空の位置に揃えます。",
            style="GlassMuted.TLabel",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(1, 4))

        # Keep the legacy file picker below the in-app selector.  It remains
        # available for older WCS files but is intentionally de-emphasized.
        calibration_row = ttk.Frame(annotation_frame, style="Glass.TFrame")
        calibration_row.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(
            calibration_row,
            text="旧形式の補正ファイル（互換用・通常は不要）:",
            style="GlassMuted.TLabel",
        ).pack(side=tk.LEFT)
        self.timelapse_annotation_calibration_entry = ttk.Entry(
            calibration_row,
            textvariable=self.timelapse_annotation_calibration_var,
        )
        self.timelapse_annotation_calibration_entry.pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5)
        )
        self.timelapse_annotation_calibration_button = ttk.Button(
            calibration_row,
            text="参照",
            command=self._choose_timelapse_annotation_calibration,
        )
        self.timelapse_annotation_calibration_button.pack(side=tk.LEFT)
        self._refresh_timelapse_annotation_models()
        reference_row = ttk.Frame(annotation_frame)
        reference_row.pack(fill=tk.X, pady=(5, 0))
        self.timelapse_annotation_reference_button = ttk.Button(
            reference_row,
            text="基準フレームを選択",
            command=self._choose_timelapse_annotation_reference_frame,
        )
        self.timelapse_annotation_reference_button.pack(side=tk.LEFT)
        self.timelapse_annotation_reference_label = ttk.Label(
            reference_row, text="自動（先頭動画）", foreground="gray"
        )
        self.timelapse_annotation_reference_label.pack(side=tk.LEFT, padx=(8, 0))
        self._toggle_timelapse_annotation_settings()

        model_frame = ttk.LabelFrame(
            main_frame,
            text="このカメラ用補正データを作る（任意）",
            style="Section.TLabelframe",
        )
        model_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(
            model_frame,
            text="指定した時間帯の動画から、レンズの歪みと星の位置を学習して登録します。\n"
                 "登録後は、上の補正データ欄からいつでも呼び出せます。",
            style="GlassMuted.TLabel", wraplength=520,
        ).pack(anchor=tk.W, pady=(0, 5))
        model_range = ttk.Frame(model_frame)
        model_range.pack(fill=tk.X)
        ttk.Label(model_range, text="開始:").pack(side=tk.LEFT)
        ttk.Entry(model_range, textvariable=self.timelapse_model_start_var, width=18).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Label(model_range, text="終了:").pack(side=tk.LEFT)
        ttk.Entry(model_range, textvariable=self.timelapse_model_end_var, width=18).pack(side=tk.LEFT, padx=4)
        ttk.Label(model_frame, text="動画は秒または時:分、フォルダは日時または時:分。空欄は全範囲。", foreground="gray").pack(anchor=tk.W, pady=(3, 5))
        ttk.Button(
            model_frame,
            text="この入力からモデルを作成",
            command=self._start_camera_model_from_timelapse,
        ).pack(anchor=tk.W)

        meteor_frame = ttk.LabelFrame(main_frame, text="流星検出動画", padding=10)
        meteor_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Checkbutton(
            meteor_frame,
            text="同じ時間帯に検出された流星動画を時刻位置へ挿入",
            variable=self.timelapse_insert_meteors_var,
        ).pack(anchor=tk.W)
        ttk.Label(
            meteor_frame,
            text="流星保存フォルダのフルサイズ動画を使用し、検出位置を半透明の黄色枠で表示します。",
            foreground="gray", wraplength=450,
        ).pack(anchor=tk.W, pady=(2, 0))

        mean_frame = ttk.LabelFrame(main_frame, text="ノイズ低減（時間平均）", padding=10)
        mean_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(
            mean_frame,
            text="各採用フレームの前後何枚を平均するか選びます。値を大きくするとノイズは減りますが、動くものは薄まります。",
            wraplength=450,
        ).pack(anchor=tk.W)

        mean_controls = ttk.Frame(mean_frame)
        mean_controls.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(mean_controls, text="前後:").pack(side=tk.LEFT)
        self.temporal_mean_scale = ttk.Scale(
            mean_controls,
            from_=0,
            to=100,
            variable=self.temporal_mean_radius_var,
            command=self._on_temporal_mean_scale,
        )
        self.temporal_mean_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 8))
        self.temporal_mean_spin = ttk.Spinbox(
            mean_controls,
            from_=0,
            to=100,
            increment=1,
            textvariable=self.temporal_mean_radius_var,
            width=4,
            command=self._update_temporal_mean_summary,
        )
        self.temporal_mean_spin.pack(side=tk.LEFT)
        ttk.Label(mean_controls, text="枚").pack(side=tk.LEFT, padx=(3, 0))

        ttk.Label(mean_frame, textvariable=self.temporal_mean_summary_var).pack(anchor=tk.W, pady=(4, 0))
        preset_frame = ttk.Frame(mean_frame)
        preset_frame.pack(anchor=tk.W, pady=(6, 0))
        ttk.Label(preset_frame, text="プリセット:").pack(side=tk.LEFT)
        for label, radius in (("なし", 0), ("軽め", 5), ("標準", 15), ("強め（今回と同じ）", 50)):
            ttk.Button(
                preset_frame,
                text=label,
                command=lambda value=radius: self._set_temporal_mean_radius(value),
            ).pack(side=tk.LEFT, padx=(5, 0))
        self.temporal_mean_spin.bind("<FocusOut>", lambda _event: self._update_temporal_mean_summary())
        self.temporal_mean_spin.bind("<Return>", lambda _event: self._update_temporal_mean_summary())
        self._update_temporal_mean_summary()
        
        # Keep the primary actions outside the scrollable canvas so they are
        # always available, even when the content is taller than the display.
        footer = ttk.Frame(self, style="GlassStrong.TFrame", padding=(15, 9))
        footer.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(
            footer,
            text="設定を確認したら作成開始",
            style="GlassMuted.TLabel",
        ).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(
            footer,
            text="キャンセル",
            style="Quiet.TButton",
            command=self.destroy,
        ).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(
            footer,
            text="タイムラプスを作成",
            style="Primary.TButton",
            command=self.start_creation,
        ).pack(side=tk.RIGHT)

        self._bind_timelapse_scroll_widgets(main_frame)
        self.after_idle(self._update_timelapse_scroll_region)

    def _update_timelapse_scroll_region(self, _event=None):
        region = self.timelapse_scroll_canvas.bbox("all")
        if region:
            self.timelapse_scroll_canvas.configure(scrollregion=region)

    def _resize_timelapse_scroll_content(self, event):
        self.timelapse_scroll_canvas.itemconfigure(
            self._timelapse_scroll_window, width=max(1, event.width)
        )
        self.after_idle(self._update_timelapse_scroll_region)

    def _bind_timelapse_scroll_widgets(self, widget):
        """Make wheel scrolling work over labels, fields, and list widgets."""
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            widget.bind(sequence, self._scroll_timelapse_window, add="+")
        for child in widget.winfo_children():
            self._bind_timelapse_scroll_widgets(child)

    def _refresh_timelapse_annotation_models(self):
        try:
            models = camera_model_catalog.discover_camera_models()
        except Exception as exc:
            models = []
            self.log_callback(f"カメラ補正データ一覧の取得に失敗しました: {exc}")
        self.timelapse_annotation_model_entries = models
        self.timelapse_annotation_model_by_display = {
            item["display_name"]: item for item in models
        }
        auto_label = "自動選択（撮影日に合う補正データ）"
        values = [auto_label] + [item["display_name"] for item in models]
        self.timelapse_annotation_model_combo.configure(values=values)
        parent_path = ""
        parent_var = getattr(self.parent, "plate_solve_model_path_var", None)
        if parent_var is not None:
            parent_path = parent_var.get().strip()
        selected_path = self.timelapse_annotation_model_path_var.get().strip() or parent_path
        selected = next(
            (item for item in models if os.path.abspath(item["path"]) == os.path.abspath(selected_path)),
            None,
        ) if selected_path else None
        if selected is not None:
            self.timelapse_annotation_model_path_var.set(selected["path"])
            self.timelapse_annotation_model_var.set(selected["display_name"])
        elif self.timelapse_annotation_model_var.get() not in values:
            self.timelapse_annotation_model_var.set(auto_label)
        self._update_timelapse_annotation_model_info()

    def _selected_timelapse_annotation_model(self):
        return self.timelapse_annotation_model_by_display.get(
            self.timelapse_annotation_model_var.get().strip()
        )

    def _update_timelapse_annotation_model_info(self):
        selected = self._selected_timelapse_annotation_model()
        self.timelapse_annotation_model_info_var.set(
            camera_model_catalog.format_model_details(selected)
        )

    def _on_timelapse_annotation_model_selected(self, _event=None):
        selected = self._selected_timelapse_annotation_model()
        self._update_timelapse_annotation_model_info()
        if selected is None:
            previous = self.timelapse_annotation_model_path_var.get().strip()
            self.timelapse_annotation_model_path_var.set("")
            if previous and os.path.abspath(self.timelapse_annotation_calibration_var.get().strip()) == os.path.abspath(previous):
                self.timelapse_annotation_calibration_var.set("")
            return
        self.timelapse_annotation_model_path_var.set(selected["path"])
        # The downstream timelapse creator already accepts one calibration
        # path.  Keep the selected model and legacy manual field synchronized
        # so the chosen model is used without another file dialog.
        self.timelapse_annotation_calibration_var.set(selected["path"])

    def _start_camera_model_from_timelapse(self):
        if not self.dropped_paths:
            messagebox.showwarning("高精度モデル", "先に動画またはフォルダを追加してください。", parent=self)
            return
        # A dropped folder is the most precise representation of a time range.
        # For several dropped files, use their common parent so minute-based
        # RTSP layouts remain selectable by the same builder.
        if len(self.dropped_paths) == 1:
            source = self.dropped_paths[0]
        else:
            source = os.path.commonpath([os.path.abspath(path) for path in self.dropped_paths])
        self.parent.camera_model_source_var.set(source)
        self.parent.camera_model_start_var.set(self.timelapse_model_start_var.get().strip())
        self.parent.camera_model_end_var.set(self.timelapse_model_end_var.get().strip())
        self.parent.start_camera_model_build()

    def _scroll_timelapse_window(self, event):
        # macOS emits MouseWheel delta; X11 uses Button-4/5.  Do not scroll
        # when the dialog content fits entirely in the current viewport.
        canvas = self.timelapse_scroll_canvas
        region = canvas.bbox("all")
        if not region or region[3] <= canvas.winfo_height():
            return
        if getattr(event, "num", None) == 4:
            amount = -1
        elif getattr(event, "num", None) == 5:
            amount = 1
        else:
            delta = getattr(event, "delta", 0)
            if not delta:
                return
            amount = -max(1, int(abs(delta) / 120)) if delta > 0 else max(1, int(abs(delta) / 120))
        canvas.yview_scroll(amount, "units")

    def _toggle_timelapse_timestamp_settings(self):
        state = tk.NORMAL if self.timelapse_timestamp_enabled_var.get() else tk.DISABLED
        self.timelapse_timestamp_position_box.config(state="readonly" if state == tk.NORMAL else tk.DISABLED)
        self.timelapse_timestamp_size_spin.config(state=state)

    def _toggle_timelapse_annotation_settings(self):
        state = tk.NORMAL if self.timelapse_annotation_enabled_var.get() else tk.DISABLED
        self.timelapse_annotation_calibration_entry.config(state=state)
        self.timelapse_annotation_calibration_button.config(state=state)
        self.timelapse_annotation_model_combo.config(
            state="readonly" if state == tk.NORMAL else "disabled"
        )
        self.timelapse_annotation_overlay_button.config(state=state)
        reference_state = state
        if state == tk.NORMAL and not self.dropped_paths:
            reference_state = tk.DISABLED
        self.timelapse_annotation_reference_button.config(state=reference_state)

    def _update_timelapse_annotation_overlay_summary(self):
        selected = []
        if self.timelapse_grid_var.get():
            selected.append("グリッド")
        if self.timelapse_constellations_var.get():
            selected.append("星座線")
        if self.timelapse_detected_stars_var.get():
            selected.append("検出星")
        self.timelapse_annotation_overlay_label.config(
            text="・".join(selected) if selected else "表示なし"
        )

    def _choose_timelapse_annotation_overlays(self):
        window = Toplevel(self)
        window.title("プレートソルブ注釈の表示設定")
        window.transient(self)
        window.grab_set()
        grid_var = tk.BooleanVar(master=window, value=self.timelapse_grid_var.get())
        constellation_var = tk.BooleanVar(
            master=window, value=self.timelapse_constellations_var.get()
        )
        detected_var = tk.BooleanVar(
            master=window, value=self.timelapse_detected_stars_var.get()
        )
        ttk.Label(
            window,
            text="タイムラプスへ重ねる表示を自由に選択してください。",
        ).pack(anchor=tk.W, padx=16, pady=(16, 8))
        ttk.Checkbutton(window, text="天球グリッドを表示", variable=grid_var).pack(
            anchor=tk.W, padx=16, pady=3
        )
        ttk.Checkbutton(
            window, text="星座線を表示（名称なし）", variable=constellation_var
        ).pack(anchor=tk.W, padx=16, pady=3)
        ttk.Checkbutton(
            window, text="実画像で検出した星を中空円で表示", variable=detected_var
        ).pack(anchor=tk.W, padx=16, pady=3)
        ttk.Label(
            window,
            text="検出星は星表ではなく、各出力フレーム上で実際に検出できた点だけです。",
            foreground="gray", wraplength=440,
        ).pack(anchor=tk.W, padx=16, pady=(5, 10))

        def apply_settings():
            self.timelapse_grid_var.set(grid_var.get())
            self.timelapse_constellations_var.set(constellation_var.get())
            self.timelapse_detected_stars_var.set(detected_var.get())
            self._update_timelapse_annotation_overlay_summary()
            window.destroy()

        controls = ttk.Frame(window)
        controls.pack(pady=(0, 14))
        ttk.Button(controls, text="適用", command=apply_settings).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="キャンセル", command=window.destroy).pack(side=tk.LEFT, padx=4)

    def _choose_timelapse_annotation_calibration(self):
        path = filedialog.askopenfilename(
            title="広角星空較正ファイルを選択",
            filetypes=[
                ("Calibration", "*.json *.wcs *.fits *.fit"),
                ("All Files", "*"),
            ],
        )
        if path:
            self.timelapse_annotation_calibration_var.set(path)

    def _timelapse_reference_samples(self):
        """Return the exact global frames that will be emitted by the timelapse."""
        images = []
        videos = []
        for path in self.dropped_paths:
            discovered_images, discovered_videos = timelapse_creator.get_files_from_path(path)
            images.extend(discovered_images)
            videos.extend(discovered_videos)
        images, videos = sorted(set(images)), sorted(set(videos))
        total, sources = timelapse_creator.count_total_frames(images, videos)
        if total <= 0:
            return None
        samples = timelapse_creator.calculate_sample_indices(total, self.duration_var.get())
        loader = timelapse_creator.FrameLoader(sources)
        target_size = None
        for global_index in samples:
            frame = loader.load_frame(global_index, (0, 0))
            if frame is not None:
                target_size = (frame.shape[1], frame.shape[0])
                break
        if target_size is None:
            loader.cleanup()
            return None
        sources = list(loader.sources)
        loader.cleanup()
        return sources, samples, target_size

    def _choose_timelapse_annotation_reference_frame(self):
        reference_data = self._timelapse_reference_samples()
        if reference_data is None:
            messagebox.showinfo(
                "基準フレーム", "タイムラプスに使えるフレームがありません。入力を確認してください。"
            )
            return
        sources, sample_indices, target_size = reference_data
        window = Toplevel(self)
        window.title("プレートソルブの基準フレーム（タイムラプス採用フレーム）")
        window.transient(self)
        window.grab_set()
        position_var = tk.IntVar(master=window, value=min(
            max(0, self.timelapse_annotation_reference_sample_var.get()), len(sample_indices) - 1
        ))
        ttk.Label(
            window,
            text="タイムラプス出力に実際に採用されるフレームを選びます。\n"
                 "プレビューは元フレームです。プレートソルブ時だけ前後を時間平均します。",
        ).pack(anchor=tk.W, padx=12, pady=(12, 2))
        preview = ttk.Label(window, text="プレビューを読み込み中…")
        preview.pack(padx=12, pady=10)
        info = ttk.Label(window, foreground="gray")
        info.pack(anchor=tk.W, padx=12)
        scale = ttk.Scale(window, from_=0, to=max(0, len(sample_indices) - 1), variable=position_var)
        scale.pack(fill=tk.X, padx=12, pady=(4, 10))
        image_ref = {"image": None}
        radius = max(0, min(100, int(self.temporal_mean_radius_var.get())))
        preview_frames = {}
        preview_metadata = {}
        preview_events = queue.Queue()
        stop_preview = threading.Event()
        scale.state(["disabled"])

        def show_frame(*_args):
            position = max(0, min(len(sample_indices) - 1, int(position_var.get())))
            global_index = sample_indices[position]
            frame = preview_frames.get(position)
            if frame is None:
                preview.config(text="時間平均プレビューを準備中…")
                return
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            image.thumbnail((640, 360))
            image_ref["image"] = ImageTk.PhotoImage(image)
            preview.config(image=image_ref["image"], text="")
            timestamp, source_name = preview_metadata.get(
                position, (datetime.now(), "不明")
            )
            capture_text = timestamp.strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
            info.config(
                text=f"採用フレーム {position + 1}/{len(sample_indices)}  "
                     f"プレビュー: 元フレーム（較正時は前後{radius}枚を平均）\n"
                     f"{source_name}  {capture_text}"
            )

        scale.configure(command=lambda _value: show_frame())

        def build_previews():
            try:
                worker_count = min(4, max(1, os.cpu_count() or 1), len(sample_indices))

                def load_preview(position, global_index):
                    # Each worker owns its decoder, so OpenCV can decode several
                    # sampled source frames concurrently without its capture lock.
                    if stop_preview.is_set():
                        return position, None, None
                    preview_loader = timelapse_creator.FrameLoader(sources)
                    try:
                        frame = preview_loader.load_frame(global_index, target_size)
                        if frame is None:
                            return position, None, None
                        thumbnail = cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA)
                        source = preview_loader._get_source_for_index(global_index)
                        return position, thumbnail, (
                            preview_loader.timestamp_for_index(global_index),
                            Path(source[0]).name if source else "不明",
                        )
                    finally:
                        preview_loader.cleanup()

                completed = 0
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    futures = [
                        executor.submit(load_preview, position, global_index)
                        for position, global_index in enumerate(sample_indices)
                    ]
                    for future in as_completed(futures):
                        if stop_preview.is_set():
                            break
                        position, thumbnail, metadata = future.result()
                        if thumbnail is not None:
                            preview_frames[position] = thumbnail
                            preview_metadata[position] = metadata
                        completed += 1
                        if completed % 8 == 0 or completed == len(sample_indices):
                            preview_events.put(("progress", completed))
                preview_events.put(("done", None))
            except Exception as exc:
                preview_events.put(("error", str(exc)))

        def poll_preview_events():
            if not window.winfo_exists():
                return
            try:
                while True:
                    kind, value = preview_events.get_nowait()
                    if kind == "progress":
                        info.config(text=f"時間平均プレビューを準備中: {value}/{len(sample_indices)}")
                    elif kind == "done":
                        scale.state(["!disabled"])
                        show_frame()
                    elif kind == "error":
                        preview.config(text=f"プレビュー作成に失敗しました: {value}")
            except queue.Empty:
                pass
            if not stop_preview.is_set():
                window.after(50, poll_preview_events)

        threading.Thread(target=build_previews, daemon=True).start()
        window.after(50, poll_preview_events)

        def confirm():
            position = max(0, min(len(sample_indices) - 1, int(position_var.get())))
            global_index = sample_indices[position]
            self.timelapse_annotation_reference_sample_var.set(global_index)
            self.timelapse_annotation_reference_selected_var.set(True)
            self.timelapse_annotation_reference_label.config(
                text=f"採用フレーム {position + 1}/{len(sample_indices)} を選択"
            )
            close_window()
            self.after(10, self._choose_timelapse_annotation_overlays)

        def close_window():
            stop_preview.set()
            window.destroy()

        controls = ttk.Frame(window)
        controls.pack(pady=(0, 12))
        ttk.Button(controls, text="このフレームを使う", command=confirm).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="キャンセル", command=close_window).pack(side=tk.LEFT, padx=4)
        window.protocol("WM_DELETE_WINDOW", close_window)

    def _on_temporal_mean_scale(self, value):
        """Scale is continuous, so convert its value to a whole frame count."""
        self._set_temporal_mean_radius(int(round(float(value))))

    def _set_temporal_mean_radius(self, value):
        radius = max(0, min(100, int(value)))
        if self.temporal_mean_radius_var.get() != radius:
            self.temporal_mean_radius_var.set(radius)
        self._update_temporal_mean_summary()

    def _update_temporal_mean_summary(self):
        try:
            radius = int(self.temporal_mean_radius_var.get())
        except (TypeError, ValueError, tk.TclError):
            radius = config.TIMELAPSE_TEMPORAL_MEAN_RADIUS_FRAMES
        radius = max(0, min(100, radius))
        if self.temporal_mean_radius_var.get() != radius:
            self.temporal_mean_radius_var.set(radius)
        if radius == 0:
            summary = "単独フレームを使用します（平均しません）"
        else:
            summary = f"前後{radius}枚を含む、合計最大{radius * 2 + 1}枚の平均画像を使用します"
        self.temporal_mean_summary_var.set(summary)
    
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
        self._toggle_timelapse_annotation_settings()
    
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
        self._toggle_timelapse_annotation_settings()
    
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
        default_output = timelapse_creator.get_default_output_path(self.dropped_paths)
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
        try:
            timestamp_size = float(self.timelapse_timestamp_size_var.get())
        except (TypeError, ValueError):
            timestamp_size = config.TIMELAPSE_TIMESTAMP_SIZE_PERCENT
        timestamp_settings = {
            "enabled": self.timelapse_timestamp_enabled_var.get(),
            "position": self.timelapse_timestamp_position_var.get(),
            "size_percent": timestamp_size,
        }
        annotation_settings = {
            "enabled": self.timelapse_annotation_enabled_var.get(),
            "calibration_path": self.timelapse_annotation_calibration_var.get().strip() or None,
            "draw_grid": self.timelapse_grid_var.get(),
            "draw_constellations": self.timelapse_constellations_var.get(),
            "draw_detected_stars": self.timelapse_detected_stars_var.get(),
            # Keep only a short detection dropout bridged; longer cloudy
            # intervals remain without constellation lines.
            "constellation_temporal_hold_frames": 3,
            "reference_sample_index": self.timelapse_annotation_reference_sample_var.get(),
            "reference_selected": self.timelapse_annotation_reference_selected_var.get(),
        }
        meteor_path_var = getattr(self.parent, "meteor_save_path_var", None)
        meteor_folder = (
            meteor_path_var.get() if meteor_path_var is not None
            else config.DEFAULT_METEOR_SAVE_PATH
        )
        meteor_insert_settings = {
            "enabled": self.timelapse_insert_meteors_var.get(),
            "meteor_folder": meteor_folder,
        }
        try:
            temporal_mean_radius = int(self.temporal_mean_radius_var.get())
        except (TypeError, ValueError, tk.TclError):
            temporal_mean_radius = config.TIMELAPSE_TEMPORAL_MEAN_RADIUS_FRAMES
        temporal_mean_radius = max(0, min(100, temporal_mean_radius))
        fixed_pattern_correction = None
        if self.timelapse_fixed_pattern_enabled_var.get():
            fixed_pattern_correction = getattr(self.parent, "rtsp_dark_frame", None)
            if fixed_pattern_correction is None:
                loader = getattr(self.parent, "load_rtsp_dark_frame", None)
                if callable(loader):
                    loader()
                fixed_pattern_correction = getattr(
                    self.parent, "rtsp_dark_frame", None
                )
            if fixed_pattern_correction is None:
                messagebox.showerror(
                    "固定パターン補正",
                    "補正マップを読み込めません。RTSP設定で補正マップを作成してください。",
                )
                return
        
        # ウィンドウを閉じる
        self.destroy()
        
        mask_status = "あり" if mask is not None else "なし"
        annotation_status = "あり（ローカル）" if annotation_settings["enabled"] else "なし"
        fixed_pattern_status = (
            "あり" if fixed_pattern_correction is not None else "なし"
        )
        self.log_callback(
            f"タイムラプス作成を開始します... (長さ: {duration}秒, "
            f"{len(paths)}個のアイテム, マスク: {mask_status}, "
            f"固定パターン補正: {fixed_pattern_status}, 星空注釈: {annotation_status})"
        )
        
        def create_task(progress_callback):
            return timelapse_creator.create_timelapse(
                paths,
                output_path,
                target_duration_seconds=duration,
                progress_callback=progress_callback,
                mask=mask,
                timestamp_settings=timestamp_settings,
                temporal_mean_radius_frames=temporal_mean_radius,
                annotation_settings=annotation_settings,
                meteor_insert_settings=meteor_insert_settings,
                fixed_pattern_correction=fixed_pattern_correction,
            )

        task_runner = getattr(self.parent, "_run_synthesis_task_async", None)
        if not callable(task_runner):
            messagebox.showerror("エラー", "タイムラプス作成機能を開始できませんでした。")
            return
        task_runner(
            create_task,
            output_path=output_path,
            item_label="タイムラプス動画",
        )


class ProcessingOptionDialog(tk.Toplevel):
    def __init__(self, parent):
        print("DEBUG: ProcessingOptionDialog initialized")
        super().__init__(parent)
        self.title("処理オプション")
        self.result = None
        self.geometry("500x360") 
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
        ttk.Label(warning_frame, text="※AI検出を選択時、VRAMが7GB未満の場合は動作が非常に遅くなる可能性があります。", font=("", 9), foreground="#FFD700").pack(anchor=tk.W)
        
        btn_frame = ttk.Frame(self.main_frame)
        btn_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        ttk.Button(btn_frame, text="キャンセル", command=self.destroy).pack(side=tk.LEFT, padx=5)
        self.next_btn = ttk.Button(btn_frame, text="確定", command=self.on_ok, state=tk.NORMAL) # 最初から有効化
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


