from gui_common import *
import ui_state
from fixed_pattern import apply_fixed_pattern_correction


class SourceMixin:
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
        self._create_processing_priority_section(scrollable_frame)
        
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

        lf_rtsp = ttk.LabelFrame(scrollable_frame, text="RTSPストリーム")
        lf_rtsp.pack(fill=tk.X, expand=True, pady=5)

        rtsp_intro = ttk.Frame(lf_rtsp)
        rtsp_intro.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(
            rtsp_intro,
            text="カメラを追加して、録画・解析・固定パターン補正をこの場所で管理します。",
            style="Hint.TLabel",
        ).pack(side=tk.LEFT)
        rtsp_info_label = ttk.Label(rtsp_intro, text="ⓘ", font=("Arial", 11), foreground="#87CEEB", cursor="hand2")
        rtsp_info_label.pack(side=tk.LEFT, padx=(6, 0))
        
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
        
        rtsp_connect = ttk.LabelFrame(lf_rtsp, text="接続を追加", style="Section.TLabelframe")
        rtsp_connect.pack(fill=tk.X, pady=(0, 6))
        rtsp_connect.columnconfigure(1, weight=1)
        ttk.Label(rtsp_connect, text="RTSP URL").grid(row=0, column=0, sticky=tk.W, padx=(0, 7))
        self.rtsp_url_entry = ttk.Entry(rtsp_connect, textvariable=self.rtsp_url_var)
        self.rtsp_url_entry.grid(row=0, column=1, sticky=tk.EW)
        ttk.Label(rtsp_connect, text="FPS").grid(row=0, column=2, sticky=tk.W, padx=(10, 5))
        fps_spin = ttk.Spinbox(rtsp_connect, from_=1, to=120, increment=1, width=5, textvariable=self.rtsp_fps_var)
        fps_spin.grid(row=0, column=3, sticky=tk.W)
        self.btn_add_rtsp = ttk.Button(rtsp_connect, text="ストリームを追加", style="Primary.TButton", command=self.add_rtsp_url)
        self.btn_add_rtsp.grid(row=0, column=4, sticky=tk.E, padx=(10, 0))

        rtsp_notification_frame = ttk.Frame(rtsp_connect)
        rtsp_notification_frame.grid(row=1, column=0, columnspan=5, sticky=tk.W, pady=(7, 0))
        ttk.Checkbutton(
            rtsp_notification_frame,
            text="流星検出時に通知音を鳴らす",
            variable=self.rtsp_notification_sound_var,
        ).pack(side=tk.LEFT)
        ttk.Button(
            rtsp_notification_frame,
            text="通知を試す",
            style="Quiet.TButton",
            command=utils.play_notification_sound,
        ).pack(side=tk.LEFT, padx=(8, 0))
        
        rtsp_list_section = ttk.LabelFrame(lf_rtsp, text="接続済みストリーム", style="Section.TLabelframe")
        rtsp_list_section.pack(fill=tk.X, expand=True, pady=(0, 6))
        rtsp_list_container = ttk.Frame(rtsp_list_section)
        rtsp_list_container.pack(fill=tk.BOTH, expand=True)
        
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
        
        rtsp_list_actions = ttk.Frame(rtsp_list_section)
        rtsp_list_actions.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(rtsp_list_actions, text="選択したストリーム:").pack(side=tk.LEFT)
        ttk.Button(rtsp_list_actions, text="削除", style="Quiet.TButton", command=self.remove_selected_rtsp).pack(side=tk.LEFT, padx=(7, 4))
        ttk.Button(rtsp_list_actions, text="すべて削除", style="Quiet.TButton", command=self.remove_all_rtsp).pack(side=tk.LEFT)

        rtsp_tools = ttk.Frame(lf_rtsp)
        rtsp_tools.pack(fill=tk.X, pady=(0, 6))
        rtsp_tools.columnconfigure(0, weight=1)
        rtsp_tools.columnconfigure(1, weight=1)

        rtsp_analysis = ttk.LabelFrame(rtsp_tools, text="解析ツール", style="Section.TLabelframe")
        rtsp_analysis.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, 3))
        ttk.Label(rtsp_analysis, text="カメラ映像から解析用の設定を作成します。", style="Hint.TLabel").pack(anchor=tk.W, pady=(0, 6))
        analysis_buttons = ttk.Frame(rtsp_analysis)
        analysis_buttons.pack(anchor=tk.W)
        self.btn_rtsp_plate_solve = ttk.Button(analysis_buttons, text="プレートソルブ", command=self.start_rtsp_plate_solve)
        self.btn_rtsp_plate_solve.pack(side=tk.LEFT, padx=(0, 5))
        self.btn_rtsp_mask = ttk.Button(analysis_buttons, text="マスク作成", command=self.create_rtsp_mask)
        self.btn_rtsp_mask.pack(side=tk.LEFT)

        rtsp_pattern = ttk.LabelFrame(rtsp_tools, text="固定パターン補正", style="Section.TLabelframe")
        rtsp_pattern.grid(row=0, column=1, sticky=tk.NSEW, padx=(3, 0))
        ttk.Label(
            rtsp_pattern,
            text="補正マップを作成し、保存する全動画・画像へ適応補正＋21フレーム平均を適用します。",
            style="Hint.TLabel",
        ).pack(anchor=tk.W, pady=(0, 6))
        rtsp_pattern_buttons = ttk.Frame(rtsp_pattern)
        rtsp_pattern_buttons.pack(anchor=tk.W)
        self.btn_rtsp_dark_capture = ttk.Button(rtsp_pattern_buttons, text="撮影（30秒）", style="Primary.TButton", command=self.capture_rtsp_dark_frame)
        self.btn_rtsp_dark_capture.pack(side=tk.LEFT, padx=(0, 5))
        self.btn_rtsp_dark_video = ttk.Button(rtsp_pattern_buttons, text="保存動画から作成", command=self.load_rtsp_dark_video)
        self.btn_rtsp_dark_video.pack(side=tk.LEFT, padx=(0, 8))
        self.chk_apply_rtsp_dark = ttk.Checkbutton(
            rtsp_pattern_buttons,
            text="保存物へ適用",
            variable=self.apply_rtsp_dark_var,
            command=self.on_apply_rtsp_dark_changed,
        )
        self.chk_apply_rtsp_dark.pack(side=tk.LEFT)
        ttk.Label(rtsp_pattern, textvariable=self.rtsp_dark_status_var, style="Hint.TLabel", anchor=tk.W).pack(fill=tk.X, pady=(6, 0))
        self.load_rtsp_dark_frame()
        
        rtsp_time_frame = ttk.LabelFrame(lf_rtsp, text="録画スケジュール", style="Section.TLabelframe")
        rtsp_time_frame.pack(fill=tk.X)
        
        rtsp_time_row1 = ttk.Frame(rtsp_time_frame)
        rtsp_time_row1.pack(fill=tk.X)
        ttk.Checkbutton(rtsp_time_row1, text="時間帯を指定して録画する", variable=self.rtsp_time_limit_var, command=self.toggle_rtsp_time_limit_frame).pack(side=tk.LEFT, anchor=tk.W)
        ttk.Button(rtsp_time_row1, text="日没・日出から設定", style="Quiet.TButton", command=self.fetch_current_location_rtsp).pack(side=tk.LEFT, padx=(10,0))
        
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
        ttk.Label(self.rtsp_time_limit_detail_frame, text="録画時間外でも、すでに保存済みの動画の解析は継続します。", style="Hint.TLabel").pack(anchor=tk.W, pady=(4,0))
        
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

    def _create_processing_priority_section(self, parent):
        """処理入力の優先順位を、ドラッグ操作で並べ替えるUI。"""
        frame = ttk.LabelFrame(parent, text="処理対象の優先順位")
        frame.pack(fill=tk.X, expand=True, pady=(0, 5))
        tk.Label(
            frame,
            text="カードをドラッグして並べ替えます。複数の入力が有効な場合は、最上位の1種類だけを実行します。",
            bg="#2E3F5B",
            fg="#A9C9EF",
            anchor=tk.W,
        ).pack(fill=tk.X, padx=8, pady=(6, 2))

        self.processing_priority_cards_frame = tk.Frame(frame, bg="#2E3F5B")
        self.processing_priority_cards_frame.pack(fill=tk.X, padx=8, pady=(2, 6))
        self._processing_priority_drag_key = None
        self._processing_priority_drop_key = None
        self._processing_priority_cards = {}

        controls = ttk.Frame(frame)
        controls.pack(fill=tk.X, padx=8, pady=(0, 6))
        ttk.Button(controls, text="標準に戻す", command=self.reset_processing_source_priority).pack(side=tk.RIGHT)
        self._refresh_processing_priority_cards()

    def set_processing_source_priority(self, priority):
        self.processing_source_priority = ui_state.normalize_source_priority(priority)
        if getattr(self, "processing_priority_cards_frame", None) is not None:
            self._refresh_processing_priority_cards()

    def reset_processing_source_priority(self):
        self.set_processing_source_priority(ui_state.SOURCE_PRIORITY_DEFAULT)

    def _refresh_processing_priority_cards(self):
        container = self.processing_priority_cards_frame
        for child in container.winfo_children():
            child.destroy()
        self._processing_priority_cards = {}

        labels = {
            "periodic": "定期スキャン",
            "rtsp": "RTSPストリーム",
            "folder": "フォルダ／動画ファイル",
        }
        for index, source_type in enumerate(self.processing_source_priority, start=1):
            card = tk.Frame(
                container, bg="#3A4D6B", highlightbackground="#6E91BF",
                highlightthickness=1, cursor="fleur", pady=6,
            )
            card.pack(fill=tk.X, pady=2)
            rank = tk.Label(
                card, text=str(index), bg="#5476A8", fg="#FFFFFF", width=3,
                font=("Segoe UI", 10, "bold"), cursor="fleur",
            )
            rank.pack(side=tk.LEFT, padx=(6, 8))
            handle = tk.Label(card, text="⠿", bg="#3A4D6B", fg="#A9C9EF", font=("Segoe UI", 14), cursor="fleur")
            handle.pack(side=tk.LEFT, padx=(0, 8))
            text = tk.Label(
                card, text=labels[source_type], bg="#3A4D6B", fg="#F0F4FA",
                font=("Segoe UI", 10, "bold"), anchor=tk.W, cursor="fleur",
            )
            text.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self._processing_priority_cards[source_type] = (card, rank, handle, text)
            for widget in (card, rank, handle, text):
                widget.bind("<ButtonPress-1>", lambda event, key=source_type: self._on_processing_priority_press(key, event))
                widget.bind("<B1-Motion>", self._on_processing_priority_motion)
                widget.bind("<ButtonRelease-1>", self._on_processing_priority_release)

    def _set_processing_priority_card_highlight(self, source_type, highlighted):
        widgets = self._processing_priority_cards.get(source_type)
        if not widgets:
            return
        card, rank, handle, text = widgets
        card_color = "#597EAE" if highlighted else "#3A4D6B"
        for widget in (card, handle, text):
            widget.configure(bg=card_color)
        card.configure(highlightbackground="#9AC4FF" if highlighted else "#6E91BF")
        rank.configure(bg="#6E9DDD" if highlighted else "#5476A8")

    def _on_processing_priority_press(self, source_type, _event):
        self._processing_priority_drag_key = source_type
        self._processing_priority_drop_key = source_type
        self._set_processing_priority_card_highlight(source_type, True)

    def _on_processing_priority_motion(self, event):
        drag_key = self._processing_priority_drag_key
        if not drag_key:
            return
        pointer_y = event.y_root
        target_key = drag_key
        for source_type in self.processing_source_priority:
            card = self._processing_priority_cards[source_type][0]
            midpoint = card.winfo_rooty() + card.winfo_height() / 2
            if pointer_y < midpoint:
                target_key = source_type
                break
            target_key = source_type
        if target_key != self._processing_priority_drop_key:
            self._set_processing_priority_card_highlight(self._processing_priority_drop_key, False)
            self._processing_priority_drop_key = target_key
            self._set_processing_priority_card_highlight(target_key, True)

    def _on_processing_priority_release(self, _event):
        drag_key = self._processing_priority_drag_key
        target_key = self._processing_priority_drop_key
        if drag_key and target_key and drag_key != target_key:
            order = list(self.processing_source_priority)
            order.pop(order.index(drag_key))
            order.insert(order.index(target_key), drag_key)
            self.set_processing_source_priority(order)
        self._processing_priority_drag_key = None
        self._processing_priority_drop_key = None

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
        self._update_live_preview_button_state()

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

    def _current_rtsp_url_for_dark(self):
        url = self.rtsp_url_var.get().strip()
        if url:
            return url
        if self.rtsp_selected_indices:
            selected_index = min(self.rtsp_selected_indices)
            if 0 <= selected_index < len(self.rtsp_urls):
                return self.rtsp_urls[selected_index]
        if self.rtsp_urls:
            return self.rtsp_urls[0]
        return ""

    def load_rtsp_dark_frame(self):
        try:
            if os.path.exists(self.rtsp_dark_file):
                with np.load(self.rtsp_dark_file, allow_pickle=False) as data:
                    if "fixed_correction" in data:
                        dark = data["fixed_correction"].astype(np.int16)
                        method = "固定補正"
                    else:
                        dark = data["dark_frame"].astype(np.uint8)
                        method = "旧ダーク"
                if dark.ndim in (2, 3):
                    self.rtsp_dark_frame = dark
                    self.rtsp_dark_status_var.set(f"{method}: 読込済み {dark.shape[1]}x{dark.shape[0]}")
                    return True
            self.rtsp_dark_frame = None
            self.rtsp_dark_status_var.set("ダーク: 未撮影")
            return False
        except Exception as e:
            self.rtsp_dark_frame = None
            self.rtsp_dark_status_var.set("ダーク: 読込失敗")
            self.append_log(f"RTSPダーク読み込み失敗: {e}")
            return False

    def on_apply_rtsp_dark_changed(self):
        if self.apply_rtsp_dark_var.get() and self.rtsp_dark_frame is None:
            if not self.load_rtsp_dark_frame():
                self.apply_rtsp_dark_var.set(False)
                messagebox.showwarning("固定パターン補正", "適用できる補正マップがありません。先に補正マップを作成してください。")
                return
        state = "ON" if self.apply_rtsp_dark_var.get() else "OFF"
        self.append_log(f"保存物への適応固定パターン＋21フレーム平均: {state}")

    def get_active_rtsp_dark_frame(self):
        if not self.apply_rtsp_dark_var.get():
            return None
        if self.rtsp_dark_frame is None:
            self.load_rtsp_dark_frame()
        return self.rtsp_dark_frame

    def apply_rtsp_dark_to_frame(self, frame):
        dark = self.get_active_rtsp_dark_frame()
        if dark is None or frame is None:
            return frame
        if dark.shape[:2] != frame.shape[:2]:
            dark = cv2.resize(dark, (frame.shape[1], frame.shape[0]))
        return apply_fixed_pattern_correction(frame, dark)

    def capture_rtsp_dark_frame(self):
        rtsp_url = self._current_rtsp_url_for_dark()
        if not rtsp_url:
            messagebox.showwarning("RTSPダーク", "RTSP URLを入力または追加してください。")
            return
        if not messagebox.askokcancel(
            "固定パターン撮影",
            "レンズにキャップをするなど、光が入らない状態にしてからOKを押してください。\n"
            "30秒間の動画を保存し、90枚の明るさ正規化中央値から固定パターンを作成します。\n"
            "滑らかな光漏れの勾配は補正対象から除外します。"
        ):
            return

        base_dir = os.path.dirname(self.rtsp_dark_file)
        capture_dir = os.path.join(base_dir, "rtsp_fixed_pattern_captures")
        filename = f"rtsp_fixed_pattern_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        command = [
            sys.executable, os.path.join(base_dir, "rtsp_dark_capture_worker.py"),
            "--url", rtsp_url, "--output", self.rtsp_dark_file,
            "--preview", self.rtsp_dark_preview_file,
            "--video", os.path.join(capture_dir, filename),
            "--duration", "30", "--samples", "90",
        ]
        self._start_rtsp_fixed_pattern_worker(command, "RTSP固定パターン撮影")

    def load_rtsp_dark_video(self):
        video_path = filedialog.askopenfilename(
            title="保存済み固定パターン動画を選択",
            filetypes=[("動画ファイル", "*.mp4 *.mov *.mkv *.avi *.m4v"), ("すべてのファイル", "*.*")],
        )
        if not video_path:
            return
        command = [
            sys.executable, os.path.join(os.path.dirname(self.rtsp_dark_file), "rtsp_dark_capture_worker.py"),
            "--input-video", video_path, "--output", self.rtsp_dark_file,
            "--preview", self.rtsp_dark_preview_file, "--samples", "90",
        ]
        self._start_rtsp_fixed_pattern_worker(command, "保存動画から固定パターン作成")

    def _start_rtsp_fixed_pattern_worker(self, command, job_name):
        self.btn_rtsp_dark_capture.config(state=tk.DISABLED)
        if hasattr(self, "btn_rtsp_dark_video"):
            self.btn_rtsp_dark_video.config(state=tk.DISABLED)
        self.rtsp_dark_status_var.set("固定補正: 作成中...")
        self.append_log(f"{job_name}開始")
        dark_queue = queue.Queue()

        def worker():
            error = None
            result_line = ""
            dark_frame = None
            try:
                proc = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

                start_time = time.time()
                while True:
                    line = proc.stdout.readline() if proc.stdout is not None else ""
                    if line:
                        line = line.strip()
                        if line.startswith("PROGRESS "):
                            _tag, captured, total = line.split()[:3]
                            dark_queue.put(("progress", captured, total))
                        elif line.startswith("CAPTURE "):
                            _tag, elapsed, total, captured, sample_total = line.split()[:5]
                            dark_queue.put(("capture", elapsed, total, captured, sample_total))
                        elif line.startswith("STATUS "):
                            message = line[len("STATUS "):]
                            dark_queue.put(("status", message))
                        elif line.startswith("RESULT "):
                            result_line = line
                            dark_queue.put(("result", line[len("RESULT "):]))
                        elif line.startswith("ERROR "):
                            error = RuntimeError(line[len("ERROR "):])
                        else:
                            dark_queue.put(("status", line))

                    if proc.poll() is not None:
                        break
                    if time.time() - start_time > 75:
                        proc.kill()
                        raise RuntimeError("固定パターン撮影がタイムアウトしました。RTSP接続を確認してください。")

                return_code = proc.wait()
                if return_code != 0:
                    raise error or RuntimeError(f"ダーク撮影ワーカーが失敗しました (exit={return_code})")

                dark_frame = None
                with np.load(self.rtsp_dark_file, allow_pickle=False) as data:
                    dark_frame = data["fixed_correction"].astype(np.int16)
            except Exception as exc:
                error = exc
                dark_frame = None

            dark_queue.put(("done", error, dark_frame, result_line[len("RESULT "):] if result_line else ""))

        def poll_dark_queue():
            finished = False
            while True:
                try:
                    item = dark_queue.get_nowait()
                except queue.Empty:
                    break

                kind = item[0]
                if kind == "progress":
                    _, captured, total = item
                    self.rtsp_dark_status_var.set(f"固定補正: 解析中 {captured}/{total}")
                    self.append_log(f"固定パターン解析中: {captured}/{total}")
                elif kind == "capture":
                    _, elapsed, total, captured, sample_total = item
                    self.rtsp_dark_status_var.set(f"固定補正: 録画中 {elapsed}/{total}秒 ({captured}/{sample_total})")
                elif kind == "status":
                    self.append_log(f"RTSPダーク: {item[1]}")
                elif kind == "result":
                    self.append_log(f"RTSPダーク撮影完了情報: {item[1]}")
                elif kind == "done":
                    _, error, dark_frame, result_summary = item
                    self.btn_rtsp_dark_capture.config(state=tk.NORMAL)
                    if hasattr(self, "btn_rtsp_dark_video"):
                        self.btn_rtsp_dark_video.config(state=tk.NORMAL)
                    if error is not None or dark_frame is None:
                        self.rtsp_dark_status_var.set("固定補正: 作成失敗")
                        self.append_log(f"固定パターン作成失敗: {error}")
                        messagebox.showerror("固定パターン", f"固定パターン作成に失敗しました:\n{error}")
                    else:
                        self.rtsp_dark_frame = dark_frame
                        self.apply_rtsp_dark_var.set(True)
                        self.rtsp_dark_status_var.set(f"固定補正: 作成済み {dark_frame.shape[1]}x{dark_frame.shape[0]}")
                        self.append_log(f"固定パターン作成完了: {self.rtsp_dark_file}")
                        if result_summary:
                            self.append_log(result_summary)
                        messagebox.showinfo("固定パターン", "固定パターンを作成し、適用をONにしました。保存動画は後日再利用できます。")
                    finished = True

            if not finished:
                self.after(100, poll_dark_queue)

        threading.Thread(target=worker, daemon=True).start()
        self.after(100, poll_dark_queue)

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
