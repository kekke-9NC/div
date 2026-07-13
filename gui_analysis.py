from gui_common import *


class AnalysisMixin:
    def create_analysis_tab(self, parent):
        """Create the '解析' tab where users can drop meteor info .txt files and run batch drawing."""
        # The tab is taller than many laptop displays.  Previously only the
        # file lists scrolled, leaving the video-concatenation controls below
        # the visible area with no way to reach them.
        tab_frame = ttk.Frame(parent)
        tab_frame.pack(fill=tk.BOTH, expand=True)
        self.analysis_tab_canvas = tk.Canvas(tab_frame, highlightthickness=0, bg="#2E3F5B")
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
        self.btn_camera_control = ttk.Button(row4, text="カメラコントロール", command=self.open_camera_control, style="Gray.TButton")
        self.btn_camera_control.pack(side=tk.LEFT, padx=(0, 5))
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
        apply_enhancement = self.video_concat_enhancement_var.get()
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
        self.append_log(
            f"設定: ビットレート={bitrate}, コーデック={codec}, FPS={fps_str}, "
            f"保存物補正={'ON' if apply_enhancement else 'OFF'}"
        )
        
        thread = threading.Thread(
            target=self._video_concat_worker,
            args=(files, output_path, bitrate, codec, fps_val, safe_mode, apply_enhancement),
            daemon=True
        )
        thread.start()

    def _video_concat_worker(self, files, output_path, bitrate, codec, fps, safe_mode, apply_enhancement):
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
                cancel_check=cancel_check,
                apply_enhancement=apply_enhancement,
                fixed_pattern_path=self.rtsp_dark_file,
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
