from gui_common import *

UI_BG = ui_theme.COLORS["content_raised"]
UI_FIELD = ui_theme.COLORS["field"]
UI_GLASS = ui_theme.COLORS["glass"]
UI_TEXT = ui_theme.COLORS["text"]
UI_MUTED = ui_theme.COLORS["text_secondary"]
UI_CYAN = ui_theme.COLORS["cyan"]
UI_BORDER = ui_theme.COLORS["border"]


class SettingsMixin:
    def create_settings_tab(self, parent):
        frame = ttk.Frame(parent)

        # スクロール可能なキャンバスとスクロールバーを作成
        canvas = tk.Canvas(frame, highlightthickness=0, bg=UI_BG)
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
            if key == 'full_video':
                # フルサイズ動画の時刻表示は、対象チェックボックスの直下にまとめる。
                timestamp_frame = ttk.Frame(lf_save)
                timestamp_frame.pack(fill=tk.X, padx=(28, 0), pady=(0, 2))
                self.full_video_timestamp_settings_widgets = []

                timestamp_check = ttk.Checkbutton(
                    timestamp_frame,
                    text="時刻を表示",
                    variable=self.full_video_timestamp_enabled_var,
                    command=self.toggle_full_video_timestamp_settings,
                )
                timestamp_check.pack(side=tk.LEFT)
                self.full_video_timestamp_settings_widgets.append(timestamp_check)

                ttk.Label(timestamp_frame, text="位置:").pack(side=tk.LEFT, padx=(12, 3))
                position_box = ttk.Combobox(
                    timestamp_frame,
                    textvariable=self.full_video_timestamp_position_var,
                    values=("右下", "左下", "右上", "左上"),
                    state="readonly",
                    width=6,
                )
                position_box.pack(side=tk.LEFT)
                self.full_video_timestamp_settings_widgets.append(position_box)

                ttk.Label(timestamp_frame, text="文字サイズ:").pack(side=tk.LEFT, padx=(12, 3))
                size_spin = ttk.Spinbox(
                    timestamp_frame,
                    from_=0.8,
                    to=4.0,
                    increment=0.1,
                    textvariable=self.full_video_timestamp_size_var,
                    width=4,
                )
                size_spin.pack(side=tk.LEFT)
                self.full_video_timestamp_settings_widgets.append(size_spin)
                ttk.Label(timestamp_frame, text="%（画面高）").pack(side=tk.LEFT, padx=(3, 0))
                preview_button = ttk.Button(
                    timestamp_frame,
                    text="プレビュー...",
                    command=self.open_full_video_timestamp_preview,
                )
                preview_button.pack(side=tk.LEFT, padx=(12, 0))
                self.full_video_timestamp_settings_widgets.append(preview_button)

                var.trace_add("write", lambda *_: self.toggle_full_video_timestamp_settings())
                self.full_video_timestamp_enabled_var.trace_add(
                    "write", lambda *_: self._render_full_video_timestamp_preview()
                )
                self.full_video_timestamp_position_var.trace_add(
                    "write", lambda *_: self._render_full_video_timestamp_preview()
                )
                self.full_video_timestamp_size_var.trace_add(
                    "write", lambda *_: self._render_full_video_timestamp_preview()
                )
                self.toggle_full_video_timestamp_settings()
            if key == 'summary':
                # ヘルプの?マーク
                summary_help_label = tk.Label(f, text="?", font=("", 9, "bold"), fg=UI_CYAN, bg=UI_BG, cursor="hand2")
                summary_help_label.pack(side=tk.LEFT, padx=(2, 0))
                summary_help_label._tooltip = None
                
                def show_summary_tooltip(event, label=summary_help_label):
                    if label._tooltip is not None:
                        return
                    tooltip = tk.Toplevel(self)
                    tooltip.wm_overrideredirect(True)
                    tooltip.wm_geometry(f"+{event.x_root + 15}+{event.y_root + 10}")
                    tooltip.configure(bg=UI_BG)
                    content_frame = tk.Frame(tooltip, bg=UI_BG, padx=8, pady=5)
                    content_frame.pack()
                    tk.Label(content_frame, text="SNSなどの投稿に便利な編集済みの動画を作成します", 
                             bg=UI_BG, fg=UI_TEXT, font=("SF Pro Text", 9)).pack()
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

        twin_frame = ttk.LabelFrame(
            scrollable_frame, text="共通ノイズ前処理（NoiseTwin / モデル不要時間平均）"
        )
        twin_frame.pack(fill=tk.X, pady=(6, 2))
        ttk.Checkbutton(
            twin_frame,
            text="NoiseTwinを検出前・保存映像へ適用",
            variable=self.noise_twin_enabled_var,
            command=self.on_noise_twin_enabled_changed,
        ).pack(anchor=tk.W, padx=4, pady=(3, 1))
        model_row = ttk.Frame(twin_frame)
        model_row.pack(fill=tk.X, padx=4, pady=2)
        ttk.Label(model_row, text="モデル:", width=9).pack(side=tk.LEFT)
        ttk.Entry(
            model_row, textvariable=self.noise_twin_model_path_var, state="readonly"
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(model_row, text="選択", command=self.select_noise_twin_model).pack(side=tk.LEFT, padx=(5, 0))
        action_row = ttk.Frame(twin_frame)
        action_row.pack(fill=tk.X, padx=4, pady=2)
        self.btn_noise_twin_video_train = ttk.Button(
            action_row, text="動画から学習", command=self.start_noise_twin_training_from_video
        )
        self.btn_noise_twin_video_train.pack(side=tk.LEFT)
        self.btn_noise_twin_rtsp_train = ttk.Button(
            action_row, text="RTSPから5分間学習", command=self.start_noise_twin_training_from_rtsp
        )
        self.btn_noise_twin_rtsp_train.pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(twin_frame, textvariable=self.noise_twin_status_var, style="Hint.TLabel").pack(
            anchor=tk.W, padx=4, pady=(2, 4)
        )
        mean_row = ttk.Frame(twin_frame)
        mean_row.pack(fill=tk.X, padx=4, pady=(2, 4))
        ttk.Label(mean_row, text="モデル不要の時間平均:").pack(side=tk.LEFT)
        for label, value in (("OFF", 0), ("3フレーム", 3), ("5フレーム", 5)):
            ttk.Radiobutton(
                mean_row,
                text=label,
                variable=self.temporal_mean_frames_var,
                value=value,
                command=self.on_temporal_mean_changed,
            ).pack(side=tk.LEFT, padx=(6, 0))
        self.rtsp_save_temporal_mean_check = ttk.Checkbutton(
            twin_frame,
            text="RTSP保存動画にも時間平均を適用",
            variable=self.rtsp_save_temporal_mean_var,
        )
        self.rtsp_save_temporal_mean_check.pack(anchor=tk.W, padx=20, pady=(0, 4))
        ttk.Label(
            twin_frame,
            text="OFFでも流星検出には時間平均を使用し、保存動画だけを原画にします。",
            style="Hint.TLabel",
        ).pack(anchor=tk.W, padx=24, pady=(0, 4))
        self._update_rtsp_save_temporal_mean_state()

        encoding_frame = ttk.LabelFrame(twin_frame, text="ノイズ低減済み連続動画の圧縮")
        encoding_frame.pack(fill=tk.X, padx=4, pady=(2, 5))
        codec_row = ttk.Frame(encoding_frame)
        codec_row.pack(fill=tk.X, padx=4, pady=(3, 1))
        ttk.Label(codec_row, text="コーデック:", width=12).pack(side=tk.LEFT)
        codec_box = ttk.Combobox(
            codec_row,
            textvariable=self.processed_video_codec_var,
            values=("H.265 / HEVC (推奨)", "H.264 / AVC", "MPEG-4 (互換)"),
            state="readonly",
            width=24,
        )
        codec_box.pack(side=tk.LEFT)
        quality_row = ttk.Frame(encoding_frame)
        quality_row.pack(fill=tk.X, padx=4, pady=1)
        ttk.Label(quality_row, text="品質:", width=12).pack(side=tk.LEFT)
        quality_box = ttk.Combobox(
            quality_row,
            textvariable=self.processed_video_quality_var,
            values=("入力品質基準（推奨）", "最高品質", "高品質", "標準", "容量優先", "カスタム", "可逆圧縮（低速）"),
            state="readonly",
            width=24,
        )
        quality_box.pack(side=tk.LEFT)
        ttk.Label(quality_row, text="ビットレート:").pack(side=tk.LEFT, padx=(12, 3))
        self.processed_video_bitrate_spin = ttk.Spinbox(
            quality_row,
            from_=5,
            to=200,
            increment=5,
            textvariable=self.processed_video_bitrate_var,
            width=5,
        )
        self.processed_video_bitrate_spin.pack(side=tk.LEFT)
        ttk.Label(quality_row, text="Mbps").pack(side=tk.LEFT, padx=(3, 0))
        ttk.Label(
            encoding_frame,
            textvariable=self.processed_video_encoding_info_var,
            style="Hint.TLabel",
        ).pack(anchor=tk.W, padx=16, pady=(1, 3))
        codec_box.bind("<<ComboboxSelected>>", lambda _event: self.update_processed_video_encoding_ui())
        quality_box.bind("<<ComboboxSelected>>", lambda _event: self.update_processed_video_encoding_ui())
        self.processed_video_bitrate_var.trace_add(
            "write", lambda *_: self.update_processed_video_encoding_ui(update_bitrate=False)
        )
        self.update_processed_video_encoding_ui()

        lf_path = ttk.LabelFrame(scrollable_frame, text="保存先")
        lf_path.pack(fill=tk.X, pady=5)
        path_frame1 = ttk.Frame(lf_path); path_frame1.pack(fill=tk.X, pady=2)
        ttk.Label(path_frame1, text="流星:", width=10).pack(side=tk.LEFT)
        ttk.Entry(path_frame1, textvariable=self.meteor_save_path_var, state='readonly').pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(path_frame1, text="選択", command=lambda: self.select_save_path(self.meteor_save_path_var)).pack(side=tk.LEFT, padx=(5,0))
        ttk.Button(path_frame1, text="📂", command=lambda: self.open_meteor_folder_in_finder()).pack(side=tk.LEFT, padx=(2,0))
        path_frame2 = ttk.Frame(lf_path); path_frame2.pack(fill=tk.X, pady=2)
        ttk.Label(path_frame2, text="非流星:", width=10).pack(side=tk.LEFT)
        ttk.Entry(path_frame2, textvariable=self.not_meteor_save_path_var, state='readonly').pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(path_frame2, text="選択", command=lambda: self.select_save_path(self.not_meteor_save_path_var)).pack(side=tk.LEFT, padx=(5,0))
        ttk.Button(path_frame2, text="📂", command=lambda: self.open_not_meteor_folder_in_finder()).pack(side=tk.LEFT, padx=(2,0))

        ml_frame = ttk.LabelFrame(scrollable_frame, text="機械学習向けデータセット")
        ml_frame.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(
            ml_frame,
            text="検出イベントを学習用フォルダへ追加する",
            variable=self.ml_training_export_enabled_var,
        ).pack(anchor=tk.W, padx=4, pady=(3, 1))
        ttk.Label(
            ml_frame,
            text="通常差分・モノクロ動画・時間画像・時刻マップを、予測クラス別の未確認フォルダへ保存します。",
            style="Hint.TLabel",
        ).pack(anchor=tk.W, padx=24, pady=(0, 4))
        ml_path_frame = ttk.Frame(ml_frame)
        ml_path_frame.pack(fill=tk.X, padx=4, pady=2)
        ttk.Label(ml_path_frame, text="保存先:", width=10).pack(side=tk.LEFT)
        ttk.Entry(
            ml_path_frame, textvariable=self.ml_training_data_root_var, state="readonly"
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(
            ml_path_frame,
            text="選択",
            command=lambda: self.select_save_path(self.ml_training_data_root_var),
        ).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Button(
            ml_frame,
            text="未確認データを目視レビュー...",
            command=self.open_ml_training_review,
        ).pack(anchor=tk.W, padx=4, pady=(4, 5))

        archive_frame = ttk.LabelFrame(scrollable_frame, text="過去動画フォルダの自動処理")
        archive_frame.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(
            archive_frame,
            text="動画ごとに空領域を判定して自動マスクを適用する",
            variable=self.auto_video_mask_enabled_var,
        ).pack(anchor=tk.W, padx=4, pady=(3, 1))
        ttk.Checkbutton(
            archive_frame,
            text="YYYYMMDDフォルダ直下は天文薄明中の動画だけ処理する",
            variable=self.date_folder_twilight_filter_enabled_var,
        ).pack(anchor=tk.W, padx=4, pady=1)
        location_row = ttk.Frame(archive_frame)
        location_row.pack(fill=tk.X, padx=24, pady=(3, 4))
        ttk.Label(location_row, text="観測地点 緯度:").pack(side=tk.LEFT)
        ttk.Entry(location_row, textvariable=self.observation_latitude_var, width=10).pack(
            side=tk.LEFT, padx=(3, 10)
        )
        ttk.Label(location_row, text="経度:").pack(side=tk.LEFT)
        ttk.Entry(location_row, textvariable=self.observation_longitude_var, width=10).pack(
            side=tk.LEFT, padx=(3, 0)
        )
        ttk.Button(
            location_row, text="現在地取得", command=self.fetch_archive_observation_location
        ).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Label(
            archive_frame,
            text="日付フォルダでは天文薄明終了後と天文薄明開始前を採用します。時刻を解析できない動画は除外しません。",
            style="Hint.TLabel",
        ).pack(anchor=tk.W, padx=24, pady=(0, 4))

        lf_astro = ttk.LabelFrame(scrollable_frame, text="プレートソルブ & マスク")
        lf_astro.pack(fill=tk.X, pady=5)
        
        # プレートソルブ説明用ヘルプアイコン
        ps_help_frame = ttk.Frame(lf_astro); ps_help_frame.pack(fill=tk.X, pady=2)
        ps_help_label = ttk.Label(ps_help_frame, text="ⓘ", font=("Arial", 11), foreground=UI_CYAN, cursor="question_arrow")
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
            tooltip.configure(bg=UI_GLASS)
            
            content_frame = tk.Frame(tooltip, bg=UI_GLASS, padx=10, pady=8)
            content_frame.pack()
            
            text_label = tk.Label(content_frame, text=ps_help_text, justify=tk.LEFT, bg=UI_GLASS, fg=UI_TEXT, font=("SF Pro Text", 10))
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

        selected_model_lf = ttk.LabelFrame(lf_astro, text="使用する固定カメラ・プレートソルブモデル")
        selected_model_lf.pack(fill=tk.X, pady=(6, 2))
        selected_model_row = ttk.Frame(selected_model_lf)
        selected_model_row.pack(fill=tk.X, pady=(3, 1))
        ttk.Label(selected_model_row, text="モデル:").pack(side=tk.LEFT, padx=(0, 6))
        self.cmb_plate_solve_model = ttk.Combobox(
            selected_model_row,
            textvariable=self.plate_solve_model_var,
            state="readonly",
            width=74,
        )
        self.cmb_plate_solve_model.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.cmb_plate_solve_model.bind(
            "<<ComboboxSelected>>", self.on_plate_solve_model_selected
        )
        ttk.Button(
            selected_model_row, text="一覧更新", command=self._refresh_plate_solve_model_choices
        ).pack(side=tk.LEFT, padx=(5, 0))
        selected_model_info = ttk.Label(
            selected_model_lf,
            textvariable=self.plate_solve_model_info_var,
            foreground=UI_CYAN,
            wraplength=1120,
            justify=tk.LEFT,
        )
        selected_model_info.pack(fill=tk.X, padx=(5, 5), pady=(1, 2))
        ttk.Label(
            selected_model_lf,
            text="選択すると直ちに適用されます。被覆率はモデルが安全に較正できる画面割合、基準夜はモデル作成時の観測夜です。",
            foreground=UI_MUTED,
            wraplength=1120,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, padx=(5, 5), pady=(0, 4))

        model_lf = ttk.LabelFrame(lf_astro, text="高精度固定カメラモデル（作成・自動更新）")
        model_lf.pack(fill=tk.X, pady=(8, 2))
        model_source = ttk.Frame(model_lf); model_source.pack(fill=tk.X, pady=2)
        ttk.Label(model_source, text="対象動画/RTSPフォルダ:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(model_source, textvariable=self.camera_model_source_var, state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(model_source, text="動画", command=self.select_camera_model_video).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Button(model_source, text="フォルダ", command=self.select_camera_model_folder).pack(side=tk.LEFT, padx=(3, 0))
        model_range = ttk.Frame(model_lf); model_range.pack(fill=tk.X, pady=2)
        ttk.Label(model_range, text="時間範囲:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(model_range, textvariable=self.camera_model_start_var, width=20).pack(side=tk.LEFT)
        ttk.Label(model_range, text=" ～ ").pack(side=tk.LEFT)
        ttk.Entry(model_range, textvariable=self.camera_model_end_var, width=20).pack(side=tk.LEFT)
        ttk.Label(model_range, text="(動画は秒/時:分、フォルダは日時または時:分)").pack(side=tk.LEFT, padx=(6, 0))
        model_options = ttk.Frame(model_lf); model_options.pack(fill=tk.X, pady=2)
        ttk.Label(model_options, text="雲量しきい値:").pack(side=tk.LEFT)
        ttk.Entry(model_options, textvariable=self.camera_model_cloud_threshold_var, width=7).pack(side=tk.LEFT, padx=(3, 10))
        ttk.Label(model_options, text="監視間隔(秒):").pack(side=tk.LEFT)
        ttk.Entry(model_options, textvariable=self.camera_model_interval_var, width=7).pack(side=tk.LEFT, padx=(3, 10))
        ttk.Checkbutton(model_options, text="作成前に雲量判定（10%未満）", variable=self.camera_model_cloud_filter_var).pack(side=tk.LEFT)
        model_actions = ttk.Frame(model_lf); model_actions.pack(fill=tk.X, pady=2)
        self.btn_build_camera_model = ttk.Button(model_actions, text="選択範囲からモデル作成", command=self.start_camera_model_build)
        self.btn_build_camera_model.pack(side=tk.LEFT)
        self.btn_toggle_camera_model_monitor = ttk.Button(model_actions, text="RTSP自動監視を開始", command=self.toggle_camera_model_monitor)
        self.btn_toggle_camera_model_monitor.pack(side=tk.LEFT, padx=(6, 0))
        model_status_row = ttk.Frame(model_lf)
        model_status_row.pack(fill=tk.X, pady=(2, 3))
        self.camera_model_status_label = ttk.Label(
            model_status_row, textvariable=self.camera_model_status_var, foreground=UI_CYAN
        )
        self.camera_model_status_label.pack(side=tk.LEFT, anchor=tk.W)
        self.camera_model_progress_label = ttk.Label(
            model_status_row, textvariable=self.camera_model_progress_var, foreground=UI_CYAN,
            anchor=tk.E,
        )
        self.camera_model_progress_label.pack(side=tk.RIGHT, anchor=tk.E)
        self.bind_camera_model_input_hover(
            model_status_row, self.camera_model_status_label, self.camera_model_progress_label
        )
        self._refresh_plate_solve_model_choices()
        ttk.Label(
            model_lf,
            text=f"利用量ログ: {os.path.basename(config.AI_USAGE_LOG_PATH)}（入力/出力トークン・経過時間）",
            foreground=UI_MUTED,
        ).pack(anchor=tk.W, pady=(0, 3))
        
        ps_wcs_frame = ttk.Frame(lf_astro); ps_wcs_frame.pack(fill=tk.X, pady=2)
        ttk.Label(ps_wcs_frame, text="既存WCSファイル:").pack(side=tk.LEFT, padx=(0,5))
        ttk.Entry(ps_wcs_frame, textvariable=self.plate_solve_wcs_path_var, state='readonly').pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(ps_wcs_frame, text="選択", command=self.select_plate_solve_wcs_file).pack(side=tk.LEFT, padx=(5,0))

        # Plate solve mode selection (local vs API)
        ps_mode_frame = ttk.Frame(lf_astro); ps_mode_frame.pack(fill=tk.X, pady=2)
        ttk.Label(ps_mode_frame, text="ソルバー:").pack(side=tk.LEFT, padx=(0,5))
        ttk.Radiobutton(ps_mode_frame, text="ローカル広角 SIP・自動次数 (推奨・API不使用)", variable=self.plate_solve_mode_var, value="local").pack(side=tk.LEFT, padx=5)
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
        
        help_label = ttk.Label(api_key_frame, text=" ? ", font=("Arial", 10, "bold"), foreground=UI_CYAN, cursor="question_arrow")
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
            tooltip.configure(bg=UI_BG)
            frame = ttk.Frame(tooltip, padding=10)
            frame.pack()
            
            # Textウィジェットでクリック可能なURLを実装
            text_widget = tk.Text(frame, wrap=tk.WORD, width=45, height=20, bg=UI_FIELD, fg=UI_TEXT,
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

        ttk.Label(lf_astro, textvariable=self.plate_solve_status_var, foreground=UI_CYAN).pack(pady=2)
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

    def toggle_full_video_timestamp_settings(self):
        """フルサイズ動画と時刻表示の選択状態に応じて設定欄を有効化する。"""
        widgets = getattr(self, "full_video_timestamp_settings_widgets", [])
        full_video_enabled = bool(self.save_options_vars['full_video'].get())
        timestamp_enabled = bool(self.full_video_timestamp_enabled_var.get())
        for index, widget in enumerate(widgets):
            try:
                if index == 0:
                    widget.configure(state=tk.NORMAL if full_video_enabled else tk.DISABLED)
                elif isinstance(widget, ttk.Button):
                    widget.configure(state=tk.NORMAL if full_video_enabled else tk.DISABLED)
                else:
                    widget.configure(
                        state=("readonly" if isinstance(widget, ttk.Combobox) else tk.NORMAL)
                        if full_video_enabled and timestamp_enabled else tk.DISABLED
                    )
            except Exception:
                pass

    def open_full_video_timestamp_preview(self):
        """時刻表示の位置と大きさを確認・選択できるプレビューを開く。"""
        win = getattr(self, "_full_video_timestamp_preview_window", None)
        if win is not None and win.winfo_exists():
            win.lift()
            win.focus_force()
            self._render_full_video_timestamp_preview()
            return

        win = tk.Toplevel(self)
        win.title("フルサイズ動画の時刻表示プレビュー")
        win.configure(bg=UI_GLASS)
        win.resizable(False, False)
        self._full_video_timestamp_preview_window = win

        body = tk.Frame(win, bg=UI_GLASS, padx=12, pady=12)
        body.pack(fill=tk.BOTH, expand=True)
        tk.Label(
            body,
            text="プレビュー内の四隅をクリックすると、時刻の位置を変更できます。",
            bg=UI_GLASS,
            fg=UI_TEXT,
            anchor=tk.W,
        ).pack(fill=tk.X, pady=(0, 8))

        canvas = tk.Canvas(
            body, width=640, height=360, bg=ui_theme.COLORS["shadow"], highlightthickness=1,
            highlightbackground=UI_BORDER, cursor="crosshair",
        )
        canvas.pack()
        canvas.bind("<Button-1>", self._on_full_video_timestamp_preview_click)
        self._full_video_timestamp_preview_canvas = canvas

        self._full_video_timestamp_preview_status = tk.Label(
            body,
            bg=UI_GLASS,
            fg=UI_MUTED,
            anchor=tk.W,
        )
        self._full_video_timestamp_preview_status.pack(fill=tk.X, pady=(8, 0))

        ttk.Button(body, text="閉じる", command=self._close_full_video_timestamp_preview).pack(
            anchor=tk.E, pady=(8, 0)
        )
        win.protocol("WM_DELETE_WINDOW", self._close_full_video_timestamp_preview)
        self._render_full_video_timestamp_preview()

    def _close_full_video_timestamp_preview(self):
        win = getattr(self, "_full_video_timestamp_preview_window", None)
        self._full_video_timestamp_preview_window = None
        self._full_video_timestamp_preview_canvas = None
        self._full_video_timestamp_preview_status = None
        if win is not None and win.winfo_exists():
            win.destroy()

    def _on_full_video_timestamp_preview_click(self, event):
        canvas = getattr(self, "_full_video_timestamp_preview_canvas", None)
        if canvas is None:
            return
        horizontal = "左" if event.x < canvas.winfo_width() / 2 else "右"
        vertical = "上" if event.y < canvas.winfo_height() / 2 else "下"
        self.full_video_timestamp_position_var.set(f"{horizontal}{vertical}")

    def _render_full_video_timestamp_preview(self):
        canvas = getattr(self, "_full_video_timestamp_preview_canvas", None)
        win = getattr(self, "_full_video_timestamp_preview_window", None)
        if canvas is None or win is None or not win.winfo_exists():
            return

        width, height = 640, 360
        canvas.delete("all")
        # 暗い空を模した背景。配置の見え方を確認しやすくするため固定配置の星を置く。
        for index in range(110):
            x = (index * 83 + 41) % width
            y = (index * 47 + 19) % height
            brightness = 65 + (index * 29) % 130
            radius = 1 if index % 5 else 2
            color = f"#{brightness:02x}{brightness:02x}{min(255, brightness + 18):02x}"
            canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color, outline="")

        try:
            size_percent = float(self.full_video_timestamp_size_var.get())
        except (TypeError, ValueError):
            size_percent = config.FULL_VIDEO_TIMESTAMP_SIZE_PERCENT
        size_percent = max(0.8, min(4.0, size_percent))
        position = self.full_video_timestamp_position_var.get()
        if position not in ("右下", "左下", "右上", "左上"):
            position = "右下"
            self.full_video_timestamp_position_var.set(position)

        text = "2026/07/10 03:24:49.350"
        # 実寸比率を保ちつつ、小さ過ぎて確認できない場合だけプレビュー上で拡大する。
        font_size = max(11, int(round(height * size_percent / 100.0)))
        font = ("Helvetica", font_size, "bold")
        margin = max(10, int(round(font_size * 0.8)))
        anchor = {
            "右下": "se", "左下": "sw", "右上": "ne", "左上": "nw",
        }[position]
        x = margin if "左" in position else width - margin
        y = margin if "上" in position else height - margin
        text_id = canvas.create_text(x, y, text=text, anchor=anchor, fill="#F5F5F5", font=font)
        left, top, right, bottom = canvas.bbox(text_id)
        padding = max(4, font_size // 3)
        background = canvas.create_rectangle(
            left - padding, top - padding, right + padding, bottom + padding,
            fill="#000000", outline="", stipple="gray50",
        )
        canvas.tag_lower(background, text_id)

        enabled = self.full_video_timestamp_enabled_var.get()
        if not enabled:
            canvas.itemconfigure(text_id, state=tk.HIDDEN)
            canvas.itemconfigure(background, state=tk.HIDDEN)

        status = getattr(self, "_full_video_timestamp_preview_status", None)
        if status is not None:
            state = "表示" if enabled else "非表示"
            status.configure(
                text=f"現在: {state} / {position} / 文字サイズ {size_percent:.1f}%（画面高）"
            )

    def _create_model_selection_section(self, parent):
        lf_model = ttk.LabelFrame(parent, text="流星分類に使用するモデル")
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
        ttk.Label(model_row2, textvariable=self.model_meta_info_var, foreground=UI_CYAN).pack(side=tk.LEFT, padx=(10, 0))
        self.refresh_model_candidates()

    def _is_lm_studio_backend_selected(self):
        return self.ai_vlm_backend_var.get() == getattr(config, "AI_VLM_BACKEND_LM_STUDIO_QWEN35_2B", "lmstudio_qwen3_5_2b")

    def _create_ai_vlm_section(self, parent):
        lf_ai_vlm = ttk.LabelFrame(parent, text="AI解析に使用するVLM")
        lf_ai_vlm.pack(fill=tk.X, pady=5)

        ai_vlm_backend_frame = ttk.Frame(lf_ai_vlm)
        ai_vlm_backend_frame.pack(fill=tk.X, pady=2)
        ttk.Label(ai_vlm_backend_frame, text="AIモデル:", width=10).pack(side=tk.LEFT)
        ttk.Radiobutton(
            ai_vlm_backend_frame,
            text="内蔵 Qwen3-VL 4B",
            variable=self.ai_vlm_backend_var,
            value=getattr(config, "AI_VLM_BACKEND_LOCAL_QWEN3_VL_4B", "local_qwen3_vl_4b"),
            command=self.on_ai_vlm_backend_changed,
        ).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(
            ai_vlm_backend_frame,
            text="LM Studio",
            variable=self.ai_vlm_backend_var,
            value=getattr(config, "AI_VLM_BACKEND_LM_STUDIO_QWEN35_2B", "lmstudio_qwen3_5_2b"),
            command=self.on_ai_vlm_backend_changed,
        ).pack(side=tk.LEFT)

        self.lm_studio_vlm_detail_frame = ttk.Frame(lf_ai_vlm)

        lm_row1 = ttk.Frame(self.lm_studio_vlm_detail_frame)
        lm_row1.pack(fill=tk.X, pady=2)
        ttk.Label(lm_row1, text="URL:", width=10).pack(side=tk.LEFT)
        ttk.Entry(lm_row1, textvariable=self.lm_studio_vlm_url_var).pack(side=tk.LEFT, fill=tk.X, expand=True)

        lm_row2 = ttk.Frame(self.lm_studio_vlm_detail_frame)
        lm_row2.pack(fill=tk.X, pady=2)
        ttk.Label(lm_row2, text="モデルID:", width=10).pack(side=tk.LEFT)
        self.cmb_lm_studio_vlm_model = ttk.Combobox(
            lm_row2,
            textvariable=self.lm_studio_vlm_model_var,
            values=(),
            state="readonly",
        )
        self.cmb_lm_studio_vlm_model.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.cmb_lm_studio_vlm_model.bind("<<ComboboxSelected>>", self.on_lm_studio_vlm_model_selected)
        self.btn_lm_studio_refresh_models = ttk.Button(
            lm_row2, text="モデル一覧を更新", command=self.refresh_lm_studio_vlm_models_async
        )
        self.btn_lm_studio_refresh_models.pack(side=tk.LEFT, padx=(5, 0))

        ai_vlm_action_row = ttk.Frame(self.lm_studio_vlm_detail_frame)
        ai_vlm_action_row.pack(fill=tk.X, pady=(4, 2))
        self.btn_load_ai_vlm = ttk.Button(ai_vlm_action_row, text="Load Model", command=self.load_ai_vlm_model_async)
        self.btn_load_ai_vlm.pack(side=tk.LEFT)
        self.btn_unload_ai_vlm = ttk.Button(ai_vlm_action_row, text="Unload Model", command=self.unload_ai_vlm_model_async)
        self.btn_unload_ai_vlm.pack(side=tk.LEFT, padx=(5, 0))
        ttk.Label(ai_vlm_action_row, textvariable=self.ai_vlm_status_var, foreground=UI_CYAN).pack(side=tk.LEFT, padx=(10, 0))

        self.lm_studio_vlm_url_var.trace_add("write", lambda *_: self._set_lm_studio_action_buttons_state())

        self.update_lm_studio_vlm_visibility()
        if self._is_lm_studio_backend_selected():
            self.after(100, self.refresh_lm_studio_vlm_models_async)

    def update_lm_studio_vlm_visibility(self):
        detail_frame = getattr(self, "lm_studio_vlm_detail_frame", None)
        if detail_frame is None:
            return
        if self._is_lm_studio_backend_selected():
            if not detail_frame.winfo_ismapped():
                detail_frame.pack(fill=tk.X, pady=(2, 0))
        else:
            detail_frame.pack_forget()

    def _set_lm_studio_action_buttons_state(self, busy=False):
        is_lm_studio = self._is_lm_studio_backend_selected()
        has_model = bool(self.lm_studio_vlm_model_var.get().strip())
        state = tk.NORMAL if is_lm_studio and has_model and not busy else tk.DISABLED
        for name in ("btn_load_ai_vlm", "btn_unload_ai_vlm"):
            button = getattr(self, name, None)
            if button is not None:
                button.configure(state=state)
        refresh_button = getattr(self, "btn_lm_studio_refresh_models", None)
        if refresh_button is not None:
            refresh_button.configure(state=tk.NORMAL if is_lm_studio and not busy else tk.DISABLED)

    def _unload_previous_ai_vlm_if_changed(self, new_backend, new_lm_model):
        old_backend = getattr(self, "_last_ai_vlm_backend", new_backend)
        old_lm_model = getattr(self, "_last_lm_studio_vlm_model", new_lm_model)
        if old_backend == new_backend and old_lm_model == new_lm_model:
            return

        old_url = self.lm_studio_vlm_url_var.get()

        def worker():
            with self._ai_vlm_operation_lock:
                try:
                    import bright_area_detector
                    if old_backend == getattr(config, "AI_VLM_BACKEND_LOCAL_QWEN3_VL_4B", "local_qwen3_vl_4b"):
                        unloaded = bright_area_detector.unload_local_model()
                        result = f"Unloaded: {getattr(bright_area_detector, 'MODEL_ID', 'local model')}" if unloaded else "Already unloaded: local model"
                    else:
                        bright_area_detector.configure_ai_backend(
                            backend=old_backend,
                            lm_studio_url=old_url,
                            lm_studio_model_id=old_lm_model,
                            lm_studio_api_key="",
                        )
                        result = bright_area_detector.unload_selected_ai_model()
                    self.append_log(f"前のAIモデルをアンロードしました: {result}")
                except Exception as e:
                    self.append_log(f"前のAIモデルのアンロードに失敗しました: {e}")
                finally:
                    try:
                        import bright_area_detector
                        bright_area_detector.configure_ai_backend(
                            backend=new_backend,
                            lm_studio_url=self.lm_studio_vlm_url_var.get(),
                            lm_studio_model_id=new_lm_model,
                            lm_studio_api_key="",
                        )
                    except Exception:
                        pass

        threading.Thread(target=worker, daemon=True).start()

    def on_ai_vlm_backend_changed(self):
        new_backend = self.ai_vlm_backend_var.get()
        new_lm_model = self.lm_studio_vlm_model_var.get()
        self._last_ai_vlm_backend = new_backend
        self._last_lm_studio_vlm_model = new_lm_model
        self.update_lm_studio_vlm_visibility()
        self.apply_ai_model_settings(show_message=False)
        if self._is_lm_studio_backend_selected():
            self.refresh_lm_studio_vlm_models_async()
        self._set_lm_studio_action_buttons_state()

    def on_lm_studio_vlm_model_selected(self, _event=None):
        new_backend = self.ai_vlm_backend_var.get()
        new_lm_model = self.lm_studio_vlm_model_var.get()
        self._last_ai_vlm_backend = new_backend
        self._last_lm_studio_vlm_model = new_lm_model
        self.apply_ai_model_settings(show_message=False)
        self.ai_vlm_status_var.set(f"選択中: {new_lm_model}")
        self._set_lm_studio_action_buttons_state()

    def refresh_lm_studio_vlm_models_async(self):
        if not self._is_lm_studio_backend_selected():
            return
        backend = self.ai_vlm_backend_var.get()
        lm_url = self.lm_studio_vlm_url_var.get().strip()
        current_model = self.lm_studio_vlm_model_var.get().strip()
        self.ai_vlm_status_var.set("Vision対応モデルを取得中...")
        self._set_lm_studio_action_buttons_state(busy=True)
        result_queue = queue.Queue(maxsize=1)

        def worker():
            try:
                import bright_area_detector
                bright_area_detector.configure_ai_backend(
                    backend=backend,
                    lm_studio_url=lm_url,
                    lm_studio_model_id=current_model,
                    lm_studio_api_key="",
                )
                model_ids = bright_area_detector.list_lm_studio_model_ids()
                result_queue.put(("ok", model_ids))
            except Exception as e:
                result_queue.put(("error", str(e)))

        def update_from_result_queue():
            """Tkのメインスレッドだけで、取得結果をウィジェットへ反映する。"""
            try:
                result_type, payload = result_queue.get_nowait()
            except queue.Empty:
                if self._is_lm_studio_backend_selected():
                    self.after(50, update_from_result_queue)
                return

            if not self._is_lm_studio_backend_selected():
                return
            if result_type == "error":
                self.ai_vlm_status_var.set("モデル一覧の取得に失敗")
                self.append_log(f"LM Studioモデル一覧の取得に失敗しました: {payload}")
                self._set_lm_studio_action_buttons_state()
                return

            model_ids = payload
            combo = getattr(self, "cmb_lm_studio_vlm_model", None)
            if combo is None:
                return
            combo.configure(values=model_ids)
            if not model_ids:
                self.lm_studio_vlm_model_var.set("")
                self.ai_vlm_status_var.set("Vision対応モデルなし（LM StudioのLocal Serverを確認）")
                self._set_lm_studio_action_buttons_state()
                return
            current = self.lm_studio_vlm_model_var.get().strip()
            if current not in model_ids:
                self.lm_studio_vlm_model_var.set(model_ids[0])
                current = model_ids[0]
            self.apply_ai_model_settings(show_message=False)
            self.ai_vlm_status_var.set(f"Vision対応モデル: {len(model_ids)}件 / 選択中: {current}")
            self._set_lm_studio_action_buttons_state()

        # after()の登録自体は必ずメインスレッドで行う。
        self.after(50, update_from_result_queue)
        threading.Thread(target=worker, daemon=True).start()

    def _run_ai_vlm_model_action_async(self, action_name, action_func_name):
        backend = self.ai_vlm_backend_var.get()
        lm_url = self.lm_studio_vlm_url_var.get().strip()
        lm_model_id = self.lm_studio_vlm_model_var.get().strip()
        if backend != getattr(config, "AI_VLM_BACKEND_LM_STUDIO_QWEN35_2B", "lmstudio_qwen3_5_2b"):
            messagebox.showinfo("AIモデル", "モデルのロード・アンロードはLM Studio選択時に利用できます。")
            return
        if not lm_model_id:
            messagebox.showwarning("AIモデル", "Vision対応モデルを選択してから実行してください。")
            return
        self.ai_vlm_status_var.set(f"{action_name}...")
        self._set_lm_studio_action_buttons_state(busy=True)
        result_queue = queue.Queue(maxsize=1)

        def worker():
            with self._ai_vlm_operation_lock:
                try:
                    import bright_area_detector
                    bright_area_detector.configure_ai_backend(
                        backend=backend,
                        lm_studio_url=lm_url,
                        lm_studio_model_id=lm_model_id,
                        lm_studio_api_key="",
                    )
                    action_func = getattr(bright_area_detector, action_func_name)
                    if action_func_name == "load_selected_ai_model":
                        result = action_func(status_callback=self.append_log)
                    else:
                        result = action_func()
                    result_queue.put(("ok", result))
                except Exception as e:
                    result_queue.put(("error", str(e)))

        def update_from_result_queue():
            try:
                result_type, payload = result_queue.get_nowait()
            except queue.Empty:
                self.after(50, update_from_result_queue)
                return
            if result_type == "ok":
                self.ai_vlm_status_var.set(payload)
                self._set_lm_studio_action_buttons_state()
                messagebox.showinfo("AIモデル", payload, parent=self)
                self.refresh_lm_studio_vlm_models_async()
                return
            self.ai_vlm_status_var.set("AIモデルエラー")
            self._set_lm_studio_action_buttons_state()
            messagebox.showerror("AIモデル", f"{action_name}に失敗しました:\n{payload}", parent=self)

        self.after(50, update_from_result_queue)
        threading.Thread(target=worker, daemon=True).start()

    def load_ai_vlm_model_async(self):
        self._run_ai_vlm_model_action_async("Load Model", "load_selected_ai_model")

    def unload_ai_vlm_model_async(self):
        self._run_ai_vlm_model_action_async("Unload Model", "unload_selected_ai_model")

    def _model_search_dirs(self):
        candidates = [
            os.path.dirname(os.path.abspath(__file__)),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
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
        if current and os.path.exists(current) and current not in discovered:
            discovered.append(current)
        self.available_model_paths = discovered
        if hasattr(self, "cmb_model_select"):
            self.cmb_model_select.configure(values=discovered)
        if (not current or not os.path.exists(current)) and discovered:
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
        architecture = meta.get("architecture", "complex_cnn_v1")
        if architecture == "meteor_fusion_universal_v1":
            threshold = float(meta.get("decision_threshold", 0.5))
            self.model_meta_info_var.set(
                f"汎用カメラモデル / 推奨閾値={threshold:.3f} / resize={resize_text}"
            )
        else:
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

    def apply_ai_model_settings(self, show_message=False):
        try:
            import bright_area_detector

            bright_area_detector.configure_ai_backend(
                backend=self.ai_vlm_backend_var.get(),
                lm_studio_url=self.lm_studio_vlm_url_var.get(),
                lm_studio_model_id=self.lm_studio_vlm_model_var.get(),
                lm_studio_api_key="",
            )
            active_name = bright_area_detector.get_active_model_name()
            self.ai_vlm_status_var.set(active_name)
            self.append_log(f"AI解析モデルを切り替えました: {active_name}")
            if show_message:
                messagebox.showinfo("AIモデル設定", f"AI解析モデルを適用しました:\n{active_name}")
            return True
        except Exception as e:
            self.ai_vlm_status_var.set("AIモデル設定エラー")
            self.append_log(f"AIモデル設定の適用に失敗しました: {e}")
            if show_message:
                messagebox.showerror("エラー", f"AIモデル設定の適用に失敗しました:\n{e}")
            return False

    def _normalize_output_directory(self, saved_path, default_path, folder_name):
        if not saved_path:
            return default_path

        saved_abs = os.path.abspath(saved_path)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        legacy_parent_path = os.path.join(os.path.dirname(script_dir), folder_name)

        if saved_abs == os.path.abspath(legacy_parent_path):
            return default_path
        if not os.path.isdir(os.path.dirname(saved_abs)):
            return default_path
        return saved_path

    def check_ai_model_connection_async(self, show_success=True, only_lm_studio=False):
        is_lm_studio = self._is_lm_studio_backend_selected()
        if only_lm_studio and not is_lm_studio:
            return
        if not self.apply_ai_model_settings(show_message=False):
            return
        self.ai_vlm_status_var.set("接続確認中...")

        def worker():
            try:
                import bright_area_detector
                connected, err = bright_area_detector.check_vlm_connection(status_callback=self.append_log, force=True)
                if connected:
                    msg = f"接続OK: {bright_area_detector.get_active_model_name()}"
                    self.after(0, lambda: self.ai_vlm_status_var.set(msg))
                    if show_success:
                        self.after(0, lambda: messagebox.showinfo("AIモデル接続確認", msg, parent=self))
                else:
                    msg = f"接続NG: {err}"
                    self.after(0, lambda: self.ai_vlm_status_var.set("接続NG"))
                    if is_lm_studio:
                        msg = (
                            "LM Studioに接続できません。\n"
                            "LM Studioを起動し、Local Serverを有効にしてください。\n\n"
                            f"詳細: {err}"
                        )
                        self.after(0, lambda: messagebox.showerror("LM Studio 未起動", msg, parent=self))
                    else:
                        self.after(0, lambda: messagebox.showerror("AIモデル接続確認", msg, parent=self))
            except Exception as e:
                msg = f"接続確認に失敗しました: {e}"
                self.after(0, lambda: self.ai_vlm_status_var.set("接続NG"))
                self.after(0, lambda: messagebox.showerror("AIモデル接続確認", msg, parent=self))

        threading.Thread(target=worker, daemon=True).start()

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

    def open_ml_training_review(self):
        root_dir = self.ml_training_data_root_var.get().strip()
        if not root_dir:
            messagebox.showerror("設定エラー", "機械学習向けデータの保存先を指定してください。")
            return
        os.makedirs(root_dir, exist_ok=True)
        ml_review.open_review_window(self, root_dir)

    def fetch_archive_observation_location(self):
        def worker():
            try:
                lat, lon = location_utils.get_current_location()
                self.after(0, lambda: self.observation_latitude_var.set(f"{lat:.6f}"))
                self.after(0, lambda: self.observation_longitude_var.set(f"{lon:.6f}"))
                self.after(0, lambda: self.append_log(
                    f"過去動画の観測地点を取得: 緯度={lat:.6f}, 経度={lon:.6f}"
                ))
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror("現在地取得エラー", str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def _set_noise_twin_training_buttons(self, enabled):
        state = tk.NORMAL if enabled else tk.DISABLED
        if hasattr(self, "btn_noise_twin_video_train"):
            self.btn_noise_twin_video_train.config(state=state)
        if hasattr(self, "btn_noise_twin_rtsp_train"):
            self.btn_noise_twin_rtsp_train.config(state=state)

    def refresh_noise_twin_status(self):
        path = self.noise_twin_model_path_var.get().strip()
        if not path:
            self.noise_twin_status_var.set("NoiseTwin: 未選択")
            return False
        try:
            metadata = noise_twin.load_metadata(path)
            validation = metadata.validation
            state = "検証済み" if validation.validated else "未検証（本番利用不可）"
            rtsp_state = (
                "RTSP可"
                if (
                    validation.realtime_test_seconds >= 60.0
                    and validation.dropped_frames == 0
                    and validation.realtime_fps >= metadata.fps
                )
                else "RTSP未検証"
            )
            self.noise_twin_status_var.set(
                f"NoiseTwin: {state}・{rtsp_state} / {metadata.width}x{metadata.height} / "
                f"{validation.realtime_fps:.1f} fps / 光量保持 {validation.flux_retention * 100:.1f}%"
            )
            return validation.validated
        except Exception as exc:
            self.noise_twin_status_var.set(f"NoiseTwin: 読込失敗 ({exc})")
            return False

    def on_noise_twin_enabled_changed(self):
        if self.noise_twin_enabled_var.get() and not self.refresh_noise_twin_status():
            self.noise_twin_enabled_var.set(False)
            messagebox.showwarning(
                "Camera Digital Twin",
                "検証基準を満たしたNoiseTwinモデルを選択または作成してください。",
            )
            return
        if self.noise_twin_enabled_var.get():
            self.temporal_mean_frames_var.set(0)
        self._update_rtsp_save_temporal_mean_state()
        self.append_log(
            f"Camera Digital Twinノイズ分離: {'ON' if self.noise_twin_enabled_var.get() else 'OFF'}"
        )

    def update_processed_video_encoding_ui(self, update_bitrate=True):
        quality = self.processed_video_quality_var.get()
        preset_bitrates = {
            "最高品質": 80,
            "高品質": 60,
            "標準": 40,
            "容量優先": 20,
        }
        if update_bitrate and quality in preset_bitrates:
            value = str(preset_bitrates[quality])
            if self.processed_video_bitrate_var.get() != value:
                self.processed_video_bitrate_var.set(value)
        editable = quality == "カスタム"
        if hasattr(self, "processed_video_bitrate_spin"):
            self.processed_video_bitrate_spin.configure(
                state=tk.NORMAL if editable else tk.DISABLED
            )
        if quality == "入力品質基準（推奨）":
            text = "各RTSP・動画の実ビットレートを測定し、情報量を超えない値へ自動設定します。"
        elif quality == "可逆圧縮（低速）":
            text = "量子化による情報損失なし。容量は映像ノイズ量に依存し、処理速度は低下します。"
        else:
            try:
                bitrate = max(5, min(200, int(float(self.processed_video_bitrate_var.get()))))
                estimated = bitrate * 60 / 8
                text = f"上限目安: 約{estimated:.0f} MB/分（実容量は通常これ以下）"
            except ValueError:
                text = "ビットレートは5〜200 Mbpsで指定してください。"
        if self.processed_video_codec_var.get().startswith("H.265"):
            text += " / Apple VideoToolbox使用"
        self.processed_video_encoding_info_var.set(text)

    def on_temporal_mean_changed(self):
        frames = int(self.temporal_mean_frames_var.get())
        if frames in (3, 5):
            self.noise_twin_enabled_var.set(False)
            self.append_log(f"モデル不要の時間平均: {frames}フレーム")
        else:
            self.append_log("モデル不要の時間平均: OFF")
        self._update_rtsp_save_temporal_mean_state()

    def _update_rtsp_save_temporal_mean_state(self):
        if not hasattr(self, "rtsp_save_temporal_mean_check"):
            return
        enabled = int(self.temporal_mean_frames_var.get()) in (3, 5)
        self.rtsp_save_temporal_mean_check.configure(
            state=tk.NORMAL if enabled else tk.DISABLED
        )

    def select_noise_twin_model(self):
        path = filedialog.askopenfilename(
            title="NoiseTwinモデルを選択",
            initialdir=self.noise_twin_model_dir,
            filetypes=[("NoiseTwinモデル", "*.pth *.pt"), ("すべてのファイル", "*.*")],
        )
        if not path:
            return
        self.noise_twin_model_path_var.set(path)
        if not self.refresh_noise_twin_status():
            self.noise_twin_enabled_var.set(False)

    def start_noise_twin_training_from_video(self):
        paths = filedialog.askopenfilenames(
            title="NoiseTwin学習に使う動画を選択",
            filetypes=[("動画ファイル", "*.mp4 *.mov *.mkv *.avi *.m4v"), ("すべてのファイル", "*.*")],
        )
        if paths:
            self._start_noise_twin_training(list(paths), "")

    def start_noise_twin_training_from_rtsp(self):
        url = ""
        if self.rtsp_urls:
            url = self.rtsp_urls[0]
        if not url:
            url = self.rtsp_url_var.get().strip()
        if not url:
            messagebox.showwarning("NoiseTwin学習", "RTSP URLを入力または追加してください。")
            return
        if not messagebox.askokcancel(
            "NoiseTwin学習",
            "RTSPを5分間収録した後、背景モデル・流星保護ゲートの学習と\n"
            "256本の高速合成流星検証、1分の連続無欠落試験を行います。\n"
            "処理中は通常検出を開始できません。",
        ):
            return
        self._start_noise_twin_training([], url)

    def _start_noise_twin_training(self, video_paths, rtsp_url):
        if self.noise_twin_training_process is not None:
            messagebox.showwarning("NoiseTwin学習", "NoiseTwin学習はすでに実行中です。")
            return
        if any((
            self.worker_thread and self.worker_thread.is_alive(),
            self.rtsp_thread and self.rtsp_thread.is_alive(),
            self.periodic_scan_thread and self.periodic_scan_thread.is_alive(),
        )):
            messagebox.showwarning("NoiseTwin学習", "通常処理を停止してから学習を開始してください。")
            return
        output = os.path.join(
            self.noise_twin_model_dir,
            f"noise_twin_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth",
        )
        if getattr(sys, "frozen", False):
            command = [sys.executable, "--noise-twin-worker", "--output", output]
        else:
            command = [
                sys.executable,
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "noise_twin_worker.py"),
                "--output",
                output,
            ]
        for path in video_paths:
            command.extend(("--video", path))
        if rtsp_url:
            command.extend(("--rtsp-url", rtsp_url, "--rtsp-duration", "300"))
        if self.apply_rtsp_dark_var.get() and os.path.exists(self.rtsp_dark_file):
            command.extend(("--correction", self.rtsp_dark_file))
        self._set_noise_twin_training_buttons(False)
        self.noise_twin_status_var.set("NoiseTwin: 学習準備中...")
        self.append_log("NoiseTwin学習を開始しました。")
        training_queue = queue.Queue()

        def worker():
            process = None
            error = None
            result = None
            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                self.noise_twin_training_process = process
                if process.stdout is not None:
                    for raw_line in process.stdout:
                        line = raw_line.strip()
                        if line.startswith("PROGRESS "):
                            parts = line.split(maxsplit=3)
                            training_queue.put(("progress", parts[1], int(parts[2]), int(parts[3])))
                        elif line.startswith("STATUS "):
                            training_queue.put(("status", line[7:]))
                        elif line.startswith("RESULT "):
                            result = json.loads(line[7:])
                        elif line.startswith("ERROR "):
                            error = line[6:]
                        elif line:
                            training_queue.put(("status", line))
                return_code = process.wait()
                if return_code != 0 and not error:
                    error = f"学習ワーカーが終了しました (exit={return_code})"
            except Exception as exc:
                error = str(exc)
            training_queue.put(("done", error, result, output))

        phase_names = {
            "capture": "RTSP収録",
            "background": "背景モデル学習",
            "gate": "流星保護ゲート学習",
            "validation": "合成流星検証",
            "realtime": "RTSP連続検証",
        }

        def poll():
            finished = False
            while True:
                try:
                    item = training_queue.get_nowait()
                except queue.Empty:
                    break
                if item[0] == "progress":
                    _, phase, done, total = item
                    label = phase_names.get(phase, phase)
                    self.noise_twin_status_var.set(f"NoiseTwin: {label} {done}/{total}")
                    if done == total or done % max(1, total // 20) == 0:
                        self.append_log(f"NoiseTwin {label}: {done}/{total}")
                elif item[0] == "status":
                    self.noise_twin_status_var.set(f"NoiseTwin: {item[1]}")
                elif item[0] == "done":
                    _, error, result, model_path = item
                    self.noise_twin_training_process = None
                    self._set_noise_twin_training_buttons(True)
                    if error or not result:
                        self.noise_twin_status_var.set("NoiseTwin: 学習失敗")
                        messagebox.showerror("NoiseTwin学習", f"学習に失敗しました:\n{error}")
                    else:
                        self.noise_twin_model_path_var.set(model_path)
                        valid = self.refresh_noise_twin_status()
                        if valid:
                            self.noise_twin_enabled_var.set(True)
                            messagebox.showinfo("NoiseTwin学習", "検証済みNoiseTwinを作成し、適用をONにしました。")
                        else:
                            messagebox.showwarning(
                                "NoiseTwin学習",
                                "モデルは保存しましたが、信号保持または誤検出低減の採用基準を満たしませんでした。",
                            )
                    finished = True
            if not finished:
                self.after(100, poll)

        threading.Thread(target=worker, daemon=True).start()
        self.after(100, poll)

    def save_settings(self):
        settings = {
            'periodic_scan_enabled': self.periodic_scan_var.get(), 'periodic_scan_directory': self.periodic_dir_var.get(),
            'periodic_scan_interval': self.periodic_interval_var.get(), 'periodic_time_limit_enabled': self.periodic_time_limit_var.get(),
            'periodic_start_hour': self.start_hour_var.get(), 'periodic_start_minute': self.start_min_var.get(),
            'periodic_end_hour': self.end_hour_var.get(), 'periodic_end_minute': self.end_min_var.get(),
            'folder_paths': self.folder_paths, 'rtsp_urls': self.rtsp_urls,
            'processing_source_priority': list(self.processing_source_priority),
            'save_options': {k: v.get() for k, v in self.save_options_vars.items()},
            'full_video_timestamp': {
                'enabled': self.full_video_timestamp_enabled_var.get(),
                'position': self.full_video_timestamp_position_var.get(),
                'size_percent': self.full_video_timestamp_size_var.get(),
            },
            'plate_solve_wcs_path': self.plate_solve_wcs_path_var.get(), 'plate_solve_video_path': self.plate_solve_video_path_var.get(),
            'plate_solve_model_path': self.plate_solve_model_path_var.get(),
            'camera_model_source': self.camera_model_source_var.get(),
            'camera_model_start': self.camera_model_start_var.get(),
            'camera_model_end': self.camera_model_end_var.get(),
            'camera_model_cloud_threshold': self.camera_model_cloud_threshold_var.get(),
            'camera_model_interval': self.camera_model_interval_var.get(),
            'camera_model_cloud_filter': self.camera_model_cloud_filter_var.get(),
            'use_plate_solve': self.use_plate_solve_var.get(), 'apply_mask': self.apply_mask_var.get(),
            'mask_path_or_status': self.mask_path_var.get(), 'concurrency': self.concurrency_var.get(),
            'interval': self.interval_var.get(), 'duration': self.duration_var.get(),
            'meteor_save_path': self.meteor_save_path_var.get(), 'not_meteor_save_path': self.not_meteor_save_path_var.get(),
            'ml_training_export_enabled': self.ml_training_export_enabled_var.get(),
            'ml_training_data_root': self.ml_training_data_root_var.get(),
            'auto_video_mask_enabled': self.auto_video_mask_enabled_var.get(),
            'date_folder_twilight_filter_enabled': self.date_folder_twilight_filter_enabled_var.get(),
            'observation_latitude': self.observation_latitude_var.get(),
            'observation_longitude': self.observation_longitude_var.get(),
            'selected_model_path': self.selected_model_path_var.get(),
            'custom_model_paths': list(self.custom_model_paths),
            'ai_vlm_backend': self.ai_vlm_backend_var.get(),
            'lm_studio_vlm_url': self.lm_studio_vlm_url_var.get(),
            'lm_studio_vlm_model_id': self.lm_studio_vlm_model_var.get(),
            'lm_studio_vlm_api_key': "",
            'has_mask_image': self.mask_image is not None, 'has_plate_solve_mask_image': self.plate_solve_mask_image is not None,
            'summary_video_config': self.summary_video_config,
            'video_concat_settings': {
                'bitrate': self.video_concat_bitrate_var.get(),
                'codec': self.video_concat_codec_var.get(),
                'fps': self.video_concat_fps_var.get(),
                'safe_mode': self.video_concat_safe_mode_var.get(),
                'apply_enhancement': self.video_concat_enhancement_var.get(),
                'timestamp_enabled': self.video_concat_timestamp_enabled_var.get(),
                'timestamp_position': self.video_concat_timestamp_position_var.get(),
                'timestamp_size_percent': self.video_concat_timestamp_size_var.get(),
                'timestamp_offset_seconds': self.video_concat_timestamp_offset_var.get(),
            },
            'auto_time_updater_enabled': self.auto_time_updater_enabled_var.get(),
            'rtsp_preset': self.rtsp_preset_var.get(),
            'rtsp_fps': self.rtsp_fps_var.get(),
            'rtsp_notification_sound': self.rtsp_notification_sound_var.get(),
            'camera_control_base_url': self.camera_control_base_url_var.get(),
            'camera_control_ev_target': self.camera_control_ev_target_var.get(),
            # RTSP time limit settings
            'rtsp_time_limit_enabled': self.rtsp_time_limit_var.get(),
            'rtsp_start_hour': self.rtsp_start_hour_var.get(), 'rtsp_start_minute': self.rtsp_start_min_var.get(),
            'apply_rtsp_dark': self.apply_rtsp_dark_var.get(),
            'rtsp_fixed_pattern_samples': int(self.rtsp_fixed_pattern_samples_var.get()),
            'noise_twin_enabled': self.noise_twin_enabled_var.get(),
            'noise_twin_model_path': self.noise_twin_model_path_var.get(),
            'temporal_mean_frames': int(self.temporal_mean_frames_var.get()),
            'rtsp_save_temporal_mean': self.rtsp_save_temporal_mean_var.get(),
            'processed_video_encoding': {
                'codec': self.processed_video_codec_var.get(),
                'quality': self.processed_video_quality_var.get(),
                'bitrate_mbps': self.processed_video_bitrate_var.get(),
            },
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
            self.rtsp_notification_sound_var.set(settings.get('rtsp_notification_sound', True))
            self.load_rtsp_dark_frame()
            self.apply_rtsp_dark_var.set(bool(settings.get('apply_rtsp_dark', False)) and self.rtsp_dark_frame is not None)
            fixed_pattern_samples = int(settings.get(
                'rtsp_fixed_pattern_samples',
                config.RTSP_FIXED_PATTERN_DEFAULT_SAMPLES,
            ))
            if fixed_pattern_samples not in config.RTSP_FIXED_PATTERN_SAMPLE_COUNTS:
                fixed_pattern_samples = config.RTSP_FIXED_PATTERN_DEFAULT_SAMPLES
            self.rtsp_fixed_pattern_samples_var.set(str(fixed_pattern_samples))
            self.noise_twin_model_path_var.set(settings.get('noise_twin_model_path', ''))
            requested_twin = bool(settings.get('noise_twin_enabled', False))
            self.noise_twin_enabled_var.set(requested_twin and self.refresh_noise_twin_status())
            mean_frames = int(settings.get('temporal_mean_frames', 0) or 0)
            self.temporal_mean_frames_var.set(
                0 if self.noise_twin_enabled_var.get() else (mean_frames if mean_frames in (3, 5) else 0)
            )
            self.rtsp_save_temporal_mean_var.set(
                bool(settings.get('rtsp_save_temporal_mean', True))
            )
            self._update_rtsp_save_temporal_mean_state()
            encoding = settings.get('processed_video_encoding', {})
            self.processed_video_codec_var.set(
                encoding.get('codec', "H.265 / HEVC (推奨)")
            )
            self.processed_video_quality_var.set(
                encoding.get('quality', "入力品質基準（推奨）")
            )
            self.processed_video_bitrate_var.set(
                str(encoding.get('bitrate_mbps', "40"))
            )
            self.update_processed_video_encoding_ui(update_bitrate=False)
            self.toggle_rtsp_time_limit_frame()

            self.folder_paths = settings.get('folder_paths', [])
            self.set_processing_source_priority(settings.get('processing_source_priority', []))
            # Clear existing items and add restored paths
            for item in self.folder_item_frames:
                item['frame'].destroy()
            self.folder_item_frames.clear()
            self.folder_selected_indices.clear()
            for p in self.folder_paths:
                # For restored paths, show path only (no FPS calculation to avoid delay)
                self._add_folder_item("--", p)
            self.rtsp_urls = settings.get('rtsp_urls', [])
            self.camera_control_base_url_var.set(settings.get('camera_control_base_url', ''))
            self.camera_control_ev_target_var.set(settings.get('camera_control_ev_target', '0.0'))
            # Clear and restore RTSP items
            for item in self.rtsp_item_frames:
                item['frame'].destroy()
            self.rtsp_item_frames.clear()
            self.rtsp_selected_indices.clear()
            for url in self.rtsp_urls:
                self._add_rtsp_item(url)

            saved_opts = settings.get('save_options', {})
            for key, var in self.save_options_vars.items(): var.set(saved_opts.get(key, True))
            timestamp_settings = settings.get('full_video_timestamp', {})
            self.full_video_timestamp_enabled_var.set(
                timestamp_settings.get('enabled', config.FULL_VIDEO_TIMESTAMP_ENABLED)
            )
            position = timestamp_settings.get('position', config.FULL_VIDEO_TIMESTAMP_POSITION)
            position = {
                'bottom_right': '右下', 'bottom_left': '左下',
                'top_right': '右上', 'top_left': '左上',
            }.get(position, position)
            if position in ("右下", "左下", "右上", "左上"):
                self.full_video_timestamp_position_var.set(position)
            self.full_video_timestamp_size_var.set(str(
                timestamp_settings.get('size_percent', config.FULL_VIDEO_TIMESTAMP_SIZE_PERCENT)
            ))
            self.toggle_full_video_timestamp_settings()

            concat_settings = settings.get('video_concat_settings', {})
            self.video_concat_bitrate_var.set(concat_settings.get('bitrate', 'Auto'))
            self.video_concat_codec_var.set(concat_settings.get('codec', 'h264'))
            self.video_concat_fps_var.set(concat_settings.get('fps', 'Auto'))
            self.video_concat_safe_mode_var.set(bool(concat_settings.get('safe_mode', True)))
            self.video_concat_enhancement_var.set(bool(concat_settings.get('apply_enhancement', False)))
            self.video_concat_timestamp_enabled_var.set(bool(concat_settings.get('timestamp_enabled', True)))
            self.video_concat_timestamp_position_var.set(concat_settings.get('timestamp_position', '右下'))
            self.video_concat_timestamp_size_var.set(str(concat_settings.get('timestamp_size_percent', 1.8)))
            self.video_concat_timestamp_offset_var.set(str(concat_settings.get('timestamp_offset_seconds', 0.0)))

            self.plate_solve_wcs_path_var.set(settings.get('plate_solve_wcs_path', ''))
            self.plate_solve_video_path_var.set(settings.get('plate_solve_video_path', ''))
            self.plate_solve_model_path_var.set(settings.get('plate_solve_model_path', ''))
            self.camera_model_source_var.set(settings.get('camera_model_source', ''))
            self.camera_model_start_var.set(settings.get('camera_model_start', ''))
            self.camera_model_end_var.set(settings.get('camera_model_end', ''))
            self.camera_model_cloud_threshold_var.set(str(settings.get('camera_model_cloud_threshold', '0.10')))
            self.camera_model_interval_var.set(str(settings.get('camera_model_interval', '60')))
            self.camera_model_cloud_filter_var.set(bool(settings.get('camera_model_cloud_filter', True)))
            self.use_plate_solve_var.set(settings.get('use_plate_solve', True))
            # Annotation now defaults to the on-device wide-angle solver even
            # for settings files created by older API-only app versions.
            self.plate_solve_mode_var.set(
                'local' if getattr(config, "LOCAL_SOLVER_ENABLED", False) else 'api'
            )
            
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
            meteor_save_path = self._normalize_output_directory(
                settings.get('meteor_save_path', config.DEFAULT_METEOR_SAVE_PATH),
                config.DEFAULT_METEOR_SAVE_PATH,
                'meteor',
            )
            not_meteor_save_path = self._normalize_output_directory(
                settings.get('not_meteor_save_path', config.DEFAULT_NOT_METEOR_SAVE_PATH),
                config.DEFAULT_NOT_METEOR_SAVE_PATH,
                'not_meteor',
            )
            self.meteor_save_path_var.set(meteor_save_path)
            self.not_meteor_save_path_var.set(not_meteor_save_path)
            self.ml_training_export_enabled_var.set(
                settings.get('ml_training_export_enabled', config.DEFAULT_ML_TRAINING_EXPORT_ENABLED)
            )
            self.ml_training_data_root_var.set(
                settings.get('ml_training_data_root', config.DEFAULT_ML_TRAINING_DATA_ROOT)
            )
            self.auto_video_mask_enabled_var.set(
                settings.get('auto_video_mask_enabled', config.DEFAULT_AUTO_VIDEO_MASK_ENABLED)
            )
            self.date_folder_twilight_filter_enabled_var.set(
                settings.get(
                    'date_folder_twilight_filter_enabled',
                    config.DEFAULT_DATE_FOLDER_TWILIGHT_FILTER_ENABLED,
                )
            )
            self.observation_latitude_var.set(
                str(settings.get('observation_latitude', config.DEFAULT_OBSERVATION_LATITUDE))
            )
            self.observation_longitude_var.set(
                str(settings.get('observation_longitude', config.DEFAULT_OBSERVATION_LONGITUDE))
            )
            self.custom_model_paths = [str(p) for p in settings.get('custom_model_paths', []) if isinstance(p, str)]
            saved_model_path = settings.get('selected_model_path', config.MODEL_PATH)
            self.selected_model_path_var.set(
                saved_model_path if saved_model_path and os.path.exists(saved_model_path) else config.MODEL_PATH
            )
            self.refresh_model_candidates()
            saved_ai_backend = settings.get(
                'ai_vlm_backend',
                getattr(config, "DEFAULT_AI_VLM_BACKEND", "local_qwen3_vl_4b"),
            )
            if (
                sys.platform == 'darwin'
                and saved_ai_backend == getattr(config, "AI_VLM_BACKEND_LOCAL_QWEN3_VL_4B", "local_qwen3_vl_4b")
            ):
                saved_ai_backend = getattr(config, "AI_VLM_BACKEND_LM_STUDIO_QWEN35_2B", "lmstudio_qwen3_5_2b")
            self.ai_vlm_backend_var.set(saved_ai_backend)
            self.lm_studio_vlm_url_var.set(settings.get('lm_studio_vlm_url', getattr(config, "DEFAULT_LM_STUDIO_VLM_URL", "http://localhost:1234/v1")))
            self.lm_studio_vlm_model_var.set(settings.get('lm_studio_vlm_model_id', getattr(config, "DEFAULT_LM_STUDIO_VLM_MODEL_ID", "qwen/qwen3-vl-4b")))
            self.lm_studio_vlm_api_key_var.set("")
            self.update_lm_studio_vlm_visibility()
            self.apply_ai_model_settings(show_message=False)
            self._last_ai_vlm_backend = self.ai_vlm_backend_var.get()
            self._last_lm_studio_vlm_model = self.lm_studio_vlm_model_var.get()
            if self._is_lm_studio_backend_selected():
                self.after(0, self.refresh_lm_studio_vlm_models_async)
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

    def open_meteor_folder_in_finder(self):
        """Open the current meteor data save folder in Finder."""
        path = self.meteor_save_path_var.get()
        if not path:
            path = config.DEFAULT_METEOR_SAVE_PATH
        if not os.path.exists(path):
            messagebox.showwarning("フォルダを開けません",
                                   f"指定されたフォルダが存在しません:\n{path}")
            return
        if sys.platform == "darwin":
            subprocess.Popen(["open", path])
        elif sys.platform == "win32":
            os.startfile(path)
        else:
            subprocess.Popen(["xdg-open", path])

    def open_not_meteor_folder_in_finder(self):
        """Open the current non-meteor data save folder in Finder."""
        path = self.not_meteor_save_path_var.get()
        if not path:
            path = config.DEFAULT_NOT_METEOR_SAVE_PATH
        if not os.path.exists(path):
            messagebox.showwarning("フォルダを開けません",
                                   f"指定されたフォルダが存在しません:\n{path}")
            return
        if sys.platform == "darwin":
            subprocess.Popen(["open", path])
        elif sys.platform == "win32":
            os.startfile(path)
        else:
            subprocess.Popen(["xdg-open", path])
