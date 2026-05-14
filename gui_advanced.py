from gui_common import *


class AdvancedSettingsMixin:
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
        self._create_model_selection_section(scrollable_frame)
        self._create_ai_vlm_section(scrollable_frame)

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

