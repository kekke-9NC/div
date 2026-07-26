from gui_common import *

UI_BG = ui_theme.COLORS["content_raised"]
UI_FIELD = ui_theme.COLORS["field"]
UI_GLASS = ui_theme.COLORS["glass"]
UI_HOVER = ui_theme.COLORS["glass_hover"]
UI_SELECTED = ui_theme.COLORS["glass_selected"]
UI_BORDER = ui_theme.COLORS["border"]
UI_TEXT = ui_theme.COLORS["text"]
UI_MUTED = ui_theme.COLORS["text_secondary"]
UI_ACCENT = ui_theme.COLORS["accent"]
UI_SUCCESS = ui_theme.COLORS["success"]


class UsageMixin:
    def create_usage_tab(self, parent):
        frame = ttk.Frame(parent)

        canvas = tk.Canvas(frame, highlightthickness=0, bg=UI_BG)
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

        pad_x = 12
        pad_y = 7
        wrap_w = 720
        base_bg = UI_BG
        phase_styles = {
            "detect": {
                "border": UI_BORDER,
                "header_bg": UI_SELECTED,
                "header_fg": UI_TEXT,
                "body_bg": UI_GLASS,
                "text_fg": UI_TEXT,
            },
            "post": {
                "border": UI_BORDER,
                "header_bg": "#1D3B39",
                "header_fg": UI_TEXT,
                "body_bg": UI_GLASS,
                "text_fg": UI_TEXT,
            },
        }

        ttk.Label(
            scrollable_frame,
            text="クイックスタート",
            font=("SF Pro Display", 18, "bold"),
            foreground=UI_TEXT,
        ).pack(pady=(15, 8), padx=pad_x, anchor="w")

        ttk.Label(
            scrollable_frame,
            text=(
                "各Stepの説明文の中にあるボタンを押すと、該当タブや操作場所へ直接移動します。"
            ),
            justify=tk.LEFT,
            wraplength=wrap_w,
        ).pack(padx=pad_x, pady=(0, 14), anchor="w")

        def make_inline_button(parent_frame, label, command, text_color=UI_ACCENT, button_bg=UI_HOVER):
            del text_color, button_bg
            return ttk.Button(
                parent_frame,
                text=label,
                command=command,
                style="Quiet.TButton",
            )

        default_font = tkfont.nametofont("TkDefaultFont")
        button_font = tkfont.Font(family="SF Pro Text", size=10, weight="bold")

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
                    btn = make_inline_button(row, label, command, color, button_bg=UI_HOVER)
                    btn.update_idletasks()
                    req_px = btn.winfo_reqwidth() + 10
                    if used_px > 0 and used_px + req_px > max_line_px:
                        btn.destroy()
                        row = new_row()
                        used_px = 0
                        btn = make_inline_button(row, label, command, color, button_bg=UI_HOVER)
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
            tk.Frame(sep, bg=UI_SUCCESS, height=1).pack(fill=tk.X, pady=(0, 6))
            tk.Label(
                sep,
                text=text,
                bg=base_bg,
                fg=UI_SUCCESS,
                font=("SF Pro Text", 10, "bold"),
            ).pack(anchor="center")
            tk.Frame(sep, bg=UI_SUCCESS, height=1).pack(fill=tk.X, pady=(6, 0))

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
                font=("SF Pro Text", 11, "bold"),
                padx=12,
                pady=8,
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
            foreground=UI_MUTED,
        ).pack(padx=pad_x, pady=(8, 14), anchor="w")

        return frame

