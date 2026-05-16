from gui_common import *
from camera_control import CameraControlClient, camera_base_url_from_rtsp


class CameraControlMixin:
    CAMERA_CONTROL_BOUNDS = {
        "brightness": (0, 255),
        "contrast": (0, 255),
        "saturation": (0, 255),
        "sharpness": (0, 255),
        "auto_exposureEx": (0, 1),
        "max_exposure": (1, 100),
        "exposure_time": (1, 100),
        "auto_gain_mode": (0, 1),
        "manual_AGain_enable": (0, 1),
        "manual_AGain": (0, 255),
        "manual_DGain_enable": (0, 1),
        "manual_DGain": (0, 255),
        "wdr_level": (0, 100),
    }
    CAMERA_CONTROL_FIELDS = [
        ("brightness", "明るさ", "basic"),
        ("contrast", "コントラスト", "basic"),
        ("saturation", "彩度", "basic"),
        ("sharpness", "シャープネス", "basic"),
        ("auto_exposureEx", "自動露出", "basic"),
        ("max_exposure", "最大露出", "basic"),
        ("exposure_time", "露出時間", "basic"),
        ("auto_gain_mode", "自動ゲイン", "basic"),
        ("manual_AGain_enable", "AGain有効", "basic"),
        ("manual_AGain", "アナログゲイン", "basic"),
        ("manual_DGain_enable", "DGain有効", "basic"),
        ("manual_DGain", "デジタルゲイン", "basic"),
        ("wdr_level", "WDR", "ex"),
    ]

    def _init_camera_control_state(self):
        self.camera_control_window = None
        self.camera_control_preview_label = None
        self.camera_control_photo = None
        self.camera_control_stop_event = None
        self.camera_control_preview_thread = None
        self.camera_control_vars = {}
        self.camera_control_original_values = {}
        self.camera_control_status_var = tk.StringVar(value="")
        self.camera_control_metrics_var = tk.StringVar(value="")
        self.camera_control_ev_target_var = tk.StringVar(value="0.0")
        self.camera_control_auto_adjust_var = tk.BooleanVar(value=False)
        self.camera_control_histogram_canvas = None
        self.camera_control_last_metrics = None
        self.camera_control_last_histogram = None
        self.camera_control_last_histogram_time = 0.0
        self.camera_control_auto_job = None
        self.camera_control_tuning_job = None
        self.camera_control_tuning_running = False
        self.camera_control_tuning_remaining = 0
        self.camera_control_reference_roi = None
        self.camera_control_selecting_reference = False
        self.camera_control_reference_drag_start = None
        self.camera_control_latest_frame_shape = None
        self.camera_control_display_size = None
        self.camera_control_tuning_candidates = []
        self.camera_control_tuning_index = 0
        self.camera_control_pending_values = None
        self.camera_control_best_values = None
        self.camera_control_best_score = None
        self.camera_control_periodic_param_index = 0
        self.camera_control_periodic_pending_values = None
        self.camera_control_periodic_previous_values = None
        self.camera_control_periodic_base_score = None

    def open_camera_control(self):
        if not self.check_admin_password():
            return

        if not self.rtsp_urls:
            messagebox.showwarning("カメラコントロール", "RTSP URLを追加してから開いてください。")
            return

        if self.camera_control_window and self.camera_control_window.winfo_exists():
            self.camera_control_window.lift()
            self.camera_control_window.focus_force()
            return

        if not hasattr(self, "camera_control_base_url_var"):
            self.camera_control_base_url_var = tk.StringVar(value="")
        if not self.camera_control_base_url_var.get().strip():
            self.camera_control_base_url_var.set(camera_base_url_from_rtsp(self.rtsp_urls[0]))

        self.camera_control_stop_event = threading.Event()

        win = Toplevel(self)
        win.title("カメラコントロール")
        win.geometry("1280x820")
        win.configure(bg="#2E3F5B")
        self.camera_control_window = win

        root = ttk.Frame(win, padding=10)
        root.pack(fill=tk.BOTH, expand=True)

        preview_frame = ttk.LabelFrame(root, text="ライブプレビュー")
        preview_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        self.camera_control_preview_label = ttk.Label(preview_frame, text="ライブプレビュー準備中...", anchor=tk.CENTER)
        self.camera_control_preview_label.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.camera_control_preview_label.bind("<ButtonPress-1>", self._on_camera_reference_press)
        self.camera_control_preview_label.bind("<B1-Motion>", self._on_camera_reference_drag)
        self.camera_control_preview_label.bind("<ButtonRelease-1>", self._on_camera_reference_release)
        self.camera_control_histogram_canvas = tk.Canvas(preview_frame, height=115, bg="#111927", highlightthickness=0)
        self.camera_control_histogram_canvas.pack(fill=tk.X, padx=6, pady=(0, 4))
        ttk.Label(preview_frame, textvariable=self.camera_control_metrics_var).pack(fill=tk.X, padx=6, pady=(0, 6))

        control_frame = ttk.LabelFrame(root, text="露出・ゲイン")
        control_frame.pack(side=tk.RIGHT, fill=tk.Y)

        url_frame = ttk.Frame(control_frame)
        url_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(url_frame, text="Camera URL").pack(anchor=tk.W)
        ttk.Entry(url_frame, textvariable=self.camera_control_base_url_var, width=34).pack(fill=tk.X)

        self.camera_control_vars = {}
        for key, label, _group in self.CAMERA_CONTROL_FIELDS:
            row = ttk.Frame(control_frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=14).pack(side=tk.LEFT)
            var = tk.StringVar(value="")
            self.camera_control_vars[key] = var
            ttk.Entry(row, textvariable=var, width=10).pack(side=tk.LEFT, padx=(4, 0))

        btns = ttk.Frame(control_frame)
        btns.pack(fill=tk.X, pady=(10, 2))
        ttk.Button(btns, text="取得", command=self.fetch_camera_control_settings).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btns, text="適用", command=self.apply_camera_control_settings).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btns, text="閉じる", command=self.close_camera_control).pack(side=tk.RIGHT)

        tune_frame = ttk.LabelFrame(control_frame, text="自動調整")
        tune_frame.pack(fill=tk.X, pady=(10, 2))
        ev_row = ttk.Frame(tune_frame)
        ev_row.pack(fill=tk.X, pady=2)
        ttk.Label(ev_row, text="目標EV", width=10).pack(side=tk.LEFT)
        ttk.Spinbox(
            ev_row,
            from_=-3.0,
            to=3.0,
            increment=0.3,
            width=7,
            textvariable=self.camera_control_ev_target_var,
            format="%.1f",
        ).pack(side=tk.LEFT)
        tune_btns = ttk.Frame(tune_frame)
        tune_btns.pack(fill=tk.X, pady=4)
        ttk.Button(tune_btns, text="自動最適化", command=self.start_camera_auto_tune).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(tune_btns, text="停止", command=self.stop_camera_auto_tune).pack(side=tk.LEFT)
        ttk.Checkbutton(
            tune_frame,
            text="10秒ごとに自動調整",
            variable=self.camera_control_auto_adjust_var,
            command=self.toggle_camera_auto_adjust,
        ).pack(anchor=tk.W, pady=(2, 0))
        roi_btns = ttk.Frame(tune_frame)
        roi_btns.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(roi_btns, text="リファレンス領域指定", command=self.start_camera_reference_selection).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(roi_btns, text="領域クリア", command=self.clear_camera_reference_selection).pack(side=tk.LEFT)

        ttk.Label(control_frame, textvariable=self.camera_control_status_var, wraplength=280).pack(fill=tk.X, pady=(8, 0))

        win.protocol("WM_DELETE_WINDOW", self.close_camera_control)
        self._set_camera_control_status("カメラ設定を取得しています...")
        self.fetch_camera_control_settings()
        self._start_camera_control_preview()

    def close_camera_control(self):
        if self.camera_control_stop_event:
            self.camera_control_stop_event.set()
        win = self.camera_control_window
        self.camera_control_window = None
        self.camera_control_preview_label = None
        self.camera_control_photo = None
        self.camera_control_auto_adjust_var.set(False)
        self.camera_control_tuning_running = False
        self.camera_control_selecting_reference = False
        if self.camera_control_auto_job is not None:
            try:
                self.after_cancel(self.camera_control_auto_job)
            except Exception:
                pass
            self.camera_control_auto_job = None
        if self.camera_control_tuning_job is not None:
            try:
                self.after_cancel(self.camera_control_tuning_job)
            except Exception:
                pass
            self.camera_control_tuning_job = None
        if win and win.winfo_exists():
            win.destroy()

    def fetch_camera_control_settings(self):
        def worker():
            try:
                client = CameraControlClient(self.camera_control_base_url_var.get())
                basic = client.get_image_adjustment()
                extra = client.get_image_adjustment_ex()
                values = {}
                values.update(extra)
                values.update(basic)
                self.after(0, lambda: self._apply_camera_values_to_ui(values))
            except Exception as exc:
                message = f"取得エラー: {exc}"
                self.after(0, lambda message=message: self._set_camera_control_status(message))

        threading.Thread(target=worker, daemon=True).start()

    def apply_camera_control_settings(self):
        basic_payload = {}
        ex_payload = {}
        for key, _label, group in self.CAMERA_CONTROL_FIELDS:
            raw = self.camera_control_vars[key].get().strip()
            if raw == "":
                continue
            try:
                value = int(raw)
            except ValueError:
                messagebox.showerror("カメラコントロール", f"{key} は整数で入力してください。")
                return
            if self.camera_control_original_values.get(key) == value:
                continue
            if group == "ex":
                ex_payload[key] = value
            else:
                basic_payload[key] = value

        if not basic_payload and not ex_payload:
            self._set_camera_control_status("変更された項目はありません。")
            return

        def worker():
            try:
                client = CameraControlClient(self.camera_control_base_url_var.get())
                if basic_payload:
                    client.set_image_adjustment(basic_payload)
                if ex_payload:
                    client.set_image_adjustment_ex(ex_payload)
                changed = sorted(list(basic_payload.keys()) + list(ex_payload.keys()))
                self.after(0, lambda: self._after_camera_apply(changed))
            except Exception as exc:
                message = f"適用エラー: {exc}"
                self.after(0, lambda message=message: self._set_camera_control_status(message))

        self._set_camera_control_status("カメラ設定を適用しています...")
        threading.Thread(target=worker, daemon=True).start()

    def start_camera_auto_tune(self):
        if not self.camera_control_last_metrics:
            self._set_camera_control_status("ライブ映像の統計を待っています。")
            return
        current = self._current_camera_values()
        if not current:
            self._set_camera_control_status("現在のカメラ値を取得してから実行してください。")
            return
        self.camera_control_tuning_running = True
        self.camera_control_tuning_candidates = self._build_noise_tuning_candidates(current)
        self.camera_control_tuning_index = 0
        self.camera_control_pending_values = None
        self.camera_control_best_values = current.copy()
        self.camera_control_best_score = self._camera_noise_score(self.camera_control_last_metrics)
        self._set_camera_control_status(
            f"自動最適化を開始しました。基準スコア {self.camera_control_best_score:.2f}"
        )
        self._run_camera_tuning_step()

    def stop_camera_auto_tune(self):
        self.camera_control_tuning_running = False
        self.camera_control_auto_adjust_var.set(False)
        if self.camera_control_auto_job is not None:
            try:
                self.after_cancel(self.camera_control_auto_job)
            except Exception:
                pass
            self.camera_control_auto_job = None
        if self.camera_control_tuning_job is not None:
            try:
                self.after_cancel(self.camera_control_tuning_job)
            except Exception:
                pass
            self.camera_control_tuning_job = None
        if self.camera_control_best_values:
            self._apply_camera_payload(self.camera_control_best_values, reason="自動調整停止時の最良値復元")
        self._set_camera_control_status("自動調整を停止しました。")

    def toggle_camera_auto_adjust(self):
        if self.camera_control_auto_adjust_var.get():
            self._set_camera_control_status("10秒ごとの自動調整を開始しました。")
            self._schedule_camera_auto_adjust(delay_ms=1000)
        else:
            if self.camera_control_auto_job is not None:
                try:
                    self.after_cancel(self.camera_control_auto_job)
                except Exception:
                    pass
                self.camera_control_auto_job = None
            self._set_camera_control_status("10秒ごとの自動調整を停止しました。")

    def _schedule_camera_auto_adjust(self, delay_ms=10000):
        if not self.camera_control_auto_adjust_var.get():
            return
        if self.camera_control_auto_job is not None:
            try:
                self.after_cancel(self.camera_control_auto_job)
            except Exception:
                pass
        self.camera_control_auto_job = self.after(delay_ms, self._run_camera_periodic_adjust)

    def _run_camera_periodic_adjust(self):
        self.camera_control_auto_job = None
        if not self.camera_control_auto_adjust_var.get():
            return
        self._apply_camera_periodic_noise_probe()
        self._schedule_camera_auto_adjust(delay_ms=10000)

    def _run_camera_tuning_step(self):
        if not self.camera_control_tuning_running:
            return

        if self.camera_control_pending_values is not None:
            score = self._camera_noise_score(self.camera_control_last_metrics)
            if self.camera_control_best_score is None or score < self.camera_control_best_score:
                self.camera_control_best_score = score
                self.camera_control_best_values = self.camera_control_pending_values.copy()
                self._set_camera_control_status(f"自動最適化: 改善 {score:.2f} を採用しました。")
            self.camera_control_pending_values = None

        if self.camera_control_tuning_index >= len(self.camera_control_tuning_candidates):
            self.camera_control_tuning_running = False
            if self.camera_control_best_values:
                self._apply_camera_payload(self.camera_control_best_values, reason="自動最適化の最良値")
            best = self.camera_control_best_score if self.camera_control_best_score is not None else 0.0
            self._set_camera_control_status(f"自動最適化が完了しました。最良スコア {best:.2f}")
            return

        candidate = self.camera_control_tuning_candidates[self.camera_control_tuning_index]
        self.camera_control_tuning_index += 1
        self.camera_control_pending_values = candidate.copy()
        self._apply_camera_payload(candidate, reason=f"自動最適化 {self.camera_control_tuning_index}/{len(self.camera_control_tuning_candidates)}")
        self.camera_control_tuning_job = self.after(3000, self._run_camera_tuning_step)

    def _build_noise_tuning_candidates(self, current):
        params = [
            ("exposure_time", [-2, -1, 1, 2]),
            ("max_exposure", [-2, -1, 1, 2]),
            ("manual_AGain", [-8, -4, 4, 8]),
            ("manual_DGain", [-6, -3, 3, 6]),
            ("brightness", [-4, -2, 2, 4]),
            ("contrast", [-4, -2, 2, 4]),
        ]
        base = current.copy()
        base.update({
            "auto_exposureEx": 0,
            "auto_gain_mode": 0,
            "manual_AGain_enable": 1,
            "manual_DGain_enable": 1,
        })
        candidates = []
        for key, deltas in params:
            if key not in base:
                continue
            for delta in deltas:
                cand = base.copy()
                cand[key] = self._clamp_camera_value(key, cand[key] + delta)
                if cand[key] != base[key]:
                    candidates.append(cand)
        return candidates

    def _apply_camera_periodic_noise_probe(self):
        metrics = self.camera_control_last_metrics
        current = self._current_camera_values()
        if not metrics or not current:
            self._set_camera_control_status("10秒自動調整: ライブ統計または現在値がありません。")
            return
        score = self._camera_noise_score(metrics)
        if self.camera_control_periodic_pending_values is not None:
            base_score = self.camera_control_periodic_base_score
            if base_score is None or score <= base_score:
                self._set_camera_control_status(f"10秒自動調整: 改善 {score:.2f} を採用しました。")
            elif self.camera_control_periodic_previous_values:
                self._apply_camera_payload(self.camera_control_periodic_previous_values, reason="10秒自動調整の復元")
                self._set_camera_control_status(f"10秒自動調整: 悪化 {score:.2f} のため復元します。")
            self.camera_control_periodic_pending_values = None
            self.camera_control_periodic_previous_values = None
            self.camera_control_periodic_base_score = None
            return

        candidates = self._build_noise_tuning_candidates(current)
        if not candidates:
            self._set_camera_control_status("10秒自動調整: 候補がありません。")
            return
        candidate = candidates[self.camera_control_periodic_param_index % len(candidates)]
        self.camera_control_periodic_param_index += 1
        self.camera_control_periodic_previous_values = current.copy()
        self.camera_control_periodic_pending_values = candidate.copy()
        self.camera_control_periodic_base_score = score
        self._apply_camera_payload(candidate, reason="10秒自動調整プローブ")

    def _camera_noise_score(self, metrics):
        if not metrics:
            return 9999.0
        return (
            float(metrics.get("reference_std", metrics.get("noise", 0.0)))
            + float(metrics.get("bright_clip", 0.0)) * 200.0
            + float(metrics.get("dark_clip", 0.0)) * 40.0
            - min(float(metrics.get("contrast", 0.0)), 80.0) * 0.03
        )

    def _clamp_camera_value(self, key, value):
        lo, hi = self.CAMERA_CONTROL_BOUNDS.get(key, (0, 255))
        return int(max(lo, min(hi, round(value))))

    def _apply_camera_auto_adjust_once(self, reason="自動調整"):
        metrics = self.camera_control_last_metrics
        if not metrics:
            self._set_camera_control_status(f"{reason}: ライブ映像の統計がまだありません。")
            return False
        payload = self._build_camera_auto_adjust_payload(metrics)
        if not payload:
            self._set_camera_control_status(f"{reason}: 調整不要です。")
            return False
        self._apply_camera_payload(payload, reason=reason)
        return True

    def _build_camera_auto_adjust_payload(self, metrics):
        current = self._current_camera_values()
        if not current:
            return {}

        target = self._camera_target_median()
        median = metrics.get("median", 0.0)
        contrast = metrics.get("contrast", 0.0)
        noise = metrics.get("noise", 0.0)
        bright_clip = metrics.get("bright_clip", 0.0)
        dark_clip = metrics.get("dark_clip", 0.0)
        error = target - median
        updates = {}

        def set_value(key, value):
            if key not in current:
                return
            lo, hi = self.CAMERA_CONTROL_BOUNDS.get(key, (0, 255))
            value = int(max(lo, min(hi, round(value))))
            if value != current.get(key):
                updates[key] = value
                current[key] = value

        set_value("auto_exposureEx", 0)
        set_value("auto_gain_mode", 0)
        set_value("manual_AGain_enable", 1)
        set_value("manual_DGain_enable", 1)

        exposure = current.get("exposure_time", current.get("max_exposure", 25))
        max_exposure = current.get("max_exposure", exposure)
        again = current.get("manual_AGain", 128)
        dgain = current.get("manual_DGain", 32)

        if error > 8:
            if exposure < max_exposure:
                set_value("exposure_time", exposure + min(3, max_exposure - exposure))
            elif max_exposure < 100:
                set_value("max_exposure", max_exposure + 2)
                set_value("exposure_time", min(exposure + 2, max_exposure + 2))
            elif noise < 12 and again < 255:
                set_value("manual_AGain", again + 6)
            elif noise < 9 and dgain < 96:
                set_value("manual_DGain", dgain + 3)
            else:
                set_value("brightness", current.get("brightness", 128) + 2)
        elif error < -8 or bright_clip > 0.01:
            if dgain > 16:
                set_value("manual_DGain", dgain - 4)
            elif again > 64:
                set_value("manual_AGain", again - 6)
            elif exposure > 1:
                set_value("exposure_time", exposure - 2)
            else:
                set_value("brightness", current.get("brightness", 128) - 2)

        if noise > 14:
            if current.get("manual_DGain", 0) > 16:
                set_value("manual_DGain", current.get("manual_DGain", 0) - 4)
            elif current.get("manual_AGain", 0) > 64:
                set_value("manual_AGain", current.get("manual_AGain", 0) - 4)
            set_value("sharpness", current.get("sharpness", 128) - 3)
            set_value("contrast", current.get("contrast", 128) - 1)
        elif contrast < 42 and noise < 10 and bright_clip < 0.005:
            set_value("contrast", current.get("contrast", 128) + 2)
            if current.get("sharpness", 128) < 150:
                set_value("sharpness", current.get("sharpness", 128) + 1)

        if bright_clip > 0.015 and current.get("wdr_level", 0) < 20:
            set_value("wdr_level", current.get("wdr_level", 0) + 2)
        elif dark_clip > 0.25 and bright_clip < 0.002 and current.get("wdr_level", 0) > 0:
            set_value("wdr_level", current.get("wdr_level", 0) - 1)

        return updates

    def _apply_camera_payload(self, updates, reason="自動調整"):
        basic_payload = {}
        ex_payload = {}
        field_groups = {key: group for key, _label, group in self.CAMERA_CONTROL_FIELDS}
        for key, value in updates.items():
            if field_groups.get(key) == "ex":
                ex_payload[key] = value
            else:
                basic_payload[key] = value

        def worker():
            try:
                client = CameraControlClient(self.camera_control_base_url_var.get())
                if basic_payload:
                    client.set_image_adjustment(basic_payload)
                if ex_payload:
                    client.set_image_adjustment_ex(ex_payload)
                changed = sorted(updates.keys())
                self.after(0, lambda changed=changed, updates=updates: self._after_camera_auto_apply(reason, changed, updates))
            except Exception as exc:
                message = f"{reason}エラー: {exc}"
                self.after(0, lambda message=message: self._set_camera_control_status(message))

        self._set_camera_control_status(f"{reason}: {', '.join(sorted(updates.keys()))} を調整中...")
        threading.Thread(target=worker, daemon=True).start()

    def _after_camera_auto_apply(self, reason, changed, updates):
        for key, value in updates.items():
            if key in self.camera_control_vars:
                self.camera_control_vars[key].set(str(value))
                self.camera_control_original_values[key] = value
        self._set_camera_control_status(f"{reason}: " + ", ".join(changed) + " を調整しました。")

    def _current_camera_values(self):
        values = {}
        for key, _label, _group in self.CAMERA_CONTROL_FIELDS:
            raw = self.camera_control_vars.get(key)
            if raw is None:
                continue
            try:
                values[key] = int(float(raw.get().strip()))
            except Exception:
                pass
        return values

    def _camera_target_median(self):
        try:
            ev = float(self.camera_control_ev_target_var.get())
        except Exception:
            ev = 0.0
            self.camera_control_ev_target_var.set("0.0")
        return max(25.0, min(180.0, 70.0 * (2.0 ** ev)))

    def start_camera_reference_selection(self):
        self.camera_control_selecting_reference = True
        self.camera_control_reference_drag_start = None
        self._set_camera_control_status("プレビュー上をドラッグしてリファレンス領域を指定してください。")

    def clear_camera_reference_selection(self):
        self.camera_control_reference_roi = None
        self.camera_control_reference_drag_start = None
        self.camera_control_selecting_reference = False
        self._set_camera_control_status("リファレンス領域をクリアしました。")

    def _on_camera_reference_press(self, event):
        if not self.camera_control_selecting_reference:
            return
        point = self._camera_display_to_frame_point(event.x, event.y)
        if point is None:
            return
        self.camera_control_reference_drag_start = point

    def _on_camera_reference_drag(self, event):
        if not self.camera_control_selecting_reference or self.camera_control_reference_drag_start is None:
            return
        point = self._camera_display_to_frame_point(event.x, event.y)
        if point is None:
            return
        x0, y0 = self.camera_control_reference_drag_start
        x1, y1 = point
        self.camera_control_reference_roi = self._normalize_camera_roi(x0, y0, x1, y1)

    def _on_camera_reference_release(self, event):
        if not self.camera_control_selecting_reference or self.camera_control_reference_drag_start is None:
            return
        point = self._camera_display_to_frame_point(event.x, event.y)
        if point is None:
            return
        x0, y0 = self.camera_control_reference_drag_start
        x1, y1 = point
        roi = self._normalize_camera_roi(x0, y0, x1, y1)
        if roi is not None:
            self.camera_control_reference_roi = roi
            self._set_camera_control_status(f"リファレンス領域を設定しました: {roi}")
        self.camera_control_selecting_reference = False
        self.camera_control_reference_drag_start = None

    def _normalize_camera_roi(self, x0, y0, x1, y1):
        if self.camera_control_latest_frame_shape is None:
            return None
        height, width = self.camera_control_latest_frame_shape
        left = max(0, min(width - 1, min(x0, x1)))
        right = max(0, min(width, max(x0, x1)))
        top = max(0, min(height - 1, min(y0, y1)))
        bottom = max(0, min(height, max(y0, y1)))
        if right - left < 10 or bottom - top < 10:
            return None
        return (int(left), int(top), int(right), int(bottom))

    def _camera_display_to_frame_point(self, x, y):
        label = self.camera_control_preview_label
        if (
            label is None
            or self.camera_control_latest_frame_shape is None
            or self.camera_control_display_size is None
        ):
            return None
        frame_h, frame_w = self.camera_control_latest_frame_shape
        display_w, display_h = self.camera_control_display_size
        label_w = max(label.winfo_width(), display_w)
        label_h = max(label.winfo_height(), display_h)
        offset_x = max(0, (label_w - display_w) // 2)
        offset_y = max(0, (label_h - display_h) // 2)
        img_x = x - offset_x
        img_y = y - offset_y
        if img_x < 0 or img_y < 0 or img_x >= display_w or img_y >= display_h:
            return None
        frame_x = int(img_x * frame_w / display_w)
        frame_y = int(img_y * frame_h / display_h)
        return frame_x, frame_y

    def _after_camera_apply(self, changed):
        self._set_camera_control_status("適用しました: " + ", ".join(changed))
        self.fetch_camera_control_settings()

    def _apply_camera_values_to_ui(self, values):
        self.camera_control_original_values = {}
        for key, _label, _group in self.CAMERA_CONTROL_FIELDS:
            if key in values:
                self.camera_control_original_values[key] = values[key]
                self.camera_control_vars[key].set(str(values[key]))
        self._set_camera_control_status("現在値を取得しました。")

    def _start_camera_control_preview(self):
        if self._is_rtsp_preview_available():
            self._set_camera_control_status("録画中のRTSP共有プレビューを表示します。")
            return

        rtsp_url = self.rtsp_urls[0] if self.rtsp_urls else ""
        if not rtsp_url:
            self._set_camera_control_preview_status("RTSP URLがありません。")
            return

        def worker():
            cap = None
            try:
                cap = utils.create_rtsp_capture(rtsp_url)
                if not cap or not cap.isOpened():
                    self.after(0, lambda: self._set_camera_control_preview_status("RTSPに接続できません。"))
                    return
                while self.camera_control_stop_event and not self.camera_control_stop_event.is_set():
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        time.sleep(0.1)
                        continue
                    self._handle_camera_control_preview_frame(frame)
            except Exception as exc:
                message = f"プレビューエラー: {exc}"
                self.after(0, lambda message=message: self._set_camera_control_preview_status(message))
            finally:
                if cap is not None:
                    cap.release()

        self.camera_control_preview_thread = threading.Thread(target=worker, daemon=True)
        self.camera_control_preview_thread.start()

    def handle_camera_control_shared_frame(self, frame):
        if not self.camera_control_window or not self.camera_control_stop_event:
            return
        if self.camera_control_stop_event.is_set():
            return
        self._handle_camera_control_preview_frame(frame)

    def _handle_camera_control_preview_frame(self, frame):
        try:
            frame = self._apply_live_preview_masks(frame)
            metrics, hist = self._analyze_camera_control_frame(frame)
            now = time.time()
            if now - self.camera_control_last_histogram_time >= 0.5:
                self.camera_control_last_histogram_time = now
                self.after(0, lambda metrics=metrics, hist=hist: self._update_camera_histogram(metrics, hist))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = rgb.shape[:2]
            self.camera_control_latest_frame_shape = (height, width)
            max_w, max_h = 820, 560
            scale = min(max_w / width, max_h / height, 1.0)
            if scale < 1.0:
                rgb = cv2.resize(rgb, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
            self.after(0, lambda img=rgb: self._show_camera_control_preview_frame(img))
        except Exception as exc:
            print(f"Camera control preview error: {exc}")

    def _analyze_camera_control_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = self.camera_control_reference_roi
        if roi is not None:
            x0, y0, x1, y1 = roi
            gray_for_stats = gray[y0:y1, x0:x1]
            if gray_for_stats.size < 100:
                gray_for_stats = gray
        else:
            gray_for_stats = gray

        sample = gray_for_stats[::4, ::4]
        valid = sample[sample > 2]
        if valid.size < 100:
            valid = sample.reshape(-1)

        median = float(np.median(valid))
        mean = float(np.mean(valid))
        p5 = float(np.percentile(valid, 5))
        p95 = float(np.percentile(valid, 95))
        contrast = p95 - p5
        dark_clip = float(np.mean(valid <= 3))
        bright_clip = float(np.mean(valid >= 252))

        blur = cv2.GaussianBlur(sample, (5, 5), 0)
        residual = cv2.absdiff(sample, blur)
        valid_residual = residual[sample > 2]
        noise = float(np.median(valid_residual) * 1.4826) if valid_residual.size else 0.0
        reference_std = float(np.std(valid)) if valid.size else 0.0
        hist = np.histogram(valid, bins=64, range=(0, 256))[0].astype(np.float32)
        metrics = {
            "mean": mean,
            "median": median,
            "contrast": contrast,
            "noise": noise,
            "reference_std": reference_std,
            "dark_clip": dark_clip,
            "bright_clip": bright_clip,
            "target": self._camera_target_median(),
            "roi": roi,
        }
        self.camera_control_last_metrics = metrics
        self.camera_control_last_histogram = hist
        return metrics, hist

    def _update_camera_histogram(self, metrics, hist):
        canvas = self.camera_control_histogram_canvas
        win = self.camera_control_window
        if canvas is None or win is None or not win.winfo_exists() or not canvas.winfo_exists():
            return
        canvas.delete("all")
        width = max(canvas.winfo_width(), 320)
        height = max(canvas.winfo_height(), 80)
        max_count = float(np.max(hist)) if hist.size else 0.0
        if max_count <= 0:
            return
        bar_w = max(1.0, width / len(hist))
        for i, count in enumerate(hist):
            x0 = i * bar_w
            x1 = x0 + bar_w
            y1 = height - 8
            y0 = y1 - (count / max_count) * (height - 18)
            canvas.create_rectangle(x0, y0, x1, y1, fill="#79A7E3", outline="")

        target_x = (metrics["target"] / 255.0) * width
        median_x = (metrics["median"] / 255.0) * width
        canvas.create_line(target_x, 4, target_x, height - 4, fill="#FFD166", width=2)
        canvas.create_line(median_x, 4, median_x, height - 4, fill="#EF476F", width=2)
        canvas.create_text(8, 8, text="Hist  黄=目標  赤=中央値", fill="#EAEAEA", anchor=tk.NW)

        self.camera_control_metrics_var.set(
            "中央値 {median:.1f} / 目標 {target:.1f} / コントラスト {contrast:.1f} / "
            "ノイズ {noise:.1f} / 標準偏差 {std:.1f} / 黒潰れ {dark:.1%} / 白飛び {bright:.1%}{roi}".format(
                median=metrics["median"],
                target=metrics["target"],
                contrast=metrics["contrast"],
                noise=metrics["noise"],
                std=metrics["reference_std"],
                dark=metrics["dark_clip"],
                bright=metrics["bright_clip"],
                roi=" / ROI" if metrics.get("roi") else "",
            )
        )

    def _show_camera_control_preview_frame(self, rgb_frame):
        label = self.camera_control_preview_label
        win = self.camera_control_window
        if label is None or win is None or not win.winfo_exists() or not label.winfo_exists():
            return
        display = rgb_frame.copy()
        self.camera_control_display_size = (display.shape[1], display.shape[0])
        if self.camera_control_reference_roi is not None and self.camera_control_latest_frame_shape is not None:
            frame_h, frame_w = self.camera_control_latest_frame_shape
            disp_h, disp_w = display.shape[:2]
            x0, y0, x1, y1 = self.camera_control_reference_roi
            sx = disp_w / frame_w
            sy = disp_h / frame_h
            p0 = (int(x0 * sx), int(y0 * sy))
            p1 = (int(x1 * sx), int(y1 * sy))
            cv2.rectangle(display, p0, p1, (255, 209, 102), 2)
        image = Image.fromarray(display)
        photo = ImageTk.PhotoImage(image)
        self.camera_control_photo = photo
        label.config(image=photo, text="")

    def _set_camera_control_preview_status(self, message):
        label = self.camera_control_preview_label
        win = self.camera_control_window
        if label is None or win is None or not win.winfo_exists() or not label.winfo_exists():
            return
        label.config(image="", text=message)

    def _set_camera_control_status(self, message):
        if hasattr(self, "camera_control_status_var"):
            self.camera_control_status_var.set(message)
        try:
            self.append_log(f"カメラコントロール: {message}")
        except Exception:
            pass
