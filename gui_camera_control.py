from gui_common import *
from camera_control import CameraControlClient, CameraControlError, camera_base_url_from_rtsp


class CameraControlMixin:
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
        win.geometry("1120x720")
        win.configure(bg="#2E3F5B")
        self.camera_control_window = win

        root = ttk.Frame(win, padding=10)
        root.pack(fill=tk.BOTH, expand=True)

        preview_frame = ttk.LabelFrame(root, text="ライブプレビュー")
        preview_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        self.camera_control_preview_label = ttk.Label(preview_frame, text="ライブプレビュー準備中...", anchor=tk.CENTER)
        self.camera_control_preview_label.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        control_frame = ttk.LabelFrame(root, text="露出・ゲイン")
        control_frame.pack(side=tk.RIGHT, fill=tk.Y)

        url_frame = ttk.Frame(control_frame)
        url_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(url_frame, text="Camera URL").pack(anchor=tk.W)
        ttk.Entry(url_frame, textvariable=self.camera_control_base_url_var, width=32).pack(fill=tk.X)

        self.camera_control_vars = {}
        for key, label, _group in self.CAMERA_CONTROL_FIELDS:
            row = ttk.Frame(control_frame)
            row.pack(fill=tk.X, pady=3)
            ttk.Label(row, text=label, width=14).pack(side=tk.LEFT)
            var = tk.StringVar(value="")
            self.camera_control_vars[key] = var
            ttk.Entry(row, textvariable=var, width=10).pack(side=tk.LEFT, padx=(4, 0))

        btns = ttk.Frame(control_frame)
        btns.pack(fill=tk.X, pady=(10, 2))
        ttk.Button(btns, text="取得", command=self.fetch_camera_control_settings).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btns, text="適用", command=self.apply_camera_control_settings).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btns, text="閉じる", command=self.close_camera_control).pack(side=tk.RIGHT)

        ttk.Label(control_frame, textvariable=self.camera_control_status_var, wraplength=260).pack(fill=tk.X, pady=(8, 0))

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
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = rgb.shape[:2]
            max_w, max_h = 700, 540
            scale = min(max_w / width, max_h / height, 1.0)
            if scale < 1.0:
                rgb = cv2.resize(rgb, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
            self.after(0, lambda img=rgb: self._show_camera_control_preview_frame(img))
        except Exception as exc:
            print(f"Camera control preview error: {exc}")

    def _show_camera_control_preview_frame(self, rgb_frame):
        label = self.camera_control_preview_label
        win = self.camera_control_window
        if label is None or win is None or not win.winfo_exists() or not label.winfo_exists():
            return
        image = Image.fromarray(rgb_frame)
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
