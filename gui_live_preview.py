from gui_common import *
from fixed_pattern import apply_fixed_pattern_correction


class LivePreviewMixin:
    def _init_live_preview_state(self):
        self.live_preview_window = None
        self.live_preview_label = None
        self.live_preview_photo = None
        self.live_preview_stop_event = None
        self.live_preview_thread = None
        self.live_preview_apply_dark_var = None
        self.live_preview_last_frame_time = 0.0
        self.live_preview_direct_started = False
        # RTSP の読み取りスレッドから Tk を直接操作しない。最新の 1 枚だけを
        # 保持し、メインスレッドのタイマーで描画することで UI のイベントキューが
        # フレーム更新で埋まり、待機表示のままになることを防ぐ。
        self.live_preview_frame_lock = threading.Lock()
        self.live_preview_pending_frame = None
        self.live_preview_render_after_id = None
        self.live_preview_last_enqueue_time = 0.0

    def _is_rtsp_shared_preview_available(self):
        return bool(
            self.rtsp_urls
            and self.rtsp_thread
            and self.rtsp_thread.is_alive()
            and not self.cancel_flag.is_set()
        )

    def _is_rtsp_preview_available(self):
        return bool(self.rtsp_urls)

    def _update_live_preview_button_state(self):
        button = getattr(self, "live_preview_button", None)
        if button is None:
            return
        button.config(state=tk.NORMAL if self._is_rtsp_preview_available() else tk.DISABLED)

    def open_rtsp_live_preview(self):
        if not self._is_rtsp_preview_available():
            messagebox.showinfo("ライブプレビュー", "RTSP URLを追加してからライブプレビューを開いてください。")
            self._update_live_preview_button_state()
            return

        if self.live_preview_window and self.live_preview_window.winfo_exists():
            self.live_preview_window.lift()
            self.live_preview_window.focus_force()
            return

        self.live_preview_stop_event = threading.Event()
        self.live_preview_last_frame_time = 0.0
        self.live_preview_direct_started = False
        self.live_preview_last_enqueue_time = 0.0
        with self.live_preview_frame_lock:
            self.live_preview_pending_frame = None

        win = Toplevel(self)
        win.title("RTSPライブプレビュー")
        win.geometry("1000x620")
        win.configure(bg="#2E3F5B")
        self.live_preview_window = win

        container = ttk.Frame(win, padding=10)
        container.pack(fill=tk.BOTH, expand=True)

        self.live_preview_label = ttk.Label(container, text="RTSP映像に接続中...", anchor=tk.CENTER)
        self.live_preview_label.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(container)
        controls.pack(fill=tk.X, pady=(8, 0))
        if self.live_preview_apply_dark_var is None:
            default_dark = False
            if hasattr(self, "apply_rtsp_dark_var"):
                default_dark = bool(self.apply_rtsp_dark_var.get())
            self.live_preview_apply_dark_var = tk.BooleanVar(value=default_dark)
        ttk.Checkbutton(
            controls,
            text="ダーク適用して表示",
            variable=self.live_preview_apply_dark_var,
            command=self._on_live_preview_dark_changed,
        ).pack(side=tk.LEFT)
        ttk.Button(controls, text="閉じる", command=self.close_rtsp_live_preview).pack(side=tk.RIGHT)

        win.protocol("WM_DELETE_WINDOW", self.close_rtsp_live_preview)
        self._schedule_live_preview_render()

        if self._is_rtsp_shared_preview_available():
            self._set_live_preview_status("録画中のRTSP映像を待機中...")
            # 録画側の共有フレームが停止していても、プレビューが永久に
            # 待機状態にならないよう、短時間後に独立接続へ切り替える。
            self.after(3000, self._start_live_preview_direct_reader_if_needed)
        else:
            self._set_live_preview_status("RTSPに直接接続中...")
            self._start_live_preview_direct_reader()

    def close_rtsp_live_preview(self):
        if self.live_preview_stop_event:
            self.live_preview_stop_event.set()
        if self.live_preview_render_after_id is not None:
            try:
                self.after_cancel(self.live_preview_render_after_id)
            except Exception:
                pass
            self.live_preview_render_after_id = None
        with self.live_preview_frame_lock:
            self.live_preview_pending_frame = None
        win = self.live_preview_window
        self.live_preview_window = None
        self.live_preview_label = None
        self.live_preview_photo = None
        self.live_preview_thread = None
        self.live_preview_direct_started = False
        if win and win.winfo_exists():
            win.destroy()
        self._update_live_preview_button_state()

    def _start_live_preview_direct_reader(self):
        if self.live_preview_direct_started:
            return
        rtsp_url = self.rtsp_urls[0] if self.rtsp_urls else ""
        if not rtsp_url:
            self._set_live_preview_status("RTSP URLがありません。")
            return
        self.live_preview_direct_started = True
        stop_event = self.live_preview_stop_event

        def worker():
            reconnect_wait = 2
            while stop_event and not stop_event.is_set():
                cap = None
                try:
                    cap = utils.create_rtsp_capture(rtsp_url)
                    if not cap or not cap.isOpened():
                        self.after(0, lambda: self._set_live_preview_status("RTSPに接続できません。再試行中..."))
                        time.sleep(reconnect_wait)
                        continue

                    consecutive_failures = 0
                    self.after(0, lambda: self._set_live_preview_status("RTSP映像を取得中..."))
                    while stop_event and not stop_event.is_set():
                        ok, frame = cap.read()
                        if not ok or frame is None:
                            consecutive_failures += 1
                            if consecutive_failures >= 30:
                                self.after(0, lambda: self._set_live_preview_status("RTSP映像が途切れました。再接続中..."))
                                break
                            time.sleep(0.1)
                            continue
                        consecutive_failures = 0
                        if self.live_preview_stop_event is not stop_event:
                            break
                        self.handle_rtsp_live_preview_frame(frame)
                except Exception as exc:
                    message = f"プレビューエラー: {exc}"
                    self.after(0, lambda message=message: self._set_live_preview_status(message))
                    time.sleep(reconnect_wait)
                finally:
                    if cap is not None:
                        cap.release()

        self.live_preview_thread = threading.Thread(target=worker, daemon=True)
        self.live_preview_thread.start()

    def _start_live_preview_direct_reader_if_needed(self):
        """共有プレビューが届かない場合だけ、カメラへ直接接続する。"""
        if self.live_preview_window is None or self.live_preview_stop_event is None:
            return
        if self.live_preview_stop_event.is_set() or self.live_preview_direct_started:
            return
        if self.live_preview_last_frame_time > 0:
            return
        self._set_live_preview_status("共有映像が届かないためRTSPに直接接続中...")
        self._start_live_preview_direct_reader()

    def handle_rtsp_live_preview_frame(self, frame):
        self.live_preview_last_frame_time = time.monotonic()
        camera_handler = getattr(self, "handle_camera_control_shared_frame", None)
        if camera_handler is not None:
            try:
                camera_handler(frame)
            except Exception as e:
                print(f"Camera control shared preview error: {e}")

        if self.live_preview_window is None or self.live_preview_stop_event is None:
            return
        if self.live_preview_stop_event.is_set():
            return
        # 録画用のフレーム受信は最大 FPS で呼ばれる。ここで毎回 after() を積むと
        # macOS の Tk イベントループが滞留し、画像表示まで到達しないことがある。
        # 10 fps に間引き、未描画分は常に最新フレームへ置き換える。
        now = time.monotonic()
        if now - self.live_preview_last_enqueue_time < 0.1:
            return
        self.live_preview_last_enqueue_time = now
        try:
            # OpenCV の capture が次フレームで内部バッファを再利用しても安全なように
            # コピーしてから UI スレッドへ渡す。
            with self.live_preview_frame_lock:
                self.live_preview_pending_frame = frame.copy()
        except Exception as e:
            print(f"ライブプレビューフレーム受信エラー: {e}")

    def _schedule_live_preview_render(self):
        if self.live_preview_window is None:
            return
        self.live_preview_render_after_id = self.after(33, self._render_live_preview_frame)

    def _render_live_preview_frame(self):
        """Tk のメインスレッドで最新フレームだけを画面へ反映する。"""
        self.live_preview_render_after_id = None
        if self.live_preview_window is None or self.live_preview_stop_event is None:
            return
        if self.live_preview_stop_event.is_set():
            return

        with self.live_preview_frame_lock:
            frame = self.live_preview_pending_frame
            self.live_preview_pending_frame = None

        if frame is not None:
            try:
                frame = self._apply_live_preview_dark(frame)
                frame = self._apply_live_preview_masks(frame)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width = rgb.shape[:2]
                max_w, max_h = 960, 540
                scale = min(max_w / width, max_h / height, 1.0)
                if scale < 1.0:
                    rgb = cv2.resize(rgb, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
                self._show_live_preview_frame(rgb)
            except Exception as e:
                print(f"ライブプレビューフレーム表示エラー: {e}")

        self._schedule_live_preview_render()

    def _on_live_preview_dark_changed(self):
        enabled = bool(self.live_preview_apply_dark_var and self.live_preview_apply_dark_var.get())
        if enabled and getattr(self, "rtsp_dark_frame", None) is None:
            if not self.load_rtsp_dark_frame():
                self.live_preview_apply_dark_var.set(False)
                messagebox.showwarning("ライブプレビュー", "適用できるダークフレームがありません。先に「ダークを撮る」を実行してください。")
                enabled = False
        state = "ON" if enabled else "OFF"
        try:
            self.append_log(f"ライブプレビューダーク表示: {state}")
        except Exception:
            pass

    def _apply_live_preview_dark(self, frame):
        if not (self.live_preview_apply_dark_var and self.live_preview_apply_dark_var.get()):
            return frame
        if getattr(self, "rtsp_dark_frame", None) is None and not self.load_rtsp_dark_frame():
            return frame
        try:
            dark = self.rtsp_dark_frame
            if dark is None:
                return frame
            if dark.shape[:2] != frame.shape[:2]:
                dark = cv2.resize(dark, (frame.shape[1], frame.shape[0]))
            return apply_fixed_pattern_correction(frame, dark)
        except Exception as e:
            print(f"ライブプレビューダーク適用エラー: {e}")
            return frame

    def _apply_live_preview_masks(self, frame):
        masked = frame.copy()
        masks = []
        if self.apply_mask_var.get() and self.mask_image is not None:
            masks.append(self.mask_image)
        if self.use_plate_solve_var.get() and self.plate_solve_mask_image is not None:
            masks.append(self.plate_solve_mask_image)

        height, width = masked.shape[:2]
        for mask in masks:
            try:
                mask_arr = mask
                if mask_arr.shape[:2] != (height, width):
                    mask_arr = cv2.resize(mask_arr, (width, height), interpolation=cv2.INTER_NEAREST)
                masked[mask_arr <= 0] = 0
            except Exception as e:
                print(f"ライブプレビューマスク適用エラー: {e}")
        return masked

    def _show_live_preview_frame(self, rgb_frame):
        label = self.live_preview_label
        win = self.live_preview_window
        if label is None or win is None or not win.winfo_exists() or not label.winfo_exists():
            return
        image = Image.fromarray(rgb_frame)
        photo = ImageTk.PhotoImage(image)
        self.live_preview_photo = photo
        label.config(image=photo, text="")

    def _set_live_preview_status(self, message):
        label = self.live_preview_label
        win = self.live_preview_window
        if label is None or win is None or not win.winfo_exists() or not label.winfo_exists():
            return
        label.config(image="", text=message)
