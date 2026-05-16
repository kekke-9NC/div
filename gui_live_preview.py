from gui_common import *


class LivePreviewMixin:
    def _init_live_preview_state(self):
        self.live_preview_window = None
        self.live_preview_label = None
        self.live_preview_photo = None
        self.live_preview_stop_event = None
        self.live_preview_thread = None

    def _is_rtsp_preview_available(self):
        return bool(
            self.rtsp_urls
            and self.rtsp_thread
            and self.rtsp_thread.is_alive()
            and not self.cancel_flag.is_set()
        )

    def _update_live_preview_button_state(self):
        button = getattr(self, "live_preview_button", None)
        if button is None:
            return
        button.config(state=tk.NORMAL if self._is_rtsp_preview_available() else tk.DISABLED)

    def open_rtsp_live_preview(self):
        if not self._is_rtsp_preview_available():
            messagebox.showinfo("ライブプレビュー", "RTSP処理中のみライブプレビューを表示できます。")
            self._update_live_preview_button_state()
            return

        if self.live_preview_window and self.live_preview_window.winfo_exists():
            self.live_preview_window.lift()
            self.live_preview_window.focus_force()
            return

        url = self.rtsp_urls[0]
        self.live_preview_stop_event = threading.Event()

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
        ttk.Button(controls, text="閉じる", command=self.close_rtsp_live_preview).pack(side=tk.RIGHT)

        win.protocol("WM_DELETE_WINDOW", self.close_rtsp_live_preview)

        self.live_preview_thread = threading.Thread(
            target=self._rtsp_live_preview_loop,
            args=(url, self.live_preview_stop_event),
            daemon=True,
        )
        self.live_preview_thread.start()

    def close_rtsp_live_preview(self):
        if self.live_preview_stop_event:
            self.live_preview_stop_event.set()
        win = self.live_preview_window
        self.live_preview_window = None
        self.live_preview_label = None
        self.live_preview_photo = None
        if win and win.winfo_exists():
            win.destroy()
        self._update_live_preview_button_state()

    def _rtsp_live_preview_loop(self, url, stop_event):
        previous_options = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
        if getattr(config, "RTSP_USE_TCP", False):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|loglevel;quiet"
            )

        previous_log_level = None
        try:
            if hasattr(cv2, "getLogLevel"):
                previous_log_level = cv2.getLogLevel()
            if hasattr(cv2, "setLogLevel"):
                cv2.setLogLevel(0)
        except Exception:
            previous_log_level = None

        cap = None
        try:
            cap = self._open_rtsp_preview_capture(url)
            consecutive_failures = 0
            while not stop_event.is_set():
                if cap is None or not cap.isOpened():
                    self.after(0, lambda: self._set_live_preview_status("RTSP映像に再接続中..."))
                    self._sleep_live_preview(stop_event, 1.0)
                    cap = self._open_rtsp_preview_capture(url)
                    continue

                try:
                    ok, frame = cap.read()
                except cv2.error:
                    ok, frame = False, None
                except Exception:
                    ok, frame = False, None

                if not ok or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures == 1:
                        self.after(0, lambda: self._set_live_preview_status("RTSP映像を待機中..."))
                    if consecutive_failures >= 10:
                        try:
                            cap.release()
                        except Exception:
                            pass
                        cap = None
                        consecutive_failures = 0
                    self._sleep_live_preview(stop_event, 0.15)
                    continue
                consecutive_failures = 0

                frame = self._apply_live_preview_masks(frame)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                height, width = rgb.shape[:2]
                max_w, max_h = 960, 540
                scale = min(max_w / width, max_h / height, 1.0)
                if scale < 1.0:
                    rgb = cv2.resize(rgb, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)

                self.after(0, self._show_live_preview_frame, rgb)
                self._sleep_live_preview(stop_event, 1 / 15)
        except Exception as e:
            self.after(0, lambda err=e: self._set_live_preview_status(f"ライブプレビューでエラーが発生しました: {err}"))
        finally:
            if cap is not None:
                cap.release()
            if previous_options is None:
                os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
            else:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = previous_options
            if previous_log_level is not None and hasattr(cv2, "setLogLevel"):
                try:
                    cv2.setLogLevel(previous_log_level)
                except Exception:
                    pass

    def _open_rtsp_preview_capture(self, url):
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, getattr(config, "RTSP_BUFFER_SIZE", 3))
        except Exception:
            pass
        return cap

    def _sleep_live_preview(self, stop_event, seconds):
        stop_event.wait(seconds)

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
