from gui_common import *
import gui_common as common


class PreviewMixin:
    def create_info_panel(self, parent):
        panel = status_panel.StatusPanel(parent, progress_queue=self.progress_queue, app=self)
        panel.pack(fill=tk.BOTH, expand=True, pady=5)
        self.status_panel = panel

        self.log_text = panel.log_text
        self._init_summary_log_hover_preview()

        status_row = ttk.Frame(parent)
        status_row.pack(fill=tk.X, pady=5)
        self.progress = ttk.Progressbar(status_row, orient=tk.HORIZONTAL, mode='determinate')
        self.progress.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=(0,10))
        self.status_label = ttk.Label(status_row, text="待機中", width=15)
        self.status_label.pack(side=tk.LEFT)

        time_frame = ttk.Frame(parent)
        time_frame.pack(fill=tk.X, pady=5)
        self.eta_label = ttk.Label(time_frame, text="ETA: --:--:--", width=20)
        self.eta_label.pack(side=tk.LEFT)
        self.elapsed_label = ttk.Label(time_frame, text="経過: 00:00:00", width=20)
        self.elapsed_label.pack(side=tk.LEFT)

        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=(10,0))
        self.start_button = ttk.Button(btn_frame, text="開始", command=self.start_processing)
        self.start_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,5))
        self.live_preview_button = ttk.Button(
            btn_frame,
            text="ライブプレビュー",
            command=self.open_rtsp_live_preview,
            state=tk.DISABLED,
        )
        self.live_preview_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.cancel_button = ttk.Button(btn_frame, text="キャンセル", command=self.cancel_processing, state=tk.DISABLED)
        self.cancel_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5,0))
        self._update_live_preview_button_state()

        # Share status callback with worker-side pipeline.
        try:
            common.STATUS_CALLBACK = panel.get_status_callback()
        except Exception:
            common.STATUS_CALLBACK = None

    def append_log(self, message: str):
        if threading.current_thread() is not threading.main_thread():
            self.after(0, lambda m=message: self.append_log(m))
            return
        if not self.log_text.winfo_exists():
            return
        self.log_text.config(state='normal')
        try:
            view_top, view_bottom = self.log_text.yview()
        except Exception:
            view_top, view_bottom = (1.0, 1.0)
        follow_tail = view_bottom >= 0.995

        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        summary_ref = self._extract_summary_video_ref(message)
        if summary_ref:
            line_no = self.log_text.index("end-2c").split('.')[0]
            self.log_text.tag_add("summary_hover", f"{line_no}.0", f"{line_no}.end")
            self._summary_log_line_map[line_no] = {
                "summary_ref": summary_ref,
                "resolved_path": self._resolve_summary_video_path(summary_ref),
            }

        if follow_tail:
            self.log_text.see(tk.END)
        else:
            try:
                self.log_text.yview_moveto(view_top)
            except Exception:
                pass
        self.log_text.config(state='disabled')

    def _init_summary_log_hover_preview(self):
        self._summary_log_line_map = {}
        self._active_summary_line = None
        self._summary_preview_window = None
        self._summary_preview_title_label = None
        self._summary_preview_image_label = None
        self._summary_preview_open_button = None
        self._summary_preview_photo = None
        self._summary_preview_capture = None
        self._summary_preview_after_id = None
        self._summary_preview_hide_after_id = None
        self._summary_preview_fps = 12.0

        self.log_text.tag_config("summary_hover", foreground="#87CEEB", underline=True)
        self.log_text.bind("<Motion>", self._on_log_text_motion_for_summary_preview, add="+")
        self.log_text.bind("<Leave>", self._on_log_text_leave_for_summary_preview, add="+")

    def _extract_summary_video_ref(self, message: str) -> Optional[str]:
        if not message:
            return None
        patterns = [
            r"->\s*Summary:\s*(.+?\.mp4)\s*$",
            r"概要動画を保存しました:\s*(.+?\.mp4)\s*$",
        ]
        for pattern in patterns:
            m = re.search(pattern, message, flags=re.IGNORECASE)
            if m:
                return m.group(1).strip().strip('"').strip("'")
        return None

    def _resolve_summary_video_path(self, summary_ref: str) -> Optional[str]:
        if not summary_ref:
            return None

        ref = summary_ref.strip().strip('"').strip("'")
        if not ref:
            return None

        candidates = []

        if os.path.isabs(ref):
            candidates.append(ref)
        else:
            candidates.append(os.path.abspath(ref))

        if hasattr(self, "meteor_save_path_var"):
            try:
                meteor_dir = self.meteor_save_path_var.get()
                if meteor_dir:
                    candidates.append(os.path.join(meteor_dir, ref))
                    candidates.append(os.path.join(meteor_dir, os.path.basename(ref)))
            except Exception:
                pass

        if hasattr(self, "not_meteor_save_path_var"):
            try:
                not_meteor_dir = self.not_meteor_save_path_var.get()
                if not_meteor_dir:
                    candidates.append(os.path.join(not_meteor_dir, ref))
                    candidates.append(os.path.join(not_meteor_dir, os.path.basename(ref)))
            except Exception:
                pass

        try:
            candidates.append(os.path.join(config.DEFAULT_METEOR_SAVE_PATH, os.path.basename(ref)))
            candidates.append(os.path.join(config.DEFAULT_NOT_METEOR_SAVE_PATH, os.path.basename(ref)))
        except Exception:
            pass

        for path in candidates:
            if path and os.path.exists(path):
                return os.path.abspath(path)
        return None

    def _on_log_text_motion_for_summary_preview(self, event):
        if not hasattr(self, "log_text") or not self.log_text.winfo_exists():
            return

        self._cancel_summary_preview_hide()

        try:
            index = self.log_text.index(f"@{event.x},{event.y}")
        except Exception:
            self._hide_summary_preview()
            return

        if "summary_hover" not in self.log_text.tag_names(index):
            self._hide_summary_preview()
            return

        line_no = index.split('.')[0]
        meta = self._summary_log_line_map.get(line_no)
        if not meta:
            self._hide_summary_preview()
            return

        if self._active_summary_line != line_no:
            self._active_summary_line = line_no
            self._show_summary_preview_for_line(meta, event)
        else:
            self._move_summary_preview(event)

    def _on_log_text_leave_for_summary_preview(self, _event):
        self._schedule_summary_preview_hide(260)

    def _cancel_summary_preview_hide(self):
        if self._summary_preview_hide_after_id is not None:
            try:
                self.after_cancel(self._summary_preview_hide_after_id)
            except Exception:
                pass
            self._summary_preview_hide_after_id = None

    def _schedule_summary_preview_hide(self, delay_ms: int = 220):
        self._cancel_summary_preview_hide()
        self._summary_preview_hide_after_id = self.after(delay_ms, self._hide_summary_preview_if_pointer_outside)

    def _hide_summary_preview_if_pointer_outside(self):
        self._summary_preview_hide_after_id = None

        preview_has_pointer = False
        if self._summary_preview_window is not None and self._summary_preview_window.winfo_exists():
            try:
                px = self.winfo_pointerx()
                py = self.winfo_pointery()
                preview_has_pointer = self._summary_preview_window.winfo_containing(px, py) is not None
            except Exception:
                preview_has_pointer = False
        if preview_has_pointer:
            return

        if hasattr(self, "log_text") and self.log_text.winfo_exists():
            try:
                px = self.winfo_pointerx()
                py = self.winfo_pointery()
                x = px - self.log_text.winfo_rootx()
                y = py - self.log_text.winfo_rooty()
                if 0 <= x < self.log_text.winfo_width() and 0 <= y < self.log_text.winfo_height():
                    index = self.log_text.index(f"@{x},{y}")
                    if "summary_hover" in self.log_text.tag_names(index):
                        return
            except Exception:
                pass

        self._hide_summary_preview()

    def _on_summary_preview_enter(self, _event):
        self._cancel_summary_preview_hide()

    def _on_summary_preview_leave(self, _event):
        self._schedule_summary_preview_hide(180)

    def _show_summary_preview_for_line(self, meta: Dict[str, Any], event):
        self._hide_summary_preview()

        summary_path = meta.get("resolved_path")
        if not summary_path:
            summary_path = self._resolve_summary_video_path(meta.get("summary_ref", ""))
            if summary_path:
                meta["resolved_path"] = summary_path

        win = tk.Toplevel(self)
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        win.configure(bg="#0F1724")
        win.bind("<Enter>", self._on_summary_preview_enter, add="+")
        win.bind("<Leave>", self._on_summary_preview_leave, add="+")

        container = tk.Frame(win, bg="#0F1724", bd=1, relief=tk.SOLID)
        container.pack(fill=tk.BOTH, expand=True)
        container.bind("<Enter>", self._on_summary_preview_enter, add="+")
        container.bind("<Leave>", self._on_summary_preview_leave, add="+")
        win.geometry("392x300")

        title = os.path.basename(summary_path) if summary_path else "summary.mp4 が見つかりません"
        if len(title) > 56:
            title = title[:53] + "..."
        self._summary_preview_title_label = tk.Label(
            container,
            text=title,
            bg="#0F1724",
            fg="#D9E5FF",
            anchor="w",
            padx=8,
            pady=4,
            font=("Segoe UI", 9, "bold"),
        )
        self._summary_preview_title_label.pack(fill=tk.X)
        self._summary_preview_title_label.bind("<Enter>", self._on_summary_preview_enter, add="+")
        self._summary_preview_title_label.bind("<Leave>", self._on_summary_preview_leave, add="+")

        preview_area = tk.Frame(container, bg="#000000", width=372, height=214)
        preview_area.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        preview_area.pack_propagate(False)
        preview_area.bind("<Enter>", self._on_summary_preview_enter, add="+")
        preview_area.bind("<Leave>", self._on_summary_preview_leave, add="+")

        self._summary_preview_image_label = tk.Label(
            preview_area,
            bg="#000000",
            fg="#EAEAEA",
            text="プレビューを読み込み中...",
        )
        self._summary_preview_image_label.pack(fill=tk.BOTH, expand=True)
        self._summary_preview_image_label.bind("<Enter>", self._on_summary_preview_enter, add="+")
        self._summary_preview_image_label.bind("<Leave>", self._on_summary_preview_leave, add="+")

        action_frame = tk.Frame(container, bg="#0F1724")
        action_frame.pack(fill=tk.X, padx=6, pady=(0, 6))
        action_frame.bind("<Enter>", self._on_summary_preview_enter, add="+")
        action_frame.bind("<Leave>", self._on_summary_preview_leave, add="+")

        self._summary_preview_open_button = ttk.Button(
            action_frame,
            text="ファイルの場所を開く",
            command=lambda m=meta: self._open_summary_file_location(m),
        )
        self._summary_preview_open_button.pack(side=tk.RIGHT)
        self._summary_preview_open_button.bind("<Enter>", self._on_summary_preview_enter, add="+")
        self._summary_preview_open_button.bind("<Leave>", self._on_summary_preview_leave, add="+")
        if not summary_path:
            self._summary_preview_open_button.configure(state=tk.DISABLED)

        self._summary_preview_window = win
        self._move_summary_preview(event)

        if not summary_path or not os.path.exists(summary_path):
            self._summary_preview_image_label.configure(text="summary.mp4 の場所を特定できません")
            return

        cap = cv2.VideoCapture(summary_path)
        if not cap.isOpened():
            cap.release()
            self._summary_preview_image_label.configure(text="summary.mp4 を開けません")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1 or fps > 120:
            fps = 12.0
        self._summary_preview_fps = float(fps)
        self._summary_preview_capture = cap
        self._update_summary_preview_frame()

    def _open_summary_file_location(self, meta: Dict[str, Any]):
        summary_path = meta.get("resolved_path")
        if not summary_path:
            summary_path = self._resolve_summary_video_path(meta.get("summary_ref", ""))
            if summary_path:
                meta["resolved_path"] = summary_path

        if not summary_path:
            self.append_log("Summary動画の場所を特定できませんでした。")
            return

        target = os.path.abspath(summary_path)
        folder = os.path.dirname(target)
        if not os.path.exists(target):
            self.append_log(f"Summary動画が見つかりません: {target}")
            if folder and os.path.isdir(folder):
                target = folder
            else:
                return

        try:
            if sys.platform.startswith("win"):
                if os.path.isfile(target):
                    select_target = target.replace("/", "\\")
                    subprocess.Popen(["explorer", "/select," + select_target])
                else:
                    os.startfile(target)
            elif sys.platform == "darwin":
                if os.path.isfile(target):
                    subprocess.Popen(["open", "-R", target])
                else:
                    subprocess.Popen(["open", target])
            else:
                open_target = target if os.path.isdir(target) else folder
                subprocess.Popen(["xdg-open", open_target])
        except Exception as e:
            self.append_log(f"ファイルの場所を開けませんでした: {e}")

    def _move_summary_preview(self, event):
        if not self._summary_preview_window or not self._summary_preview_window.winfo_exists():
            return

        self._summary_preview_window.update_idletasks()
        ww = self._summary_preview_window.winfo_width()
        wh = self._summary_preview_window.winfo_height()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()

        if hasattr(self, "log_text") and self.log_text.winfo_exists():
            try:
                lx = self.log_text.winfo_rootx()
                ly = self.log_text.winfo_rooty()
                lw = self.log_text.winfo_width()
                lh = self.log_text.winfo_height()
            except Exception:
                lx = event.x_root
                ly = event.y_root
                lw = 0
                lh = 0
        else:
            lx = event.x_root
            ly = event.y_root
            lw = 0
            lh = 0

        x = lx + lw + 12
        if x + ww > sw - 8:
            x = max(8, lx - ww - 12)

        if ly <= event.y_root <= (ly + lh):
            y = event.y_root - (wh // 3)
        else:
            y = ly + 8
        y = max(8, y)
        if y + wh > sh - 8:
            y = max(8, sh - wh - 8)

        self._summary_preview_window.geometry(f"+{x}+{y}")

    def _update_summary_preview_frame(self):
        cap = self._summary_preview_capture
        label = self._summary_preview_image_label

        if cap is None or label is None or not label.winfo_exists():
            return

        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            if not ok:
                label.configure(text="動画フレームを取得できません")
                return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        max_w, max_h = 360, 202
        scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        if (new_w, new_h) != (w, h):
            frame_rgb = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        photo = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
        self._summary_preview_photo = photo
        label.configure(image=photo, text="")

        delay = int(1000 / max(1.0, self._summary_preview_fps))
        delay = max(30, min(150, delay))
        self._summary_preview_after_id = self.after(delay, self._update_summary_preview_frame)

    def _hide_summary_preview(self):
        self._active_summary_line = None
        self._cancel_summary_preview_hide()

        if self._summary_preview_after_id is not None:
            try:
                self.after_cancel(self._summary_preview_after_id)
            except Exception:
                pass
            self._summary_preview_after_id = None

        if self._summary_preview_capture is not None:
            try:
                self._summary_preview_capture.release()
            except Exception:
                pass
            self._summary_preview_capture = None

        if self._summary_preview_window is not None:
            try:
                if self._summary_preview_window.winfo_exists():
                    self._summary_preview_window.destroy()
            except Exception:
                pass
            self._summary_preview_window = None

        self._summary_preview_title_label = None
        self._summary_preview_image_label = None
        self._summary_preview_open_button = None
        self._summary_preview_photo = None

    def _run_on_main_thread(self, func):
        if threading.current_thread() is threading.main_thread():
            return func()
        result_queue = queue.Queue(maxsize=1)

        def wrapper():
            try:
                result_queue.put((True, func()))
            except Exception as e:
                result_queue.put((False, e))

        self.after(0, wrapper)
        ok, payload = result_queue.get()
        if ok:
            return payload
        raise payload

    def _format_size_bytes(size_bytes: int) -> str:
        units = ["B", "KB", "MB", "GB", "TB"]
        size = float(max(0, size_bytes))
        for unit in units:
            if size < 1024 or unit == units[-1]:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size_bytes} B"

    def _estimate_llm_storage_requirements(self, detector_module) -> Dict[str, Any]:
        repo_id = getattr(detector_module, "MODEL_ID", "Qwen/Qwen3-VL-4B-Instruct")
        download_bytes = int(10.0 * (1024 ** 3))
        final_bytes = int(4.5 * (1024 ** 3))
        overhead_bytes = int(1.5 * (1024 ** 3))
        fetched_metadata = False

        try:
            from huggingface_hub import HfApi
            info = HfApi().model_info(repo_id, files_metadata=True)
            file_sizes = [s.size for s in getattr(info, "siblings", []) if getattr(s, "size", None)]
            if file_sizes:
                download_bytes = int(sum(file_sizes))
                final_bytes = max(int(download_bytes * 0.45), int(3.0 * (1024 ** 3)))
                fetched_metadata = True
        except Exception as e:
            self.append_log(f"モデル容量情報の取得に失敗したため既定値で見積もります: {e}")

        temporary_bytes = download_bytes + final_bytes + overhead_bytes
        free_bytes = shutil.disk_usage(os.path.abspath(".")).free

        return {
            "repo_id": repo_id,
            "download_bytes": download_bytes,
            "final_bytes": final_bytes,
            "temporary_bytes": temporary_bytes,
            "free_bytes": free_bytes,
            "fetched_metadata": fetched_metadata,
        }

