from gui_common import *


class ToolsMixin:
    def create_long_exposure_map_callback(self):
        """Callback for the 'Create Long Exposure Map' button."""
        if not self.check_admin_password():
            return

        if not self.folder_paths:
            messagebox.showwarning("情報", "ソース選択タブでフォルダまたは動画ファイルを追加してください。")
            return

        output_path = filedialog.asksaveasfilename(
            title="長時間輝線マップの保存先",
            defaultextension=".jpg",
            filetypes=[("JPEG Image", "*.jpg"), ("PNG Image", "*.png"), ("All Files", "*")]
        )
        
        if not output_path:
            return

        def run_task():
            self.append_log("長時間輝線マップの作成を開始します...")
            success = long_exposure_map.create_long_exposure_map(
                self.folder_paths, 
                output_path, 
                progress_callback=self.append_log
            )
            if success:
                messagebox.showinfo("完了", "長時間輝線マップの作成が完了しました。")
                self.append_log("長時間輝線マップの作成が完了しました。")
            else:
                messagebox.showerror("エラー", "長時間輝線マップの作成に失敗しました。ログを確認してください。")
                self.append_log("長時間輝線マップの作成に失敗しました。")

        threading.Thread(target=run_task, daemon=True).start()

    def apply_distortion_correction_callback(self):
        """Callback for the 'Distortion Correction' button."""
        if not self.check_admin_password():
            return

        if not self.folder_paths:
            messagebox.showwarning("情報", "ソース選択タブでフォルダまたは動画ファイルを追加してください。")
            return

        output_path = filedialog.asksaveasfilename(
            title="ゆがみ補正画像の保存先",
            defaultextension=".jpg",
            filetypes=[("JPEG Image", "*.jpg"), ("PNG Image", "*.png"), ("All Files", "*")]
        )
        
        if not output_path:
            return

        # Distortion maps are stored next to this module.
        module_dir = os.path.dirname(os.path.abspath(__file__))
        map_x_path = os.path.join(module_dir, "distortion_map_x.npy")
        map_y_path = os.path.join(module_dir, "distortion_map_y.npy")

        def run_task():
            self.append_log("ゆがみ補正処理を開始します...")
            success = distortion_correction.apply_distortion_correction(
                self.folder_paths, 
                output_path, 
                map_x_path,
                map_y_path,
                progress_callback=self.append_log
            )
            if success:
                messagebox.showinfo("完了", "ゆがみ補正画像の作成が完了しました。")
                self.append_log("ゆがみ補正画像の作成が完了しました。")
            else:
                messagebox.showerror("エラー", "ゆがみ補正画像の作成に失敗しました。ログを確認してください。")
                self.append_log("ゆがみ補正画像の作成に失敗しました。")

        threading.Thread(target=run_task, daemon=True).start()

    def _select_selfcal_mask_mode(self) -> Optional[str]:
        """Select mask mode for night self-calibration."""
        selection = {"mode": None}
        style = ttk.Style(self)
        bg_color = self.cget("background") or style.lookup("TFrame", "background") or "#2E3F5B"

        dialog = Toplevel(self)
        dialog.title("自己校正マスク設定")
        dialog.geometry("520x220")
        dialog.transient(self)
        dialog.grab_set()
        dialog.resizable(False, False)
        dialog.configure(background=bg_color)

        container = tk.Frame(dialog, bg=bg_color, padx=16, pady=14)
        container.pack(fill=tk.BOTH, expand=True)

        ttk.Label(container, text="夜間自己校正で使うマスク方式を選択してください。").pack(anchor=tk.W, pady=(0, 8))
        tk.Label(
            container,
            text="手動マスクは『除外したい領域を塗る』方式です（時刻表示・地上・強いかぶり等）。",
            bg=bg_color,
            fg="#AFC0DA",
            justify=tk.LEFT,
            wraplength=480
        ).pack(anchor=tk.W, pady=(0, 10))

        btns = ttk.Frame(container)
        btns.pack(fill=tk.X, pady=4)

        def choose(mode: str):
            selection["mode"] = mode
            dialog.destroy()

        ttk.Button(btns, text="自動+手動 (推奨)", command=lambda: choose("auto_plus_manual")).pack(fill=tk.X, pady=3)
        ttk.Button(btns, text="自動のみ", command=lambda: choose("auto_only")).pack(fill=tk.X, pady=3)
        ttk.Button(btns, text="手動のみ", command=lambda: choose("manual_only")).pack(fill=tk.X, pady=3)
        ttk.Button(container, text="キャンセル", command=dialog.destroy).pack(pady=(10, 0))

        dialog.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - dialog.winfo_width()) // 2
        y = self.winfo_y() + (self.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")

        self.wait_window(dialog)
        return selection["mode"]

    def _read_frame_for_selfcal_mask(self, video_path: str) -> Optional[np.ndarray]:
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                return None
            return frame
        except Exception:
            return None

    def _draw_manual_exclusion_mask_on_frame(
        self,
        frame: np.ndarray,
        title: str = "自己校正用手動マスク作成",
        existing_mask: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """Open a drawing dialog and return final mask (255=use, 0=exclude)."""
        if frame is None:
            return None

        result = {"mask": None}
        win = Toplevel(self)
        win.title(title)
        win.geometry("1060x760")
        win.transient(self)
        win.grab_set()

        orig_h, orig_w = frame.shape[:2]
        disp_w, disp_h = 980, 600
        scale = min(disp_w / orig_w, disp_h / orig_h)
        disp_w, disp_h = int(orig_w * scale), int(orig_h * scale)

        frame_disp = cv2.resize(frame, (disp_w, disp_h))
        frame_rgb = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB)
        bg_photo = ImageTk.PhotoImage(Image.fromarray(frame_rgb))

        canvas = Canvas(win, width=disp_w, height=disp_h, cursor="circle", bg="black")
        canvas.pack(pady=6)
        canvas.create_image(0, 0, anchor=tk.NW, image=bg_photo)
        canvas.image = bg_photo

        mask_data_disp = Image.new("L", (disp_w, disp_h), 0)
        draw = ImageDraw.Draw(mask_data_disp)

        # If an existing mask is provided, preload it (excluded area -> white paint).
        if existing_mask is not None:
            try:
                m = existing_mask
                if m.ndim == 3:
                    m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
                if m.shape[:2] != (orig_h, orig_w):
                    m = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                m_disp = cv2.resize(m, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
                painted_disp = cv2.bitwise_not(m_disp)
                mask_data_disp = Image.fromarray(painted_disp.astype(np.uint8), mode="L")
                draw = ImageDraw.Draw(mask_data_disp)
            except Exception:
                pass

        overlay_item = None
        overlay_photo_holder = {"photo": None}

        def refresh_overlay():
            nonlocal overlay_item
            mask_np = np.array(mask_data_disp, dtype=np.uint8)
            if mask_np.max() == 0:
                if overlay_item is not None:
                    canvas.delete(overlay_item)
                    overlay_item = None
                overlay_photo_holder["photo"] = None
                return

            rgba = np.zeros((disp_h, disp_w, 4), dtype=np.uint8)
            rgba[..., 0] = 255  # red
            rgba[..., 3] = (mask_np > 0).astype(np.uint8) * 100
            overlay_photo = ImageTk.PhotoImage(Image.fromarray(rgba, mode="RGBA"))
            overlay_photo_holder["photo"] = overlay_photo
            if overlay_item is None:
                overlay_item = canvas.create_image(0, 0, anchor=tk.NW, image=overlay_photo)
            else:
                canvas.itemconfig(overlay_item, image=overlay_photo)

        refresh_overlay()

        brush_radius = tk.IntVar(value=35)
        draw_mode = tk.StringVar(value="paint")  # paint=exclude, erase=restore

        def apply_brush(event):
            r = int(brush_radius.get())
            x0, y0, x1, y1 = event.x - r, event.y - r, event.x + r, event.y + r
            if draw_mode.get() == "paint":
                draw.ellipse((x0, y0, x1, y1), fill=255)
            else:
                draw.ellipse((x0, y0, x1, y1), fill=0)
            refresh_overlay()

        def paint(event):
            draw_mode.set("paint")
            apply_brush(event)

        def erase(event):
            draw_mode.set("erase")
            apply_brush(event)

        canvas.bind("<B1-Motion>", paint)
        canvas.bind("<Button-1>", paint)
        canvas.bind("<B3-Motion>", erase)
        canvas.bind("<Button-3>", erase)

        controls = ttk.Frame(win)
        controls.pack(fill=tk.X, padx=10)
        ttk.Label(controls, text="ブラシサイズ:").pack(side=tk.LEFT)
        ttk.Scale(controls, from_=5, to=150, orient=tk.HORIZONTAL, variable=brush_radius).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        ttk.Label(controls, text="左クリック: 除外 / 右クリック: 復元").pack(side=tk.LEFT, padx=(6, 0))

        sub_controls = ttk.Frame(win)
        sub_controls.pack(fill=tk.X, padx=10, pady=(4, 0))

        def clear_mask():
            draw.rectangle([0, 0, disp_w, disp_h], fill=0)
            refresh_overlay()

        ttk.Button(sub_controls, text="クリア", command=clear_mask).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Label(
            win,
            text="塗った領域は自己校正で除外されます。空以外の地上、時刻表示、強いかぶり、雲の出やすい部分を除外してください。",
            foreground="#87CEEB"
        ).pack(anchor=tk.W, padx=10, pady=(6, 0))

        def on_ok():
            mask_np_disp = np.array(mask_data_disp, dtype=np.uint8)
            if mask_np_disp.max() > 0:
                excluded_resized = cv2.resize(mask_np_disp, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                final_mask = cv2.bitwise_not(excluded_resized)
            else:
                final_mask = np.full((orig_h, orig_w), 255, dtype=np.uint8)
            result["mask"] = final_mask
            win.destroy()

        btn_frame = ttk.Frame(win)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="キャンセル", command=win.destroy).pack(side=tk.LEFT, padx=5)

        self.wait_window(win)
        return result["mask"]

    def estimate_distortion_map_night_callback(self):
        """Generate distortion maps from ~20 minutes of a night-sky video inside the app."""
        if not self.check_admin_password():
            return

        if not self.folder_paths:
            messagebox.showwarning("情報", "ソース選択タブで夜空動画のフォルダまたは動画ファイルを追加してください。")
            return

        initial_dir = None
        try:
            first_source = self.folder_paths[0]
            if os.path.isdir(first_source):
                initial_dir = first_source
            elif os.path.isfile(first_source):
                initial_dir = os.path.dirname(first_source)
        except Exception:
            initial_dir = None
        if not initial_dir:
            initial_dir = os.path.expanduser("~")

        selected_start_video = filedialog.askopenfilename(
            title="夜間自己校正の開始動画を選択 (この動画から後続分割動画を連続利用)",
            initialdir=initial_dir,
            filetypes=[
                ("動画ファイル", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("すべてのファイル", "*.*"),
            ]
        )
        if not selected_start_video:
            return

        mask_mode = self._select_selfcal_mask_mode()
        if not mask_mode:
            return

        use_auto_mask = (mask_mode != "manual_only")
        use_manual_mask = (mask_mode != "auto_only")
        selected_manual_mask = None

        if use_manual_mask:
            reuse_existing = False
            if self.selfcal_mask_image is not None:
                reuse_existing = messagebox.askyesno(
                    "手動自己校正マスク",
                    "既存の自己校正用手動マスクがあります。\n再利用しますか？\n\n"
                    "「いいえ」を選ぶと、開始動画のフレームで描き直します。"
                )

            if reuse_existing:
                selected_manual_mask = self.selfcal_mask_image.copy()
            else:
                frame_for_mask = self._read_frame_for_selfcal_mask(selected_start_video)
                if frame_for_mask is None:
                    messagebox.showerror("エラー", "手動マスク用に開始動画の先頭フレームを読み込めませんでした。")
                    return
                drawn_mask = self._draw_manual_exclusion_mask_on_frame(
                    frame_for_mask,
                    title="自己校正用手動マスク作成",
                    existing_mask=self.selfcal_mask_image
                )
                if drawn_mask is None:
                    return
                self.selfcal_mask_image = drawn_mask
                selected_manual_mask = drawn_mask.copy()

        if not messagebox.askyesno(
            "夜間自己校正マップ生成",
            "選択した開始動画から、ソース内の後続分割動画を連続利用して20分分の自己校正マップを生成します。\n\n"
            f"開始動画:\n{selected_start_video}\n\n"
            f"マスク方式: {'自動+手動' if mask_mode == 'auto_plus_manual' else ('自動のみ' if mask_mode == 'auto_only' else '手動のみ')}\n\n"
            "対策として自動マスク(時刻表示/グロー領域)を作成し、データ欠損区間は自動スキップします。\n"
            "既存の distortion_map_x.npy / distortion_map_y.npy は上書きされます。\n\n"
            "続行しますか？"
        ):
            return

        module_dir = os.path.dirname(os.path.abspath(__file__))
        map_x_path = os.path.join(module_dir, "distortion_map_x.npy")
        map_y_path = os.path.join(module_dir, "distortion_map_y.npy")
        auto_mask_output_path = os.path.join(module_dir, "distortion_selfcal_auto_mask.png")
        manual_mask_output_path = os.path.join(module_dir, "distortion_selfcal_manual_mask.png")
        metadata_output_path = os.path.join(module_dir, "distortion_selfcal_meta.json")

        # Backup current maps if they exist so the user can roll back easily.
        backup_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_map_x = None
        backup_map_y = None
        try:
            if os.path.exists(map_x_path):
                backup_map_x = os.path.join(module_dir, f"distortion_map_x.backup_{backup_ts}.npy")
                shutil.copy2(map_x_path, backup_map_x)
            if os.path.exists(map_y_path):
                backup_map_y = os.path.join(module_dir, f"distortion_map_y.backup_{backup_ts}.npy")
                shutil.copy2(map_y_path, backup_map_y)
        except Exception as e:
            self.append_log(f"既存ゆがみマップのバックアップ作成に失敗しました: {e}")

        def run_task():
            self.append_log("夜間自己校正マップ生成を開始します... (先頭20分, 自動マスク有効)")
            self.append_log(f"開始動画: {selected_start_video}")
            self.append_log("この動画から後続の分割動画を連続利用して20分分を処理します。")
            self.append_log(f"マスク方式: {'自動+手動' if mask_mode == 'auto_plus_manual' else ('自動のみ' if mask_mode == 'auto_only' else '手動のみ')}")
            self.append_log("注意: 固定カメラの夜空動画を前提とします。欠損区間は自動スキップします。")
            try:
                if selected_manual_mask is not None:
                    try:
                        cv2.imwrite(manual_mask_output_path, selected_manual_mask)
                        self.append_log(f"手動自己校正マスクを保存しました: {manual_mask_output_path}")
                    except Exception as e_save_mask:
                        self.append_log(f"手動自己校正マスクの保存に失敗しました: {e_save_mask}")

                result = distortion_correction.estimate_distortion_map_from_night_sources(
                    sources=self.folder_paths,
                    map_x_path=map_x_path,
                    map_y_path=map_y_path,
                    duration_minutes=20.0,
                    sample_interval_sec=2.0,
                    progress_callback=self.append_log,
                    auto_mask_output_path=auto_mask_output_path,
                    metadata_output_path=metadata_output_path,
                    strength=0.5,
                    start_video_path=selected_start_video,
                    manual_mask=selected_manual_mask,
                    use_auto_mask=use_auto_mask,
                )
                stats = result.get("stats", {}) if isinstance(result, dict) else {}
                sample_count = stats.get("residual_samples_before_fit", "N/A")
                used_obs = stats.get("track_observations_used", "N/A")
                sampled_ok = stats.get("frames_sampled_success", "N/A")
                sampled_ng = stats.get("frames_sampled_failed", "N/A")
                p95_resid = stats.get("p95_residual_mag_px", None)
                used_videos = stats.get("videos_touched_count", "N/A")
                start_video_meta = stats.get("video_path_start", selected_start_video)

                summary_lines = [
                    "夜間自己校正マップ生成が完了しました。",
                    f"開始動画: {start_video_meta}",
                    f"使用動画数: {used_videos}",
                    f"マスク方式: {'自動+手動' if mask_mode == 'auto_plus_manual' else ('自動のみ' if mask_mode == 'auto_only' else '手動のみ')}",
                    f"map_x: {map_x_path}",
                    f"map_y: {map_y_path}",
                    f"自動マスク: {auto_mask_output_path}",
                    f"手動マスク: {manual_mask_output_path if selected_manual_mask is not None else '(未使用)'}",
                    f"メタ情報: {metadata_output_path}",
                    f"サンプル成功/失敗: {sampled_ok} / {sampled_ng}",
                    f"残差サンプル数: {sample_count} (観測使用数: {used_obs})",
                ]
                if p95_resid is not None:
                    try:
                        summary_lines.append(f"残差95%値: {float(p95_resid):.3f} px")
                    except Exception:
                        pass
                if backup_map_x or backup_map_y:
                    backup_info = []
                    if backup_map_x:
                        backup_info.append(f"X: {backup_map_x}")
                    if backup_map_y:
                        backup_info.append(f"Y: {backup_map_y}")
                    summary_lines.append("バックアップ作成済み")
                    summary_lines.extend(backup_info)

                self.append_log("夜間自己校正マップ生成が完了しました。")
                self.append_log(f"  map_x: {map_x_path}")
                self.append_log(f"  map_y: {map_y_path}")
                self.append_log(f"  自動マスク: {auto_mask_output_path}")
                self.append_log(f"  メタ情報: {metadata_output_path}")
                messagebox.showinfo("完了", "\n".join(summary_lines))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.append_log(f"夜間自己校正マップ生成に失敗しました: {e}")
                messagebox.showerror(
                    "エラー",
                    "夜間自己校正マップ生成に失敗しました。\n"
                    "ログを確認してください。\n\n"
                    f"詳細: {e}"
                )

        threading.Thread(target=run_task, daemon=True).start()

    def visualize_distortion_map_callback(self):
        """Visualize distortion map (map_x/map_y) on a Tk canvas as heatmap + vector field."""
        if not self.check_admin_password():
            return

        module_dir = os.path.dirname(os.path.abspath(__file__))
        default_map_x = os.path.join(module_dir, "distortion_map_x.npy")
        default_map_y = os.path.join(module_dir, "distortion_map_y.npy")
        default_meta = os.path.join(module_dir, "distortion_selfcal_meta.json")

        map_x_path = default_map_x
        map_y_path = default_map_y

        if not (os.path.exists(map_x_path) and os.path.exists(map_y_path)):
            messagebox.showinfo(
                "情報",
                "既定のゆがみマップが見つからないため、map_x / map_y ファイルを選択してください。"
            )
            map_x_path = filedialog.askopenfilename(
                title="distortion_map_x.npy を選択",
                initialdir=module_dir,
                filetypes=[("NumPy Array", "*.npy"), ("All Files", "*.*")]
            )
            if not map_x_path:
                return

            inferred_map_y = map_x_path.replace("map_x", "map_y")
            if os.path.exists(inferred_map_y):
                map_y_path = inferred_map_y
            else:
                map_y_path = filedialog.askopenfilename(
                    title="distortion_map_y.npy を選択",
                    initialdir=os.path.dirname(map_x_path),
                    filetypes=[("NumPy Array", "*.npy"), ("All Files", "*.*")]
                )
                if not map_y_path:
                    return

        try:
            map_x = np.load(map_x_path).astype(np.float32)
            map_y = np.load(map_y_path).astype(np.float32)
        except Exception as e:
            messagebox.showerror("エラー", f"ゆがみマップの読み込みに失敗しました:\n{e}")
            return

        if map_x.ndim != 2 or map_y.ndim != 2 or map_x.shape != map_y.shape:
            messagebox.showerror(
                "エラー",
                f"map_x / map_y の形状が不正です。\n"
                f"map_x: shape={getattr(map_x, 'shape', None)}\n"
                f"map_y: shape={getattr(map_y, 'shape', None)}"
            )
            return

        h, w = map_x.shape[:2]
        yy, xx = np.indices((h, w), dtype=np.float32)
        dx = map_x - xx
        dy = map_y - yy
        mag = np.hypot(dx, dy)

        finite_mask = np.isfinite(mag)
        if not np.any(finite_mask):
            messagebox.showerror("エラー", "ゆがみマップがすべて非数値です。")
            return

        valid_mag = mag[finite_mask]
        mag_max = float(np.max(valid_mag))
        mag_p50 = float(np.percentile(valid_mag, 50.0))
        mag_p95 = float(np.percentile(valid_mag, 95.0))
        mag_p99 = float(np.percentile(valid_mag, 99.0))
        norm_denom = max(1e-6, mag_p99 if mag_p99 > 0 else mag_max)

        heat_norm = np.clip(mag / norm_denom, 0.0, 1.0)
        heat_u8 = (heat_norm * 255.0).astype(np.uint8)
        heat_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)
        heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

        max_disp_w = 1100
        max_disp_h = 720
        scale = min(max_disp_w / float(w), max_disp_h / float(h), 1.0)
        disp_w = max(1, int(round(w * scale)))
        disp_h = max(1, int(round(h * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_NEAREST
        heat_disp = cv2.resize(heat_rgb, (disp_w, disp_h), interpolation=interp)

        meta_info = {}
        if os.path.exists(default_meta):
            try:
                with open(default_meta, "r", encoding="utf-8") as f:
                    meta_info = json.load(f)
            except Exception:
                meta_info = {}

        win = Toplevel(self)
        win.title("ゆがみマップ可視化")
        win.geometry(f"{min(disp_w + 80, 1280)}x{min(disp_h + 220, 980)}")
        win.transient(self)

        header = ttk.Frame(win, padding=10)
        header.pack(fill=tk.X)

        info_lines = [
            f"map_x: {map_x_path}",
            f"map_y: {map_y_path}",
            f"サイズ: {w} x {h}",
            f"変位量 [px]  p50={mag_p50:.3f}, p95={mag_p95:.3f}, p99={mag_p99:.3f}, max={mag_max:.3f}",
            "表示: 背景=変位量ヒートマップ, 矢印=map_x/map_y の変位ベクトル",
        ]
        if isinstance(meta_info, dict):
            vstart = meta_info.get("video_path_start") or meta_info.get("video_path")
            vcount = meta_info.get("videos_touched_count")
            if vstart:
                info_lines.append(f"自己校正開始動画: {vstart}")
            if vcount is not None:
                info_lines.append(f"自己校正で使用した動画数: {vcount}")
        ttk.Label(header, text="\n".join(info_lines), justify=tk.LEFT).pack(anchor=tk.W)

        canvas_frame = ttk.Frame(win, padding=(10, 0, 10, 10))
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        canvas = Canvas(canvas_frame, bg="#111111", highlightthickness=1, highlightbackground="#444444")
        canvas.pack(fill=tk.BOTH, expand=True)

        tk_img = ImageTk.PhotoImage(Image.fromarray(heat_disp))
        canvas.create_image(0, 0, image=tk_img, anchor="nw")
        canvas.config(scrollregion=(0, 0, disp_w, disp_h))
        win._distortion_map_preview_tk = tk_img  # prevent GC

        # Draw a sparse vector field on top of the heatmap.
        grid_step = max(48, min(120, int(min(w, h) / 10)))
        mag_thresh = max(0.1, mag_p95 * 0.08)
        vec_scale = float(np.clip(24.0 / max(mag_p95, 0.2), 3.0, 35.0))

        def rgb_to_hex(rgb_arr):
            r, g, b = int(rgb_arr[0]), int(rgb_arr[1]), int(rgb_arr[2])
            return f"#{r:02x}{g:02x}{b:02x}"

        for y in range(grid_step // 2, h, grid_step):
            for x in range(grid_step // 2, w, grid_step):
                m = float(mag[y, x])
                if not np.isfinite(m) or m < mag_thresh:
                    continue
                x0 = x * scale
                y0 = y * scale
                vx = float(dx[y, x]) * scale * vec_scale
                vy = float(dy[y, x]) * scale * vec_scale
                x1 = x0 + vx
                y1 = y0 + vy
                color = rgb_to_hex(heat_rgb[y, x])
                canvas.create_line(
                    x0, y0, x1, y1,
                    fill=color,
                    width=2 if m >= mag_p95 else 1,
                    arrow=tk.LAST,
                    arrowshape=(8, 10, 3)
                )
                canvas.create_oval(x0 - 1.5, y0 - 1.5, x0 + 1.5, y0 + 1.5, fill=color, outline="")

        # Corner/center guides
        guide_color = "#FFFFFF"
        for gx, gy, label in (
            (0, 0, "TL"),
            (w - 1, 0, "TR"),
            (0, h - 1, "BL"),
            (w - 1, h - 1, "BR"),
            (w / 2, h / 2, "C"),
        ):
            px = gx * scale
            py = gy * scale
            canvas.create_line(px - 8, py, px + 8, py, fill=guide_color, width=1)
            canvas.create_line(px, py - 8, px, py + 8, fill=guide_color, width=1)
            canvas.create_text(px + 12, py + 12, text=label, fill=guide_color, anchor="nw", font=("Arial", 9, "bold"))

        footer = ttk.Frame(win, padding=(10, 0, 10, 10))
        footer.pack(fill=tk.X)
        ttk.Label(
            footer,
            text="注: この可視化は map_x/map_y の変位を表示するもので、補正の良し悪しは別途プレートソルブ誤差で評価してください。",
            foreground="#87CEEB"
        ).pack(anchor=tk.W)

    def analyze_angles_callback(self):
        """Callback for the 'Angle Distribution Analysis' button."""
        if not self.check_admin_password():
            return

        info_files = filedialog.askopenfilenames(
            title="角度分布分析の対象ファイル",
            filetypes=(("info.txt", "*.txt"), ("すべてのファイル", "*.*")),
        )
        if not info_files:
            return

        ra_str = simpledialog.askstring("放射点入力", "放射点の赤経 (RA) を度数 (deg) で入力してください:\n(例: 45.0)")
        if ra_str is None: return
        try:
            radiant_ra = float(ra_str)
        except ValueError:
            messagebox.showerror("エラー", "有効な数値を入力してください。")
            return

        dec_str = simpledialog.askstring("放射点入力", "放射点の赤緯 (Dec) を度数 (deg) で入力してください:\n(例: 30.0)")
        if dec_str is None: return
        try:
            radiant_dec = float(dec_str)
        except ValueError:
            messagebox.showerror("エラー", "有効な数値を入力してください。")
            return

        output_path = filedialog.asksaveasfilename(
            title="角度分布グラフの保存先",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("All Files", "*")]
        )
        
        if not output_path:
            return

        def run_task():
            self.append_log("角度分布分析を開始します...")
            success, msg = meteor_angle_analysis.analyze_angles(
                info_files,
                radiant_ra,
                radiant_dec,
                output_path
            )
            if success:
                messagebox.showinfo("完了", msg)
                self.append_log(f"角度分布分析完了: {msg}")
            else:
                messagebox.showerror("エラー", msg)
                self.append_log(f"角度分布分析失敗: {msg}")

        threading.Thread(target=run_task, daemon=True).start()

