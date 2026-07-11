from gui_common import *


class TimelapseDragDropWindow(Toplevel):
    """タイムラプス作成用のドラッグ＆ドロップウィンドウ"""
    
    def __init__(self, parent, log_callback):
        super().__init__(parent)
        self.parent = parent
        self.log_callback = log_callback
        self.dropped_paths = []
        self.timelapse_mask = None  # タイムラプス用マスク
        self.timelapse_timestamp_enabled_var = tk.BooleanVar(
            value=config.TIMELAPSE_TIMESTAMP_ENABLED
        )
        self.timelapse_timestamp_position_var = tk.StringVar(value="右下")
        self.timelapse_timestamp_size_var = tk.StringVar(
            value=str(config.TIMELAPSE_TIMESTAMP_SIZE_PERCENT)
        )
        self.temporal_mean_radius_var = tk.IntVar(
            value=config.TIMELAPSE_TEMPORAL_MEAN_RADIUS_FRAMES
        )
        self.temporal_mean_summary_var = tk.StringVar()
        self.timelapse_annotation_enabled_var = tk.BooleanVar(
            value=getattr(config, "TIMELAPSE_LOCAL_ANNOTATION_ENABLED", False)
        )
        self.timelapse_annotation_calibration_var = tk.StringVar(
            value=getattr(config, "TIMELAPSE_ANNOTATION_CALIBRATION_PATH", "") or ""
        )
        
        self.title("タイムラプス作成")
        self.geometry("500x870")
        self.resizable(True, True)
        
        self.setup_ui()
        
        self.transient(parent)
        self.grab_set()
    
    def setup_ui(self):
        main_frame = ttk.Frame(self, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        drop_frame = ttk.LabelFrame(main_frame, text="ファイル / フォルダ", padding=10)
        drop_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.drop_label = ttk.Label(
            drop_frame, 
            text="ここにフォルダや動画ファイルを\nドラッグ＆ドロップしてください",
            relief=tk.SOLID, 
            padding=30, 
            anchor=tk.CENTER,
            borderwidth=2,
            justify=tk.CENTER
        )
        self.drop_label.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.drop_label.drop_target_register(DND_FILES)
        self.drop_label.dnd_bind('<<Drop>>', self.on_drop)
        
        list_frame = ttk.Frame(drop_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.listbox = tk.Listbox(
            list_frame, 
            height=5, 
            bg="#3A4D6B", 
            fg="#EAEAEA",
            relief=tk.FLAT,
            highlightthickness=0
        )
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.config(yscrollcommand=scrollbar.set)
        
        ttk.Button(drop_frame, text="リストをクリア", command=self.clear_list).pack(anchor=tk.E, pady=(5, 0))
        
        duration_frame = ttk.LabelFrame(main_frame, text="動画の長さ", padding=10)
        duration_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.duration_var = tk.IntVar(value=30)
        
        duration_options = ttk.Frame(duration_frame)
        duration_options.pack()
        
        ttk.Radiobutton(duration_options, text="15秒", variable=self.duration_var, value=15).pack(side=tk.LEFT, padx=15)
        ttk.Radiobutton(duration_options, text="30秒", variable=self.duration_var, value=30).pack(side=tk.LEFT, padx=15)
        ttk.Radiobutton(duration_options, text="60秒", variable=self.duration_var, value=60).pack(side=tk.LEFT, padx=15)
        
        mask_frame = ttk.LabelFrame(main_frame, text="マスク設定", padding=10)
        mask_frame.pack(fill=tk.X, pady=(0, 10))
        
        mask_controls = ttk.Frame(mask_frame)
        mask_controls.pack(fill=tk.X)
        
        self.mask_btn = ttk.Button(mask_controls, text="マスク作成", command=self.create_timelapse_mask)
        self.mask_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_mask_btn = ttk.Button(mask_controls, text="マスクをクリア", command=self.clear_timelapse_mask, state=tk.DISABLED)
        self.clear_mask_btn.pack(side=tk.LEFT, padx=5)
        
        self.mask_status_label = ttk.Label(mask_controls, text="マスクなし")
        self.mask_status_label.pack(side=tk.LEFT, padx=10)

        timestamp_frame = ttk.LabelFrame(main_frame, text="時刻表示", padding=10)
        timestamp_frame.pack(fill=tk.X, pady=(0, 10))

        timestamp_check = ttk.Checkbutton(
            timestamp_frame,
            text="時刻を表示（ファイル作成時刻を基準）",
            variable=self.timelapse_timestamp_enabled_var,
            command=self._toggle_timelapse_timestamp_settings,
        )
        timestamp_check.pack(side=tk.LEFT)
        ttk.Label(timestamp_frame, text="位置:").pack(side=tk.LEFT, padx=(12, 3))
        self.timelapse_timestamp_position_box = ttk.Combobox(
            timestamp_frame,
            textvariable=self.timelapse_timestamp_position_var,
            values=("右下", "左下", "右上", "左上"),
            state="readonly",
            width=6,
        )
        self.timelapse_timestamp_position_box.pack(side=tk.LEFT)
        ttk.Label(timestamp_frame, text="文字サイズ:").pack(side=tk.LEFT, padx=(10, 3))
        self.timelapse_timestamp_size_spin = ttk.Spinbox(
            timestamp_frame,
            from_=0.8,
            to=4.0,
            increment=0.1,
            textvariable=self.timelapse_timestamp_size_var,
            width=4,
        )
        self.timelapse_timestamp_size_spin.pack(side=tk.LEFT)
        ttk.Label(timestamp_frame, text="%（画面高）").pack(side=tk.LEFT, padx=(3, 0))
        self._toggle_timelapse_timestamp_settings()

        annotation_frame = ttk.LabelFrame(
            main_frame,
            text="ローカル広角星空注釈",
            padding=10,
        )
        annotation_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Checkbutton(
            annotation_frame,
            text="歪み補正した天球グリッドを描画（外部API不使用）",
            variable=self.timelapse_annotation_enabled_var,
            command=self._toggle_timelapse_annotation_settings,
        ).pack(anchor=tk.W)
        ttk.Label(
            annotation_frame,
            text="当晩の広角歪み・向きを自動較正します。初回は通常より時間がかかります。",
            foreground="gray",
        ).pack(anchor=tk.W, pady=(2, 4))
        calibration_row = ttk.Frame(annotation_frame)
        calibration_row.pack(fill=tk.X)
        ttk.Label(calibration_row, text="較正ファイル（任意）:").pack(side=tk.LEFT)
        self.timelapse_annotation_calibration_entry = ttk.Entry(
            calibration_row,
            textvariable=self.timelapse_annotation_calibration_var,
        )
        self.timelapse_annotation_calibration_entry.pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5)
        )
        self.timelapse_annotation_calibration_button = ttk.Button(
            calibration_row,
            text="選択",
            command=self._choose_timelapse_annotation_calibration,
        )
        self.timelapse_annotation_calibration_button.pack(side=tk.LEFT)
        self._toggle_timelapse_annotation_settings()

        mean_frame = ttk.LabelFrame(main_frame, text="ノイズ低減（時間平均）", padding=10)
        mean_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(
            mean_frame,
            text="各採用フレームの前後何枚を平均するか選びます。値を大きくするとノイズは減りますが、動くものは薄まります。",
            wraplength=450,
        ).pack(anchor=tk.W)

        mean_controls = ttk.Frame(mean_frame)
        mean_controls.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(mean_controls, text="前後:").pack(side=tk.LEFT)
        self.temporal_mean_scale = ttk.Scale(
            mean_controls,
            from_=0,
            to=100,
            variable=self.temporal_mean_radius_var,
            command=self._on_temporal_mean_scale,
        )
        self.temporal_mean_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 8))
        self.temporal_mean_spin = ttk.Spinbox(
            mean_controls,
            from_=0,
            to=100,
            increment=1,
            textvariable=self.temporal_mean_radius_var,
            width=4,
            command=self._update_temporal_mean_summary,
        )
        self.temporal_mean_spin.pack(side=tk.LEFT)
        ttk.Label(mean_controls, text="枚").pack(side=tk.LEFT, padx=(3, 0))

        ttk.Label(mean_frame, textvariable=self.temporal_mean_summary_var).pack(anchor=tk.W, pady=(4, 0))
        preset_frame = ttk.Frame(mean_frame)
        preset_frame.pack(anchor=tk.W, pady=(6, 0))
        ttk.Label(preset_frame, text="プリセット:").pack(side=tk.LEFT)
        for label, radius in (("なし", 0), ("軽め", 5), ("標準", 15), ("強め", 50)):
            ttk.Button(
                preset_frame,
                text=label,
                command=lambda value=radius: self._set_temporal_mean_radius(value),
            ).pack(side=tk.LEFT, padx=(5, 0))
        self.temporal_mean_spin.bind("<FocusOut>", lambda _event: self._update_temporal_mean_summary())
        self.temporal_mean_spin.bind("<Return>", lambda _event: self._update_temporal_mean_summary())
        self._update_temporal_mean_summary()
        
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X)
        
        ttk.Button(btn_frame, text="作成開始", command=self.start_creation).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="キャンセル", command=self.destroy).pack(side=tk.LEFT, padx=5)

    def _toggle_timelapse_timestamp_settings(self):
        state = tk.NORMAL if self.timelapse_timestamp_enabled_var.get() else tk.DISABLED
        self.timelapse_timestamp_position_box.config(state="readonly" if state == tk.NORMAL else tk.DISABLED)
        self.timelapse_timestamp_size_spin.config(state=state)

    def _toggle_timelapse_annotation_settings(self):
        state = tk.NORMAL if self.timelapse_annotation_enabled_var.get() else tk.DISABLED
        self.timelapse_annotation_calibration_entry.config(state=state)
        self.timelapse_annotation_calibration_button.config(state=state)

    def _choose_timelapse_annotation_calibration(self):
        path = filedialog.askopenfilename(
            title="広角星空較正ファイルを選択",
            filetypes=[
                ("Calibration", "*.json *.wcs *.fits *.fit"),
                ("All Files", "*"),
            ],
        )
        if path:
            self.timelapse_annotation_calibration_var.set(path)

    def _on_temporal_mean_scale(self, value):
        """Scale is continuous, so convert its value to a whole frame count."""
        self._set_temporal_mean_radius(int(round(float(value))))

    def _set_temporal_mean_radius(self, value):
        radius = max(0, min(100, int(value)))
        if self.temporal_mean_radius_var.get() != radius:
            self.temporal_mean_radius_var.set(radius)
        self._update_temporal_mean_summary()

    def _update_temporal_mean_summary(self):
        try:
            radius = int(self.temporal_mean_radius_var.get())
        except (TypeError, ValueError, tk.TclError):
            radius = config.TIMELAPSE_TEMPORAL_MEAN_RADIUS_FRAMES
        radius = max(0, min(100, radius))
        if self.temporal_mean_radius_var.get() != radius:
            self.temporal_mean_radius_var.set(radius)
        if radius == 0:
            summary = "単独フレームを使用します（平均しません）"
        else:
            summary = f"前後{radius}枚を含む、合計最大{radius * 2 + 1}枚の平均画像を使用します"
        self.temporal_mean_summary_var.set(summary)
    
    def on_drop(self, event):
        """ドラッグ＆ドロップのイベントハンドラ"""
        # splitlistを使用してパスを分解
        try:
            paths = self.tk.splitlist(event.data)
        except:
            paths = [event.data]
        
        for path in paths:
            path = path.strip('{}')
            if path and path not in self.dropped_paths:
                self.dropped_paths.append(path)
                self.listbox.insert(tk.END, os.path.basename(path))
        
        self.update_drop_label()
    
    def update_drop_label(self):
        """ドロップラベルのテキストを更新"""
        if self.dropped_paths:
            self.drop_label.config(text=f"{len(self.dropped_paths)}個のアイテムが追加されました\n(さらに追加できます)")
        else:
            self.drop_label.config(text="ここにフォルダや動画ファイルを\nドラッグ＆ドロップしてください")
    
    def clear_list(self):
        """リストをクリア"""
        self.dropped_paths.clear()
        self.listbox.delete(0, tk.END)
        self.update_drop_label()
    
    def create_timelapse_mask(self):
        """タイムラプス用マスクを作成"""
        if not self.dropped_paths:
            messagebox.showwarning("警告", "先にファイルまたはフォルダをドロップしてください。")
            return
        
        # 最初のファイルからフレームを取得
        from pathlib import Path
        from PIL import Image, ImageTk, ImageDraw
        
        first_frame = None
        for path in self.dropped_paths:
            if os.path.isfile(path):
                ext = Path(path).suffix.lower()
                if ext in {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v'}:
                    cap = cv2.VideoCapture(path)
                    ret, first_frame = cap.read()
                    cap.release()
                    if ret:
                        break
                elif ext in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}:
                    first_frame = cv2.imread(path)
                    if first_frame is not None:
                        break
            elif os.path.isdir(path):
                # ディレクトリから最初の動画または画像を探す
                for f in sorted(os.listdir(path)):
                    fpath = os.path.join(path, f)
                    if os.path.isfile(fpath):
                        ext = Path(fpath).suffix.lower()
                        if ext in {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v'}:
                            cap = cv2.VideoCapture(fpath)
                            ret, first_frame = cap.read()
                            cap.release()
                            if ret:
                                break
                        elif ext in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}:
                            first_frame = cv2.imread(fpath)
                            if first_frame is not None:
                                break
                if first_frame is not None:
                    break
        
        if first_frame is None:
            messagebox.showerror("エラー", "フレームを取得できませんでした。")
            return
        
        mask_win = Toplevel(self)
        mask_win.title("タイムラプス用マスク作成")
        mask_win.geometry("1000x700")
        mask_win.grab_set()
        mask_win.transient(self)
        
        orig_h, orig_w = first_frame.shape[:2]
        disp_w, disp_h = 960, 540
        scale = min(disp_w / orig_w, disp_h / orig_h)
        disp_w, disp_h = int(orig_w * scale), int(orig_h * scale)
        
        frame_disp = cv2.resize(first_frame, (disp_w, disp_h))
        tk_image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB)))
        
        canvas = Canvas(mask_win, width=disp_w, height=disp_h, cursor="circle")
        canvas.pack(pady=5)
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
        canvas.image = tk_image
        
        mask_data_disp = Image.new("L", (disp_w, disp_h), 0)
        draw = ImageDraw.Draw(mask_data_disp)
        
        brush_radius = tk.IntVar(value=30)
        
        def paint(event):
            r = brush_radius.get()
            canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r, fill='red', outline='red', tags="paint")
            draw.ellipse((event.x - r, event.y - r, event.x + r, event.y + r), fill=255)
        
        canvas.bind("<B1-Motion>", paint)
        canvas.bind("<Button-1>", paint)
        
        controls = ttk.Frame(mask_win)
        controls.pack(fill=tk.X, padx=10)
        ttk.Label(controls, text="ブラシサイズ:").pack(side=tk.LEFT)
        ttk.Scale(controls, from_=5, to=100, orient=tk.HORIZONTAL, variable=brush_radius).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        def clear_drawing():
            canvas.delete("paint")
            draw.rectangle([0, 0, disp_w, disp_h], fill=0)
        
        ttk.Button(controls, text="クリア", command=clear_drawing).pack(side=tk.LEFT)
        
        def on_ok():
            mask_np_disp = np.array(mask_data_disp)
            # 白黒反転（描画部分=マスク=0、非描画部分=表示=255）
            if mask_np_disp.max() > 0:
                mask_resized = cv2.resize(mask_np_disp, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                final_mask = cv2.bitwise_not(mask_resized)
            else:
                final_mask = None  # マスクなし
            
            self.timelapse_mask = final_mask
            if final_mask is not None:
                self.mask_status_label.config(text="マスクあり")
                self.clear_mask_btn.config(state=tk.NORMAL)
            mask_win.destroy()
        
        btn_frame = ttk.Frame(mask_win)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="キャンセル", command=mask_win.destroy).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(mask_win, text="※塗りつぶした領域が黒くマスクされます", foreground="gray").pack()
    
    def clear_timelapse_mask(self):
        """タイムラプス用マスクをクリア"""
        self.timelapse_mask = None
        self.mask_status_label.config(text="マスクなし")
        self.clear_mask_btn.config(state=tk.DISABLED)
    
    def start_creation(self):
        """タイムラプス作成を開始"""
        if not self.dropped_paths:
            messagebox.showwarning("警告", "ファイルまたはフォルダをドロップしてください。")
            return
        
        # 保存先を選択
        default_output = timelapse_creator.get_default_output_path()
        output_path = filedialog.asksaveasfilename(
            title="タイムラプス動画の保存先",
            initialdir=os.path.dirname(default_output),
            initialfile=os.path.basename(default_output),
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4"), ("AVI Video", "*.avi"), ("All Files", "*")]
        )
        
        if not output_path:
            return
        try:
            output_path = self.parent._ensure_date_prefix(output_path)
        except Exception:
            pass
        
        duration = self.duration_var.get()
        paths = list(self.dropped_paths)
        mask = self.timelapse_mask  # マスクを保存
        try:
            timestamp_size = float(self.timelapse_timestamp_size_var.get())
        except (TypeError, ValueError):
            timestamp_size = config.TIMELAPSE_TIMESTAMP_SIZE_PERCENT
        timestamp_settings = {
            "enabled": self.timelapse_timestamp_enabled_var.get(),
            "position": self.timelapse_timestamp_position_var.get(),
            "size_percent": timestamp_size,
        }
        annotation_settings = {
            "enabled": self.timelapse_annotation_enabled_var.get(),
            "calibration_path": self.timelapse_annotation_calibration_var.get().strip() or None,
        }
        try:
            temporal_mean_radius = int(self.temporal_mean_radius_var.get())
        except (TypeError, ValueError, tk.TclError):
            temporal_mean_radius = config.TIMELAPSE_TEMPORAL_MEAN_RADIUS_FRAMES
        temporal_mean_radius = max(0, min(100, temporal_mean_radius))
        
        # ウィンドウを閉じる
        self.destroy()
        
        mask_status = "あり" if mask is not None else "なし"
        annotation_status = "あり（ローカル）" if annotation_settings["enabled"] else "なし"
        self.log_callback(
            f"タイムラプス作成を開始します... (長さ: {duration}秒, "
            f"{len(paths)}個のアイテム, マスク: {mask_status}, 星空注釈: {annotation_status})"
        )
        
        def create_task(progress_callback):
            return timelapse_creator.create_timelapse(
                paths,
                output_path,
                target_duration_seconds=duration,
                progress_callback=progress_callback,
                mask=mask,
                timestamp_settings=timestamp_settings,
                temporal_mean_radius_frames=temporal_mean_radius,
                annotation_settings=annotation_settings,
            )

        task_runner = getattr(self.parent, "_run_synthesis_task_async", None)
        if not callable(task_runner):
            messagebox.showerror("エラー", "タイムラプス作成機能を開始できませんでした。")
            return
        task_runner(
            create_task,
            output_path=output_path,
            item_label="タイムラプス動画",
        )


class ProcessingOptionDialog(tk.Toplevel):
    def __init__(self, parent):
        print("DEBUG: ProcessingOptionDialog initialized")
        super().__init__(parent)
        self.title("処理オプション")
        self.result = None
        self.geometry("500x320") 
        self.resizable(False, False)
        
        # メインフレームを作成して全体に配置（テーマの背景色を適用するため）
        self.main_frame = ttk.Frame(self, padding="20 20 20 10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(self.main_frame, text="比較明合成オプション", font=("", 14, "bold")).pack(anchor=tk.W, pady=(0, 15))
        
        mode_frame = ttk.LabelFrame(self.main_frame, text="モード選択", padding=10)
        mode_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.mode_var = tk.IntVar(value=0)
        
        ttk.Radiobutton(mode_frame, text="通常合成 (AIを使用しない)", variable=self.mode_var, value=0).pack(anchor=tk.W, pady=5)
        self.rb_bright = ttk.Radiobutton(mode_frame, text="明るいエリアをマスク (AI検出)", variable=self.mode_var, value=1)
        self.rb_bright.pack(anchor=tk.W, pady=5)
        self.rb_meteor = ttk.Radiobutton(mode_frame, text="流星のみ合成 (AI検出)", variable=self.mode_var, value=2)
        self.rb_meteor.pack(anchor=tk.W, pady=5)
        
        # VRAM warning
        warning_frame = ttk.Frame(self.main_frame)
        warning_frame.pack(fill=tk.X, pady=(5, 10))
        # 黄色 (#FFD700) に変更
        ttk.Label(warning_frame, text="※AI検出を選択時、VRAMが7GB未満の場合は", font=("", 9), foreground="#FFD700").pack(anchor=tk.W)
        ttk.Label(warning_frame, text="  動作が非常に遅くなる可能性があります。", font=("", 9), foreground="#FFD700").pack(anchor=tk.W)
        
        btn_frame = ttk.Frame(self.main_frame)
        btn_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        ttk.Button(btn_frame, text="キャンセル", command=self.destroy).pack(side=tk.RIGHT, padx=5)
        self.next_btn = ttk.Button(btn_frame, text="次へ", command=self.on_ok, state=tk.NORMAL) # 最初から有効化
        self.next_btn.pack(side=tk.RIGHT, padx=5)
        
        self.transient(parent)
        self.grab_set()
        
        self.update_idletasks()
        try:
            x = parent.winfo_x() + (parent.winfo_width() // 2) - (self.winfo_width() // 2)
            y = parent.winfo_y() + (parent.winfo_height() // 2) - (self.winfo_height() // 2)
            self.geometry(f"+{x}+{y}")
        except:
            pass
            
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.wait_window(self)
    
    def on_ok(self):
        self.result = self.mode_var.get()
        self.destroy()
    
    def _setup_help_tooltip(self, widget):
        pass


