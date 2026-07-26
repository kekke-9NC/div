from gui_common import *


class MaskMixin:
    def create_summary_settings_window(self):
        win = Toplevel(self)
        win.title("概要動画設定")
        win.geometry("500x450")
        win.grab_set()
        win.transient(self)

        temp_config = [item.copy() for item in self.summary_video_config]
        
        ttk.Label(win, text="概要動画に含める項目と順序:").pack(pady=5, padx=10, anchor='w')
        list_frame = ttk.Frame(win); list_frame.pack(fill=tk.BOTH, expand=True, padx=10)
        listbox = tk.Listbox(
            list_frame,
            selectmode=tk.SINGLE,
            exportselection=False,
            bg=ui_theme.COLORS["field"],
            fg=ui_theme.COLORS["text"],
            selectbackground=ui_theme.COLORS["selection"],
            selectforeground=ui_theme.COLORS["text"],
            highlightthickness=0,
        )
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        check_vars = [tk.BooleanVar(value=item['enabled']) for item in temp_config]
        for i, item in enumerate(temp_config):
            listbox.insert(tk.END, item['name'])

        def move_item(direction):
            idx = listbox.curselection()[0] if listbox.curselection() else -1
            if idx == -1: return
            new_idx = idx + direction
            if 0 <= new_idx < listbox.size():
                item = listbox.get(idx)
                listbox.delete(idx); listbox.insert(new_idx, item)
                listbox.selection_set(new_idx); listbox.activate(new_idx)
                temp_config.insert(new_idx, temp_config.pop(idx))
                check_vars.insert(new_idx, check_vars.pop(idx))

        btn_panel = ttk.Frame(list_frame); btn_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(5,0))
        ttk.Button(btn_panel, text="↑", command=lambda: move_item(-1)).pack(pady=2)
        ttk.Button(btn_panel, text="↓", command=lambda: move_item(1)).pack(pady=2)

        check_frame = ttk.Frame(win); check_frame.pack(fill=tk.X, padx=10, pady=5)
        for i, item in enumerate(temp_config):
            ttk.Checkbutton(check_frame, text=item['name'], variable=check_vars[i]).pack(anchor='w')

        def on_ok():
            for i, var in enumerate(check_vars):
                temp_config[i]['enabled'] = var.get()
            self.summary_video_config = temp_config
            self.append_log("概要動画の設定を更新しました。")
            win.destroy()

        ok_cancel_frame = ttk.Frame(win); ok_cancel_frame.pack(pady=10)
        ttk.Button(ok_cancel_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(ok_cancel_frame, text="キャンセル", command=win.destroy).pack(side=tk.LEFT, padx=5)

    def create_mask_window(self, is_plate_solve_mask: bool):
        base_image_path = None
        is_rtsp_source = False
        if is_plate_solve_mask:
            base_image_path = self.plate_solve_video_path_var.get()
            if not base_image_path:
                messagebox.showwarning("情報", "まずプレートソルブ用の動画を選択してください。")
                return
            window_title = "プレートソルブ用マスク作成"
        else:
            window_title = "検出マスク作成"
            source_folders = self.folder_paths or ([self.periodic_dir_var.get()] if self.periodic_scan_var.get() and self.periodic_dir_var.get() else [])
            for folder in source_folders:
                videos = sorted([p for p in Path(folder).rglob('*') if p.suffix.lower() in config.PERIODIC_VIDEO_EXTENSIONS])
                if videos:
                    base_image_path = str(videos[0])
                    break
            
            # フォルダに動画がなくRTSP URLがある場合はRTSPから取得
            if not base_image_path and self.rtsp_urls:
                base_image_path = self.rtsp_urls[0]
                is_rtsp_source = True
        
        if not base_image_path:
            messagebox.showwarning("情報", "マスク作成の元となる動画ソースが見つかりません。")
            return

        try:
            cap = cv2.VideoCapture(base_image_path)
            if is_rtsp_source:
                # RTSPのタイムアウト設定
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise ValueError("動画またはRTSPストリームからフレームを読み込めませんでした。")
        except Exception as e:
            messagebox.showerror("エラー", f"マスク作成用画像の読み込みに失敗しました:\n{e}")
            return


        win = Toplevel(self)
        win.title(window_title)
        win.geometry("1000x700")
        win.grab_set()

        orig_h, orig_w = frame.shape[:2]
        disp_w, disp_h = 960, 540
        scale = min(disp_w / orig_w, disp_h / orig_h)
        disp_w, disp_h = int(orig_w * scale), int(orig_h * scale)
        
        frame_disp = cv2.resize(frame, (disp_w, disp_h))
        tk_image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB)))

        canvas = Canvas(win, width=disp_w, height=disp_h, cursor="circle")
        canvas.pack(pady=5)
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
        canvas.image = tk_image

        mask_data_disp = Image.new("L", (disp_w, disp_h), 0)
        draw = ImageDraw.Draw(mask_data_disp)
        
        brush_radius = tk.IntVar(value=30)
        def paint(event):
            r = brush_radius.get()
            canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r, fill='white', outline='white', tags="paint")
            draw.ellipse((event.x - r, event.y - r, event.x + r, event.y + r), fill=255)
        canvas.bind("<B1-Motion>", paint)
        canvas.bind("<Button-1>", paint)

        controls = ttk.Frame(win); controls.pack(fill=tk.X, padx=10)
        ttk.Label(controls, text="ブラシサイズ:").pack(side=tk.LEFT)
        ttk.Scale(controls, from_=5, to=100, orient=tk.HORIZONTAL, variable=brush_radius).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(controls, text="クリア", command=lambda: [canvas.delete("paint"), draw.rectangle([0,0,disp_w,disp_h], fill=0)]).pack(side=tk.LEFT)

        def on_ok():
            mask_np_disp = np.array(mask_data_disp)
            final_mask = cv2.bitwise_not(cv2.resize(mask_np_disp, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)) if mask_np_disp.max() > 0 else np.full((orig_h, orig_w), 255, dtype=np.uint8)
            
            if is_plate_solve_mask:
                self.plate_solve_mask_image = final_mask
                self.preview_mask(self.plate_solve_mask_image, self.ps_mask_preview_label, "PSマスク")
            else:
                self.mask_image = final_mask
                self.mask_path_var.set("作成済み (描画)")
                self.apply_mask_var.set(True)
                self.preview_mask(self.mask_image, self.mask_preview_label, "検出マスク")
            win.destroy()
        btn_frame = ttk.Frame(win); btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="キャンセル", command=win.destroy).pack(side=tk.LEFT, padx=5)

    def create_rtsp_mask(self):
        """RTSPストリームからマスクを作成する"""
        # 選択されているRTSP URLを取得、選択がなければ最初のURLを使用
        if self.rtsp_selected_indices:
            selected_index = min(self.rtsp_selected_indices)
            rtsp_url = self.rtsp_urls[selected_index]
        elif self.rtsp_urls:
            rtsp_url = self.rtsp_urls[0]
        else:
            messagebox.showwarning("警告", "RTSPストリームを追加してください。")
            return

        is_plate_solve_mask = self._select_rtsp_mask_type()
        if is_plate_solve_mask is None:
            return
        
        progress_win = Toplevel(self)
        progress_win.title("接続中")
        progress_win.geometry("300x100")
        progress_win.transient(self)
        progress_win.grab_set()
        progress_win.resizable(False, False)
        
        ttk.Label(progress_win, text="RTSPストリームに接続中...\nしばらくお待ちください。").pack(pady=15)
        cancel_flag = threading.Event()
        
        def on_cancel():
            cancel_flag.set()
            progress_win.destroy()
        
        cancel_btn = ttk.Button(progress_win, text="キャンセル", command=on_cancel)
        cancel_btn.pack(pady=5)
        
        progress_win.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - progress_win.winfo_width()) // 2
        y = self.winfo_y() + (self.winfo_height() - progress_win.winfo_height()) // 2
        progress_win.geometry(f"+{x}+{y}")
        
        result_holder = {'frame': None, 'error': None}
        
        def fetch_frame():
            try:
                cap = utils.create_rtsp_capture(rtsp_url)
                if not cap.isOpened():
                    result_holder['error'] = "RTSPストリームを開けませんでした。"
                    return
                ret, frame = cap.read()
                cap.release()
                if cancel_flag.is_set():
                    return
                if not ret or frame is None:
                    result_holder['error'] = "RTSPストリームからフレームを読み込めませんでした。"
                else:
                    frame = self.apply_rtsp_dark_to_frame(frame)
                    result_holder['frame'] = frame
            except Exception as e:
                result_holder['error'] = str(e)
        
        fetch_thread = threading.Thread(target=fetch_frame, daemon=True)
        fetch_thread.start()
        
        def check_thread():
            if cancel_flag.is_set():
                return
            if fetch_thread.is_alive():
                self.after(100, check_thread)
            else:
                try:
                    progress_win.destroy()
                except tk.TclError:
                    pass
                if result_holder['error']:
                    messagebox.showerror("エラー", f"RTSPからのフレーム取得に失敗しました:\n{result_holder['error']}")
                elif result_holder['frame'] is not None:
                    self._open_rtsp_mask_window(result_holder['frame'], is_plate_solve_mask)
        
        self.after(100, check_thread)

    def _select_rtsp_mask_type(self) -> Optional[bool]:
        """RTSPマスク作成時にマスクの用途を選択する"""
        selection = {'is_plate_solve_mask': None}
        style = ttk.Style(self)
        bg_color = self.cget("background") or style.lookup("TFrame", "background") or "#2E3F5B"
        sub_fg_color = "#AFC0DA"

        dialog = Toplevel(self)
        dialog.title("マスク種別の選択")
        dialog.geometry("440x190")
        dialog.transient(self)
        dialog.grab_set()
        dialog.resizable(False, False)
        dialog.configure(background=bg_color)
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)

        container = tk.Frame(dialog, bg=bg_color, padx=18, pady=14)
        container.pack(fill=tk.BOTH, expand=True)

        ttk.Label(container, text="作成するマスクを選択してください。").pack(pady=(4, 8))
        tk.Label(
            container,
            text="RTSPから取得したフレームでマスクを作成します。",
            bg=bg_color,
            fg=sub_fg_color,
            font=("Segoe UI", 10)
        ).pack(pady=(0, 14))

        btn_frame = ttk.Frame(container)
        btn_frame.pack(pady=4)

        def choose_detection_mask():
            selection['is_plate_solve_mask'] = False
            dialog.destroy()

        def choose_plate_solve_mask():
            selection['is_plate_solve_mask'] = True
            dialog.destroy()

        ttk.Button(btn_frame, text="検出用マスク", command=choose_detection_mask).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="プレートソルブ用マスク", command=choose_plate_solve_mask).pack(side=tk.LEFT, padx=6)
        ttk.Button(container, text="キャンセル", command=dialog.destroy).pack(pady=(12, 0))

        dialog.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - dialog.winfo_width()) // 2
        y = self.winfo_y() + (self.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")

        self.wait_window(dialog)
        return selection['is_plate_solve_mask']

    def _open_rtsp_mask_window(self, frame, is_plate_solve_mask: bool):
        """RTSPから取得したフレームでマスク作成ウィンドウを開く"""
        
        # マスク作成ウィンドウを開く
        win = Toplevel(self)
        mask_label = "プレートソルブ用マスク" if is_plate_solve_mask else "検出用マスク"
        win.title(f"RTSPから{mask_label}作成")
        win.geometry("1000x700")
        win.grab_set()

        orig_h, orig_w = frame.shape[:2]
        disp_w, disp_h = 960, 540
        scale = min(disp_w / orig_w, disp_h / orig_h)
        disp_w, disp_h = int(orig_w * scale), int(orig_h * scale)
        
        frame_disp = cv2.resize(frame, (disp_w, disp_h))
        tk_image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB)))

        canvas = Canvas(win, width=disp_w, height=disp_h, cursor="circle")
        canvas.pack(pady=5)
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
        canvas.image = tk_image

        # マスクデータをselfに一時保存して確実に参照可能にする
        self._rtsp_mask_data = Image.new("L", (disp_w, disp_h), 0)
        self._rtsp_mask_draw = ImageDraw.Draw(self._rtsp_mask_data)
        self._rtsp_mask_orig_size = (orig_w, orig_h)
        
        brush_radius = tk.IntVar(value=30)
        
        def paint(event):
            r = brush_radius.get()
            canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r, fill='white', outline='white', tags="paint")
            self._rtsp_mask_draw.ellipse((event.x - r, event.y - r, event.x + r, event.y + r), fill=255)
        
        canvas.bind("<B1-Motion>", paint)
        canvas.bind("<Button-1>", paint)

        def clear_mask():
            canvas.delete("paint")
            self._rtsp_mask_draw.rectangle([0, 0, disp_w, disp_h], fill=0)

        controls = ttk.Frame(win)
        controls.pack(fill=tk.X, padx=10)
        ttk.Label(controls, text="ブラシサイズ:").pack(side=tk.LEFT)
        ttk.Scale(controls, from_=5, to=100, orient=tk.HORIZONTAL, variable=brush_radius).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(controls, text="クリア", command=clear_mask).pack(side=tk.LEFT)

        def on_ok():
            try:
                # selfからマスクデータを取得
                mask_np_disp = np.array(self._rtsp_mask_data)
                orig_w, orig_h = self._rtsp_mask_orig_size
                
                print(f"マスクデータ確認: max={mask_np_disp.max()}, min={mask_np_disp.min()}, shape={mask_np_disp.shape}")
                
                # 描画がある場合は反転したマスクを作成、ない場合は全て255（マスクなし）
                if mask_np_disp.max() > 0:
                    # ディスプレイサイズから元のサイズにリサイズ
                    mask_resized = cv2.resize(mask_np_disp, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                    # 白黒反転（描画部分=255をマスク=0に）
                    final_mask = cv2.bitwise_not(mask_resized)
                else:
                    # 描画がなければマスクなし（全て255）
                    final_mask = np.full((orig_h, orig_w), 255, dtype=np.uint8)
                
                # メインアプリのマスクを更新
                if is_plate_solve_mask:
                    self.plate_solve_mask_image = final_mask
                    self.preview_mask(self.plate_solve_mask_image, self.ps_mask_preview_label, "PSマスク")
                else:
                    self.mask_image = final_mask
                    self.mask_path_var.set("作成済み (RTSP)")
                    self.apply_mask_var.set(True)
                    self.preview_mask(self.mask_image, self.mask_preview_label, "検出マスク")
                
                print(f"RTSP{mask_label}作成完了: shape={final_mask.shape}, max={final_mask.max()}, min={final_mask.min()}")
                
                # 一時データを削除
                del self._rtsp_mask_data
                del self._rtsp_mask_draw
                del self._rtsp_mask_orig_size
                
                # ウィンドウを閉じる
                win.destroy()
                
            except Exception as e:
                messagebox.showerror("エラー", f"マスク作成中にエラーが発生しました:\n{e}")
                import traceback
                traceback.print_exc()

        btn_frame = ttk.Frame(win)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="キャンセル", command=win.destroy).pack(side=tk.LEFT, padx=5)

    def preview_mask(self, mask_data, target_label, label_text):
        if mask_data is None:
            target_label.config(image='', text=f"{label_text}なし")
            target_label.image = None
            return
        try:
            preview_data = cv2.bitwise_not(mask_data)
            mask_pil = Image.fromarray(preview_data)
            mask_pil.thumbnail((80, 80))
            mask_photo = ImageTk.PhotoImage(mask_pil)
            target_label.config(image=mask_photo, text=f"{label_text}あり", compound=tk.TOP)
            target_label.image = mask_photo
        except Exception as e:
            print(f"マスクプレビューエラー: {e}")
            target_label.config(image='', text=f"{label_text} (エラー)")

    def download_mask(self):
        """検出マスクをPNG形式（1920x1080）でダウンロードする（マスク部分は透明）"""
        if self.mask_image is None:
            messagebox.showwarning("警告", "検出マスクが作成されていません。\n先にマスクを作成してください。")
            return
        
        # ファイル保存ダイアログを表示
        save_path = filedialog.asksaveasfilename(
            title="マスクを保存",
            defaultextension=".png",
            filetypes=[("PNG画像", "*.png")],
            initialfile="detection_mask.png"
        )
        
        if not save_path:
            return  # キャンセルされた場合
        
        try:
            # マスクを1920x1080にリサイズ（横長）
            target_size = (1920, 1080)  # (width, height)
            resized_mask = cv2.resize(self.mask_image, target_size, interpolation=cv2.INTER_NEAREST)
            
            # RGBAに変換
            # mask_imageでは 255=検出可能領域（非マスク）、0=マスク領域（塗った部分）
            # 出力では: マスク領域（塗った部分）=黒で不透明、非マスク領域=透明
            height, width = resized_mask.shape
            rgba = np.zeros((height, width, 4), dtype=np.uint8)
            
            # アルファチャンネル: マスク領域(0)=不透明(255)、非マスク領域(255)=透明(0)
            rgba[:, :, 3] = 255 - resized_mask  # 反転してマスク部分を不透明に
            # RGB channels stay 0 (black) for mask areas
            
            # PILを使用してPNGとして保存（アルファチャンネル対応）
            pil_image = Image.fromarray(rgba, mode='RGBA')
            pil_image.save(save_path, 'PNG')
            
            messagebox.showinfo("保存完了", f"マスクを保存しました:\n{save_path}")
            self.append_log(f"検出マスクを保存しました: {save_path}")
        except Exception as e:
            messagebox.showerror("エラー", f"マスクの保存に失敗しました:\n{e}")

