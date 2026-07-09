from gui_common import *


class PlateSolveMixin:
    def select_plate_solve_video(self):
        file_path = filedialog.askopenfilename(title="プレートソルブ用動画を選択", filetypes=[("動画ファイル", "*.mp4 *.avi *.mov"), ("すべてのファイル", "*.*")])
        if file_path: self.plate_solve_video_path_var.set(file_path)

    def select_plate_solve_wcs_file(self):
        file_path = filedialog.askopenfilename(title="既存のWCSファイルを選択", filetypes=[("WCS/FITSファイル", "*.wcs *.fits"), ("すべてのファイル", "*.*")])
        if file_path:
            try:
                ps_datetime = None
                # まずWCSファイル(FITS)のヘッダーから'DATE-OBS'を読み込もうと試みる
                try:
                    with fits.open(file_path) as hdul:
                        header = hdul[0].header
                        if not WCS(header).is_celestial:
                            raise ValueError("有効な天球WCSではありません。")
                        
                        if 'DATE-OBS' in header:
                            date_obs_str = header['DATE-OBS']
                            ps_datetime = datetime.fromisoformat(date_obs_str)
                            print(f"WCSヘッダーから基準時刻を読み込みました: {ps_datetime}")
                except Exception as fits_e:
                    print(f"FITSヘッダーの読み込みまたは解析に失敗しました: {fits_e}")
                    # FITSとして開けなかった場合や'DATE-OBS'がない場合は、従来の方法に進む
                    pass

                # ヘッダーから時刻が取得できなかった場合、ファイルパスから抽出を試みる
                if ps_datetime is None:
                    print("WCSヘッダーに基準時刻が見つからないため、ファイルパスから推定します。")
                    ps_datetime = astrometry.extract_datetime_from_file_path(file_path)

                # それでも時刻が取得できない場合、最終手段として現在時刻を使用する
                if ps_datetime is None:
                    print("ファイルパスからも基準時刻を推定できませんでした。現在時刻を使用します。")
                    ps_datetime = datetime.now()

                self.global_wcs_info = {'wcs_file': file_path, 'plate_solve_datetime': ps_datetime}
                self.plate_solve_wcs_path_var.set(file_path)
                self.plate_solve_status_var.set(f"プレートソルブ: 成功 (既存WCS) @ {ps_datetime.strftime('%H:%M')}")
                messagebox.showinfo("成功", f"既存WCSファイルをロードしました。\n参照時刻: {ps_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
                self.update_start_button_state()

            except Exception as e:
                self.global_wcs_info = None
                self.plate_solve_wcs_path_var.set("")
                self.plate_solve_status_var.set("プレートソルブ: 失敗")
                messagebox.showerror("エラー", f"WCSファイルのロード/検証に失敗しました:\n{e}")

    def start_plate_solve(self):
        self.apply_advanced_settings_to_config()
        threading.Thread(target=self.execute_plate_solve_thread, daemon=True).start()

    def start_rtsp_plate_solve(self):
        """RTSPストリームからプレートソルブを実行する"""
        self.apply_advanced_settings_to_config()
        # 選択されているRTSP URLを取得、選択がなければ最初のURLを使用
        if self.rtsp_selected_indices:
            selected_index = min(self.rtsp_selected_indices)
            rtsp_url = self.rtsp_urls[selected_index]
        elif self.rtsp_urls:
            rtsp_url = self.rtsp_urls[0]
        else:
            messagebox.showwarning("警告", "RTSPストリームを追加してください。")
            return
        threading.Thread(target=self.execute_rtsp_plate_solve_thread, args=(rtsp_url,), daemon=True).start()

    def execute_rtsp_plate_solve_thread(self, rtsp_url: str):
        """RTSPストリームからフレームを取得してプレートソルブを実行するスレッド"""
        self.plate_solve_status_var.set("プレートソルブ: RTSP接続中...")
        self.progress_queue.put((f"RTSPプレートソルブを実行中: {rtsp_url}", None))
        try:
            cap = utils.create_rtsp_capture(rtsp_url)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
            
            if not cap.isOpened():
                raise IOError(f"RTSPストリームを開けません: {rtsp_url}")
            
            self.plate_solve_status_var.set("プレートソルブ: フレーム取得中...")
            
            # 約10秒分のフレームを取得（25fps前提で250フレーム）
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or fps > 120:
                fps = config.RTSP_FPS  # デフォルトのRTSP FPSを使用
            num_frames = int(fps * 10)
            
            frames = []
            for _ in range(num_frames):
                ret, frame = cap.read()
                if ret and frame is not None:
                    frame = self.apply_rtsp_dark_to_frame(frame)
                    frames.append(frame)
                else:
                    # フレーム取得に失敗した場合、少し待って再試行
                    time.sleep(0.01)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        frame = self.apply_rtsp_dark_to_frame(frame)
                        frames.append(frame)
            cap.release()
            
            if len(frames) < 10:
                raise ValueError(f"RTSPストリームから十分なフレームを取得できませんでした。取得フレーム数: {len(frames)}")
            
            self.plate_solve_status_var.set("プレートソルブ: 合成画像作成中...")
            self.progress_queue.put((f"RTSPから{len(frames)}フレームを取得しました。合成画像を作成中...", None))
            
            composite_image = np.max(np.array(frames), axis=0).astype(np.uint8)
            temp_composite_path = os.path.join(config.TEMP_CLIP_DIR, f"rtsp_composite_{time.time_ns()}.jpg")
            os.makedirs(config.TEMP_CLIP_DIR, exist_ok=True)
            cv2.imwrite(temp_composite_path, composite_image)
            
            self.plate_solve_status_var.set("プレートソルブ: 実行中...")
            self.progress_queue.put(("Astrometry.netにアップロード中...", None))
            
            # RTSPプレートソルブでは検出マスク（RTSPから作成したマスク）を使用
            rtsp_mask = self.mask_image if self.apply_mask_var.get() else None
            use_local_solver = (self.plate_solve_mode_var.get() == "local")
            plate_solve_result = astrometry.plate_solve_image(
                temp_composite_path, mask=rtsp_mask,
                plate_solve_video_path=rtsp_url, cancel_flag=self.cancel_flag,
                scale_lower=config.RTSP_SCALE_LOWER, scale_upper=config.RTSP_SCALE_UPPER,
                use_local=use_local_solver
            )
            if os.path.exists(temp_composite_path):
                os.remove(temp_composite_path)
            
            if plate_solve_result and 'wcs_file' in plate_solve_result:
                self.global_wcs_info = plate_solve_result
                ps_datetime = self.global_wcs_info.get('plate_solve_datetime', datetime.now())
                self.plate_solve_status_var.set(f"プレートソルブ: 成功 (RTSP) @ {ps_datetime.strftime('%H:%M')}")
                self.plate_solve_wcs_path_var.set(self.global_wcs_info['wcs_file'])
                self.progress_queue.put((f"RTSPプレートソルブ成功: {self.global_wcs_info['wcs_file']}", None))
                messagebox.showinfo("成功", f"RTSPからのプレートソルブに成功しました。\n参照時刻: {ps_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
                self.update_start_button_state()
            else:
                self.global_wcs_info = None
                self.plate_solve_status_var.set("プレートソルブ: 失敗")
                self.progress_queue.put(("RTSPプレートソルブ失敗", None))
                messagebox.showerror("失敗", "RTSPからのプレートソルブに失敗しました。\nストリーム内容、ネットワーク、APIキーを確認してください。")
                
        except Exception as e:
            self.global_wcs_info = None
            self.plate_solve_status_var.set("プレートソルブ: エラー")
            error_message = f"RTSPプレートソルブ中にエラーが発生しました: {e}"
            self.progress_queue.put((error_message, None))
            messagebox.showerror("エラー", error_message)

    def execute_plate_solve_thread(self):
        video_file_path = self.plate_solve_video_path_var.get()
        if not video_file_path:
            messagebox.showwarning("警告", "プレートソルブに使用する動画を選択してください。")
            self.plate_solve_status_var.set("プレートソルブ: 未実行")
            return

        self.plate_solve_status_var.set("プレートソルブ: 実行中...")
        self.progress_queue.put(("プレートソルブを実行中...", None))
        try:
            cap = cv2.VideoCapture(video_file_path)
            if not cap.isOpened(): raise IOError("動画ファイルを開けません。")
            fps = cap.get(cv2.CAP_PROP_FPS) or config.DEFAULT_FPS
            num_frames = int(fps * 10)
            frames = [cap.read()[1] for _ in range(num_frames) if cap.isOpened() and cap.read()[0]]
            cap.release()
            if not frames: raise ValueError("動画からフレームを取得できませんでした。")
            
            composite_image = np.max(np.array(frames), axis=0).astype(np.uint8)
            temp_composite_path = os.path.join(config.TEMP_CLIP_DIR, f"temp_composite_{time.time_ns()}.jpg")
            os.makedirs(config.TEMP_CLIP_DIR, exist_ok=True)
            cv2.imwrite(temp_composite_path, composite_image)

            use_local_solver = (self.plate_solve_mode_var.get() == "local")
            plate_solve_result = astrometry.plate_solve_image(
                temp_composite_path, mask=self.plate_solve_mask_image,
                plate_solve_video_path=video_file_path, cancel_flag=self.cancel_flag,
                use_local=use_local_solver
            )
            if os.path.exists(temp_composite_path): os.remove(temp_composite_path)

            if plate_solve_result and 'wcs_file' in plate_solve_result:
                self.global_wcs_info = plate_solve_result
                ps_datetime = self.global_wcs_info.get('plate_solve_datetime', datetime.now())
                self.plate_solve_status_var.set(f"プレートソルブ: 成功 @ {ps_datetime.strftime('%H:%M')}")
                self.plate_solve_wcs_path_var.set(self.global_wcs_info['wcs_file'])
                self.progress_queue.put((f"プレートソルブ成功: {self.global_wcs_info['wcs_file']}", None))
                messagebox.showinfo("成功", f"プレートソルブに成功しました。\n参照時刻: {ps_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
                self.update_start_button_state()
            else:
                self.global_wcs_info = None
                self.plate_solve_status_var.set("プレートソルブ: 失敗")
                self.progress_queue.put(("プレートソルブ失敗", None))
                messagebox.showerror("失敗", "プレートソルブに失敗しました。APIキー、ネットワーク、画像内容を確認してください。")

        except Exception as e:
            self.global_wcs_info = None
            self.plate_solve_status_var.set("プレートソルブ: エラー")
            error_message = f"プレートソルブ中にエラーが発生しました: {e}"
            self.progress_queue.put((error_message, None))
            messagebox.showerror("エラー", error_message)

