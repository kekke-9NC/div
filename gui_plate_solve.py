from gui_common import *


class PlateSolveMixin:
    def _handle_plate_solve_ui(self, payload):
        """Apply one plate-solve UI event; called only by the Tk poller."""
        action = payload.get("action")
        if action == "status":
            self.plate_solve_status_var.set(payload.get("text", ""))
        elif action == "message":
            getattr(messagebox, payload.get("level", "showinfo"))(
                payload.get("title", ""), payload.get("message", ""), parent=self
            )
        elif action == "result":
            result = payload["result"]
            self.global_wcs_info = result
            self.plate_solve_wcs_path_var.set(result["wcs_file"])
            self.plate_solve_status_var.set(payload.get("status_text", "プレートソルブ: 成功"))
            self.update_start_button_state()

    def _queue_plate_solve_ui(self, payload):
        self.progress_queue.put((None, {"plate_solve_ui": payload}))

    def _set_plate_solve_status(self, text):
        """Tk変数は必ずメインスレッドで更新する。"""
        if threading.current_thread() is threading.main_thread():
            self.plate_solve_status_var.set(text)
        else:
            self._queue_plate_solve_ui({"action": "status", "text": text})

    def _show_plate_solve_message(self, level, title, message):
        if threading.current_thread() is threading.main_thread():
            getattr(messagebox, level)(title, message, parent=self)
        else:
            self._queue_plate_solve_ui({
                "action": "message", "level": level, "title": title, "message": message
            })

    def _set_plate_solve_result(self, result, status_text):
        if threading.current_thread() is threading.main_thread():
            self.global_wcs_info = result
            self.plate_solve_wcs_path_var.set(result['wcs_file'])
            self.plate_solve_status_var.set(status_text)
            self.update_start_button_state()
        else:
            self._queue_plate_solve_ui({
                "action": "result", "result": result, "status_text": status_text
            })

    def select_plate_solve_video(self):
        file_path = filedialog.askopenfilename(title="プレートソルブ用動画を選択", filetypes=[("動画ファイル", "*.mp4 *.avi *.mov"), ("すべてのファイル", "*.*")])
        if file_path: self.plate_solve_video_path_var.set(file_path)

    def select_plate_solve_wcs_file(self):
        file_path = filedialog.askopenfilename(title="既存のWCSファイルを選択", filetypes=[("WCS/FITSファイル", "*.wcs *.fits"), ("すべてのファイル", "*.*")])
        if file_path:
            try:
                ps_datetime = None
                local_wideangle_wcs = False
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
                        local_wideangle_wcs = header.get('CALTYPE') == 'LOCAL-SIP'
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

                self.global_wcs_info = {
                    'wcs_file': file_path,
                    'plate_solve_datetime': ps_datetime,
                    'job_id': (
                        'local-wideangle-manual' if local_wideangle_wcs else 'manual-wcs'
                    ),
                }
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
        video_path = self.plate_solve_video_path_var.get().strip()
        if not video_path:
            messagebox.showwarning("警告", "プレートソルブに使用する動画を選択してください。")
            self.plate_solve_status_var.set("プレートソルブ: 未実行")
            return
        use_local = self.plate_solve_mode_var.get() == "local"
        plate_mask = self.plate_solve_mask_image
        threading.Thread(
            target=self.execute_plate_solve_thread,
            args=(video_path, use_local, plate_mask),
            daemon=True,
            name="local-wideangle-calibration",
        ).start()

    def start_rtsp_plate_solve(self):
        """RTSPストリームからプレートソルブを実行する"""
        self.apply_advanced_settings_to_config()
        # 選択されているRTSP URLを取得、選択がなければ最初のURLを使用
        selected_indices = [index for index in self.rtsp_selected_indices if 0 <= index < len(self.rtsp_urls)]
        if selected_indices:
            selected_index = min(selected_indices)
            rtsp_url = self.rtsp_urls[selected_index]
        elif self.rtsp_urls:
            rtsp_url = self.rtsp_urls[0]
        else:
            messagebox.showwarning("警告", "RTSPストリームを追加してください。")
            return
        use_local = self.plate_solve_mode_var.get() == "local"
        rtsp_mask = self.mask_image if self.apply_mask_var.get() else None
        fixed_pattern = self.rtsp_dark_frame if self.apply_rtsp_dark_var.get() else None
        threading.Thread(
            target=self.execute_rtsp_plate_solve_thread,
            args=(rtsp_url, use_local, rtsp_mask, fixed_pattern),
            daemon=True,
            name="rtsp-wideangle-calibration",
        ).start()

    def execute_rtsp_plate_solve_thread(
        self, rtsp_url: str, use_local_solver=False, rtsp_mask=None, fixed_pattern=None
    ):
        """RTSPストリームからフレームを取得してプレートソルブを実行するスレッド"""
        self._set_plate_solve_status("プレートソルブ: RTSP接続中...")
        self.progress_queue.put((f"RTSPプレートソルブを実行中: {rtsp_url}", None))
        cap = None
        try:
            cap = utils.create_rtsp_capture(rtsp_url)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
            
            if not cap.isOpened():
                raise IOError(f"RTSPストリームを開けません: {rtsp_url}")
            
            self._set_plate_solve_status("プレートソルブ: フレーム取得中...")
            
            # 約10秒分のフレームを取得（25fps前提で250フレーム）
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or fps > 120:
                fps = config.RTSP_FPS  # デフォルトのRTSP FPSを使用
            num_frames = int(fps * 10)
            
            frames = []
            maximum = None
            local_stride = max(1, num_frames // 30)
            for frame_index in range(num_frames):
                ret, frame = cap.read()
                if ret and frame is not None:
                    if fixed_pattern is not None:
                        frame = apply_fixed_pattern_correction(frame, fixed_pattern)
                    if use_local_solver:
                        if frame_index < 50 or frame_index % local_stride == 0:
                            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                    elif maximum is None:
                        maximum = frame.copy()
                    else:
                        cv2.max(maximum, frame, dst=maximum)
                else:
                    # フレーム取得に失敗した場合、少し待って再試行
                    time.sleep(0.01)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        if fixed_pattern is not None:
                            frame = apply_fixed_pattern_correction(frame, fixed_pattern)
                        if use_local_solver:
                            if frame_index < 50 or frame_index % local_stride == 0:
                                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                        elif maximum is None:
                            maximum = frame.copy()
                        else:
                            cv2.max(maximum, frame, dst=maximum)
            cap.release()
            cap = None
            
            if use_local_solver and len(frames) < 10:
                raise ValueError(f"RTSPストリームから十分なフレームを取得できませんでした。取得フレーム数: {len(frames)}")
            if not use_local_solver and maximum is None:
                raise ValueError("RTSPストリームからフレームを取得できませんでした。")
            
            self._set_plate_solve_status("プレートソルブ: 実行中...")
            if use_local_solver:
                import local_wideangle_astrometry

                plate_solve_result = local_wideangle_astrometry.solve_frames_local(
                    frames,
                    source_identity=f"rtsp_{datetime.now():%Y%m%d_%H%M%S}.mp4",
                    observation_datetime=datetime.now(),
                    progress_callback=lambda message: self.progress_queue.put((str(message), None)),
                )
            else:
                self.progress_queue.put(("Astrometry.netにアップロード中...", None))
                temp_composite_path = os.path.join(
                    config.TEMP_CLIP_DIR, f"rtsp_composite_{time.time_ns()}.jpg"
                )
                os.makedirs(config.TEMP_CLIP_DIR, exist_ok=True)
                cv2.imwrite(temp_composite_path, maximum)
                try:
                    plate_solve_result = astrometry.plate_solve_image(
                        temp_composite_path, mask=rtsp_mask,
                        plate_solve_video_path=rtsp_url, cancel_flag=self.cancel_flag,
                        scale_lower=config.RTSP_SCALE_LOWER, scale_upper=config.RTSP_SCALE_UPPER,
                        use_local=False,
                    )
                finally:
                    if os.path.exists(temp_composite_path):
                        os.remove(temp_composite_path)
            
            if plate_solve_result and 'wcs_file' in plate_solve_result:
                ps_datetime = plate_solve_result.get('plate_solve_datetime', datetime.now())
                status_text = f"プレートソルブ: 成功 (RTSP) @ {ps_datetime.strftime('%H:%M')}"
                self._set_plate_solve_result(plate_solve_result, status_text)
                self.progress_queue.put((f"RTSPプレートソルブ成功: {plate_solve_result['wcs_file']}", None))
                self._show_plate_solve_message("showinfo", "成功", f"RTSPからのプレートソルブに成功しました。\n参照時刻: {ps_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                self._set_plate_solve_status("プレートソルブ: 失敗")
                self.progress_queue.put(("RTSPプレートソルブ失敗", None))
                self._show_plate_solve_message("showerror", "失敗", "RTSPからのプレートソルブに失敗しました。\nストリーム内容、ネットワーク、APIキーを確認してください。")

        except Exception as e:
            self._set_plate_solve_status("プレートソルブ: エラー")
            error_message = f"RTSPプレートソルブ中にエラーが発生しました: {e}"
            self.progress_queue.put((error_message, None))
            self._show_plate_solve_message("showerror", "エラー", error_message)
        finally:
            if cap is not None:
                cap.release()

    def execute_plate_solve_thread(self, video_file_path, use_local_solver=False, plate_mask=None):
        self._set_plate_solve_status("プレートソルブ: 実行中...")
        self.progress_queue.put(("プレートソルブを実行中...", None))
        cap = None
        try:
            if use_local_solver:
                import local_wideangle_astrometry

                plate_solve_result = local_wideangle_astrometry.solve_video_local(
                    video_file_path,
                    progress_callback=lambda message: self.progress_queue.put((str(message), None)),
                )
            else:
                cap = cv2.VideoCapture(video_file_path)
                if not cap.isOpened():
                    raise IOError("動画ファイルを開けません。")
                fps = cap.get(cv2.CAP_PROP_FPS) or config.DEFAULT_FPS
                num_frames = int(fps * 10)
                maximum = None
                for _ in range(num_frames):
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                    if maximum is None:
                        maximum = frame.copy()
                    else:
                        cv2.max(maximum, frame, dst=maximum)
                cap.release()
                cap = None
                if maximum is None:
                    raise ValueError("動画からフレームを取得できませんでした。")
                temp_composite_path = os.path.join(
                    config.TEMP_CLIP_DIR, f"temp_composite_{time.time_ns()}.jpg"
                )
                os.makedirs(config.TEMP_CLIP_DIR, exist_ok=True)
                cv2.imwrite(temp_composite_path, maximum)
                try:
                    plate_solve_result = astrometry.plate_solve_image(
                        temp_composite_path, mask=plate_mask,
                        plate_solve_video_path=video_file_path,
                        cancel_flag=self.cancel_flag, use_local=False,
                    )
                finally:
                    if os.path.exists(temp_composite_path):
                        os.remove(temp_composite_path)

            if plate_solve_result and 'wcs_file' in plate_solve_result:
                ps_datetime = plate_solve_result.get('plate_solve_datetime', datetime.now())
                status_text = f"プレートソルブ: 成功 @ {ps_datetime.strftime('%H:%M')}"
                self._set_plate_solve_result(plate_solve_result, status_text)
                self.progress_queue.put((f"プレートソルブ成功: {plate_solve_result['wcs_file']}", None))
                self._show_plate_solve_message("showinfo", "成功", f"プレートソルブに成功しました。\n参照時刻: {ps_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                self._set_plate_solve_status("プレートソルブ: 失敗")
                self.progress_queue.put(("プレートソルブ失敗", None))
                self._show_plate_solve_message("showerror", "失敗", "プレートソルブに失敗しました。APIキー、ネットワーク、画像内容を確認してください。")

        except Exception as e:
            self._set_plate_solve_status("プレートソルブ: エラー")
            error_message = f"プレートソルブ中にエラーが発生しました: {e}"
            self.progress_queue.put((error_message, None))
            self._show_plate_solve_message("showerror", "エラー", error_message)
        finally:
            if cap is not None:
                cap.release()

