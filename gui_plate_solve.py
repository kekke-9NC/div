from gui_common import *
from camera_model_builder import CameraModelBuildRequest, build_camera_model
from camera_model_monitor import RTSPCameraModelMonitor


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
        if file_path:
            self.plate_solve_video_path_var.set(file_path)
            if not self.camera_model_source_var.get().strip():
                self.camera_model_source_var.set(file_path)

    def select_camera_model_video(self):
        file_path = filedialog.askopenfilename(
            title="高精度モデル用動画を選択",
            filetypes=[("動画ファイル", "*.mp4 *.avi *.mov *.mkv *.m4v *.ts"), ("すべてのファイル", "*.*")],
        )
        if file_path:
            self.camera_model_source_var.set(file_path)
            self.plate_solve_video_path_var.set(file_path)

    def select_camera_model_folder(self):
        folder = filedialog.askdirectory(title="高精度モデル用RTSP保存フォルダを選択")
        if folder:
            self.camera_model_source_var.set(folder)

    def _set_camera_model_status(self, text: str):
        def apply():
            try:
                self.camera_model_status_var.set(text)
            except tk.TclError:
                pass
        if threading.current_thread() is threading.main_thread():
            apply()
        else:
            try:
                self.after(0, apply)
            except tk.TclError:
                pass

    def _camera_model_request(self) -> CameraModelBuildRequest:
        source = self.camera_model_source_var.get().strip() or self.plate_solve_video_path_var.get().strip()
        if not source:
            raise ValueError("モデル作成対象の動画またはRTSP保存フォルダを選択してください")
        try:
            threshold = float(self.camera_model_cloud_threshold_var.get())
        except ValueError as exc:
            raise ValueError("雲量しきい値は数値で指定してください") from exc
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("雲量しきい値は0.0〜1.0で指定してください")
        return CameraModelBuildRequest(
            source=source,
            start=self.camera_model_start_var.get().strip(),
            end=self.camera_model_end_var.get().strip(),
            cloud_threshold=threshold,
            use_cloud_filter=bool(self.camera_model_cloud_filter_var.get()),
            backend=self.ai_vlm_backend_var.get(),
            lm_studio_url=self.lm_studio_vlm_url_var.get(),
            lm_studio_model_id=self.lm_studio_vlm_model_var.get(),
            lm_studio_api_key=self.lm_studio_vlm_api_key_var.get(),
        )

    def start_camera_model_build(self):
        try:
            request = self._camera_model_request()
        except Exception as exc:
            messagebox.showwarning("高精度モデル", str(exc), parent=self)
            return
        self.btn_build_camera_model.configure(state=tk.DISABLED)
        self._set_camera_model_status("高精度モデル: 作成中...")

        def progress(message):
            self._set_camera_model_status(f"高精度モデル: {message}")
            try:
                self.progress_queue.put((message, None))
            except Exception:
                pass

        def worker():
            result = build_camera_model(request, progress_callback=progress)

            def finished():
                self.btn_build_camera_model.configure(state=tk.NORMAL)
                if result.success and result.enabled:
                    self.camera_model_status_var.set(
                        f"高精度モデル: 登録済み（被覆率 {result.support_fraction * 100:.0f}% / "
                        f"p95 {result.residual_p95_px:.2f}px）"
                    )
                    self.plate_solve_wcs_path_var.set(result.model_path)
                    self.plate_solve_status_var.set("プレートソルブ: 高精度固定カメラモデルを適用")
                    self.global_wcs_info = {
                        "wcs_file": result.model_path,
                        "calibration_path": result.model_path,
                        "model_path": result.model_path,
                        "job_id": "local-wideangle-camera-model",
                    }
                    self.update_start_button_state()
                    if result.target_met:
                        messagebox.showinfo("高精度モデル", "選択範囲から高精度固定カメラモデルを作成し、登録しました。", parent=self)
                    else:
                        messagebox.showwarning(
                            "高精度モデル",
                            "モデルは登録しましたが、目標被覆率80%またはp95 2pxの基準には未達です。\n"
                            f"検証レポート:\n{result.report_path}", parent=self,
                        )
                elif result.success:
                    self.camera_model_status_var.set(
                        f"高精度モデル: 候補保存（被覆率 {result.support_fraction * 100:.0f}% / "
                        f"p95 {result.residual_p95_px:.2f}px、未適用）"
                    )
                    messagebox.showwarning(
                        "高精度モデル",
                        "候補モデルは保存しましたが、安全な被覆率または誤差基準に届かないため適用しません。\n"
                        f"検証レポート:\n{result.report_path}", parent=self,
                    )
                else:
                    self.camera_model_status_var.set(f"高精度モデル: 失敗（{result.error}）")
                    messagebox.showerror("高精度モデル", result.error, parent=self)

            self.after(0, finished)

        threading.Thread(target=worker, name="camera-model-build", daemon=True).start()

    def toggle_camera_model_monitor(self):
        monitor = getattr(self, "camera_model_monitor", None)
        if monitor is not None and monitor.thread and monitor.thread.is_alive():
            monitor.stop()
            self.camera_model_monitor = None
            self.btn_toggle_camera_model_monitor.configure(text="RTSP自動監視を開始")
            self._set_camera_model_status("高精度モデル: RTSP自動監視を停止")
            return
        selected_indices = [i for i in self.rtsp_selected_indices if 0 <= i < len(self.rtsp_urls)]
        rtsp_url = self.rtsp_urls[selected_indices[0]] if selected_indices else (self.rtsp_urls[0] if self.rtsp_urls else "")
        if not rtsp_url:
            messagebox.showwarning("高精度モデル", "RTSP URLを追加してください。", parent=self)
            return
        try:
            interval = max(10, int(float(self.camera_model_interval_var.get())))
            threshold = float(self.camera_model_cloud_threshold_var.get())
        except ValueError:
            messagebox.showwarning("高精度モデル", "監視間隔と雲量しきい値を確認してください。", parent=self)
            return
        monitor = RTSPCameraModelMonitor(
            rtsp_url, interval_seconds=interval, cloud_threshold=threshold,
            backend=self.ai_vlm_backend_var.get(), lm_studio_url=self.lm_studio_vlm_url_var.get(),
            lm_studio_model_id=self.lm_studio_vlm_model_var.get(), lm_studio_api_key=self.lm_studio_vlm_api_key_var.get(),
            status_callback=self._set_camera_model_status,
        )
        self.camera_model_monitor = monitor
        monitor.start()
        self.btn_toggle_camera_model_monitor.configure(text="RTSP自動監視を停止")
        self._set_camera_model_status(f"高精度モデル: RTSPを{interval}秒間隔で監視中")

    def select_plate_solve_wcs_file(self):
        file_path = filedialog.askopenfilename(title="既存のWCS/固定カメラモデルを選択", filetypes=[("WCS/FITS/モデル", "*.wcs *.fits *.fit *.json"), ("すべてのファイル", "*.*")])
        if file_path:
            try:
                ps_datetime = None
                local_wideangle_wcs = False
                if file_path.lower().endswith(".json"):
                    import local_wideangle_astrometry
                    metadata, _model = local_wideangle_astrometry._load_calibration(file_path)
                    if metadata.get("model_type") != "fixed-camera-stg-poly":
                        raise ValueError("固定カメラモデルJSONではありません。")
                    reference_value = metadata.get("reference_datetime")
                    if reference_value:
                        ps_datetime = datetime.fromisoformat(str(reference_value).replace("Z", "+00:00"))
                    local_wideangle_wcs = True
                    self.global_wcs_info = {
                        "wcs_file": file_path,
                        "calibration_path": file_path,
                        "plate_solve_datetime": ps_datetime or datetime.now(),
                        "job_id": "local-wideangle-camera-model",
                    }
                    self.plate_solve_wcs_path_var.set(file_path)
                    self.plate_solve_status_var.set(
                        f"プレートソルブ: 高精度モデル適用 @ {(ps_datetime or datetime.now()).strftime('%H:%M')}"
                    )
                    messagebox.showinfo("成功", "固定カメラモデルをロードしました。", parent=self)
                    self.update_start_button_state()
                    return
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

