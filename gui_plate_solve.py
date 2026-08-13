from gui_common import *
from camera_model_builder import CameraModelBuildRequest, build_camera_model
from camera_model_monitor import RTSPCameraModelMonitor
from camera_plate_model import MODEL_TYPE as FIXED_CAMERA_MODEL_TYPE
from trajectory_camera_model import TrajectoryBuildRequest, build_trajectory_camera_model
import camera_model_catalog


def _usable_trajectory_seed(path: str) -> bool:
    """Return whether path contains the fixed-camera parameters trajectories need."""
    try:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return bool(
        payload.get("model_type") == FIXED_CAMERA_MODEL_TYPE
        and payload.get("stg_parameters")
        and payload.get("correction_coefficients") is not None
        and payload.get("reference_datetime")
    )


def _build_camera_model_for_app(
    request: CameraModelBuildRequest,
    *,
    use_trajectory: bool,
    initial_model_path: str = "",
    progress_callback=None,
):
    """Run the model path selected in the GUI, including seed creation."""
    seed_path = initial_model_path if _usable_trajectory_seed(initial_model_path) else ""
    if not use_trajectory:
        return build_camera_model(request, progress_callback=progress_callback)
    if not seed_path:
        if progress_callback:
            progress_callback("絶対座標の基準となる初期モデルを作成中...")
        seed_result = build_camera_model(request, progress_callback=progress_callback)
        if not seed_result.success or not seed_result.model_path:
            return seed_result
        seed_path = seed_result.model_path
    if progress_callback:
        progress_callback("動画内の恒星を追跡して投影モデルを学習中...")
    return build_trajectory_camera_model(
        TrajectoryBuildRequest(
            source=request.source,
            initial_model_path=seed_path,
            start=request.start,
            end=request.end,
            cache_root=request.cache_root,
        ),
        progress_callback=progress_callback,
    )


class PlateSolveMixin:
    _AUTO_CAMERA_MODEL_LABEL = "自動選択（撮影日に合う補正データ）"
    _TRAJECTORY_MODEL_METHOD = "動画の星の動き（推奨）"
    _STATIC_MODEL_METHOD = "静止画プレートソルブ"

    def _refresh_plate_solve_model_choices(self):
        """Refresh the in-app camera-correction selector."""
        try:
            models = camera_model_catalog.discover_camera_models()
        except Exception as exc:
            models = []
            logger = getattr(self, "append_log", None)
            if callable(logger):
                logger(f"カメラ補正データ一覧の取得に失敗しました: {exc}")
        self.plate_solve_model_entries = models
        self.plate_solve_model_by_display = {
            item["display_name"]: item for item in models
        }
        values = [self._AUTO_CAMERA_MODEL_LABEL] + [
            item["display_name"] for item in models
        ]
        combo = getattr(self, "cmb_plate_solve_model", None)
        if combo is not None:
            combo.configure(values=values)
        selected_path = self.plate_solve_model_path_var.get().strip()
        selected = next(
            (item for item in models if os.path.abspath(item["path"]) == os.path.abspath(selected_path)),
            None,
        ) if selected_path else None
        if selected is not None:
            self.plate_solve_model_var.set(selected["display_name"])
        elif self.plate_solve_model_var.get() not in values:
            self.plate_solve_model_var.set(self._AUTO_CAMERA_MODEL_LABEL)
        self._update_plate_solve_model_info()

    def _selected_plate_solve_model(self):
        value = self.plate_solve_model_var.get().strip()
        if value == self._AUTO_CAMERA_MODEL_LABEL:
            return None
        return getattr(self, "plate_solve_model_by_display", {}).get(value)

    def _update_plate_solve_model_info(self):
        selected = self._selected_plate_solve_model()
        try:
            self.plate_solve_model_info_var.set(
                camera_model_catalog.format_model_details(selected)
            )
        except Exception:
            self.plate_solve_model_info_var.set("モデル情報を読み込めません")

    def on_plate_solve_model_selected(self, _event=None):
        """Apply the model chosen in the combobox without opening Finder."""
        selected = self._selected_plate_solve_model()
        self._update_plate_solve_model_info()
        if selected is None:
            self.plate_solve_model_path_var.set("")
            # Do not discard a manually loaded WCS unless it was a model that
            # this selector had applied previously.
            current = self.global_wcs_info or {}
            if current.get("job_id") == "local-wideangle-camera-model":
                self.global_wcs_info = None
                self.plate_solve_wcs_path_var.set("")
                self.plate_solve_status_var.set("プレートソルブ: 撮影日に合う補正データを自動選択")
            self.update_start_button_state()
            return
        self._apply_plate_solve_model(selected)

    def _apply_plate_solve_model(self, selected):
        try:
            import local_wideangle_astrometry
            metadata, _model = local_wideangle_astrometry._load_calibration(selected["path"])
            reference_value = metadata.get("reference_datetime")
            reference = (
                datetime.fromisoformat(str(reference_value).replace("Z", "+00:00"))
                if reference_value else datetime.now()
            )
            self.plate_solve_model_path_var.set(selected["path"])
            self.plate_solve_wcs_path_var.set(selected["path"])
            self.global_wcs_info = {
                "wcs_file": selected["path"],
                "calibration_path": selected["path"],
                "model_path": selected["path"],
                "plate_solve_datetime": reference,
                "job_id": "local-wideangle-camera-model",
                "model_label": selected["model_label"],
                "support_fraction": selected["support_fraction"],
                "reference_night": selected["reference_night"],
            }
            self.plate_solve_status_var.set(
                f"プレートソルブ: {selected['model_label']}を適用"
            )
            self.update_start_button_state()
        except Exception as exc:
            self.plate_solve_status_var.set("プレートソルブ: モデル適用失敗")
            self._show_plate_solve_message(
                "showerror", "カメラ補正データ", f"補正データを適用できませんでした:\n{exc}"
            )

    def _handle_camera_model_status(self, text: str):
        try:
            self.camera_model_status_var.set(str(text or ""))
        except tk.TclError:
            pass

    def _handle_camera_model_progress(self, payload):
        """Apply an automatic camera-model progress event on the Tk thread."""
        if not isinstance(payload, dict):
            return
        percent = max(0, min(100, int(payload.get("percent", 0) or 0)))
        self.camera_model_progress_var.set(f"自動作成: {percent}%")
        image_path = str(payload.get("input_image_path", "") or "")
        if image_path:
            self.camera_model_input_image_path = image_path
        classification = payload.get("classification")
        if isinstance(classification, dict):
            self.camera_model_input_image_info = dict(classification)
        if payload.get("input_image_at"):
            self.camera_model_input_image_info["input_image_at"] = payload["input_image_at"]

    def _queue_camera_model_progress(self, payload):
        try:
            self.progress_queue.put((None, {"camera_model_progress": dict(payload)}))
        except Exception:
            pass

    def bind_camera_model_input_hover(self, *widgets):
        for widget in widgets:
            widget.bind("<Enter>", self._camera_model_hover_enter, add="+")
            widget.bind("<Leave>", self._camera_model_schedule_hover_close, add="+")

    def _camera_model_cancel_hover_close(self, _event=None):
        if self._camera_model_hover_close_job is not None:
            try:
                self.after_cancel(self._camera_model_hover_close_job)
            except (tk.TclError, ValueError):
                pass
            self._camera_model_hover_close_job = None

    def _camera_model_schedule_hover_close(self, _event=None):
        self._camera_model_cancel_hover_close()
        self._camera_model_hover_close_job = self.after(220, self._hide_camera_model_hover)

    def _camera_model_hover_enter(self, _event=None):
        self._camera_model_cancel_hover_close()
        if self._camera_model_hover_popup is not None:
            return
        image_path = getattr(self, "camera_model_input_image_path", "")
        if not image_path or not os.path.isfile(image_path):
            return
        try:
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((720, 405), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
        except (OSError, ValueError):
            return

        popup = Toplevel(self)
        popup.title("自動監視の入力画像")
        popup.transient(self)
        popup.configure(background=ui_theme.COLORS["content"])
        popup.protocol("WM_DELETE_WINDOW", self._hide_camera_model_hover)
        image_label = ttk.Label(popup, image=photo)
        image_label.pack(padx=10, pady=(10, 6))
        info = getattr(self, "camera_model_input_image_info", {}) or {}
        cloud_fraction = info.get("cloud_fraction")
        if cloud_fraction is not None:
            try:
                cloud_text = f"雲量判定: {float(cloud_fraction) * 100:.1f}%"
            except (TypeError, ValueError):
                cloud_text = "雲量判定: -"
        else:
            cloud_text = "雲量判定: -"
        source_text = str(info.get("source", ""))
        timestamp = str(info.get("input_image_at", ""))
        caption = f"実入力画像  /  {cloud_text}"
        if source_text:
            caption += f"  /  {source_text}"
        if timestamp:
            caption += f"  /  {timestamp.replace('T', ' ')[:19]}"
        ttk.Label(popup, text=caption).pack(padx=10, pady=(0, 10))
        popup.bind("<Enter>", self._camera_model_cancel_hover_close, add="+")
        popup.bind("<Leave>", self._camera_model_schedule_hover_close, add="+")
        image_label.bind("<Enter>", self._camera_model_cancel_hover_close, add="+")
        image_label.bind("<Leave>", self._camera_model_schedule_hover_close, add="+")
        popup.update_idletasks()
        x = self.winfo_pointerx() + 14
        y = self.winfo_pointery() + 14
        popup.geometry(f"+{x}+{y}")
        self._camera_model_hover_popup = popup
        self._camera_model_hover_photo = photo

    def _hide_camera_model_hover(self):
        self._camera_model_cancel_hover_close()
        popup = self._camera_model_hover_popup
        self._camera_model_hover_popup = None
        self._camera_model_hover_photo = None
        if popup is not None:
            try:
                popup.destroy()
            except tk.TclError:
                pass

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
        if threading.current_thread() is threading.main_thread():
            self._handle_camera_model_status(text)
        else:
            try:
                self.progress_queue.put((None, {"camera_model_status": str(text)}))
            except Exception:
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
        method = self.camera_model_method_var.get().strip()
        use_trajectory = method != self._STATIC_MODEL_METHOD
        current = dict(self.global_wcs_info or {})
        initial_model_path = self.plate_solve_model_path_var.get().strip()
        if not initial_model_path:
            initial_model_path = str(
                current.get("model_path") or current.get("calibration_path") or ""
            ).strip()
        if initial_model_path and not os.path.isfile(initial_model_path):
            initial_model_path = ""
        model_kind = "星の動きモデル" if use_trajectory else "静止画モデル"
        self._set_camera_model_status(f"高精度モデル: {model_kind}を作成中...")

        def progress(message):
            self._set_camera_model_status(f"高精度モデル: {message}")
            try:
                self.progress_queue.put((message, None))
            except Exception:
                pass

        def worker():
            result = _build_camera_model_for_app(
                request,
                use_trajectory=use_trajectory,
                initial_model_path=initial_model_path,
                progress_callback=progress,
            )

            def finished():
                self.btn_build_camera_model.configure(state=tk.NORMAL)
                if result.success and result.enabled:
                    trajectory_count = int(getattr(result, "trajectory_count", 0) or 0)
                    trajectory_text = f" / 恒星軌跡 {trajectory_count}本" if trajectory_count else ""
                    self.camera_model_status_var.set(
                        f"高精度モデル: 登録済み（被覆率 {result.support_fraction * 100:.0f}% / "
                        f"p95 {result.residual_p95_px:.2f}px{trajectory_text}）"
                    )
                    self.plate_solve_model_path_var.set(result.model_path)
                    self._refresh_plate_solve_model_choices()
                    selected = next(
                        (item for item in self.plate_solve_model_entries
                         if os.path.abspath(item["path"]) == os.path.abspath(result.model_path)),
                        None,
                    )
                    if selected is not None:
                        self.plate_solve_model_var.set(selected["display_name"])
                        self._update_plate_solve_model_info()
                    self.plate_solve_wcs_path_var.set(result.model_path)
                    self.plate_solve_status_var.set("プレートソルブ: 高精度カメラ補正データを適用")
                    self.global_wcs_info = {
                        "wcs_file": result.model_path,
                        "calibration_path": result.model_path,
                        "model_path": result.model_path,
                        "job_id": "local-wideangle-camera-model",
                    }
                    self.update_start_button_state()
                    if bool(getattr(result, "target_met", False)):
                        messagebox.showinfo("高精度モデル", "選択範囲から高精度カメラ補正データを作成し、登録しました。", parent=self)
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
            self.camera_model_progress_var.set("")
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
            progress_callback=self._queue_camera_model_progress,
        )
        self.camera_model_monitor = monitor
        monitor.start()
        self.btn_toggle_camera_model_monitor.configure(text="RTSP自動監視を停止")
        self.camera_model_progress_var.set("自動作成: 0%")
        self._set_camera_model_status(f"高精度モデル: RTSPを{interval}秒間隔で監視中")

    def select_plate_solve_wcs_file(self):
        file_path = filedialog.askopenfilename(title="既存のWCS/カメラ補正データを選択", filetypes=[("WCS/FITS/補正データ", "*.wcs *.fits *.fit *.json"), ("すべてのファイル", "*.*")])
        if file_path:
            try:
                ps_datetime = None
                local_wideangle_wcs = False
                if file_path.lower().endswith(".json"):
                    import local_wideangle_astrometry
                    metadata, _model = local_wideangle_astrometry._load_calibration(file_path)
                    if metadata.get("model_type") != "fixed-camera-stg-poly":
                        raise ValueError("カメラ補正データJSONではありません。")
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
                    messagebox.showinfo("成功", "カメラ補正データを読み込みました。", parent=self)
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

