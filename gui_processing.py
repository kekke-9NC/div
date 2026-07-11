from gui_common import *
import gui_common as common
import ui_state


class ProcessingMixin:
    def start_processing(self):
        # 詳細設定をconfigに適用
        self.apply_advanced_settings_to_config()
        if not self.apply_selected_model(silent=True):
            messagebox.showerror("設定エラー", "有効な学習モデルを選択してください。")
            return
        if (self.worker_thread and self.worker_thread.is_alive()) or \
           (self.rtsp_thread and self.rtsp_thread.is_alive()) or \
           (self.periodic_scan_thread and self.periodic_scan_thread.is_alive()):
            messagebox.showwarning("情報", "別のプロセスが実行中です。")
            return

        self.cancel_flag.clear()
        self.append_log("処理準備中...")

        try:
            params = {
                'max_workers': int(self.concurrency_var.get()),
                'interval_sec': float(self.interval_var.get()),
                'duration_sec': float(self.duration_var.get()),
                'save_options': {k: v.get() for k, v in self.save_options_vars.items()},
                'meteor_save_path': self.meteor_save_path_var.get(),
                'not_meteor_save_path': self.not_meteor_save_path_var.get(),
                'mask': self.mask_image if self.apply_mask_var.get() else None,
                'global_wcs_info': self.global_wcs_info if self.use_plate_solve_var.get() else None,
                'plate_solve_mask': self.plate_solve_mask_image,
                'fixed_pattern_correction': self.rtsp_dark_frame if self.apply_rtsp_dark_var.get() else None,
                'rtsp_notification_sound': self.rtsp_notification_sound_var.get(),
                'summary_config': [item.copy() for item in self.summary_video_config]
            }
            try:
                timestamp_size_percent = float(self.full_video_timestamp_size_var.get())
            except ValueError:
                timestamp_size_percent = config.FULL_VIDEO_TIMESTAMP_SIZE_PERCENT
            params['save_options'].update({
                'full_video_timestamp_enabled': self.full_video_timestamp_enabled_var.get(),
                'full_video_timestamp_position': self.full_video_timestamp_position_var.get(),
                'full_video_timestamp_size_percent': timestamp_size_percent,
                'ml_training_export_enabled': self.ml_training_export_enabled_var.get(),
                'ml_training_data_root': self.ml_training_data_root_var.get().strip(),
                'auto_video_mask_enabled': self.auto_video_mask_enabled_var.get(),
                'auto_video_mask_cache_dir': config.AUTO_VIDEO_MASK_CACHE_DIR,
            })
            if params['save_options']['ml_training_export_enabled']:
                if not params['save_options']['ml_training_data_root']:
                    raise ValueError("機械学習向けデータの保存先が空です。")
                os.makedirs(params['save_options']['ml_training_data_root'], exist_ok=True)
            
            if self.rtsp_preset_var.get() == "clear":
                preset = config.RTSP_PRESET_CLEAR_SKY
            else:
                preset = config.RTSP_PRESET_CLOUDY
            config.RTSP_MIN_LINE_LENGTH = preset['min_line_length']
            config.RTSP_HOUGH_THRESHOLD = preset['hough_threshold']
            config.RTSP_CANNY_THRESH1 = preset['canny_thresh1']
            config.RTSP_CANNY_THRESH2 = preset['canny_thresh2']
            
            try:
                config.RTSP_FPS = int(self.rtsp_fps_var.get())
            except ValueError:
                messagebox.showwarning("設定警告", f"FPS値が無効です。デフォルト値({config.RTSP_FPS})を使用します。")
            
            self.append_log(f"RTSP検出プリセット: {preset['name']}, FPS: {config.RTSP_FPS}")
            os.makedirs(params['meteor_save_path'], exist_ok=True)
            os.makedirs(params['not_meteor_save_path'], exist_ok=True)
            # set temp_video dir path on the App instance so GUI can shorten logs
            module_dir = os.path.dirname(os.path.abspath(__file__))
            self.temp_video_dir = os.path.join(module_dir, 'temp_video')
            os.makedirs(self.temp_video_dir, exist_ok=True)
        except (ValueError, Exception) as e:
            messagebox.showerror("設定エラー", f"パラメータ値が無効です: {e}")
            return

        self.start_button.config(state=tk.DISABLED)
        self.cancel_button.config(state=tk.NORMAL)
        self.status_label.config(text="処理中...")
        self.progress['value'] = 0
        self.eta_label.config(text="ETA: 計算中...")
        self.elapsed_label.config(text="経過: 00:00:00")
        self.start_time_gui = time.time()

        is_periodic = self.periodic_scan_var.get()
        selected_source = ui_state.select_source_by_priority(
            getattr(self, "processing_source_priority", ui_state.SOURCE_PRIORITY_DEFAULT),
            periodic_enabled=is_periodic,
            has_rtsp=bool(self.rtsp_urls),
            has_folder=bool(self.folder_paths),
        )
        active_source_count = sum((is_periodic, bool(self.rtsp_urls), bool(self.folder_paths)))
        if active_source_count > 1 and selected_source:
            source_labels = {
                "periodic": "定期スキャン",
                "rtsp": "RTSPストリーム",
                "folder": "フォルダ／動画ファイル",
            }
            self.append_log(f"複数の入力が有効です。優先順位により「{source_labels[selected_source]}」のみ実行します。")

        if selected_source == "periodic":
            periodic_dir = self.periodic_dir_var.get().strip()
            if not periodic_dir or not os.path.isdir(periodic_dir):
                messagebox.showerror("設定エラー", "定期スキャン用の有効な監視フォルダを選択してください。")
                self.cancel_processing(restore_button_state=True)
                return
            
            log_msg = f"定期スキャン開始 (フォルダ: {periodic_dir})"
            monitor_kwargs = {
                'directory': periodic_dir, 'scan_interval': int(self.periodic_interval_var.get()),
                'progress_callback': self.progress_queue.put, 'mask': params['mask'], 
                'global_wcs_info': params['global_wcs_info'], 'plate_solve_mask': params['plate_solve_mask'], 
                'meteor_save_path': params['meteor_save_path'], 'not_meteor_save_path': params['not_meteor_save_path'], 
                'cancel_flag': self.cancel_flag, 'save_options': params['save_options'], 
                'interval': params['interval_sec'], 'duration': params['duration_sec'], 
                'min_length': config.MIN_LINE_LENGTH, 'summary_video_config': params['summary_config'],
                'time_limit_enabled': self.periodic_time_limit_var.get(),
                'start_hour': int(self.start_hour_var.get()), 'start_minute': int(self.start_min_var.get()),
                'end_hour': int(self.end_hour_var.get()), 'end_minute': int(self.end_min_var.get()),
                'fixed_pattern_correction': params['fixed_pattern_correction'],
            }
            if monitor_kwargs['time_limit_enabled']:
                log_msg += f", 時間制限: {monitor_kwargs['start_hour']:02d}:{monitor_kwargs['start_minute']:02d} - {monitor_kwargs['end_hour']:02d}:{monitor_kwargs['end_minute']:02d}"
            self.append_log(log_msg)

            self.periodic_scan_thread = threading.Thread(target=file_utils.monitor_directory, kwargs=monitor_kwargs, daemon=True)
            self.periodic_scan_thread.start()

        elif selected_source == "rtsp":
            url = self.rtsp_urls[0]
            rtsp_time_limit = self.rtsp_time_limit_var.get()
            rtsp_sh = int(self.rtsp_start_hour_var.get())
            rtsp_sm = int(self.rtsp_start_min_var.get())
            rtsp_eh = int(self.rtsp_end_hour_var.get())
            rtsp_em = int(self.rtsp_end_min_var.get())
            
            log_msg = f"RTSP処理開始 (URL: {url}, 並列処理数: {params['max_workers']})"
            if rtsp_time_limit:
                log_msg += f", 録画時間制限: {rtsp_sh:02d}:{rtsp_sm:02d} - {rtsp_eh:02d}:{rtsp_em:02d}"
            self.append_log(log_msg)
            self.append_log(
                f"流星検出通知音: {'ON' if params['rtsp_notification_sound'] else 'OFF'}"
            )
            if params['fixed_pattern_correction'] is not None:
                self.append_log("保存物補正: 適応固定パターン＋21フレーム平均 ON")
            
            rtsp_args = (
                url, config.RTSP_SAVE_ROOT, config.RTSP_SEGMENT_DURATION, 60, self.progress_queue.put,
                params['mask'], params['global_wcs_info'], params['plate_solve_mask'],
                params['meteor_save_path'], params['not_meteor_save_path'], self.cancel_flag,
                params['save_options'], params['interval_sec'], params['duration_sec'],
                config.MIN_LINE_LENGTH, params['summary_config'],
                rtsp_time_limit, rtsp_sh, rtsp_sm, rtsp_eh, rtsp_em,
                params['max_workers'], self.handle_rtsp_live_preview_frame, params['fixed_pattern_correction'],
                params['rtsp_notification_sound']
            )
            self.rtsp_thread = threading.Thread(target=file_utils.rtsp_save_and_process_thread_target, args=rtsp_args, daemon=True)
            self.rtsp_thread.start()
            self._update_live_preview_button_state()

        elif selected_source == "folder":
            sources_to_process = []
            self.append_log(f"{len(self.folder_paths)}個の項目を処理します...")
            try:
                observation_lat = float(self.observation_latitude_var.get())
                observation_lon = float(self.observation_longitude_var.get())
            except ValueError:
                messagebox.showerror("設定エラー", "観測地点の緯度・経度を数値で入力してください。")
                self.cancel_processing(restore_button_state=True)
                return
            for path_item in self.folder_paths:
                p = Path(path_item)
                if p.is_dir():
                    found = sorted([p for p in p.rglob('*') if p.suffix.lower() in config.PERIODIC_VIDEO_EXTENSIONS])
                    if self.date_folder_twilight_filter_enabled_var.get():
                        filtered, filter_info = observation_time_filter.filter_date_root_videos(
                            p, found, observation_lat, observation_lon
                        )
                        if filter_info.get('applied'):
                            dawn = filter_info['astro_dawn'].strftime('%H:%M')
                            dusk = filter_info['astro_dusk'].strftime('%H:%M')
                            self.append_log(
                                f"{p.name}: 天文薄明フィルタ {dawn}以前 / {dusk}以後 "
                                f"({len(found)}本 → {len(filtered)}本)"
                            )
                            found = filtered
                    sources_to_process.extend([{'path': str(fp), 'is_rtsp': False} for fp in found])
                elif p.is_file() and p.suffix.lower() in config.PERIODIC_VIDEO_EXTENSIONS:
                    sources_to_process.append({'path': str(p), 'is_rtsp': False})
            
            if not sources_to_process:
                messagebox.showwarning("情報", "選択されたフォルダに動画ファイルが見つかりませんでした。")
                self.cancel_processing(restore_button_state=True)
                return

            total_videos = len(sources_to_process)
            self.append_log(f"合計 {total_videos} 個の動画ファイルを処理します。")
            self.progress['maximum'] = total_videos
            self.progress_queue.put((f"処理開始 ({total_videos} ファイル)", (0, total_videos)))

            worker_args = (
                self.progress_queue, sources_to_process, params['max_workers'], params['interval_sec'], 
                params['duration_sec'], params['mask'], params['global_wcs_info'], params['plate_solve_mask'],
                params['meteor_save_path'], params['not_meteor_save_path'], self.cancel_flag,
                params['save_options'], params['summary_config'], params['fixed_pattern_correction']
            )
            self.worker_thread = threading.Thread(target=worker_main_loop, args=worker_args, daemon=True)
            self.worker_thread.start()
        else:
            messagebox.showerror("エラー", "処理対象がありません。")
            self.cancel_processing(restore_button_state=True)

    def cancel_processing(self, restore_button_state=False):
        # リクエストが来たら直ちにキャンセルフラグを立て、UIを更新する
        if not self.cancel_flag.is_set():
            self.append_log("キャンセル要求を受け付けました...")
        else:
            self.append_log("キャンセル要求 (再送) ...")

        # Notify workers and update UI state immediately
        self.cancel_flag.set()
        try:
            self.cancel_button.config(state=tk.DISABLED)
        except Exception:
            pass
        self.status_label.config(text="キャンセル中...")

        self.start_time_gui = None

        # allow Start to be pressed again immediately after cancel requested
        try:
            self.update_start_button_state()
        except Exception:
            pass

        if restore_button_state:
            # restore start button state and label when requested by caller
            self.update_start_button_state()
            self.status_label.config(text="停止")
        self.close_rtsp_live_preview()
        self._update_live_preview_button_state()

    def update_progress(self):
        if self.start_time_gui:
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - self.start_time_gui))
            self.elapsed_label.config(text=f"経過: {elapsed_str}")

        try:
            while True:
                message, value = self.progress_queue.get_nowait()
                if isinstance(value, tuple) and len(value) == 2:
                    current, total = value
                    if total > 0:
                        self.progress['maximum'] = total
                        self.progress['value'] = max(0, min(current, total))
                        self.status_label.config(text=f"処理中... ({int(self.progress['value'])}/{int(self.progress['maximum'])})")
                if self.start_time_gui and self.progress['maximum'] > 0 and self.progress['value'] > 0:
                    elapsed = time.time() - self.start_time_gui
                    avg_time = elapsed / self.progress['value']
                    eta_sec = avg_time * (self.progress['maximum'] - self.progress['value'])
                    self.eta_label.config(text=f"ETA: {time.strftime('%H:%M:%S', time.gmtime(eta_sec))}")

                if message:
                    # Shorten messages that reference temporary copied files
                    msg = message
                    try:
                        tmp_root = getattr(self, 'temp_video_dir', None)
                        if tmp_root and isinstance(msg, str):
                            idx = msg.find(tmp_root)
                            if idx != -1:
                                # find bounds of path in the message (try quotes first)
                                start_q = msg.rfind('"', 0, idx)
                                end_q = msg.find('"', idx)
                                if start_q == -1:
                                    start = idx
                                else:
                                    start = start_q + 1
                                if end_q == -1:
                                    # fallback: space or end
                                    space_pos = msg.find(' ', idx)
                                    end = space_pos if (space_pos != -1) else len(msg)
                                else:
                                    end = end_q

                                full_path = msg[idx:end]
                                norm = os.path.normpath(full_path)
                                parts = norm.split(os.sep)
                                # find netcopy_<id> segment
                                net_idx = next((i for i, p in enumerate(parts) if p.startswith('netcopy_')), None)
                                if net_idx is not None and len(parts) > net_idx + 2:
                                    # skip netcopy and drive-name segments
                                    simp_parts = parts[net_idx + 2:]
                                    simp = os.sep.join(simp_parts)
                                else:
                                    simp = os.path.basename(norm)

                                # replace the full path in the message with the simplified path (preserve quotes if present)
                                if start_q != -1 and end_q != -1:
                                    msg = msg[:start_q+1] + simp + msg[end_q:]
                                else:
                                    msg = msg.replace(full_path, simp)
                    except Exception:
                        pass

                    self.append_log(msg)
                    # Consider the run complete only on explicit completion/cancel messages.
                    # Avoid treating transient error words in exception text as 'complete',
                    # because many libraries surface English words like 'failed' in tracebacks
                    # and that would erroneously stop ETA updates.
                    is_complete = (
                        "すべての処理が完了しました" in message or
                        "監視を終了しました" in message or
                        "統合処理終了" in message or
                        "処理はキャンセルされました" in message
                    )
                    if is_complete:
                        self.update_start_button_state()
                        self.cancel_button.config(state=tk.DISABLED)
                        self.close_rtsp_live_preview()
                        self.status_label.config(text="完了/停止")
                        if "すべての処理が完了しました" in message:
                            self.progress['value'] = self.progress['maximum']
                        self.start_time_gui = None
                        self.cancel_flag.clear()
        except queue.Empty:
            pass

        self.after(100, self.update_progress)



def worker_main_loop(
    progress_queue: queue.Queue, sources: List[Dict[str, Any]], max_workers: int, interval: float, duration: float,
    mask: Optional[np.ndarray], global_wcs_info: Optional[Dict], plate_solve_mask: Optional[np.ndarray],
    meteor_save_path: str, not_meteor_save_path: str, cancel_flag: threading.Event,
    save_options: Dict[str, bool], summary_video_config: List[Dict[str, Any]],
    fixed_pattern_correction: Optional[np.ndarray] = None,
):
    total_videos = len(sources)
    if total_videos == 0: return

    # Use a two-stage pipeline (downloaders + processors) implemented in download_pipeline
    module_dir = os.path.dirname(os.path.abspath(__file__))
    tmp_root_dir = os.path.join(module_dir, 'temp_video')
    os.makedirs(tmp_root_dir, exist_ok=True)

    try:
        download_pipeline.run_pipeline(
            sources=sources,
            max_workers=max_workers,
            interval=interval,
            duration=duration,
            mask=mask,
            global_wcs_info=global_wcs_info,
            plate_solve_mask=plate_solve_mask,
            meteor_save_path=meteor_save_path,
            not_meteor_save_path=not_meteor_save_path,
            cancel_flag=cancel_flag,
            progress_callback=progress_queue.put,
            save_options=save_options,
            summary_video_config=summary_video_config,
            tmp_root=tmp_root_dir,
            status_callback=common.STATUS_CALLBACK,
            fixed_pattern_correction=fixed_pattern_correction,
        )

        if not cancel_flag.is_set():
            progress_queue.put(("すべての処理が完了しました。", None))
        else:
            progress_queue.put(("処理はキャンセルされました。", None))

    except Exception as e:
        progress_queue.put((f"パイプライン実行中に例外が発生しました: {e}", None))
