from gui_common import *
from gui_dialogs import ProcessingOptionDialog, TimelapseDragDropWindow
from concurrent.futures import ThreadPoolExecutor, as_completed


class SynthesisMixin:
    def _run_synthesis_task_async(self, task, *, output_path, item_label, cleanup=None):
        """Run a CPU/IO-heavy synthesis task without ever touching Tk from its worker.

        Tcl/Tk (especially on macOS) is not thread-safe.  A number of synthesis
        paths used to call ``append_log`` and ``messagebox`` directly from their
        worker thread, which could terminate the application instead of merely
        reporting a failed conversion.  The worker now communicates through a
        queue and this method performs every UI operation on the Tk thread.
        """
        events = queue.Queue()
        completed = {"value": False}

        def poll_events():
            try:
                while True:
                    event_type, *payload = events.get_nowait()
                    if event_type == "log":
                        self.append_log(payload[0])
                    elif event_type == "done":
                        completed["value"] = True
                        success = payload[0]
                        if success:
                            messagebox.showinfo(
                                "完了",
                                f"{item_label}の作成が完了しました。\n保存先: {output_path}",
                                parent=self,
                            )
                            self.append_log(f"{item_label}の作成が完了しました: {output_path}")
                        else:
                            messagebox.showerror(
                                "エラー",
                                f"{item_label}の作成に失敗しました。ログを確認してください。",
                                parent=self,
                            )
                            self.append_log(f"{item_label}の作成に失敗しました。")
                    elif event_type == "error":
                        completed["value"] = True
                        self.append_log(f"{item_label}の作成中にエラーが発生しました: {payload[0]}")
                        messagebox.showerror(
                            "エラー", f"{item_label}の作成中にエラーが発生しました。\n{payload[0]}", parent=self
                        )
            except queue.Empty:
                pass
            if not completed["value"]:
                self.after(50, poll_events)

        def run_task():
            try:
                success = bool(task(lambda message: events.put(("log", message))))
                events.put(("done", success))
            except Exception as exc:
                import traceback
                traceback.print_exc()
                events.put(("error", str(exc)))
            finally:
                if cleanup is not None:
                    try:
                        cleanup()
                    except Exception:
                        pass

        self.after(50, poll_events)
        threading.Thread(target=run_task, daemon=True).start()

    def create_lighten_blend_video_callback(self):
        """Callback for the 'Create Lighten Blend Video' button."""
        initial_dir = self.meteor_save_path_var.get()
        if not initial_dir or not os.path.exists(initial_dir):
            initial_dir = os.path.expanduser("~")
        
        video_files = filedialog.askopenfilenames(
            title="比較明合成する動画ファイルを選択（複数可）",
            initialdir=initial_dir,
            filetypes=[
                ("動画ファイル", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("すべてのファイル", "*.*")
            ]
        )
        
        if not video_files:
            return
        
        if len(video_files) < 2:
            messagebox.showwarning("警告", "比較明合成を行うには2つ以上の動画ファイルを選択してください。")
            return
        
        default_output = lighten_blend_video.get_default_output_path()
        
        output_path = filedialog.asksaveasfilename(
            title="比較明合成動画の保存先",
            initialdir=os.path.dirname(default_output),
            initialfile=os.path.basename(default_output),
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4"), ("AVI Video", "*.avi"), ("All Files", "*")]
        )
        
        if not output_path:
            return
        output_path = self._ensure_date_prefix(output_path)
        
        self.append_log(f"比較明合成動画の作成を開始します... ({len(video_files)}個の動画)")
        
        dialog = ProcessingOptionDialog(self)
        if dialog.result is None:  # キャンセル
            return
            
        mode = dialog.result  # 0:通常, 1:明るいエリアマスク, 2:流星のみ
        # mode=1(明るいエリア)は現状 mode=2(流星のみ)と同じ扱いにする。
        if mode == 0:
            self._run_synthesis_task_async(
                lambda progress_callback: lighten_blend_video.create_lighten_blend_video(
                    list(video_files),
                    output_path,
                    progress_callback=progress_callback,
                ),
                output_path=output_path,
                item_label="比較明合成動画",
            )
        else:
            self._create_lighten_blend_video_with_meteor_detection(video_files, output_path)

    def _create_lighten_blend_video_with_meteor_detection(self, video_files, output_path):
        """AI流星検出を使用して比較明合成動画を作成（各動画ごとに検出）"""
        import detection_preview
        import bright_area_detector
        import gc
        
        self.append_log("各動画から比較明合成画像を作成し、流星を検出します...")
        
        # 動画ごとの一時ファイルパスを保存（画像はメモリに保持しない）
        video_composites = {}  # {video_path: {'temp_path': str, 'filename': str, 'shape': (h, w)}}
        prep_events = queue.Queue()
        prep_finished = {"value": False}

        def _run_prep_task():
            worker_log = lambda message: prep_events.put(("log", message))
            if not self._ensure_ai_model_loaded(bright_area_detector, log_callback=worker_log):
                prep_events.put(("error", "AIモデルのロードまたは接続確認に失敗しました。"))
                return

            # Step 1: 各動画から個別に比較明合成画像を作成
            for i, vp in enumerate(video_files):
                worker_log(f"動画 {i+1}/{len(video_files)} から合成画像を作成中: {os.path.basename(vp)}")
                
                composite_image = lighten_blend_video.create_composite_from_videos(
                    [vp],  # 1つの動画のみ
                    progress_callback=None,  # 個別のログは抑制
                    sample_interval=1  # 全フレームを使用（流星を見逃さないため）
                )
                
                if composite_image is not None:
                    # 一時ファイルとして保存
                    temp_path = os.path.join(config.TEMP_CLIP_DIR, f"temp_composite_{i}_{os.path.basename(vp)}.png")
                    h, w = composite_image.shape[:2]
                    cv2.imwrite(temp_path, composite_image)
                    
                    video_composites[vp] = {
                        'temp_path': temp_path,
                        'filename': os.path.basename(temp_path),
                        'shape': (h, w)  # サイズ情報のみ保持
                    }
                    
                    # メモリ解放
                    del composite_image
                    gc.collect()
                else:
                    worker_log(f"警告: 動画から合成画像を作成できませんでした: {os.path.basename(vp)}")
            
            if not video_composites:
                prep_events.put(("error", "有効な合成画像を作成できませんでした。"))
                return
            
            worker_log(f"{len(video_composites)}個の合成画像を作成しました。")
            
            # メインスレッドでプレビューウィンドウを開く
            def open_preview():
                # 合成開始コールバック
                def start_video_synthesis_with_results(results):
                    # 動画ごとの個別マスクを作成（和集合ではなく、各動画に対応するマスクのみ適用）
                    per_video_masks = {}
                    base_shape = None
                    has_detections = False
                    
                    for vp, data in video_composites.items():
                        filename = data['filename']
                        if filename in results:
                            boxes = results[filename]['boxes']
                            if boxes:
                                has_detections = True
                                h, w = data['shape']
                                if base_shape is None:
                                    base_shape = (h, w)
                                
                                mask = bright_area_detector.create_inclusion_mask_from_boxes(
                                    (h, w), boxes, margin=40
                                )
                                
                                # サイズが異なる場合はリサイズ
                                if base_shape != (h, w):
                                    mask = cv2.resize(mask, (base_shape[1], base_shape[0]))
                                
                                # 動画パスをキーとして個別マスクを保存
                                per_video_masks[vp] = mask
                    
                    if not has_detections:
                        if not messagebox.askyesno("確認", "流星が検出されていないか、選択されていません。\\nマスクなしで（通常の比較明合成として）作成しますか？"):
                            # 一時ファイルのクリーンアップ
                            for data in video_composites.values():
                                try:
                                    if os.path.exists(data['temp_path']):
                                        os.remove(data['temp_path'])
                                except:
                                    pass
                            return
                    
                    # 動画ごとのマスクを適用して動画作成
                    def video_synthesis_task(progress_callback):
                        progress_callback("動画ごとのマスクを適用して動画を作成中...")
                        return lighten_blend_video.create_lighten_blend_video(
                            list(video_files),
                            output_path,
                            progress_callback=progress_callback,
                            per_video_masks=per_video_masks if per_video_masks else None
                        )

                    def cleanup_temp_files():
                        for data in video_composites.values():
                            try:
                                if os.path.exists(data['temp_path']):
                                    os.remove(data['temp_path'])
                            except:
                                pass

                    self._run_synthesis_task_async(
                        video_synthesis_task,
                        output_path=output_path,
                        item_label="比較明合成動画",
                        cleanup=cleanup_temp_files,
                    )
                
                # プレビューウィンドウ作成
                preview_window = detection_preview.DetectionPreviewWindow(
                    self, start_video_synthesis_with_results
                )
                detection_events = queue.Queue()
                detection_finished = {"value": False}

                def poll_detection_events():
                    try:
                        while True:
                            event_type, *payload = detection_events.get_nowait()
                            if event_type == "log":
                                self.append_log(payload[0])
                            elif event_type == "start" and preview_window.winfo_exists():
                                preview_window.start_analysis(payload[0])
                            elif event_type == "item" and preview_window.winfo_exists():
                                filename, temp_path, boxes = payload
                                preview_window.add_item(
                                    filename,
                                    temp_path,
                                    boxes,
                                    lambda image: bright_area_detector.detect_meteors_with_boxes(image) or (None, []),
                                )
                            elif event_type == "done":
                                detection_finished["value"] = True
                                if preview_window.winfo_exists():
                                    preview_window.finalize_analysis()
                            elif event_type == "error":
                                detection_finished["value"] = True
                                self.append_log(f"AI検出中にエラーが発生しました: {payload[0]}")
                                if preview_window.winfo_exists():
                                    preview_window.destroy()
                                messagebox.showerror("AI解析エラー", payload[0], parent=self)
                    except queue.Empty:
                        pass
                    if not detection_finished["value"]:
                        self.after(50, poll_detection_events)
                
                # 各動画の合成画像に対して検出実行
                def run_detection():
                    try:
                        total = len(video_composites)
                        detection_events.put(("log", f"AIによる流星検出を開始します... ({total}個の画像)"))
                        detection_events.put(("start", total))

                        for i, (vp, data) in enumerate(video_composites.items()):
                            detection_events.put(("log", f"検出中 ({i+1}/{total}): {os.path.basename(vp)}"))

                            # 一時ファイルから画像を読み込み（メモリ節約）
                            composite_img = imread_with_japanese_path(data['temp_path'])
                            if composite_img is None:
                                detection_events.put(("log", f"警告: 画像を読み込めませんでした: {data['filename']}"))
                                continue

                            res = bright_area_detector.detect_meteors_with_boxes(
                                composite_img,
                                progress_callback=None,  # 個別ログ抑制
                            )
                            boxes = res[1] if res else []
                            del composite_img
                            gc.collect()

                            filename = data['filename']
                            temp_path = data['temp_path']
                            detection_events.put(("item", filename, temp_path, boxes))

                        detection_events.put(("log", "全画像の検出が完了しました。結果を確認してください。"))
                        detection_events.put(("done",))
                    except Exception as exc:
                        import traceback
                        traceback.print_exc()
                        detection_events.put(("error", str(exc)))
                
                self.after(50, poll_detection_events)
                threading.Thread(target=run_detection, daemon=True).start()
            
            prep_events.put(("ready", open_preview))

        def run_prep_task():
            try:
                _run_prep_task()
            except Exception as exc:
                import traceback
                traceback.print_exc()
                prep_events.put(("error", str(exc)))
        
        def poll_prep_events():
            try:
                while True:
                    event_type, *payload = prep_events.get_nowait()
                    if event_type == "log":
                        self.append_log(payload[0])
                    elif event_type == "error":
                        prep_finished["value"] = True
                        self.append_log(f"AI解析準備中にエラーが発生しました: {payload[0]}")
                        messagebox.showerror("AI解析エラー", payload[0], parent=self)
                    elif event_type == "ready":
                        prep_finished["value"] = True
                        payload[0]()
            except queue.Empty:
                pass
            if not prep_finished["value"]:
                self.after(50, poll_prep_events)

        self.after(50, poll_prep_events)
        threading.Thread(target=run_prep_task, daemon=True).start()

    def create_lighten_blend_image_callback(self):
        """比較明合成画像作成ボタンのコールバック"""
        initial_dir = self.meteor_save_path_var.get()
        if not initial_dir or not os.path.exists(initial_dir):
            initial_dir = os.path.expanduser("~")
        
        # ファイル選択ダイアログで複数の画像/動画ファイルを選択
        file_paths_tuple = filedialog.askopenfilenames(
            title="比較明合成する画像・動画ファイルを選択（複数可）",
            initialdir=initial_dir,
            filetypes=[
                ("画像・動画ファイル", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.mp4 *.avi *.mov *.mkv *.wmv"),
                ("画像ファイル", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("動画ファイル", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("すべてのファイル", "*.*")
            ]
        )
        
        if not file_paths_tuple:
            return
            
        file_paths = set(file_paths_tuple)
        
        if len(file_paths) < 1:
            messagebox.showwarning("警告", "有効なファイルが見つかりません。")
            return

        # 初期ディレクトリ設定
        default_output = "composite.png"
        if len(file_paths) == 1:
            first_path = list(file_paths)[0]
            if os.path.isdir(first_path):
                default_output = os.path.join(os.path.dirname(first_path), f"{os.path.basename(first_path)}_composite.png")
            else:
                 default_output = os.path.join(os.path.dirname(first_path), f"{os.path.splitext(os.path.basename(first_path))[0]}_composite.png")
        
        # ユーザーに保存先を確認
        output_path = filedialog.asksaveasfilename(
            title="比較明合成画像の保存先",
            initialdir=os.path.dirname(default_output),
            initialfile=os.path.basename(default_output),
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("All Files", "*")]
        )
        
        if not output_path:
            return
        output_path = self._ensure_date_prefix(output_path)

        dialog = ProcessingOptionDialog(self)
        if dialog.result is None:  # キャンセル
            return
            
        mode = dialog.result  # 0:通常, 1:明るいエリアマスク, 2:流星のみ
        is_ai_mode = (mode != 0)
        is_meteor_mode = (mode == 2)
        print(f"DEBUG: ProcessingOptionDialog result: mode={mode}, is_ai_mode={is_ai_mode}, is_meteor_mode={is_meteor_mode}")

        if not is_ai_mode:
            self._run_synthesis_task_async(
                lambda progress_callback: lighten_blend_image.create_lighten_blend_image(
                    list(file_paths),
                    output_path,
                    progress_callback=progress_callback,
                ),
                output_path=output_path,
                item_label="比較明合成画像",
            )
            return

        # AI解析モードの場合
        print("DEBUG: AI解析モードに入りました")
        import detection_preview
        import bright_area_detector
        import cv2

        detector_func = bright_area_detector.detect_meteors_with_boxes if is_meteor_mode else bright_area_detector.detect_bright_areas_with_boxes
        print(f"DEBUG: detector_func = {detector_func.__name__}")
        analysis_cache_paths = {}
        cache_run_dir = [None]
        
        def start_synthesis_with_results(results):
            def synthesis_task(progress_callback):
                progress_callback("AI解析結果に基づく合成処理を開始します...")
                # Mask generation must follow the same expanded file order as synthesis.
                
                all_files = []
                image_ext, video_ext = lighten_blend_image.get_supported_extensions()
                for path in file_paths:
                    if os.path.isdir(path):
                        all_files.extend(lighten_blend_image.collect_files_from_folder(path))
                    elif os.path.isfile(path):
                        all_files.append(path)
                
                all_files.sort()  # 名前順で処理されると仮定
                
                # インデックス管理用
                synthesis_items = [
                    {'path': analysis_cache_paths.get(path, path), 'result_key': os.path.basename(path)}
                    for path in all_files
                ]
                synthesis_files = [item['path'] for item in synthesis_items]
                file_index = [0]
                
                def mask_generator(img):
                    if file_index[0] >= len(synthesis_items):
                        return None
                    
                    item = synthesis_items[file_index[0]]
                    filename = item['result_key']
                    file_index[0] += 1
                    
                    if filename in results:
                        boxes = results[filename]['boxes']
                        h, w = img.shape[:2]
                        if is_meteor_mode:
                            return bright_area_detector.create_inclusion_mask_from_boxes((h, w), boxes)
                        else:
                            return bright_area_detector.create_mask_from_boxes((h, w), boxes)
                    return None

                success = lighten_blend_image.create_lighten_blend_image(
                    synthesis_files,
                    output_path,
                    progress_callback=progress_callback,
                    mask_generator=mask_generator,
                    inclusion_mode=is_meteor_mode
                )
                return success

            self._run_synthesis_task_async(
                synthesis_task,
                output_path=output_path,
                item_label="比較明合成画像",
                cleanup=lambda: cache_run_dir[0] and shutil.rmtree(cache_run_dir[0], ignore_errors=True),
            )

        preview_window = detection_preview.DetectionPreviewWindow(self, start_synthesis_with_results)
        ai_config = {
            "backend": self.ai_vlm_backend_var.get(),
            "lm_studio_url": self.lm_studio_vlm_url_var.get(),
            "lm_studio_model_id": self.lm_studio_vlm_model_var.get(),
        }
        ui_events = queue.Queue()
        analysis_finished = {"value": False}

        def poll_analysis_events():
            """解析ワーカーの結果を、Tkメインスレッドでだけ表示する。"""
            try:
                while True:
                    event_type, *payload = ui_events.get_nowait()
                    if event_type == "log":
                        self.append_log(payload[0])
                    elif event_type == "start" and preview_window.winfo_exists():
                        preview_window.start_analysis(payload[0])
                    elif event_type == "item" and preview_window.winfo_exists():
                        path, filename, display_path, boxes = payload
                        analysis_cache_paths[path] = display_path
                        preview_window.add_item(
                            filename, display_path, boxes,
                            lambda image: detector_func(image) or (None, []),
                        )
                    elif event_type == "done":
                        analysis_finished["value"] = True
                        if preview_window.winfo_exists():
                            preview_window.finalize_analysis()
                            messagebox.showinfo(
                                "解析完了",
                                "全画像の解析が完了しました。\n未検出の画像が上部に表示されています。\n"
                                "プレビュー画面で結果を確認し、「修正を確定して合成を開始」ボタンを押してください。",
                                parent=self,
                            )
                    elif event_type == "error":
                        analysis_finished["value"] = True
                        self.append_log(f"AI解析中にエラーが発生しました: {payload[0]}")
                        if preview_window.winfo_exists():
                            preview_window.destroy()
                        messagebox.showerror("AI解析エラー", payload[0], parent=self)
            except queue.Empty:
                pass
            if not analysis_finished["value"] and preview_window.winfo_exists():
                self.after(50, poll_analysis_events)

        def run_analysis_task():
            try:
                ui_events.put(("log", "AIによる画像解析を開始します..."))
                if not self._ensure_ai_model_loaded(
                    bright_area_detector,
                    ai_config,
                    log_callback=lambda message: ui_events.put(("log", message)),
                ):
                    ui_events.put(("error", "AIモデルのロードまたは接続確認に失敗しました。"))
                    return

                all_files = []
                for path in file_paths:
                    if os.path.isdir(path):
                        all_files.extend(lighten_blend_image.collect_files_from_folder(path))
                    elif os.path.isfile(path):
                        all_files.append(path)
                all_files.sort()
                _, video_ext = lighten_blend_image.get_supported_extensions()
                cache_run_dir[0] = os.path.join(
                    config.LIGHTEN_BLEND_CACHE_DIR,
                    datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                )
                os.makedirs(cache_run_dir[0], exist_ok=True)

                # 動画はまず比較明合成画像へ変換し、その後のVLM判定を並列化する。
                prepared_items = []
                for index, path in enumerate(all_files):
                    filename = os.path.basename(path)
                    display_path = path
                    if Path(path).suffix.lower() in video_ext:
                        ui_events.put(("log", f"動画から比較明合成画像を作成中: {filename}"))
                        image = lighten_blend_video.create_composite_from_videos(
                            [path], progress_callback=None, sample_interval=1
                        )
                        if image is None:
                            ui_events.put(("log", f"警告: 合成画像を作成できませんでした: {filename}"))
                            continue
                        safe_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', os.path.splitext(filename)[0])
                        display_path = os.path.join(cache_run_dir[0], f"{index:04d}_{safe_name}_composite.png")
                        cv2.imwrite(display_path, image)
                    prepared_items.append((path, filename, display_path))

                if not prepared_items:
                    ui_events.put(("error", "解析できる画像または動画がありません。"))
                    return

                uses_local_model = bool(getattr(bright_area_detector, "uses_local_model", lambda: True)())
                max_workers = 1 if uses_local_model else min(
                    config.AI_VLM_MAX_PARALLEL_REQUESTS, len(prepared_items)
                )
                ui_events.put(("log", f"VLM解析を並列数 {max_workers} で開始します。"))
                ui_events.put(("start", len(prepared_items)))

                def analyze_item(item):
                    path, filename, display_path = item
                    image = imread_with_japanese_path(display_path)
                    if image is None:
                        raise RuntimeError(f"画像を読み込めませんでした: {filename}")
                    result = detector_func(image)
                    boxes = result[1] if result else []
                    return path, filename, display_path, boxes

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(analyze_item, item) for item in prepared_items]
                    for future in as_completed(futures):
                        try:
                            ui_events.put(("item", *future.result()))
                        except Exception as exc:
                            ui_events.put(("log", f"警告: 個別解析に失敗しました: {exc}"))

                ui_events.put(("log", "全画像の解析が完了しました。検出結果を確認・修正してください。"))
                ui_events.put(("done",))
            except Exception as exc:
                import traceback
                traceback.print_exc()
                ui_events.put(("error", str(exc)))

        self.after(50, poll_analysis_events)
        threading.Thread(target=run_analysis_task, daemon=True).start()

    def _handle_synthesis_result(self, success, output_path):
        if success:
            messagebox.showinfo("完了", f"比較明合成画像の作成が完了しました。\n保存先: {output_path}")
            self.append_log(f"比較明合成画像の作成が完了しました: {output_path}")
        else:
            messagebox.showerror("エラー", "比較明合成画像の作成に失敗しました。ログを確認してください。")
            self.append_log("比較明合成画像の作成に失敗しました。")

    def _ensure_ai_model_loaded(self, detector_module, ai_config=None, log_callback=None) -> bool:
        """AI合成前に内部LLMのロード完了を保証する。"""
        log = log_callback or self.append_log
        log("AIモデルのロードを確認中...")
        if hasattr(detector_module, "configure_ai_backend"):
            selected_config = ai_config or {
                "backend": self.ai_vlm_backend_var.get(),
                "lm_studio_url": self.lm_studio_vlm_url_var.get(),
                "lm_studio_model_id": self.lm_studio_vlm_model_var.get(),
            }
            detector_module.configure_ai_backend(
                backend=selected_config["backend"],
                lm_studio_url=selected_config["lm_studio_url"],
                lm_studio_model_id=selected_config["lm_studio_model_id"],
                lm_studio_api_key="",
            )
            if hasattr(detector_module, "get_active_model_name"):
                log(f"使用AIモデル: {detector_module.get_active_model_name()}")

        uses_local_model = True
        uses_local_model_fn = getattr(detector_module, "uses_local_model", None)
        if callable(uses_local_model_fn):
            uses_local_model = bool(uses_local_model_fn())

        if not uses_local_model:
            try:
                load_result = detector_module.load_selected_ai_model(status_callback=log)
                log(f"AIモデルを準備しました: {load_result}")
            except Exception as exc:
                log(f"LM Studioモデルのロードに失敗しました: {exc}")
                return False
            connected, err = detector_module.check_vlm_connection(status_callback=log, force=True)
            if connected:
                log("AIモデルの接続確認が完了しました。")
                return True

            error_message = f"AIモデルの接続確認に失敗しました: {err}"
            log(error_message)
            return False

        local_model_dir = getattr(detector_module, "LOCAL_MODEL_DIR", "./quantized_model")
        has_local_model = False
        try:
            has_model_fn = getattr(detector_module, "has_quantized_model", None)
            if callable(has_model_fn):
                has_local_model = bool(has_model_fn())
            else:
                has_local_model = os.path.isdir(local_model_dir)
        except Exception as e:
            log(f"ローカルモデル状態の確認中にエラー: {e}")
            has_local_model = os.path.isdir(local_model_dir)

        if not has_local_model:
            log(f"ローカルLLMモデルが見つかりません: {local_model_dir}")
            # A worker must never open a Tk dialog.  Synthesis calls this method
            # from a worker, so let the caller show a normal error instead.
            if threading.current_thread() is not threading.main_thread():
                log("ローカルモデルを先に「Load Model」で読み込んでから再実行してください。")
                return False

            log("必要なディスク容量を見積もり中...")
            req = self._estimate_llm_storage_requirements(detector_module)

            info_lines = [
                "ローカルLLMモデルが見つかりませんでした。",
                f"対象モデル: {req['repo_id']}",
                f"一時的に必要な空き容量 (目安): {self._format_size_bytes(req['temporary_bytes'])}",
                f"最終的に必要な容量 (目安): {self._format_size_bytes(req['final_bytes'])}",
                f"現在の空き容量: {self._format_size_bytes(req['free_bytes'])}",
                "",
                "モデルをダウンロードしますか？",
            ]
            if not req["fetched_metadata"]:
                info_lines.insert(4, "※ 容量は取得失敗のため既定値での目安です。")
            if req["free_bytes"] < req["temporary_bytes"]:
                info_lines.insert(5, "※ 警告: 空き容量が一時必要容量を下回っています。")

            should_download = bool(
                self._run_on_main_thread(
                    lambda msg="\n".join(info_lines): messagebox.askyesno("モデルダウンロード確認", msg, parent=self)
                )
            )

            if not should_download:
                log("ユーザーがモデルダウンロードをキャンセルしました。")
                return False

            log("モデルダウンロードを開始します。")
            mirror_stream = _StderrProgressStream(log, passthrough=sys.stderr)
            with contextlib.redirect_stderr(mirror_stream), contextlib.redirect_stdout(mirror_stream):
                connected, err = detector_module.check_vlm_connection(status_callback=log)
            mirror_stream.flush()
        else:
            connected, err = detector_module.check_vlm_connection(status_callback=log)

        if connected:
            log("AIモデルのロードが完了しました。")
            return True

        error_message = f"AIモデルのロードに失敗しました: {err}"
        log(error_message)
        return False

    def create_timelapse_callback(self):
        """タイムラプス作成ボタンのコールバック。ドラッグ＆ドロップウィンドウを表示する。"""
        TimelapseDragDropWindow(self, self.append_log)
