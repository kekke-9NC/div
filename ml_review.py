"""Tk review window for pending machine-learning training events."""

from __future__ import annotations

import json
import math
import queue
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk

import ml_training_data


DEFAULT_CONFIRM_STYLE = "TButton"
PREDICTED_CONFIRM_STYLE = "PredictedConfirm.TButton"


def confirmation_button_styles(predicted_label: object) -> tuple[str, str]:
    """Return the meteor and not_meteor button styles for a prediction."""
    label = str(predicted_label).strip().lower()
    return (
        PREDICTED_CONFIRM_STYLE if label == "meteor" else DEFAULT_CONFIRM_STYLE,
        PREDICTED_CONFIRM_STYLE if label == "not_meteor" else DEFAULT_CONFIRM_STYLE,
    )


class TrainingDataReviewWindow(tk.Toplevel):
    def __init__(
        self,
        parent: tk.Misc,
        root_dir: str,
        media_directories: Optional[Dict[str, str]] = None,
    ):
        super().__init__(parent)
        self.root_dir = str(Path(root_dir).expanduser())
        self.media_directories = dict(media_directories or {})
        self.title("機械学習データ目視レビュー")
        self.geometry("1320x760")
        self.minsize(1000, 650)
        self.events: List[Path] = []
        self.index = 0
        self.undo_stack: List[Dict[str, Any]] = []
        self.skip_stack: List[Dict[str, Any]] = []
        self._load_busy = False
        self._load_failed = False
        self._closing = False
        self._closed = False
        self._load_cancel = threading.Event()
        self._load_results = queue.Queue()
        self._load_after_id = None
        self._pending_media_directories = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.video_after_id = None
        self.video_frame_index = 0
        self._photo_refs: List[ImageTk.PhotoImage] = []
        self.status_var = tk.StringVar()
        self.detail_var = tk.StringVar()
        self.undo_count_var = tk.IntVar(value=1)
        self.skip_count_var = tk.IntVar(value=10)
        self.undo_skip_count_var = tk.IntVar(value=1)
        self.skip_status_var = tk.StringVar()
        self.bulk_dialog = None
        self._build_ui()
        self._review_controls = []
        def collect_controls(widget):
            for child in widget.winfo_children():
                if isinstance(child, (ttk.Button, ttk.Spinbox)):
                    self._review_controls.append(child)
                collect_controls(child)
        collect_controls(self)
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.bind("<Destroy>", self._on_destroy, add="+")
        self.bind("<Left>", lambda _e: self._previous())
        self.bind("<Right>", lambda _e: self._next())
        self.bind("<Key-m>", lambda _e: self._classify("meteor"))
        self.bind("<Key-n>", lambda _e: self._classify("not_meteor"))
        self.bind("<Key-s>", lambda _e: self._skip_current())
        self.bind("<BackSpace>", lambda _e: self._undo())
        self.refresh(synchronize_media=True)

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        style.configure(
            PREDICTED_CONFIRM_STYLE,
            background="#FFD54F",
            foreground="#17130A",
            bordercolor="#F9A825",
            lightcolor="#FFD54F",
            darkcolor="#F9A825",
        )
        style.map(
            PREDICTED_CONFIRM_STYLE,
            background=[("pressed", "#F9A825"), ("active", "#FFE082")],
            foreground=[("pressed", "#17130A"), ("active", "#17130A")],
        )

        toolbar = ttk.Frame(self, padding=8)
        toolbar.pack(fill=tk.X)
        ttk.Label(toolbar, textvariable=self.status_var).pack(side=tk.LEFT)
        ttk.Label(toolbar, textvariable=self.skip_status_var).pack(side=tk.LEFT, padx=(18, 0))
        ttk.Button(toolbar, text="流星確率で一括選別", command=self._open_bulk_review).pack(side=tk.RIGHT, padx=8)
        self.refresh_button = ttk.Button(toolbar, text="再読み込み", command=self.refresh)
        self.refresh_button.pack(side=tk.RIGHT)

        ttk.Label(
            self, textvariable=self.detail_var, padding=(10, 0, 10, 6), justify=tk.LEFT
        ).pack(fill=tk.X)

        preview = ttk.Frame(self, padding=(8, 0))
        preview.pack(fill=tk.BOTH, expand=True)
        diff_box = ttk.LabelFrame(preview, text="通常差分")
        diff_box.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        temporal_box = ttk.LabelFrame(preview, text="時間の向き（赤→緑→青）")
        temporal_box.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)
        video_box = ttk.LabelFrame(preview, text="モノクロ切り出し動画")
        video_box.grid(row=0, column=2, sticky="nsew", padx=4, pady=4)
        self.diff_label = ttk.Label(diff_box, anchor=tk.CENTER)
        self.diff_label.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.temporal_label = ttk.Label(temporal_box, anchor=tk.CENTER)
        self.temporal_label.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.video_label = ttk.Label(video_box, anchor=tk.CENTER)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        for column in range(3):
            preview.columnconfigure(column, weight=1, uniform="preview")
        preview.rowconfigure(0, weight=1)

        controls = ttk.Frame(self, padding=(10, 6, 10, 4))
        controls.pack(fill=tk.X)
        navigation = ttk.Frame(controls)
        navigation.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(navigation, text="← 前の未確認", command=self._previous).pack(side=tk.LEFT)
        ttk.Button(navigation, text="次の未確認 →", command=self._next).pack(side=tk.LEFT, padx=5)
        self.meteor_confirm_button = ttk.Button(
            navigation, text="流星として確定  [M]", command=lambda: self._classify("meteor")
        )
        self.meteor_confirm_button.pack(side=tk.LEFT, padx=(25, 5), expand=True, fill=tk.X)
        self.not_meteor_confirm_button = ttk.Button(
            navigation, text="非流星として確定  [N]", command=lambda: self._classify("not_meteor")
        )
        self.not_meteor_confirm_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        skip_controls = ttk.LabelFrame(controls, text="今は学習に使わない（次回から非表示・後で復元可）", padding=5)
        skip_controls.pack(fill=tk.X)
        ttk.Button(skip_controls, text="この1件をスキップ [S]", command=self._skip_current).pack(side=tk.LEFT)
        ttk.Label(skip_controls, text="ここから").pack(side=tk.LEFT, padx=(12, 3))
        ttk.Spinbox(skip_controls, from_=1, to=10000, width=6, textvariable=self.skip_count_var).pack(side=tk.LEFT)
        ttk.Button(skip_controls, text="件をスキップ", command=self._skip_from_here).pack(side=tk.LEFT, padx=(3, 8))
        ttk.Button(skip_controls, text="この日の未確認を全てスキップ", command=self._skip_same_day).pack(side=tk.LEFT)
        ttk.Label(skip_controls, text="復元:").pack(side=tk.LEFT, padx=(18, 3))
        ttk.Spinbox(skip_controls, from_=1, to=10000, width=5, textvariable=self.undo_skip_count_var).pack(side=tk.LEFT)
        ttk.Button(skip_controls, text="スキップを戻す", command=self._undo_skip).pack(side=tk.LEFT, padx=(3, 0))

        undo_controls = ttk.Frame(controls)
        undo_controls.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(undo_controls, text="確定判定を戻す件数:").pack(side=tk.LEFT)
        ttk.Spinbox(undo_controls, from_=1, to=100, width=4, textvariable=self.undo_count_var).pack(side=tk.LEFT)
        ttk.Button(undo_controls, text="判定を戻す [Backspace]", command=self._undo).pack(side=tk.LEFT, padx=5)

        ttk.Label(
            self,
            text="M/N: 確定　S: 1件スキップ　←/→: 移動　Backspace: 確定判定を戻す",
            padding=(10, 0, 10, 6),
        ).pack(fill=tk.X)

    def refresh(self, *, synchronize_media=False, include_history=True) -> None:
        if self._load_busy or self._closing or self._closed:
            return
        if self.bulk_dialog is not None and self.bulk_dialog.winfo_exists() and self.bulk_dialog.busy:
            return
        current_name = self.current_event.name if self.current_event else None
        self._stop_video()
        self._load_busy = True
        self._load_cancel.clear()
        self.status_var.set("未確認データを読み込み中…")
        self.detail_var.set("一覧と判定履歴を読み込んでいます。")
        for control in self._review_controls:
            control.state(["disabled"])
        root_dir, directories = self.root_dir, dict(self.media_directories)
        results, cancel = self._load_results, self._load_cancel

        def worker():
            moved_count, sync_error = 0, None
            try:
                Path(root_dir).mkdir(parents=True, exist_ok=True)
                if synchronize_media and not cancel.is_set():
                    try:
                        moved_count = ml_training_data.sync_reviewed_media(
                            root_dir, directories, cancel=cancel)
                    except Exception as exc:
                        sync_error = str(exc)
                if cancel.is_set():
                    raise InterruptedError()
                events = ml_training_data.pending_events(root_dir, cancel=cancel)
                reviews, skips = (ml_training_data.undoable_history(root_dir, cancel=cancel, strict=True)
                                  if include_history else (None, None))
                if cancel.is_set():
                    raise InterruptedError()
                results.put(("ready", (events, reviews, skips, current_name, moved_count, sync_error)))
            except InterruptedError:
                results.put(("cancelled", None))
            except Exception as exc:
                results.put(("error", str(exc)))

        # Non-daemon: finish/roll back the current media move if the app closes.
        # Workers only communicate through a queue; even Tk.after stays on Tk.
        threading.Thread(target=worker, name="MLReviewLoad", daemon=False).start()
        self._load_after_id = self.after(50, self._poll_load)

    def _poll_load(self) -> None:
        self._load_after_id = None
        if self._closed:
            return
        try:
            kind, value = self._load_results.get_nowait()
        except queue.Empty:
            self._load_after_id = self.after(50, self._poll_load)
            return
        self._load_busy = False
        if self._closing:
            self.destroy()
            return
        if kind == "action":
            self._finish_action(*value)
            return
        for control in self._review_controls:
            control.state(["!disabled"])
        if self._pending_media_directories is not None:
            self.media_directories = self._pending_media_directories
            self._pending_media_directories = None
            self.refresh(synchronize_media=True)
            return
        if kind != "ready":
            self._load_failed = True
            for control in self._review_controls:
                control.state(["disabled"])
            self.refresh_button.state(["!disabled"])
            self.status_var.set("読み込みに失敗しました。再読み込みしてください。")
            self._notify_bulk_loaded()
            if kind == "error":
                messagebox.showerror("レビュー読み込みエラー", value, parent=self)
            return
        self._load_failed = False
        self.events, reviews, skips, current_name, moved_count, sync_error = value
        if reviews is not None:
            self.undo_stack, self.skip_stack = reviews, skips
        if current_name:
            for i, event in enumerate(self.events):
                if event.name == current_name:
                    self.index = i
                    break
            else:
                self.index = min(self.index, max(0, len(self.events) - 1))
        else:
            self.index = min(self.index, max(0, len(self.events) - 1))
        self.skip_status_var.set(f"スキップ中: {len(self.skip_stack)}件")
        self._show_current()
        self._notify_bulk_loaded()
        if sync_error:
            messagebox.showerror("検出メディアの分類更新エラー", sync_error, parent=self)
        elif moved_count:
            messagebox.showinfo(
                "検出メディアの分類更新",
                f"既に確定済みの分類に合わせて、動画・画像 {moved_count}件を保存フォルダ間で移動しました。",
                parent=self,
            )

    @staticmethod
    def _record_key(record):
        return record.get("action"), record.get("timestamp"), record.get("to")

    def _update_history_records(self, operation, records):
        keys = {self._record_key(record) for record in records}
        if operation in ("undo", "undo_review", "undo_skip"):
            self.undo_stack = [record for record in self.undo_stack if self._record_key(record) not in keys]
            self.skip_stack = [record for record in self.skip_stack if self._record_key(record) not in keys]
        else:
            existing = {self._record_key(record) for record in self.undo_stack + self.skip_stack}
            for record in records:
                if self._record_key(record) not in existing:
                    (self.undo_stack if record["action"] == "review" else self.skip_stack).append(record)
                    existing.add(self._record_key(record))

    def apply_batch_result(self, operation, result):
        self._update_history_records(operation, result.records)
        # The transaction already returned its exact committed records. Only
        # rescan the small pending inventory, not the full historical log.
        self.refresh(include_history=bool(result.changed or result.error))

    def _start_action(self, operation, function):
        if self._load_busy or self._closing or self._load_failed:
            return
        if self.bulk_dialog is not None and self.bulk_dialog.winfo_exists() and self.bulk_dialog.busy:
            return
        self._stop_video()
        self._load_busy = True
        self._load_cancel.clear()
        self.status_var.set("保存しています…")
        for control in self._review_controls:
            control.state(["disabled"])
        results, cancel = self._load_results, self._load_cancel
        def worker():
            try:
                result = function(cancel)
            except Exception as exc:
                result = dict(records=[], error=str(exc))
            results.put(("action", (operation, result)))
        threading.Thread(target=worker, name="MLReviewSave", daemon=False).start()
        self._load_after_id = self.after(50, self._poll_load)

    def _finish_action(self, operation, result):
        if operation == "prepare_skip_day":
            for control in self._review_controls:
                control.state(["!disabled"])
            selected = result.get("selected", [])
            if result.get("error"):
                messagebox.showerror("日付検索エラー", result["error"], parent=self)
            elif selected and messagebox.askyesno(
                "同一日をスキップ",
                f"{result['day']} の未確認 {len(selected)}件をスキップしますか？\n"
                "後から「スキップを戻す」で復元できます。", parent=self,
            ):
                self._skip_events(selected, f"same_date:{result['day']}")
                return
            self._show_current()
            return
        self._update_history_records(operation, result.get("records", []))
        if result.get("error"):
            messagebox.showerror("レビュー保存エラー", result["error"], parent=self)
        elif operation.startswith("undo_") and not result.get("restored"):
            messagebox.showinfo("判定を戻す", "戻せる判定はありません。", parent=self)
        if operation in ("review", "skip") and not result.get("error"):
            moved = {Path(record["from"]) for record in result.get("records", [])}
            self.events = [event for event in self.events if event not in moved]
            self.index = min(self.index, max(0, len(self.events) - 1))
            self.skip_status_var.set(f"スキップ中: {len(self.skip_stack)}件")
            if self._pending_media_directories is not None:
                self.media_directories = self._pending_media_directories
                self._pending_media_directories = None
                self.refresh(synchronize_media=True)
                return
            for control in self._review_controls:
                control.state(["!disabled"])
            self._show_current()
            self._notify_bulk_loaded()
            return
        self.refresh(include_history=bool(result.get("error")))

    def _notify_bulk_loaded(self) -> None:
        if self.bulk_dialog is not None and self.bulk_dialog.winfo_exists():
            self.bulk_dialog.reviewer_loaded()

    def _on_destroy(self, event) -> None:
        if event.widget is not self:
            return
        self._closed = True
        self._load_cancel.set()
        if self._load_after_id is not None:
            try:
                self.after_cancel(self._load_after_id)
            except tk.TclError:
                pass
            self._load_after_id = None
        self._stop_video()

    def _open_bulk_review(self) -> None:
        if self._load_busy or self._closing or self._load_failed:
            return
        if self.bulk_dialog is not None and self.bulk_dialog.winfo_exists():
            self.bulk_dialog.lift()
            return
        from ml_bulk_review_ui import BulkReviewDialog
        self.bulk_dialog = BulkReviewDialog(self)
        self.bulk_dialog.grab_set()

    @property
    def current_event(self) -> Optional[Path]:
        if not self.events:
            return None
        return self.events[self.index]

    def _fit_photo(self, path: Path, max_size=(400, 420)) -> Optional[ImageTk.PhotoImage]:
        try:
            image = Image.open(path).convert("RGB")
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(image)
        except Exception:
            return None

    def _show_current(self) -> None:
        self._stop_video()
        event = self.current_event
        if event is None:
            self._highlight_predicted_confirmation(None)
            self.status_var.set("未確認データはありません")
            self.detail_var.set("すべて目視確認済みです。")
            for label in (self.diff_label, self.temporal_label, self.video_label):
                label.configure(image="", text="")
            self._photo_refs = []
            return

        try:
            meta = ml_training_data.load_metadata(event)
            if not isinstance(meta, dict):
                raise ValueError("メタデータの形式が不正です。")
        except Exception as exc:
            self._highlight_predicted_confirmation(None)
            self.detail_var.set(f"メタデータを読み込めません: {exc}")
            return
        predicted_label = meta.get("predicted_label")
        self._highlight_predicted_confirmation(predicted_label)
        self.status_var.set(f"未確認 {self.index + 1} / {len(self.events)}")
        try:
            raw_probability = meta["meteor_probability"]
            if isinstance(raw_probability, bool):
                raise ValueError("Invalid probability")
            probability = float(raw_probability)
            probability_text = f"{probability:.3f}" if math.isfinite(probability) and 0 <= probability <= 1 else "不明"
        except (KeyError, TypeError, ValueError, OverflowError):
            probability_text = "不明"
        self.detail_var.set(
            f"予測: {predicted_label}　流星確率: {probability_text}\n"
            f"日時: {meta.get('detection_time', '')}　元動画: {meta.get('source', '')}"
        )

        diff = self._fit_photo(event / "diff.png")
        temporal = self._fit_photo(event / "temporal_rgb.png")
        self._photo_refs = [p for p in (diff, temporal) if p is not None]
        self.diff_label.configure(image=diff or "", text="通常差分" if diff is None else "")
        self.temporal_label.configure(image=temporal or "", text="時間画像" if temporal is None else "")
        thread_property = getattr(cv2, "CAP_PROP_N_THREADS", None)
        if thread_property is not None:
            try:
                self.cap = cv2.VideoCapture(
                    str(event / "clip.mp4"), cv2.CAP_FFMPEG,
                    [int(thread_property), 1],
                )
            except (cv2.error, TypeError, ValueError):
                self.cap = cv2.VideoCapture(str(event / "clip.mp4"))
        else:
            self.cap = cv2.VideoCapture(str(event / "clip.mp4"))
        self.video_frame_index = 0
        self._video_photo_size = None
        self._play_next_frame()

    def _highlight_predicted_confirmation(self, predicted_label: object) -> None:
        meteor_style, not_meteor_style = confirmation_button_styles(predicted_label)
        self.meteor_confirm_button.configure(style=meteor_style)
        self.not_meteor_confirm_button.configure(style=not_meteor_style)

    def _play_next_frame(self) -> None:
        self.video_after_id = None
        if self.cap is None or not self.cap.isOpened():
            self.video_label.configure(text="動画なし", image="")
            return
        ok, frame = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        if not ok:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((420, 420), Image.Resampling.LANCZOS)
        if getattr(self, "_video_photo_size", None) == image.size:
            # Reuse the displayed Tk image; avoid widget relayout and image
            # allocation/destruction on every frame (costly on macOS/Tk).
            self._video_photo.paste(image)
        else:
            self._video_photo = ImageTk.PhotoImage(image)
            self._video_photo_size = image.size
            self.video_label.configure(image=self._video_photo, text="")
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 15.0
        self.video_after_id = self.after(max(30, int(1000 / fps)), self._play_next_frame)

    def _pause_video(self) -> None:
        if self.video_after_id is not None:
            try:
                self.after_cancel(self.video_after_id)
            except Exception:
                pass
            self.video_after_id = None

    def _resume_video(self) -> None:
        if self.video_after_id is None and self.cap is not None:
            self._play_next_frame()

    def _stop_video(self) -> None:
        self._pause_video()
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _previous(self) -> None:
        if self._load_busy or self._closing or self._load_failed:
            return
        if self.events:
            self.index = (self.index - 1) % len(self.events)
            self._show_current()

    def _next(self) -> None:
        if self._load_busy or self._closing or self._load_failed:
            return
        if self.events:
            self.index = (self.index + 1) % len(self.events)
            self._show_current()

    def _classify(self, label: str) -> None:
        if self._load_busy or self._closing or self._load_failed:
            return
        event = self.current_event
        if event is None:
            return
        root, directories = self.root_dir, dict(self.media_directories)
        def classify(_cancel):
            return dict(records=[ml_training_data.review_event(event, root, label, directories)])
        self._start_action("review", classify)

    def _skip_events(self, events: List[Path], reason: str) -> None:
        if not events:
            return
        selected, root = list(events), self.root_dir
        def skip(cancel):
            result = dict(records=[], error="")
            for event in selected:
                if cancel.is_set():
                    break
                try:
                    result["records"].append(ml_training_data.skip_event(event, root, reason))
                except Exception as exc:
                    result["error"] = str(exc)
                    break
            return result
        self._start_action("skip", skip)

    def _skip_current(self) -> None:
        event = self.current_event
        if event is not None:
            self._skip_events([event], "single")

    def _skip_from_here(self) -> None:
        if not self.events:
            return
        try:
            count = max(1, int(self.skip_count_var.get()))
        except (TypeError, ValueError, tk.TclError):
            count = 1
        selected = self.events[self.index:self.index + count]
        self._skip_events(selected, f"from_here:{len(selected)}")

    def _skip_same_day(self) -> None:
        event = self.current_event
        if event is None:
            return
        events = list(self.events)
        def select_day(cancel):
            day = ml_training_data.detection_date(event)
            if not day:
                raise ValueError("この検出の日付を読み取れません。")
            selected = []
            for item in events:
                if cancel.is_set():
                    break
                if ml_training_data.detection_date(item) == day:
                    selected.append(item)
            return dict(selected=selected, day=day)
        self._start_action("prepare_skip_day", select_day)

    def _start_undo(self, operation, stack, count):
        records, root = list(reversed(stack)), self.root_dir
        undo = ml_training_data.undo_skip if operation == "undo_skip" else ml_training_data.undo_review
        def restore(cancel):
            result = dict(records=[], restored=0, error="")
            for record in records:
                if cancel.is_set() or result["restored"] >= count:
                    break
                try:
                    restored = undo(record, root)
                    # A destination conflict does not undo an active review.
                    # Keep it in the local stack just as a history reload would.
                    if restored is not None or not Path(record.get("to", "")).exists():
                        result["records"].append(record)
                    result["restored"] += restored is not None
                except Exception as exc:
                    result["error"] = str(exc)
                    break
            return result
        self._start_action(operation, restore)

    def _undo_skip(self) -> None:
        try:
            count = max(1, int(self.undo_skip_count_var.get()))
        except (TypeError, ValueError, tk.TclError):
            count = 1
        self._start_undo("undo_skip", self.skip_stack, count)

    def _undo(self) -> None:
        try:
            count = max(1, int(self.undo_count_var.get()))
        except (TypeError, ValueError, tk.TclError):
            count = 1
        self._start_undo("undo_review", self.undo_stack, count)

    def _close(self) -> None:
        if self.bulk_dialog is not None and self.bulk_dialog.winfo_exists():
            if self.bulk_dialog.busy:
                self.bulk_dialog._close()
                self.after(100, self._close)
                return
            self.bulk_dialog.destroy()
        self._stop_video()
        if self._load_busy:
            self._closing = True
            self._load_cancel.set()
            self.status_var.set("閉じています…")
            return
        self.destroy()


def open_review_window(
    parent: tk.Misc,
    root_dir: str,
    media_directories: Optional[Dict[str, str]] = None,
) -> TrainingDataReviewWindow:
    existing = getattr(parent, "_ml_training_review_window", None)
    if existing is not None and existing.winfo_exists():
        requested_root = str(Path(root_dir).expanduser().absolute())
        if existing._closing or str(Path(existing.root_dir).absolute()) != requested_root:
            existing._close()
            if existing.winfo_exists():
                # Let the old window finish its current media transaction first.
                parent._ml_training_review_request = (root_dir, dict(media_directories or {}))
                if getattr(parent, "_ml_training_review_reopen_after", None) is None:
                    def reopen():
                        parent._ml_training_review_reopen_after = None
                        request = parent._ml_training_review_request
                        open_review_window(parent, *request)
                    parent._ml_training_review_reopen_after = parent.after(50, reopen)
                return existing
        else:
            _cancel_pending_reopen(parent)
            existing.deiconify()
            existing.lift()
            directories = dict(media_directories or {})
            if existing._load_busy:
                if directories != existing.media_directories:
                    existing._pending_media_directories = directories
            elif directories != existing.media_directories or existing._load_failed:
                existing.media_directories = directories
                existing.refresh(synchronize_media=True)
            return existing
    _cancel_pending_reopen(parent)
    window = TrainingDataReviewWindow(parent, root_dir, media_directories)
    parent._ml_training_review_window = window
    return window


def _cancel_pending_reopen(parent):
    after_id = getattr(parent, "_ml_training_review_reopen_after", None)
    if after_id is not None:
        parent.after_cancel(after_id)
        parent._ml_training_review_reopen_after = None
