"""Tk review window for pending machine-learning training events."""

from __future__ import annotations

import json
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
    def __init__(self, parent: tk.Misc, root_dir: str):
        super().__init__(parent)
        self.root_dir = str(Path(root_dir).expanduser())
        self.title("機械学習データ目視レビュー")
        self.geometry("1320x760")
        self.minsize(1000, 650)
        self.events: List[Path] = []
        self.index = 0
        self.undo_stack: List[Dict[str, Any]] = ml_training_data.undoable_reviews(root_dir)
        self.skip_stack: List[Dict[str, Any]] = ml_training_data.undoable_skips(root_dir)
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
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.bind("<Left>", lambda _e: self._previous())
        self.bind("<Right>", lambda _e: self._next())
        self.bind("<Key-m>", lambda _e: self._classify("meteor"))
        self.bind("<Key-n>", lambda _e: self._classify("not_meteor"))
        self.bind("<Key-s>", lambda _e: self._skip_current())
        self.bind("<BackSpace>", lambda _e: self._undo())
        self.refresh()

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
        ttk.Button(toolbar, text="再読み込み", command=self.refresh).pack(side=tk.RIGHT)

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

    def refresh(self) -> None:
        current_name = self.current_event.name if self.current_event else None
        self.events = ml_training_data.pending_events(self.root_dir)
        if current_name:
            for i, event in enumerate(self.events):
                if event.name == current_name:
                    self.index = i
                    break
            else:
                self.index = min(self.index, max(0, len(self.events) - 1))
        else:
            self.index = min(self.index, max(0, len(self.events) - 1))
        self.skip_stack = ml_training_data.undoable_skips(self.root_dir)
        self.skip_status_var.set(f"スキップ中: {len(self.skip_stack)}件")
        self._show_current()

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
        except Exception as exc:
            self._highlight_predicted_confirmation(None)
            self.detail_var.set(f"メタデータを読み込めません: {exc}")
            return
        predicted_label = meta.get("predicted_label")
        self._highlight_predicted_confirmation(predicted_label)
        self.status_var.set(f"未確認 {self.index + 1} / {len(self.events)}")
        self.detail_var.set(
            f"予測: {predicted_label}　流星確率: {meta.get('meteor_probability', 0):.3f}\n"
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
        self._play_next_frame()

    def _highlight_predicted_confirmation(self, predicted_label: object) -> None:
        meteor_style, not_meteor_style = confirmation_button_styles(predicted_label)
        self.meteor_confirm_button.configure(style=meteor_style)
        self.not_meteor_confirm_button.configure(style=not_meteor_style)

    def _play_next_frame(self) -> None:
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
        photo = ImageTk.PhotoImage(image)
        self._video_photo = photo
        self.video_label.configure(image=photo, text="")
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 15.0
        self.video_after_id = self.after(max(30, int(1000 / fps)), self._play_next_frame)

    def _stop_video(self) -> None:
        if self.video_after_id is not None:
            try:
                self.after_cancel(self.video_after_id)
            except Exception:
                pass
            self.video_after_id = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _previous(self) -> None:
        if self.events:
            self.index = (self.index - 1) % len(self.events)
            self._show_current()

    def _next(self) -> None:
        if self.events:
            self.index = (self.index + 1) % len(self.events)
            self._show_current()

    def _classify(self, label: str) -> None:
        event = self.current_event
        if event is None:
            return
        self._stop_video()
        try:
            record = ml_training_data.review_event(event, self.root_dir, label)
            self.undo_stack.append(record)
            self.events.pop(self.index)
            self.index = min(self.index, max(0, len(self.events) - 1))
            self._show_current()
        except Exception as exc:
            messagebox.showerror("レビュー保存エラー", str(exc), parent=self)
            self.refresh()

    def _skip_events(self, events: List[Path], reason: str) -> None:
        if not events:
            return
        self._stop_video()
        skipped = 0
        for event in events:
            try:
                record = ml_training_data.skip_event(event, self.root_dir, reason)
                self.skip_stack.append(record)
                skipped += 1
            except Exception as exc:
                messagebox.showerror("スキップ保存エラー", str(exc), parent=self)
                break
        self.events = [event for event in self.events if event.exists()]
        self.index = min(self.index, max(0, len(self.events) - 1))
        self.skip_status_var.set(f"スキップ中: {len(self.skip_stack)}件（今回 {skipped}件）")
        self._show_current()

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
        target_date = ml_training_data.detection_date(event)
        if not target_date:
            messagebox.showwarning("日付不明", "この検出の日付を読み取れません。", parent=self)
            return
        selected = [item for item in self.events if ml_training_data.detection_date(item) == target_date]
        if not selected:
            return
        if not messagebox.askyesno(
            "同一日をスキップ",
            f"{target_date} の未確認 {len(selected)}件をスキップしますか？\n"
            "後から「スキップを戻す」で復元できます。",
            parent=self,
        ):
            return
        self._skip_events(selected, f"same_date:{target_date}")

    def _undo_skip(self) -> None:
        try:
            count = max(1, int(self.undo_skip_count_var.get()))
        except (TypeError, ValueError, tk.TclError):
            count = 1
        restored = 0
        while restored < count and self.skip_stack:
            record = self.skip_stack.pop()
            try:
                if ml_training_data.undo_skip(record, self.root_dir) is not None:
                    restored += 1
            except Exception as exc:
                messagebox.showerror("スキップを戻せません", str(exc), parent=self)
                break
        if restored == 0:
            messagebox.showinfo("スキップを戻す", "戻せるスキップはありません。", parent=self)
        self.refresh()

    def _undo(self) -> None:
        try:
            count = max(1, int(self.undo_count_var.get()))
        except (TypeError, ValueError):
            count = 1
        restored = 0
        while restored < count and self.undo_stack:
            record = self.undo_stack.pop()
            try:
                if ml_training_data.undo_review(record, self.root_dir) is not None:
                    restored += 1
            except Exception as exc:
                messagebox.showerror("判定を戻せません", str(exc), parent=self)
                break
        if restored == 0:
            messagebox.showinfo(
                "判定を戻す", "戻せる確定済み判定はありません。", parent=self
            )
        self.refresh()

    def _close(self) -> None:
        self._stop_video()
        self.destroy()


def open_review_window(parent: tk.Misc, root_dir: str) -> TrainingDataReviewWindow:
    return TrainingDataReviewWindow(parent, root_dir)
