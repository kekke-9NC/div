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


class TrainingDataReviewWindow(tk.Toplevel):
    def __init__(self, parent: tk.Misc, root_dir: str):
        super().__init__(parent)
        self.root_dir = str(Path(root_dir).expanduser())
        self.title("機械学習データ目視レビュー")
        self.geometry("1120x820")
        self.minsize(820, 620)
        self.events: List[Path] = []
        self.index = 0
        self.undo_stack: List[Dict[str, Any]] = ml_training_data.undoable_reviews(root_dir)
        self.cap: Optional[cv2.VideoCapture] = None
        self.video_after_id = None
        self.video_frame_index = 0
        self._photo_refs: List[ImageTk.PhotoImage] = []
        self.status_var = tk.StringVar()
        self.detail_var = tk.StringVar()
        self.undo_count_var = tk.IntVar(value=1)
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.bind("<Left>", lambda _e: self._previous())
        self.bind("<Right>", lambda _e: self._next())
        self.bind("<Key-m>", lambda _e: self._classify("meteor"))
        self.bind("<Key-n>", lambda _e: self._classify("not_meteor"))
        self.bind("<BackSpace>", lambda _e: self._undo())
        self.refresh()

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self, padding=8)
        toolbar.pack(fill=tk.X)
        ttk.Label(toolbar, textvariable=self.status_var).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="再読み込み", command=self.refresh).pack(side=tk.RIGHT)

        preview = ttk.Frame(self, padding=(8, 0))
        preview.pack(fill=tk.BOTH, expand=True)
        self.diff_label = ttk.Label(preview, anchor=tk.CENTER)
        self.diff_label.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self.temporal_label = ttk.Label(preview, anchor=tk.CENTER)
        self.temporal_label.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)
        self.video_label = ttk.Label(preview, anchor=tk.CENTER)
        self.video_label.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=4, pady=4)
        preview.columnconfigure(0, weight=1)
        preview.columnconfigure(1, weight=1)
        preview.rowconfigure(0, weight=1)
        preview.rowconfigure(1, weight=1)

        ttk.Label(self, textvariable=self.detail_var, padding=8, justify=tk.LEFT).pack(fill=tk.X)

        controls = ttk.Frame(self, padding=10)
        controls.pack(fill=tk.X)
        ttk.Button(controls, text="← 前の未確認", command=self._previous).pack(side=tk.LEFT)
        ttk.Button(controls, text="次の未確認 →", command=self._next).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            controls, text="流星として確定  [M]", command=lambda: self._classify("meteor")
        ).pack(side=tk.LEFT, padx=(25, 5), expand=True, fill=tk.X)
        ttk.Button(
            controls, text="非流星として確定  [N]", command=lambda: self._classify("not_meteor")
        ).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        ttk.Label(controls, text="戻す件数:").pack(side=tk.LEFT, padx=(20, 3))
        ttk.Spinbox(controls, from_=1, to=100, width=4, textvariable=self.undo_count_var).pack(side=tk.LEFT)
        ttk.Button(controls, text="判定を戻す [Backspace]", command=self._undo).pack(side=tk.LEFT, padx=5)

        ttk.Label(
            self,
            text="左: 通常差分　右: 前半=赤・中盤=緑・後半=青の時間画像　下: モノクロ動画",
            padding=(10, 0, 10, 8),
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
        self._show_current()

    @property
    def current_event(self) -> Optional[Path]:
        if not self.events:
            return None
        return self.events[self.index]

    def _fit_photo(self, path: Path, max_size=(480, 270)) -> Optional[ImageTk.PhotoImage]:
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
            self.status_var.set("未確認データはありません")
            self.detail_var.set("すべて目視確認済みです。")
            for label in (self.diff_label, self.temporal_label, self.video_label):
                label.configure(image="", text="")
            self._photo_refs = []
            return

        try:
            meta = ml_training_data.load_metadata(event)
        except Exception as exc:
            self.detail_var.set(f"メタデータを読み込めません: {exc}")
            return
        self.status_var.set(f"未確認 {self.index + 1} / {len(self.events)}")
        self.detail_var.set(
            f"予測: {meta.get('predicted_label')}　流星確率: {meta.get('meteor_probability', 0):.3f}\n"
            f"日時: {meta.get('detection_time', '')}　元動画: {meta.get('source', '')}"
        )

        diff = self._fit_photo(event / "diff.png")
        temporal = self._fit_photo(event / "temporal_rgb.png")
        self._photo_refs = [p for p in (diff, temporal) if p is not None]
        self.diff_label.configure(image=diff or "", text="通常差分" if diff is None else "")
        self.temporal_label.configure(image=temporal or "", text="時間画像" if temporal is None else "")
        self.cap = cv2.VideoCapture(str(event / "clip.mp4"))
        self.video_frame_index = 0
        self._play_next_frame()

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
        image.thumbnail((720, 300), Image.Resampling.LANCZOS)
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
