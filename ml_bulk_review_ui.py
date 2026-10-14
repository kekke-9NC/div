"""Background bulk review dialog for the legacy dataset reviewer."""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

import ml_bulk_review as bulk
import ml_training_data


class BulkReviewDialog(tk.Toplevel):
    def __init__(self, reviewer):
        super().__init__(reviewer)
        self.reviewer = reviewer
        self.title("流星確率で一括選別")
        self.geometry("720x340")
        self.resizable(False, False)
        self.transient(reviewer)
        self.selection = None
        self.busy = False
        self.closing = False
        self.cancel = threading.Event()
        self.messages = queue.Queue()
        self.threshold = tk.StringVar(self, "0.10")
        self.comparison = tk.StringVar(self, "以下（≤）")
        self.day = tk.StringVar(self)
        self.predicted = tk.StringVar(self, "すべて")
        self.status = tk.StringVar(self, "条件を設定して「対象件数を確認」を押してください。")
        self.batch_records = bulk.latest_batch_records(reviewer.undo_stack + reviewer.skip_stack)
        self.inputs = []
        self._build_ui()
        for variable in (self.threshold, self.comparison, self.day, self.predicted):
            variable.trace_add("write", self._invalidate_selection)
        self.protocol("WM_DELETE_WINDOW", self._close)
        self._update_buttons()

    def _build_ui(self):
        body = ttk.Frame(self, padding=16)
        body.pack(fill=tk.BOTH, expand=True)
        ttk.Label(body, text="未確認データだけが対象です。確定済み・スキップ済みは変更しません。").pack(anchor=tk.W)
        conditions = ttk.Frame(body)
        conditions.pack(fill=tk.X, pady=(14, 4))
        ttk.Label(conditions, text="検出時の流星確率:").pack(side=tk.LEFT)
        entry = ttk.Entry(conditions, textvariable=self.threshold, width=9)
        entry.pack(side=tk.LEFT, padx=6)
        comparison = ttk.Combobox(conditions, textvariable=self.comparison,
                                 values=("以下（≤）", "以上（≥）"), width=10, state="readonly")
        comparison.pack(side=tk.LEFT)
        ttk.Label(conditions, text="（0〜1、境界値を含む）").pack(side=tk.LEFT, padx=8)
        self.inputs.extend((entry, comparison))
        date_row = ttk.Frame(body)
        date_row.pack(fill=tk.X, pady=4)
        ttk.Label(date_row, text="対象日:").pack(side=tk.LEFT)
        date_entry = ttk.Entry(date_row, textvariable=self.day, width=14)
        date_entry.pack(side=tk.LEFT, padx=6)
        date_button = ttk.Button(date_row, text="表示中の検出日", command=self._current_day)
        date_button.pack(side=tk.LEFT)
        ttk.Label(date_row, text="空欄なら全日付 / YYYY-MM-DD").pack(side=tk.LEFT, padx=8)
        self.inputs.extend((date_entry, date_button))
        prediction_row = ttk.Frame(body)
        prediction_row.pack(fill=tk.X, pady=4)
        ttk.Label(prediction_row, text="予測ラベル:").pack(side=tk.LEFT)
        prediction = ttk.Combobox(prediction_row, textvariable=self.predicted,
                                 values=("すべて", "meteor", "not_meteor"), state="readonly", width=16)
        prediction.pack(side=tk.LEFT, padx=6)
        self.inputs.append(prediction)
        buttons = ttk.Frame(body)
        buttons.pack(fill=tk.X, pady=(12, 4))
        self.scan_button = ttk.Button(buttons, text="対象件数を確認", command=self._scan)
        self.scan_button.pack(side=tk.LEFT)
        self.actions = []
        for label in ("skip", "meteor", "not_meteor"):
            button = ttk.Button(buttons, text=f"一括 {label}", command=lambda value=label: self._apply(value))
            button.pack(side=tk.LEFT, padx=(8, 0))
            self.actions.append(button)
        ttk.Label(body, textvariable=self.status, wraplength=675, justify=tk.LEFT).pack(fill=tk.X, pady=10)
        footer = ttk.Frame(body)
        footer.pack(fill=tk.X)
        self.undo_button = ttk.Button(footer, text="前回の一括処理を戻す", command=self._undo)
        self.undo_button.pack(side=tk.LEFT)
        self.cancel_button = ttk.Button(footer, text="処理を中断", command=self.cancel.set)
        self.cancel_button.pack(side=tk.RIGHT)

    def _criteria(self):
        return bulk.ReviewFilter(
            self.threshold.get(), "below" if self.comparison.get() == "以下（≤）" else "above",
            self.day.get().strip(), "" if self.predicted.get() == "すべて" else self.predicted.get(),
        )

    def _invalidate_selection(self, *_args):
        self.selection = None
        if not self.busy:
            self.status.set("条件を変更しました。「対象件数を確認」を押してください。")
        self._update_buttons()

    def _current_day(self):
        event = self.reviewer.current_event
        day = ml_training_data.detection_date(event) if event else None
        if day:
            self.day.set(day)
        else:
            messagebox.showinfo("対象日", "表示中の検出日を読み取れません。", parent=self)

    def _update_buttons(self):
        loading = self.reviewer._load_busy or self.reviewer._closing
        unavailable = self.busy or loading
        for widget in self.inputs + [self.scan_button]:
            widget.state(["disabled" if unavailable else "!disabled"])
        allowed = not unavailable and self.selection is not None and bool(self.selection.candidates)
        for button in self.actions:
            button.state(["!disabled" if allowed else "disabled"])
        self.undo_button.state(["!disabled" if not unavailable and self.batch_records else "disabled"])
        self.cancel_button.state(["!disabled" if self.busy else "disabled"])

    def _start(self, operation, function):
        if self.busy or self.reviewer._load_busy or self.reviewer._closing:
            return
        self.busy = True
        self.cancel.clear()
        self.reviewer._stop_video()
        self._update_buttons()
        self.status.set("対象を検索中…" if operation == "scan" else "処理中…（中断すると処理済みの分は保持します）")

        def worker():
            try:
                self.messages.put((operation, function()))
            except Exception as exc:
                self.messages.put(("error", str(exc)))
        threading.Thread(target=worker, daemon=True).start()
        self.after(50, self._poll)

    def _progress(self, done, total):
        self.messages.put(("progress", (done, total)))

    def _scan(self):
        try:
            criteria = self._criteria()
        except (ValueError, TypeError) as exc:
            messagebox.showerror("条件エラー", str(exc), parent=self)
            return
        self.selection = None
        self._start("scan", lambda: bulk.select_events(self.reviewer.root_dir, criteria, self.cancel, self._progress))

    def _apply(self, action):
        if self.busy or self.selection is None:
            return
        selection = self.selection
        if selection.criteria != self._criteria():
            self._invalidate_selection()
            return
        sign = "以下" if selection.criteria.comparison == "below" else "以上"
        if not messagebox.askyesno(
            "一括選別を確認",
            f"流星確率 {selection.criteria.threshold:g} {sign} の未確認 {len(selection.candidates)}件を\n"
            f"{action} に選別しますか？\n前回の一括処理を戻すことができます。",
            parent=self,
        ):
            return
        self._start("apply", lambda: bulk.apply_selection(self.reviewer.root_dir, selection, action,
                    self.cancel, self.reviewer.media_directories, self._progress))

    def _undo(self):
        if self.busy or not self.batch_records:
            return
        records = list(self.batch_records)
        if messagebox.askyesno("一括処理を戻す", f"前回の一括処理 {len(records)}件を未確認に戻しますか？", parent=self):
            self._start("undo", lambda: bulk.undo_batch(self.reviewer.root_dir, records, self.cancel, self._progress))

    def _poll(self):
        finished = False
        while True:
            try:
                operation, value = self.messages.get_nowait()
            except queue.Empty:
                break
            if operation == "progress":
                self.status.set(f"処理中: {value[0]:,} / {value[1]:,}件")
                continue
            finished = True
            self.busy = False
            if operation == "scan":
                self.selection = None if value.cancelled else value
                self.status.set(f"{'検索中断' if value.cancelled else '検索完了'}: 対象 {len(value.candidates):,}件 / 未確認 {value.scanned:,}件\n"
                                f"確率・メタデータ等を読み取れず除外: {value.unreadable:,}件")
            elif operation in ("apply", "undo"):
                self.selection = None
                self.status.set(f"{'中断' if value.cancelled else '停止' if value.error else '完了'}: {len(value.records):,}件\n"
                                f"検索後に変更・移動されたため除外: {value.changed:,}件")
                if value.error:
                    messagebox.showerror("一括処理エラー", value.error + "\n処理済みの分は復元できます。", parent=self)
            else:
                self.selection = None
                messagebox.showerror("一括処理エラー", value, parent=self)
            self.reviewer.refresh()
            self._update_buttons()
        if self.closing and finished:
            self.destroy()
        elif self.busy:
            self.after(50, self._poll)

    def reviewer_loaded(self):
        self.batch_records = bulk.latest_batch_records(self.reviewer.undo_stack + self.reviewer.skip_stack)
        self._update_buttons()

    def _close(self):
        if self.busy:
            self.closing = True
            self.cancel.set()
            self.status.set("処理を中断して閉じています…")
        else:
            self.destroy()
