import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from typing import Callable, Any, Dict, List
import ui_theme


class StatusPanel(ttk.Frame):
    """A composite panel that contains a Log tab and a Processing Status tab.

    Usage:
      panel = StatusPanel(parent, progress_queue, app)
      panel.pack(fill=tk.BOTH, expand=True)

    Provides:
      - panel.log_text: ScrolledText widget used for logs (so App.append_log can reuse it)
      - panel.get_status_callback(): returns a callable(status_dict) suitable to pass
        into the pipeline. The pipeline may call this from background threads; the
        callback marshals updates to the Tk main thread.
    """

    def __init__(self, parent, progress_queue, app, refresh_ms: int = 200):
        super().__init__(parent)
        self.parent = parent
        self.app = app
        self.progress_queue = progress_queue
        self.refresh_ms = refresh_ms

        c = ui_theme.COLORS
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Log tab
        self.log_frame = ttk.Frame(self.notebook)
        self.log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = ScrolledText(
            self.log_frame,
            wrap=tk.WORD,
            state="disabled",
            height=15,
            bg=c["field"],
            fg=c["text_secondary"],
            insertbackground=c["text"],
            selectbackground=c["selection"],
            selectforeground=c["text"],
            font=("SF Mono", 10),
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=c["border"],
            padx=10,
            pady=10,
            spacing1=2,
            spacing3=2,
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=(8, 6))
        
        # ログ保存ボタン
        log_btn_frame = ttk.Frame(self.log_frame)
        log_btn_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        save_log_btn = ttk.Button(
            log_btn_frame,
            text="ログを保存",
            style="Quiet.TButton",
            command=self._save_log,
        )
        open_meteor_btn = ttk.Button(
            log_btn_frame,
            text="結果をFinderで開く",
            style="Quiet.TButton",
            command=lambda: self._open_meteor_folder(),
        )
        save_log_btn.pack(side=tk.RIGHT)
        open_meteor_btn.pack(side=tk.RIGHT, padx=(0, 5))

        # Processing status tab
        self.status_frame = ttk.Frame(self.notebook)
        self.status_frame.pack(fill=tk.BOTH, expand=True)

        self.notebook.add(self.log_frame, text="イベントログ")
        self.notebook.add(self.status_frame, text="処理キュー")

        # Processing visuals
        self._build_status_ui()

        # internal state updated by status_callback
        self._last_status = {}

    def _build_status_ui(self):
        pad = 8
        # Download queue label + canvas
        c = ui_theme.COLORS
        dl_label = ttk.Label(self.status_frame, text="ダウンロード")
        dl_label.pack(anchor='w', padx=pad, pady=(pad, 0))
        self.dl_canvas = tk.Canvas(
            self.status_frame,
            height=52,
            bg=c["field"],
            highlightthickness=1,
            highlightbackground=c["border"],
        )
        self.dl_canvas.pack(fill=tk.X, padx=pad, pady=(2, pad))

        # Processing queue label + canvas
        pr_label = ttk.Label(self.status_frame, text="ワーカー")
        pr_label.pack(anchor='w', padx=pad, pady=(pad, 0))
        self.pr_canvas = tk.Canvas(
            self.status_frame,
            height=86,
            bg=c["field"],
            highlightthickness=1,
            highlightbackground=c["border"],
        )
        self.pr_canvas.pack(fill=tk.X, padx=pad, pady=(2, pad))

        # legend
        legend_frame = ttk.Frame(self.status_frame)
        legend_frame.pack(anchor='w', padx=pad, pady=(0, pad))
        ttk.Label(legend_frame, text="● 処理中", foreground=c["accent"]).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(legend_frame, text="○ 待機", foreground=c["text_secondary"]).pack(side=tk.LEFT)

    def _save_log(self):
        """ログ内容をファイルに保存する"""
        from tkinter import filedialog, messagebox
        from datetime import datetime
        
        # ログテキストの内容を取得
        log_content = self.log_text.get("1.0", tk.END).strip()
        if not log_content:
            messagebox.showinfo("情報", "保存するログがありません。")
            return
        
        # デフォルトファイル名を生成
        default_filename = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        # 保存先を選択
        file_path = filedialog.asksaveasfilename(
            title="ログを保存",
            defaultextension=".txt",
            initialfile=default_filename,
            filetypes=[("テキストファイル", "*.txt"), ("すべてのファイル", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(log_content)
                messagebox.showinfo("成功", f"ログを保存しました:\n{file_path}")
            except Exception as e:
                messagebox.showerror("エラー", f"ログの保存に失敗しました:\n{e}")


    def _open_meteor_folder(self):
        """Open the meteor save folder in Finder via the app."""
        if hasattr(self.app, "open_meteor_folder_in_finder"):
            self.app.open_meteor_folder_in_finder()

    def get_status_callback(self) -> Callable[[Dict[str, Any]], None]:
        """Return a callback which never enters Tcl from a worker thread.

        ``tkinter.after`` is itself a Tcl command.  Calling it from a Python
        worker is therefore not a safe way to marshal work on every Tcl/Tk
        build (in particular, this can corrupt Tk's pending-command state on
        macOS under sustained load).  The app already owns a thread-safe
        progress queue which the Tk thread polls, so status updates use that
        same bridge.
        """
        def cb(status: Dict[str, Any]):
            try:
                self.progress_queue.put((None, {"pipeline_status": dict(status)}))
            except Exception:
                # if something goes wrong, ignore — it's only informational
                pass

        return cb

    def update_status(self, status: Dict[str, Any]):
        """Called on GUI thread. Status dict keys:
           - download_queue_size: int
           - pending_sources: int
           - processors_busy: List[bool] (length = concurrency)
        """
        self._last_status = status
        # update download canvas
        dl_q = int(status.get('download_queue_size', 0))
        pending = int(status.get('pending_sources', 0))
        self._draw_download_queue(dl_q, pending)

        processors_busy: List[bool] = status.get('processors_busy', []) or []
        self._draw_processors(processors_busy)

    def _draw_download_queue(self, qsize: int, pending: int):
        c = self.dl_canvas
        c.delete('all')
        w = c.winfo_width() or c.winfo_reqwidth() or 600
        h = c.winfo_height() or 48
        colors = ui_theme.COLORS
        # draw up to 12 compact capsules; extra shown as +N
        max_boxes = 12
        box_w = 26
        gap = 6
        x = 8
        y = (h - box_w) // 2
        draw_boxes = min(qsize, max_boxes)
        for i in range(draw_boxes):
            c.create_rectangle(
                x,
                y,
                x + box_w,
                y + box_w,
                outline=colors["border_bright"],
                fill=colors["glass_selected"],
                width=1,
            )
            x += box_w + gap
        if qsize > max_boxes:
            c.create_text(x + 10, h // 2, text=f'+{qsize - max_boxes}', fill=colors["text"], anchor='w')
            x += 30
        # show pending count on right
        right_text = f'待機 {pending}'
        c.create_text(w - 10, h // 2, text=right_text, fill=colors["text_secondary"], anchor='e')

    def _draw_processors(self, busy: List[bool]):
        c = self.pr_canvas
        c.delete('all')
        w = c.winfo_width() or c.winfo_reqwidth() or 600
        h = c.winfo_height() or 80
        colors = ui_theme.COLORS
        n = max(1, len(busy))
        box_w = min(48, max(24, (w - 20) // n - 6))
        gap = 8
        x = 8
        y = (h - box_w) // 2
        for i, b in enumerate(busy):
            color = colors["accent_pressed"] if b else colors["field"]
            outline = colors["accent"] if b else colors["border_bright"]
            c.create_rectangle(x, y, x + box_w, y + box_w, outline=outline, fill=color, width=2 if b else 1)
            c.create_text(
                x + box_w // 2,
                y + box_w + 12,
                text=str(i + 1),
                fill=colors["text"] if b else colors["text_secondary"],
            )
            x += box_w + gap


# end of file
