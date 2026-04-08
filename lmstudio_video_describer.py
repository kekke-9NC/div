import base64
import json
import os
import queue
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import requests
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from tkinterdnd2 import DND_FILES, TkinterDnD


DEFAULT_SERVER_URL = "http://127.0.0.1:1234"
DEFAULT_MODEL_KEY = "gemma-4-e2b-uncensored-hauhaucs-aggressive"
DEFAULT_API_TOKEN = os.environ.get("LM_STUDIO_API_TOKEN") or os.environ.get("LM_API_TOKEN", "")
SUPPORTED_VIDEO_SUFFIXES = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".m4v",
    ".wmv",
    ".mpg",
    ".mpeg",
}
SUMMARY_GROUP_SIZE = 8


class LMStudioError(Exception):
    pass


class AnalysisCancelled(Exception):
    pass


@dataclass
class BatchResult:
    batch_index: int
    frame_start: int
    frame_end: int
    analysis: str


def is_video_file(path: str) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_VIDEO_SUFFIXES


def format_seconds(seconds: float) -> str:
    total_millis = max(0, int(round(seconds * 1000)))
    minutes, millis_part = divmod(total_millis, 60_000)
    secs, millis = divmod(millis_part, 1000)
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"


def normalize_server_url(raw_url: str) -> str:
    value = (raw_url or DEFAULT_SERVER_URL).strip().rstrip("/")
    if not value:
        return DEFAULT_SERVER_URL
    for suffix in ("/api/v1", "/v1"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    return value or DEFAULT_SERVER_URL


def encode_frame_as_data_uri(frame_bgr, max_edge: int, jpeg_quality: int) -> str:
    height, width = frame_bgr.shape[:2]
    if max_edge > 0 and max(height, width) > max_edge:
        scale = max_edge / float(max(height, width))
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        frame_bgr = cv2.resize(frame_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)

    success, encoded = cv2.imencode(
        ".jpg",
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), max(40, min(int(jpeg_quality), 100))],
    )
    if not success:
        raise LMStudioError("フレームのJPEGエンコードに失敗しました。")
    encoded_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded_b64}"


class LMStudioClient:
    def __init__(self, server_url: str, api_token: str = ""):
        self.server_url = normalize_server_url(server_url)
        self.api_token = (api_token or "").strip()
        self.session = requests.Session()

    def _headers(self, json_body: bool = False) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        if json_body:
            headers["Content-Type"] = "application/json"
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        return headers

    def _request(self, method: str, path: str, *, timeout: int = 30, payload: Optional[dict] = None) -> dict:
        url = f"{self.server_url}{path}"
        try:
            response = self.session.request(
                method=method,
                url=url,
                headers=self._headers(json_body=payload is not None),
                json=payload,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            raise LMStudioError(
                f"LM Studio に接続できませんでした: {self.server_url}\n{exc}"
            ) from exc

        if not response.ok:
            try:
                error_payload = response.json()
                error_text = json.dumps(error_payload, ensure_ascii=False, indent=2)
            except ValueError:
                error_text = response.text.strip()
            raise LMStudioError(
                f"LM Studio API エラー ({response.status_code}):\n{error_text or '詳細不明'}"
            )

        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError as exc:
            raise LMStudioError("LM Studio の応答がJSONではありませんでした。") from exc

    def list_models(self) -> List[dict]:
        payload = self._request("GET", "/api/v1/models", timeout=20)
        return payload.get("models", [])

    def check_server(self) -> None:
        self.list_models()

    def load_model(self, model_key: str, context_length: int) -> dict:
        payload = {
            "model": model_key,
            "context_length": int(context_length),
            "echo_load_config": True,
        }
        return self._request("POST", "/api/v1/models/load", payload=payload, timeout=180)

    def unload_model(self, instance_id: str) -> dict:
        return self._request(
            "POST",
            "/api/v1/models/unload",
            payload={"instance_id": instance_id},
            timeout=120,
        )

    def get_loaded_instance_id(self, model_key: str) -> Optional[str]:
        for model in self.list_models():
            if model.get("key") != model_key:
                continue
            loaded_instances = model.get("loaded_instances") or []
            if loaded_instances:
                instance = loaded_instances[0]
                return instance.get("id") or instance.get("instance_id")
        return None

    def chat_completion(
        self,
        model_key: str,
        system_prompt: str,
        user_content: Sequence[dict],
        *,
        max_tokens: int,
        temperature: float = 0.1,
        timeout: int = 300,
    ) -> str:
        payload = {
            "model": model_key,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": list(user_content)},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        data = self._request("POST", "/v1/chat/completions", payload=payload, timeout=timeout)
        choices = data.get("choices") or []
        if not choices:
            raise LMStudioError("LM Studio から有効な応答が返りませんでした。")
        content = choices[0].get("message", {}).get("content", "")
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") in {"text", "output_text"} and item.get("text"):
                        text_parts.append(item["text"])
            return "\n".join(text_parts).strip()
        return str(content).strip()


class VideoDescriberApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("LM Studio Video Describer")
        self.geometry("1360x900")
        self.minsize(1180, 760)

        self.ui_queue: "queue.Queue[tuple]" = queue.Queue()
        self.worker_thread: Optional[threading.Thread] = None
        self.cancel_event = threading.Event()
        self.video_path: Optional[Path] = None
        self.preview_image = None
        self.model_map: Dict[str, dict] = {}
        self.last_server_ok = False

        self.server_var = tk.StringVar(value=DEFAULT_SERVER_URL)
        self.api_token_var = tk.StringVar(value=DEFAULT_API_TOKEN)
        self.model_var = tk.StringVar(value=DEFAULT_MODEL_KEY)
        self.connection_var = tk.StringVar(value="LM Studio: 確認中")
        self.loaded_model_var = tk.StringVar(value="ロード状態: 不明")
        self.video_info_var = tk.StringVar(value="動画をドラッグ＆ドロップするか、選択してください。")
        self.progress_var = tk.StringVar(value="待機中")
        self.context_length_var = tk.IntVar(value=8192)
        self.frames_per_batch_var = tk.IntVar(value=6)
        self.max_edge_var = tk.IntVar(value=768)
        self.jpeg_quality_var = tk.IntVar(value=82)
        self.max_tokens_var = tk.IntVar(value=600)

        self._setup_icon()
        self._setup_style()
        self._build_ui()
        self._bind_drag_and_drop()

        self.after(100, self._drain_ui_queue)
        self.after(250, lambda: self.refresh_models_async(show_popup=True))

    def _setup_icon(self) -> None:
        try:
            icon_path = Path(__file__).with_name("icon.ico")
            if icon_path.exists():
                self.iconbitmap(icon_path)
        except Exception:
            pass

    def _setup_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Drop.TFrame", background="#101926")
        style.configure("Drop.TLabel", background="#101926", foreground="#f0f4f8")
        style.configure("Section.TLabelframe", padding=10)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=12)
        left.grid(row=0, column=0, sticky="nsew")
        left.columnconfigure(0, weight=1)

        right = ttk.Frame(self, padding=(0, 12, 12, 12))
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        self._build_connection_panel(left)
        self._build_video_panel(left)
        self._build_options_panel(left)
        self._build_action_panel(left)
        self._build_output_panel(right)

    def _build_connection_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="LM Studio 接続", style="Section.TLabelframe")
        frame.grid(row=0, column=0, sticky="ew")
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Server URL").grid(row=0, column=0, sticky="w", pady=(0, 6))
        ttk.Entry(frame, textvariable=self.server_var).grid(row=0, column=1, sticky="ew", pady=(0, 6))

        ttk.Label(frame, text="API Token").grid(row=1, column=0, sticky="w", pady=(0, 6))
        ttk.Entry(frame, textvariable=self.api_token_var, show="*").grid(row=1, column=1, sticky="ew", pady=(0, 6))

        ttk.Label(frame, text="Model").grid(row=2, column=0, sticky="w")
        self.model_combo = ttk.Combobox(frame, textvariable=self.model_var)
        self.model_combo.grid(row=2, column=1, sticky="ew")

        buttons = ttk.Frame(frame)
        buttons.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        buttons.columnconfigure((0, 1, 2), weight=1)

        self.refresh_button = ttk.Button(buttons, text="一覧更新", command=lambda: self.refresh_models_async(show_popup=True))
        self.refresh_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self.load_button = ttk.Button(buttons, text="モデルをロード", command=self.on_load_model)
        self.load_button.grid(row=0, column=1, sticky="ew", padx=4)

        self.unload_button = ttk.Button(buttons, text="モデルをアンロード", command=self.on_unload_model)
        self.unload_button.grid(row=0, column=2, sticky="ew", padx=(4, 0))

        ttk.Label(frame, textvariable=self.connection_var).grid(row=4, column=0, columnspan=2, sticky="w", pady=(10, 2))
        ttk.Label(frame, textvariable=self.loaded_model_var).grid(row=5, column=0, columnspan=2, sticky="w")

    def _build_video_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="動画入力", style="Section.TLabelframe")
        frame.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)
        parent.rowconfigure(1, weight=1)

        top = ttk.Frame(frame)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        top.columnconfigure(0, weight=1)

        ttk.Button(top, text="動画を選択", command=self.on_pick_video).grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ttk.Label(top, textvariable=self.video_info_var, wraplength=380, justify="left").grid(row=0, column=1, sticky="w")

        self.drop_frame = ttk.Frame(frame, style="Drop.TFrame", height=180)
        self.drop_frame.grid(row=1, column=0, sticky="nsew")
        self.drop_frame.columnconfigure(0, weight=1)
        self.drop_frame.rowconfigure(1, weight=1)

        ttk.Label(
            self.drop_frame,
            text="動画をここへドラッグ＆ドロップ",
            style="Drop.TLabel",
            font=("Yu Gothic UI", 15, "bold"),
        ).grid(row=0, column=0, sticky="n", pady=(18, 6))

        self.preview_label = ttk.Label(self.drop_frame, text="プレビューなし")
        self.preview_label.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))

    def _build_options_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="解析設定", style="Section.TLabelframe")
        frame.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Context Length").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(frame, from_=2048, to=131072, increment=1024, textvariable=self.context_length_var).grid(row=0, column=1, sticky="ew")

        ttk.Label(frame, text="Frames / Request").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(frame, from_=1, to=20, textvariable=self.frames_per_batch_var).grid(row=1, column=1, sticky="ew", pady=(6, 0))

        ttk.Label(frame, text="Max Edge(px)").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(frame, from_=256, to=2048, increment=64, textvariable=self.max_edge_var).grid(row=2, column=1, sticky="ew", pady=(6, 0))

        ttk.Label(frame, text="JPEG Quality").grid(row=3, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(frame, from_=50, to=95, textvariable=self.jpeg_quality_var).grid(row=3, column=1, sticky="ew", pady=(6, 0))

        ttk.Label(frame, text="Max Tokens").grid(row=4, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(frame, from_=128, to=2048, increment=64, textvariable=self.max_tokens_var).grid(row=4, column=1, sticky="ew", pady=(6, 0))

        ttk.Label(frame, text="AIへの指示").grid(row=5, column=0, sticky="nw", pady=(10, 0))
        self.prompt_text = ScrolledText(frame, height=8, wrap="word")
        self.prompt_text.grid(row=5, column=1, sticky="nsew", pady=(10, 0))
        self.prompt_text.insert(
            "1.0",
            (
                "この動画の全フレームを順番に確認し、各フレームで何が写っているかを日本語で正確に説明してください。"
                "見えていないものは推測で断定せず、不確実なら不確実と明記してください。"
                "動画内の動きや一瞬だけ現れる物体があれば、それも書いてください。"
            ),
        )
        frame.rowconfigure(5, weight=1)

    def _build_action_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="実行", style="Section.TLabelframe")
        frame.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        self.run_button = ttk.Button(frame, text="全フレーム解析を開始", command=self.on_run_analysis)
        self.run_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.cancel_button = ttk.Button(frame, text="キャンセル", command=self.on_cancel_analysis, state=tk.DISABLED)
        self.cancel_button.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        self.progress_bar = ttk.Progressbar(frame, mode="determinate", maximum=100)
        self.progress_bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(12, 4))
        ttk.Label(frame, textvariable=self.progress_var).grid(row=2, column=0, columnspan=2, sticky="w")

    def _build_output_panel(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="結果", font=("Yu Gothic UI", 16, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 8))

        notebook = ttk.Notebook(parent)
        notebook.grid(row=1, column=0, sticky="nsew")

        result_tab = ttk.Frame(notebook, padding=10)
        result_tab.columnconfigure(0, weight=1)
        result_tab.rowconfigure(0, weight=1)
        self.result_text = ScrolledText(result_tab, wrap="word", font=("Yu Gothic UI", 11))
        self.result_text.grid(row=0, column=0, sticky="nsew")
        notebook.add(result_tab, text="AI回答")

        log_tab = ttk.Frame(notebook, padding=10)
        log_tab.columnconfigure(0, weight=1)
        log_tab.rowconfigure(0, weight=1)
        self.log_text = ScrolledText(log_tab, wrap="word", font=("Consolas", 10))
        self.log_text.grid(row=0, column=0, sticky="nsew")
        notebook.add(log_tab, text="ログ")

    def _bind_drag_and_drop(self) -> None:
        self.drop_target_register(DND_FILES)
        self.dnd_bind("<<Drop>>", self.on_drop)
        self.drop_frame.drop_target_register(DND_FILES)
        self.drop_frame.dnd_bind("<<Drop>>", self.on_drop)

    def _drain_ui_queue(self) -> None:
        while True:
            try:
                callback, args, kwargs = self.ui_queue.get_nowait()
            except queue.Empty:
                break
            callback(*args, **kwargs)
        self.after(100, self._drain_ui_queue)

    def _call_in_ui(self, callback, *args, **kwargs) -> None:
        self.ui_queue.put((callback, args, kwargs))

    def _append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{timestamp}] {message}\n")
        self.log_text.see("end")

    def _set_status(self, message: str) -> None:
        self.progress_var.set(message)

    def _set_connection_state(self, connected: bool) -> None:
        self.last_server_ok = connected
        self.connection_var.set("LM Studio: 接続済み" if connected else "LM Studio: 未接続")

    def _set_progress(self, current: int, total: int, detail: str) -> None:
        total = max(1, total)
        self.progress_bar["maximum"] = total
        self.progress_bar["value"] = min(current, total)
        self.progress_var.set(detail)

    def _set_result_text(self, content: str) -> None:
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", content)

    def _set_busy(self, busy: bool) -> None:
        state = tk.DISABLED if busy else tk.NORMAL
        self.run_button.configure(state=state)
        self.load_button.configure(state=state)
        self.unload_button.configure(state=state)
        self.refresh_button.configure(state=state)
        self.cancel_button.configure(state=tk.NORMAL if busy else tk.DISABLED)

    def _set_model_choices(self, values: List[str]) -> None:
        self.model_combo["values"] = values

    def _show_popup(self, level: str, title: str, message: str) -> None:
        if level == "error":
            messagebox.showerror(title, message)
        elif level == "warning":
            messagebox.showwarning(title, message)
        else:
            messagebox.showinfo(title, message)

    def build_client(self) -> LMStudioClient:
        return LMStudioClient(self.server_var.get(), self.api_token_var.get())

    def refresh_models_async(self, *, show_popup: bool) -> None:
        def worker():
            try:
                client = self.build_client()
                models = client.list_models()
                self._call_in_ui(self._set_connection_state, True)
                self._call_in_ui(self._update_model_list, models)
                self._call_in_ui(self._append_log, f"モデル一覧を取得しました: {len(models)} 件")
            except LMStudioError as exc:
                self._call_in_ui(self._set_connection_state, False)
                self._call_in_ui(self._append_log, str(exc))
                if show_popup:
                    self._call_in_ui(
                        self._show_popup,
                        "warning",
                        "LM Studio 未起動",
                        (
                            "LM Studio に接続できませんでした。\n"
                            "LM Studio の Local Server が起動しているか確認してください。\n\n"
                            f"{exc}"
                        ),
                    )

        threading.Thread(target=worker, daemon=True).start()

    def _update_model_list(self, models: List[dict]) -> None:
        self.model_map.clear()
        ordered_keys: List[str] = []
        vision_keys: List[str] = []
        other_keys: List[str] = []

        for model in models:
            key = model.get("key")
            if not key:
                continue
            self.model_map[key] = model
            capabilities = model.get("capabilities") or {}
            has_vision = bool(capabilities.get("vision")) or bool(model.get("vision"))
            if has_vision:
                vision_keys.append(key)
            else:
                other_keys.append(key)

        ordered_keys.extend(sorted(vision_keys))
        ordered_keys.extend(sorted(other_keys))

        current = self.model_var.get().strip()
        if current and current not in ordered_keys:
            ordered_keys.insert(0, current)
        elif not current and ordered_keys:
            self.model_var.set(ordered_keys[0])

        self._set_model_choices(ordered_keys)
        self._update_loaded_model_label()

    def _update_loaded_model_label(self) -> None:
        selected = self.model_var.get().strip()
        info = self.model_map.get(selected)
        if not info:
            self.loaded_model_var.set("ロード状態: モデル情報なし")
            return
        loaded_instances = info.get("loaded_instances") or []
        if loaded_instances:
            instance_id = loaded_instances[0].get("id") or loaded_instances[0].get("instance_id") or "(unknown)"
            self.loaded_model_var.set(f"ロード状態: 読み込み済み ({instance_id})")
        else:
            self.loaded_model_var.set("ロード状態: 未ロード")

    def on_pick_video(self) -> None:
        file_path = filedialog.askopenfilename(
            title="動画を選択",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v *.wmv *.mpg *.mpeg")],
        )
        if file_path:
            self._set_video(Path(file_path))

    def on_drop(self, event) -> None:
        paths = self.tk.splitlist(event.data)
        chosen = None
        for raw_path in paths:
            candidate = Path(raw_path)
            if candidate.exists() and is_video_file(str(candidate)):
                chosen = candidate
                break
        if chosen is None:
            self._show_popup("warning", "動画ではありません", "対応している動画ファイルをドロップしてください。")
            return
        self._set_video(chosen)

    def _set_video(self, path: Path) -> None:
        self.video_path = path
        metadata = self._read_video_metadata(path)
        if metadata is None:
            return
        frame_count, fps, duration = metadata
        self.video_info_var.set(
            f"{path}\nFrames: {frame_count:,} / FPS: {fps:.2f} / Duration: {duration:.2f}s"
        )
        self._append_log(f"動画を選択: {path}")
        self._load_preview(path)

    def _read_video_metadata(self, path: Path) -> Optional[tuple]:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            self._show_popup("error", "動画を開けません", f"動画を開けませんでした:\n{path}")
            return None
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 1.0
        duration = frame_count / fps if frame_count > 0 else 0.0
        capture.release()
        return frame_count, fps, duration

    def _load_preview(self, path: Path) -> None:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            self.preview_label.configure(text="プレビューを作成できません。", image="")
            return
        ok, frame = capture.read()
        capture.release()
        if not ok:
            self.preview_label.configure(text="プレビューを作成できません。", image="")
            return
        preview_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(preview_rgb)
        image.thumbnail((360, 220))
        self.preview_image = ImageTk.PhotoImage(image)
        self.preview_label.configure(image=self.preview_image, text="")

    def on_load_model(self) -> None:
        selected_model = self.model_var.get().strip()
        if not selected_model:
            self._show_popup("warning", "モデル未指定", "ロードするモデルを入力してください。")
            return

        def worker():
            try:
                self._call_in_ui(self._set_busy, True)
                self._call_in_ui(self._set_status, "モデルをロード中...")
                client = self.build_client()
                response = client.load_model(selected_model, self.context_length_var.get())
                self._call_in_ui(self._append_log, f"モデルをロードしました: {json.dumps(response, ensure_ascii=False)}")
                self._call_in_ui(self._set_connection_state, True)
                self._call_in_ui(self._set_status, "モデルのロードが完了しました。")
                self.refresh_models_async(show_popup=False)
            except LMStudioError as exc:
                self._call_in_ui(self._set_connection_state, False)
                self._call_in_ui(self._append_log, str(exc))
                self._call_in_ui(
                    self._show_popup,
                    "warning",
                    "LM Studio 未起動またはロード失敗",
                    (
                        "モデルのロードに失敗しました。\n"
                        "LM Studio が起動していない場合は Local Server を開始してください。\n\n"
                        f"{exc}"
                    ),
                )
                self._call_in_ui(self._set_status, "モデルのロードに失敗しました。")
            finally:
                self._call_in_ui(self._set_busy, False)

        threading.Thread(target=worker, daemon=True).start()

    def on_unload_model(self) -> None:
        selected_model = self.model_var.get().strip()
        if not selected_model:
            self._show_popup("warning", "モデル未指定", "アンロードするモデルを入力してください。")
            return

        def worker():
            try:
                self._call_in_ui(self._set_busy, True)
                self._call_in_ui(self._set_status, "モデルをアンロード中...")
                client = self.build_client()
                instance_id = client.get_loaded_instance_id(selected_model)
                if not instance_id:
                    raise LMStudioError("選択中のモデルは現在ロードされていません。")
                response = client.unload_model(instance_id)
                self._call_in_ui(self._append_log, f"モデルをアンロードしました: {json.dumps(response, ensure_ascii=False)}")
                self._call_in_ui(self._set_connection_state, True)
                self._call_in_ui(self._set_status, "モデルのアンロードが完了しました。")
                self.refresh_models_async(show_popup=False)
            except LMStudioError as exc:
                self._call_in_ui(self._append_log, str(exc))
                self._call_in_ui(
                    self._show_popup,
                    "warning",
                    "アンロード失敗",
                    f"モデルのアンロードに失敗しました。\n\n{exc}",
                )
                self._call_in_ui(self._set_status, "モデルのアンロードに失敗しました。")
            finally:
                self._call_in_ui(self._set_busy, False)

        threading.Thread(target=worker, daemon=True).start()

    def on_run_analysis(self) -> None:
        if self.video_path is None:
            self._show_popup("warning", "動画未選択", "先に動画を選択してください。")
            return
        if self.worker_thread and self.worker_thread.is_alive():
            self._show_popup("warning", "実行中", "すでに解析が実行中です。")
            return

        selected_model = self.model_var.get().strip()
        if not selected_model:
            self._show_popup("warning", "モデル未指定", "解析に使うモデルを入力してください。")
            return

        info = self.model_map.get(selected_model)
        if info:
            capabilities = info.get("capabilities") or {}
            has_vision = bool(capabilities.get("vision")) or bool(info.get("vision"))
            if not has_vision:
                self._show_popup(
                    "warning",
                    "非Visionモデルの可能性",
                    "選択中モデルは画像入力非対応の可能性があります。Vision対応モデルを選ぶことをおすすめします。",
                )

        self.cancel_event.clear()
        self.worker_thread = threading.Thread(target=self._analysis_worker, daemon=True)
        self.worker_thread.start()

    def on_cancel_analysis(self) -> None:
        self.cancel_event.set()
        self._append_log("キャンセル要求を受け付けました。現在の処理単位が終わり次第停止します。")
        self._set_status("キャンセル要求を送信しました...")

    def _analysis_worker(self) -> None:
        try:
            self._call_in_ui(self._set_busy, True)
            self._call_in_ui(self._set_result_text, "")
            self._call_in_ui(self._set_progress, 0, 100, "LM Studio に接続中...")

            client = self.build_client()
            client.check_server()
            self._call_in_ui(self._set_connection_state, True)
            self._call_in_ui(self._append_log, "LM Studio への接続を確認しました。")

            model_key = self.model_var.get().strip()
            instance_id = client.get_loaded_instance_id(model_key)
            if instance_id:
                self._call_in_ui(self._append_log, f"モデルは既にロード済みです: {instance_id}")
            else:
                self._call_in_ui(self._append_log, f"モデルをロードします: {model_key}")
                response = client.load_model(model_key, self.context_length_var.get())
                self._call_in_ui(self._append_log, f"ロード完了: {json.dumps(response, ensure_ascii=False)}")
                self.refresh_models_async(show_popup=False)

            assert self.video_path is not None
            batch_results = self._describe_video_frames(client, self.video_path, model_key)
            final_summary = self._build_final_summary(client, model_key, self.video_path, batch_results)
            output_paths = self._save_outputs(self.video_path, model_key, batch_results, final_summary)

            display_text = (
                f"動画: {self.video_path}\n"
                f"モデル: {model_key}\n"
                f"生成日時: {datetime.now().isoformat(timespec='seconds')}\n"
                f"保存先:\n- {output_paths['summary']}\n- {output_paths['json']}\n\n"
                f"===== 総合要約 =====\n{final_summary}\n\n"
                "===== バッチごとの説明 =====\n"
            )
            for item in batch_results:
                display_text += (
                    f"\n[Batch {item.batch_index} | frames {item.frame_start}-{item.frame_end}]\n"
                    f"{item.analysis}\n"
                )

            self._call_in_ui(self._set_result_text, display_text)
            self._call_in_ui(
                self._set_progress,
                len(batch_results),
                max(1, len(batch_results)),
                "全フレーム解析が完了しました。",
            )
            self._call_in_ui(
                self._append_log,
                f"解析完了。結果を保存しました: {output_paths['summary']} / {output_paths['json']}",
            )
        except AnalysisCancelled:
            self._call_in_ui(self._append_log, "解析をキャンセルしました。")
            self._call_in_ui(self._set_status, "解析をキャンセルしました。")
        except LMStudioError as exc:
            self._call_in_ui(self._set_connection_state, False)
            self._call_in_ui(self._append_log, str(exc))
            self._call_in_ui(
                self._show_popup,
                "warning",
                "LM Studio 未起動または解析失敗",
                (
                    "解析中に LM Studio へ接続できませんでした。\n"
                    "LM Studio の Local Server が起動しているか確認してください。\n\n"
                    f"{exc}"
                ),
            )
            self._call_in_ui(self._set_status, "解析に失敗しました。")
        except Exception as exc:
            self._call_in_ui(self._append_log, f"予期しないエラー: {exc}")
            self._call_in_ui(self._show_popup, "error", "エラー", f"予期しないエラーが発生しました。\n\n{exc}")
            self._call_in_ui(self._set_status, "予期しないエラーで停止しました。")
        finally:
            self._call_in_ui(self._set_busy, False)
            self._call_in_ui(self._update_loaded_model_label)

    def _raise_if_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise AnalysisCancelled()

    def _describe_video_frames(self, client: LMStudioClient, video_path: Path, model_key: str) -> List[BatchResult]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise LMStudioError(f"動画を開けませんでした: {video_path}")

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 1.0
        frames_per_batch = max(1, int(self.frames_per_batch_var.get()))
        max_edge = max(128, int(self.max_edge_var.get()))
        jpeg_quality = max(50, min(95, int(self.jpeg_quality_var.get())))
        max_tokens = max(128, int(self.max_tokens_var.get()))
        prompt_text = self.prompt_text.get("1.0", "end").strip()

        batch_results: List[BatchResult] = []
        current_batch: List[dict] = []
        current_meta: List[dict] = []
        batch_index = 1
        frame_index = 0
        batches_total = max(1, (max(total_frames, 1) + frames_per_batch - 1) // frames_per_batch)

        try:
            while True:
                self._raise_if_cancelled()
                ok, frame = capture.read()
                if not ok:
                    break

                timestamp_sec = frame_index / fps
                current_batch.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": encode_frame_as_data_uri(frame, max_edge, jpeg_quality)},
                    }
                )
                current_meta.append({"frame_index": frame_index, "timestamp_sec": timestamp_sec})
                frame_index += 1

                if len(current_batch) < frames_per_batch:
                    self._call_in_ui(
                        self._set_progress,
                        frame_index,
                        max(total_frames, 1),
                        f"フレーム抽出中: {frame_index}/{max(total_frames, 1)}",
                    )
                    continue

                result = self._submit_batch(
                    client=client,
                    model_key=model_key,
                    video_path=video_path,
                    batch_index=batch_index,
                    batches_total=batches_total,
                    frame_payload=current_batch,
                    frame_meta=current_meta,
                    prompt_text=prompt_text,
                    max_tokens=max_tokens,
                )
                batch_results.append(result)
                self._call_in_ui(
                    self._set_progress,
                    frame_index,
                    max(total_frames, 1),
                    f"解析中: {frame_index}/{max(total_frames, 1)} フレーム処理済み",
                )
                current_batch = []
                current_meta = []
                batch_index += 1

            if current_batch:
                result = self._submit_batch(
                    client=client,
                    model_key=model_key,
                    video_path=video_path,
                    batch_index=batch_index,
                    batches_total=batches_total,
                    frame_payload=current_batch,
                    frame_meta=current_meta,
                    prompt_text=prompt_text,
                    max_tokens=max_tokens,
                )
                batch_results.append(result)
                self._call_in_ui(
                    self._set_progress,
                    frame_index,
                    max(total_frames, 1),
                    f"解析中: {frame_index}/{max(total_frames, 1)} フレーム処理済み",
                )

            self._append_summary_log(frame_index, len(batch_results))
            return batch_results
        finally:
            capture.release()

    def _append_summary_log(self, frame_count: int, batch_count: int) -> None:
        self._call_in_ui(
            self._append_log,
            f"全フレーム送信完了: {frame_count} フレーム / {batch_count} バッチ",
        )

    def _submit_batch(
        self,
        *,
        client: LMStudioClient,
        model_key: str,
        video_path: Path,
        batch_index: int,
        batches_total: int,
        frame_payload: List[dict],
        frame_meta: List[dict],
        prompt_text: str,
        max_tokens: int,
    ) -> BatchResult:
        self._raise_if_cancelled()
        frame_range = f"{frame_meta[0]['frame_index']}-{frame_meta[-1]['frame_index']}"
        self._call_in_ui(
            self._append_log,
            f"Batch {batch_index}/{batches_total} を送信中 (frames {frame_range})",
        )

        frame_lines = []
        for idx, meta in enumerate(frame_meta, start=1):
            frame_lines.append(
                f"Image {idx} = frame {meta['frame_index']} at {format_seconds(meta['timestamp_sec'])}"
            )

        user_text = (
            f"動画ファイル: {video_path.name}\n"
            f"以下は連続した {len(frame_meta)} 枚のフレームです。画像の順番はタイムライン順です。\n"
            "画像の対応は次の通りです:\n"
            f"{chr(10).join(frame_lines)}\n\n"
            "各フレームについて1行ずつ日本語で説明し、その後にこの区間全体の要約を書いてください。\n"
            f"追加指示:\n{prompt_text}\n\n"
            "出力形式:\n"
            "- Frame 123 (00:01.234): 説明\n"
            "- Frame 124 (00:01.267): 説明\n"
            "Chunk summary: 区間全体の説明"
        )

        user_content = [{"type": "text", "text": user_text}]
        user_content.extend(frame_payload)
        system_prompt = (
            "You are a careful video frame analyst. "
            "Describe only what is visible in the provided images. "
            "Do not invent unseen details. Always answer in Japanese."
        )
        analysis = client.chat_completion(
            model_key=model_key,
            system_prompt=system_prompt,
            user_content=user_content,
            max_tokens=max_tokens,
            timeout=360,
        )
        self._call_in_ui(
            self._append_log,
            f"Batch {batch_index}/{batches_total} 応答受信",
        )
        return BatchResult(
            batch_index=batch_index,
            frame_start=frame_meta[0]["frame_index"],
            frame_end=frame_meta[-1]["frame_index"],
            analysis=analysis,
        )

    def _build_final_summary(
        self,
        client: LMStudioClient,
        model_key: str,
        video_path: Path,
        batch_results: List[BatchResult],
    ) -> str:
        if not batch_results:
            raise LMStudioError("解析対象のフレームがありませんでした。")

        groups = [
            {
                "label": f"frames {result.frame_start}-{result.frame_end}",
                "text": result.analysis,
            }
            for result in batch_results
        ]
        level = 1

        while len(groups) > 1:
            next_groups = []
            for offset in range(0, len(groups), SUMMARY_GROUP_SIZE):
                self._raise_if_cancelled()
                chunk = groups[offset : offset + SUMMARY_GROUP_SIZE]
                joined = "\n\n".join(
                    f"[{item['label']}]\n{item['text']}"
                    for item in chunk
                )
                prompt = (
                    f"動画 {video_path.name} の部分要約を統合してください。\n"
                    "書いてほしい内容:\n"
                    "1. 何が写っているかの全体像\n"
                    "2. 時系列で起きている変化\n"
                    "3. 一瞬だけ見えるものや不確実な点\n"
                    "4. 断定しにくい点はその旨\n\n"
                    f"{joined}"
                )
                summary = client.chat_completion(
                    model_key=model_key,
                    system_prompt=(
                        "You merge video frame analyses into a faithful summary. "
                        "Keep the answer in Japanese and avoid adding facts not supported by the input."
                    ),
                    user_content=[{"type": "text", "text": prompt}],
                    max_tokens=max(256, int(self.max_tokens_var.get())),
                    timeout=240,
                )
                label = f"level{level + 1}:{chunk[0]['label']}..{chunk[-1]['label']}"
                next_groups.append({"label": label, "text": summary})
                self._call_in_ui(
                    self._append_log,
                    f"統合要約を作成しました: {label}",
                )
            groups = next_groups
            level += 1

        final_prompt = (
            f"動画 {video_path.name} の最終要約を作成してください。\n"
            "必ず日本語で、次の観点をまとめてください。\n"
            "1. 動画全体で何が写っているか\n"
            "2. フレームを追って見える変化\n"
            "3. 一瞬だけ現れるものや見落としやすい点\n"
            "4. 不確実な点や断定できない点\n\n"
            f"{groups[0]['text']}"
        )
        return client.chat_completion(
            model_key=model_key,
            system_prompt=(
                "You write a faithful final video summary in Japanese. "
                "Stay grounded in the provided analyses and avoid unsupported claims."
            ),
            user_content=[{"type": "text", "text": final_prompt}],
            max_tokens=max(256, int(self.max_tokens_var.get())),
            timeout=240,
        )

    def _save_outputs(
        self,
        video_path: Path,
        model_key: str,
        batch_results: List[BatchResult],
        final_summary: str,
    ) -> Dict[str, Path]:
        timestamp = datetime.now().isoformat(timespec="seconds")
        summary_path = video_path.with_name(f"{video_path.stem}_lmstudio_summary.txt")
        json_path = video_path.with_name(f"{video_path.stem}_lmstudio_analysis.json")

        text_lines = [
            f"Video: {video_path}",
            f"Model: {model_key}",
            f"Generated: {timestamp}",
            "",
            "===== Overall Summary =====",
            final_summary,
            "",
            "===== Batch Analyses =====",
        ]
        for item in batch_results:
            text_lines.extend(
                [
                    "",
                    f"[Batch {item.batch_index} | frames {item.frame_start}-{item.frame_end}]",
                    item.analysis,
                ]
            )

        summary_path.write_text("\n".join(text_lines), encoding="utf-8")

        payload = {
            "video_path": str(video_path),
            "model": model_key,
            "generated_at": timestamp,
            "batch_results": [
                {
                    "batch_index": item.batch_index,
                    "frame_start": item.frame_start,
                    "frame_end": item.frame_end,
                    "analysis": item.analysis,
                }
                for item in batch_results
            ],
            "final_summary": final_summary,
        }
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"summary": summary_path, "json": json_path}


def main() -> None:
    app = VideoDescriberApp()
    app.mainloop()


if __name__ == "__main__":
    main()
