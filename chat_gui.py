import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import threading
import re
import bright_area_detector

# Define styles to match main_gui
BG_COLOR = "#2E3F5B"
FG_COLOR = "#EAEAEA"
SELECT_BG = "#4A6A9B"
FRAME_BG = "#263347"
ENTRY_BG = "#3A4D6B"

class ChatTab(ttk.Frame):
    def __init__(self, parent, app=None):
        super().__init__(parent)
        self.configure(style="TFrame")
        self.conversation_history = []
        self.app = app  # Reference to main application
        self._setup_ui()

    def _setup_ui(self):
        self.history_frame = ttk.Frame(self)
        self.history_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.history_area = ScrolledText(
            self.history_frame, 
            state='disabled', 
            wrap=tk.WORD, 
            font=("Segoe UI", 10),
            bg=FRAME_BG,
            fg=FG_COLOR,
            insertbackground=FG_COLOR, # caret color
            bd=0,
            highlightthickness=1,
            highlightbackground=SELECT_BG
        )
        self.history_area.pack(fill=tk.BOTH, expand=True)

        # Tag configuration for colors
        self.history_area.tag_config("user", foreground="#87CEEB", font=("Segoe UI", 10, "bold"))
        self.history_area.tag_config("ai", foreground="#98FB98", font=("Segoe UI", 10, "bold"))
        self.history_area.tag_config("system", foreground="#AAAAAA", font=("Segoe UI", 9, "italic"))
        self.history_area.tag_config("error", foreground="#FF6B6B")
        # Link tag for clickable shortcuts
        self.history_area.tag_config("link", foreground="#FFD700", underline=True, font=("Segoe UI", 10, "bold"))
        self.history_area.tag_bind("link", "<Enter>", lambda e: self.history_area.configure(cursor="hand2"))
        self.history_area.tag_bind("link", "<Leave>", lambda e: self.history_area.configure(cursor=""))
        
        # Suggestions / Quick Actions Frame
        # Feature explanation label
        self.suggestion_label = ttk.Label(self, text="💡 クリックしてAIに質問:", font=("Segoe UI", 15), foreground="#AAAAAA")
        self.suggestion_label.pack(fill=tk.X, padx=12, pady=(5, 2))

        self.suggestion_frame_1 = ttk.Frame(self)
        self.suggestion_frame_1.pack(fill=tk.X, padx=10, pady=(0, 2))
        
        self.suggestion_frame_2 = ttk.Frame(self)
        self.suggestion_frame_2.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        suggestions_1 = [
            ("🔰 基本操作", "このアプリの基本的な使い方を教えてください"),
            ("📂 動画から流星解析", "動画を追加して解析する手順を教えて"),
            ("📹 RTSP", "RTSPストリームの使い方と設定方法は？"),
        ]
        
        suggestions_2 = [
            ("🎭 マスク作成", "マスク作成機能（RTSPマスク、検出マスク、プレートソルブマスク）について教えて"),
            ("📊 解析機能", "解析タブでできることを教えて")
        ]
        
        def create_btn(parent, label, query):
            btn = tk.Button(
                parent,
                text=label,
                command=lambda q=query: self.send_quick_message(q),
                bg=ENTRY_BG,
                fg="#EAEAEA",
                activebackground=SELECT_BG,
                activeforeground="#FFFFFF",
                font=("Segoe UI", 11),
                relief="flat",
                bd=0,
                padx=8,
                pady=2,
                cursor="hand2"
            )
            btn.pack(side=tk.LEFT, padx=(0, 5))
            
            # Hover effect
            def on_enter(e, b=btn):
                b.configure(bg=SELECT_BG)
            def on_leave(e, b=btn):
                b.configure(bg=ENTRY_BG)
                
            btn.bind("<Enter>", on_enter)
            btn.bind("<Leave>", on_leave)

        for label, query in suggestions_1:
            create_btn(self.suggestion_frame_1, label, query)
            
        for label, query in suggestions_2:
            create_btn(self.suggestion_frame_2, label, query)
        
        # Input area container
        input_container = ttk.Frame(self)
        input_container.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Input field
        self.input_field = ttk.Entry(input_container) # Style is handled by TEntry in main_gui
        self.input_field.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.input_field.bind("<Return>", self.send_message)
        
        # Send button
        self.send_btn = ttk.Button(input_container, text="送信", command=self.send_message)
        self.send_btn.pack(side=tk.RIGHT, padx=(5, 0))

        # Clear button
        self.clear_btn = ttk.Button(input_container, text="🗑️", width=3, command=self.clear_history)
        self.clear_btn.pack(side=tk.RIGHT, padx=(5, 0))

        # Status Label
        self.status_label = ttk.Label(self, text="", font=("Segoe UI", 9), foreground="#87CEEB")
        self.status_label.pack(fill=tk.X, padx=12, pady=(0, 10))

        # Initial message
        self.append_message("System", "Let's ask the AI about how to use this app.\nこのアプリの使い方についてAIに聞いてみましょう。", "system")

        # System prompt with all knowledge included
        self.system_prompt = """あなたは「Meteor Detector」（流星検出アプリ）のアシスタントです。

【重要なルール】
- 質問されたことにのみ簡潔に回答してください
- 聞かれていない機能や詳細は説明しないでください
- 回答は1〜5行程度を目安にしてください
- 挨拶には挨拶だけで返してください
- 箇条書きは最小限にしてください

【アプリの基本】
- 動画ファイル（MP4/AVI/MOV）やRTSPストリームから流星を自動検出するアプリです
- 「ソース選択」タブで動画ファイルやフォルダをドラッグ＆ドロップして追加します
- 検出結果は「meteor」フォルダに保存されます
- 「解析」タブでは検出された動画の比較明合成、録画データのタイムラプス、動画の連結などを行うことができます

【動画の追加と解析手順】
1. 「ソース選択」タブに動画ファイル/フォルダをドラッグ＆ドロップ、またはRTSP URLを入力して追加します。
2. 誤検知（木や建物など）を防ぐため、事前にマスク設定を行うことを推奨します。
3. 「解析」タブで「開始」ボタンを押すと解析が始まります。

【検出結果の保存先】
検出結果は「meteor」フォルダに保存されます。非流星と判定されたものは「not_meteor」フォルダに入ります。保存先のフォルダは「保存設定」タブで変更できます。

【解析機能】
「解析」タブでは、検出された流星の.txtファイルをドロップして軌道可視化、比較明合成画像/動画の作成、長時間輝線マップの作成などができます。

【詳細設定】
⚙️タブの詳細設定で感度調整ができます。Min Line Length（最小線長）を上げると偽陽性が減ります。Meteor Probability（流星確率）でCNNの判定閾値を調整できます。

【RTSPストリーム】
RTSPストリーム（ライブカメラ）を使う場合、URLを入力して追加します。(追加ボタンを忘れずに押してください。)外部GPUがない場合はCPU負荷が高くなることがあります。録画時間制限も設定可能です。

【定期スキャン機能】
定期スキャン機能は、指定したフォルダを一定間隔で監視し、新しいファイルを自動的に解析します。Atom Cam等のネットワークカメラと連携する際に便利です。

【ショートカットボタン機能】
ユーザーの質問に対して、以下の操作が役立つと判断した場合は、回答の最後に対応するマーカーを追加してください。
マーカーはユーザーには表示されず、アプリがボタンを表示するために使用します。複数のマーカーを同時に使用可能です。
- ソース選択タブ（動画追加の説明時）: [SHORTCUT:SOURCE]
- 開始ボタン（解析開始の説明時）: [SHORTCUT:START]
- RTSP入力欄（RTSPストリームの説明時）: [SHORTCUT:RTSP]
- RTSPマスク作成（RTSPマスクの説明時）: [SHORTCUT:MASK_RTSP]
- 検出マスク作成（マスク作成の説明や、検出の手順の中で出てきた時）: [SHORTCUT:MASK_DETECTION]
- プレートソルブマスク作成（マスク作成の説明時）: [SHORTCUT:MASK_PLATESOLVE]
- 解析タブの機能（比較明合成、タイムラプスなど）について: [SHORTCUT:ANALYSIS_ACTIONS]

【マスク作成機能】
木や建物など、誤検知の原因となる領域を除外するためにマスクを作成できます。
- RTSPマスク: RTSPストリームの映像を使ってマスクを作成します。
- 検出マスク: 通常の検出処理で除外する領域を指定します。
- プレートソルブマスク: 星座判定（プレートソルブ）を行う際に計算から除外する領域を指定します。

挨拶や一般的な質問にはマーカーを付けないでください。"""

    def send_quick_message(self, text):
        if self.input_field.cget('state') == 'disabled':
            return
        self.input_field.delete(0, tk.END)
        self.input_field.insert(0, text)
        self.send_message()

    def clear_history(self):
        """Clear conversation history and text area."""
        self.conversation_history = []
        self.history_area.configure(state='normal')
        self.history_area.delete("1.0", tk.END)
        self.history_area.configure(state='disabled')
        # Re-add initial system message if desired, or just leave empty
        self.append_message("System", "会話履歴をクリアしました。", "system")

    def send_message(self, event=None):
        msg = self.input_field.get()
        if not msg.strip():
            return
            
        self.append_message("User", msg, "user")
        self.input_field.delete(0, tk.END)
        self.input_field.config(state='disabled')
        self.send_btn.config(state='disabled')
        
        # Thread generation
        threading.Thread(target=self._get_response, args=(msg,), daemon=True).start()

    def append_message(self, sender, message, tag=None):
        self.history_area.configure(state='normal')
        self.history_area.insert(tk.END, f"\n[{sender}]:\n", tag)
        self.history_area.insert(tk.END, f"{message}\n")
        self.history_area.configure(state='disabled')
        self.history_area.see(tk.END)

    def _update_status(self, text):
        self.after(0, lambda: self.status_label.config(text=text))

    def _prepare_streaming_ui(self):
        self.history_area.configure(state='normal')
        self.history_area.insert(tk.END, f"\n[AI]:\n", "ai")
        self.history_area.configure(state='disabled')
        self.history_area.see(tk.END)

    def _append_stream_chunk(self, chunk):
        self.history_area.configure(state='normal')
        self.history_area.insert(tk.END, chunk)
        self.history_area.configure(state='disabled')
        self.history_area.see(tk.END)

    def _finalize_streaming_ui(self):
        self.history_area.configure(state='normal')
        self.history_area.insert(tk.END, "\n")
        self.history_area.configure(state='disabled')
        self.history_area.see(tk.END)
        
        self._update_status("")
        self.input_field.config(state='normal')
        self.send_btn.config(state='normal')
        self.input_field.focus()

    def _parse_shortcuts_from_response(self, response):
        """Parse AI response for shortcut markers and return clean response + detected shortcuts."""
        shortcuts = {
            'source': False,
            'start': False,
            'rtsp': False,
            'mask_rtsp': False,
            'mask_detection': False,
            'mask_ps': False,
            'analysis_actions': False
        }
        
        # Detect shortcut markers
        if '[SHORTCUT:SOURCE]' in response:
            shortcuts['source'] = True
        if '[SHORTCUT:START]' in response:
            shortcuts['start'] = True
        if '[SHORTCUT:RTSP]' in response:
            shortcuts['rtsp'] = True
        if '[SHORTCUT:MASK_RTSP]' in response:
            shortcuts['mask_rtsp'] = True
        if '[SHORTCUT:MASK_DETECTION]' in response:
            shortcuts['mask_detection'] = True
        if '[SHORTCUT:MASK_PLATESOLVE]' in response:
            shortcuts['mask_ps'] = True
        if '[SHORTCUT:ANALYSIS_ACTIONS]' in response:
            shortcuts['analysis_actions'] = True
        
        # Remove markers from displayed response
        clean_response = re.sub(r'\[SHORTCUT:(SOURCE|START|RTSP|MASK_RTSP|MASK_DETECTION|MASK_PLATESOLVE|ANALYSIS_ACTIONS)\]', '', response)
        clean_response = clean_response.strip()
        
        return clean_response, shortcuts

    def _create_shortcut_button(self, text, command, text_color):
        """Helper to create a styled button for shortcuts."""
        return tk.Button(
            self.history_area,
            text=text,
            command=command,
            bg=ENTRY_BG,
            fg=text_color,
            activebackground=SELECT_BG,
            activeforeground=text_color,
            font=("Segoe UI", 9, "bold"),
            relief="raised",
            bd=1,
            cursor="hand2",
            padx=10,
            pady=2
        )

    def _append_source_shortcut(self):
        """Append a clickable shortcut button to navigate to source selection tab."""
        if not self.app:
            return
        
        self.history_area.configure(state='normal')
        self.history_area.insert(tk.END, "\n")
        
        btn = self._create_shortcut_button(
            "📂 ソース選択タブを開く", 
            self._navigate_to_source,
            "#FFD700"
        )
        
        self.history_area.window_create(tk.END, window=btn, padx=5, pady=5)
        self.history_area.insert(tk.END, " ← クリックでドラッグ＆ドロップエリアに移動\n", "system")
        self.history_area.configure(state='disabled')
        self.history_area.see(tk.END)

    def _append_start_button_shortcut(self):
        """Append a clickable shortcut to highlight the start button."""
        if not self.app:
            return
        
        self.history_area.configure(state='normal')
        self.history_area.insert(tk.END, "\n")
        
        btn = self._create_shortcut_button(
            "▶ 開始ボタンを表示", 
            self._navigate_to_start,
            "#90EE90"
        )
        
        self.history_area.window_create(tk.END, window=btn, padx=5, pady=5)
        self.history_area.insert(tk.END, " ← クリックで開始ボタンをハイライト表示\n", "system")
        self.history_area.configure(state='disabled')
        self.history_area.see(tk.END)

    def _append_rtsp_shortcut(self):
        """Append a clickable shortcut to highlight the RTSP URL entry."""
        if not self.app:
            return
        
        self.history_area.configure(state='normal')
        self.history_area.insert(tk.END, "\n")
        
        btn = self._create_shortcut_button(
            "📹 RTSP入力欄を表示", 
            self._navigate_to_rtsp,
            "#87CEEB"
        )
        
        self.history_area.window_create(tk.END, window=btn, padx=5, pady=5)
        self.history_area.insert(tk.END, " ← クリックでURL入力欄をハイライト表示\n", "system")
        self.history_area.configure(state='disabled')
        self.history_area.see(tk.END)

    def _navigate_to_source(self):
        """Navigate to source selection and highlight drop area."""
        if self.app and hasattr(self.app, 'navigate_to_source_drop_area'):
            self.app.navigate_to_source_drop_area()

    def _navigate_to_start(self):
        """Highlight the start button."""
        if self.app and hasattr(self.app, 'navigate_to_start_button'):
            self.app.navigate_to_start_button()

    def _navigate_to_rtsp(self):
        """Navigate to RTSP entry and highlight it."""
        if self.app and hasattr(self.app, 'navigate_to_rtsp_entry'):
            self.app.navigate_to_rtsp_entry()

    def _navigate_to_rtsp_mask(self):
        if self.app and hasattr(self.app, 'navigate_to_rtsp_mask_button'):
            self.app.navigate_to_rtsp_mask_button()

    def _navigate_to_detection_mask(self):
        if self.app and hasattr(self.app, 'navigate_to_detection_mask_button'):
            self.app.navigate_to_detection_mask_button()

    def _navigate_to_ps_mask(self):
        if self.app and hasattr(self.app, 'navigate_to_ps_mask_button'):
            self.app.navigate_to_ps_mask_button()

    def _append_rtsp_mask_shortcut(self):
        if not self.app: return
        self.history_area.configure(state='normal')
        self.history_area.insert(tk.END, "\n")
        btn = self._create_shortcut_button("🎭 RTSPマスク作成を表示", self._navigate_to_rtsp_mask, "#FFD700")
        self.history_area.window_create(tk.END, window=btn, padx=5, pady=5)
        self.history_area.insert(tk.END, " ← RTSPマスク作成ボタンを表示\n", "system")
        self.history_area.configure(state='disabled')
        self.history_area.see(tk.END)

    def _append_detection_mask_shortcut(self):
        if not self.app: return
        self.history_area.configure(state='normal')
        self.history_area.insert(tk.END, "\n")
        btn = self._create_shortcut_button("🎭 検出マスク作成を表示", self._navigate_to_detection_mask, "#FFD700")
        self.history_area.window_create(tk.END, window=btn, padx=5, pady=5)
        self.history_area.insert(tk.END, " ← 検出マスク作成ボタンを表示\n", "system")
        self.history_area.configure(state='disabled')
        self.history_area.see(tk.END)

    def _append_ps_mask_shortcut(self):
        if not self.app: return
        self.history_area.configure(state='normal')
        self.history_area.insert(tk.END, "\n")
        btn = self._create_shortcut_button("🎭 プレートソルブマスク作成を表示", self._navigate_to_ps_mask, "#FFD700")
        self.history_area.window_create(tk.END, window=btn, padx=5, pady=5)
        self.history_area.insert(tk.END, " ← プレートソルブ用マスク作成ボタンを表示\n", "system")
        self.history_area.configure(state='disabled')
        self.history_area.configure(state='disabled')
        self.history_area.see(tk.END)

    def _navigate_to_analysis_actions(self):
        if self.app and hasattr(self.app, 'navigate_to_analysis_actions'):
            self.app.navigate_to_analysis_actions()

    def _append_analysis_actions_shortcut(self):
        if not self.app: return
        self.history_area.configure(state='normal')
        self.history_area.insert(tk.END, "\n")
        btn = self._create_shortcut_button("📊 解析機能ボタンを表示", self._navigate_to_analysis_actions, "#FFD700")
        self.history_area.window_create(tk.END, window=btn, padx=5, pady=5)
        self.history_area.insert(tk.END, " ← 解析タブの各機能ボタンをハイライト表示\n", "system")
        self.history_area.configure(state='disabled')
        self.history_area.see(tk.END)

    def _get_response(self, user_msg):
        self._update_status("準備中...")
        
        # Collect streaming chunks to parse markers after completion
        self.streaming_chunks = []
        
        # UI側でAIのメッセージ開始枠を作る
        self.after(0, self._prepare_streaming_ui)
        
        def stream_handler(chunk):
            self.streaming_chunks.append(chunk)
            # Filter out shortcut markers from displayed chunks
            display_chunk = re.sub(r'\[SHORTCUT:(SOURCE|START|RTSP|MASK_RTSP|MASK_DETECTION|MASK_PLATESOLVE|ANALYSIS_ACTIONS)\]', '', chunk)
            if display_chunk:
                self.after(0, self._append_stream_chunk, display_chunk)
            
        try:
            # ストリーミング実行（戻り値の全文は無視して、ストリームで表示済みのものを正とする）
            # 履歴の過去10件を取得
            history_context = self.conversation_history[-10:] if len(self.conversation_history) > 10 else self.conversation_history
            
            full_response = bright_area_detector.generate_response(
                user_prompt=user_msg,
                system_prompt=self.system_prompt,
                history=history_context,
                status_callback=self._update_status,
                stream_callback=stream_handler
            )
            
            # Parse shortcuts from full response
            clean_response, shortcuts = self._parse_shortcuts_from_response(full_response)
            
            # 成功したら履歴に追加 (clean response without markers)
            self.conversation_history.append({"role": "user", "content": user_msg})
            self.conversation_history.append({"role": "assistant", "content": clean_response})
            
            # Add shortcut links based on AI's decision
            if self.app:
                if shortcuts['source']:
                    self.after(0, self._append_source_shortcut)
                if shortcuts['start']:
                    self.after(0, self._append_start_button_shortcut)
                if shortcuts['rtsp']:
                    self.after(0, self._append_rtsp_shortcut)
                if shortcuts['mask_rtsp']:
                    self.after(0, self._append_rtsp_mask_shortcut)
                if shortcuts['mask_detection']:
                    self.after(0, self._append_detection_mask_shortcut)
                if shortcuts['mask_ps']:
                    self.after(0, self._append_ps_mask_shortcut)
                if shortcuts['analysis_actions']:
                    self.after(0, self._append_analysis_actions_shortcut)
                
        except Exception as e:
            # エラー時は追記
            self.after(0, lambda: self.append_message("System", f"Error: {e}", "error"))
        
        # 完了処理
        self.after(0, self._finalize_streaming_ui)



def create_tab(parent, app=None):
    return ChatTab(parent, app=app)
