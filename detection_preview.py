"""
検出結果プレビュー＆編集モジュール

AI検出結果の確認、再検出、手動修正を行い、
確定したデータで合成処理を開始するためのウィンドウ。
"""

import tkinter as tk
from tkinter import ttk, Canvas, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
from typing import List, Tuple, Optional, Callable, Dict
import threading
import time
import math
import cv2
import numpy as np

class DetectionPreviewWindow(tk.Toplevel):
    
    # アイテムの表示高さ（概算px）
    ITEM_HEIGHT = 380

    def __init__(self, parent, start_synthesis_callback: Callable[[Dict], None]):
        super().__init__(parent)
        self.title("検出結果の確認と編集")
        self.geometry("1000x800")
        self.start_synthesis_callback = start_synthesis_callback
        
        # 仮想スクロール管理
        self.current_top_index = 0
        self.visible_items_count = 3  # 初期値
        
        # データ管理
        # key: filename, value: {'boxes': [(x1,y1,x2,y2)...], 'image_path': str, 'image_shape': (h, w)}
        self.results: Dict[str, dict] = {}
        self.photo_images = {}  # メモリ保持用

        # 順序と進捗管理
        self.item_order: List[str] = []
        self.analysis_start_time: Optional[float] = None
        self.total_items: int = 0
        self.processed_items: int = 0
        
        # UIセットアップ
        self.setup_ui()
        
        # モーダル設定（親ウィンドウ操作禁止ではないが、最前面に）
        self.transient(parent)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
    def setup_ui(self):
        """UI構築"""
        # 上部コントロール
        top_frame = ttk.Frame(self)
        top_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(top_frame, text="検出結果を確認・修正してください", font=("", 12, "bold")).pack(side=tk.LEFT)
        
        # 進捗表示フレーム
        progress_frame = ttk.Frame(self)
        progress_frame.pack(fill=tk.X, padx=10, pady=5)

        self.progress = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, mode='determinate')
        self.progress.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=(0, 10))
        self.status_label = ttk.Label(progress_frame, text="待機中", width=15)
        self.status_label.pack(side=tk.LEFT)

        time_frame = ttk.Frame(self)
        time_frame.pack(fill=tk.X, padx=10, pady=2)
        self.eta_label = ttk.Label(time_frame, text="ETA: --:--:--", width=20)
        self.eta_label.pack(side=tk.LEFT)
        self.elapsed_label = ttk.Label(time_frame, text="経過: 00:00:00", width=20)
        self.elapsed_label.pack(side=tk.LEFT)
        
        # 合成開始ボタン
        self.btn_start = ttk.Button(top_frame, text="修正を確定して合成を開始", command=self.on_start_synthesis, state=tk.DISABLED)
        self.btn_start.pack(side=tk.RIGHT, padx=5)
        
        # 表示エリアのコンテナ
        container = ttk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # スクロールバー
        self.scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=self._on_scroll_command)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # アイテム表示用フレーム（Canvasは使わない）
        self.items_frame = ttk.Frame(container)
        self.items_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # イベントバインド
        self.items_frame.bind("<Configure>", self._on_resize)
        self.bind_all("<MouseWheel>", self._on_mousewheel)  # Windows
        # Linux/Mac用
        self.bind_all("<Button-4>", self._on_mousewheel)
        self.bind_all("<Button-5>", self._on_mousewheel)
        
    def _on_resize(self, event):
        """ウィンドウサイズ変更時に表示可能数を再計算"""
        height = event.height
        if height > 10:
            # 少し余裕を持たせて多めに描画する
            self.visible_items_count = math.ceil(height / self.ITEM_HEIGHT) + 1
            self.refresh_view()

    def _on_mousewheel(self, event):
        """マウスホイール処理"""
        if not self.item_order:
            return
            
        # Windows: event.delta, Linux: event.num
        delta = 0
        if hasattr(event, 'delta') and event.delta != 0:
            delta = event.delta
        elif hasattr(event, 'num'):
            if event.num == 4: delta = 120
            elif event.num == 5: delta = -120
            
        if delta == 0: return

        # スクロール方向（正=上へ=インデックス減、負=下へ=インデックス増）
        step = -1 if delta > 0 else 1
        
        new_top = self.current_top_index + step
        self._set_scroll_pos(new_top)

    def _on_scroll_command(self, *args):
        """スクロールバーの操作"""
        if not self.item_order:
            return
            
        total = len(self.item_order)
        if total == 0: return

        if args[0] == 'moveto':
            ratio = float(args[1])
            new_top = int(total * ratio)
            self._set_scroll_pos(new_top)
        elif args[0] == 'scroll':
            unit = args[1] # 'units' or 'pages'
            amount = int(args[1])
            step = 1
            if unit == 'pages':
                step = self.visible_items_count
            
            new_top = self.current_top_index + (amount * step)
            self._set_scroll_pos(new_top)

    def _set_scroll_pos(self, new_top):
        """スクロール位置を更新してリフレッシュ"""
        total = len(self.item_order)
        max_top = max(0, total - 1) # 少なくとも1つは表示
        
        # 範囲制限
        new_top = max(0, min(new_top, total - self.visible_items_count + 1))
        # 下端の微調整: 最後のアイテムが見えるように
        if total > self.visible_items_count:
             new_top = max(0, min(new_top, total - 1)) # 簡易的に
        else:
             new_top = 0
             
        if new_top != self.current_top_index:
            self.current_top_index = new_top
            self.refresh_view()
            self._update_scrollbar_display()

    def _update_scrollbar_display(self):
        """スクロールバーのつまみ位置を更新"""
        total = len(self.item_order)
        if total == 0:
            self.scrollbar.set(0, 1)
            return
            
        start_ratio = self.current_top_index / total
        end_ratio = min(1.0, (self.current_top_index + self.visible_items_count) / total)
        self.scrollbar.set(start_ratio, end_ratio)

    def refresh_view(self):
        """現在のスクロール位置に基づいてアイテムを表示"""
        # 既存ウィジェットの破棄
        for child in self.items_frame.winfo_children():
            child.destroy()
            
        # 画像参照のクリア（表示されなくなった分を解放）
        self.photo_images.clear()
        
        if not self.item_order:
            return
            
        total = len(self.item_order)
        start = self.current_top_index
        end = min(total, start + self.visible_items_count)
        
        for i in range(start, end):
            filename = self.item_order[i]
            self._create_item_widget(filename)
            
        self._update_scrollbar_display()

    def start_analysis(self, total_count: int):
        """解析開始時に呼び出す"""
        self.total_items = total_count
        self.processed_items = 0
        self.analysis_start_time = time.time()
        self.progress['maximum'] = total_count
        self.progress['value'] = 0
        self.status_label.config(text="解析中")
        self._update_time_display()

    def _update_time_display(self):
        """経過時間とETAを更新"""
        if self.analysis_start_time is None:
            return
            
        elapsed = time.time() - self.analysis_start_time
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        self.elapsed_label.config(text=f"経過: {elapsed_str}")
        
        if self.processed_items > 0 and self.total_items > 0:
            avg_time = elapsed / self.processed_items
            remaining = (self.total_items - self.processed_items) * avg_time
            if remaining > 0:
                eta_str = time.strftime("%H:%M:%S", time.gmtime(remaining))
                self.eta_label.config(text=f"ETA: {eta_str}")
            else:
                self.eta_label.config(text="ETA: 完了")
        elif self.processed_items == self.total_items and self.total_items > 0:
             self.eta_label.config(text="ETA: 完了")

    def finalize_analysis(self):
        """解析完了時に呼び出す。未検出画像を先頭に移動"""
        # 未検出画像を先頭に、検出済み画像を後ろにソート
        no_detection = []
        with_detection = []
        
        for filename in self.item_order:
            if filename in self.results:
                if len(self.results[filename]['boxes']) == 0:
                    no_detection.append(filename)
                else:
                    with_detection.append(filename)
        
        # 新しい順序: 未検出 -> 検出あり
        new_order = no_detection + with_detection
        
        # UIを再構築（順序を反映）
        self._rebuild_ui(new_order)
        
        # 状態更新
        self.item_order = new_order
        self.eta_label.config(text="ETA: 完了")
        self.status_label.config(text="解析完了")
        # 最終経過時間を更新
        if self.analysis_start_time:
             elapsed = time.time() - self.analysis_start_time
             elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
             self.elapsed_label.config(text=f"経過: {elapsed_str}")


    def _rebuild_ui(self, new_order: List[str]):
        """指定された順序でUIを再構築（仮想スクロールのリフレッシュ）"""
        # 仮想スクロールでは refresh_view を呼ぶだけでよい
        # ただし scroll position をリセットするかどうか。
        # 解析完了時はトップに戻すのが自然。
        self.current_top_index = 0
        self.refresh_view()

    def add_item(self, filename: str, image_path: str, boxes: List[Tuple], reanalyze_callback: Callable):
        """
        リストにアイテムを追加（または更新）
        """
        # データ登録
        full_image = cv2.imread(image_path)
        if full_image is None:
            return
            
        height, width = full_image.shape[:2]
        self.results[filename] = {
            'boxes': boxes,
            'image_path': image_path,
            'image_shape': (height, width),
            'reanalyze_callback': reanalyze_callback
        }
        
        # 順序管理（新規追加時のみ）
        if filename not in self.item_order:
            self.item_order.append(filename)
            self.processed_items += 1
            self.progress['value'] = self.processed_items
            self._update_time_display()
            
            # 追加されたアイテムが表示範囲内ならリフレッシュ
            # 基本的には末尾に追加されるので、現在のビューが末尾なら更新が必要
            # 頻繁なリフレッシュは重いので、一定間隔or完了時のみでも良いが、
            # プレビューが見たいのですぐ反映する。
            self._update_scrollbar_display()
            
            # もし現在の表示範囲に影響するならリフレッシュ
            if self.current_top_index + self.visible_items_count >= len(self.item_order) - 1:
                self.refresh_view()
        else:
             # 更新の場合もリフレッシュ
             self.refresh_view()
        
        # 合成開始ボタン有効化
        self.btn_start.config(state=tk.NORMAL)

    def _create_item_widget(self, filename: str):
        """指定されたファイルのUIアイテムを作成・表示（packする）"""
        if filename not in self.results:
            return

        data = self.results[filename]
        boxes = data['boxes']
        image_path = data['image_path']
        reanalyze_callback = data['reanalyze_callback']

        # アイテムフレーム作成
        frame = ttk.Frame(self.items_frame, relief=tk.RIDGE, borderwidth=1)
        frame.pack(fill=tk.X, padx=5, pady=5)
        # 識別用にタグ付けなどは不要、毎回作り直しなので
        
        # 画像表示エリア（左） 固定サイズ 400x320
        # Canvas自体を固定サイズにする
        canvas_w, canvas_h = 400, 320
        img_canvas = Canvas(frame, width=canvas_w, height=canvas_h, bg="black", highlightthickness=0)
        img_canvas.pack(side=tk.LEFT, padx=5, pady=5)
        
        # 情報・操作エリア（右）
        ctrl_frame = ttk.Frame(frame)
        ctrl_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        ttk.Label(ctrl_frame, text=filename, font=("", 11, "bold")).pack(anchor="nw")
        coord_label = ttk.Label(ctrl_frame, text=self._format_boxes(boxes))
        coord_label.pack(anchor="nw", pady=5)
        
        # ボタン群
        btn_box = ttk.Frame(ctrl_frame)
        btn_box.pack(anchor="nw", pady=10)
        
        ttk.Button(btn_box, text="再検出 (AI)", 
                  command=lambda: self._handle_reanalyze(filename, reanalyze_callback)).pack(side=tk.LEFT, padx=2)
        
        btn_manual = ttk.Button(btn_box, text="手動で枠を追加", 
                   command=lambda: self._enable_manual_draw(filename, img_canvas))
        btn_manual.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(btn_box, text="枠をクリア", 
                  command=lambda: self._clear_boxes(filename, img_canvas, coord_label)).pack(side=tk.LEFT, padx=2)
        
        # 画像読み込みと描画
        full_image = cv2.imread(image_path)
        if full_image is not None:
            self._draw_image_on_canvas(filename, img_canvas, full_image, boxes)

    def _format_boxes(self, boxes):
        if not boxes:
            return "検出なし"
        return "座標:\n" + "\n".join([f"({x1},{y1},{x2},{y2})" for x1,y1,x2,y2 in boxes])

    def _draw_image_on_canvas(self, filename, canvas: Canvas, image: np.ndarray, boxes: List[Tuple]):
        """キャンバスに画像と枠を描画（固定サイズでセンタリング）"""
        h, w = image.shape[:2]
        
        # Canvasのサイズ取得（固定値を想定）
        canvas_w = int(canvas['width'])
        canvas_h = int(canvas['height'])
        
        # アスペクト比維持で収まるように計算
        scale = min(canvas_w / w, canvas_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        img_resized = cv2.resize(image, (new_w, new_h))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # PILで描画
        draw = ImageDraw.Draw(pil_img)
        margin_px = 40  # 40pxのマージン
        
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            # 正規化座標(0-1000) -> 元画像のピクセル座標
            orig_x1 = x1 * w / 1000
            orig_y1 = y1 * h / 1000
            orig_x2 = x2 * w / 1000
            orig_y2 = y2 * h / 1000
            
            # マージン適用
            orig_x1 -= margin_px
            orig_y1 -= margin_px
            orig_x2 += margin_px
            orig_y2 += margin_px
            
            # クリップ
            orig_x1 = max(0, orig_x1)
            orig_y1 = max(0, orig_y1)
            orig_x2 = min(w, orig_x2)
            orig_y2 = min(h, orig_y2)
            
            # 表示用リサイズ座標に変換
            sx1 = orig_x1 * scale
            sy1 = orig_y1 * scale
            sx2 = orig_x2 * scale
            sy2 = orig_y2 * scale
            
            draw.rectangle([sx1, sy1, sx2, sy2], outline="red", width=2)
            draw.text((sx1, max(0, sy1-15)), f"#{i+1}", fill="red")
            
        photo = ImageTk.PhotoImage(pil_img)
        self.photo_images[filename] = photo  # 参照保持
        
        # センタリング座標
        pos_x = (canvas_w - new_w) // 2
        pos_y = (canvas_h - new_h) // 2
        
        canvas.delete("all")
        canvas.create_image(pos_x, pos_y, image=photo, anchor="nw")
        
        # スケール情報とオフセットを保存（手動追加用）
        canvas.scale_factor = scale
        canvas.offset_x = pos_x
        canvas.offset_y = pos_y

    def _handle_reanalyze(self, filename, callback):
        """再検出実行（表示順序を維持）- バックグラウンドスレッドで処理"""
        data = self.results[filename]
        image_path = data['image_path']
        
        # 画像読み込み
        img = cv2.imread(image_path)
        if img is None:
            return
        
        # カーソルを待機状態に
        self.config(cursor="watch")
        self.update()
        
        def run_detection():
            """バックグラウンドで検出処理を実行"""
            try:
                # コールバック実行 (bright_area_detectorの関数を呼ぶ想定)
                result = callback(img)
                if result is None:
                    boxes = []
                else:
                    mask, boxes = result
                
                # メインスレッドでUI更新
                def update_ui():
                    if not self.winfo_exists():
                        return
                    try:
                        # データ更新
                        self.results[filename]['boxes'] = boxes
                        
                        # UI再構築（順序を維持）
                        self._rebuild_ui(self.item_order)
                    finally:
                        self.config(cursor="")
                
                self.after(0, update_ui)
                
            except Exception as e:
                # エラー時もカーソルを戻す
                def reset_cursor():
                    if self.winfo_exists():
                        self.config(cursor="")
                self.after(0, reset_cursor)
                print(f"再検出エラー: {e}")
        
        # バックグラウンドスレッドで実行
        threading.Thread(target=run_detection, daemon=True).start()

    def _clear_boxes(self, filename, canvas, label_widget):
        self.results[filename]['boxes'] = []
        image_path = self.results[filename]['image_path']
        img = cv2.imread(image_path)
        if img is not None:
             self._draw_image_on_canvas(filename, canvas, img, [])
        label_widget.config(text="検出なし")

    def _enable_manual_draw(self, filename, canvas):
        """手動描画モード有効化"""
        canvas.config(cursor="cross")
        
        # 一時的な状態保持
        self.drawing_state = {
            'start_x': 0, 'start_y': 0,
            'rect_id': None,
            'filename': filename
        }
        
        # イベントバインド
        canvas.bind("<ButtonPress-1>", lambda e: self._on_draw_start(e, canvas))
        canvas.bind("<B1-Motion>", lambda e: self._on_draw_move(e, canvas))
        canvas.bind("<ButtonRelease-1>", lambda e: self._on_draw_end(e, canvas))

    def _on_draw_start(self, event, canvas):
        self.drawing_state['start_x'] = event.x
        self.drawing_state['start_y'] = event.y
        self.drawing_state['rect_id'] = canvas.create_rectangle(
            event.x, event.y, event.x, event.y, outline="yellow", width=2
        )

    def _on_draw_move(self, event, canvas):
        if self.drawing_state['rect_id']:
            canvas.coords(self.drawing_state['rect_id'], 
                         self.drawing_state['start_x'], self.drawing_state['start_y'],
                         event.x, event.y)

    def _on_draw_end(self, event, canvas):
        # 描画終了
        canvas.config(cursor="")
        canvas.unbind("<ButtonPress-1>")
        canvas.unbind("<B1-Motion>")
        canvas.unbind("<ButtonRelease-1>")
        
        # 座標計算
        start_x, start_y = self.drawing_state['start_x'], self.drawing_state['start_y']
        end_x, end_y = event.x, event.y
        
        # 描画オフセットを考慮して画像内の座標へ変換
        off_x = getattr(canvas, 'offset_x', 0)
        off_y = getattr(canvas, 'offset_y', 0)
        
        start_x -= off_x
        start_y -= off_y
        end_x -= off_x
        end_y -= off_y
        
        # 左上・右下に正規化
        x1, x2 = sorted([start_x, end_x])
        y1, y2 = sorted([start_y, end_y])
        
        # スケール戻し
        scale = getattr(canvas, 'scale_factor', 1.0)
        if scale <= 0: return

        # 画像内座標 -> 正規化座標(0-1000)
        img_h, img_w = self.results[self.drawing_state['filename']]['image_shape']
        
        norm_x1 = int(x1 / scale / img_w * 1000)
        norm_y1 = int(y1 / scale / img_h * 1000)
        norm_x2 = int(x2 / scale / img_w * 1000)
        norm_y2 = int(y2 / scale / img_h * 1000)
        
        # 0-1000クリップ
        norm_x1 = max(0, min(1000, norm_x1))
        norm_y1 = max(0, min(1000, norm_y1))
        norm_x2 = max(0, min(1000, norm_x2))
        norm_y2 = max(0, min(1000, norm_y2))
        
        # 新しいボックスを追加
        if abs(norm_x2 - norm_x1) > 10 and abs(norm_y2 - norm_y1) > 10: # 極小は無視
            filename = self.drawing_state['filename']
            current_boxes = self.results[filename]['boxes']
            current_boxes.append((norm_x1, norm_y1, norm_x2, norm_y2))
            
            # 再描画
            self.refresh_view()
        else:
            # 描画した矩形を消す（キャンセル扱い）
            canvas.delete(self.drawing_state['rect_id'])

    def on_start_synthesis(self):
        """合成開始"""
        if not self.results:
            return
            
        # 結果データだけ渡して自分は閉じる
        # results = {filename: {'boxes': [], ...}}
        self.start_synthesis_callback(self.results)
        self.destroy()

    def on_close(self):
        # 閉じるボタンが押されたらキャンセル扱い（合成しない）
        if messagebox.askyesno("確認", "検出結果の編集を終了して、合成処理をキャンセルしますか？"):
            self.destroy()
