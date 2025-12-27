"""
自動時間制限更新モジュール

このモジュールは、定期的に時間制限設定を自動更新する機能を提供します。
毎日午前11時に実行され、現在位置の日の出・日の入り時刻に基づいて
時間制限の開始・終了時刻を自動的に更新します。
"""

import threading
import time
from datetime import datetime, timedelta
from typing import Callable, Optional
import location_utils
import sun_times


class AutoTimeUpdater:
    """時間制限の自動更新を管理するクラス"""
    
    def __init__(self):
        self.enabled = False
        self.update_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.last_update_date: Optional[datetime] = None
        self.update_callback: Optional[Callable] = None
        self.log_callback: Optional[Callable[[str], None]] = None
        
    def set_update_callback(self, callback: Callable):
        """
        時刻更新時に呼び出されるコールバックを設定
        
        Args:
            callback: (start_hour, start_min, end_hour, end_min) を引数に取る関数
        """
        self.update_callback = callback
        
    def set_log_callback(self, callback: Callable[[str], None]):
        """
        ログ出力用のコールバックを設定
        
        Args:
            callback: メッセージ文字列を引数に取る関数
        """
        self.log_callback = callback
    
    def _log(self, message: str):
        """ログ出力"""
        if self.log_callback:
            self.log_callback(message)
        print(f"[AutoTimeUpdater] {message}")
    
    def start(self):
        """自動更新を開始"""
        if self.enabled and self.update_thread and self.update_thread.is_alive():
            self._log("自動更新は既に実行中です。")
            return
        
        self.enabled = True
        self.stop_event.clear()
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        self._log("時間制限の自動更新を開始しました。")
    
    def stop(self):
        """自動更新を停止"""
        self.enabled = False
        self.stop_event.set()
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=2)
        self._log("時間制限の自動更新を停止しました。")
    
    def _should_update_now(self) -> bool:
        """
        現在が更新タイミングかどうかを判定
        
        Returns:
            bool: 午前11時で、かつ今日まだ更新していない場合True
        """
        now = datetime.now()
        
        # 午前11時かどうかチェック
        if now.hour != 11:
            return False
        
        # 今日既に更新済みかチェック
        if self.last_update_date and self.last_update_date.date() == now.date():
            return False
        
        return True
    
    def _perform_update(self):
        """時間制限の更新を実行"""
        try:
            self._log("時間制限を自動更新中...")
            
            # 現在位置を取得
            lat, lon = location_utils.get_current_location()
            self._log(f"位置情報取得: 緯度={lat:.6f}, 経度={lon:.6f}")
            
            # 日の出・日の入り時刻を計算
            times = sun_times.get_sun_times(lat, lon)
            
            # 推奨される夜間時間帯を計算
            period = sun_times.compute_night_period(lat, lon)
            start_dt = period.get('start')
            end_dt = period.get('end')
            
            if start_dt and end_dt and self.update_callback:
                sh, sm = start_dt.hour, start_dt.minute
                eh, em = end_dt.hour, end_dt.minute
                
                # コールバックで時刻を更新
                self.update_callback(sh, sm, eh, em)
                
                self._log(f"時間制限を更新: 開始={sh:02d}:{sm:02d}, 終了={eh:02d}:{em:02d}")
                self.last_update_date = datetime.now()
            else:
                self._log("時間制限の計算に失敗しました。")
                
        except Exception as e:
            self._log(f"時間制限の自動更新中にエラーが発生しました: {e}")
    
    def _update_loop(self):
        """更新ループのメインロジック"""
        self._log("自動更新ループを開始しました。毎日午前11時に更新します。")
        
        while self.enabled and not self.stop_event.is_set():
            try:
                if self._should_update_now():
                    self._perform_update()
                
                # 1分ごとにチェック
                self.stop_event.wait(timeout=60)
                
            except Exception as e:
                self._log(f"更新ループでエラーが発生しました: {e}")
                self.stop_event.wait(timeout=60)
        
        self._log("自動更新ループを終了しました。")
    
    def force_update(self):
        """手動で即座に更新を実行"""
        if not self.enabled:
            self._log("自動更新が無効です。")
            return
        
        threading.Thread(target=self._perform_update, daemon=True).start()
