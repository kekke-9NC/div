# app_state.py

import threading
import queue
from typing import Optional, Dict, Any
import numpy as np

import config

# --- アプリケーション状態変数 ---

# マスク画像 (NumPy配列)
mask_image: Optional[np.ndarray] = None
plate_solve_mask_image: Optional[np.ndarray] = None

# プレートソルブ結果
# {'wcs_file': str, 'job_id': int, 'plate_solve_datetime': datetime} の形式
global_wcs_info: Optional[Dict[str, Any]] = None

# 定期スキャン関連の状態
periodic_scan_mode_enabled: bool = False
periodic_scan_directory: Optional[str] = None
periodic_scan_interval: int = config.DEFAULT_SCAN_INTERVAL

# キャンセルフラグ (複数のスレッドから参照・設定される)
cancel_flag = threading.Event()

# 進捗報告用キュー (ワーカースレッド -> GUI)
# スレッドセーフなのでロックは不要
progress_queue = queue.Queue()

# 保存オプション (GUIと連動させる想定)
save_options: Dict[str, bool] = {
    'video': config.DEFAULT_SAVE_VIDEO_CLIP,
    'cutout': config.DEFAULT_SAVE_CUTOUT_DIFF,
    'full': config.DEFAULT_SAVE_FULL_DIFF,
    'composite': config.DEFAULT_SAVE_COMPOSITE,
    'info': config.DEFAULT_SAVE_DETECTION_INFO,
}
# 注意: GUIのチェックボックスが変更された際に、この辞書も更新する処理が main_gui.py に必要です。


# --- 状態更新用のロック (参考例) ---
# 例: 複数のスレッドから同時に状態を変更する場合
# state_lock = threading.Lock()
#
# def set_mask_image(new_mask):
#     global mask_image
#     with state_lock:
#         mask_image = new_mask
#
# def get_mask_image():
#     with state_lock:
#         return mask_image.copy() if mask_image is not None else None

# --- モジュール単体でのテスト ---
if __name__ == '__main__':
    print("app_state.py が直接実行されました。")
    print("定義されている状態変数 (一部):")
    print(f"  mask_image: {type(mask_image)}")
    print(f"  global_wcs_info: {type(global_wcs_info)}")
    print(f"  periodic_scan_mode_enabled: {periodic_scan_mode_enabled}")
    print(f"  periodic_scan_interval: {periodic_scan_interval}")
    print(f"  cancel_flag: {cancel_flag.is_set()}")
    print(f"  progress_queue: {type(progress_queue)}")
    print(f"  save_options: {save_options}")

    # キューのテスト
    progress_queue.put(("テストメッセージ", 1))
    item = progress_queue.get()
    print(f"キューから取得: {item}")

    # キャンセルフラグのテスト
    cancel_flag.set()
    print(f"キャンセル後: cancel_flag.is_set() -> {cancel_flag.is_set()}")
    cancel_flag.clear()
    print(f"クリア後: cancel_flag.is_set() -> {cancel_flag.is_set()}")
