# config.py (整数値確認版)

import os
import sys

IS_MAC = sys.platform == "darwin"
IS_WINDOWS = sys.platform == "win32"

# --- Determine Base Paths for Portable/Frozen execution ---
if getattr(sys, 'frozen', False):
    # Running as compiled exe
    BUNDLE_DIR = sys._MEIPASS
    EXE_DIR = os.path.dirname(sys.executable)
else:
    # Running as script (Development)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Keep generated working folders next to the source files on macOS/dev runs.
    EXE_DIR = SCRIPT_DIR

# Resolve model path relative to this file (portable/exe-friendly).
_MODEL_CANDIDATES = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "meteor_fusion_universal_v1.pth"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_epoch_47.pth"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_latest_1.pth"),
]
MODEL_PATH = _MODEL_CANDIDATES[0]
for _candidate in _MODEL_CANDIDATES:
    if os.path.exists(_candidate):
        MODEL_PATH = _candidate
        break

# --- 画像・モデル関連 ---
IMG_HEIGHT = 224  # モデルの入力画像の高さ
IMG_WIDTH = 224   #  モデルの入力画像の幅
NUM_FRAMES = 16   #  モデルが期待するフレーム数（もし3Dモデル等で使用する場合）
# MODEL_PATH is now set above
CUTOUT_SIZE = 256 # 切り出し画像のサイズ

# --- 動画処理・検出関連 ---
MAX_CLIP_DURATION = 2   # (詳細検出を使わない場合の)検出前後を切り出す動画クリップの最大秒数 (floatでも可)
CLIP_DURATION_SECONDS = 3.0 # (詳細検出を使わない場合の)クリップ秒数目安 (float)
BORDER_SIZE = 30        # 画像処理時に無視する画像の端のピクセル数
MIN_LINE_LENGTH = 25    # 粗検出する直線の最小長（ピクセル）
DEFAULT_FPS = 15        # 動画のFPSが取得できない場合のデフォルト値
DUPLICATE_DETECTION_THRESHOLD = 100 # 重複検出とみなす距離(ピクセル)
METEOR_PROBABILITY_THRESHOLD = 0.5 # 流星と判定する確率閾値 (float)

# --- 詳細検出関連 ---
FINER_DETECT_WINDOW_SECONDS = 4.0 # 詳細検出用に取得する秒数 (float)
FINER_COMPOSITE_STEP = 3          # 詳細検出時の比較明合成ステップ数（デフォルト値、FPSから動的計算される）
FINER_DETECT_MIN_LENGTH = 15      # 詳細検出時の最小線長(ピクセル)
# 流星の前後に残す余白。従来の 0.5 秒から 0.25 秒へ短縮し、
# 実際の発光が短い流星で空白の方が長くならないようにする。
FINER_DETECT_PADDING_SECONDS = 0.25 # 詳細検出で特定した範囲へのパディング秒数 (float)
FINER_CUTOUT_SIZE = 384           # 詳細検出時のカットアウトサイズ（線検出用、保存用CUTOUTとは別）
# 詳細検出で、粗検出した流星候補と同じ軌跡とみなす許容値。
# 雲の縁など、カットアウト内の無関係な線でクリップ時間が延びるのを防ぐ。
FINER_LINE_MATCH_DISTANCE = 70    # 粗検出線分からの最大距離 (px)
FINER_LINE_MATCH_ANGLE_DEG = 35   # 粗検出線分との最大角度差 (deg)

# --- オブジェクトトラッキング・飛行機判定関連 ---
TRACKED_OBJECT_POSITIONS_MAXLEN = 30 # オブジェクトが保持する過去の位置情報の最大数
PAST_DETECTIONS_MAXLEN = 10          # 飛行機判定で使用する過去の検出座標の最大数
TRACKING_DISTANCE_THRESHOLD = 200    # 同一オブジェクトとみなす最大距離（ピクセル）
AIRPLANE_DURATION_THRESHOLD = 7      # 飛行機と判定する継続時間の閾値（秒）
AIRPLANE_FRAME_THRESHOLD = 7         # 飛行機と判定する条件を満たすフレーム数の閾値（10フレーム中）
NON_ZERO_DETECTION_THRESHOLD_FOR_AIRPLANE = 5 # 過去Nフレーム中、非ゼロ座標がこの数以上あれば飛行機と判定する閾値
BRIGHTNESS_CONSISTENCY_THRESHOLD = 20 # 飛行機判定用の輝度一貫性チェックの閾値
VELOCITY_CONSISTENCY_THRESHOLD = 2    # 飛行機判定用の速度一貫性チェックの閾値（ピクセル/フレーム）

# --- 保存パス関連 ---
DEFAULT_METEOR_SAVE_PATH = os.path.join(EXE_DIR, "meteor")
DEFAULT_NOT_METEOR_SAVE_PATH = os.path.join(EXE_DIR, "not_meteor")
TEMP_CLIP_DIR = os.path.join(EXE_DIR, "temp_clips")
LIGHTEN_BLEND_CACHE_DIR = os.path.join(EXE_DIR, "lighten_blend_cache")
VIDEO_FOURCC = 'avc1'

# --- Astrometry.net 関連 ---
ASTROMETRY_API_KEY = ""  # app_settings.json から読み込まれます
FIELD_OF_VIEW_DEG = 105   # カメラのおおよその視野角（度） (floatでも可)
SCALE_UNITS = 'degwidth'
SCALE_LOWER = 95          # 視野角推定の下限（度） (floatでも可)
SCALE_UPPER = 115         # 視野角推定の上限（度） (floatでも可)
PLATE_SOLVE_IMAGE_WIDTH = 1920 # プレートソルブにアップロードする画像の幅
PLATE_SOLVE_IMAGE_HEIGHT = 1080# プレートソルブにアップロードする画像の高さ
ASTROMETRY_TIMEOUT = 120   # Astrometry.netのソルブ結果待ちタイムアウト（秒）
ASTROMETRY_INTERVAL = 10   # Astrometry.netのソルブ結果確認間隔（秒）
ASTROMETRY_RATE_LIMIT_WAIT = 20 # Astrometry.net APIのレートリミットのための待機時間（秒）

# --- Local Plate Solve 関連 ---
# macOSはPython版Astrometry.net + Tycho-2インデックを使い、APIへ送信しない。
# Windowsでは従来のWSL solve-fieldも継続して利用できる。
LOCAL_SOLVER_ENABLED = True
LOCAL_SOLVER_INDEX_DIR = "/usr/share/astrometry/data"  # WSL内のインデックスファイルパス
TIMELAPSE_LOCAL_ANNOTATION_ENABLED = False
TIMELAPSE_ANNOTATION_CALIBRATION_PATH = ""
# The local calibration is already tied to the camera's star solution, so
# constellation sticks are a useful default whenever local annotation is on.
TIMELAPSE_CONSTELLATIONS_ENABLED = True

# --- 天体カタログ関連 ---
VIZIER_STAR_MAG_LIMIT = 3 # Vizierから取得して画像に表示する星の最大等級 (floatでも可)

# --- 定期スキャン・RTSP関連 ---
PERIODIC_VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov')
DEFAULT_SCAN_INTERVAL = 60 # 定期スキャンのデフォルト間隔（秒）
RTSP_SAVE_ROOT = os.path.join(EXE_DIR, "rtsp")
RTSP_FPS = 25 # RTSPストリームの前提FPS
RTSP_SEGMENT_FRAMES = 1500 # 旧OpenCV保存用の互換値。現在のRTSP保存長はRTSP_SEGMENT_DURATIONとFPSから計算。
RTSP_SEGMENT_DURATION = 60 # RTSP動画の1セグメントあたりの秒数（参考値）
RTSP_BUFFER_DURATION = 10 # RTSP処理時にメモリ上に保持するバッファの秒数 (float)

# --- RTSP用Astrometry設定 (視野角) ---
RTSP_SCALE_LOWER = 85     # RTSP用視野角推定の下限（度）- 水平約88.9°
RTSP_SCALE_UPPER = 100    # RTSP用視野角推定の上限（度）- 対角約96.9°

# --- RTSP最適化設定 ---
RTSP_USE_TCP = True  # TCPトランスポートを使用（パケット損失防止）
RTSP_BUFFER_SIZE = 3  # VideoCaptureバッファサイズ（フレーム数、遅延低減のため最小値）
RTSP_USE_NVIDIA_HWACCEL = not IS_MAC  # NVIDIA GPU (RTX/GTX) のハードウェアデコードを使用
RTSP_PARALLEL_ENABLED = True  # RTSP並列処理の有効/無効
RTSP_PARALLEL_WORKERS = None  # 並列ワーカー数 (None=CPU数自動, 数値=固定)
RTSP_FIXED_PATTERN_SAMPLE_COUNTS = (90, 180, 360)
RTSP_FIXED_PATTERN_DEFAULT_SAMPLES = 360

# --- RTSP用検出パラメータ プリセット ---
# 雲が少ないとき（感度高め）
RTSP_PRESET_CLEAR_SKY = {
    'name': '雲が少ないとき',
    'min_line_length': 20,
    'hough_threshold': 25,
    'canny_thresh1': 75,
    'canny_thresh2': 180,
}

# 雲が多いとき（ノイズ対策）
RTSP_PRESET_CLOUDY = {
    'name': '雲が多いとき',
    'min_line_length': 25,
    'hough_threshold': 35,
    'canny_thresh1': 100,
    # 3/5フレーム平均では、1フレームだけ発光する流星の輝度差も
    # 平均化される。上限300ではそのエッジが強エッジにならず、
    # 雲を抑えるプリセットだけ流星を粗検出できないことがある。
    # Hough閾値は高いまま維持し、Canny上限だけ実測値に合わせる。
    'canny_thresh2': 240,
}

# デフォルト値（後方互換性のため）
RTSP_MIN_LINE_LENGTH = 25
RTSP_HOUGH_THRESHOLD = 25
RTSP_CANNY_THRESH1 = 80
RTSP_CANNY_THRESH2 = 240


# --- GUI関連 ---
GUI_WINDOW_TITLE = "Meteor Detector v0.8.27"
GUI_WINDOW_GEOMETRY = "1200x900"
GUI_RESIZED_WIDTH = 640      # GUIで動画フレームをリサイズ表示する際の幅
GUI_RESIZED_HEIGHT = 300     # GUIで動画フレームをリサイズ表示する際の高さ
GUI_DEBUG_WINDOW_GEOMETRY = "1280x720"
GUI_MASK_WINDOW_GEOMETRY = "1600x800"
GUI_PLATE_SOLVE_MASK_WINDOW_GEOMETRY = "700x600"
MASK_PREVIEW_SIZE = (100, 100) # Tuple of ints

# --- AI / VLM model selection ---
AI_VLM_BACKEND_LOCAL_QWEN3_VL_4B = "local_qwen3_vl_4b"
AI_VLM_BACKEND_LM_STUDIO_QWEN35_2B = "lmstudio_qwen3_5_2b"
DEFAULT_AI_VLM_BACKEND = AI_VLM_BACKEND_LM_STUDIO_QWEN35_2B if IS_MAC else AI_VLM_BACKEND_LOCAL_QWEN3_VL_4B
DEFAULT_LM_STUDIO_VLM_URL = "http://localhost:1234/v1"
DEFAULT_LM_STUDIO_VLM_MODEL_ID = "qwen/qwen3-vl-4b"
DEFAULT_LM_STUDIO_VLM_API_KEY = ""
AI_VLM_MAX_PARALLEL_REQUESTS = 4

# --- 処理設定のデフォルト値 ---
DEFAULT_CONCURRENCY = 4 # デフォルトの同時処理数
DEFAULT_INTERVAL = 1  # デフォルトの差分作成間隔（秒） (float)
DEFAULT_DURATION = 1  # デフォルトの差分作成期間（秒） (float)

# --- 画像処理パラメータ (詳細設定が必要な場合) ---
# CANNY_THRESHOLD1 = 50
# CANNY_THRESHOLD2 = 150
# HOUGH_THRESHOLD = 25
# HOUGH_MAX_GAP = 5

# --- サウンド ---
# NOTIFICATION_SOUND_PATH = "path/to/sound.wav"

# --- 保存オプションのデフォルト ---
DEFAULT_SAVE_VIDEO_CLIP = True
DEFAULT_SAVE_CUTOUT_DIFF = True
DEFAULT_SAVE_FULL_DIFF = False
DEFAULT_SAVE_COMPOSITE = True
DEFAULT_SAVE_DETECTION_INFO = True
DEFAULT_SAVE_FULL_VIDEO = False  # フルサイズ動画（トリミングなし）
DEFAULT_SAVE_DENOISED_FULL_VIDEO = False  # 旧個別出力。共通保存物補正へ移行。

# --- 機械学習向けイベントデータセット ---
DEFAULT_ML_TRAINING_EXPORT_ENABLED = False
DEFAULT_ML_TRAINING_DATA_ROOT = os.path.join(EXE_DIR, "ml_training_data")

# --- 過去動画フォルダの自動前処理 ---
DEFAULT_AUTO_VIDEO_MASK_ENABLED = True
AUTO_VIDEO_MASK_CACHE_DIR = os.path.join(EXE_DIR, "auto_video_masks")
DEFAULT_DATE_FOLDER_TWILIGHT_FILTER_ENABLED = True
DEFAULT_OBSERVATION_LATITUDE = 35.0
DEFAULT_OBSERVATION_LONGITUDE = 135.0

# --- 検出クリップの閲覧用ノイズ低減 ---
# 5フレーム（前後2枚）の時間中央値を主体にし、検出線付近の明るい過渡成分を原画から戻す。
DENOISE_TEMPORAL_RADIUS = 2
DENOISE_ORIGINAL_BLEND = 0.12
DENOISE_TRANSIENT_THRESHOLD = 10.0
DENOISE_PROTECT_LINE_WIDTH = 32

# --- フルサイズ動画の時刻表示 ---
# 画面高に対する比率で文字サイズを指定するため、HD/4K どちらでも大きくなり過ぎない。
FULL_VIDEO_TIMESTAMP_ENABLED = True
FULL_VIDEO_TIMESTAMP_POSITION = "bottom_right"
FULL_VIDEO_TIMESTAMP_SIZE_PERCENT = 1.8

# --- タイムラプス動画の時刻表示 ---
TIMELAPSE_TIMESTAMP_ENABLED = True
TIMELAPSE_TIMESTAMP_POSITION = "bottom_right"
TIMELAPSE_TIMESTAMP_SIZE_PERCENT = 1.8

# --- 動画連結の実時刻表示 ---
VIDEO_CONCAT_TIMESTAMP_ENABLED = True
VIDEO_CONCAT_TIMESTAMP_POSITION = "bottom_right"
VIDEO_CONCAT_TIMESTAMP_SIZE_PERCENT = 1.8
VIDEO_CONCAT_TIMESTAMP_OFFSET_SECONDS = 0.0

# --- タイムラプスの時間平均 ---
TIMELAPSE_TEMPORAL_MEAN_RADIUS_FRAMES = 50
