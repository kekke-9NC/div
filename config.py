# config.py (整数値確認版)

import os
import sys

# --- Determine Base Paths for Portable/Frozen execution ---
IS_FROZEN = bool(getattr(sys, 'frozen', False))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if IS_FROZEN:
    # Running as compiled exe
    BUNDLE_DIR = sys._MEIPASS
    EXE_DIR = os.path.dirname(sys.executable)
    LEGACY_BUNDLE_DIR = BUNDLE_DIR
    LEGACY_EXE_DIR = EXE_DIR
    LEGACY_INSTALL_ROOTS = ()

    if sys.platform == "win32":
        _user_data_base = (
            os.environ.get("LOCALAPPDATA")
            or os.environ.get("APPDATA")
            or os.path.join(os.path.expanduser("~"), "AppData", "Local")
        )
    elif sys.platform == "darwin":
        _user_data_base = os.path.join(os.path.expanduser("~"), "Library", "Application Support")
    else:
        _user_data_base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    USER_DATA_DIR = os.path.join(_user_data_base, "MeteorDetector")
    APP_DATA_DIR = USER_DATA_DIR
else:
    # Running as script (Development)
    # Assuming standard structure: my_app/div/config.py
    # So base dir for resources/output should be parent of div
    PARENT_DIR = os.path.dirname(SCRIPT_DIR)
    
    # Decide where base ref is. Original code was hardcoded to C:\Users\kekke\Desktop\my_app
    # calculated PARENT_DIR should match that.
    EXE_DIR = PARENT_DIR 
    BUNDLE_DIR = SCRIPT_DIR
    LEGACY_BUNDLE_DIR = None
    LEGACY_EXE_DIR = None
    LEGACY_INSTALL_ROOTS = ()
    USER_DATA_DIR = None
    APP_DATA_DIR = EXE_DIR


def is_path_within(path, directory):
    """Return whether path is directory itself or a descendant of it."""
    try:
        normalized_path = os.path.normcase(os.path.abspath(path))
        normalized_directory = os.path.normcase(os.path.abspath(directory))
        return os.path.commonpath((normalized_path, normalized_directory)) == normalized_directory
    except (OSError, ValueError):
        return False

# Resolve model path relative to this file (portable/exe-friendly).
MODEL_FILENAMES = ("model_epoch_47.pth", "model_latest_1.pth")
_MODEL_CANDIDATES = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    for filename in MODEL_FILENAMES
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
FINER_DETECT_PADDING_SECONDS = 0.5 # 詳細検出で特定した範囲へのパディング秒数 (float)
FINER_CUTOUT_SIZE = 384           # 詳細検出時のカットアウトサイズ（線検出用、保存用CUTOUTとは別）

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
DEFAULT_METEOR_SAVE_PATH = os.path.join(APP_DATA_DIR, "meteor")
DEFAULT_NOT_METEOR_SAVE_PATH = os.path.join(APP_DATA_DIR, "not_meteor")
TEMP_CLIP_DIR = os.path.join(APP_DATA_DIR, "temp_clips")
TEMP_VIDEO_DIR = (
    os.path.join(APP_DATA_DIR, "temp_video") if IS_FROZEN
    else os.path.join(SCRIPT_DIR, "temp_video")
)
DISTORTION_MAP_DIR = (
    os.path.join(APP_DATA_DIR, "distortion_maps") if IS_FROZEN
    else SCRIPT_DIR
)

# Preserve access to output folders used by older frozen builds. Source runs
# already use the same directories as the legacy layout.
LEGACY_METEOR_SAVE_PATH = os.path.join(EXE_DIR, "meteor") if IS_FROZEN else None
LEGACY_NOT_METEOR_SAVE_PATH = os.path.join(EXE_DIR, "not_meteor") if IS_FROZEN else None
LEGACY_DISTORTION_MAP_DIRS = tuple(
    dict.fromkeys(
        path for path in (LEGACY_BUNDLE_DIR, LEGACY_EXE_DIR)
        if path and os.path.abspath(path) != os.path.abspath(DISTORTION_MAP_DIR)
    )
)
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

# --- Local Plate Solve (WSL solve-field) 関連 ---
LOCAL_SOLVER_ENABLED = True           # ローカルソルバーを有効化（WSLが利用可能な場合）
LOCAL_SOLVER_INDEX_DIR = "/usr/share/astrometry/data"  # WSL内のインデックスファイルパス

# --- 天体カタログ関連 ---
VIZIER_STAR_MAG_LIMIT = 3 # Vizierから取得して画像に表示する星の最大等級 (floatでも可)

# --- 定期スキャン・RTSP関連 ---
PERIODIC_VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov')
DEFAULT_SCAN_INTERVAL = 60 # 定期スキャンのデフォルト間隔（秒）
RTSP_SAVE_ROOT = os.path.join(APP_DATA_DIR, "rtsp")
LEGACY_RTSP_SAVE_ROOTS = (
    (os.path.join(EXE_DIR, "rtsp"),)
    if IS_FROZEN and os.path.abspath(os.path.join(EXE_DIR, "rtsp")) != os.path.abspath(RTSP_SAVE_ROOT)
    else ()
)


def is_rtsp_recorded_file(path):
    """Recognize recordings written by this and earlier frozen releases."""
    return any(
        is_path_within(path, root)
        for root in (RTSP_SAVE_ROOT, *LEGACY_RTSP_SAVE_ROOTS)
    )


def register_legacy_install_root(root):
    """Keep data from a confirmed prior install discoverable after it moves."""
    global LEGACY_INSTALL_ROOTS, LEGACY_RTSP_SAVE_ROOTS, LEGACY_DISTORTION_MAP_DIRS
    if not IS_FROZEN or not root:
        return
    root = os.path.abspath(root)
    normalized_root = os.path.normcase(os.path.abspath(root))
    existing_install_roots = {
        os.path.normcase(os.path.abspath(path))
        for path in LEGACY_INSTALL_ROOTS
    }
    if normalized_root not in existing_install_roots:
        LEGACY_INSTALL_ROOTS = (*LEGACY_INSTALL_ROOTS, root)
    rtsp_root = os.path.join(root, "rtsp")
    rtsp_roots = {
        os.path.normcase(os.path.abspath(path))
        for path in LEGACY_RTSP_SAVE_ROOTS
    }
    if os.path.normcase(os.path.abspath(rtsp_root)) not in rtsp_roots:
        LEGACY_RTSP_SAVE_ROOTS = (*LEGACY_RTSP_SAVE_ROOTS, rtsp_root)
    existing_map_roots = {
        os.path.normcase(os.path.abspath(path))
        for path in LEGACY_DISTORTION_MAP_DIRS
    }
    for map_root in (root, os.path.join(root, "_internal")):
        normalized_map_root = os.path.normcase(os.path.abspath(map_root))
        if normalized_map_root not in existing_map_roots:
            LEGACY_DISTORTION_MAP_DIRS = (*LEGACY_DISTORTION_MAP_DIRS, map_root)
            existing_map_roots.add(normalized_map_root)
RTSP_FPS = 25 # RTSPストリームの前提FPS
RTSP_SEGMENT_FRAMES = 1500 # 既定値。実際の録画フレーム数はFPSとセグメント秒数から算出。
RTSP_SEGMENT_DURATION = 60 # RTSP動画の1セグメントあたりの秒数（参考値）
RTSP_BUFFER_DURATION = 10 # RTSP処理時にメモリ上に保持するバッファの秒数 (float)

# --- RTSP用Astrometry設定 (視野角) ---
RTSP_SCALE_LOWER = 85     # RTSP用視野角推定の下限（度）- 水平約88.9°
RTSP_SCALE_UPPER = 100    # RTSP用視野角推定の上限（度）- 対角約96.9°

# --- RTSP最適化設定 ---
RTSP_USE_TCP = True  # TCPトランスポートを使用（パケット損失防止）
RTSP_BUFFER_SIZE = 3  # VideoCaptureバッファサイズ（フレーム数、遅延低減のため最小値）
RTSP_USE_NVIDIA_HWACCEL = True  # NVIDIA GPU (RTX/GTX) のハードウェアデコードを使用
RTSP_PARALLEL_ENABLED = True  # RTSP並列処理の有効/無効
RTSP_PARALLEL_WORKERS = None  # 並列ワーカー数 (None=CPU数自動, 数値=固定)

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
    'canny_thresh2': 300,
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
DEFAULT_AI_VLM_BACKEND = AI_VLM_BACKEND_LOCAL_QWEN3_VL_4B
DEFAULT_LM_STUDIO_VLM_URL = "http://localhost:1234/v1"
DEFAULT_LM_STUDIO_VLM_MODEL_ID = "qwen3.5-2b"
DEFAULT_LM_STUDIO_VLM_API_KEY = ""

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

