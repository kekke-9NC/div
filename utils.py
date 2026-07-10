# file_utils.py

import os
import glob
import time
import threading
from pathlib import Path
from datetime import datetime, time as dt_time
import queue
from typing import Set, Optional, Callable, Dict, Any, Tuple
import cv2
import numpy as np
import platform
import subprocess

import config
import video_processing

# winsound は Windows 専用
if platform.system() == "Windows":
    try:
        import winsound
    except ImportError:
        print("winsoundモジュールが見つかりません。通知音は再生されません。")
        winsound = None
else:
    winsound = None


# --- グローバル変数 ---
rtsp_processed_files: Set[str] = set()

# --- 定期スキャン関連 ---

def process_video_file_periodic(
    file_path: str,
    progress_callback: Optional[Callable[[Tuple[str, Optional[Any]]], None]] = None,
    mask: Optional[np.ndarray] = None,
    global_wcs_info: Optional[Dict] = None,
    plate_solve_mask: Optional[np.ndarray] = None,
    meteor_save_path: str = config.DEFAULT_METEOR_SAVE_PATH,
    not_meteor_save_path: str = config.DEFAULT_NOT_METEOR_SAVE_PATH,
    cancel_flag: Optional[threading.Event] = None,
    save_options: Optional[Dict[str, bool]] = None,
    interval: float = config.DEFAULT_INTERVAL,
    duration: float = config.DEFAULT_DURATION,
    min_length: int = config.MIN_LINE_LENGTH
):
    """
    定期スキャンで見つかった新規動画ファイルに対して流星検出処理を実行する。
    video_processing.create_line_video_clips のラッパー関数。
    """
    # 開始ログはvideo_processing側で出力されるため、ここでは出力しない
    try:
        video_processing.create_line_video_clips(
            source=file_path,
            is_rtsp=False, # 通常ファイルとして処理
            interval=interval,
            duration=duration,
            min_length=min_length,
            mask=mask,
            roi=None,
            progress_callback=progress_callback,
            meteor_save_path=meteor_save_path,
            not_meteor_save_path=not_meteor_save_path,
            use_plate_solve=(global_wcs_info is not None),
            global_wcs_info=global_wcs_info,
            plate_solve_mask=plate_solve_mask,
            cancel_flag=cancel_flag,
            save_options=save_options,
            notify_on_detection=True # 定期スキャン中は通知する
        )
        # 完了ログはvideo_processing側で出力されるため、ここでは出力しない

    except Exception as e:
        message = f"[定期スキャン] 処理中にエラーが発生 ({file_path}): {e}"
        print(f"エラー: {message}")
        if progress_callback:
            progress_callback((message, None))
        import traceback
        traceback.print_exc()

def is_within_time_range(
    current_dt: datetime,
    limit_enabled: bool,
    start_h: int, start_m: int, end_h: int, end_m: int
) -> bool:
    """
    現在時刻が指定された時間範囲内（定期スキャンを実行する時間帯）にあるか判定する。
    """
    if not limit_enabled:
        return True

    current_time = current_dt.time()
    start_time = dt_time(start_h, start_m)
    end_time = dt_time(end_h, end_m)

    if start_time <= end_time:
        # 日をまたがない場合 (例: 09:00 ～ 17:00)
        return start_time <= current_time < end_time
    else:
        # 日をまたぐ場合 (例: 17:00 ～ 07:00)
        return current_time >= start_time or current_time < end_time


def monitor_directory(
    directory: str,
    scan_interval: int,
    progress_callback: Optional[Callable[[Tuple[str, Optional[Any]]], None]] = None,
    mask: Optional[np.ndarray] = None,
    global_wcs_info: Optional[Dict] = None,
    plate_solve_mask: Optional[np.ndarray] = None,
    meteor_save_path: str = config.DEFAULT_METEOR_SAVE_PATH,
    not_meteor_save_path: str = config.DEFAULT_NOT_METEOR_SAVE_PATH,
    cancel_flag: Optional[threading.Event] = None,
    save_options: Optional[Dict[str, bool]] = None,
    interval: float = config.DEFAULT_INTERVAL,
    duration: float = config.DEFAULT_DURATION,
    min_length: int = config.MIN_LINE_LENGTH,
    time_limit_enabled: bool = False,
    start_hour: int = 17,
    start_minute: int = 0,
    end_hour: int = 7,
    end_minute: int = 0
):
    """
    指定ディレクトリを定期的にスキャンし、新しい動画ファイルを見つけたら処理を実行する。
    指定された時間帯のみスキャン・処理を行うことも可能。
    """
    processed_files: Set[str] = set()
    video_extensions = config.PERIODIC_VIDEO_EXTENSIONS

    if not os.path.isdir(directory):
        message = f"監視対象ディレクトリが見つかりません: {directory}"
        print(f"エラー: {message}")
        if progress_callback:
            progress_callback((message, None))
        return

    # 初回スキャンで既存ファイルを記録し、処理対象外とする
    try:
        print(f"[定期スキャン] 初回スキャン中: {directory}")
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(video_extensions):
                    full_path = os.path.join(root, file)
                    processed_files.add(full_path)
        message = f"[定期スキャン] 初回スキャン完了。既存の {len(processed_files)} ファイルは処理対象外。"
        print(message)
        if progress_callback:
            progress_callback((message, None))
    except Exception as e:
         message = f"[定期スキャン] 初回スキャン中にエラー: {e}"
         print(f"エラー: {message}")
         if progress_callback:
              progress_callback((message, None))
         return

    # 定期スキャンループ
    while cancel_flag is None or not cancel_flag.is_set():
        try:
            now = datetime.now()
            if not is_within_time_range(now, time_limit_enabled, start_hour, start_minute, end_hour, end_minute):
                wait_message = (f"[定期スキャン] 時間外のため待機中... "
                                f"(現在: {now.strftime('%H:%M:%S')}, "
                                f"有効時間帯: {start_hour:02d}:{start_minute:02d}～{end_hour:02d}:{end_minute:02d})")
                if time_limit_enabled:
                     print(wait_message)
                     if progress_callback:
                          progress_callback((wait_message, None))
                else:
                     print(f"[定期スキャン] 次のスキャンまで {scan_interval} 秒待機します。")

                for _ in range(scan_interval):
                    if cancel_flag is not None and cancel_flag.is_set(): break
                    time.sleep(1)
                continue

            print(f"[定期スキャン] ディレクトリをスキャン中 ({now.strftime('%H:%M:%S')}): {directory}")
            new_files_found = []
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith(video_extensions):
                        full_path = os.path.join(root, file)
                        if full_path not in processed_files:
                            new_files_found.append(full_path)
                            processed_files.add(full_path)

            new_files_found.sort(key=os.path.getmtime)
            if new_files_found:
                 message = f"[定期スキャン] {len(new_files_found)} 個の新規動画ファイルを検出。"
                 print(message)
                 if progress_callback:
                      progress_callback((message, None))

                 for file_path in new_files_found:
                      if cancel_flag is not None and cancel_flag.is_set(): break
                      process_video_file_periodic(
                          file_path, progress_callback, mask, global_wcs_info, plate_solve_mask,
                          meteor_save_path, not_meteor_save_path, cancel_flag, save_options,
                          interval, duration, min_length
                      )
            else:
                 print("[定期スキャン] 新規ファイルは見つかりませんでした。")

            wait_message = f"[定期スキャン] スキャン完了。次のスキャンまで {scan_interval} 秒待機します。"
            print(wait_message)

            for _ in range(scan_interval):
                if cancel_flag is not None and cancel_flag.is_set(): break
                time.sleep(1)

        except Exception as e:
            message = f"[定期スキャン] ループ中にエラー: {e}"
            print(f"エラー: {message}")
            if progress_callback:
                progress_callback((message, None))
            for _ in range(scan_interval):
                if cancel_flag is not None and cancel_flag.is_set(): break
                time.sleep(1)

    print("[定期スキャン] 監視スレッドを終了します。")
    if progress_callback:
        progress_callback(("[定期スキャン] 監視を終了しました。", None))


# --- RTSP関連 ---

def create_rtsp_capture(rtsp_url: str, use_tcp: bool = None, use_nvidia_hwaccel: bool = None) -> cv2.VideoCapture:
    """
    最適化されたRTSP用VideoCaptureオブジェクトを作成する。
    NVIDIAハードウェアデコードが有効な場合、システムFFmpegのh264_cuvidを使用する。
    
    Args:
        rtsp_url: RTSPストリームのURL
        use_tcp: TCPトランスポートを使用するか。NoneならRTSP_USE_TCP設定を使用。
        use_nvidia_hwaccel: NVIDIAハードウェアデコードを使用するか。NoneならRTSP_USE_NVIDIA_HWACCEL設定を使用。
    
    Returns:
        cv2.VideoCapture: 設定済みのVideoCaptureオブジェクト
    """
    import os
    import subprocess
    import shutil
    
    if use_tcp is None:
        use_tcp = config.RTSP_USE_TCP
    if use_nvidia_hwaccel is None:
        use_nvidia_hwaccel = config.RTSP_USE_NVIDIA_HWACCEL
    
    # システムFFmpegが利用可能かチェック
    ffmpeg_path = shutil.which("ffmpeg")
    
    # NVIDIAハードウェアデコードを使用したい場合のログ出力（参考情報）
    # 注: OpenCVはFFmpegパイプを直接サポートしていないため、環境変数方式を使用
    if use_nvidia_hwaccel and ffmpeg_path and rtsp_url.startswith("rtsp://"):
        try:
            result = subprocess.run([ffmpeg_path, '-decoders'], capture_output=True, text=True, timeout=5)
            if 'h264_cuvid' in result.stdout:
                print(f"[RTSP] h264_cuvid利用可能。OpenCV標準方式で接続します。")
        except Exception as e:
            print(f"[RTSP] h264_cuvid確認中にエラー: {e}")
    
    # 標準のOpenCV方式（環境変数でできるだけ設定）
    ffmpeg_options = []
    
    # TCPトランスポート設定
    if use_tcp:
        ffmpeg_options.append("rtsp_transport;tcp")
        print(f"[RTSP] TCPトランスポートを使用")
    
    # 環境変数を設定
    if ffmpeg_options:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "|".join(ffmpeg_options)
        print(f"[RTSP] FFmpegオプション: {os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS']}")
    
    if rtsp_url.startswith("rtsp://"):
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, config.RTSP_BUFFER_SIZE)
    else:
        cap = cv2.VideoCapture(rtsp_url)
    
    return cap





def save_rtsp_video_segments(
    rtsp_url: str,
    save_root: str = config.RTSP_SAVE_ROOT,
    segment_duration: int = config.RTSP_SEGMENT_DURATION,
    cancel_flag: Optional[threading.Event] = None,
    use_tcp: bool = None
):
    """
    RTSPストリームから720フレーム（1分 @ 12fps）ごとに動画ファイルを保存する。
    保存先: save_root/YYYYMMDD/HH/MM.mp4
    再接続失敗時はリトライを継続する。
    
    Args:
        use_tcp: TCPトランスポートを使用するか。NoneならRTSP_USE_TCP設定を使用。
    """
    if use_tcp is None:
        use_tcp = config.RTSP_USE_TCP
    
    # リトライ設定
    MAX_RETRY_ATTEMPTS = 10  # 最大リトライ回数
    BASE_RETRY_WAIT = 5  # 基本待機時間（秒）
    MAX_RETRY_WAIT = 60  # 最大待機時間（秒）
    retry_count = 0
    
    cap = create_rtsp_capture(rtsp_url, use_tcp)
    if use_tcp:
        print(f"[RTSP保存] TCPトランスポートを使用します")
    
    if not cap.isOpened():
        print(f"エラー: RTSPストリームを開けませんでした: {rtsp_url}")
        # 初回接続失敗時もリトライ
        while (cancel_flag is None or not cancel_flag.is_set()) and retry_count < MAX_RETRY_ATTEMPTS:
            retry_count += 1
            wait_time = min(BASE_RETRY_WAIT * (2 ** (retry_count - 1)), MAX_RETRY_WAIT)
            print(f"[RTSP保存] 初回接続失敗。{wait_time}秒後にリトライします ({retry_count}/{MAX_RETRY_ATTEMPTS})...")
            for _ in range(int(wait_time)):
                if cancel_flag is not None and cancel_flag.is_set():
                    print("[RTSP保存] キャンセルされました。")
                    return
                time.sleep(1)
            cap = create_rtsp_capture(rtsp_url, use_tcp)
            if cap.isOpened():
                print(f"[RTSP保存] 接続成功 (リトライ {retry_count}回目)")
                retry_count = 0
                break
        else:
            if retry_count >= MAX_RETRY_ATTEMPTS:
                print(f"[RTSP保存] 最大リトライ回数 ({MAX_RETRY_ATTEMPTS}) に達しました。終了します。")
            return

    fps = config.RTSP_FPS  # 12fps前提
    segment_frames = config.RTSP_SEGMENT_FRAMES  # 720フレーム (12fps * 60秒)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_FOURCC)

    print(f"[RTSP保存] ストリーム開始 ({width}x{height} @ {fps} fps, {segment_frames}フレーム/セグメント)")

    while cancel_flag is None or not cancel_flag.is_set():
        out = None
        temp_file_path = ""
        try:
            now = datetime.now()
            date_str = now.strftime("%Y%m%d")
            hour_str = now.strftime("%H")
            minute_str = now.strftime("%M")

            dir_path = os.path.join(save_root, date_str, hour_str)
            os.makedirs(dir_path, exist_ok=True)

            # 不完全なファイルの上書きを避けるため、一時ファイルに書き込む
            temp_file_path = os.path.join(dir_path, f"{minute_str}_temp_{time.time_ns()}.mp4")
            final_file_path = os.path.join(dir_path, f"{minute_str}.mp4")

            out = cv2.VideoWriter(temp_file_path, fourcc, fps, (width, height))
            if not out.isOpened():
                 print(f"エラー: VideoWriter を開けませんでした: {temp_file_path}")
                 time.sleep(5)
                 continue

            print(f"[RTSP保存] セグメント開始 (一時ファイル): {temp_file_path}")
            frames_written = 0
            stream_error = False

            # フレーム数ベースで保存 (720フレーム = 1分 @ 12fps)
            while frames_written < segment_frames:
                if cancel_flag is not None and cancel_flag.is_set():
                    stream_error = True
                    break

                ret, frame = cap.read()
                if not ret:
                    print("エラー: RTSPストリームのフレーム読み込みに失敗しました。")
                    stream_error = True
                    break

                out.write(frame)
                frames_written += 1

            out.release()
            out = None

            if stream_error:
                 print(f"[RTSP保存] セグメント中断またはエラー。一時ファイルを削除: {temp_file_path}")
                 if os.path.exists(temp_file_path): os.remove(temp_file_path)
                 # ストリームエラー時は再接続を試みる（リトライロジック）
                 cap.release()
                 
                 reconnected = False
                 while (cancel_flag is None or not cancel_flag.is_set()) and retry_count < MAX_RETRY_ATTEMPTS:
                     retry_count += 1
                     wait_time = min(BASE_RETRY_WAIT * (2 ** (retry_count - 1)), MAX_RETRY_WAIT)
                     print(f"[RTSP保存] 再接続失敗。{wait_time}秒後にリトライします ({retry_count}/{MAX_RETRY_ATTEMPTS})...")
                     for _ in range(int(wait_time)):
                         if cancel_flag is not None and cancel_flag.is_set():
                             break
                         time.sleep(1)
                     if cancel_flag is not None and cancel_flag.is_set():
                         break
                     cap = create_rtsp_capture(rtsp_url, use_tcp)
                     if cap.isOpened():
                         print(f"[RTSP保存] 再接続成功 (リトライ {retry_count}回目)")
                         retry_count = 0
                         # 解像度が変わっている可能性があるのでチェック
                         new_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                         new_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                         if new_width != width or new_height != height:
                             print(f"[RTSP保存] 解像度変更検出: {width}x{height} -> {new_width}x{new_height}")
                             width, height = new_width, new_height
                         reconnected = True
                         break
                 
                 if not reconnected:
                     if retry_count >= MAX_RETRY_ATTEMPTS:
                         print(f"[RTSP保存] 最大リトライ回数 ({MAX_RETRY_ATTEMPTS}) に達しました。終了します。")
                         break
                     if cancel_flag is not None and cancel_flag.is_set():
                         break

            elif frames_written > 0:
                # 成功した場合、一時ファイルを最終ファイル名にリネーム (アトミックな操作)
                try:
                    os.replace(temp_file_path, final_file_path)
                    print(f"[RTSP保存] セグメント保存完了: {final_file_path} ({frames_written} フレーム)")
                    retry_count = 0  # 成功したらリトライカウンタをリセット
                except OSError as e:
                    print(f"エラー: 一時ファイルのリネーム/上書きに失敗 ({temp_file_path} -> {final_file_path}): {e}")
                    if os.path.exists(temp_file_path): os.remove(temp_file_path)

            else:
                print(f"[RTSP保存] セグメントにフレームが書き込まれませんでした。一時ファイルを削除: {temp_file_path}")
                if os.path.exists(temp_file_path): os.remove(temp_file_path)

        except Exception as e:
            print(f"[RTSP保存] ループ中にエラー: {e}")
            if out is not None and out.isOpened(): out.release()
            if temp_file_path and os.path.exists(temp_file_path):
                try: os.remove(temp_file_path)
                except OSError: pass
            
            # エラー発生後も再接続を試みる（リトライロジック）
            if cap.isOpened(): cap.release()
            
            reconnected = False
            while (cancel_flag is None or not cancel_flag.is_set()) and retry_count < MAX_RETRY_ATTEMPTS:
                retry_count += 1
                wait_time = min(BASE_RETRY_WAIT * (2 ** (retry_count - 1)), MAX_RETRY_WAIT)
                print(f"[RTSP保存] エラー後リトライ。{wait_time}秒後に再接続します ({retry_count}/{MAX_RETRY_ATTEMPTS})...")
                for _ in range(int(wait_time)):
                    if cancel_flag is not None and cancel_flag.is_set():
                        break
                    time.sleep(1)
                if cancel_flag is not None and cancel_flag.is_set():
                    break
                cap = create_rtsp_capture(rtsp_url, use_tcp)
                if cap.isOpened():
                    print(f"[RTSP保存] 再接続成功 (リトライ {retry_count}回目)")
                    retry_count = 0
                    reconnected = True
                    break
            
            if not reconnected:
                if retry_count >= MAX_RETRY_ATTEMPTS:
                    print(f"[RTSP保存] 最大リトライ回数 ({MAX_RETRY_ATTEMPTS}) に達しました。終了します。")
                    break
                if cancel_flag is not None and cancel_flag.is_set():
                    break

    if cap.isOpened():
        cap.release()
    print("[RTSP保存] 保存スレッドを終了します。")



def process_new_rtsp_files(
    rtsp_root: str = config.RTSP_SAVE_ROOT,
    processed_files_set: Set[str] = rtsp_processed_files,
    progress_callback: Optional[Callable[[Tuple[str, Optional[Any]]], None]] = None,
    mask: Optional[np.ndarray] = None,
    global_wcs_info: Optional[Dict] = None,
    plate_solve_mask: Optional[np.ndarray] = None,
    meteor_save_path: str = config.DEFAULT_METEOR_SAVE_PATH,
    not_meteor_save_path: str = config.DEFAULT_NOT_METEOR_SAVE_PATH,
    cancel_flag: Optional[threading.Event] = None,
    save_options: Optional[Dict[str, bool]] = None,
    interval: float = config.DEFAULT_INTERVAL,
    duration: float = config.DEFAULT_DURATION,
    min_length: int = config.MIN_LINE_LENGTH
):
    """
    RTSP動画が保存されているディレクトリをスキャンし、未処理の新しいファイルを解析する。
    """
    new_files_to_process = []
    video_extensions = config.PERIODIC_VIDEO_EXTENSIONS
    try:
        for root_dir, _, files in os.walk(rtsp_root):
            for file in files:
                # 一時ファイルはスキップ
                if file.lower().endswith(video_extensions) and '_temp_' not in file:
                    full_path = os.path.join(root_dir, file)
                    if full_path not in processed_files_set:
                        new_files_to_process.append(full_path)

        new_files_to_process.sort(key=os.path.getmtime)

        if new_files_to_process:
            message = f"[RTSP解析] {len(new_files_to_process)} 個の新規保存動画ファイルを検出。"
            print(message)
            if progress_callback: progress_callback((message, None))

            for file_path in new_files_to_process:
                if cancel_flag is not None and cancel_flag.is_set(): break
                process_video_file_periodic(
                     file_path, progress_callback, mask, global_wcs_info, plate_solve_mask,
                     meteor_save_path, not_meteor_save_path, cancel_flag, save_options,
                     interval, duration, min_length
                )
                processed_files_set.add(file_path)
        else:
             print("[RTSP解析] 解析対象の新規ファイルは見つかりませんでした。")

    except Exception as e:
        message = f"[RTSP解析] スキャンまたは処理中にエラー: {e}"
        print(f"エラー: {message}")
        if progress_callback: progress_callback((message, None))


def rtsp_save_and_process_thread_target(
    rtsp_url: str,
    save_root: str = config.RTSP_SAVE_ROOT,
    segment_duration: int = config.RTSP_SEGMENT_DURATION,
    scan_interval: int = 60,
    progress_callback: Optional[Callable[[Tuple[str, Optional[Any]]], None]] = None,
    mask: Optional[np.ndarray] = None,
    global_wcs_info: Optional[Dict] = None,
    plate_solve_mask: Optional[np.ndarray] = None,
    meteor_save_path: str = config.DEFAULT_METEOR_SAVE_PATH,
    not_meteor_save_path: str = config.DEFAULT_NOT_METEOR_SAVE_PATH,
    cancel_flag: Optional[threading.Event] = None,
    save_options: Optional[Dict[str, bool]] = None,
    interval: float = config.DEFAULT_INTERVAL,
    duration: float = config.DEFAULT_DURATION,
    min_length: int = config.MIN_LINE_LENGTH
):
    """
    RTSPの保存と新規ファイルの解析を並行して行うためのスレッド関数。
    録画スレッドの生存を監視し、停止した場合は自動で再起動する。
    """
    global rtsp_processed_files

    # スレッド再起動設定
    MAX_RESPAWN_ATTEMPTS = 5  # 連続再起動の上限
    respawn_count = 0
    last_alive_time = time.time()
    
    def create_save_thread():
        return threading.Thread(
            target=save_rtsp_video_segments,
            args=(rtsp_url, save_root, segment_duration, cancel_flag),
            daemon=True
        )

    save_thread = create_save_thread()
    save_thread.start()
    print("[RTSP統合] 保存スレッドを開始しました。")
    if progress_callback: progress_callback(("[RTSP] 保存スレッド開始", None))

    time.sleep(5)

    while cancel_flag is None or not cancel_flag.is_set():
        # 録画スレッドの生存確認
        if not save_thread.is_alive():
            if respawn_count < MAX_RESPAWN_ATTEMPTS:
                respawn_count += 1
                warning_msg = f"[RTSP統合] 警告: 保存スレッドが停止しました。再起動します ({respawn_count}/{MAX_RESPAWN_ATTEMPTS})..."
                print(warning_msg)
                if progress_callback:
                    progress_callback((warning_msg, None))
                
                # 少し待ってから再起動
                time.sleep(2)
                
                save_thread = create_save_thread()
                save_thread.start()
                print("[RTSP統合] 保存スレッドを再起動しました。")
                if progress_callback:
                    progress_callback(("[RTSP] 保存スレッド再起動完了", None))
            else:
                error_msg = f"[RTSP統合] エラー: 保存スレッドの連続再起動上限 ({MAX_RESPAWN_ATTEMPTS}回) に達しました。処理を終了します。"
                print(error_msg)
                if progress_callback:
                    progress_callback((error_msg, None))
                break
        else:
            # 正常動作中は再起動カウントをリセット
            current_time = time.time()
            # 最後に生存確認してから60秒以上経過していたらカウントリセット
            if current_time - last_alive_time > 60:
                if respawn_count > 0:
                    print(f"[RTSP統合] 保存スレッドが安定稼働中。再起動カウントをリセット ({respawn_count} -> 0)")
                respawn_count = 0
            last_alive_time = current_time
        
        # ファイル解析処理
        process_new_rtsp_files(
            rtsp_root, rtsp_processed_files, progress_callback, mask, global_wcs_info, plate_solve_mask,
            meteor_save_path, not_meteor_save_path, cancel_flag, save_options,
            interval, duration, min_length
        )

        wait_message = f"[RTSP統合] 解析スキャン完了。次のスキャンまで {scan_interval} 秒待機。"
        print(wait_message)

        for _ in range(scan_interval):
            if cancel_flag is not None and cancel_flag.is_set(): break
            # 待機中も録画スレッドの生存を確認
            if not save_thread.is_alive():
                print("[RTSP統合] 待機中に保存スレッドの停止を検出。")
                break
            time.sleep(1)

    print("[RTSP統合] 統合処理スレッドを終了します。")
    if progress_callback: progress_callback(("[RTSP] 統合処理終了", None))
    
    # 録画スレッドの終了を待機
    if save_thread.is_alive():
        print("[RTSP統合] 保存スレッドの終了を待機...")
        save_thread.join(timeout=10)


def play_notification_sound() -> bool:
    """検出時の通知音をOS標準の方法で非同期再生する。"""
    try:
        system = platform.system()
        if system == "Windows" and winsound:
            winsound.MessageBeep(winsound.MB_ICONASTERISK)
            return True

        if system == "Darwin":
            # Finderから起動した場合も確実に見つかる絶対パスを使用する。
            # Popenにして、RTSPの解析スレッドを音の再生中に停止させない。
            sound_path = "/System/Library/Sounds/Glass.aiff"
            if os.path.isfile(sound_path):
                subprocess.Popen(
                    ["/usr/bin/afplay", sound_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True

        # その他の環境、またはmacOS標準音がない場合のフォールバック。
        print("\a", end="", flush=True)
        return True
    except Exception as e:
        print(f"通知音の再生中にエラー: {e}")
        return False


if __name__ == '__main__':
    print("file_utils.py が直接実行されました。")
    # 単体での実行テストは困難です。

    # is_within_time_range のテスト
    print("\n--- is_within_time_range テスト ---")
    now = datetime.now()
    print(f"現在 {now.strftime('%H:%M')}, 範囲 09:00-17:00 -> {is_within_time_range(now, True, 9, 0, 17, 0)}")
    print(f"現在 {now.strftime('%H:%M')}, 範囲 17:00-07:00 -> {is_within_time_range(now, True, 17, 0, 7, 0)}")
    print(f"現在 {now.strftime('%H:%M')}, 時間指定なし -> {is_within_time_range(now, False, 17, 0, 7, 0)}")
