import os
import time
import threading
from typing import Set, Optional, Callable, Dict, Any, Tuple, List
import cv2
import numpy as np
from datetime import datetime, time as dt_time

import config
import video_processing
import utils  # RTSP最適化用

rtsp_processed_files: Set[str] = set()

def get_unique_file_path(base_path: str) -> str:
    """
    ファイルパスが既に存在する場合、_1, _2 ... を付けてユニークなパスを返す。
    例: 30.mp4 が存在 → 30_1.mp4, さらに存在 → 30_2.mp4
    """
    if not os.path.exists(base_path):
        return base_path
    
    base, ext = os.path.splitext(base_path)
    counter = 1
    while True:
        new_path = f"{base}_{counter}{ext}"
        if not os.path.exists(new_path):
            return new_path
        counter += 1

def is_within_time_range(
    current_dt: datetime, limit_enabled: bool,
    start_h: int, start_m: int, end_h: int, end_m: int
) -> bool:
    if not limit_enabled:
        return True
    current_time = current_dt.time()
    start_time = dt_time(start_h, start_m)
    end_time = dt_time(end_h, end_m)
    if start_time <= end_time:
        return start_time <= current_time < end_time
    else:
        return current_time >= start_time or current_time < end_time

def is_rtsp_file_within_time_range(
    file_path: str, limit_enabled: bool,
    start_h: int, start_m: int, end_h: int, end_m: int
) -> bool:
    """RTSP保存ファイルの HH/MM.mp4 形式の時刻が録画時間内か判定する。"""
    if not limit_enabled:
        return True
    try:
        minute_text = os.path.splitext(os.path.basename(file_path))[0].split("_", 1)[0]
        hour = int(os.path.basename(os.path.dirname(file_path)))
        minute = int(minute_text)
        file_dt = datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
        return is_within_time_range(file_dt, True, start_h, start_m, end_h, end_m)
    except (TypeError, ValueError):
        return True

def process_video_file_periodic(
    file_path: str,
    progress_callback: Optional[Callable[[Tuple[str, Optional[float]]], None]] = None,
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
    summary_video_config: Optional[List[Dict[str, Any]]] = None,
    notify_on_detection: bool = True
) -> bool:
    # 開始ログはvideo_processing側で出力されるため、ここでは出力しない
    try:
        video_processing.create_line_video_clips(
            source=file_path, is_rtsp=False, interval=interval, duration=duration,
            min_length=min_length, mask=mask, roi=None, progress_callback=progress_callback,
            meteor_save_path=meteor_save_path, not_meteor_save_path=not_meteor_save_path,
            use_plate_solve=(global_wcs_info is not None), global_wcs_info=global_wcs_info,
            plate_solve_mask=plate_solve_mask, cancel_flag=cancel_flag, save_options=save_options,
            notify_on_detection=notify_on_detection, summary_video_config=summary_video_config
        )
        # 完了ログはvideo_processing側で出力されるため、ここでは出力しない
        return True
    except IOError as e:
        message = f"[定期スキャン] ファイルを開けませんでした。次回スキャン時に再試行します ({os.path.basename(file_path)}): {e}"
        print(f"警告: {message}")
        if progress_callback:
            progress_callback((message, None))
        return False
    except Exception as e:
        message = f"[定期スキャン] 処理中にエラーが発生 ({os.path.basename(file_path)}): {e}"
        print(f"エラー: {message}")
        if progress_callback:
            progress_callback((message, None))
        import traceback; traceback.print_exc()
        return True

def monitor_directory(
    directory: str, scan_interval: int,
    progress_callback: Optional[Callable[[Tuple[str, Optional[Any]]], None]] = None,
    mask: Optional[np.ndarray] = None, global_wcs_info: Optional[Dict] = None,
    plate_solve_mask: Optional[np.ndarray] = None, meteor_save_path: str = config.DEFAULT_METEOR_SAVE_PATH,
    not_meteor_save_path: str = config.DEFAULT_NOT_METEOR_SAVE_PATH, cancel_flag: Optional[threading.Event] = None,
    save_options: Optional[Dict[str, bool]] = None, interval: float = config.DEFAULT_INTERVAL,
    duration: float = config.DEFAULT_DURATION, min_length: int = config.MIN_LINE_LENGTH,
    summary_video_config: Optional[List[Dict[str, Any]]] = None, time_limit_enabled: bool = False,
    start_hour: int = 17, start_minute: int = 0, end_hour: int = 7, end_minute: int = 0
):
    processed_files: Set[str] = set()
    video_extensions = config.PERIODIC_VIDEO_EXTENSIONS

    if not os.path.isdir(directory):
        message = f"監視対象ディレクトリが見つかりません: {directory}"
        print(f"エラー: {message}")
        if progress_callback: progress_callback((message, None))
        return

    was_outside_time_range = True
    is_first_scan_after_start = False

    try:
        print(f"[定期スキャン] 初回スキャン中 (アプリケーション起動時): {directory}")
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(video_extensions):
                    processed_files.add(os.path.join(root, file))
        message = f"[定期スキャン] アプリ起動時の初回スキャン完了。既存の {len(processed_files)} ファイルは処理対象外。"
        print(message)
        if progress_callback: progress_callback((message, None))
    except Exception as e:
         message = f"[定期スキャン] 初回スキャン中にエラー: {e}"
         print(f"エラー: {message}")
         if progress_callback: progress_callback((message, None))
         return

    while cancel_flag is None or not cancel_flag.is_set():
        try:
            now = datetime.now()
            currently_within_time = is_within_time_range(now, time_limit_enabled, start_hour, start_minute, end_hour, end_minute)

            if time_limit_enabled:
                if not currently_within_time:
                    was_outside_time_range = True
                elif currently_within_time and was_outside_time_range:
                    is_first_scan_after_start = True
                    was_outside_time_range = False
                    first_scan_msg = f"[定期スキャン] 時間帯開始 ({start_hour:02d}:{start_minute:02d})。初回スキャン（ファイルチェックのみ）を行います。"
                    print(first_scan_msg)
                    if progress_callback: progress_callback((first_scan_msg, None))

            if not currently_within_time:
                # When a time limit is enabled, run silently during outside-hours.
                # Do not print or call the progress_callback so no messages appear in the GUI log.
                if not time_limit_enabled:
                    print(f"[定期スキャン] 次のスキャンまで {scan_interval} 秒待機します。")

                for _ in range(scan_interval):
                    if cancel_flag is not None and cancel_flag.is_set(): break
                    time.sleep(1)
                continue

            scan_log_msg = f"[定期スキャン] ディレクトリをスキャン中 ({now.strftime('%H:%M:%S')}): {directory}"
            if is_first_scan_after_start: scan_log_msg += " (時間帯開始後の初回チェック)"
            print(scan_log_msg)

            new_files_found = []
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith(video_extensions):
                        full_path = os.path.join(root, file)
                        if full_path not in processed_files:
                            new_files_found.append(full_path)

            new_files_found.sort(key=os.path.getmtime)
            if new_files_found:
                 message = f"[定期スキャン] {len(new_files_found)} 個の新規動画ファイルを検出。"
                 print(message)
                 if progress_callback: progress_callback((message, None))

                 for file_path in new_files_found:
                      if cancel_flag is not None and cancel_flag.is_set(): break
                      if is_first_scan_after_start:
                          skip_msg = f"  -> スキップ (初回チェック): {os.path.basename(file_path)}"
                          print(skip_msg)
                          if progress_callback: progress_callback((skip_msg, None))
                          processed_files.add(file_path)
                      else:
                          processed_successfully = process_video_file_periodic(
                              file_path, progress_callback, mask, global_wcs_info, plate_solve_mask,
                              meteor_save_path, not_meteor_save_path, cancel_flag, save_options,
                              interval, duration, min_length, summary_video_config
                          )
                          if processed_successfully:
                              processed_files.add(file_path)
            else:
                 print("[定期スキャン] 新規ファイルは見つかりませんでした。")

            if is_first_scan_after_start:
                 is_first_scan_after_start = False
                 print("[定期スキャン] 初回チェック完了。次回から新規ファイルの検出処理を開始します。")

            wait_message = f"[定期スキャン] スキャン完了。次のスキャンまで {scan_interval} 秒待機します。"
            print(wait_message)

            for _ in range(scan_interval):
                if cancel_flag is not None and cancel_flag.is_set(): break
                time.sleep(1)

        except Exception as e:
            message = f"[定期スキャン] ループ中にエラー: {e}"
            print(f"エラー: {message}")
            if progress_callback: progress_callback((message, None))
            is_first_scan_after_start = False
            for _ in range(scan_interval):
                if cancel_flag is not None and cancel_flag.is_set(): break
                time.sleep(1)

    print("[定期スキャン] 監視スレッドを終了します。")
    if progress_callback: progress_callback(("[定期スキャン] 監視を終了しました。", None))

def save_rtsp_video_segments_ffmpeg(
    rtsp_url: str, save_root: str = config.RTSP_SAVE_ROOT,
    segment_duration: int = config.RTSP_SEGMENT_DURATION, cancel_flag: Optional[threading.Event] = None,
    time_limit_enabled: bool = False, start_hour: int = 17, start_minute: int = 0,
    end_hour: int = 7, end_minute: int = 0,
    preview_callback: Optional[Callable[[np.ndarray], None]] = None
):
    """
    FFmpegのsegment muxerを使用してRTSPストリームを連続録画する。
    接続を維持したまま自動的にセグメントファイルに分割するため、隙間なく録画できる。
    
    ネットワーク監視機能:
    - 最新セグメントファイルのサイズを監視し、0kbps（増加なし）が続いたら切断と判断
    - 切断検出時は即座にプロセスを終了し再接続
    """
    import subprocess
    import shutil
    import glob
    
    # ネットワーク監視パラメータ
    STALL_TIMEOUT = 15     # 15秒間活動がなければ切断と判断（セグメント切り替え猶予を含む）
    GRACE_PERIOD = 15      # FFmpeg起動直後は監視しない（接続確立待ち）
    
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        print("[RTSP保存] FFmpegが見つかりません。OpenCV方式にフォールバック")
        return save_rtsp_video_segments(rtsp_url, save_root, segment_duration, cancel_flag)
    
    # h264_cuvidサポート確認 + CUDAデバイス利用可否の確認
    use_cuvid = False
    try:
        result = subprocess.run([ffmpeg_path, '-decoders'], capture_output=True, text=True, timeout=5)
        if 'h264_cuvid' in result.stdout:
            # FFmpegがh264_cuvidをサポートしている場合、実際にCUDAデバイスが利用可能かテスト
            # 短いテストコマンドを実行してCUDAが使えるか確認
            test_result = subprocess.run(
                [ffmpeg_path, '-hide_banner', '-hwaccel', 'cuda', '-f', 'lavfi', '-i', 'nullsrc=s=64x64:d=0.1', '-f', 'null', '-'],
                capture_output=True, text=True, timeout=10
            )
            if test_result.returncode == 0:
                use_cuvid = True
            else:
                # CUDAエラーをチェック
                if 'CUDA_ERROR_NO_DEVICE' in test_result.stderr or 'no CUDA-capable device' in test_result.stderr:
                    print("[RTSP保存] CUDAデバイスが見つかりません。ソフトウェアデコードを使用します。")
                elif 'cuda' in test_result.stderr.lower() and 'error' in test_result.stderr.lower():
                    print(f"[RTSP保存] CUDAが利用できません。ソフトウェアデコードを使用します。")
                else:
                    # その他のエラーでもソフトウェアデコードにフォールバック
                    print(f"[RTSP保存] ハードウェアアクセラレーションのテストに失敗。ソフトウェアデコードを使用します。")
    except Exception as e:
        print(f"[RTSP保存] デコーダー確認中にエラー: {e}。ソフトウェアデコードを使用します。")
        use_cuvid = False
    
    if preview_callback is not None and use_cuvid:
        use_cuvid = False
        print("[RTSP保存] ライブプレビュー有効時は同一入力からプレビューを分岐するため、ソフトウェアデコードを使用します。")

    if use_cuvid:
        print("[RTSP保存] NVIDIAハードウェアデコード (h264_cuvid) を使用 - 連続録画モード")
    else:
        print("[RTSP保存] ソフトウェアデコードを使用 - 連続録画モード")
    
    duration_secs = int(segment_duration)
    
    # 時間外の場合に表示するログを抑制するためのフラグ
    was_outside_time_range = True
    
    while cancel_flag is None or not cancel_flag.is_set():
        if time_limit_enabled:
            now = datetime.now()
            if not is_within_time_range(now, time_limit_enabled, start_hour, start_minute, end_hour, end_minute):
                if was_outside_time_range:
                    print(f"[RTSP保存] 録画時間外です ({start_hour:02d}:{start_minute:02d} - {end_hour:02d}:{end_minute:02d})。録画を一時停止中...")
                    was_outside_time_range = False
                # 30秒ごとにチェック
                for _ in range(30):
                    if cancel_flag is not None and cancel_flag.is_set():
                        break
                    time.sleep(1)
                continue
            elif not was_outside_time_range:
                # 時間内に戻った
                print(f"[RTSP保存] 録画時間帯に入りました。録画を再開します...")
                was_outside_time_range = True
        
        process = None
        try:
            # 出力ディレクトリ（日付別・時間別）を事前に作成
            # Windows では -strftime_mkdir が正しく動作しないため手動で作成
            now = datetime.now()
            session_start_date = now.strftime("%Y%m%d")  # 日付跨ぎ検出用
            base_dir = os.path.join(save_root, session_start_date)
            hour_dir = os.path.join(base_dir, now.strftime("%H"))
            os.makedirs(hour_dir, exist_ok=True)
            
            # セグメントファイル名パターン: YYYYMMDD/HH/MM.mp4
            # strftimeを使って時刻ベースのファイル名を生成
            segment_pattern = os.path.join(base_dir, "%H", "%M.mp4")
            
            # FFmpegコマンドを構築（segment muxer使用）
            ffmpeg_cmd = [ffmpeg_path, '-y']
            
            if use_cuvid:
                ffmpeg_cmd.extend(['-hwaccel', 'cuda', '-c:v', 'h264_cuvid'])
            
            if config.RTSP_USE_TCP:
                ffmpeg_cmd.extend(['-rtsp_transport', 'tcp'])
            
            # 入力と出力設定（segment muxer）
            ffmpeg_cmd.extend(['-i', rtsp_url])

            ffmpeg_cmd.extend([
                '-map', '0:v:0',
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-force_key_frames', f'expr:gte(t,n_forced*{duration_secs})',
                '-an',  # オーディオなし
                '-f', 'segment',  # セグメントmuxer
                '-segment_time', str(duration_secs),  # セグメント長（秒）
                '-segment_time_delta', '0.5',
                '-segment_format', 'mp4',
                '-reset_timestamps', '1',  # 各セグメントのタイムスタンプをリセット
                '-strftime', '1',  # strftimeでファイル名生成
                '-strftime_mkdir', '1',  # 必要なディレクトリを自動作成
                segment_pattern
            ])

            if preview_callback is not None:
                ffmpeg_cmd.extend([
                    '-map', '0:v:0',
                    '-vf', 'fps=3,scale=960:-2',
                    '-q:v', '5',
                    '-an',
                    '-f', 'mjpeg',
                    'pipe:1'
                ])
            
            print(f"[RTSP保存] 連続録画開始 (FFmpeg segment muxer): {segment_pattern}")
            
            # FFmpegプロセスを実行
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE if preview_callback is not None else subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )

            def read_preview_stdout():
                if preview_callback is None or process.stdout is None:
                    return
                buffer = bytearray()
                try:
                    while process.poll() is None and (cancel_flag is None or not cancel_flag.is_set()):
                        chunk = process.stdout.read(8192)
                        if not chunk:
                            break
                        buffer.extend(chunk)
                        while True:
                            start = buffer.find(b'\xff\xd8')
                            end = buffer.find(b'\xff\xd9', start + 2) if start != -1 else -1
                            if start == -1:
                                if len(buffer) > 1024 * 1024:
                                    del buffer[:-2]
                                break
                            if end == -1:
                                if start > 0:
                                    del buffer[:start]
                                break
                            jpg = bytes(buffer[start:end + 2])
                            del buffer[:end + 2]
                            img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                            if img is not None:
                                preview_callback(img)
                except Exception as e:
                    print(f"[RTSPライブプレビュー] プレビューフレーム読み取りエラー: {e}")
            
            # FFmpeg stderrを非同期で読み取るスレッド
            ffmpeg_errors = []
            ffmpeg_fatal_error = threading.Event()  # 致命的エラー検出用フラグ
            def read_ffmpeg_stderr():
                try:
                    for line in iter(process.stderr.readline, b''):
                        if line:
                            decoded = line.decode('utf-8', errors='ignore').strip()
                            if decoded:
                                lower_decoded = decoded.lower()
                                if "error while decoding mb" in lower_decoded or "concealing" in lower_decoded:
                                    continue
                                ffmpeg_errors.append(decoded)
                                # エラー関連のキーワードを含む場合のみ出力
                                if any(kw in lower_decoded for kw in ['error', 'fail', 'invalid', 'disconnect']):
                                    print(f"[FFmpeg] {decoded}")
                                # ネットワーク関連の致命的エラーを検出
                                # -10054: WSAECONNRESET (接続がリセットされた)
                                # -10053: WSAECONNABORTED (接続が中断された)
                                # -10060: WSAETIMEDOUT (接続がタイムアウト)
                                if any(err in decoded for err in ['-10054', '-10053', '-10060', 'Connection reset', 'demuxing']):
                                    print(f"[RTSP保存] 致命的ネットワークエラーを検出: {decoded}")
                                    ffmpeg_fatal_error.set()
                except Exception:
                    pass
            
            stderr_thread = threading.Thread(target=read_ffmpeg_stderr, daemon=True)
            stderr_thread.start()
            if preview_callback is not None:
                preview_thread = threading.Thread(target=read_preview_stdout, daemon=True)
                preview_thread.start()
            
            # ネットワーク監視変数
            start_time = time.time()
            last_total_size = 0
            last_max_mtime = 0  # 最新ファイルの更新時刻を追跡
            stall_start_time = None
            cancelled = False
            stalled = False
            paused_for_time_limit = False
            last_segment_count = 0
            
            print("[RTSP保存] ネットワーク監視を開始（連続録画中）...")
            
            # プロセス実行中は継続監視（キャンセルまたは切断検出まで継続）
            while process.poll() is None:
                elapsed = time.time() - start_time
                
                # キャンセルチェック
                if cancel_flag is not None and cancel_flag.is_set():
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    print("[RTSP保存] キャンセルされました")
                    cancelled = True
                    break

                if time_limit_enabled:
                    now_for_limit = datetime.now()
                    if not is_within_time_range(now_for_limit, True, start_hour, start_minute, end_hour, end_minute):
                        print(f"[RTSP保存] 録画終了時刻に達しました ({start_hour:02d}:{start_minute:02d} - {end_hour:02d}:{end_minute:02d})。FFmpeg録画を停止します。")
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        paused_for_time_limit = True
                        was_outside_time_range = False
                        break
                
                # 起動直後はスキップ（接続確立待ち）
                if elapsed < GRACE_PERIOD:
                    time.sleep(1)
                    continue
                
                # FFmpegの致命的エラーを検出した場合は即座に再接続
                if ffmpeg_fatal_error.is_set():
                    print("[RTSP保存] FFmpegで致命的エラーを検出。即座に再接続します...")
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    stalled = True
                    break
                
                # 現在の時間ディレクトリを監視（現在の時間フォルダのみ）
                # 時間が変わった場合は新しいディレクトリを作成
                current_now = datetime.now()
                current_date = current_now.strftime("%Y%m%d")
                
                # 日付が変わった場合は、新しい日付のディレクトリに保存するためにFFmpegを再起動
                if current_date != session_start_date:
                    print(f"[RTSP保存] 日付が変わりました ({session_start_date} -> {current_date})。FFmpegを再起動して新しい日付のディレクトリに録画します...")
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    break  # 外側のwhileループに戻り、新しい日付で再開
                
                current_hour_dir = os.path.join(base_dir, current_now.strftime("%H"))
                os.makedirs(current_hour_dir, exist_ok=True)  # 時間が変わっても対応
                current_total_size = 0
                current_segment_count = 0
                current_max_mtime = 0  # 最新ファイルの更新時刻
                try:
                    if os.path.isdir(current_hour_dir):
                        for f in os.listdir(current_hour_dir):
                            if f.endswith('.mp4'):
                                try:
                                    fpath = os.path.join(current_hour_dir, f)
                                    current_total_size += os.path.getsize(fpath)
                                    current_segment_count += 1
                                    # ファイルの更新時刻を取得
                                    mtime = os.path.getmtime(fpath)
                                    if mtime > current_max_mtime:
                                        current_max_mtime = mtime
                                except OSError:
                                    pass
                except Exception:
                    pass
                
                # 新しいセグメントが作られたらログ
                if current_segment_count > last_segment_count:
                    print(f"[RTSP保存] 新規セグメント検出 (計 {current_segment_count} ファイル, {current_total_size // 1024} KB)")
                    last_segment_count = current_segment_count
                    # 新しいセグメント作成 = 確実に活動中
                    stall_start_time = None
                    last_total_size = current_total_size
                    last_max_mtime = current_max_mtime
                elif current_total_size > last_total_size:
                    # ファイルサイズが増加 - 正常
                    last_total_size = current_total_size
                    last_max_mtime = current_max_mtime
                    stall_start_time = None
                elif current_max_mtime > last_max_mtime:
                    # サイズは増えていないが、更新時刻が更新されている - 正常
                    # (セグメント切り替え中でバッファリング中の可能性)
                    last_max_mtime = current_max_mtime
                    stall_start_time = None
                else:
                    # サイズも更新時刻も変化なし - 切断の可能性
                    if stall_start_time is None:
                        stall_start_time = time.time()
                    elif time.time() - stall_start_time >= STALL_TIMEOUT:
                        # 切断検出！
                        print(f"[RTSP保存] ネットワーク切断検出 ({STALL_TIMEOUT}秒間活動なし)。再接続します...")
                        process.terminate()
                        try:
                            process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        stalled = True
                        break
                
                time.sleep(1)
            
            if cancelled:
                break  # 外側のwhileループも抜ける

            if paused_for_time_limit:
                continue
            
            # 切断検出時 - すぐに再接続（stderrの読み取りでブロックしないように）
            if stalled:
                print("[RTSP保存] 再接続中...")
                time.sleep(1)  # 少し待機
                continue  # すぐに次のセッションを開始（再接続）
            
            # stderrを読み取り（デバッグ用）- プロセスが自然に終了した場合のみ
            stderr_output = ""
            if process.stderr:
                try:
                    # ノンブロッキングで読み取り（既にバッファにある分のみ）
                    import select
                    # Windows互換の方法でタイムアウト付き読み取り
                    stderr_output = "".join(ffmpeg_errors[-10:])  # 最後の10エラーを使用
                except Exception:
                    pass
            
            # FFmpegが予期せず終了した場合
            if process.returncode is not None and process.returncode != 0:
                if stderr_output:
                    print(f"[RTSP保存] FFmpegエラー: {stderr_output[-500:]}")
                print(f"[RTSP保存] FFmpegが終了しました (code={process.returncode})。再接続します...")
                continue
            
            # 正常終了（通常はここには来ない、segment muxerは継続動作のため）
            print("[RTSP保存] FFmpegが終了しました。再起動します...")
            time.sleep(2)  # 再接続前に少し待機
            continue  # 次のイテレーションで再接続
                
        except Exception as e:
            print(f"[RTSP保存] エラー: {e}")
            import traceback
            traceback.print_exc()
            if process is not None:
                try:
                    process.terminate()
                    process.wait(timeout=2)
                except Exception:
                    pass
            # エラー後も即座に再接続
            print("[RTSP保存] 再接続します...")
            time.sleep(2)
            continue
    
    print("[RTSP保存] 連続録画スレッドを終了します。")



def save_rtsp_video_segments(
    rtsp_url: str, save_root: str = config.RTSP_SAVE_ROOT,
    segment_duration: int = config.RTSP_SEGMENT_DURATION, cancel_flag: Optional[threading.Event] = None,
    time_limit_enabled: bool = False, start_hour: int = 17, start_minute: int = 0,
    end_hour: int = 7, end_minute: int = 0,
    preview_callback: Optional[Callable[[np.ndarray], None]] = None,
    dark_frame: Optional[np.ndarray] = None
):
    """
    RTSPストリームから設定されたフレーム数ごとに動画ファイルを保存する。
    NVIDIA GPU対応環境ではFFmpegモードを使用。
    
    ネットワーク監視機能:
    - 連続フレーム読み込み失敗で切断と判断
    - 切断検出時は即座に再接続（無限リトライ）
    """
    # NVIDIAハードウェアデコードが有効ならFFmpegモードを使用
    if config.RTSP_USE_NVIDIA_HWACCEL and dark_frame is None:
        import shutil
        if shutil.which("ffmpeg"):
            return save_rtsp_video_segments_ffmpeg(
                rtsp_url, save_root, segment_duration, cancel_flag,
                time_limit_enabled, start_hour, start_minute, end_hour, end_minute,
                preview_callback
            )
    
    # ネットワーク監視パラメータ
    MAX_READ_FAILURES = 30  # 連続30回失敗で切断と判断（約3秒 @0.1秒間隔）
    RECONNECT_WAIT = 2      # 再接続前の待機時間（秒）
    
    # 初回接続
    cap = None
    width = 0
    height = 0
    fps = config.RTSP_FPS
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    dark_frame_cache = None
    dark_frame_cache_shape = None

    def apply_dark_frame(frame: np.ndarray) -> np.ndarray:
        nonlocal dark_frame_cache, dark_frame_cache_shape
        if dark_frame is None:
            return frame
        if dark_frame_cache is None or dark_frame_cache_shape != frame.shape:
            if dark_frame.shape[:2] != frame.shape[:2]:
                dark_frame_cache = cv2.resize(dark_frame, (frame.shape[1], frame.shape[0]))
            else:
                dark_frame_cache = dark_frame
            if dark_frame_cache.dtype != np.uint8:
                dark_frame_cache = np.clip(dark_frame_cache, 0, 255).astype(np.uint8)
            dark_frame_cache_shape = frame.shape
        return cv2.subtract(frame, dark_frame_cache)
    
    def connect_rtsp():
        """RTSPストリームに接続。成功するまで無限リトライ"""
        nonlocal cap, width, height, fps
        while cancel_flag is None or not cancel_flag.is_set():
            print(f"[RTSP保存] ストリームに接続中: {rtsp_url}")
            cap = utils.create_rtsp_capture(rtsp_url)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                detected_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
                if detected_fps > 0:
                    fps = detected_fps
                else:
                    fps = config.RTSP_FPS
                print(f"[RTSP保存] ストリーム接続成功 ({width}x{height} @ {fps} fps)")
                return True
            print(f"[RTSP保存] 接続失敗。{RECONNECT_WAIT}秒後に再試行...")
            for _ in range(RECONNECT_WAIT):
                if cancel_flag is not None and cancel_flag.is_set():
                    return False
                time.sleep(1)
        return False
    
    # 時間外の場合に表示するログを抑制するためのフラグ
    was_outside_time_range_cv = True
    
    while cancel_flag is None or not cancel_flag.is_set():
        # 時間制限チェック（録画のみに適用）
        if time_limit_enabled:
            now = datetime.now()
            if not is_within_time_range(now, time_limit_enabled, start_hour, start_minute, end_hour, end_minute):
                if was_outside_time_range_cv:
                    print(f"[RTSP保存] 録画時間外です ({start_hour:02d}:{start_minute:02d} - {end_hour:02d}:{end_minute:02d})。録画を一時停止中...")
                    was_outside_time_range_cv = False
                if cap is not None and cap.isOpened():
                    cap.release()
                for _ in range(30):
                    if cancel_flag is not None and cancel_flag.is_set():
                        break
                    time.sleep(1)
                continue
            elif not was_outside_time_range_cv:
                print(f"[RTSP保存] 録画時間帯に入りました。録画を再開します...")
                was_outside_time_range_cv = True
        
        out = None
        temp_file_path = ""
        try:
            if cap is None or not cap.isOpened():
                if not connect_rtsp():
                    break
            segment_frames = max(1, int(round(fps * segment_duration)))

            now = datetime.now()
            dir_path = os.path.join(save_root, now.strftime("%Y%m%d"), now.strftime("%H"))
            os.makedirs(dir_path, exist_ok=True)

            minute_str = now.strftime("%M")
            base_file_path = os.path.join(dir_path, f"{minute_str}.mp4")
            temp_file_path = os.path.join(dir_path, f"{minute_str}_temp_{time.time_ns()}.mp4")

            out = cv2.VideoWriter(temp_file_path, fourcc, fps, (width, height))
            if not out.isOpened():
                print(f"エラー: VideoWriter を開けませんでした: {temp_file_path}")
                time.sleep(2)
                continue

            print(f"[RTSP保存] セグメント開始 (一時ファイル): {os.path.basename(temp_file_path)}")
            frames_written = 0
            stream_error = False
            paused_for_time_limit = False
            consecutive_read_failures = 0

            # フレーム数ベースで保存 (例: 720フレーム = 1分 @ 12fps)
            while frames_written < segment_frames:
                if cancel_flag is not None and cancel_flag.is_set():
                    stream_error = True
                    break

                if time_limit_enabled:
                    now_for_limit = datetime.now()
                    if not is_within_time_range(now_for_limit, True, start_hour, start_minute, end_hour, end_minute):
                        print(f"[RTSP保存] 録画終了時刻に達しました ({start_hour:02d}:{start_minute:02d} - {end_hour:02d}:{end_minute:02d})。現在のセグメントで録画を停止します。")
                        paused_for_time_limit = True
                        was_outside_time_range_cv = False
                        break

                ret, frame = cap.read()
                if not ret:
                    consecutive_read_failures += 1
                    if consecutive_read_failures >= MAX_READ_FAILURES:
                        # 切断検出！
                        print(f"[RTSP保存] ネットワーク切断検出 (連続 {MAX_READ_FAILURES} 回読み込み失敗)。即座に再接続します...")
                        stream_error = True
                        break
                    # 少し待って再試行
                    time.sleep(0.1)
                    continue
                
                # 成功時はリセット
                consecutive_read_failures = 0
                frame = apply_dark_frame(frame)
                if preview_callback is not None:
                    try:
                        preview_callback(frame)
                    except Exception as e:
                        print(f"[RTSPライブプレビュー] プレビューフレーム送信エラー: {e}")
                out.write(frame)
                frames_written += 1

            out.release()
            out = None

            if paused_for_time_limit:
                if os.path.exists(temp_file_path):
                    try:
                        file_size = os.path.getsize(temp_file_path)
                        if file_size > 10000 and frames_written > 10:
                            final_file_path = get_unique_file_path(base_file_path)
                            os.replace(temp_file_path, final_file_path)
                            print(f"[RTSP保存] 時間制限による部分セグメント保存: {final_file_path} ({frames_written} フレーム)")
                        else:
                            os.remove(temp_file_path)
                    except OSError:
                        pass
                if cap is not None and cap.isOpened():
                    cap.release()
                continue

            if stream_error:
                # 部分的なデータがあれば保存を試みる
                if os.path.exists(temp_file_path):
                    try:
                        file_size = os.path.getsize(temp_file_path)
                        if file_size > 10000 and frames_written > 10:  # 10KB以上かつ10フレーム以上
                            final_file_path = get_unique_file_path(base_file_path)
                            os.replace(temp_file_path, final_file_path)
                            print(f"[RTSP保存] 部分セグメント保存: {final_file_path} ({frames_written} フレーム)")
                        else:
                            os.remove(temp_file_path)
                    except OSError:
                        pass
                
                # キャンセルの場合は終了
                if cancel_flag is not None and cancel_flag.is_set():
                    break
                
                # 再接続
                if cap is not None and cap.isOpened():
                    cap.release()
                if not connect_rtsp():
                    break
                continue
                
            elif frames_written > 0:
                # 正常完了
                final_file_path = get_unique_file_path(base_file_path)
                os.replace(temp_file_path, final_file_path)
                print(f"[RTSP保存] セグメント保存完了: {final_file_path} ({frames_written} フレーム)")
            else:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

        except Exception as e:
            print(f"[RTSP保存] ループ中にエラー: {e}")
            import traceback
            traceback.print_exc()
            if out is not None and out.isOpened():
                out.release()
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except OSError:
                    pass
            
            # エラー後も即座に再接続
            print("[RTSP保存] 再接続します...")
            if cap is not None and cap.isOpened():
                cap.release()
            if not connect_rtsp():
                break
            continue

    if cap is not None and cap.isOpened():
        cap.release()
    print("[RTSP保存] 保存スレッドを終了します。")



def process_new_rtsp_files(
    rtsp_root: str = config.RTSP_SAVE_ROOT, processed_files_set: Set[str] = rtsp_processed_files,
    progress_callback: Optional[Callable[[Tuple[str, Optional[float]]], None]] = None,
    mask: Optional[np.ndarray] = None, global_wcs_info: Optional[Dict] = None,
    plate_solve_mask: Optional[np.ndarray] = None, meteor_save_path: str = config.DEFAULT_METEOR_SAVE_PATH,
    not_meteor_save_path: str = config.DEFAULT_NOT_METEOR_SAVE_PATH, cancel_flag: Optional[threading.Event] = None,
    save_options: Optional[Dict[str, bool]] = None, interval: float = config.DEFAULT_INTERVAL,
    duration: float = config.DEFAULT_DURATION, min_length: int = config.MIN_LINE_LENGTH,
    summary_video_config: Optional[List[Dict[str, Any]]] = None,
    max_workers: int = 1,
    time_limit_enabled: bool = False, start_hour: int = 17, start_minute: int = 0,
    end_hour: int = 7, end_minute: int = 0,
    notify_on_detection: bool = True
):
    new_files_to_process = []
    video_extensions = config.PERIODIC_VIDEO_EXTENSIONS
    
    # 現在の分のファイル名（書き込み中の可能性があるためスキップ）
    current_minute_filename = datetime.now().strftime("%M.mp4")
    current_time = time.time()
    # 書き込み中と判定する閾値（最終更新から5秒以内は書き込み中とみなす）
    WRITING_THRESHOLD_SECONDS = 5
    
    try:
        for root_dir, _, files in os.walk(rtsp_root):
            for file in files:
                if file.lower().endswith(video_extensions) and '_temp_' not in file:
                    # 現在書き込み中のファイルはスキップ
                    if file == current_minute_filename:
                        continue
                    full_path = os.path.join(root_dir, file)
                    if full_path not in processed_files_set:
                        if not is_rtsp_file_within_time_range(
                            full_path, time_limit_enabled, start_hour, start_minute, end_hour, end_minute
                        ):
                            print(f"[RTSP解析] スキップ (録画時間外): {full_path}")
                            processed_files_set.add(full_path)
                            continue
                        # ファイルの最終更新時刻をチェック
                        try:
                            file_mtime = os.path.getmtime(full_path)
                            time_since_modified = current_time - file_mtime
                            if time_since_modified < WRITING_THRESHOLD_SECONDS:
                                # 書き込み中の可能性が高いためスキップ
                                print(f"[RTSP解析] スキップ (書き込み中の可能性): {os.path.basename(full_path)} (更新 {time_since_modified:.1f}秒前)")
                                continue
                        except OSError:
                            # ファイルアクセスエラーの場合もスキップ
                            continue
                        new_files_to_process.append(full_path)
        new_files_to_process.sort(key=os.path.getmtime)

        if new_files_to_process:
            message = f"[RTSP解析] {len(new_files_to_process)} 個の新規保存動画ファイルを検出。(並列処理数: {max_workers})"
            print(message)
            if progress_callback: progress_callback((message, None))
            
            if max_workers > 1:
                # 並列処理モード
                from concurrent.futures import ThreadPoolExecutor, as_completed
                
                def process_one_file(fp):
                    if cancel_flag is not None and cancel_flag.is_set():
                        return fp, False
                    result = process_video_file_periodic(
                        fp, progress_callback, mask, global_wcs_info, plate_solve_mask,
                        meteor_save_path, not_meteor_save_path, cancel_flag, save_options,
                        interval, duration, min_length, summary_video_config, notify_on_detection
                    )
                    return fp, result
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(process_one_file, fp): fp for fp in new_files_to_process}
                    for future in as_completed(futures):
                        if cancel_flag is not None and cancel_flag.is_set():
                            break
                        try:
                            file_path, success = future.result()
                            if success:
                                processed_files_set.add(file_path)
                        except Exception as e:
                            print(f"[RTSP解析] 並列処理中に例外: {e}")
            else:
                # 逐次処理モード（従来どおり）
                for file_path in new_files_to_process:
                    if cancel_flag is not None and cancel_flag.is_set(): break
                    processed_successfully = process_video_file_periodic(
                         file_path, progress_callback, mask, global_wcs_info, plate_solve_mask,
                         meteor_save_path, not_meteor_save_path, cancel_flag, save_options,
                         interval, duration, min_length, summary_video_config, notify_on_detection
                    )
                    if processed_successfully:
                        processed_files_set.add(file_path)
        else:
             print("[RTSP解析] 解析対象の新規ファイルは見つかりませんでした。")
    except Exception as e:
        message = f"[RTSP解析] スキャンまたは処理中にエラー: {e}"
        print(f"エラー: {message}")
        if progress_callback: progress_callback((message, None))

def rtsp_save_and_process_thread_target(
    rtsp_url: str, save_root: str = config.RTSP_SAVE_ROOT,
    segment_duration: int = config.RTSP_SEGMENT_DURATION, scan_interval: int = 60,
    progress_callback: Optional[Callable[[Tuple[str, Optional[float]]], None]] = None,
    mask: Optional[np.ndarray] = None, global_wcs_info: Optional[Dict] = None,
    plate_solve_mask: Optional[np.ndarray] = None, meteor_save_path: str = config.DEFAULT_METEOR_SAVE_PATH,
    not_meteor_save_path: str = config.DEFAULT_NOT_METEOR_SAVE_PATH, cancel_flag: Optional[threading.Event] = None,
    save_options: Optional[Dict[str, bool]] = None, interval: float = config.DEFAULT_INTERVAL,
    duration: float = config.DEFAULT_DURATION, min_length: int = config.MIN_LINE_LENGTH,
    summary_video_config: Optional[List[Dict[str, Any]]] = None,
    time_limit_enabled: bool = False, start_hour: int = 17, start_minute: int = 0,
    end_hour: int = 7, end_minute: int = 0,
    max_workers: int = 1,
    preview_callback: Optional[Callable[[np.ndarray], None]] = None,
    dark_frame: Optional[np.ndarray] = None,
    notify_on_detection: bool = True
):
    global rtsp_processed_files
    video_extensions = config.PERIODIC_VIDEO_EXTENSIONS

    if os.path.isdir(save_root):
        try:
            initial_scan_message = f"[RTSP統合] 初回スキャン中: {save_root}"
            print(initial_scan_message)
            if progress_callback: progress_callback((initial_scan_message, None))
            for root, _, files in os.walk(save_root):
                for file in files:
                    if file.lower().endswith(video_extensions):
                        rtsp_processed_files.add(os.path.join(root, file))
            completion_message = f"[RTSP統合] 初回スキャン完了。既存の {len(rtsp_processed_files)} ファイルは処理対象外。"
            print(completion_message)
            if progress_callback: progress_callback((completion_message, None))
        except Exception as e:
            error_message = f"[RTSP統合] 初回スキャン中にエラー: {e}"
            print(f"エラー: {error_message}")
            if progress_callback: progress_callback((error_message, None))
    else:
        print(f"[RTSP統合] 保存ディレクトリが見つからないため、初回スキャンをスキップします: {save_root}")

    # 時間制限は録画とRTSP保存ファイルの解析対象判定に適用する
    save_thread = threading.Thread(
        target=save_rtsp_video_segments, 
        args=(rtsp_url, save_root, segment_duration, cancel_flag,
              time_limit_enabled, start_hour, start_minute, end_hour, end_minute, preview_callback, dark_frame), 
        daemon=True
    )
    save_thread.start()
    if time_limit_enabled:
        print(f"[RTSP統合] 保存スレッドを開始しました。録画時間制限: {start_hour:02d}:{start_minute:02d} - {end_hour:02d}:{end_minute:02d}")
    else:
        print("[RTSP統合] 保存スレッドを開始しました。")

    while cancel_flag is None or not cancel_flag.is_set():
        process_new_rtsp_files(
            save_root, rtsp_processed_files, progress_callback, mask, global_wcs_info, plate_solve_mask,
            meteor_save_path, not_meteor_save_path, cancel_flag, save_options,
            interval, duration, min_length, summary_video_config, max_workers,
            time_limit_enabled, start_hour, start_minute, end_hour, end_minute,
            notify_on_detection
        )
        wait_message = f"[RTSP統合] 解析スキャン完了。次のスキャンまで {scan_interval} 秒待機。"
        print(wait_message)
        for _ in range(scan_interval):
            if cancel_flag is not None and cancel_flag.is_set(): break
            time.sleep(1)

    print("[RTSP統合] 統合処理スレッドを終了します。")
