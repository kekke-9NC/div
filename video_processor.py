"""
動画連結モジュール

FFmpegを使用して複数の動画ファイルを連結する機能を提供します。
メモリ効率を重視し、subprocess経由でFFmpegを呼び出すストリーム処理を採用。
NVIDIA GPUが利用可能な場合はNVENCを使用してハードウェアエンコードを行う。
"""

import os
import subprocess
import tempfile
import json
import re
import time
from typing import List, Callable, Optional, Tuple

# NVENCの利用可否をキャッシュ
_nvenc_available: Optional[bool] = None


def get_ffmpeg_path() -> str:
    """FFmpegの実行パスを取得する"""
    return "ffmpeg"


def get_ffprobe_path() -> str:
    """FFprobeの実行パスを取得する"""
    return "ffprobe"


def check_nvenc_available() -> bool:
    """
    NVIDIA NVENCが利用可能かチェックする
    
    Returns:
        NVENCが利用可能な場合True
    """
    global _nvenc_available
    
    if _nvenc_available is not None:
        return _nvenc_available
    
    try:
        # FFmpegでNVENCエンコーダが利用可能かテスト
        cmd = [get_ffmpeg_path(), "-hide_banner", "-encoders"]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=10)
        
        if result.returncode == 0:
            output = result.stdout
            # h264_nvenc または hevc_nvenc が含まれていればNVENCが利用可能
            if "h264_nvenc" in output or "hevc_nvenc" in output:
                _nvenc_available = True
                return True
    except Exception:
        pass
    
    _nvenc_available = False
    return False


def get_video_info(video_path: str) -> Optional[dict]:
    """
    ffprobeを使って動画情報を取得する
    
    Args:
        video_path: 動画ファイルのパス
        
    Returns:
        動画情報の辞書、失敗時はNone
    """
    try:
        cmd = [
            get_ffprobe_path(),
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception as e:
        print(f"動画情報の取得に失敗: {e}")
    return None


def get_video_duration(video_path: str) -> float:
    """
    動画の長さ（秒）を取得する
    
    Args:
        video_path: 動画ファイルのパス
        
    Returns:
        動画の長さ（秒）、失敗時は0.0
    """
    info = get_video_info(video_path)
    if info and 'format' in info:
        try:
            return float(info['format'].get('duration', 0))
        except (ValueError, TypeError):
            pass
    return 0.0


def get_video_fps(video_path: str) -> float:
    """
    動画のフレームレート(FPS)を取得する
    
    Args:
        video_path: 動画ファイルのパス
        
    Returns:
        FPS（float）、失敗時は0.0
    """
    info = get_video_info(video_path)
    if info and 'streams' in info:
        for stream in info['streams']:
            if stream.get('codec_type') == 'video':
                # avg_frame_rate は "30/1" や "2997/100" の形式
                avg_fps = stream.get('avg_frame_rate', '0/0')
                if '/' in avg_fps:
                    try:
                        num, den = map(int, avg_fps.split('/'))
                        if den != 0:
                            return num / den
                    except (ValueError, ZeroDivisionError):
                        pass
                
                # avg_frame_rateがダメな場合は r_frame_rate を試す
                r_fps = stream.get('r_frame_rate', '0/0')
                if '/' in r_fps:
                    try:
                        num, den = map(int, r_fps.split('/'))
                        if den != 0:
                            return num / den
                    except (ValueError, ZeroDivisionError):
                        pass
    return 0.0


def get_total_duration(video_files: List[str]) -> float:
    """
    複数の動画の合計再生時間を取得する
    
    Args:
        video_files: 動画ファイルのリスト
        
    Returns:
        合計再生時間（秒）
    """
    total = 0.0
    for f in video_files:
        total += get_video_duration(f)
    return total


def format_time(seconds: float) -> str:
    """
    秒数を HH:MM:SS 形式にフォーマットする
    
    Args:
        seconds: 秒数
        
    Returns:
        フォーマットされた時間文字列
    """
    if seconds < 0:
        return "--:--:--"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def parse_ffmpeg_progress(line: str, total_duration: float) -> Optional[float]:
    """
    FFmpegの出力から進捗率を解析する
    
    Args:
        line: FFmpegの出力行
        total_duration: 合計再生時間（秒）
        
    Returns:
        進捗率（0.0〜1.0）、解析できない場合はNone
    """
    # time=00:01:23.45 形式をパース
    match = re.search(r'time=(\d+):(\d+):(\d+\.?\d*)', line)
    if match and total_duration > 0:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = float(match.group(3))
        current_time = hours * 3600 + minutes * 60 + seconds
        progress = min(current_time / total_duration, 1.0)
        return progress
    return None


def concatenate_videos(
    input_files: List[str],
    output_path: str,
    bitrate: str = "8000k",
    codec: str = "h264",
    fps: Optional[float] = None,
    safe_mode: bool = False,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None
) -> Tuple[bool, str]:
    """
    複数の動画ファイルを連結する
    
    FFmpegのconcat demuxerを使用してメモリ効率の良い連結を行う。
    NVIDIA GPUが利用可能な場合はNVENCを使用してハードウェアエンコードを行う。
    
    Args:
        input_files: 連結する動画ファイルのリスト（順序通りに連結）
        output_path: 出力ファイルのパス
        bitrate: 出力ビットレート（例: "8000k"）
        codec: コーデック（"h264" または "h265"）
        fps: 出力フレームレート（Noneの場合は自動）
        safe_mode: Trueの場合、一時的にMPEG-TS形式に変換してから連結する（タイムスタンプエラー回避用）
        progress_callback: 進捗コールバック(進捗率, メッセージ)
        cancel_check: キャンセル確認コールバック（Trueを返すとキャンセル）
        
    Returns:
        (成功フラグ, メッセージ)のタプル
    """
    if not input_files:
        return False, "連結する動画ファイルが指定されていません"
    
    if len(input_files) < 2:
        return False, "連結には2つ以上の動画ファイルが必要です"
    
    # NVENCの利用可否をチェック
    use_nvenc = check_nvenc_available()
    
    # コーデックの設定
    if codec.lower() == "h265" or codec.lower() == "hevc":
        if use_nvenc:
            video_codec = "hevc_nvenc"
            codec_params = ["-tag:v", "hvc1", "-preset", "p4", "-rc", "vbr"]
        else:
            video_codec = "libx265"
            codec_params = ["-tag:v", "hvc1"]
    else:
        if use_nvenc:
            video_codec = "h264_nvenc"
            codec_params = ["-preset", "p4", "-rc", "vbr"]
        else:
            video_codec = "libx264"
            codec_params = []
    
    # FPSの設定
    # -r は -fps_mode passthrough と競合するため、-vf filterを使用する
    video_filters = []
    if fps and fps > 0:
        video_filters = ["-vf", f"fps={fps}"]
    
    encoder_info = f"GPU (NVENC)" if use_nvenc else "CPU"
    
    # 入力ファイルの事前検証（有効なファイルのみをフィルタリング）
    if progress_callback:
        progress_callback(0.0, f"入力ファイルを検証中... ({len(input_files)}ファイル)")
    
    valid_files = []
    skipped_files = []
    total_duration = 0.0
    
    for i, f in enumerate(input_files):
        if not os.path.isfile(f):
            skipped_files.append((f, "ファイルが存在しません"))
            continue
        
        # ファイルサイズをチェック
        try:
            file_size = os.path.getsize(f)
            if file_size < 1000:  # 1KB未満は無効とみなす
                skipped_files.append((f, f"ファイルサイズが小さすぎます ({file_size} bytes)"))
                continue
        except OSError as e:
            skipped_files.append((f, f"ファイルアクセスエラー: {e}"))
            continue
        
        # ffprobeで動画の整合性を詳細にチェック
        try:
            probe_cmd = [
                get_ffprobe_path(),
                "-v", "error",  # エラーのみ表示
                "-select_streams", "v:0",  # 映像ストリームのみ
                "-show_entries", "stream=codec_type,duration,nb_frames",
                "-of", "json",
                f
            ]
            probe_result = subprocess.run(
                probe_cmd, 
                capture_output=True, 
                text=True, 
                encoding='utf-8', 
                errors='replace',
                timeout=30
            )
            
            # stderrにエラーがあればスキップ
            if probe_result.stderr and probe_result.stderr.strip():
                error_preview = probe_result.stderr.strip()[:100]
                skipped_files.append((f, f"ファイル破損の可能性: {error_preview}"))
                if progress_callback:
                    progress_callback(0.0, f"警告: {os.path.basename(f)} をスキップ（破損の可能性）")
                continue
            
            # 動画情報を取得できなければスキップ
            if probe_result.returncode != 0:
                skipped_files.append((f, f"ffprobeエラー (code: {probe_result.returncode})"))
                if progress_callback:
                    progress_callback(0.0, f"警告: {os.path.basename(f)} をスキップ（解析失敗）")
                continue
                
        except subprocess.TimeoutExpired:
            skipped_files.append((f, "ffprobe タイムアウト"))
            if progress_callback:
                progress_callback(0.0, f"警告: {os.path.basename(f)} をスキップ（タイムアウト）")
            continue
        except Exception as e:
            skipped_files.append((f, f"検証エラー: {e}"))
            if progress_callback:
                progress_callback(0.0, f"警告: {os.path.basename(f)} をスキップ（検証エラー）")
            continue
        
        valid_files.append(f)
        duration = get_video_duration(f)
        total_duration += duration
        
        # 進捗を更新（10ファイルごと）
        if progress_callback and i % 10 == 0:
            progress_callback(0.0, f"入力ファイルを検証中... ({i+1}/{len(input_files)})")
    
    # スキップされたファイルをログ出力
    if skipped_files and progress_callback:
        progress_callback(0.0, f"警告: {len(skipped_files)}ファイルをスキップしました")
    
    if len(valid_files) < 2:
        skip_msg = "\n".join([f"- {os.path.basename(f)}: {reason}" for f, reason in skipped_files[:10]])
        return False, f"有効な動画ファイルが2つ未満です（{len(valid_files)}ファイル）\nスキップされたファイル:\n{skip_msg}"
    
    if progress_callback:
        progress_callback(0.0, f"有効なファイル: {len(valid_files)}/{len(input_files)} (エンコーダ: {encoder_info})")
    
    # デバッグ用: 合計再生時間をログ出力
    if progress_callback:
        progress_callback(0.0, f"[DEBUG] 合計再生時間: {total_duration:.2f}秒 ({total_duration/60:.2f}分)")

    # ファイルごとの開始時間を計算（デバッグ用）
    file_schedule = []
    current_offset = 0.0
    for f in valid_files:
        dur = get_video_duration(f)
        file_schedule.append({
            'file': f,
            'start': current_offset,
            'end': current_offset + dur
        })
        current_offset += dur
    # 一番最後のファイルの終了時間がtotal_durationと一致しない場合の微修正（get_video_durationの誤差などで）は許容

    # セーフモード: すべてのファイルを一時MPEG-TSに変換
    ts_temp_files = []
    files_to_concat = valid_files # デフォルトは元のファイル

    if safe_mode:
        if progress_callback:
            progress_callback(0.0, "セーフモード: 一時ファイルを作成中（タイムスタンプ補正）...")
        
        try:
            ts_dir = tempfile.mkdtemp(prefix="safe_concat_")
            
            for i, vf in enumerate(valid_files):
                # キャンセル確認
                if cancel_check and cancel_check():
                    # Cleanup below will handle removing created files
                    pass

                ts_path = os.path.join(ts_dir, f"temp_{i:04d}.ts")
                
                # ストリームコピーでTS変換 (高速)
                # -bsf:v h264_mp4toannexb は自動で適用されることが多いが明示的に指定も可
                # ここではシンプルにコンテナ変換
                ts_cmd = [
                    get_ffmpeg_path(),
                    "-y", "-hide_banner", "-loglevel", "error",
                    "-i", vf,
                    "-c", "copy",
                    "-bsf:v", "h264_mp4toannexb", # H.264の場合必須に近い
                    "-f", "mpegts",
                    ts_path
                ]
                
                # H.265の場合は hevc_mp4toannexb
                # 入力のコーデックを簡易判定するか、あるいはFFmpegのオートに任せる
                # いったんオートで試すが、エラーが出るならフィルタ指定が必要
                # H.264/HEVC両対応のため、まずはフィルタなしで試行し、だめなら...
                # 実は -bsf:v h264_mp4toannexb はH264以外に使うとエラーになる可能性
                # 安全策: フィルタ指定なしで実行し、FFmpegの自動解決に任せる
                ts_cmd = [
                    get_ffmpeg_path(),
                    "-y", "-hide_banner", "-loglevel", "error",
                    "-i", vf,
                    "-c", "copy",
                    "-f", "mpegts",
                    ts_path
                ]

                res = subprocess.run(ts_cmd, capture_output=True, text=True, encoding='utf-8')
                
                # 変換成功チェック
                conversion_success = (res.returncode == 0)
                
                if not conversion_success:
                    if progress_callback:
                        progress_callback(0.0, f"警告: ストリームコピー失敗 ({os.path.basename(vf)}) -> 再エンコードを試行します...")
                    
                    # 再エンコードでの修復を試みる
                    # 高速化のため ultrafast を使用、画質より修復優先
                    repair_cmd = [
                        get_ffmpeg_path(),
                        "-y", "-hide_banner", "-loglevel", "error",
                        "-i", vf,
                        "-c:v", "libx264", "-preset", "ultrafast",
                        "-c:a", "aac",
                        "-f", "mpegts",
                        ts_path
                    ]
                    
                    res_repair = subprocess.run(repair_cmd, capture_output=True, text=True, encoding='utf-8')
                    if res_repair.returncode == 0:
                        conversion_success = True
                        if progress_callback:
                            progress_callback(0.0, f"修復成功: {os.path.basename(vf)}")
                    else:
                        if progress_callback:
                            progress_callback(0.0, f"エラー: ファイルの修復に失敗しました ({os.path.basename(vf)}) -> このファイルをスキップします")
                
                if conversion_success:
                    ts_temp_files.append(ts_path)
                
                if progress_callback and i % 5 == 0:
                     progress_callback(0.0, f"セーフモード: 変換中 ({i+1}/{len(valid_files)})")
            
            # 変換できたファイルのみを連結対象とする
            if ts_temp_files:
                files_to_concat = ts_temp_files
                if progress_callback:
                    progress_callback(0.0, f"セーフモード: 変換完了 ({len(ts_temp_files)}/{len(valid_files)}ファイル)")
            else:
                if progress_callback:
                    progress_callback(0.0, "エラー: 有効なファイルが一つも変換できませんでした -> 元ファイルを使用します")
                files_to_concat = valid_files

        except Exception as e:
            if progress_callback:
                progress_callback(0.0, f"セーフモード初期化エラー: {e} -> 通常モードで続行")
            files_to_concat = valid_files

    
    # concat demuxer用の一時ファイルを作成
    # デバッグ用: 一時ファイルを出力先フォルダにも保存
    debug_concat_list_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            concat_list_path = f.name
            for video_file in files_to_concat:
                # パスを正規化（バックスラッシュをスラッシュに統一）
                normalized_path = video_file.replace('\\', '/')
                # パス内のシングルクォートをエスケープ
                safe_path = normalized_path.replace("'", "'\\''")
                f.write(f"file '{safe_path}'\n")
        
        # デバッグ用: 出力先フォルダにconcat listを保存
        try:
            output_dir = os.path.dirname(output_path)
            debug_concat_list_path = os.path.join(output_dir, "debug_concat_list.txt")
            import shutil
            shutil.copy(concat_list_path, debug_concat_list_path)
            if progress_callback:
                progress_callback(0.0, f"[DEBUG] concat listを保存: {debug_concat_list_path}")
        except Exception as e:
            if progress_callback:
                progress_callback(0.0, f"[DEBUG] concat list保存に失敗: {e}")
                
    except Exception as e:
        return False, f"一時ファイルの作成に失敗しました: {e}"
    
    # エラーログを収集する変数
    error_lines = []
    all_output_lines = []  # デバッグ用: 全出力を記録
    
    try:
        # FFmpegコマンドを構築（DTS/PTSタイムスタンプ問題を修正するオプション追加）
        cmd = [
            get_ffmpeg_path(),
            "-y",  # 上書き確認をスキップ
            "-loglevel", "verbose", # デバッグログを詳細に出力
            "-err_detect", "ignore_err",  # エラーを無視して続行
            "-fflags", "+genpts+igndts+discardcorrupt",  # タイムスタンプを再生成、破損フレームを破棄
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list_path,
            "-fps_mode", "passthrough",  # フレームレートモードをパススルーに変更（DTSエラー回避）
            # "-async", "1",  # 廃止されたため削除
            "-max_muxing_queue_size", "9999",  # muxingキューサイズを大幅に増加
            "-c:v", video_codec,
            "-b:v", bitrate,
            *codec_params,
            *video_filters,
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            "-progress", "pipe:1",
            output_path
        ]
        
        # デバッグ用: コマンドをログ出力
        if progress_callback:
            cmd_str = ' '.join(cmd)
            progress_callback(0.0, f"[DEBUG] FFmpegコマンド: {cmd_str[:200]}...")
        
        if progress_callback:
            progress_callback(0.0, f"連結処理を開始中... (エンコーダ: {encoder_info})")
        
        # FFmpegを実行（stderrを別途キャプチャ）
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            encoding='utf-8',
            errors='replace'
        )
        
        # 進捗を監視
        start_time = time.time()
        last_reported_percent = -1  # 最後に報告したパーセント（1%単位）
        
        import threading
        
        # stderrを別スレッドで読み取る
        def read_stderr():
            for line in process.stderr:
                error_lines.append(line.strip())
        
        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        stderr_thread.start()
        
        while True:
            # キャンセル確認
            if cancel_check and cancel_check():
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                return False, "処理がキャンセルされました"
            
            line = process.stdout.readline()
            if not line:
                if process.poll() is not None:
                    break
                continue
            
            # 進捗を解析
            if progress_callback:
                progress = parse_ffmpeg_progress(line, total_duration)
                if progress is not None:
                    current_percent = int(progress * 100)
                    
                    # 1%単位でのみ報告
                    if current_percent > last_reported_percent:
                        last_reported_percent = current_percent
                        
                        # 経過時間と残り時間を計算
                        elapsed = time.time() - start_time
                        if progress > 0:
                            estimated_total = elapsed / progress
                            remaining = estimated_total - elapsed
                        else:
                            remaining = -1
                        
                        elapsed_str = format_time(elapsed)
                        remaining_str = format_time(remaining)
                        
                        # 現在処理中のファイルを特定
                        current_file_name = ""
                        # parse_ffmpeg_progressで秒数が取れないため再計算
                        match_time = re.search(r'time=(\d+):(\d+):(\d+\.?\d*)', line)
                        if match_time:
                            h, m, s = int(match_time.group(1)), int(match_time.group(2)), float(match_time.group(3))
                            curr_secs = h * 3600 + m * 60 + s
                            
                            for item in file_schedule:
                                if item['start'] <= curr_secs < item['end']:
                                    current_file_name = f" | File: {os.path.basename(item['file'])}"
                                    break
                                    
                        progress_callback(
                            progress, 
                            f"連結中: {current_percent}% | 経過: {elapsed_str} | 残り: {remaining_str}{current_file_name}"
                        )
        
        # stderrスレッドの終了を待つ
        stderr_thread.join(timeout=5)
        
        # 終了コードを確認
        return_code = process.wait()
        
        elapsed = time.time() - start_time
        elapsed_str = format_time(elapsed)
        
        # デバッグ用: stderrを出力先フォルダにも保存
        try:
            output_dir = os.path.dirname(output_path)
            debug_stderr_path = os.path.join(output_dir, "debug_ffmpeg_stderr.txt")
            with open(debug_stderr_path, 'w', encoding='utf-8') as f:
                f.write(f"=== FFmpeg Stderr Log ===\n")
                f.write(f"Total input files: {len(valid_files)}\n")
                f.write(f"Expected duration: {total_duration:.2f}s\n")
                f.write(f"Return code: {return_code}\n")
                f.write(f"=== Error lines ({len(error_lines)}) ===\n")
                for line in error_lines:
                    f.write(line + '\n')
            if progress_callback:
                progress_callback(1.0, f"[DEBUG] stderrログを保存: {debug_stderr_path}")
        except Exception as e:
            if progress_callback:
                progress_callback(1.0, f"[DEBUG] stderrログ保存に失敗: {e}")
        
        if return_code == 0:
            # 出力ファイルの長さを確認
            output_duration = get_video_duration(output_path)
            duration_ratio = output_duration / total_duration if total_duration > 0 else 0
            
            # デバッグ用: 期待値と実際の値を表示
            if progress_callback:
                progress_callback(1.0, f"[DEBUG] 期待再生時間: {total_duration:.2f}秒, 実際: {output_duration:.2f}秒 ({duration_ratio*100:.1f}%)")
            
            if duration_ratio < 0.9:  # 90%未満の場合は警告
                # 問題のありそうなファイルを特定
                suspicious_file = "不明"
                if len(file_schedule) > 0:
                    for item in file_schedule:
                        if item['start'] <= output_duration < item['end']:
                            suspicious_file = f"{os.path.basename(item['file'])} (範囲: {item['start']:.1f}s - {item['end']:.1f}s)"
                            break
                    # もし最後のファイルまで行っていれば
                    if output_duration >= file_schedule[-1]['start']:
                         suspicious_file = f"{os.path.basename(file_schedule[-1]['file'])} (最後のファイル)"

                warning_msg = f"警告: 出力ファイルの長さが期待より短いです（{duration_ratio*100:.1f}%）\n停止位置: {output_duration:.1f}秒\n疑わしいファイル: {suspicious_file}"
                
                if error_lines:
                    # 最後の数行のエラーを追加
                    recent_errors = error_lines[-10:] if len(error_lines) > 10 else error_lines
                    warning_msg += f"\nFFmpegエラー(抜粋): {'; '.join(recent_errors)}"
                    
                if progress_callback:
                    progress_callback(1.0, f"連結完了（警告あり） (処理時間: {elapsed_str})")
                
                return True, f"{warning_msg}\n出力パス: {output_path}\n詳細ログ: {debug_stderr_path}"
            
            if progress_callback:
                progress_callback(1.0, f"連結完了 (処理時間: {elapsed_str})")
            return True, f"連結が完了しました: {output_path}"
        else:
            # エラー詳細を取得
            error_detail = ""
            if error_lines:
                recent_errors = error_lines[-10:] if len(error_lines) > 10 else error_lines
                error_detail = "\n".join(recent_errors)
            return False, f"FFmpegがエラーで終了しました（コード: {return_code}）\n{error_detail}"
            
    except Exception as e:
        return False, f"連結処理中にエラーが発生しました: {e}"
    finally:
        # 一時ファイルを削除（デバッグ用にconcat_listは残す）
        try:
            os.unlink(concat_list_path)
        except:
            pass
        
        # セーフモードの一時ファイルを削除
        for ts_file in ts_temp_files:
            try:
                os.unlink(ts_file)
            except:
                pass
        # 一時ディレクトリ削除
        if ts_temp_files:
            try:
                os.rmdir(os.path.dirname(ts_temp_files[0]))
            except:
                pass


def get_supported_video_extensions() -> List[str]:
    """
    サポートされている動画ファイルの拡張子を返す
    
    Returns:
        拡張子のリスト（小文字、ドット付き）
    """
    return ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.ts', '.mts', '.m2ts']


def is_video_file(filepath: str) -> bool:
    """
    ファイルが動画ファイルかどうかを判定する
    
    Args:
        filepath: ファイルパス
        
    Returns:
        動画ファイルの場合True
    """
    ext = os.path.splitext(filepath)[1].lower()
    return ext in get_supported_video_extensions()

