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
import shutil
import math
from collections import deque
from datetime import datetime, timedelta
from typing import List, Callable, Optional, Tuple, Dict

from PIL import Image, ImageDraw, ImageFont

import video_enhancement
import media_time

# NVENCの利用可否をキャッシュ
_nvenc_available: Optional[bool] = None
_videotoolbox_available = {}
SAFE_CONCAT_DECODER_THREADS = 4


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


def check_videotoolbox_available(codec: str) -> bool:
    """Return whether macOS VideoToolbox can actually encode the codec."""
    normalized = "hevc" if codec.lower() in ("h265", "hevc") else "h264"
    if normalized in _videotoolbox_available:
        return _videotoolbox_available[normalized]

    encoder = f"{normalized}_videotoolbox"
    try:
        # Listing the encoder is insufficient on some Macs, so perform a tiny
        # real encode once and cache the result.
        result = subprocess.run(
            [
                get_ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.1",
                "-frames:v", "1", "-c:v", encoder, "-f", "null", "-",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        available = result.returncode == 0
    except Exception:
        available = False

    _videotoolbox_available[normalized] = available
    return available


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


def _normalize_concat_timestamp_settings(settings: Optional[Dict]) -> Dict:
    settings = settings or {}
    position_map = {
        "右下": "bottom_right", "左下": "bottom_left",
        "右上": "top_right", "左上": "top_left",
    }
    try:
        size_percent = float(settings.get("size_percent", 1.8))
    except (TypeError, ValueError):
        size_percent = 1.8
    try:
        offset_seconds = float(settings.get("offset_seconds", 0.0))
    except (TypeError, ValueError):
        offset_seconds = 0.0
    raw_position = str(settings.get("position", "bottom_right"))
    return {
        "enabled": bool(settings.get("enabled", False)),
        "position": position_map.get(raw_position, raw_position),
        "size_percent": max(0.8, min(4.0, size_percent)),
        "offset_seconds": offset_seconds,
    }


def _build_concat_schedule(source_files: List[str], offset_seconds: float = 0.0) -> List[Dict]:
    schedule = []
    output_offset = 0.0
    fallback_start = None
    for source in source_files:
        duration = get_video_duration(source)
        start_time, time_source = media_time.get_media_start_time(source)
        if start_time is None:
            start_time = fallback_start or datetime.now()
            time_source = "直前区間から推定"
        start_time += timedelta(seconds=offset_seconds)
        schedule.append({
            "file": source,
            "start": output_offset,
            "end": output_offset + duration,
            "duration": duration,
            "source_start_time": start_time,
            "time_source": time_source,
        })
        output_offset += duration
        fallback_start = start_time + timedelta(seconds=duration)
    return schedule


def _timestamp_for_output_second(schedule: List[Dict], output_second: float) -> datetime:
    for item in schedule:
        if item["start"] <= output_second < item["end"]:
            return item["source_start_time"] + timedelta(seconds=output_second - item["start"])
    if schedule:
        last = schedule[-1]
        return last["source_start_time"] + timedelta(seconds=max(0.0, output_second - last["start"]))
    return datetime.now()


def _load_timestamp_font(font_size: int):
    candidates = [
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "C:/Windows/Fonts/consola.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, font_size)
            except OSError:
                pass
    return ImageFont.load_default()


def _create_timestamp_overlay(
    schedule: List[Dict],
    width: int,
    height: int,
    settings: Dict,
    temp_dir: str,
    cancel_check=None,
    process_callback=None,
) -> Tuple[str, str]:
    """Create a compact 1fps timestamp video and return path/overlay position."""
    font_size = max(12, int(round(height * settings["size_percent"] / 100.0)))
    font = _load_timestamp_font(font_size)
    timezone_label = media_time.local_timezone_label()
    sample_text = f"2000-00-00 00:00:00 {timezone_label}"
    sample_box = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), sample_text, font=font)
    padding_x = max(8, font_size // 2)
    padding_y = max(5, font_size // 3)
    overlay_width = sample_box[2] - sample_box[0] + padding_x * 2
    overlay_height = sample_box[3] - sample_box[1] + padding_y * 2
    overlay_path = os.path.join(temp_dir, "timestamp_overlay.mkv")
    command = [
        get_ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{overlay_width}x{overlay_height}", "-r", "1", "-i", "-",
        "-an", "-c:v", "ffv1", "-level", "3", overlay_path,
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    if process_callback:
        process_callback(process)
    total_seconds = max(1, int(math.ceil(schedule[-1]["end"] if schedule else 1.0)) + 1)
    try:
        for second in range(total_seconds):
            if cancel_check and cancel_check():
                process.terminate()
                process.wait(timeout=3)
                raise RuntimeError("処理がキャンセルされました")
            timestamp = _timestamp_for_output_second(schedule, float(second))
            text = f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} {timezone_label}"
            frame = Image.new("RGB", (overlay_width, overlay_height), (0, 0, 0))
            draw = ImageDraw.Draw(frame)
            draw.text((padding_x, padding_y - sample_box[1]), text, font=font, fill=(255, 255, 255))
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"タイムスタンプ映像の作成に失敗しました: {stderr[-500:]}")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if process_callback:
            process_callback(None)

    margin = max(8, int(round(height * 0.01)))
    positions = {
        "top_left": f"{margin}:{margin}",
        "top_right": f"W-w-{margin}:{margin}",
        "bottom_left": f"{margin}:H-h-{margin}",
        "bottom_right": f"W-w-{margin}:H-h-{margin}",
    }
    return overlay_path, positions.get(settings["position"], positions["bottom_right"])


def _write_concat_timeline(output_path: str, schedule: List[Dict], settings: Dict) -> str:
    timeline_path = os.path.splitext(output_path)[0] + ".timeline.json"
    payload = {
        "version": 1,
        "timezone": media_time.local_timezone_label(),
        "output_start_time": schedule[0]["source_start_time"].isoformat() if schedule else None,
        "timestamp_offset_seconds": settings["offset_seconds"],
        "segments": [
            {
                "source_file": item["file"],
                "output_start_seconds": item["start"],
                "output_end_seconds": item["end"],
                "source_start_time": item["source_start_time"].isoformat(),
                "time_source": item["time_source"],
            }
            for item in schedule
        ],
    }
    with open(timeline_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)
    return timeline_path


def _build_independent_concat_filter(
    input_count: int,
    fps: Optional[float],
    timestamp_overlay_position: Optional[str] = None,
) -> Tuple[str, str]:
    """Build a filter that resets and concatenates independently decoded files."""
    if input_count < 1:
        raise ValueError("連結する入力動画がありません")

    chains = [
        f"[{index}:v]settb=AVTB,setpts=PTS-STARTPTS[segment{index}]"
        for index in range(input_count)
    ]
    segment_inputs = "".join(f"[segment{index}]" for index in range(input_count))
    chains.append(f"{segment_inputs}concat=n={input_count}:v=1:a=0[joined]")

    base_label = "joined"
    if fps and fps > 0:
        chains.append(f"[joined]fps={fps}[base]")
        base_label = "base"

    output_label = base_label
    if timestamp_overlay_position:
        overlay_index = input_count
        chains.append(f"[{overlay_index}:v]setpts=PTS-STARTPTS[clock]")
        chains.append(
            f"[{base_label}][clock]overlay={timestamp_overlay_position}:shortest=1[v]"
        )
        output_label = "v"

    return ";".join(chains), output_label


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


def parse_quality_metrics(text: str) -> Tuple[float, float]:
    """Parse FFmpeg SSIM and PSNR summaries."""
    ssim_matches = re.findall(r"All:([0-9.]+)", text)
    psnr_matches = re.findall(r"average:([0-9.]+)", text)
    ssim = float(ssim_matches[-1]) if ssim_matches else 0.0
    psnr = float(psnr_matches[-1]) if psnr_matches else 0.0
    return ssim, psnr


def automatic_bitrate_candidates(codec: str, width: int, height: int, fps: float) -> List[str]:
    """Return a small ascending benchmark set scaled to resolution and FPS."""
    scale = max(0.35, (width * height * max(fps, 1.0)) / (1920 * 1080 * 25.0))
    base = [750, 1000, 1500, 2000, 3000, 4000, 6000, 8000, 10000, 12000]
    if codec.lower() in ("h265", "hevc"):
        base = [500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 8000]
    values = sorted({max(500, int(round(value * scale / 250) * 250)) for value in base})
    return [f"{value}k" for value in values]


def benchmark_automatic_bitrate(
    source: str,
    codec: str,
    fps: Optional[float] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Tuple[str, float, float]:
    """Quickly choose the lowest bitrate meeting conservative SSIM/PSNR targets."""
    info = get_video_info(source) or {}
    stream = next((item for item in info.get("streams", []) if item.get("codec_type") == "video"), {})
    width = int(stream.get("width", 1920) or 1920)
    height = int(stream.get("height", 1080) or 1080)
    effective_fps = fps or get_video_fps(source) or 25.0
    duration = get_video_duration(source)
    sample_duration = min(2.0, max(0.5, duration))
    sample_start = max(0.0, (duration - sample_duration) / 2.0)
    candidates = automatic_bitrate_candidates(codec, width, height, effective_fps)
    encoder = "libx265" if codec.lower() in ("h265", "hevc") else "libx264"
    selected = candidates[-1]
    selected_ssim = 0.0
    selected_psnr = 0.0
    benchmark_results: List[Tuple[str, float, float]] = []

    with tempfile.TemporaryDirectory(prefix="bitrate_benchmark_") as temp_dir:
        reference = os.path.join(temp_dir, "reference.mkv")
        extract = subprocess.run(
            [
                get_ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{sample_start:.3f}", "-t", f"{sample_duration:.3f}",
                "-i", source, "-an", "-c:v", "ffv1", reference,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if extract.returncode != 0:
            raise RuntimeError(f"自動ビットレート用サンプルを作成できません: {extract.stderr[-300:]}")

        for index, candidate in enumerate(candidates, 1):
            if progress_callback:
                progress_callback(0.0, f"自動ビットレート試験 {index}/{len(candidates)}: {candidate}")
            encoded = os.path.join(temp_dir, f"candidate_{candidate}.mp4")
            encode = subprocess.run(
                [
                    get_ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
                    "-i", reference, "-an", "-c:v", encoder, "-preset", "fast",
                    "-b:v", candidate, "-pix_fmt", "yuv420p", encoded,
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if encode.returncode != 0:
                continue
            ssim_run = subprocess.run(
                [get_ffmpeg_path(), "-hide_banner", "-i", reference, "-i", encoded,
                 "-lavfi", "ssim", "-f", "null", "-"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            psnr_run = subprocess.run(
                [get_ffmpeg_path(), "-hide_banner", "-i", reference, "-i", encoded,
                 "-lavfi", "psnr", "-f", "null", "-"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            ssim, _ = parse_quality_metrics(ssim_run.stderr)
            _, psnr = parse_quality_metrics(psnr_run.stderr)
            benchmark_results.append((candidate, ssim, psnr))
            if progress_callback:
                progress_callback(0.0, f"  品質: SSIM={ssim:.4f}, PSNR={psnr:.2f}dB")
        if benchmark_results:
            # Use the highest tested bitrate as the local visual-quality ceiling,
            # then choose the smallest bitrate whose loss is practically negligible.
            _, ceiling_ssim, ceiling_psnr = benchmark_results[-1]
            selected, selected_ssim, selected_psnr = benchmark_results[-1]
            for candidate, ssim, psnr in benchmark_results:
                if (
                    ssim >= max(0.975, ceiling_ssim - 0.006)
                    and psnr >= max(40.0, ceiling_psnr - 3.0)
                ):
                    selected, selected_ssim, selected_psnr = candidate, ssim, psnr
                    break
    return selected, selected_ssim, selected_psnr


def concatenate_videos(
    input_files: List[str],
    output_path: str,
    bitrate: str = "8000k",
    codec: str = "h264",
    fps: Optional[float] = None,
    safe_mode: bool = False,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    process_callback: Optional[Callable[[Optional[subprocess.Popen]], None]] = None,
    apply_enhancement: bool = False,
    fixed_pattern_path: Optional[str] = None,
    timestamp_settings: Optional[Dict] = None,
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
        safe_mode: Trueの場合、入力タイムスタンプを再生成して連結する
        progress_callback: 進捗コールバック(進捗率, メッセージ)
        cancel_check: キャンセル確認コールバック（Trueを返すとキャンセル）
        process_callback: 実行中FFmpegプロセスの開始・終了通知
        timestamp_settings: 実時刻の焼き込み設定
        
    Returns:
        (成功フラグ, メッセージ)のタプル
    """
    if not input_files:
        return False, "連結する動画ファイルが指定されていません"
    
    if len(input_files) < 2:
        return False, "連結には2つ以上の動画ファイルが必要です"

    timestamp_settings = _normalize_concat_timestamp_settings(timestamp_settings)
    
    # ハードウェアエンコーダの利用可否をチェック
    use_nvenc = check_nvenc_available()
    use_videotoolbox = not use_nvenc and check_videotoolbox_available(codec)
    
    # コーデックの設定
    if codec.lower() == "h265" or codec.lower() == "hevc":
        if use_nvenc:
            video_codec = "hevc_nvenc"
            codec_params = ["-tag:v", "hvc1", "-preset", "p4", "-rc", "vbr"]
        elif use_videotoolbox:
            video_codec = "hevc_videotoolbox"
            codec_params = ["-tag:v", "hvc1", "-realtime", "1", "-prio_speed", "1"]
        else:
            video_codec = "libx265"
            codec_params = ["-tag:v", "hvc1"]
    else:
        if use_nvenc:
            video_codec = "h264_nvenc"
            codec_params = ["-preset", "p4", "-rc", "vbr"]
        elif use_videotoolbox:
            video_codec = "h264_videotoolbox"
            codec_params = ["-realtime", "1", "-prio_speed", "1"]
        else:
            video_codec = "libx264"
            codec_params = []
    
    # FPSの設定
    # -r は -fps_mode passthrough と競合するため、-vf filterを使用する
    video_filters = []
    if fps and fps > 0:
        video_filters = ["-vf", f"fps={fps}"]
    
    if use_nvenc:
        encoder_info = "GPU (NVENC)"
    elif use_videotoolbox:
        encoder_info = "Apple VideoToolbox (ハードウェア)"
    else:
        encoder_info = "CPU (自動マルチスレッド)"
    # FFmpeg/libx264/libx265 normally selects this automatically, but make it
    # explicit so CPU encoding is not accidentally constrained to one thread.
    encoder_thread_params = [] if (use_nvenc or use_videotoolbox) else ["-threads", "0"]
    
    # 入力ファイルの事前検証（有効なファイルのみをフィルタリング）
    if progress_callback:
        progress_callback(0.0, f"入力ファイルを検証中... ({len(input_files)}ファイル)")
    
    valid_files = []
    skipped_files = []
    total_duration = 0.0
    
    for i, f in enumerate(input_files):
        if cancel_check and cancel_check():
            return False, "処理がキャンセルされました"
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

    enhancement_temp_dir = None
    enhancement_temp_files: List[str] = []
    base_files = valid_files
    if apply_enhancement:
        if not fixed_pattern_path or not os.path.isfile(fixed_pattern_path):
            return False, "固定パターン＋21フレーム平均を適用するには、有効な補正マップが必要です"
        try:
            correction = video_enhancement.load_fixed_correction(fixed_pattern_path)
            enhancement_temp_dir = tempfile.mkdtemp(prefix="enhanced_concat_")
            for index, source_file in enumerate(valid_files):
                if cancel_check and cancel_check():
                    if enhancement_temp_dir:
                        shutil.rmtree(enhancement_temp_dir, ignore_errors=True)
                    return False, "処理がキャンセルされました"
                output_file = os.path.join(enhancement_temp_dir, f"enhanced_{index:04d}.mp4")
                if progress_callback:
                    progress_callback(
                        0.0,
                        f"保存物補正 {index + 1}/{len(valid_files)}: {os.path.basename(source_file)}",
                    )

                def enhancement_progress(done, total, item=index):
                    if progress_callback and (done == total or done % 50 == 0):
                        progress_callback(
                            0.0,
                            f"21フレーム平均中 {item + 1}/{len(valid_files)}: {done}/{total}",
                        )

                strength = video_enhancement.enhance_video_file(
                    source_file,
                    output_file,
                    correction,
                    progress_callback=enhancement_progress,
                )
                enhancement_temp_files.append(output_file)
                if progress_callback:
                    progress_callback(0.0, f"適応固定パターン強度: {strength:.3f}倍")
            base_files = enhancement_temp_files
        except Exception as exc:
            if enhancement_temp_dir:
                shutil.rmtree(enhancement_temp_dir, ignore_errors=True)
            return False, f"固定パターン＋21フレーム平均の前処理に失敗しました: {exc}"

    automatic_bitrate_summary = ""
    if str(bitrate).strip().lower() == "auto":
        try:
            bitrate, benchmark_ssim, benchmark_psnr = benchmark_automatic_bitrate(
                base_files[0], codec, fps=fps, progress_callback=progress_callback
            )
            automatic_bitrate_summary = (
                f"自動ビットレート={bitrate} "
                f"(SSIM={benchmark_ssim:.4f}, PSNR={benchmark_psnr:.2f}dB)"
            )
            if progress_callback:
                progress_callback(0.0, automatic_bitrate_summary)
        except Exception as exc:
            bitrate = "8000k"
            automatic_bitrate_summary = f"自動試験失敗のため8000kを使用: {exc}"
            if progress_callback:
                progress_callback(0.0, automatic_bitrate_summary)
    
    # デバッグ用: 合計再生時間をログ出力
    if progress_callback:
        progress_callback(0.0, f"[DEBUG] 合計再生時間: {total_duration:.2f}秒 ({total_duration/60:.2f}分)")

    # The final job already decodes and re-encodes every frame with regenerated
    # timestamps. Remuxing first is redundant and can hide per-file MPEG-4 VOL
    # header changes when RTSP FPS shifts (for example 25.0 -> 24.963).
    files_to_concat = base_files # デフォルトは元ファイルまたは補正済み一時ファイル
    concat_source_files = list(valid_files)

    if safe_mode and progress_callback:
        progress_callback(
            0.0,
            "セーフモード: ファイル別デコードで連結（PTS再生成・4スレッド）",
        )

    file_schedule = _build_concat_schedule(
        concat_source_files, timestamp_settings["offset_seconds"]
    )
    total_duration = file_schedule[-1]["end"] if file_schedule else 0.0

    
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
    # A corrupted RTSP recording can emit hundreds of thousands of decoder
    # messages. Keep the useful tail without letting a diagnostic log exhaust
    # memory or create a huge file.
    error_lines = deque(maxlen=2000)
    error_line_count = 0
    all_output_lines = []  # デバッグ用: 全出力を記録
    timestamp_temp_dir = None
    timestamp_overlay_path = None
    timestamp_overlay_position = None

    if timestamp_settings["enabled"]:
        try:
            info = get_video_info(files_to_concat[0]) or {}
            video_stream = next(
                (stream for stream in info.get("streams", []) if stream.get("codec_type") == "video"),
                {},
            )
            frame_width = int(video_stream.get("width", 1920) or 1920)
            frame_height = int(video_stream.get("height", 1080) or 1080)
            timestamp_temp_dir = tempfile.mkdtemp(prefix="concat_timestamp_")
            if progress_callback:
                progress_callback(0.0, "実時刻タイムスタンプ映像を作成中...")
            timestamp_overlay_path, timestamp_overlay_position = _create_timestamp_overlay(
                file_schedule,
                frame_width,
                frame_height,
                timestamp_settings,
                timestamp_temp_dir,
                cancel_check=cancel_check,
                process_callback=process_callback,
            )
        except Exception as exc:
            if timestamp_temp_dir:
                shutil.rmtree(timestamp_temp_dir, ignore_errors=True)
            try:
                os.unlink(concat_list_path)
            except OSError:
                pass
            if enhancement_temp_dir:
                shutil.rmtree(enhancement_temp_dir, ignore_errors=True)
            return False, f"実時刻タイムスタンプの作成に失敗しました: {exc}"
    
    try:
        # FFmpegコマンドを構築（DTS/PTSタイムスタンプ問題を修正するオプション追加）
        cmd = [
            get_ffmpeg_path(),
            "-y",  # 上書き確認をスキップ
            "-loglevel", "verbose", # デバッグログを詳細に出力
            "-err_detect", "ignore_err",  # エラーを無視して続行
            "-fflags", "+genpts+igndts+discardcorrupt",  # タイムスタンプを再生成、破損フレームを破棄
        ]

        output_filter_params = []
        audio_params = ["-c:a", "aac", "-b:a", "192k"]
        if safe_mode:
            # Each source gets its own decoder so MPEG-4 VOL/extradata changes
            # are re-read at every RTSP file boundary. Resetting PTS per input
            # also removes large capture-clock gaps before concatenation.
            for video_file in files_to_concat:
                cmd.extend(["-threads", str(SAFE_CONCAT_DECODER_THREADS), "-i", video_file])
            if timestamp_overlay_path:
                cmd.extend(["-threads", "1", "-i", timestamp_overlay_path])
            filter_graph, output_label = _build_independent_concat_filter(
                len(files_to_concat), fps, timestamp_overlay_position
            )
            output_filter_params = [
                "-filter_complex", filter_graph,
                "-map", f"[{output_label}]", "-an",
            ]
            audio_params = []
        else:
            cmd.extend(["-f", "concat", "-safe", "0", "-i", concat_list_path])
            if timestamp_overlay_path:
                cmd.extend(["-i", timestamp_overlay_path])
                fps_filter = f"fps={fps}," if fps and fps > 0 else ""
                filter_graph = (
                    f"[0:v]{fps_filter}setpts=PTS-STARTPTS[base];"
                    f"[1:v]setpts=PTS-STARTPTS[clock];"
                    f"[base][clock]overlay={timestamp_overlay_position}:shortest=1[v]"
                )
                output_filter_params = [
                    "-filter_complex", filter_graph,
                    "-map", "[v]", "-map", "0:a?",
                ]
            else:
                output_filter_params = video_filters

        creation_time_metadata = ""
        if file_schedule:
            local_tz = datetime.now().astimezone().tzinfo
            creation_time_metadata = file_schedule[0]["source_start_time"].replace(
                tzinfo=local_tz
            ).isoformat()

        cmd.extend([
            "-fps_mode", "passthrough",  # フレームレートモードをパススルーに変更（DTSエラー回避）
            # "-async", "1",  # 廃止されたため削除
            "-c:v", video_codec,
            *encoder_thread_params,
            "-b:v", bitrate,
            *codec_params,
            *output_filter_params,
            *audio_params,
            "-metadata", f"creation_time={creation_time_metadata}",
            "-metadata", "comment=Absolute capture timeline is stored in the companion .timeline.json file",
            "-movflags", "+faststart",
            "-progress", "pipe:1",
            output_path
        ])
        
        # デバッグ用: コマンドをログ出力
        if progress_callback:
            cmd_str = ' '.join(cmd)
            progress_callback(0.0, f"[DEBUG] FFmpegコマンド: {cmd_str[:200]}...")
        
        if progress_callback:
            progress_callback(0.0, f"連結処理を開始中... (エンコーダ: {encoder_info})")

        if cancel_check and cancel_check():
            return False, "処理がキャンセルされました"
        
        # FFmpegを実行（stderrを別途キャプチャ）
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            encoding='utf-8',
            errors='replace'
        )
        if process_callback:
            process_callback(process)
        
        # 進捗を監視
        start_time = time.time()
        last_reported_percent = -1  # 最後に報告したパーセント（1%単位）
        
        import threading
        
        # stderrを別スレッドで読み取る
        def read_stderr():
            nonlocal error_line_count
            for line in process.stderr:
                error_line_count += 1
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
                    process.wait(timeout=2)
                try:
                    os.remove(output_path)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass
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
                f.write(
                    f"=== Captured error lines ({len(error_lines)} of {error_line_count}) ===\n"
                )
                for line in error_lines:
                    f.write(line + '\n')
            if progress_callback:
                progress_callback(1.0, f"[DEBUG] stderrログを保存: {debug_stderr_path}")
        except Exception as e:
            if progress_callback:
                progress_callback(1.0, f"[DEBUG] stderrログ保存に失敗: {e}")
        
        if return_code == 0:
            timeline_path = None
            try:
                timeline_path = _write_concat_timeline(
                    output_path, file_schedule, timestamp_settings
                )
                if progress_callback:
                    progress_callback(1.0, f"解析用タイムラインを保存: {timeline_path}")
            except Exception as exc:
                if progress_callback:
                    progress_callback(1.0, f"警告: 解析用タイムラインを保存できませんでした: {exc}")

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
                    recent_errors = list(error_lines)[-10:]
                    warning_msg += f"\nFFmpegエラー(抜粋): {'; '.join(recent_errors)}"
                    
                if progress_callback:
                    progress_callback(1.0, f"連結完了（警告あり） (処理時間: {elapsed_str})")
                
                auto_line = f"\n{automatic_bitrate_summary}" if automatic_bitrate_summary else ""
                timeline_line = f"\n解析用タイムライン: {timeline_path}" if timeline_path else ""
                return True, f"{warning_msg}\n出力パス: {output_path}{auto_line}{timeline_line}\n詳細ログ: {debug_stderr_path}"
            
            if progress_callback:
                progress_callback(1.0, f"連結完了 (処理時間: {elapsed_str})")
            auto_line = f"\n{automatic_bitrate_summary}" if automatic_bitrate_summary else ""
            timeline_line = f"\n解析用タイムライン: {timeline_path}" if timeline_path else ""
            return True, f"連結が完了しました: {output_path}{auto_line}{timeline_line}"
        else:
            # エラー詳細を取得
            error_detail = ""
            if error_lines:
                recent_errors = list(error_lines)[-10:]
                error_detail = "\n".join(recent_errors)
            return False, f"FFmpegがエラーで終了しました（コード: {return_code}）\n{error_detail}"
            
    except Exception as e:
        return False, f"連結処理中にエラーが発生しました: {e}"
    finally:
        if process_callback:
            try:
                process_callback(None)
            except Exception:
                pass
        # 一時ファイルを削除（デバッグ用にconcat_listは残す）
        try:
            os.unlink(concat_list_path)
        except:
            pass
        
        if enhancement_temp_dir:
            try:
                shutil.rmtree(enhancement_temp_dir)
            except OSError:
                pass
        if timestamp_temp_dir:
            shutil.rmtree(timestamp_temp_dir, ignore_errors=True)


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

