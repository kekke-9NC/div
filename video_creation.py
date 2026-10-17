import cv2
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import subprocess
import sys
import os

def create_summary_video(
    summary_video_config: List[Dict[str, Any]],
    composite_image_path: str,
    annotated_image_path: str,
    cutout_video_path: str,
    output_video_path: str,
    cutout_rect: Tuple[int, int, int, int],
    frame_rate: float,
    output_resolution: Tuple[int, int] = (1920, 1080),
    full_video_path: Optional[str] = None,
    full_video_frames: Optional[List[np.ndarray]] = None
) -> bool:
    """
    ユーザー設定に基づき、検出結果をまとめた概要動画を生成する。
    FFMPEGを利用して高速に動画を書き出す。

    Args:
        summary_video_config (List[Dict[str, Any]]): 動画の構成要素と順序を定義した設定リスト。
        composite_image_path (str): 比較明合成画像のパス。
        annotated_image_path (str): 注釈付き画像のパス。
        cutout_video_path (str): 切り出し動画クリップのパス。
        output_video_path (str): 生成する動画の保存パス。
        cutout_rect (Tuple[int, int, int, int]): 元画像における切り出し領域 (x_start, y_start, x_end, y_end)。
        frame_rate (float): 動画のフレームレート。
        output_resolution (Tuple[int, int]): 出力動画の解像度 (幅, 高さ)。
        full_video_path (Optional[str]): フルサイズ動画のパス（オプション）。
    """
    print(f"概要動画の生成を開始: {output_video_path}")
    
    W, H = output_resolution

    # --- FFMPEG を使用した高速な動画書き出し設定 ---
    command = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{W}x{H}', '-pix_fmt', 'bgr24', '-r', str(frame_rate),
        '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'ultrafast',
        '-tune', 'zerolatency', '-pix_fmt', 'yuv420p', output_video_path
    ]

    try:
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print("\n" + "="*50 + "\nエラー: `ffmpeg` が見つかりません。\n" + "="*50)
        return False
    except Exception as e:
        print(f"ffmpegの起動に失敗しました: {e}")
        return False

    # --- ヘルパー関数 ---
    def _add_image_sequence(img_path, duration_sec, proc_handle):
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 画像を読み込めませんでした: {img_path}")
            return False
        
        h, w = img.shape[:2]
        scale = min(W / w, H / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_img = cv2.resize(img, (new_w, new_h))

        top = (H - new_h) // 2; bottom = H - new_h - top
        left = (W - new_w) // 2; right = W - new_w - left
        
        final_frame = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        num_frames = int(duration_sec * frame_rate)
        for _ in range(num_frames):
            if proc_handle.stdin:
                proc_handle.stdin.write(final_frame.tobytes())
        return True

    def _add_zoom_sequence(img_path, duration_sec, proc_handle):
        annotated_img = cv2.imread(img_path)
        if annotated_img is None:
             print("警告: ズーム用の画像が読み込めません。ズームをスキップします。")
             return False
        
        img_h, img_w = annotated_img.shape[:2]
        num_zoom_frames = int(duration_sec * frame_rate)
        start_rect = np.array([0, 0, img_w, img_h])

        x_s, y_s, x_e, y_e = cutout_rect
        cutout_cx, cutout_cy = (x_s + x_e) / 2, (y_s + y_e) / 2
        zoom_target_w, zoom_target_h = 455, 256
        end_rect = np.array([cutout_cx - zoom_target_w / 2, cutout_cy - zoom_target_h / 2, zoom_target_w, zoom_target_h])

        for i in range(num_zoom_frames):
            progress = i / (num_zoom_frames - 1) if num_zoom_frames > 1 else 1.0
            progress = 0.5 * (1 - np.cos(np.pi * progress))
            
            current_rect = start_rect * (1 - progress) + end_rect * progress
            cx, cy, cw, ch = current_rect.astype(int)
            cx1, cy1 = max(0, cx), max(0, cy)
            cx2, cy2 = min(img_w, cx + cw), min(img_h, cy + ch)

            if cx2 <= cx1 or cy2 <= cy1:
                frame = np.zeros((H, W, 3), dtype=np.uint8)
            else:
                cropped = annotated_img[cy1:cy2, cx1:cx2]
                frame = cv2.resize(cropped, (W, H), interpolation=cv2.INTER_LANCZOS4)
            
            if proc_handle.stdin:
                proc_handle.stdin.write(frame.tobytes())
        return True
    
    def _add_video_frame(frame, proc_handle):
        h, w = frame.shape[:2]
        scale = min(W / w, H / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_frame = cv2.resize(frame, (new_w, new_h))

        top = (H - new_h) // 2; bottom = H - new_h - top
        left = (W - new_w) // 2; right = W - new_w - left

        final_frame = cv2.copyMakeBorder(resized_frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        if proc_handle.stdin:
            proc_handle.stdin.write(final_frame.tobytes())

    def _add_video_clip(video_path, proc_handle):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"警告: カットアウト動画を読み込めませんでした: {video_path}")
            cap.release()
            return False

        decoded_frames = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                _add_video_frame(frame, proc_handle)
                decoded_frames += 1
        finally:
            cap.release()

        if decoded_frames == 0:
            print(f"警告: カットアウト動画からフレームを読み込めませんでした: {video_path}")
            return False
        return True

    def _add_video_frame_sequence(frames, proc_handle):
        frame_count = 0
        for frame in frames:
            _add_video_frame(frame, proc_handle)
            frame_count += 1
        return frame_count > 0

    generation_failed = False
    try:
        # --- 設定に基づいて動画シーケンスを動的に生成 ---
        step_counter = 1
        total_steps = sum(1 for item in summary_video_config if item.get('enabled'))

        for component in summary_video_config:
            if not component.get('enabled'):
                continue

            name = component.get('name')
            duration = component.get('duration')
            print(f"  - ステップ{step_counter}/{total_steps}: '{name}' を追加中...")

            if name == "Composite Image":
                if not _add_image_sequence(composite_image_path, duration, proc):
                    generation_failed = True
            elif name == "Annotated Image":
                if not _add_image_sequence(annotated_image_path, duration, proc):
                    generation_failed = True
            elif name == "Zoom Sequence":
                if not _add_zoom_sequence(annotated_image_path, duration, proc):
                    generation_failed = True
            elif name == "Cutout Video":
                if not _add_video_clip(cutout_video_path, proc):
                    generation_failed = True
            elif name == "Full Size Video":
                if full_video_path:
                    if not _add_video_clip(full_video_path, proc):
                        generation_failed = True
                elif full_video_frames is not None:
                    if not _add_video_frame_sequence(full_video_frames, proc):
                        generation_failed = True
                else:
                    print("警告: フルサイズ動画の入力がありません。スキップします。")
                    generation_failed = True
            
            step_counter += 1

    except Exception as e:
        generation_failed = True
        print(f"概要動画の生成中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            # communicate closes stdin, drains stderr, and waits for ffmpeg. In
            # particular, it handles BrokenPipeError when ffmpeg exits early.
            stderr_output = proc.communicate()[1]
        except Exception as cleanup_error:
            generation_failed = True
            print(f"FFMPEGの終了処理に失敗しました: {cleanup_error}")
            try:
                proc.kill()
            except OSError:
                pass
            try:
                stderr_output = proc.communicate()[1]
            except Exception as final_cleanup_error:
                stderr_output = b""
                print(f"FFMPEGプロセスを回収できませんでした: {final_cleanup_error}")
                try:
                    proc.wait()
                except Exception as wait_error:
                    print(f"FFMPEGの終了待機に失敗しました: {wait_error}")

        if proc.returncode != 0:
            print("="*50 + "\nFFMPEGがエラーを返しました:\n" + "="*50)
            try:
                print(stderr_output.decode('utf-8', errors='ignore'))
            except Exception as decode_err:
                print(f"エラーメッセージのデコードに失敗: {decode_err}")
            print("="*50)
        else:
            print(f"概要動画の生成が完了しました: {output_video_path}")

    try:
        output_exists = os.path.isfile(output_video_path) and os.path.getsize(output_video_path) > 0
    except OSError:
        output_exists = False
    return not generation_failed and proc.returncode == 0 and output_exists
