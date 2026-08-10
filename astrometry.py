# astrometry.py

import os
import json
import time
import io
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import requests
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import threading
import subprocess
import platform
import textwrap

from astropy.io import fits
from astropy.wcs import WCS, Sip
from astropy.coordinates import SkyCoord, Angle
from astropy import units as u
from astropy.units import hourangle, deg

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# config モジュールが存在することを前提とします
try:
    import config
except ImportError:
    print("警告: config.py が見つかりません。デフォルト値を使用します。")
    class DummyConfig:
        PLATE_SOLVE_IMAGE_WIDTH = 1920; PLATE_SOLVE_IMAGE_HEIGHT = 1080
        ASTROMETRY_API_KEY = "YOUR_API_KEY"; ASTROMETRY_RATE_LIMIT_WAIT = 20
        ASTROMETRY_TIMEOUT = 120; ASTROMETRY_INTERVAL = 10
        SCALE_UNITS = 'degwidth'; SCALE_LOWER = 95; SCALE_UPPER = 115
    config = DummyConfig()
    if config.ASTROMETRY_API_KEY == "YOUR_API_KEY": print("警告: APIキー未設定")


astrometry_session: Optional[str] = None
last_upload_time: Optional[float] = None

def extract_datetime_from_video_path(video_path: str) -> Optional[datetime]:
    path_obj = Path(video_path); parts = path_obj.parts
    try:
        if len(parts) >= 4:
            date_part, hour_part, minute_part_base = parts[-3], parts[-2], path_obj.stem
            if '_' in minute_part_base and minute_part_base.split('_')[-1].isdigit(): minute_part = minute_part_base.split('_')[-1][:2]
            elif minute_part_base.isdigit(): minute_part = minute_part_base[:2]
            else: raise ValueError(f"分解析不可: {path_obj.name}")
            return datetime.strptime(f"{date_part}_{hour_part}_{minute_part}", "%Y%m%d_%H_%M")
        else: print(f"パス階層不足: {video_path}"); return None
    except (IndexError, ValueError, TypeError) as e: print(f"動画日時解析失敗 ({video_path}): {e}"); return None

def extract_datetime_from_file_path(file_path: str) -> Optional[datetime]:
    path = Path(file_path)
    try:
        if len(path.parts) >= 4 and path.parent.parent.name.isdigit() and path.parent.name.isdigit():
            date_part, hour_part, minute_str = path.parent.parent.name, path.parent.name, path.stem.split('.')[0]
            if minute_str.isdigit() and len(minute_str) >= 2: return datetime.strptime(f"{date_part}_{hour_part}_{minute_str[:2]}", "%Y%m%d_%H_%M")
        filename_parts = path.stem.split('_')
        if len(filename_parts) >= 2 and len(filename_parts[0]) == 8 and filename_parts[0].isdigit() and len(filename_parts[1]) == 6 and filename_parts[1].isdigit():
            return datetime.strptime(f"{filename_parts[0]}_{filename_parts[1]}", "%Y%m%d_%H%M%S")
        print(f"ファイル日時解析パターン不一致: {file_path}"); return None
    except (IndexError, ValueError, TypeError) as e: print(f"ファイル日時解析失敗 ({file_path}): {e}"); return None


# --- Local Plate Solve (WSL solve-field) ---
_wsl_solver_available: Optional[bool] = None  # キャッシュ用

def check_wsl_solver_available(index_dir: Optional[str] = None) -> bool:
    """WSLとsolve-fieldが利用可能かチェック (結果をキャッシュ)"""
    global _wsl_solver_available
    
    if _wsl_solver_available is not None:
        return _wsl_solver_available
    
    if platform.system() != "Windows":
        print("ローカルソルバー: Windows以外の環境ではWSL経由は利用不可")
        _wsl_solver_available = False
        return False
    
    try:
        # WSLが利用可能かチェック
        wsl_check = subprocess.run(["wsl", "--version"], capture_output=True, text=True, errors='ignore', timeout=10)
        if wsl_check.returncode != 0:
            print("ローカルソルバー: WSLが利用できません")
            _wsl_solver_available = False
            return False
        
        # solve-fieldが存在するかチェック
        solve_check = subprocess.run(["wsl", "which", "solve-field"], capture_output=True, text=True, errors='ignore', timeout=10)
        if solve_check.returncode != 0 or not solve_check.stdout.strip():
            print("ローカルソルバー: WSL内にsolve-fieldがインストールされていません")
            _wsl_solver_available = False
            return False
        
        # インデックスファイルの確認
        if index_dir is None:
            index_dir = getattr(config, 'LOCAL_SOLVER_INDEX_DIR', '/usr/share/astrometry/data')
        
        idx_check = subprocess.run(
            ["wsl", "bash", "-c", f"ls {index_dir}/index-*.fits 2>/dev/null | head -n1"],
            capture_output=True, text=True, errors='ignore', timeout=10
        )
        if idx_check.returncode != 0 or not idx_check.stdout.strip():
            print(f"ローカルソルバー: インデックスファイルが見つかりません ({index_dir})")
            print("  インストール方法: sudo apt-get install astrometry-data-2mass-*")
            _wsl_solver_available = False
            return False
        
        print(f"ローカルソルバー: WSL solve-field 利用可能 (インデックス: {index_dir})")
        _wsl_solver_available = True
        return True
        
    except subprocess.TimeoutExpired:
        print("ローカルソルバー: WSLチェックがタイムアウトしました")
        _wsl_solver_available = False
        return False
    except Exception as e:
        print(f"ローカルソルバー: チェック中にエラー: {e}")
        _wsl_solver_available = False
        return False


def plate_solve_image_local(
    image_path: str,
    mask: Optional[np.ndarray] = None,
    plate_solve_video_path: Optional[str] = None,
    cancel_flag: Optional[threading.Event] = None,
    scale_lower: Optional[float] = None,
    scale_upper: Optional[float] = None
) -> Optional[Dict]:
    """Use the platform-local solver without uploading the image."""
    if platform.system() != "Windows":
        try:
            import local_wideangle_astrometry

            return local_wideangle_astrometry.solve_image_local(
                image_path,
                source_path=plate_solve_video_path or image_path,
            )
        except Exception as exc:
            print(f"ローカル広角Plate Solveエラー: {exc}")
            return None

    # Windows keeps the existing WSL solve-field implementation.
    
    if cancel_flag and cancel_flag.is_set():
        return None
    
    print(f"ローカルPlate Solve開始: {os.path.basename(image_path)}")
    
    # 画像読み込みとマスク適用
    image = cv2.imread(image_path)
    if image is None:
        print(f"エラー: 画像読込失敗: {image_path}")
        return None
    
    if mask is not None:
        try:
            if mask.shape[:2] != image.shape[:2]:
                mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
            else:
                mask_resized = mask
            if mask_resized.ndim == 3:
                mask_gray = cv2.cvtColor(mask_resized, cv2.COLOR_BGR2GRAY)
            else:
                mask_gray = mask_resized
            _, mask_binary = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)
            mask_binary = mask_binary.astype(np.uint8)
            image = cv2.bitwise_and(image, image, mask=mask_binary)
        except Exception as e:
            print(f"マスク適用エラー: {e}")

    # リサイズと一時ファイル保存
    try:
        image_resized = cv2.resize(image, (config.PLATE_SOLVE_IMAGE_WIDTH, config.PLATE_SOLVE_IMAGE_HEIGHT))
    except Exception as e:
        print(f"リサイズエラー: {e}")
        return None
    
    temp_image_path = image_path + "_temp_for_local_solve.jpg"
    if not cv2.imwrite(temp_image_path, image_resized):
        print("一時画像保存失敗")
        return None
    
    # 日時取得
    if plate_solve_video_path:
        plate_solve_datetime = extract_datetime_from_video_path(plate_solve_video_path)
    else:
        plate_solve_datetime = extract_datetime_from_file_path(image_path)
    if plate_solve_datetime is None:
        plate_solve_datetime = datetime.now()
        print("日時取得失敗、現在時刻使用")
    
    wsl_temp_dir = None
    try:
        if cancel_flag and cancel_flag.is_set():
            return None
        
        # WSL一時ディレクトリ作成
        wsl_temp_dir = subprocess.check_output(['wsl', 'mktemp', '-d'], text=True, errors='ignore').strip()
        
        # 設定ファイル作成
        index_dir = getattr(config, 'LOCAL_SOLVER_INDEX_DIR', '/usr/share/astrometry/data')
        cfg_text = textwrap.dedent(f"""\
            add_path {index_dir}
            inparallel
            autoindex
        """)
        wsl_cfg_path = f"{wsl_temp_dir}/temp_config.cfg"
        subprocess.run(['wsl', 'bash', '-c', f'echo "{cfg_text}" > {wsl_cfg_path}'], check=True, capture_output=True)
        
        # Windowsパス -> WSLパス変換
        windows_path = os.path.abspath(temp_image_path)
        drive = windows_path[0].lower()
        wsl_src_path = f"/mnt/{drive}/{windows_path[3:].replace(os.sep, '/')}"
        sanitized_filename = os.path.basename(windows_path).replace(" ", "_")
        wsl_dst_path = f"{wsl_temp_dir}/{sanitized_filename}"
        
        # ファイルコピー
        subprocess.run(['wsl', 'cp', wsl_src_path, wsl_dst_path], check=True, capture_output=True)
        
        if cancel_flag and cancel_flag.is_set():
            return None
        
        # solve-field実行
        print("solve-field実行中...")
        cmd = [
            'wsl', '/usr/bin/solve-field',
            '--config', wsl_cfg_path,
            '--no-plots',
            '--overwrite',
            '--dir', wsl_temp_dir,
            wsl_dst_path,
            '--verbose'
        ]
        
        # スケール指定がある場合は追加
        if scale_lower is not None and scale_upper is not None:
            cmd.extend(['--scale-low', str(scale_lower), '--scale-high', str(scale_upper), '--scale-units', 'degwidth'])
        
        proc = subprocess.run(cmd, text=True, capture_output=True, errors='ignore', timeout=300)
        
        if cancel_flag and cancel_flag.is_set():
            return None
        
        # 結果確認
        wcs_result_path_wsl = f'{wsl_temp_dir}/{os.path.splitext(sanitized_filename)[0]}.wcs'
        check_result = subprocess.run(['wsl', 'test', '-f', wcs_result_path_wsl])
        
        if check_result.returncode == 0:
            # WCSファイルをWindowsにコピー
            corrected_wcs_path = image_path + '_corrected.wcs'
            drive_win = os.path.abspath(corrected_wcs_path)[0].lower()
            wsl_dest_wcs = f"/mnt/{drive_win}/{os.path.abspath(corrected_wcs_path)[3:].replace(os.sep, '/')}"
            
            # 一時WCSとして取得
            temp_wcs_path = image_path + '_temp_local.wcs'
            wsl_temp_wcs_dest = f"/mnt/{drive_win}/{os.path.abspath(temp_wcs_path)[3:].replace(os.sep, '/')}"
            subprocess.run(['wsl', 'cp', wcs_result_path_wsl, wsl_temp_wcs_dest], check=True, capture_output=True)
            
            # 修正済みWCS作成
            if create_corrected_wcs(temp_wcs_path, corrected_wcs_path, plate_solve_datetime=plate_solve_datetime):
                os.remove(temp_wcs_path)
                print(f"ローカルPlate Solve成功: {corrected_wcs_path}")
                return {'wcs_file': corrected_wcs_path, 'job_id': 'local', 'plate_solve_datetime': plate_solve_datetime}
            else:
                print("修正WCS作成失敗")
                if os.path.exists(temp_wcs_path):
                    os.remove(temp_wcs_path)
                return None
        else:
            # .solvedファイルの存在確認（solve-fieldが処理完了したかのインジケータ）
            solved_path_wsl = f'{wsl_temp_dir}/{os.path.splitext(sanitized_filename)[0]}.solved'
            solved_check = subprocess.run(['wsl', 'test', '-f', solved_path_wsl])
            
            if solved_check.returncode == 0:
                # solvedファイルはあるがwcsがない = 解けなかった
                print("solve-field: 実行完了しましたが、画像の解析に失敗しました（星が認識できない等）")
            else:
                # solvedファイルもない = 実行自体に問題がある可能性
                print("solve-field: 実行が完了しなかったか、解析に失敗しました")
            
            # デバッグ情報
            print(f"\n--- solve-field デバッグ情報 ---")
            print(f"Exit code: {proc.returncode}")
            if proc.stdout:
                # 重要な情報を抽出
                stdout_lines = proc.stdout.strip().split('\n')
                key_lines = [l for l in stdout_lines if any(k in l.lower() for k in ['error', 'fail', 'match', 'solved', 'hit', 'field'])]
                if key_lines:
                    print(f"重要な出力:\n  " + "\n  ".join(key_lines[-10:]))
                else:
                    print(f"標準出力 (末尾):\n  " + "\n  ".join(stdout_lines[-5:]))
            if proc.stderr:
                stderr_lines = [l for l in proc.stderr.strip().split('\n') if 'pnm' not in l.lower() and 'reading' not in l.lower()]
                if stderr_lines:
                    print(f"エラー出力:\n  " + "\n  ".join(stderr_lines[-5:]))
            print("--------------------------------\n")
            return None
            
    except subprocess.TimeoutExpired:
        print("ローカルPlate Solve: タイムアウト (300秒)")
        return None
    except Exception as e:
        print(f"ローカルPlate Solve中にエラー: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # クリーンアップ
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        if wsl_temp_dir:
            subprocess.run(['wsl', 'rm', '-rf', wsl_temp_dir], capture_output=True)


def plate_solve_image( image_path: str, mask: Optional[np.ndarray] = None, plate_solve_video_path: Optional[str] = None, cancel_flag: Optional[threading.Event] = None, scale_lower: Optional[float] = None, scale_upper: Optional[float] = None, use_local: Optional[bool] = None ) -> Optional[Dict]:
    """
    Plate Solveを実行する。use_localがNoneの場合は自動判定。
    ローカルソルバーが利用可能な場合はローカルを使用し、そうでなければAPIを使用。
    """
    # scale_lower/scale_upper が指定されていない場合はconfigのデフォルト値を使用
    if scale_lower is None:
        scale_lower = config.SCALE_LOWER
    if scale_upper is None:
        scale_upper = config.SCALE_UPPER
    
    if cancel_flag and cancel_flag.is_set():
        return None
    
    # ローカルソルバーの使用判定
    if use_local is None:
        if platform.system() == "Windows":
            use_local = (
                getattr(config, 'LOCAL_SOLVER_ENABLED', False)
                and check_wsl_solver_available()
            )
        else:
            try:
                import local_wideangle_astrometry
                use_local = (
                    getattr(config, 'LOCAL_SOLVER_ENABLED', False)
                    and local_wideangle_astrometry.is_available()
                )
            except Exception:
                use_local = False
    
    if use_local:
        print("ローカルソルバー（外部API不使用）を使用します")
        return plate_solve_image_local(
            image_path, mask, plate_solve_video_path, cancel_flag, scale_lower, scale_upper
        )
    
    print("Astrometry.net API を使用します")
    # --- 以下は既存のAPI処理 ---
    global astrometry_session, last_upload_time
    image = cv2.imread(image_path)
    if image is None: print(f"エラー: 画像読込失敗: {image_path}"); return None
    if mask is not None:
        try:
            if mask.shape[:2] != image.shape[:2]: mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
            else: mask_resized = mask
            if mask_resized.ndim == 3: mask_gray = cv2.cvtColor(mask_resized, cv2.COLOR_BGR2GRAY)
            else: mask_gray = mask_resized
            _, mask_binary = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY); mask_binary = mask_binary.astype(np.uint8)
            image = cv2.bitwise_and(image, image, mask=mask_binary)
        except Exception as e: print(f"マスク適用エラー: {e}")
    try: image_resized = cv2.resize(image, (config.PLATE_SOLVE_IMAGE_WIDTH, config.PLATE_SOLVE_IMAGE_HEIGHT))
    except Exception as e: print(f"リサイズエラー: {e}"); return None
    temp_image_path = image_path + "_temp_for_astrometry.jpg"
    if not cv2.imwrite(temp_image_path, image_resized): print("一時画像保存失敗"); return None
    if plate_solve_video_path: plate_solve_datetime = extract_datetime_from_video_path(plate_solve_video_path); dt_source = "video"
    else: plate_solve_datetime = extract_datetime_from_file_path(image_path); dt_source = "image"
    if plate_solve_datetime is None: plate_solve_datetime = datetime.now(); print(f"日時取得失敗({dt_source})、現在時刻使用")
    try:
        if astrometry_session is None:
            if not config.ASTROMETRY_API_KEY or config.ASTROMETRY_API_KEY == "YOUR_API_KEY": print("APIキー未設定"); return None
            url_login = 'http://nova.astrometry.net/api/login'; login_data = {'request-json': json.dumps({'apikey': config.ASTROMETRY_API_KEY})}
            response = requests.post(url_login, data=login_data, timeout=30); response.raise_for_status(); result = response.json(); astrometry_session = result.get('session')
            if not astrometry_session: print("セッションID取得失敗"); return None
            print("Astrometry.netログイン成功")
        if last_upload_time is not None:
            time_since_last = time.time() - last_upload_time
            if time_since_last < config.ASTROMETRY_RATE_LIMIT_WAIT:
                sleep_time = config.ASTROMETRY_RATE_LIMIT_WAIT - time_since_last; print(f"レートリミット待機 {sleep_time:.1f}秒...")
                wait_start = time.time()
                while time.time() - wait_start < sleep_time:
                    if cancel_flag and cancel_flag.is_set(): print("待機中キャンセル"); return None
                    time.sleep(0.1)
        if cancel_flag and cancel_flag.is_set(): print("キャンセル"); return None
        upload_params = {'session': astrometry_session, 'allow_commercial_use': 'd', 'allow_modifications': 'd', 'publicly_visible': 'n', 'scale_units': config.SCALE_UNITS, 'scale_lower': scale_lower, 'scale_upper': scale_upper, 'scale_type': 'ul'}
        url_upload = 'http://nova.astrometry.net/api/upload'
        with open(temp_image_path, 'rb') as f:
            files = {'file': (os.path.basename(temp_image_path), f, 'image/jpeg')}
            response = requests.post(url_upload, data={'request-json': json.dumps(upload_params)}, files=files, timeout=60)
            response.raise_for_status()
        result = response.json()
        if result.get('status') != 'success': print(f"アップロード失敗: {result.get('errormessage', '不明')}"); return None
        submission_id = result.get('subid')
        if not submission_id: print("サブミッションID取得失敗"); return None
        print(f"アップロード成功 ID: {submission_id}"); last_upload_time = time.time()
        start_wait_time = time.time(); job_id = None
        while time.time() - start_wait_time < config.ASTROMETRY_TIMEOUT:
            if cancel_flag and cancel_flag.is_set(): print("結果待機中キャンセル"); return None
            url_sub_status = f'http://nova.astrometry.net/api/submissions/{submission_id}'; response = requests.get(url_sub_status, timeout=10); response.raise_for_status(); sub_result = response.json()
            jobs = sub_result.get('jobs', [])
            if jobs and jobs[0] is not None:
                job_id = jobs[0]; url_job_status = f'http://nova.astrometry.net/api/jobs/{job_id}/info/'; response = requests.get(url_job_status, timeout=10); response.raise_for_status(); job_result = response.json()
                job_status = job_result.get('status')
                if job_status == 'success': print(f"プレートソルブ成功 Job ID: {job_id}"); break
                elif job_status == 'failure': print("プレートソルブ失敗"); return None
                print(f"状況: {job_status} (経過: {int(time.time() - start_wait_time)}秒)")
            else: print(f"サブミッション確認中... (経過: {int(time.time() - start_wait_time)}秒)")
            time.sleep(config.ASTROMETRY_INTERVAL)
        else: print("タイムアウト"); return None
        if job_id:
            url_wcs = f'http://nova.astrometry.net/wcs_file/{job_id}'; response = requests.get(url_wcs, timeout=60); response.raise_for_status()
            original_wcs_path = image_path + '_original.wcs'
            # Preserve the WCS file exactly as delivered (binary FITS or text header).
            with open(original_wcs_path, 'wb') as f: f.write(response.content)
            corrected_wcs_path = image_path + '_corrected.wcs'
            if create_corrected_wcs(
                original_wcs_path,
                corrected_wcs_path,
                plate_solve_datetime=plate_solve_datetime
            ):
                os.remove(original_wcs_path)
                return {'wcs_file': corrected_wcs_path, 'job_id': job_id, 'plate_solve_datetime': plate_solve_datetime}
            else:
                print("修正WCS作成失敗"); os.remove(original_wcs_path); return None
        else:
            print("Job ID 不明"); return None
    except requests.exceptions.RequestException as e: print(f"API通信エラー: {e}"); return None
    except json.JSONDecodeError as e: print(f"APIレスポンス解析エラー: {e}"); return None
    except Exception as e: print(f"プレートソルブ中エラー: {e}"); return None
    finally:
        if os.path.exists(temp_image_path): os.remove(temp_image_path)

def create_corrected_wcs(
    original_wcs_filename: str,
    new_wcs_filename: str,
    naxis1: int = config.PLATE_SOLVE_IMAGE_WIDTH,
    naxis2: int = config.PLATE_SOLVE_IMAGE_HEIGHT,
    plate_solve_datetime: Optional[datetime] = None
) -> bool:
    try:
        source_header = None
        wcs = None

        # Prefer FITS parsing so SIP/distortion keywords are preserved.
        try:
            with fits.open(original_wcs_filename) as hdul:
                source_header = hdul[0].header.copy()
            wcs = WCS(source_header, relax=True, fix=False)
        except Exception:
            # Fallback: some sources may provide a plain-text WCS header.
            with open(original_wcs_filename, 'rb') as f:
                raw = f.read()
            header_str = raw.decode('ascii', errors='ignore')
            wcs = WCS(header_str, relax=True, fix=False)

        # relax=True keeps non-standard but widely-used distortion terms (e.g. SIP).
        new_header = wcs.to_header(relax=True)

        # If the original header had explicit SIP keys, keep them even if Astropy omits any
        # during serialization (defensive against version differences).
        if source_header is not None:
            sip_prefixes = ("A_", "B_", "AP_", "BP_")
            for key, value in source_header.items():
                if key in ("A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER") or key.startswith(sip_prefixes):
                    if key not in new_header:
                        new_header[key] = value

        new_header['SIMPLE'] = True
        new_header['BITPIX'] = 8
        new_header['NAXIS'] = 2
        new_header['NAXIS1'] = naxis1
        new_header['NAXIS2'] = naxis2
        
        if plate_solve_datetime:
            new_header['DATE-OBS'] = plate_solve_datetime.isoformat()
            new_header.set('COMMENT', 'Timestamp for plate-solve reference.')
            print(f"基準時刻 {plate_solve_datetime.isoformat()} をWCSヘッダーに保存します。")

        empty_data = np.zeros((naxis2, naxis1), dtype=np.uint8)
        hdu = fits.PrimaryHDU(data=empty_data, header=new_header)
        hdul = fits.HDUList([hdu])
        hdul.writeto(new_wcs_filename, overwrite=True)
        hdul.close()
        print(f"修正済みWCSファイル作成 (Astropy使用): {new_wcs_filename}")
        return True
    except FileNotFoundError:
        print(f"エラー: 元WCSファイルなし: {original_wcs_filename}")
        return False
    except Exception as e:
        print(f"修正済みWCS作成エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_adjusted_skycoord_for_rotation(
    original_skycoord: SkyCoord,
    plate_solve_datetime: Optional[datetime],
    detection_datetime: Optional[datetime]
) -> SkyCoord:
    """
    基準時刻と検出時刻の差に基づき、地球の自転を考慮して天空座標を補正する。
    """
    if not (plate_solve_datetime and detection_datetime):
        print("自転補正スキップ: 時刻情報が不十分です。")
        return original_skycoord

    try:
        delta_t = (detection_datetime - plate_solve_datetime).total_seconds() * u.second
        if abs(delta_t.value) < 1.0:
            return original_skycoord # 1秒未満の差は無視

        # 恒星時日 (23.9344696 hours) に基づく回転率
        EARTH_ROTATION_RATE = Angle(360 * u.deg) / (23.9344696 * u.hour).to(u.second)
        delta_ra = delta_t * EARTH_ROTATION_RATE

        original_ra = original_skycoord.ra
        new_ra = (original_ra + delta_ra).wrap_at(360 * u.deg)

        adjusted_skycoord = SkyCoord(ra=new_ra, dec=original_skycoord.dec, frame=original_skycoord.frame)

        # デバッグ情報出力
        print("\n--- 地球自転による座標補正を実行 ---")
        print(f"  プレートソルブ基準時刻: {plate_solve_datetime.isoformat()}")
        print(f"  流星検出時刻:         {detection_datetime.isoformat()}")
        print(f"  時間差 (Δt):         {delta_t.to(u.minute):.2f}")
        print(f"  赤経補正量 (ΔRA):      {delta_ra.to(u.deg):.4f}")
        print(f"  元の座標 (RA, Dec):    ({original_skycoord.ra.deg:.4f}, {original_skycoord.dec.deg:.4f})")
        print(f"  補正後の座標 (RA, Dec):  ({adjusted_skycoord.ra.deg:.4f}, {adjusted_skycoord.dec.deg:.4f})")
        print("----------------------------------\n")

        return adjusted_skycoord

    except Exception as e:
        print(f"座標の自転補正中にエラーが発生しました。元の座標を使用します: {e}")
        import traceback; traceback.print_exc()
        return original_skycoord

def _annotate_stars_and_grid(ax: plt.Axes):
    try:
        ra = ax.coords[0]; dec = ax.coords[1]
        ra.set_ticks(spacing=30 * u.deg, color='white', alpha=0.5); dec.set_ticks(spacing=30 * u.deg, color='white', alpha=0.5)
        ax.coords.grid(True, color='white', ls='solid', alpha=0.5, linewidth=0.5)
        ra.set_axislabel(''); dec.set_axislabel('')
        ra.set_ticklabel_visible(False); dec.set_ticklabel_visible(False)
        ax.coords.frame.set_color('none')
    except Exception as e: print(f"グリッド描画中にエラー: {e}")

def _create_flipped_wcs(original_wcs: WCS, image_shape: Tuple[int, int]) -> WCS:
    flipped_wcs = original_wcs.deepcopy()
    height = image_shape[0]
    flipped_wcs.wcs.crpix[1] = height - original_wcs.wcs.crpix[1] + 1
    if flipped_wcs.wcs.has_cd():
        flipped_wcs.wcs.cd[0, 1] *= -1
        flipped_wcs.wcs.cd[1, 1] *= -1
    elif flipped_wcs.wcs.has_pc():
        flipped_wcs.wcs.pc[0, 1] *= -1
        flipped_wcs.wcs.pc[1, 1] *= -1
    else:
        raise ValueError("WCSにCDまたはPCマトリックスがありません。")
    if original_wcs.sip is not None:
        def flip_coefficients(coefficients, invert_y_output=False):
            if coefficients is None:
                return None
            transformed = np.array(coefficients, dtype=float, copy=True)
            for y_power in range(transformed.shape[1]):
                sign = -1.0 if y_power % 2 else 1.0
                if invert_y_output:
                    sign *= -1.0
                transformed[:, y_power] *= sign
            return transformed

        sip = original_wcs.sip
        flipped_wcs.sip = Sip(
            flip_coefficients(sip.a),
            flip_coefficients(sip.b, invert_y_output=True),
            flip_coefficients(sip.ap),
            flip_coefficients(sip.bp, invert_y_output=True),
            flipped_wcs.wcs.crpix,
        )
    return flipped_wcs


def _draw_meteor_marker(
    image: np.ndarray,
    detected_line: Tuple[Tuple[int, int], Tuple[int, int]],
) -> np.ndarray:
    """流星の検出範囲を黄色い枠だけで強調した画像を返す。

    WCS の有無にかかわらず、概要動画で一目で検出箇所を確認できるようにする。
    ``image`` は RGB/RGBA のどちらでも扱い、元の型・スケールを維持する。
    """
    if image is None or image.size == 0:
        return image

    was_float = np.issubdtype(image.dtype, np.floating)
    normalized = was_float and float(np.nanmax(image)) <= 1.0
    canvas = np.clip(image * 255.0 if normalized else image, 0, 255).astype(np.uint8).copy()
    height, width = canvas.shape[:2]
    (x1, y1), (x2, y2) = detected_line
    x1, x2 = int(np.clip(x1, 0, width - 1)), int(np.clip(x2, 0, width - 1))
    y1, y2 = int(np.clip(y1, 0, height - 1)), int(np.clip(y2, 0, height - 1))

    line_length = max(1.0, float(np.hypot(x2 - x1, y2 - y1)))
    padding = int(max(36, min(120, line_length * 0.55)))
    left, right = max(0, min(x1, x2) - padding), min(width - 1, max(x1, x2) + padding)
    top, bottom = max(0, min(y1, y2) - padding), min(height - 1, max(y1, y2) + padding)
    thickness = max(2, int(round(min(width, height) / 420)))
    marker_color = (255, 220, 0)  # RGB: 視認性の高い黄色

    cv2.rectangle(canvas, (left, top), (right, bottom), marker_color, thickness, cv2.LINE_AA)
    return canvas.astype(np.float32) / 255.0 if normalized else canvas


def _annotate_local_wideangle_image(
    image_path: str,
    wcs_info: Dict,
    line_centers: Optional[List[Tuple[float, float]]],
    detection_datetime: Optional[datetime],
    timestamp: Optional[str],
    flip_vertically: bool,
    detected_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]],
) -> Optional[str]:
    """Render a local SIP calibration without extrapolating past verified stars."""
    import local_wideangle_astrometry

    frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if frame is None:
        raise IOError(f"画像を読み込めません: {image_path}")
    calibration_path = wcs_info.get("calibration_path") or wcs_info["wcs_file"]
    metadata, local_wcs = local_wideangle_astrometry._load_calibration(calibration_path)
    reference_value = metadata.get("reference_datetime") or wcs_info.get("plate_solve_datetime")
    if isinstance(reference_value, datetime):
        reference_datetime = reference_value
    else:
        reference_datetime = datetime.fromisoformat(str(reference_value).replace("Z", "+00:00"))
    target_datetime = detection_datetime or reference_datetime
    if reference_datetime.tzinfo is not None and target_datetime.tzinfo is None:
        target_datetime = target_datetime.replace(tzinfo=reference_datetime.tzinfo)
    elif reference_datetime.tzinfo is None and target_datetime.tzinfo is not None:
        target_datetime = target_datetime.replace(tzinfo=None)
    output = local_wideangle_astrometry.annotate_frame(
        frame,
        target_datetime,
        calibration_path=calibration_path,
        draw_grid=True,
        draw_constellations=bool(wcs_info.get("draw_constellations", False)),
    )
    support_mask = local_wideangle_astrometry._forward_grid_model(
        local_wcs, metadata, output.shape[1], output.shape[0]
    )["support_mask"]
    if detected_line:
        rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        output = cv2.cvtColor(_draw_meteor_marker(rgb, detected_line), cv2.COLOR_RGB2BGR)
    if line_centers:
        for x, y in line_centers:
            pixel_x, pixel_y = int(round(x)), int(round(y))
            supported = (
                0 <= pixel_x < output.shape[1]
                and 0 <= pixel_y < output.shape[0]
                and support_mask[pixel_y, pixel_x] > 0
            )
            if supported:
                sky = local_wcs.pixel_to_world(float(x), float(y))
                sky = get_adjusted_skycoord_for_rotation(
                    sky, reference_datetime, target_datetime
                )
                labels = (
                    f"RA: {sky.ra.to_string(unit=u.hourangle, sep=':', precision=2)}",
                    f"Dec: {sky.dec.to_string(unit=u.deg, sep=':', precision=2, alwayssign=True)}",
                )
                label_color = (80, 255, 80)
            else:
                # Do not present an extrapolated edge coordinate as measured.
                labels = ("RA/Dec unavailable", "outside calibrated area")
                label_color = (80, 190, 255)
            anchor_x = int(np.clip(round(x), 5, output.shape[1] - 5))
            anchor_y = int(np.clip(round(y) - 30, 36, output.shape[0] - 12))
            for index, label in enumerate(labels):
                text_y = anchor_y + index * 22
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
                )
                left = int(np.clip(anchor_x - text_width // 2 - 5, 0, output.shape[1] - 1))
                right = int(np.clip(left + text_width + 10, 0, output.shape[1] - 1))
                cv2.rectangle(output, (left, text_y - text_height - 5), (right, text_y + 4), (0, 0, 0), -1)
                cv2.putText(output, label, (left + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, label_color, 1, cv2.LINE_AA)
    if timestamp:
        (text_width, text_height), _ = cv2.getTextSize(
            timestamp, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        x = max(8, (output.shape[1] - text_width) // 2)
        y = output.shape[0] - 24
        cv2.rectangle(output, (x - 8, y - text_height - 7), (x + text_width + 8, y + 7), (0, 0, 0), -1)
        cv2.putText(output, timestamp, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
    if flip_vertically:
        output = cv2.flip(output, 0)
    base, _ = os.path.splitext(image_path)
    if base.endswith('_composite'):
        base = base[:-len('_composite')]
    annotated_image_path = f"{base}_annotated.png"
    if not cv2.imwrite(annotated_image_path, output, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise IOError(f"注釈付き画像を保存できません: {annotated_image_path}")
    return annotated_image_path

def annotate_image_with_wcs(
    image_path: str,
    wcs_info: Dict,
    line_centers: Optional[List[Tuple[float, float]]] = None,
    detection_datetime: Optional[datetime] = None,
    timestamp: Optional[str] = None,
    cancel_flag: Optional[threading.Event] = None,
    flip_vertically: bool = False,
    detected_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
) -> Optional[str]:
    """
    WCS情報を使用して画像に注釈を描画し、保存する。
    地球の自転を考慮して、流星の「座標」を補正する。

    出力画像は、検出・保存時の画面向きを既定で維持する。過去のWCS表示との
    互換が必要な場合だけ ``flip_vertically=True`` を明示的に指定する。
    """
    # WCS情報がない場合はPILでタイムスタンプのみ描画
    if not wcs_info or 'wcs_file' not in wcs_info or not os.path.exists(wcs_info['wcs_file']):
        try:
            with Image.open(image_path) as image:
                if detected_line:
                    image = Image.fromarray(_draw_meteor_marker(np.asarray(image.convert("RGB")), detected_line))
                if flip_vertically: image = image.transpose(Image.FLIP_TOP_BOTTOM)
                draw = ImageDraw.Draw(image)
                if timestamp:
                    try: font = ImageFont.truetype("arial.ttf", size=24)
                    except IOError: font = ImageFont.load_default()
                    if hasattr(draw, 'textbbox'): bbox = draw.textbbox((0, 0), timestamp, font=font); text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    else: text_width, text_height = draw.textsize(timestamp, font=font)
                    img_width, img_height = image.size; x = (img_width - text_width) / 2; y = img_height - text_height - 15
                    draw.rectangle((x - 5, y - 5, x + text_width + 5, y + text_height + 5), fill=(0, 0, 0, 128))
                    draw.text((x, y), timestamp, fill="white", font=font)
                base, ext = os.path.splitext(image_path)
                if base.endswith('_composite'): base = base[:-len('_composite')]
                annotated_image_path = f"{base}_annotated{ext}"
                image.save(annotated_image_path)
                return annotated_image_path
        except Exception as e: print(f"PILを使用したタイムスタンプ描画中にエラー: {e}"); return None

    fig = None
    try:
        if cancel_flag and cancel_flag.is_set(): return None

        wcs_file_path = wcs_info['wcs_file']
        with fits.open(wcs_file_path) as hdul:
            calibration_type = hdul[0].header.get("CALTYPE")
        if (
            calibration_type == "LOCAL-SIP"
            or str(wcs_info.get("job_id", "")).startswith("local-wideangle")
        ):
            return _annotate_local_wideangle_image(
                image_path, wcs_info, line_centers, detection_datetime, timestamp,
                flip_vertically, detected_line,
            )
        image_data = plt.imread(image_path)
        if detected_line:
            image_data = _draw_meteor_marker(image_data, detected_line)
        image_shape = image_data.shape[:2]
        
        with fits.open(wcs_file_path) as hdul:
            header = hdul[0].header
            original_wcs = WCS(header, relax=True, fix=False)
            if not (original_wcs.is_celestial and original_wcs.pixel_n_dim == 2): raise ValueError("WCSが無効です。")
            
            plate_solve_datetime = None
            if 'DATE-OBS' in header:
                try: plate_solve_datetime = datetime.fromisoformat(header['DATE-OBS'])
                except (ValueError, TypeError): print(f"警告: FITSヘッダーのDATE-OBS '{header['DATE-OBS']}' の形式が不正です。")
            
            if plate_solve_datetime is None:
                plate_solve_datetime = wcs_info.get('plate_solve_datetime')
                if plate_solve_datetime: print("警告: WCSヘッダーに時刻情報がありません。wcs_infoの時刻を使用します。")

        # WCSAxes always uses a FITS-style lower-left origin, while camera and
        # OpenCV images use a top-left origin.  Convert both the raster and WCS
        # so the saved summary retains the camera's normal orientation.  The
        # explicit compatibility option intentionally requests the old,
        # vertically inverted presentation.
        if flip_vertically:
            image_to_show = image_data
            wcs_for_plotting = original_wcs
        else:
            image_to_show = np.flipud(image_data)
            wcs_for_plotting = _create_flipped_wcs(original_wcs, image_shape)

        output_dpi = 300
        fig_width_inch = image_shape[1] / output_dpi
        fig_height_inch = image_shape[0] / output_dpi
        fig = plt.figure(figsize=(fig_width_inch, fig_height_inch), dpi=output_dpi)
        ax = fig.add_subplot(1, 1, 1, projection=wcs_for_plotting)

        ax.imshow(image_to_show, origin='lower', aspect='auto', interpolation='none')
        _annotate_stars_and_grid(ax)

        if line_centers:
            for x, y in line_centers:
                # 1. 元のWCSを使って、ピクセル座標を「基準時刻」の天空座標に変換
                sky_coord_original = original_wcs.pixel_to_world(x, y)
                
                # 2. その天空座標を、自転を考慮して「検出時刻」の座標に補正
                sky_coord_adjusted = get_adjusted_skycoord_for_rotation(
                    sky_coord_original,
                    plate_solve_datetime,
                    detection_datetime
                )
                
                # 3. 補正後の座標を文字列にフォーマットして表示
                ra_str = sky_coord_adjusted.ra.to_string(unit=u.hourangle, sep=':', precision=2)
                dec_str = sky_coord_adjusted.dec.to_string(unit=u.deg, sep=':', precision=2, alwayssign=True)
                label = f"RA: {ra_str}\nDec: {dec_str}"
                
                y_for_text = y if flip_vertically else (image_shape[0] - 1 - y)
                ax.text(x, y_for_text, label, color='lime', fontsize=6, ha='center', va='bottom',
                        bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2', edgecolor='none'))

        if timestamp:
            ax.text(0.5, 0.05, timestamp, transform=ax.transAxes, fontsize=8, color='white', ha='center', va='center',
                    bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3', edgecolor='none'))
        
        base, _ = os.path.splitext(image_path)
        if base.endswith('_composite'): base = base[:-len('_composite')]
        annotated_image_path = f"{base}_annotated.png"
        plt.savefig(annotated_image_path, dpi=output_dpi, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
        print(f"注釈付き画像を保存しました: {annotated_image_path}")
        return annotated_image_path

    except Exception as e:
        print(f"注釈処理中にエラーが発生しました: {e}"); import traceback; traceback.print_exc(); return None
    finally:
        if fig: plt.close(fig)

if __name__ == '__main__':
    print("astrometry.py が直接実行されました。")
    test_path1 = "/path/to/20250408/17/14.mp4"; test_path2 = "C:\\data\\20250408\\17\\clip_abc_14.mp4"
    test_path3 = "C:\\data\\20250408\\17\\20250408_171430_meteor_xyz.jpg"; test_path4 = "C:\\data\\20250408\\17\\14.mp4_composite.jpg_corrected.wcs"
    print(f"日時抽出 (動画パス1): {extract_datetime_from_video_path(test_path1)}")
    print(f"日時抽出 (動画パス2): {extract_datetime_from_video_path(test_path2)}")
    print(f"日時抽出 (画像パス): {extract_datetime_from_file_path(test_path3)}")
    print(f"日時抽出 (WCSパス): {extract_datetime_from_file_path(test_path4)}")
