#!/usr/bin/env python3
"""Project an explicitly approved fixed-camera model into a transparent OBS layer."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import signal
import sys
import tempfile
import time
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2
import numpy as np
from PIL import Image
from camera_plate_model import FixedCameraPlateModel
from local_wideangle_astrometry import _draw_constellation_lines, SIDEREAL_DAY_SECONDS
from legacy_detection_mask import LegacyDetectionMask, default_settings_path

STOP = False


def atomic_json(path, data):
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', mode='w', encoding='utf-8', delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(data, handle, ensure_ascii=False, allow_nan=False)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def render_layer(payload, when, mask=None, opacity=0.45):
    width, height = int(payload['width']), int(payload['height'])
    reference = datetime.fromisoformat(payload['reference_datetime'])
    if reference.tzinfo is not None and when.tzinfo is None:
        when = when.astimezone()
    delta_ra = (when-reference).total_seconds()/SIDEREAL_DAY_SECONDS*360
    support = cv2.resize(np.asarray(payload['support_grid'],dtype=np.uint8), (width,height), interpolation=cv2.INTER_NEAREST)*255
    if mask is not None:
        support = cv2.bitwise_and(support, cv2.resize(mask,(width,height),interpolation=cv2.INTER_NEAREST))
    lines = np.zeros((height,width,3),dtype=np.uint8)
    _draw_constellation_lines(lines,FixedCameraPlateModel(payload),delta_ra,support,allow_partial_segments=True)
    rgb = cv2.cvtColor(lines,cv2.COLOR_BGR2RGB)
    strength = np.max(rgb,axis=2)
    # Modest opacity keeps faint stars and meteors visible through the sticks.
    opacity = float(opacity)
    if not np.isfinite(opacity) or not 0 <= opacity <= 1:
        raise ValueError('線の不透明度は0〜1で指定してください')
    alpha = np.rint(strength.astype(float)*opacity).astype(np.uint8)
    rgba = np.dstack((rgb,alpha))
    rgba[support==0] = 0
    return rgba


def approved_model(config_path):
    config = json.loads(config_path.read_text())
    if not config.get('enabled') or not config.get('visually_verified_at'):
        raise ValueError('目視確認済みのモデルが有効になっていません')
    path = Path(config['model_path']).expanduser()
    if not path.is_absolute():
        path = config_path.parent/path
    contents = path.read_bytes()
    revision = hashlib.sha256(contents).hexdigest()
    if config.get('approved_model_sha256') != revision:
        raise ValueError('補正モデルが変更されました。再度目視確認してください')
    payload = json.loads(contents)
    if not payload.get('enabled') or (payload.get('width'),payload.get('height')) != (1920,1080):
        raise ValueError('この配信の解像度・品質条件に対応しないモデルです')
    return config,payload,revision


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--parent-pid',type=int)
    parser.add_argument('--legacy-settings',type=Path,default=default_settings_path())
    args=parser.parse_args()
    cv2.setNumThreads(1)
    os.nice(10)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    mask=LegacyDetectionMask(args.legacy_settings)
    def stop(*_):
        global STOP
        STOP=True
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    last_error=None
    while not STOP:
        if args.parent_pid:
            try:
                os.kill(args.parent_pid,0)
            except ProcessLookupError:
                break
        started=time.monotonic()
        try:
            config,payload,revision=approved_model(args.output_dir/'sky_config.json')
            mask.refresh()
            epoch=time.time()
            when=datetime.fromtimestamp(epoch-float(config.get('video_delay_seconds',0)))
            rgba=render_layer(payload,when,mask.mask,config.get('line_opacity',0.45))
            with tempfile.NamedTemporaryFile(dir=args.output_dir,suffix='.png',delete=False) as handle:
                temporary=Path(handle.name)
            try:
                Image.fromarray(rgba,'RGBA').save(temporary)
                os.replace(temporary,args.output_dir/'sky_overlay.png')
            finally:
                temporary.unlink(missing_ok=True)
            atomic_json(args.output_dir/'sky_overlay.json',dict(enabled=True,updated_at_epoch=epoch,
                generation=f'{epoch:.6f}',model_revision=revision,mask_revision=mask.revision,width=1920,height=1080))
            last_error=None
        except Exception as exc:
            if str(exc)!=last_error:
                print(f'[OBS sky] {exc}',file=sys.stderr,flush=True)
                last_error=str(exc)
            atomic_json(args.output_dir/'sky_overlay.json',dict(enabled=False,updated_at_epoch=time.time()))
        while not STOP and time.monotonic()-started<5:
            time.sleep(.25)
    atomic_json(args.output_dir/'sky_overlay.json',dict(enabled=False,updated_at_epoch=time.time()))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
