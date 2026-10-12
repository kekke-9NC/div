#!/usr/bin/env python3
"""Project an explicitly approved fixed-camera model into a transparent OBS layer."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
import argparse
from collections import Counter
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
from PIL import Image, ImageDraw, ImageFont
from astropy.coordinates import SkyCoord, get_constellation
from astropy import units as u
from camera_plate_model import FixedCameraPlateModel
from local_wideangle_astrometry import _draw_constellation_lines, SIDEREAL_DAY_SECONDS
try:
    from .legacy_detection_mask import LegacyDetectionMask, default_settings_path
except ImportError:
    from obs_meteor_overlay.legacy_detection_mask import LegacyDetectionMask, default_settings_path

STOP = False
_CONSTELLATION_FEATURES = None
_LABEL_FONT = None


def atomic_json(path, data):
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', mode='w', encoding='utf-8', delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(data, handle, ensure_ascii=False, allow_nan=False)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_constellation_features():
    """Load named IAU stick figures once, preserving their catalog identity."""
    global _CONSTELLATION_FEATURES
    if _CONSTELLATION_FEATURES is not None:
        return _CONSTELLATION_FEATURES
    asset = Path(__file__).resolve().parent.parent / 'assets' / 'constellations.lines.json'
    features = []
    try:
        catalog = json.loads(asset.read_text(encoding='utf-8'))
        for feature in catalog.get('features', []):
            lines = []
            for line in feature.get('geometry', {}).get('coordinates', []):
                points = np.asarray(line, dtype=float)
                if points.ndim == 2 and points.shape[0] >= 2 and points.shape[1] >= 2:
                    lines.append(points[:, :2])
            if not lines:
                continue
            coordinates = np.concatenate(lines, axis=0)
            sky = SkyCoord(ra=coordinates[:, 0] * u.deg,
                           dec=coordinates[:, 1] * u.deg, frame='icrs')
            abbreviations = np.asarray(get_constellation(sky, short_name=True), dtype=str)
            names = np.asarray(get_constellation(sky, short_name=False), dtype=str)
            identifier = str(feature.get('id', ''))
            matching = np.flatnonzero(abbreviations == identifier)
            name = names[matching[0]] if len(matching) else Counter(names).most_common(1)[0][0]
            if name == 'Ophiucus':
                name = 'Ophiuchus'
            features.append((name, int(feature.get('properties', {}).get('rank', 3)), coordinates))
    except (OSError, ValueError, TypeError, KeyError, IndexError):
        features = []
    _CONSTELLATION_FEATURES = tuple(features)
    return _CONSTELLATION_FEATURES


def _constellation_label_font():
    global _LABEL_FONT
    if _LABEL_FONT is None:
        candidates = (
            '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
            '/System/Library/Fonts/Supplemental/Arial.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        )
        for candidate in candidates:
            try:
                _LABEL_FONT = ImageFont.truetype(candidate, 22)
                break
            except OSError:
                continue
        if _LABEL_FONT is None:
            _LABEL_FONT = ImageFont.load_default()
    return _LABEL_FONT


def _draw_constellation_labels(rgba, model, delta_ra, support):
    """Add a few unobtrusive English names at the currently projected positions."""
    height, width = support.shape
    font = _constellation_label_font()
    text_layer = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_layer)
    candidates = []
    for name, rank, coordinates in _load_constellation_features():
        ra = (coordinates[:, 0] - delta_ra) % 360.0
        x, y = model.world_to_pixel_values(ra, coordinates[:, 1])
        points = np.column_stack((x, y))
        usable = np.isfinite(points).all(axis=1)
        usable &= (x >= 0) & (x < width) & (y >= 0) & (y < height)
        indices = np.flatnonzero(usable)
        if len(indices):
            rounded = np.rint(points[indices]).astype(np.int32)
            inside = ((rounded[:, 0] >= 0) & (rounded[:, 0] < width)
                      & (rounded[:, 1] >= 0) & (rounded[:, 1] < height))
            usable[indices[~inside]] = False
            supported = indices[inside]
            if len(supported):
                safe_points = rounded[inside]
                usable[supported] = support[safe_points[:, 1], safe_points[:, 0]] > 0
        visible = points[usable]
        if len(visible) < 3:
            continue
        span = np.ptp(visible, axis=0)
        diagonal = float(np.hypot(span[0], span[1]))
        if diagonal < 85:
            continue
        center = np.median(visible, axis=0)
        candidates.append((rank, -diagonal, name, center))

    candidates.sort()
    placed = []
    # A handful of labels keeps the wide shot readable when several figures
    # share the frame. Rank-1/2 constellations are preferred, then larger figures.
    for _, _, name, center in candidates:
        if len(placed) >= 8:
            break
        bbox = draw.textbbox((0, 0), name, font=font, stroke_width=3)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        offsets = ((0, -28), (0, 28), (36, 0), (-36, 0), (0, -52), (0, 0))
        for dx, dy in offsets:
            cx, cy = center[0] + dx, center[1] + dy
            left = int(round(cx - text_width / 2 - bbox[0]))
            top = int(round(cy - text_height / 2 - bbox[1]))
            box = (left + bbox[0], top + bbox[1], left + bbox[2], top + bbox[3])
            x0, y0, x1, y1 = box
            # The sky viewport is cropped above the lower information panels.
            if x0 < 5 or y0 < 5 or x1 >= width - 5 or y1 >= 1004:
                continue
            if any(x0 < ox1 + 12 and x1 + 12 > ox0 and y0 < oy1 + 10 and y1 + 10 > oy0
                   for ox0, oy0, ox1, oy1 in placed):
                continue
            region = support[y0:y1 + 1:4, x0:x1 + 1:4]
            if region.size == 0 or float(np.mean(region > 0)) < 0.97:
                continue
            draw.text((left, top), name, font=font, fill=(188, 231, 249, 224),
                      stroke_width=3, stroke_fill=(4, 12, 22, 238))
            placed.append(box)
            break

    if placed:
        labels = np.asarray(text_layer).copy()
        labels[:, :, 3] = np.where(support > 0, labels[:, :, 3], 0)
        return np.asarray(Image.alpha_composite(Image.fromarray(rgba, 'RGBA'),
                                               Image.fromarray(labels, 'RGBA')))
    return rgba


def render_layer(payload, when, mask=None, opacity=0.45):
    width, height = int(payload['width']), int(payload['height'])
    reference = datetime.fromisoformat(payload['reference_datetime'])
    if reference.tzinfo is not None and when.tzinfo is None:
        when = when.astimezone()
    delta_ra = (when-reference).total_seconds()/SIDEREAL_DAY_SECONDS*360
    support = cv2.resize(np.asarray(payload['support_grid'],dtype=np.uint8), (width,height), interpolation=cv2.INTER_NEAREST)*255
    if mask is not None:
        support = cv2.bitwise_and(support, cv2.resize(mask,(width,height),interpolation=cv2.INTER_NEAREST))
    model = FixedCameraPlateModel(payload)
    lines = np.zeros((height,width,3),dtype=np.uint8)
    _draw_constellation_lines(lines,model,delta_ra,support,allow_partial_segments=True)
    rgb = cv2.cvtColor(lines,cv2.COLOR_BGR2RGB)
    strength = np.max(rgb,axis=2)
    # Modest opacity keeps faint stars and meteors visible through the sticks.
    opacity = float(opacity)
    if not np.isfinite(opacity) or not 0 <= opacity <= 1:
        raise ValueError('線の不透明度は0〜1で指定してください')
    alpha = np.rint(strength.astype(float)*opacity).astype(np.uint8)
    rgba = np.dstack((rgb,alpha))
    rgba[support==0] = 0
    return _draw_constellation_labels(rgba, model, delta_ra, support)


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
