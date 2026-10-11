#!/usr/bin/env python3
"""Measure recording-to-broadcast time using frames, without another RTSP reader."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
import argparse
import base64
import hashlib
import json
from pathlib import Path
import signal
import sys
import time
import uuid
import cv2
import numpy as np
import websocket

STOP = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
try:
    import config as app_config
except ImportError:
    app_config = None


def atomic_json(path, payload):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


class OBS:
    def __init__(self):
        config = json.loads((Path.home() / 'Library/Application Support/obs-studio/plugin_config/obs-websocket/config.json').read_text())
        self.ws = websocket.create_connection(f"ws://127.0.0.1:{config['server_port']}", timeout=5)
        hello = json.loads(self.ws.recv())['d']
        identify = {'rpcVersion': 1, 'eventSubscriptions': 0}
        if 'authentication' in hello:
            auth = hello['authentication']
            secret = base64.b64encode(hashlib.sha256((config['server_password'] + auth['salt']).encode()).digest()).decode()
            identify['authentication'] = base64.b64encode(hashlib.sha256((secret + auth['challenge']).encode()).digest()).decode()
        self.ws.send(json.dumps({'op': 1, 'd': identify}))
        if json.loads(self.ws.recv())['op'] != 2:
            raise ValueError('OBS authentication failed')

    def request(self, kind, data=None):
        request_id = str(uuid.uuid4())
        self.ws.send(json.dumps({'op': 6, 'd': {'requestType': kind, 'requestId': request_id, 'requestData': data or {}}}))
        while True:
            response = json.loads(self.ws.recv())
            if response['op'] == 7 and response['d']['requestId'] == request_id:
                result = response['d']
                if not result['requestStatus']['result']:
                    raise ValueError(f'OBS request failed: {kind}')
                return result.get('responseData', {})

    def close(self):
        self.ws.close()


def features(frame):
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Avoid masked roof/ground edges and the overlay; compare moving cloud detail.
    small = cv2.resize(gray[160:880, 320:1600], (256, 144), interpolation=cv2.INTER_AREA).astype(np.float32)
    return (small - cv2.GaussianBlur(small, (0, 0), 2)).ravel()


def match_frame(image, wall_epoch, source):
    birth = source.stat().st_birthtime
    cap = cv2.VideoCapture(str(source))
    try:
        fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or total <= 0:
            return None
        first, last = 0, total - 1
        if last - first < fps * 2:
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        frames = []
        for _ in range(first, last + 1):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(features(frame))
        if len(frames) < fps * 2:
            return None
        matrix = np.asarray(frames)
        mean = matrix.mean(axis=0)
        matrix -= mean
        target = features(image) - mean
        scores = (matrix @ target) / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(target) + 1e-6)
        best = int(np.argmax(scores))
        distant = np.abs(np.arange(len(scores)) - best) > fps * 2
        margin = float(scores[best] - np.max(scores[distant])) if distant.any() else 0
        offset = wall_epoch - birth - (first + best) / fps
        if scores[best] < .72 or margin < .04 or abs(offset) > 8 or best < 2 or best >= len(scores) - 2:
            return None
        frame_number = first + best
        return {'source': str(source), 'frame': frame_number, 'fps': fps,
                'wall_epoch': wall_epoch, 'source_epoch': birth + frame_number / fps,
                'offset_seconds': offset,
                'score': float(scores[best]), 'margin': margin}
    finally:
        cap.release()


def camera_source_name(output_dir):
    try:
        configured = (output_dir / 'camera_source.txt').read_text(encoding='utf-8').strip()
    except OSError:
        configured = ''
    return configured or 'メディア'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--parent-pid', type=int)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    os.nice(10)
    def stop(*_):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    output = args.output_dir / 'broadcast_timing.json'
    root = Path(app_config.RTSP_SAVE_ROOT) if app_config is not None else PROJECT_ROOT / 'rtsp'
    pending, anchors, session_start = [], [], None
    try:
        previous = json.loads(output.read_text())
        if previous.get('calibrated') and time.time() - previous['matched_at_epoch'] < 180:
            anchors, session_start = previous['anchors'], previous['session_start_epoch']
    except (OSError, ValueError, KeyError):
        pass
    last_capture = 0
    while not STOP:
        try:
            if args.parent_pid:
                os.kill(args.parent_pid, 0)
            now = time.time()
            if now - last_capture >= 20:
                obs = OBS()
                try:
                    source_name = camera_source_name(args.output_dir)
                    status = obs.request('GetMediaInputStatus', {'inputName': source_name})
                    if status['mediaState'] != 'OBS_MEDIA_STATE_PLAYING':
                        raise ValueError('Camera is not playing')
                    connection_start = time.time() - status['mediaCursor'] / 1000
                    if session_start is None or abs(connection_start - session_start) > 5:
                        pending, anchors, session_start = [], [], connection_start
                        atomic_json(output, {'calibrated': False, 'updated_at_epoch': now})
                    before = time.time()
                    screenshot = obs.request('GetSourceScreenshot', {'sourceName': source_name, 'imageFormat': 'png'})
                    after = time.time()
                    if after - before < .5:
                        image = cv2.imdecode(np.frombuffer(base64.b64decode(screenshot['imageData'].split(',', 1)[1]), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                        if image is not None and image.shape == (1080, 1920):
                            pending.append(((before + after) / 2, image))
                    last_capture = now
                finally:
                    obs.close()
            # Only finished files; their birth time survives the recorder's rename.
            for stamp, image in list(pending):
                if now - stamp < 75:
                    continue
                from datetime import datetime
                folders = {root / datetime.fromtimestamp(stamp + delta).strftime('%Y%m%d/%H') for delta in (-60, 0, 60)}
                candidates = [p for folder in folders for p in folder.glob('*.mp4')
                              if '_temp_' not in p.name and stamp - 75 < p.stat().st_birthtime < stamp + 8]
                matches = [match for p in candidates if (match := match_frame(image, stamp, p)) is not None]
                pending = [(other_stamp, other_image) for other_stamp, other_image in pending if other_stamp != stamp]
                if matches:
                    anchors.append(max(matches, key=lambda item: item['score']))
            anchors = [a for a in anchors if now - a['wall_epoch'] < 900][-12:]
            if len(anchors) >= 2:
                offsets = np.asarray([a['offset_seconds'] for a in anchors])
                median = float(np.median(offsets))
                consistent = [a for a in anchors if abs(a['offset_seconds'] - median) < .4]
                if len(consistent) >= 2:
                    offset = float(np.median([a['offset_seconds'] for a in consistent]))
                    atomic_json(output, {'calibrated': True, 'updated_at_epoch': now,
                        'matched_at_epoch': max(a['wall_epoch'] for a in consistent),
                        'session_start_epoch': session_start, 'offset_seconds': offset,
                        'uncertainty_seconds': max(.25, max(abs(a['offset_seconds'] - offset) for a in consistent)),
                        'anchors': consistent})
            pending = [(stamp, image) for stamp, image in pending if now - stamp < 180]
        except ProcessLookupError:
            break
        except Exception as exc:
            # Never log credentials, input settings, or raw WebSocket responses.
            print(f'[broadcast timing] retry: {type(exc).__name__}', flush=True)
        for _ in range(5):
            if STOP:
                break
            time.sleep(1)


if __name__ == '__main__':
    main()
