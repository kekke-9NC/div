"""Use an actually active meteor frame and a measured OBS clock alignment."""
import json
from pathlib import Path
import re
import time
import cv2
import numpy as np


class EventTiming:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.cache = {}
        cv2.setNumThreads(1)

    def resolve(self, source):
        from latest_meteor_gif import event_datetime, raw_zoom_request, saved_media_path
        detected = event_datetime(source)
        original = detected.timestamp() if detected else None
        fallback = {'replay_at_epoch': original, 'timestamp_basis': 'detection_record'}
        try:
            info = saved_media_path(source.parent, source, '_info.txt')
            raw = raw_zoom_request(source, source.parent)
            if original is None or raw is None:
                return fallback
            text = info.read_text(encoding='utf-8')
            path, first, last, cx, cy = raw
            key = (str(source), info.stat().st_mtime_ns, path.stat().st_mtime_ns)
            if key not in self.cache:
                start_match = re.search(r'^Motion Start Frame:\s*(\d+)', text, re.M)
                end_match = re.search(r'^Motion End Frame:\s*(\d+)', text, re.M)
                start = max(first, int(start_match[1])) if start_match else first
                end = min(last, int(end_match[1])) if end_match else last
                if end < start or end - start > 250:
                    return fallback
                cap = cv2.VideoCapture(str(path))
                try:
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if fps <= 0:
                        return fallback
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                    brightness = []
                    for frame_number in range(start, end + 1):
                        ok, frame = cap.read()
                        if not ok:
                            break
                        h, w = frame.shape[:2]
                        # A local positive brightness change is more visible than
                        # the midpoint of a clip containing pre/post padding.
                        crop = frame[max(0, int(cy)-128):min(h, int(cy)+128), max(0, int(cx)-128):min(w, int(cx)+128)]
                        brightness.append(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32))
                    if len(brightness) != end - start + 1:
                        return fallback
                    stack = np.asarray(brightness)
                    background = np.median(stack, axis=0)
                    positive = np.maximum(stack - background - 8, 0)
                    region = np.zeros(background.shape, dtype=np.uint8)
                    points = [re.search(rf'^Detected Line {name} \(px\):\s*\(([-+]?\d+(?:\.\d+)?),\s*([-+]?\d+(?:\.\d+)?)\)', text, re.M) for name in ('Start', 'End')]
                    left, top = max(0, int(cx)-128), max(0, int(cy)-128)
                    if all(points):
                        endpoints = [(round(float(point[1])-left), round(float(point[2])-top)) for point in points]
                        cv2.line(region, endpoints[0], endpoints[1], 1, 24)
                    else:
                        cv2.circle(region, (round(cx-left), round(cy-top)), 32, 1, -1)
                    positive *= region
                    score = positive.reshape(len(stack), -1).sum(axis=1)
                    peak = start + int(np.argmax(score)) if float(score.max()) > 0 else (start + end) // 2
                    # Legacy event name uses integer midpoint of buffered clip.
                    base = original - ((first + last) // 2) / fps
                    self.cache[key] = {'peak': base + peak / fps, 'start': base + start / fps, 'end': base + end / fps, 'frame': peak}
                    if len(self.cache) > 30:
                        self.cache.pop(next(iter(self.cache)))
                finally:
                    cap.release()
            timing = self.cache[key]
            result = {'replay_at_epoch': timing['peak'], 'visible_start_epoch': timing['start'],
                      'visible_end_epoch': timing['end'], 'replay_frame': timing['frame'], 'timestamp_basis': 'recording_frame'}
            alignment = json.loads((self.output_dir / 'broadcast_timing.json').read_text())
            if (alignment.get('calibrated') and time.time() - alignment['matched_at_epoch'] < 180
                    and timing['start'] > alignment['session_start_epoch']):
                offset = float(alignment['offset_seconds'])
                for field in ('replay_at_epoch', 'visible_start_epoch', 'visible_end_epoch'):
                    result[field] += offset
                result.update(timestamp_basis='obs_frame_match', broadcast_offset_seconds=offset,
                              timing_uncertainty_seconds=alignment['uncertainty_seconds'])
            return result
        except (OSError, ValueError, KeyError, cv2.error):
            return locals().get('result', fallback)
