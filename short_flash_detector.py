"""Full resolution, chronological proposals for brief and faint meteors.

This supplies evidence to the classifier; it never labels a proposal a meteor.
The state spans coarse windows, so a two-frame flash at a window boundary is
not lost. Stationary stars, fading trails and long lived moving points are
filtered using the original frame times, rather than a maximum composite.
"""
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class FlashCandidate:
    start_frame: int
    end_frame: int
    start: tuple[float, float]
    end: tuple[float, float]
    peak: float
    fit_r2: float

    @property
    def line(self):
        return tuple(tuple(int(round(v)) for v in p) for p in (self.start, self.end))

    def classifier_rect(self, width, height):
        # Keep faint activity visible to the event model, while preserving the
        # full trajectory and enough surrounding sky. This is not a new model
        # threshold or a position/time exception for the regression recording.
        minimum = 160 if self.peak < 40 else 256
        size = min(width, height, max(minimum, int(np.max(np.abs(np.subtract(self.end, self.start))) + 50)))
        cx, cy = np.mean([self.start, self.end], axis=0)
        x = int(np.clip(cx - size / 2, 0, width - size))
        y = int(np.clip(cy - size / 2, 0, height - size))
        return x, y, x + size, y + size


def candidate_from_points(points, frame_rate=25.0):
    """Validate a component track, keeping its luminous head, not its decay."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2 or pts[-1, 0] - pts[0, 0] > frame_rate * 3:
        return None
    peak = float(pts[:, 6].max())
    pts = pts[pts[:, 6] >= max(8.0, peak * .20)]
    if len(pts) < 2:
        return None
    times, xy = pts[:, 0], pts[:, 1:3]
    span = float(times[-1] - times[0])
    displacement = float(np.linalg.norm(xy[-1] - xy[0]))
    if (span <= 0 or span > frame_rate * 1.6 or
            len(pts) / (span + 1) < .7 or displacement < 4 or
            displacement / span < .75 or peak < 12 or pts[:, 4:6].max() < 8):
        return None
    design = np.column_stack([times - times.mean(), np.ones(len(times))])
    coefficients = np.linalg.lstsq(design, xy, rcond=None)[0]
    variance = float(np.square(xy - xy.mean(axis=0)).sum())
    r2 = 1 - float(np.square(xy - design @ coefficients).sum()) / max(1e-5, variance)
    if r2 < .75:
        return None
    fitted = design[[0, -1]] @ coefficients
    return FlashCandidate(int(times[0]), int(times[-1]), tuple(fitted[0]),
                          tuple(fitted[1]), peak, r2)


class ShortFlashDetector:
    def __init__(self, frame_rate=25.0, border=30):
        self.frame_rate = max(1., float(frame_rate))
        self.border = max(0, int(border))
        self.background = None
        self.threshold = None
        self.active = []
        self.last_frame = -1

    def consume(self, frames: Sequence[np.ndarray], first_frame: int, mask=None, roi=None,
                final=False):
        result = []
        for offset, frame in enumerate(frames):
            index = first_frame + offset
            if index <= self.last_frame:
                continue
            if self.last_frame >= 0 and index != self.last_frame + 1:
                # Discontinuous samples cannot establish motion or duration.
                self.active = []
                self.background = None
            gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            smooth = cv2.GaussianBlur(gray, (3, 3), 0)
            height, width = gray.shape
            if self.background is None or self.background.shape != gray.shape:
                self.background = smooth.astype(np.float32)
                self.threshold = None
                self.active = []
            residual = cv2.subtract(smooth, cv2.convertScaleAbs(self.background))
            if self.threshold is None or index % 100 == 0:
                base = cv2.convertScaleAbs(self.background)
                contrast = cv2.subtract(base, cv2.GaussianBlur(base, (0, 0), 3))
                contrast = cv2.dilate(contrast, np.ones((9, 9), np.uint8))
                self.threshold = np.minimum(25., 3.5 + contrast.astype(np.float32) * .18)
            binary = (residual >= self.threshold).astype(np.uint8)
            b = self.border
            if b:
                binary[:b] = binary[-b:] = 0
                binary[:, :b] = binary[:, -b:] = 0
            if mask is not None:
                allowed = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                binary[allowed <= 0] = 0
            if roi is not None:
                x1, y1, x2, y2 = [int(v) for v in roi]
                permitted = np.zeros_like(binary)
                permitted[max(0, y1):min(height, y2), max(0, x1):min(width, x2)] = 1
                binary &= permitted
            count, labels, stats, centers = cv2.connectedComponentsWithStats(binary, 8)
            detections = []
            for k in range(1, count):
                x, y, w, h, area = map(int, stats[k])
                if area < 3 or area > 1000 or max(w, h) > 350:
                    continue
                values = residual[y:y+h, x:x+w][labels[y:y+h, x:x+w] == k]
                peak, score = int(values.max()), int(values.sum())
                if peak >= 5 and score >= 20:
                    detections.append([*centers[k], score, w, h, peak])
            # Bound work in clouds; rank by signal, never by screen position.
            detections.sort(key=lambda d: d[2], reverse=True)
            detections = detections[:400]
            if detections:
                tree = cKDTree(np.asarray(detections)[:, :2])
                choices = []
                for tid, tr in enumerate(self.active):
                    last = tr[-1]
                    gap = index - last[0]
                    velocity = np.zeros(2)
                    if len(tr) >= 2:
                        previous = tr[-2]
                        velocity = (np.asarray(last[1:3]) - previous[1:3]) / max(1, last[0] - previous[0])
                    distance, di = tree.query(np.asarray(last[1:3]) + velocity * gap)
                    gate = 40 if len(tr) < 2 else max(12, min(90, np.linalg.norm(velocity) * gap * .7 + 12))
                    if distance < gate:
                        choices.append((float(distance), tid, int(di)))
                used_tracks, used_detections = set(), set()
                for _, tid, di in sorted(choices):
                    if tid in used_tracks or di in used_detections:
                        continue
                    used_tracks.add(tid)
                    used_detections.add(di)
                    self.active[tid].append([index, *detections[di]])
                for di, detection in enumerate(detections):
                    if di not in used_detections:
                        self.active.append([[index, *detection]])
            remaining = []
            for tr in self.active:
                if index - tr[-1][0] > 4:
                    candidate = candidate_from_points(tr, self.frame_rate)
                    if candidate is not None:
                        result.append(candidate)
                else:
                    # Long tracks are kept as duration evidence but bounded in
                    # memory. Their original start ensures they stay rejected.
                    if len(tr) > 150:
                        tr[:] = [tr[0], *tr[-100:]]
                    remaining.append(tr)
            self.active = remaining
            cv2.accumulateWeighted(smooth, self.background, .025)
            self.last_frame = index
        if final:
            result.extend(c for tr in self.active
                          if (c := candidate_from_points(tr, self.frame_rate)) is not None)
            self.active = []
        return result
