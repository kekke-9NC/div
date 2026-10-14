"""Full resolution, chronological proposals for brief and faint meteors.

This supplies evidence to the classifier; it never labels a proposal a meteor.
The state spans coarse windows, so a two-frame flash at a window boundary is
not lost. Stationary stars, fading trails and long lived moving points are
filtered using the original frame times, rather than a maximum composite.
"""
from dataclasses import dataclass
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
import threading
from typing import Sequence

import cv2
import numpy as np
from scipy.spatial import cKDTree
import component_statistics

_component_pools = {}
_component_pool_lock = threading.Lock()


def _component_executor(workers):
    with _component_pool_lock:
        if workers not in _component_pools:
            _component_pools[workers] = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='flash-components')
        return _component_pools[workers]


def _frame_components(residual, binary):
    if not cv2.countNonZero(binary):
        return []
    _, labels, stats, centers = cv2.connectedComponentsWithStats(binary, 8)
    return _component_detections(residual, labels, stats, centers)


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
    if len(points) < 2:
        return None
    # These existing rejection conditions depend only on the original scalar
    # observations. Check them before allocating a matrix for noise tracks.
    if points[-1][0] - points[0][0] > frame_rate * 3:
        return None
    peak_value = max(point[6] for point in points)
    if peak_value < 12:
        return None
    selected = [point for point in points if point[6] >= max(8.0, peak_value * .20)]
    if len(selected) < 2 or max(max(point[4:6]) for point in selected) < 8:
        return None
    pts = np.asarray(points, dtype=np.float64)
    if pts[-1, 0] - pts[0, 0] > frame_rate * 3:
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


def _component_detections(residual, labels, stats, centers):
    """Exact component scoring with stable signal ranking and bounded Python work."""
    count = len(stats)
    detections = []
    # Keep label order (including stable signal-score ties). Only
    # components passing the existing size tests need a Python loop.
    eligible = np.flatnonzero(
        (stats[1:, cv2.CC_STAT_AREA] >= 3)
        & (stats[1:, cv2.CC_STAT_AREA] <= 1000)
        & (stats[1:, cv2.CC_STAT_WIDTH] <= 350)
        & (stats[1:, cv2.CC_STAT_HEIGHT] <= 350)
    ) + 1
    if len(eligible) > 400:
        # Aggregate exact integer sums in one image pass. The old loop
        # visited each cloud/noise component in Python. The same aggregation
        # is also cheaper for moderately dense frames; retain the small-loop
        # path for sparse images. Bincount is exact for these uint8 sums.
        aggregated = component_statistics.aggregate(labels, residual, eligible, count)
        if aggregated is None:
            keep = np.zeros(count, dtype=bool)
            keep[eligible] = True
            pixels = keep[labels.ravel()]
            flat_labels = labels.ravel()[pixels]
            flat_values = residual.ravel()[pixels]
            scores = np.bincount(flat_labels, weights=flat_values, minlength=count)
            peaks = np.zeros(count, dtype=residual.dtype)
            np.maximum.at(peaks, flat_labels, flat_values)
        else:
            scores, peaks = aggregated
        eligible = eligible[(scores[eligible] >= 20) & (peaks[eligible] >= 5)]
        eligible = eligible[np.argsort(-scores[eligible], kind='stable')[:400]]
        # All component peaks are now known from that same sparse pixel pass.
        # Avoid hundreds of temporary crop masks and reductions per frame.
        detections = [list(row) for row in zip(
            centers[eligible, 0].tolist(), centers[eligible, 1].tolist(),
            scores[eligible].astype(np.int64).tolist(),
            stats[eligible, cv2.CC_STAT_WIDTH].tolist(),
            stats[eligible, cv2.CC_STAT_HEIGHT].tolist(),
            peaks[eligible].astype(np.int64).tolist(),
        )]
    else:
        for k in eligible:
            x, y, w, h, area = map(int, stats[k])
            values = residual[y:y+h, x:x+w][labels[y:y+h, x:x+w] == k]
            peak, score = int(values.max()), int(values.sum())
            if peak >= 5 and score >= 20:
                detections.append([*centers[k], score, w, h, peak])
        detections.sort(key=lambda d: d[2], reverse=True)
    # Bound work in clouds; rank by signal, never by screen position.
    detections = detections[:400]
    return detections


class ShortFlashDetector:
    def __init__(self, frame_rate=25.0, border=30, component_workers=1):
        self.component_workers = max(1, min(4, int(component_workers)))
        self.frame_rate = max(1., float(frame_rate))
        self.border = max(0, int(border))
        self.background = None
        self.threshold = None
        self.threshold_u8 = None
        self.active = []
        self.last_frame = -1

    def consume(self, frames: Sequence[np.ndarray], first_frame: int, mask=None, roi=None,
                final=False):
        result = []
        pending = deque()
        executor = _component_executor(self.component_workers) if self.component_workers > 1 else None
        try:
            for index, residual, binary, reset in self._foreground_frames(frames, first_frame, mask, roi):
                if executor is None:
                    self._update_tracks(_frame_components(residual, binary), index, reset, result)
                else:
                    # Components depend only on these pixels. Background and
                    # track state still advance in the original frame order.
                    future = executor.submit(_frame_components, residual, binary.copy())
                    pending.append((future, index, reset))
                    if len(pending) >= self.component_workers:
                        task, at, clear = pending.popleft()
                        self._update_tracks(task.result(), at, clear, result)
            while pending:
                task, at, clear = pending.popleft()
                self._update_tracks(task.result(), at, clear, result)
        finally:
            wait([packet[0] for packet in pending])
        if final:
            result.extend(c for tr in self.active
                          if (c := candidate_from_points(tr, self.frame_rate)) is not None)
            self.active = []
        return result

    def _foreground_frames(self, frames, first_frame, mask, roi):
        allowed = permitted = excluded = binary = None
        permitted_shape = None
        last_read = self.last_frame
        for offset, frame in enumerate(frames):
            index = first_frame + offset
            if index <= last_read:
                continue
            reset = last_read >= 0 and index != last_read + 1
            if reset:
                self.background = None
            gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            smooth = cv2.GaussianBlur(gray, (3, 3), 0)
            height, width = gray.shape
            if self.background is None or self.background.shape != gray.shape:
                self.background = smooth.astype(np.float32)
                self.threshold = None
                reset = True
            residual = cv2.subtract(smooth, cv2.convertScaleAbs(self.background))
            if self.threshold is None or index % 100 == 0:
                base = cv2.convertScaleAbs(self.background)
                contrast = cv2.subtract(base, cv2.GaussianBlur(base, (0, 0), 3))
                contrast = cv2.dilate(contrast, np.ones((9, 9), np.uint8))
                self.threshold = np.minimum(25., 3.5 + contrast.astype(np.float32) * .18)
                self.threshold_u8 = np.ceil(self.threshold).astype(np.uint8)
            if binary is None or binary.shape != gray.shape:
                binary = np.empty_like(residual)
            np.greater_equal(residual, self.threshold_u8, out=binary)
            b = self.border
            if b:
                binary[:b] = binary[-b:] = 0
                binary[:, :b] = binary[:, -b:] = 0
            if mask is not None:
                if allowed is None or allowed.shape != gray.shape:
                    allowed = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                    excluded = allowed <= 0
                binary[excluded] = 0
            if roi is not None:
                if permitted_shape != gray.shape:
                    x1, y1, x2, y2 = [int(v) for v in roi]
                    permitted = np.zeros_like(binary)
                    permitted[max(0, y1):min(height, y2), max(0, x1):min(width, x2)] = 1
                    permitted_shape = gray.shape
                binary &= permitted
            yield index, residual, binary, reset
            cv2.accumulateWeighted(smooth, self.background, .025)
            last_read = index

    def _update_tracks(self, detections, index, reset, result):
        if reset:
            self.active = []
        if detections:
            tree = cKDTree(np.asarray(detections)[:, :2])
            choices = []
            positions, gates = _track_predictions(self.active, index)
            if len(positions):
                # Exact nearest-neighbour queries, with the same tree,
                # arithmetic, and greedy tie ordering as individual calls.
                distances, indices = tree.query(np.asarray(positions))
                # A batched two-component dot can round differently from
                # scalar BLAS norm. Recheck every boundary-near comparison
                # with the original arithmetic; greedy choices stay exact.
                for tid in np.flatnonzero(np.abs(distances - gates) <= 1e-10):
                    tr = self.active[tid]
                    if len(tr) >= 2:
                        last, previous = tr[-1], tr[-2]
                        velocity = (np.asarray(last[1:3]) - previous[1:3]) / max(1, last[0] - previous[0])
                        gates[tid] = max(12, min(90, np.linalg.norm(velocity) * (index-last[0]) * .7 + 12))
                for tid in np.flatnonzero(distances < gates):
                    choices.append((float(distances[tid]), int(tid), int(indices[tid])))
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
        self.last_frame = index


def _track_predictions(active, index):
    """Predict all tracks with the original elementwise arithmetic and order."""
    if not active:
        return np.empty((0, 2)), np.empty(0)
    lasts = [track[-1] for track in active]
    previous = [track[-2] if len(track) >= 2 else track[-1] for track in active]
    xy = np.asarray([point[1:3] for point in lasts])
    before = np.asarray([point[1:3] for point in previous])
    times = np.asarray([point[0] for point in lasts], dtype=np.int64)
    before_times = np.asarray([point[0] for point in previous], dtype=np.int64)
    gaps = index - times
    velocities = (xy - before) / np.maximum(1, times - before_times)[:, None]
    positions = xy + velocities * gaps[:, None]
    norms = np.sqrt(np.einsum('ij,ij->i', velocities, velocities))
    gates = np.maximum(12, np.minimum(90, norms * gaps * .7 + 12))
    gates[np.fromiter((len(track) < 2 for track in active), dtype=bool, count=len(active))] = 40
    return positions, gates
