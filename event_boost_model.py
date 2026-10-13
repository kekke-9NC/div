"""Event morphology and temporal statistics with portable boosted trees.

No CNN, camera identity, timestamps, filenames or previous predictions enter
the classifier. The checkpoint contains tensors and primitives only; sklearn
is needed for training, but never imported by the application runtime.
"""
from __future__ import annotations

import cv2
import numpy as np

ARCHITECTURE_NAME = "meteor_event_boost_v1"
PREPROCESS_VERSION = 2
_STAT_QUANTILES = np.array([0, .1, .25, .5, .75, .9, .99, 1], dtype=np.float64)


def _stat_quantiles(values):
    # Tiny curves dominate the call count. Sort their private copy once and
    # retain NumPy's exact linear interpolation arithmetic without constructing
    # a general axis/NaN/quantile-dispatch stack for each curve.
    if len(values)>256 or not np.isfinite(values).all():
        return np.quantile(values,_STAT_QUANTILES)
    ordered = np.sort(values)
    positions = (len(values)-1)*_STAT_QUANTILES
    low = np.floor(positions).astype(np.intp)
    high = np.minimum(low+1,len(values)-1)
    weight = positions-low
    lower,upper = ordered[low],ordered[high]
    difference = upper-lower
    result = lower+difference*weight
    np.subtract(upper,difference*(1.-weight),out=result,where=weight>=.5)
    return result


def _stats(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    if not len(values):
        return [0.0] * 10
    return [float(values.mean()), float(values.std()),
            *_stat_quantiles(values).tolist()]


def _curve_features(curve):
    curve = np.asarray(curve, dtype=np.float64)
    peak = max(float(curve.max()), 1e-8)
    normalized = curve / peak
    result = _stats(curve) + _stats(np.diff(normalized))
    result += cv2.resize(normalized[None], (24, 1), interpolation=cv2.INTER_AREA).ravel().tolist()
    for threshold in (.1, .25, .5, .75):
        active = normalized > threshold
        starts = np.flatnonzero(np.diff(np.r_[False, active, False].astype(int)) == 1)
        ends = np.flatnonzero(np.diff(np.r_[False, active, False].astype(int)) == -1)
        result += [float(active.mean()), float(len(starts)),
                   float(max(ends - starts, default=0) / len(curve))]
    centered = normalized - normalized.mean()
    power = np.abs(np.fft.rfft(centered, n=128)) ** 2
    power /= max(float(power.sum()), 1e-8)
    result += [float(chunk.sum()) for chunk in np.array_split(power[1:], 8)]
    result += [float(np.argmax(curve) / max(1, len(curve) - 1))]
    return result


def extract_event_features(image, kymograph, physical):
    """Shared train/runtime features from the versioned event representation."""
    image = np.asarray(image, dtype=np.float32)
    kymograph = np.asarray(kymograph, dtype=np.float32)
    maximum = image[0]
    result = np.asarray(physical, dtype=np.float32).tolist()
    for channel in image[:4]:
        result += _stats(channel)
        result += [float((channel > value).mean()) for value in (.05, .1, .2, .4, .6)]

    # Rotation-invariant shape, connectedness and time ordering of bright pixels.
    yy, xx = np.indices(maximum.shape, dtype=np.float64)
    for threshold in (.1, .25, .5):
        mask = maximum > max(float(maximum.max()) * threshold, .03)
        weight = maximum * mask
        mass = max(float(weight.sum()), 1e-8)
        cx, cy = float((weight * xx).sum() / mass), float((weight * yy).sum() / mass)
        dx, dy = xx - cx, yy - cy
        covariance = np.array([[(weight * dx * dx).sum(), (weight * dx * dy).sum()],
                               [(weight * dx * dy).sum(), (weight * dy * dy).sum()]]) / mass
        eig, axes = np.linalg.eigh(covariance)
        major = dx * axes[0, 1] + dy * axes[1, 1]
        minor = dx * axes[0, 0] + dy * axes[1, 0]
        result += [float(mask.mean()), float(eig[0]), float(eig[1]),
                   float(eig[0] / max(eig[1], 1e-8))]
        for coordinate in (major, minor):
            variance = max(float((weight * coordinate ** 2).sum() / mass), 1e-8)
            result += [float((weight * coordinate ** 3).sum() / mass / variance ** 1.5),
                       float((weight * coordinate ** 4).sum() / mass / variance ** 2)]
        count, _, components, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        areas = components[1:, cv2.CC_STAT_AREA]
        result += [float(count - 1), float(max(areas, default=0) / max(1, mask.sum()))]
        times = image[4][mask]
        result += _stats(times)
        if mask.sum() > 2:
            centered_time = image[4] - float((weight * image[4]).sum() / mass)
            time_var = max(float((weight * centered_time ** 2).sum() / mass), 1e-8)
            result += [float(abs((weight * centered_time * major).sum() / mass)
                             / np.sqrt(time_var * max(eig[1], 1e-8)))]
        else:
            result += [0.0]

    # Fixed time grid preserves flash shape, blinking, persistence and motion.
    for channel in kymograph:
        result += _stats(channel)
        result += _curve_features(channel.max(axis=1))
        result += _curve_features(channel.mean(axis=1))
        result += cv2.resize(channel, (8, 12), interpolation=cv2.INTER_AREA).ravel().tolist()
        mass = np.maximum(channel.sum(axis=1), 1e-8)
        positions = (channel * np.linspace(0, 1, channel.shape[1])).sum(axis=1) / mass
        active = channel.max(axis=1) > max(float(channel.max()) * .2, .03)
        result += _stats(np.diff(positions)[active[1:] & active[:-1]])
        if active.sum() >= 3:
            t = np.linspace(0, 1, len(positions))[active]
            p = positions[active]
            slope, intercept = np.polyfit(t, p, 1)
            result += [float(abs(slope)), float(np.std(p - slope * t - intercept))]
        else:
            result += [0., 0.]

    for first, second in ((1, 2), (2, 3), (1, 3)):
        a, b = image[first], image[second]
        result += [float(np.minimum(a, b).sum() / max(float(np.maximum(a, b).sum()), 1e-8)),
                   float(np.abs(a - b).mean())]
    # Align the spatial signal by its own principal axis, independently of the
    # detector's sometimes inaccurate Hough line. Keep the event's morphology.
    weight = np.maximum(maximum - max(float(maximum.max()) * .15, .03), 0.)
    mass = max(float(weight.sum()), 1e-8)
    cx, cy = float((weight * xx).sum() / mass), float((weight * yy).sum() / mass)
    dx, dy = xx - cx, yy - cy
    covariance = np.array([[(weight * dx * dx).sum(), (weight * dx * dy).sum()],
                           [(weight * dx * dy).sum(), (weight * dy * dy).sum()]]) / mass
    eig, axes = np.linalg.eigh(covariance)
    angle = float(np.degrees(np.arctan2(axes[1, 1], axes[0, 1])))
    scale = float(np.clip(40. / max(np.sqrt(max(eig[1], 0.)), 5.), .5, 3.))
    matrix = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    matrix[:, 2] += np.array([64. - cx, 32. - cy])
    for channel in image:
        aligned = cv2.warpAffine(channel, matrix, (128, 64))
        # Sum opposing orientations to eliminate the arbitrary PCA-axis sign.
        symmetric = .5 * (aligned + np.flip(aligned, (0, 1)))
        result += cv2.resize(symmetric, (16, 8), interpolation=cv2.INTER_AREA).ravel().tolist()
        result += _stats(aligned.max(axis=0)) + _stats(aligned.max(axis=1))
    gradients_x = cv2.Sobel(maximum, cv2.CV_32F, 1, 0)
    gradients_y = cv2.Sobel(maximum, cv2.CV_32F, 0, 1)
    magnitude, direction = cv2.cartToPolar(gradients_x, gradients_y)
    histogram = np.histogram(direction % np.pi, bins=16, range=(0, np.pi), weights=magnitude)[0]
    histogram /= max(float(histogram.sum()), 1e-8)
    result += sorted(histogram.tolist())
    return np.nan_to_num(np.asarray(result, dtype=np.float32), nan=0., posinf=1e6, neginf=-1e6)


def extract_motion_features(frames, rect=None, frame_rate=15.):
    """Measure the actual frame sequence, retaining speed and component shape."""
    from event_representation import _crop_square
    if not frames:
        raise ValueError("EventBoost requires at least one frame")
    stack = np.stack([_crop_square(frame, rect, 128) for frame in frames])
    return extract_motion_features_from_crops(stack,frame_rate)


def extract_motion_features_from_crops(stack,frame_rate):
    from event_representation import _robust_residual_cube
    response, _ = _robust_residual_cube(stack)
    curves = [[] for _ in range(8)]
    positions = []
    peaks = response.max(axis=(1, 2))
    means = response.mean(axis=(1, 2))
    percentiles = np.percentile(response, 99.5, axis=(1, 2))
    for frame, peak, mean, percentile in zip(response, peaks, means, percentiles):
        peak = float(peak)
        mask = (frame > max(3., peak * .3)).astype(np.uint8)
        count, labels, components, centroids = cv2.connectedComponentsWithStats(mask, 8)
        if count > 1:
            best = 1 + int(np.argmax(components[1:, cv2.CC_STAT_AREA]))
            area = float(components[best, cv2.CC_STAT_AREA])
            yy, xx = np.nonzero(labels == best)
            covariance = np.cov(np.stack([xx, yy]), bias=True) if len(xx) > 1 else np.zeros((2, 2))
            eig = np.linalg.eigvalsh(covariance)
            position = centroids[best] / 128.
        else:
            area, eig, position = 0., np.zeros(2), np.zeros(2)
        values = [peak, float(mean), float(percentile),
                  float(mask.mean()), area / 16384., float(count - 1),
                  float(np.sqrt(max(eig[1], 0.)) / 128.),
                  float(eig[0] / max(eig[1], 1e-8))]
        for curve, value in zip(curves, values):
            curve.append(value)
        positions.append(position)
    result = [len(stack) / max(float(frame_rate), 1.), float(len(stack))]
    for curve in curves:
        result += _curve_features(np.array(curve))
    positions = np.array(positions)
    valid = np.array(curves[4]) > 0.
    if valid.sum() >= 3:
        times = np.arange(len(stack))[valid] / max(float(frame_rate), 1.)
        p = positions[valid]
        coefficients = np.polyfit(times, p, 1)
        residual = p - (times[:, None] * coefficients[0] + coefficients[1])
        speeds = np.linalg.norm(np.diff(p, axis=0), axis=1) / np.maximum(np.diff(times), 1e-8)
        result += [float(np.linalg.norm(coefficients[0])), float(np.sqrt((residual ** 2).mean())),
                   float(np.linalg.norm(p[-1] - p[0])),
                   float(np.linalg.norm(p[-1] - p[0]) / max(np.linalg.norm(np.diff(p, axis=0), axis=1).sum(), 1e-8))]
        result += _stats(speeds)
    else:
        result += [0.] * 14
    return np.nan_to_num(np.array(result, dtype=np.float32), nan=0., posinf=1e6, neginf=-1e6)


class EventBoostClassifier:
    """Numerical tree traversal; safe weights-only checkpoint loading."""

    def __init__(self, checkpoint):
        if checkpoint.get("architecture") not in (ARCHITECTURE_NAME, "meteor_event_transformer_boost_v1"):
            raise ValueError("Unsupported EventBoost architecture")
        if checkpoint.get("preprocess_version") != PREPROCESS_VERSION:
            raise ValueError("Unsupported EventBoost preprocessing version")
        self.feature_count = int(checkpoint["feature_count"])
        self.members = [dict(member) for member in checkpoint["members"]]
        self.linear_members = [{key: value.numpy() if hasattr(value, "numpy") else value
                                for key, value in member.items()}
                               for member in checkpoint.get("linear_members", [])]
        if not self.members and not self.linear_members:
            raise ValueError("EventBoost checkpoint has no trees")
        for member in self.members:
            member["trees"] = [
                {key: value.numpy() if hasattr(value, "numpy") else np.asarray(value)
                 for key, value in tree.items()}
                for tree in member["trees"]
            ]
        # Live calls traverse one row. Python scalars avoid repeated NumPy
        # scalar boxing/int conversions at every node, with identical float64
        # thresholds, leaf values, tree order and accumulation arithmetic.
        self.scalar_members = [
            (float(member['bias']), [tuple(tree[key].tolist() for key in
                ('leaf', 'feature', 'threshold', 'left', 'right', 'value'))
                for tree in member['trees']]) for member in self.members]

    def predict_proba(self, features):
        values = np.atleast_2d(np.asarray(features, dtype=np.float64))
        if values.shape[1] != self.feature_count or not np.isfinite(values).all():
            raise ValueError("Invalid EventBoost feature vector")
        linear_probability = np.zeros(len(values), dtype=np.float64)
        for member in self.linear_members:
            normalized = (values - member["mean"]) / member["scale"]
            logits = normalized @ member["coef"] + member["intercept"]
            linear_probability += 1. / (1. + np.exp(-np.clip(logits, -700, 700)))
        member_count = len(self.members) + len(self.linear_members)
        if not self.members:
            return linear_probability / member_count
        if len(values) == 1:
            # Live detection usually submits one event. Avoid thousands of
            # temporary boolean arrays for single-row numerical traversal.
            probability = 0.
            row = values[0]
            for bias, trees in self.scalar_members:
                logit = bias
                for leaf, feature, threshold, left, right, value in trees:
                    node = 0
                    while not leaf[node]:
                        node = left[node] if row[feature[node]] <= threshold[node] else right[node]
                    logit += value[node]
                probability += 1. / (1. + np.exp(-np.clip(logit, -700, 700)))
            return (np.array([probability]) + linear_probability) / member_count
        probability = np.zeros(len(values), dtype=np.float64)
        rows = np.arange(len(values))
        for member in self.members:
            logits = np.full(len(values), float(member["bias"]), dtype=np.float64)
            for tree in member["trees"]:
                nodes = np.zeros(len(values), dtype=np.int64)
                active = ~tree["leaf"][nodes].astype(bool)
                while active.any():
                    selected = nodes[active]
                    left = values[rows[active], tree["feature"][selected]] <= tree["threshold"][selected]
                    nodes[active] = np.where(left, tree["left"][selected], tree["right"][selected])
                    active = ~tree["leaf"][nodes].astype(bool)
                logits += tree["value"][nodes]
            probability += 1. / (1. + np.exp(-np.clip(logits, -700, 700)))
        return (probability + linear_probability) / member_count


def export_boosted_trees(estimators, feature_count):
    """Convert fitted sklearn HistGradientBoosting models to portable tensors."""
    import torch
    members = []
    for estimator in estimators:
        trees = []
        for stage in estimator._predictors:
            nodes = stage[0].nodes
            if nodes["is_categorical"].any():
                raise ValueError("Only numerical features are supported")
            tree = {}
            for key, field in (("feature", "feature_idx"), ("threshold", "num_threshold"),
                               ("left", "left"), ("right", "right"),
                               ("leaf", "is_leaf"), ("value", "value")):
                dtype = np.float64 if key in ("threshold", "value") else np.int64
                tree[key] = torch.from_numpy(nodes[field].astype(dtype).copy())
            trees.append(tree)
        members.append({"bias": float(estimator._baseline_prediction[0, 0]), "trees": trees})
    return {"architecture": ARCHITECTURE_NAME, "preprocess_version": PREPROCESS_VERSION,
            "feature_count": int(feature_count), "members": members}
