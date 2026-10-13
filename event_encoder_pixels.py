"""CPU-only DINO event pixel preparation, usable by spawned workers."""
import cv2
import numpy as np


def encoder_pixels(image):
    maximum = np.asarray(image[0], dtype=np.float32)
    scale = max(float(np.quantile(maximum, .999)), .05)
    gray = np.repeat(maximum[None], 3, axis=0)
    temporal = image[1:4]
    views = []
    for view in (gray, temporal):
        resized = np.stack([cv2.resize(channel, (224, 224), interpolation=cv2.INTER_LINEAR)
                            for channel in np.clip(view / scale, 0., 1.)])
        views.append((resized - np.array([.485, .456, .406], dtype=np.float32)[:, None, None])
                     / np.array([.229, .224, .225], dtype=np.float32)[:, None, None])
    return np.stack(views)
