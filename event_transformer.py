"""Offline DINOv2 event encoder; all weights are embedded in our checkpoint."""
from __future__ import annotations

import cv2
import numpy as np
import torch

ARCHITECTURE_NAME = "meteor_event_transformer_boost_v1"
ENCODER_INPUT_VERSION = 1
ENCODER_ID = "facebook/dinov2-small"
ENCODER_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"


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


class EventImageEncoder:
    def __init__(self, checkpoint, device="cpu"):
        from transformers import Dinov2Config, Dinov2Model
        self.device = torch.device(device)
        self.network = Dinov2Model(Dinov2Config(**checkpoint["config"]))
        self.network.load_state_dict(checkpoint["state_dict"])
        self.network.to(self.device).eval()

    def encode(self, images):
        pixels = np.concatenate([encoder_pixels(image) for image in images])
        with torch.inference_mode():
            hidden = self.network(pixel_values=torch.from_numpy(pixels).to(self.device)).last_hidden_state
            vectors = torch.cat([hidden[:, 0], hidden[:, 1:].mean(dim=1)], dim=1)
            return vectors.reshape(len(images), -1).cpu().numpy().astype(np.float32)
