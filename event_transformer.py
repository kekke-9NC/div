"""Offline DINOv2 event encoder; all weights are embedded in our checkpoint."""
from __future__ import annotations

import cv2
import numpy as np
import torch
from event_encoder_pixels import encoder_pixels

ARCHITECTURE_NAME = "meteor_event_transformer_boost_v1"
ENCODER_INPUT_VERSION = 1
ENCODER_ID = "facebook/dinov2-small"
ENCODER_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"


class EventImageEncoder:
    def __init__(self, checkpoint, device="cpu"):
        from transformers import Dinov2Config, Dinov2Model
        self.device = torch.device(device)
        self.network = Dinov2Model(Dinov2Config(**checkpoint["config"]))
        self.network.load_state_dict(checkpoint["state_dict"])
        self.network.to(self.device).eval()

    def encode(self, images):
        return self.encode_prepared([encoder_pixels(image) for image in images])

    def encode_prepared(self, prepared_pixels):
        pixels = np.concatenate(prepared_pixels)
        with torch.inference_mode():
            hidden = self.network(pixel_values=torch.from_numpy(pixels).to(self.device)).last_hidden_state
            vectors = torch.cat([hidden[:, 0], hidden[:, 1:].mean(dim=1)], dim=1)
            return vectors.reshape(len(prepared_pixels), -1).cpu().numpy().astype(np.float32)
