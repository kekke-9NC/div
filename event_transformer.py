"""Offline DINOv2 event encoder; all weights are embedded in our checkpoint."""
from __future__ import annotations

import cv2
import numpy as np
import torch
import threading
import warnings
from event_encoder_pixels import encoder_pixels

ARCHITECTURE_NAME = "meteor_event_transformer_boost_v1"
ENCODER_INPUT_VERSION = 1
ENCODER_ID = "facebook/dinov2-small"
ENCODER_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"


class _EncoderProjection(torch.nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, pixels):
        hidden = self.network(pixel_values=pixels).last_hidden_state
        return torch.cat([hidden[:,0],hidden[:,1:].mean(dim=1)],dim=1)


class EventImageEncoder:
    def __init__(self, checkpoint, device="cpu"):
        from transformers import Dinov2Config, Dinov2Model
        self.device = torch.device(device)
        self.network = Dinov2Model(Dinov2Config(**checkpoint["config"]))
        self.network.load_state_dict(checkpoint["state_dict"])
        self.network.to(self.device).eval()
        # Dinov2Config and weights are owned by this loaded encoder. Model
        # reload constructs a new instance; replacing its network is eager.
        self._projection_owner = self.network
        self._trace_lock = threading.Lock()
        self._trace_checked = False
        self._single_projection = None

    def _projection_for(self, pixels):
        # Only the fixed single-event live geometry is traced. Offline batches,
        # other devices/dtypes and training retain their original forward path.
        if (self.device.type!='mps' or pixels.device.type!='mps' or pixels.dtype!=torch.float32
                or tuple(pixels.shape)!=(2,3,224,224)
                or self.network is not getattr(self,'_projection_owner',None) or self.network.training):
            return None
        with self._trace_lock:
            if self._trace_checked:
                return self._single_projection
            self._trace_checked = True
            try:
                projection = _EncoderProjection(self.network).eval()
                # Channel-count/attention-shape branches are constant for the
                # guarded geometry. No image-value branches are specialized.
                with warnings.catch_warnings(),torch.inference_mode():
                    warnings.simplefilter('ignore',torch.jit.TracerWarning)
                    traced = torch.jit.trace(projection,pixels,check_trace=False,strict=False)
                    probes = (pixels,torch.zeros_like(pixels),
                              torch.linspace(-2.2,2.6,pixels.numel(),device=pixels.device,dtype=pixels.dtype).reshape_as(pixels))
                    for probe in probes:
                        expected,observed = projection(probe).cpu().numpy(),traced(probe).cpu().numpy()
                        if not np.array_equal(expected,observed):
                            return None
                self._single_projection = traced
            except Exception as exc:
                print(f'Encoder acceleration unavailable; retaining eager inference: {exc}')
            return self._single_projection

    def encode(self, images):
        return self.encode_prepared([encoder_pixels(image) for image in images])

    def _eager_vectors(self, pixels):
        hidden = self.network(pixel_values=pixels).last_hidden_state
        return torch.cat([hidden[:,0],hidden[:,1:].mean(dim=1)],dim=1)

    def encode_prepared(self, prepared_pixels):
        pixels = np.concatenate(prepared_pixels)
        with torch.inference_mode():
            tensor = torch.from_numpy(pixels).to(self.device)
            projection = self._projection_for(tensor) if len(prepared_pixels)==1 else None
            if projection is not None:
                try:
                    vectors = projection(tensor)
                except Exception as exc:
                    with self._trace_lock:
                        self._single_projection = None
                    print(f'Encoder acceleration failed; retaining eager inference: {exc}')
                    vectors = self._eager_vectors(tensor)
            else:
                vectors = self._eager_vectors(tensor)
            return vectors.reshape(len(prepared_pixels), -1).cpu().numpy().astype(np.float32)
