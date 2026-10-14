"""Camera-invariant event representation and lightweight meteor classifier."""
from __future__ import annotations
from typing import Sequence, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
from event_representation import (
    ARCHITECTURE_NAME, PREPROCESS_VERSION, IMAGE_SIZE, KYMO_HEIGHT, KYMO_WIDTH, FEATURE_COUNT, _gray, _crop_square, _uint8_median, _line_in_crop, _robust_residual_cube, _compress_response, _track_aligned_cube, _centroid_track, _physical_features, build_universal_inputs
)

def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class SeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.depth_norm = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.point_norm = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.activation = nn.SiLU(inplace=True)
        self.skip = None
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(_group_count(out_channels), out_channels),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs if self.skip is None else self.skip(inputs)
        output = self.activation(self.depth_norm(self.depthwise(inputs)))
        output = self.point_norm(self.pointwise(output))
        return self.activation(output + residual)


class UniversalEncoder(nn.Module):
    def __init__(self, in_channels: int, widths: Sequence[int]):
        super().__init__()
        first = int(widths[0])
        layers = [
            nn.Conv2d(in_channels, first, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(_group_count(first), first),
            nn.SiLU(inplace=True),
        ]
        current = first
        for width in widths:
            width = int(width)
            layers.append(SeparableBlock(current, width, stride=2 if current != width else 1))
            layers.append(SeparableBlock(width, width, stride=1))
            current = width
        self.network = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output_channels = current

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.pool(self.network(inputs)).flatten(1)


class MeteorFusionUniversal(nn.Module):
    """Small GroupNorm network intended to be robust to unseen cameras."""

    def __init__(self, feature_count: int = FEATURE_COUNT):
        super().__init__()
        self.image_encoder = UniversalEncoder(5, (24, 40, 72, 128, 192))
        self.kymo_encoder = UniversalEncoder(3, (16, 32, 64, 96))
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_count, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        total = self.image_encoder.output_channels + self.kymo_encoder.output_channels + 64
        self.classifier = nn.Sequential(
            nn.Linear(total, 160),
            nn.LayerNorm(160),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(160, 1),
        )

    def forward(
        self,
        image: torch.Tensor,
        kymograph: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        fused = torch.cat(
            [
                self.image_encoder(image),
                self.kymo_encoder(kymograph),
                self.feature_encoder(features),
            ],
            dim=1,
        )
        return self.classifier(fused).squeeze(1)


def augment_universal_inputs(
    image: torch.Tensor,
    kymograph: torch.Tensor,
    features: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sensor and geometry augmentation that preserves temporal semantics."""
    if torch.rand(()) < 0.5:
        image = torch.flip(image, dims=(-1,))
        kymograph = torch.flip(kymograph, dims=(-1,))
    if torch.rand(()) < 0.5:
        image = torch.flip(image, dims=(-2,))
    rotations = int(torch.randint(0, 4, ()).item())
    if rotations:
        image = torch.rot90(image, rotations, dims=(-2, -1))
    if torch.rand(()) < 0.25:
        image = image.clone()
        image[[1, 3]] = image[[3, 1]]
        signal = image[0] > 0.02
        image[4] = torch.where(signal, 1.0 - image[4], image[4])
        kymograph = torch.flip(kymograph, dims=(-2,))

    strength = float(torch.empty(()).uniform_(0.75, 1.25))
    gamma = float(torch.empty(()).uniform_(0.8, 1.25))
    image = image.clone()
    image[:4] = torch.clamp(image[:4] * strength, 0.0, 1.0).pow(gamma)
    kymograph = torch.clamp(kymograph * strength, 0.0, 1.0).pow(gamma)
    noise_sigma = float(torch.empty(()).uniform_(0.0, 0.035))
    if noise_sigma > 0.0:
        image[:4] = torch.clamp(
            image[:4] + torch.randn_like(image[:4]) * noise_sigma,
            0.0,
            1.0,
        )
        kymograph = torch.clamp(
            kymograph + torch.randn_like(kymograph) * noise_sigma,
            0.0,
            1.0,
        )
    return image, kymograph, features
