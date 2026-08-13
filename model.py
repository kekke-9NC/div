import os
import multiprocessing
import threading
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

import config
import model_catalog
from universal_meteor_model import (
    ARCHITECTURE_NAME as UNIVERSAL_ARCHITECTURE_NAME,
    MeteorFusionUniversal,
    build_universal_inputs,
)


def _configure_torch_threads() -> None:
    """Keep PyTorch's CPU helper pool inside the app-wide CPU budget."""
    try:
        requested = int(os.environ.get("METEOR_NATIVE_THREADS", "4"))
    except ValueError:
        requested = 4
    requested = max(1, min(requested, os.cpu_count() or 1))
    try:
        torch.set_num_threads(requested)
    except (RuntimeError, ValueError):
        pass
    try:
        # Inter-op parallelism is unnecessary for this single-network inference
        # path and can only be set before parallel work begins.
        torch.set_num_interop_threads(1)
    except (RuntimeError, ValueError):
        pass


_configure_torch_threads()


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class ComplexCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(ComplexCNN, self).__init__()
        self.in_channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride=s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def _select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = _select_device()
print(f"Using device: {device}")


model: Optional[nn.Module] = None
transform = None
_model_ready = False
_model_lock = threading.RLock()
_active_model_info: Dict = {
    "model_path": "",
    "mean": list(model_catalog.DEFAULT_MEAN),
    "std": list(model_catalog.DEFAULT_STD),
    "input_resize": list(model_catalog.DEFAULT_INPUT_RESIZE),
    "class_names": list(model_catalog.DEFAULT_CLASS_NAMES),
    "meteor_class_index": 0,
    "architecture": "complex_cnn_v1",
}


def _build_transform(mean, std, input_resize):
    ops = []
    if input_resize is not None:
        try:
            h = int(input_resize[0])
            w = int(input_resize[1])
            if h > 0 and w > 0:
                ops.append(transforms.Resize((h, w)))
        except Exception:
            pass
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return transforms.Compose(ops)


def _load_state_dict(model_path: str):
    try:
        return torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(model_path, map_location=device)


def _reload_model_unlocked(model_path: Optional[str] = None, metadata: Optional[Dict] = None):
    global model, transform, _model_ready, _active_model_info

    target_path = model_path or config.MODEL_PATH
    if not target_path:
        return False, "Model path is empty."
    if not os.path.exists(target_path):
        _model_ready = False
        return False, f"Model file not found: {target_path}"

    meta = metadata if isinstance(metadata, dict) else model_catalog.load_model_metadata(target_path)

    try:
        state_dict = _load_state_dict(target_path)
        architecture = str(meta.get("architecture", "complex_cnn_v1"))
        if architecture == UNIVERSAL_ARCHITECTURE_NAME:
            loaded = MeteorFusionUniversal(
                feature_count=int(meta.get("feature_count", 12))
            ).to(device)
        else:
            loaded = ComplexCNN(num_classes=2).to(device)
        loaded.load_state_dict(state_dict)
        loaded.eval()

        mean = meta.get("mean", list(model_catalog.DEFAULT_MEAN))
        std = meta.get("std", list(model_catalog.DEFAULT_STD))
        input_resize = meta.get("input_resize", list(model_catalog.DEFAULT_INPUT_RESIZE))

        model = loaded
        transform = _build_transform(mean, std, input_resize)
        _model_ready = True

        _active_model_info = {
            "model_path": target_path,
            "mean": [float(x) for x in mean],
            "std": [float(x) for x in std],
            "input_resize": input_resize,
            "class_names": list(meta.get("class_names", model_catalog.DEFAULT_CLASS_NAMES)),
            "meteor_class_index": int(meta.get("meteor_class_index", 0)),
            "metadata_path": meta.get("metadata_path", model_catalog.metadata_path_for_model(target_path)),
            "architecture": architecture,
            "decision_threshold": float(
                meta.get("decision_threshold", config.METEOR_PROBABILITY_THRESHOLD)
            ),
            "target_recall": meta.get("target_recall"),
        }
        config.MODEL_PATH = target_path
        print(f"Loaded model: {target_path}")
        print(f"Normalization mean/std: {_active_model_info['mean']} / {_active_model_info['std']}")
        print(f"Resize policy: {_active_model_info['input_resize']}")
        print(f"Architecture: {architecture}")
        return True, "ok"
    except Exception as e:
        _model_ready = False
        return False, str(e)


def reload_model(model_path: Optional[str] = None, metadata: Optional[Dict] = None):
    """Atomically replace the classifier while inference is quiescent."""
    with _model_lock:
        return _reload_model_unlocked(model_path=model_path, metadata=metadata)


def get_active_model_info() -> Dict:
    return dict(_active_model_info)


def get_meteor_decision_threshold() -> float:
    """Return the calibrated threshold for the active classifier.

    Universal event models store an operating threshold selected for their
    validation recall target.  Older image-only models do not have that
    calibration and continue to use the GUI/configuration threshold.
    """
    fallback = float(getattr(config, "METEOR_PROBABILITY_THRESHOLD", 0.5))
    if _active_model_info.get("architecture") != UNIVERSAL_ARCHITECTURE_NAME:
        return fallback
    try:
        threshold = float(_active_model_info.get("decision_threshold", fallback))
    except (TypeError, ValueError):
        return fallback
    return min(1.0, max(0.0, threshold))


def predict_meteor_probability(image_path: str) -> float:
    global model, transform
    # MPS command submission and the mutable global model are deliberately
    # serialized.  Archive workers still decode and detect in parallel; only
    # the short classifier step enters this critical section.
    with _model_lock:
        if not _model_ready or model is None or transform is None:
            print("Warning: prediction requested before model is ready.")
            return 0.0
        try:
            with Image.open(image_path) as source_image:
                pil_image = source_image.convert("RGB")
            transpose = getattr(Image, "Transpose", Image)
            variants = [
                pil_image,
                pil_image.transpose(transpose.FLIP_LEFT_RIGHT),
                pil_image.transpose(transpose.FLIP_TOP_BOTTOM),
            ]
            # Submit one batch instead of three independent MPS command graphs.
            image_batch = torch.stack([transform(image) for image in variants]).to(device)
            with torch.inference_mode():
                avg_probabilities = torch.softmax(model(image_batch), dim=1).mean(dim=0)
            meteor_idx = int(_active_model_info.get("meteor_class_index", 0))
            if meteor_idx < 0 or meteor_idx >= avg_probabilities.numel():
                meteor_idx = 0
            return float(avg_probabilities[meteor_idx].item())
        except FileNotFoundError:
            print(f"Error: image not found: {image_path}")
            return 0.0
        except Exception as e:
            print(f"Prediction error ({image_path}): {e}")
            return 0.0


def predict_meteor_event(
    frames: Sequence,
    cutout_rect,
    detected_line,
    frame_rate: float,
    fallback_image_path: Optional[str] = None,
) -> float:
    """Classify a complete event when supported, with legacy-image fallback."""
    with _model_lock:
        if (
            _active_model_info.get("architecture") != UNIVERSAL_ARCHITECTURE_NAME
            or not _model_ready
            or model is None
        ):
            if fallback_image_path:
                return predict_meteor_probability(fallback_image_path)
            return 0.0
        try:
            image, kymograph, features = build_universal_inputs(
                frames=frames,
                rect=cutout_rect,
                detected_line=detected_line,
                frame_rate=frame_rate,
            )
            image_tensor = torch.from_numpy(image).unsqueeze(0).to(device)
            kymograph_tensor = torch.from_numpy(kymograph).unsqueeze(0).to(device)
            feature_tensor = torch.from_numpy(features).unsqueeze(0).to(device)
            with torch.inference_mode():
                logit = model(image_tensor, kymograph_tensor, feature_tensor)
                return float(torch.sigmoid(logit)[0].item())
        except Exception as exc:
            print(f"Universal event prediction error: {exc}")
            if fallback_image_path:
                return predict_meteor_probability(fallback_image_path)
            return 0.0


def _should_eager_load() -> bool:
    """Only the owner process may initialize an accelerator-backed model."""
    try:
        return multiprocessing.current_process().name == "MainProcess"
    except Exception:
        return True


if _should_eager_load():
    _ok, _msg = reload_model(config.MODEL_PATH)
    if not _ok:
        print(f"Warning: failed to load initial model '{config.MODEL_PATH}': {_msg}")


if __name__ == "__main__":
    print("model.py self-check")
    print(f"Model ready: {_model_ready}")
    print(f"Device: {device}")
    print(f"Active model info: {get_active_model_info()}")
