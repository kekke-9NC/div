"""Read the saved detection mask used by the legacy Tk UI, not SwiftUI."""

from pathlib import Path
import hashlib
import json
import os
import tempfile

import cv2
import numpy as np
from PIL import Image


def default_settings_path() -> Path:
    installed = Path.home() / "Library/Application Support/MeteorDetector/app_settings.json"
    return installed if installed.exists() else Path(__file__).resolve().parent.parent / "app_settings.json"


class LegacyDetectionMask:
    def __init__(self, settings_path: Path):
        self.settings_path = settings_path.expanduser().resolve()
        self.mask_path = self.settings_path.with_name("app_masks.npz")
        self.mask = None
        self.revision = ""

    def refresh(self) -> bool:
        # Read on each poll: both the selected path and enable checkbox can
        # change. A failed/partial save raises before replacing the last mask.
        settings = json.loads(self.settings_path.read_text(encoding="utf-8"))
        if not isinstance(settings, dict):
            raise ValueError("旧UI設定はJSONオブジェクトである必要があります")
        configured = settings.get("detection_mask_path")
        if configured is not None and not isinstance(configured, str):
            raise ValueError("検出マスクの保存先が不正です")
        path = Path(configured).expanduser() if configured else self.settings_path.with_name("app_masks.npz")
        if not path.is_absolute():
            path = self.settings_path.parent / path
        mask = None
        if settings.get("apply_mask", False) and settings.get("has_mask_image", True):
            with np.load(path, allow_pickle=False) as archive:
                if "mask_image" in archive:
                    raw = archive["mask_image"]
                    if raw.ndim != 2 or not raw.size or not np.issubdtype(raw.dtype, np.number):
                        raise ValueError("検出マスクは空でない2次元の数値配列である必要があります")
                    # Legacy live preview blacks out zero pixels and retains
                    # every positive pixel. Use nearest neighbour for resizing.
                    mask = np.where(raw > 0, 255, 0).astype(np.uint8)
        revision = hashlib.sha256(
            str(path.resolve()).encode() +
            (b"disabled" if mask is None else str(mask.shape).encode() + mask.tobytes())
        ).hexdigest()
        changed = revision != self.revision
        self.mask, self.mask_path, self.revision = mask, path.resolve(), revision
        return changed

    def resized(self, width: int, height: int) -> np.ndarray:
        if self.mask is None:
            return np.full((height, width), 255, dtype=np.uint8)
        return cv2.resize(self.mask, (width, height), interpolation=cv2.INTER_NEAREST)

    def write_png(self, output: Path, width: int | None = None, height: int | None = None) -> None:
        if width is None or height is None:
            height, width = self.mask.shape if self.mask is not None else (1080, 1920)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".png", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            Image.fromarray(self.resized(width, height)).convert("RGB").save(temporary)
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)

    def write_state(self, output: Path) -> None:
        data = dict(revision=self.revision, enabled=self.mask is not None,
                    settings_path=str(self.settings_path), mask_path=str(self.mask_path))
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, output)
