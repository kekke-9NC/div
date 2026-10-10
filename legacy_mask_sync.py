"""Persist legacy UI mask changes without rewriting unrelated settings."""

from pathlib import Path
import json
import os
import tempfile

import numpy as np


def persist_detection_mask(settings_file, masks_file, mask_image, plate_solve_mask_image, enabled):
    settings_path = Path(settings_file)
    masks_path = Path(masks_file)
    settings = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.exists() else {}
    arrays = {}
    if mask_image is not None:
        arrays["mask_image"] = mask_image
    if plate_solve_mask_image is not None:
        arrays["plate_solve_mask_image"] = plate_solve_mask_image
    masks_path.parent.mkdir(parents=True, exist_ok=True)
    # Write the archive first. Readers retain the previous mask if a save is
    # briefly incomplete; atomic replacement also avoids partial ZIP reads.
    with tempfile.NamedTemporaryFile(dir=masks_path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, masks_path)
    finally:
        temporary.unlink(missing_ok=True)
    settings.update(
        apply_mask=bool(enabled),
        has_mask_image=mask_image is not None,
        has_plate_solve_mask_image=plate_solve_mask_image is not None,
        detection_mask_path=str(masks_path.resolve()),
    )
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=settings_path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(settings, handle, ensure_ascii=False, indent=4)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, settings_path)
    finally:
        temporary.unlink(missing_ok=True)
