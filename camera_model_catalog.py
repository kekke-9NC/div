"""Discovery and presentation helpers for registered fixed-camera models.

The GUI uses this module to present astrometric models as human-readable
choices instead of exposing cache paths or a Finder file picker.
"""

from __future__ import annotations

from datetime import datetime
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


MODEL_TYPE = "fixed-camera-stg-poly"


def _default_cache_root() -> Path:
    configured = os.environ.get("METEOR_ASTROMETRY_CACHE")
    if configured:
        return Path(configured).expanduser().resolve()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "MeteorDetector" / "astrometry"
    return Path.home() / ".cache" / "meteor_detector" / "astrometry"


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if result == result else default


def _date_label(value: Any) -> str:
    """Render compact model dates in a form a person can scan quickly."""
    text = str(value).strip()
    if re.fullmatch(r"\d{8}", text):
        return f"{text[:4]}/{text[4:6]}/{text[6:]}"
    return text[:10].replace("-", "/") if text else ""


def _reference_night(metadata: Dict[str, Any]) -> str:
    valid_dates = metadata.get("valid_dates") or metadata.get("training_dates") or []
    if isinstance(valid_dates, str):
        valid_dates = [valid_dates]
    dates = [_date_label(item) for item in valid_dates if str(item).strip()]
    if dates:
        return ", ".join(dates)
    reference = str(metadata.get("reference_datetime", "")).strip()
    if reference:
        return _date_label(reference[:10])
    return "不明"


def _reference_datetime(metadata: Dict[str, Any]) -> str:
    value = str(metadata.get("reference_datetime", "")).strip()
    if not value:
        return "不明"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return value.replace("T", " ")[:19]


def _support_fraction(metadata: Dict[str, Any]) -> float:
    explicit = metadata.get("support_fraction", metadata.get("sip_support_fraction"))
    if explicit is not None:
        value = _as_float(explicit)
        return value / 100.0 if value > 1.0 else max(0.0, min(1.0, value))
    grid = metadata.get("support_grid") or metadata.get("support_grid_validation")
    if isinstance(grid, dict):
        grid = grid.get("grid") or grid.get("values")
    try:
        values = [bool(cell) for row in grid for cell in row]
    except (TypeError, ValueError):
        return 0.0
    return sum(values) / len(values) if values else 0.0


def _quality_p95(metadata: Dict[str, Any]) -> Optional[float]:
    fit_stats = metadata.get("fit_stats") or {}
    for key in ("residual_p95_px", "sip_residual_p95_px"):
        value = fit_stats.get(key) if key in fit_stats else metadata.get(key)
        if value is not None:
            parsed = _as_float(value, float("nan"))
            if parsed == parsed:
                return parsed
    return None


def _display_name(metadata: Dict[str, Any], path: Path) -> str:
    label = str(metadata.get("model_label") or path.parent.name).strip()
    width = int(metadata.get("width", 0) or 0)
    height = int(metadata.get("height", 0) or 0)
    size = f"{width}×{height}" if width and height else "解像度不明"
    coverage = f"{_support_fraction(metadata) * 100:.0f}%"
    night = _reference_night(metadata)
    state = "使用可" if metadata.get("enabled", False) else "候補・未適用"
    return f"{label}  /  被覆率 {coverage}  /  基準夜 {night}  /  {size}  /  {state}"


def _selection_kind(metadata: Dict[str, Any], path: Path) -> str:
    """Return a short Japanese label suitable for a dropdown row."""
    label = str(metadata.get("model_label") or path.parent.name).strip().upper()
    if "GAIA" in label:
        return "星の動き＋Gaia"
    if "STAR-TRAJECTORY" in label or "TRAJECTORY" in label:
        return "星の動きモデル"
    if "MOSAIC" in label or "STABLE REGION" in label:
        return "固定カメラ（安定領域）"
    if "TRIPOD" in label:
        return "固定カメラ（検証済み）"
    if "AUTO FIXED" in label:
        return "固定カメラ（自動）"
    if "FIXED CAMERA" in label or "STG+POLY" in label:
        return "固定カメラモデル"
    return "カメラ補正モデル"


def _selection_name(metadata: Dict[str, Any], path: Path) -> str:
    """Build a compact, scannable choice for the model combobox."""
    coverage = f"{_support_fraction(metadata) * 100:.0f}%"
    night = _reference_night(metadata)
    state = "使用可" if metadata.get("enabled", False) else "候補・未適用"
    return f"{_selection_kind(metadata, path)}  |  {coverage}  |  {night}  |  {state}"


def _read_model(path: Path) -> Optional[Dict[str, Any]]:
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    if metadata.get("model_type") != MODEL_TYPE:
        return None
    wcs_path = str(metadata.get("wcs_path", "")).strip()
    if wcs_path:
        resolved_wcs = Path(wcs_path).expanduser()
        if not resolved_wcs.is_absolute():
            resolved_wcs = path.parent / resolved_wcs
        if not resolved_wcs.exists():
            return None
    support = _support_fraction(metadata)
    quality = _quality_p95(metadata)
    return {
        "path": str(path.resolve()),
        "metadata": metadata,
        "display_name": _display_name(metadata, path),
        "selection_name": _selection_name(metadata, path),
        "model_label": str(metadata.get("model_label") or path.parent.name),
        "width": int(metadata.get("width", 0) or 0),
        "height": int(metadata.get("height", 0) or 0),
        "support_fraction": support,
        "support_percent": support * 100.0,
        "reference_night": _reference_night(metadata),
        "reference_datetime": _reference_datetime(metadata),
        "valid_dates": list(metadata.get("valid_dates") or []),
        "quality_p95_px": quality,
        "enabled": bool(metadata.get("enabled", False)),
        "target_met": bool(metadata.get("target_met", metadata.get("coverage_target_met", False))),
    }


def discover_camera_models(cache_root: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return registered fixed-camera models in UI-ready form.

    Invalid or stale entries are skipped so the model selector never presents
    a choice that cannot be loaded.
    """
    root = Path(cache_root).expanduser().resolve() if cache_root else _default_cache_root()
    models: List[Dict[str, Any]] = []
    for path in sorted((root / "camera_models").glob("*/camera_model.json")):
        item = _read_model(path)
        if item is not None:
            models.append(item)
    models.sort(
        key=lambda item: (
            not item["enabled"],
            -item["support_fraction"],
            item["reference_night"],
            item["display_name"],
        )
    )
    # The compact label is used only in selectors. Keep display_name for
    # diagnostics and make duplicate compact labels unambiguous.
    seen: Dict[str, int] = {}
    for item in models:
        choice = str(item.get("selection_name") or item["display_name"])
        seen[choice] = seen.get(choice, 0) + 1
        if seen[choice] > 1:
            item["selection_name"] = f"{choice}  #{seen[choice]}"
    return models


def format_model_details(model: Optional[Dict[str, Any]]) -> str:
    """Build a compact detail string for the model selector."""
    if not model:
        return "未選択（撮影日に合うカメラ補正データを自動選択）"
    quality = model.get("quality_p95_px")
    quality_text = f" / p95誤差 {quality:.2f}px" if quality is not None else ""
    dates = ", ".join(str(value) for value in model.get("valid_dates", []) if str(value).strip())
    valid_text = f" / 適用夜 {dates}" if dates and dates != model.get("reference_night") else ""
    metadata = model.get("metadata") or {}
    selection = metadata.get("selection_summary") or {}
    if selection.get("mode") == "automatic":
        count = selection.get("selected_video_count", "-")
        selection_text = f" / 学習動画 {count}本（夜間を自動選択）"
    elif selection.get("mode") == "manual":
        selection_text = " / 学習動画 手動指定"
    else:
        selection_text = ""
    state = "使用可" if model.get("enabled") else "候補・未適用"
    return (
        f"種類: {model['model_label']}  /  状態: {state}\n"
        f"解像度: {model['width']}×{model['height']}  /  画面被覆率: {model['support_percent']:.0f}%  /  "
        f"基準夜: {model['reference_night']}\n"
        f"投影基準時刻: {model['reference_datetime']}{selection_text}{valid_text}{quality_text}"
    )
