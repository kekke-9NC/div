"""Translate legacy saved settings into the non-Tk processing runtime.

The JSON settings file is shared by the two interfaces.  This module keeps
the conversion at the process boundary, so individual workers receive the
same options that the legacy ProcessingMixin constructs before a run.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping


# Saved names, config names, type, and the range shown by the legacy controls.
_ADVANCED_FIELDS = {
    "min_line_length": ("MIN_LINE_LENGTH", int, 1, 100),
    "border_size": ("BORDER_SIZE", int, 0, 100),
    "duplicate_thresh": ("DUPLICATE_DETECTION_THRESHOLD", int, 10, 500),
    "meteor_prob": ("METEOR_PROBABILITY_THRESHOLD", float, 0.1, 1.0),
    "finer_window_sec": ("FINER_DETECT_WINDOW_SECONDS", float, 1.0, 10.0),
    "finer_comp_step": ("FINER_COMPOSITE_STEP", int, 1, 20),
    "finer_min_length": ("FINER_DETECT_MIN_LENGTH", int, 5, 50),
    "finer_padding_sec": ("FINER_DETECT_PADDING_SECONDS", float, 0.1, 2.0),
    "finer_cutout_size": ("FINER_CUTOUT_SIZE", int, 128, 1024),
    "airplane_dur_thresh": ("AIRPLANE_DURATION_THRESHOLD", int, 1, 120),
    "airplane_frame_thresh": ("AIRPLANE_FRAME_THRESHOLD", int, 1, 120),
    "tracking_dist_thresh": ("TRACKING_DISTANCE_THRESHOLD", int, 1, 2000),
    "max_clip_dur": ("MAX_CLIP_DURATION", float, 1, 30),
    "clip_dur_sec": ("CLIP_DURATION_SECONDS", float, 1, 30),
    "cutout_size": ("CUTOUT_SIZE", int, 128, 1023),
    "rtsp_scale_lower": ("RTSP_SCALE_LOWER", float, 10, 180),
    "rtsp_scale_upper": ("RTSP_SCALE_UPPER", float, 10, 180),
}

_QUALITIES = {
    "入力品質基準（推奨）": "source",
    "最高品質": "maximum",
    "高品質": "high",
    "標準": "standard",
    "容量優先": "compact",
    "カスタム": "custom",
    "可逆圧縮（低速）": "lossless",
}


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name}はオブジェクト形式で指定してください")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name}は真偽値で指定してください")
    return value


def _number(value: Any, name: str, lower: float, upper: float, integer: bool = False) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{name}は数値で指定してください")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}は数値で指定してください") from exc
    if not math.isfinite(parsed) or not lower <= parsed <= upper:
        raise ValueError(f"{name}は{lower:g}〜{upper:g}の範囲で指定してください")
    return int(parsed) if integer else parsed


def _text(value: Any, name: str, *, maximum: int = 4096) -> str:
    if not isinstance(value, str) or len(value) > maximum:
        raise ValueError(f"{name}は文字列で指定してください")
    return value.strip()


@dataclass(frozen=True)
class RuntimeSettings:
    advanced_config: Dict[str, Any]
    save_option_extras: Dict[str, Any]
    encoding: Dict[str, Any]
    ai_backend: str
    lm_studio_url: str
    lm_studio_model_id: str
    lm_studio_api_key: str
    use_plate_solve: bool
    plate_solve_mode: str
    auto_time_updater_enabled: bool

    @contextmanager
    def applied_config(self, config: Any) -> Iterator[None]:
        """Keep module globals in sync with a single serialized worker run."""
        missing = object()
        previous = {name: getattr(config, name, missing) for name in self.advanced_config}
        for name, value in self.advanced_config.items():
            setattr(config, name, value)
        try:
            yield
        finally:
            for name, value in previous.items():
                if value is missing:
                    delattr(config, name)
                else:
                    setattr(config, name, value)

    def configure_ai_backend(self) -> None:
        # Import lazily.  The bridge must answer ping/settings without loading
        # the computer-vision stack, and this module never imports Tkinter.
        import bright_area_detector

        bright_area_detector.configure_ai_backend(
            backend=self.ai_backend,
            lm_studio_url=self.lm_studio_url,
            lm_studio_model_id=self.lm_studio_model_id,
            lm_studio_api_key=self.lm_studio_api_key,
        )


def from_legacy_settings(raw: Any, root: Path) -> RuntimeSettings:
    """Validate known runtime options while preserving unknown JSON elsewhere."""
    settings = _mapping(raw, "legacySettings")
    advanced = _mapping(settings.get("advanced_settings", {}), "advanced_settings")
    import config

    advanced_config: Dict[str, Any] = {}
    for key, (config_name, kind, lower, upper) in _ADVANCED_FIELDS.items():
        if key in advanced:
            advanced_config[config_name] = _number(
                advanced[key], f"advanced_settings.{key}", lower, upper, kind is int
            )
    lower = advanced_config.get("RTSP_SCALE_LOWER", getattr(config, "RTSP_SCALE_LOWER", 85.0))
    upper = advanced_config.get("RTSP_SCALE_UPPER", getattr(config, "RTSP_SCALE_UPPER", 100.0))
    if lower >= upper:
        raise ValueError("advanced_settingsの視野角LowerはUpperより小さくしてください")

    timestamp = _mapping(settings.get("full_video_timestamp", {}), "full_video_timestamp")
    timestamp_enabled = _boolean(
        timestamp.get("enabled", getattr(config, "FULL_VIDEO_TIMESTAMP_ENABLED", True)),
        "full_video_timestamp.enabled",
    )
    timestamp_position = _text(
        timestamp.get("position", "右下"), "full_video_timestamp.position", maximum=32
    )
    if timestamp_position not in {"右下", "左下", "右上", "左上"}:
        raise ValueError("full_video_timestamp.positionの指定が不正です")
    timestamp_size = _number(
        timestamp.get("size_percent", getattr(config, "FULL_VIDEO_TIMESTAMP_SIZE_PERCENT", 1.8)),
        "full_video_timestamp.size_percent", 0.8, 4.0,
    )

    ml_enabled = _boolean(
        settings.get("ml_training_export_enabled", getattr(config, "DEFAULT_ML_TRAINING_EXPORT_ENABLED", False)),
        "ml_training_export_enabled",
    )
    ml_root = _text(
        settings.get("ml_training_data_root", getattr(config, "DEFAULT_ML_TRAINING_DATA_ROOT", str(root / "ml_training_data"))),
        "ml_training_data_root",
    )
    if ml_enabled and not ml_root:
        raise ValueError("機械学習向けデータの保存先が空です")
    ml_boundary_fraction = (
        _number(
            settings.get(
                "ml_training_boundary_fraction",
                getattr(config, "DEFAULT_ML_TRAINING_BOUNDARY_FRACTION", 0.30),
            ),
            "ml_training_boundary_fraction", 0.0, 1.0,
        )
        if ml_enabled
        else getattr(config, "DEFAULT_ML_TRAINING_BOUNDARY_FRACTION", 0.30)
    )
    auto_mask = _boolean(
        settings.get("auto_video_mask_enabled", getattr(config, "DEFAULT_AUTO_VIDEO_MASK_ENABLED", True)),
        "auto_video_mask_enabled",
    )
    save_extras = {
        "full_video_timestamp_enabled": timestamp_enabled,
        "full_video_timestamp_position": timestamp_position,
        "full_video_timestamp_size_percent": timestamp_size,
        "ml_training_export_enabled": ml_enabled,
        "ml_training_data_root": str(Path(ml_root).expanduser().resolve()) if ml_root else "",
        "ml_training_boundary_fraction": ml_boundary_fraction,
        "auto_video_mask_enabled": auto_mask,
        "auto_video_mask_cache_dir": str(root / "auto_video_masks"),
    }

    encoding = _mapping(settings.get("processed_video_encoding", {}), "processed_video_encoding")
    codec_label = _text(
        encoding.get("codec", "H.265 / HEVC (推奨)"), "processed_video_encoding.codec", maximum=64
    )
    codecs = {"H.265 / HEVC (推奨)": "hevc", "H.264 / AVC": "h264", "MPEG-4 (互換)": "mpeg4"}
    if codec_label not in codecs:
        raise ValueError("processed_video_encoding.codecの指定が不正です")
    quality_label = _text(
        encoding.get("quality", "入力品質基準（推奨）"), "processed_video_encoding.quality", maximum=64
    )
    if quality_label not in _QUALITIES:
        raise ValueError("processed_video_encoding.qualityの指定が不正です")
    bitrate = _number(
        encoding.get("bitrate_mbps", "40"), "processed_video_encoding.bitrate_mbps", 5, 200, True
    )
    if _QUALITIES[quality_label] == "lossless" and codecs[codec_label] == "mpeg4":
        raise ValueError("可逆圧縮ではH.265またはH.264を選択してください")
    normalized_encoding = {
        "codec": codecs[codec_label],
        "quality": _QUALITIES[quality_label],
        "bitrate_mbps": bitrate,
    }

    ai_backend = _text(
        settings.get("ai_vlm_backend", getattr(config, "DEFAULT_AI_VLM_BACKEND", "lmstudio_qwen3_5_2b")),
        "ai_vlm_backend", maximum=64,
    )
    if ai_backend not in {
        getattr(config, "AI_VLM_BACKEND_LOCAL_QWEN3_VL_4B", "local_qwen3_vl_4b"),
        getattr(config, "AI_VLM_BACKEND_LM_STUDIO_QWEN35_2B", "lmstudio_qwen3_5_2b"),
    }:
        raise ValueError("ai_vlm_backendの指定が不正です")
    lm_url = _text(
        settings.get("lm_studio_vlm_url", getattr(config, "DEFAULT_LM_STUDIO_VLM_URL", "http://localhost:1234/v1")),
        "lm_studio_vlm_url", maximum=2048,
    )
    lm_model = _text(
        settings.get("lm_studio_vlm_model_id", getattr(config, "DEFAULT_LM_STUDIO_VLM_MODEL_ID", "qwen/qwen3-vl-4b")),
        "lm_studio_vlm_model_id", maximum=256,
    )
    lm_key = _text(
        settings.get("lm_studio_vlm_api_key", ""),
        "lm_studio_vlm_api_key", maximum=2048,
    )
    astrometry_key = _text(
        settings.get("astrometry_api_key", ""),
        "astrometry_api_key", maximum=2048,
    )
    if astrometry_key:
        advanced_config["ASTROMETRY_API_KEY"] = astrometry_key
    use_plate_solve = _boolean(settings.get("use_plate_solve", False), "use_plate_solve")
    plate_solve_mode = _text(settings.get("plate_solve_mode", "local"), "plate_solve_mode", maximum=16)
    if plate_solve_mode not in {"local", "api"}:
        raise ValueError("plate_solve_modeの指定が不正です")
    auto_time = _boolean(
        settings.get("auto_time_updater_enabled", False), "auto_time_updater_enabled"
    )
    return RuntimeSettings(
        advanced_config=advanced_config,
        save_option_extras=save_extras,
        encoding=normalized_encoding,
        ai_backend=ai_backend,
        lm_studio_url=lm_url,
        lm_studio_model_id=lm_model,
        lm_studio_api_key=lm_key,
        use_plate_solve=use_plate_solve,
        plate_solve_mode=plate_solve_mode,
        auto_time_updater_enabled=auto_time,
    )
