import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


DEFAULT_MEAN = [0.033, 0.033, 0.033]
DEFAULT_STD = [0.047, 0.047, 0.047]
DEFAULT_INPUT_RESIZE = [224, 224]
DEFAULT_CLASS_NAMES = ["meteor", "not_meteor"]


def metadata_path_for_model(model_path: str) -> str:
    return f"{model_path}.meta.json"


def _to_float_triplet(value, fallback: Sequence[float]) -> List[float]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            return [float(value[0]), float(value[1]), float(value[2])]
        except (TypeError, ValueError):
            pass
    return [float(fallback[0]), float(fallback[1]), float(fallback[2])]


def _normalize_resize(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            h = int(value[0])
            w = int(value[1])
            if h > 0 and w > 0:
                return [h, w]
        except (TypeError, ValueError):
            pass
    return [int(DEFAULT_INPUT_RESIZE[0]), int(DEFAULT_INPUT_RESIZE[1])]


def load_model_metadata(model_path: str) -> Dict:
    meta_path = metadata_path_for_model(model_path)
    payload = {}
    if Path(meta_path).exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                payload = loaded
        except Exception:
            payload = {}

    class_names = payload.get("class_names")
    if not isinstance(class_names, list) or not class_names:
        class_names = list(DEFAULT_CLASS_NAMES)
    else:
        class_names = [str(x) for x in class_names]

    meteor_index = payload.get("meteor_class_index")
    try:
        meteor_index = int(meteor_index)
    except (TypeError, ValueError):
        meteor_index = 0
        for idx, name in enumerate(class_names):
            if str(name).strip().lower() == "meteor":
                meteor_index = idx
                break

    if meteor_index < 0 or meteor_index >= len(class_names):
        meteor_index = 0

    normalized = {
        "model_path": str(model_path),
        "metadata_path": meta_path,
        "mean": _to_float_triplet(payload.get("mean"), DEFAULT_MEAN),
        "std": _to_float_triplet(payload.get("std"), DEFAULT_STD),
        "input_resize": _normalize_resize(payload.get("input_resize", DEFAULT_INPUT_RESIZE)),
        "class_names": class_names,
        "meteor_class_index": meteor_index,
    }
    # Architecture-specific fields are intentionally preserved so the runtime
    # can load newer event models without weakening legacy model metadata.
    for key in (
        "architecture",
        "preprocess_version",
        "decision_threshold",
        "target_recall",
        "kymograph_size",
        "feature_count",
        "validation_metrics",
        "camera_policy",
    ):
        if key in payload:
            normalized[key] = payload[key]
    return normalized


def save_model_metadata(
    model_path: str,
    mean: Sequence[float],
    std: Sequence[float],
    class_names: Optional[Sequence[str]] = None,
    input_resize: Optional[Sequence[int]] = None,
    meteor_class_index: Optional[int] = None,
    extra: Optional[Dict] = None,
) -> str:
    class_names = list(class_names) if class_names else list(DEFAULT_CLASS_NAMES)
    if meteor_class_index is None:
        meteor_class_index = 0
        for idx, name in enumerate(class_names):
            if str(name).strip().lower() == "meteor":
                meteor_class_index = idx
                break

    payload = {
        "mean": _to_float_triplet(mean, DEFAULT_MEAN),
        "std": _to_float_triplet(std, DEFAULT_STD),
        "class_names": [str(x) for x in class_names],
        "input_resize": _normalize_resize(input_resize),
        "meteor_class_index": int(meteor_class_index),
    }
    if extra:
        payload.update(extra)

    meta_path = metadata_path_for_model(model_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return meta_path


def discover_model_paths(
    search_dirs: Iterable[str],
    extra_paths: Optional[Iterable[str]] = None,
    recursive: bool = False,
) -> List[str]:
    found = []
    seen = set()

    for directory in search_dirs:
        if not directory:
            continue
        base = Path(directory)
        if not base.exists() or not base.is_dir():
            continue
        pattern = "**/*.pth" if recursive else "*.pth"
        for path in base.glob(pattern):
            try:
                resolved = str(path.resolve())
            except OSError:
                resolved = str(path)
            if resolved not in seen:
                seen.add(resolved)
                found.append(resolved)

    if extra_paths:
        for p in extra_paths:
            if not p:
                continue
            pp = Path(p)
            if not pp.exists() or not pp.is_file():
                continue
            try:
                resolved = str(pp.resolve())
            except OSError:
                resolved = str(pp)
            if resolved not in seen:
                seen.add(resolved)
                found.append(resolved)

    found.sort(key=lambda x: Path(x).name.lower())
    return found
