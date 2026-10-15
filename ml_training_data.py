"""Export and review detector events as a training-oriented dataset."""

from __future__ import annotations

import json
import bisect
import math
import os
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from temporal_statistics import temporal_median
import ml_collection_policy


SCHEMA_VERSION = 1
_log_lock = threading.Lock()
_history_cache_lock = threading.Lock()
_history_cache = {}
_media_names_cache = {}
_sync_skip_cache = {}


def _file_identity(path):
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _matching_media(source_dir, prefix):
    """Index directory names; only stat names belonging to this event."""
    identity = _file_identity(source_dir)
    cached = _media_names_cache.get(source_dir)
    if cached is None or cached[0] != identity:
        with os.scandir(source_dir) as entries:
            names = sorted(entry.name for entry in entries)
        if _file_identity(source_dir) == identity:
            if len(_media_names_cache) >= 16:
                _media_names_cache.clear()
            _media_names_cache[source_dir] = (identity, names)
    else:
        names = cached[1]
    candidates = set()
    exact = bisect.bisect_left(names, prefix)
    if exact < len(names) and names[exact] == prefix:
        candidates.add(prefix)
    for start in (prefix + "_", prefix + "."):
        index = bisect.bisect_left(names, start)
        while index < len(names) and names[index].startswith(start):
            name = names[index]
            if start.endswith("_") or Path(name).stem == prefix:
                candidates.add(name)
            index += 1
    return [source_dir / name for name in sorted(candidates)
            if (source_dir / name).is_file()]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _extract_square(frame: np.ndarray, rect: Sequence[int], size: int) -> np.ndarray:
    x1, y1, x2, y2 = (int(v) for v in rect)
    crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if crop.size == 0:
        return np.zeros((size, size), dtype=np.uint8)
    if crop.shape[:2] != (size, size):
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return crop


def build_temporal_representations(
    frames: Sequence[np.ndarray], rect: Sequence[int], size: int
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """Return grayscale crops, early/middle/late RGB response, and time-of-peak map."""
    crops = [_gray(_extract_square(frame, rect, size)) for frame in frames]
    if not crops:
        blank = np.zeros((size, size), dtype=np.uint8)
        return [], cv2.cvtColor(blank, cv2.COLOR_GRAY2BGR), blank

    stack = np.stack(crops).astype(np.int16)
    background = temporal_median(stack).astype(np.int16)
    response = np.abs(stack - background).astype(np.uint8)

    groups = np.array_split(np.arange(len(crops)), 3)
    temporal_channels = []
    for group in groups:
        if len(group):
            temporal_channels.append(response[group].max(axis=0))
        else:
            temporal_channels.append(np.zeros((size, size), dtype=np.uint8))
    # OpenCV stores BGR. The semantic channel order is early, middle, late.
    temporal_bgr = cv2.merge(
        [temporal_channels[2], temporal_channels[1], temporal_channels[0]]
    )

    peak_strength = response.max(axis=0)
    peak_index = response.argmax(axis=0)
    denom = max(1, len(crops) - 1)
    time_of_peak = np.round(peak_index * 255.0 / denom).astype(np.uint8)
    # Background-only pixels have no meaningful event time.
    time_of_peak[peak_strength < 3] = 0
    return crops, temporal_bgr, time_of_peak


def _write_clip(path: Path, crops: Sequence[np.ndarray], fps: float) -> None:
    if not crops:
        return
    height, width = crops[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, float(fps)), (width, height)
    )
    if not writer.isOpened():
        raise IOError(f"Could not open training clip writer: {path}")
    try:
        for crop in crops:
            writer.write(cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR))
    finally:
        writer.release()


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
    with _log_lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())


def export_training_event(
    *,
    root_dir: str,
    predicted_label: str,
    probability: float,
    event_id: str,
    source: str,
    detection_time: datetime,
    frames: Sequence[np.ndarray],
    classification_diff: np.ndarray,
    cutout_rect: Sequence[int],
    cutout_size: int,
    frame_rate: float,
    detected_line: Sequence[Sequence[int]],
    frame_range: Sequence[int],
    media_directories: Optional[Dict[str, str]] = None,
    minimum_probability: float = 0.15,
    decision_threshold: Optional[float] = None,
    boundary_fraction: float = ml_collection_policy.DEFAULT_BOUNDARY_FRACTION,
    model_info: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Export a review candidate; use the legacy floor only without a model threshold."""
    if isinstance(minimum_probability, bool):
        raise ValueError("minimum_probability must be between 0 and 1")
    try:
        minimum_probability = float(minimum_probability)
        probability = float(probability)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("probability and minimum_probability must be numeric") from exc
    if not math.isfinite(minimum_probability) or not 0.0 <= minimum_probability <= 1.0:
        raise ValueError("minimum_probability must be between 0 and 1")
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1")
    collection = None
    root = Path(root_dir).expanduser().resolve()
    if decision_threshold is not None:
        collection = ml_collection_policy.collection_decision(
            probability=probability, decision_threshold=decision_threshold,
            boundary_fraction=boundary_fraction, source=source,
            detection_time=detection_time, frame_range=frame_range, cutout_rect=cutout_rect,
        )
        if collection is None:
            return None
    elif probability < minimum_probability:
        return None

    reserved = collection is not None and collection["reason"] == "low_score_sample"
    if reserved and not ml_collection_policy.reserve_low_score(
        root, collection["observation_night"], collection["event_key"],
    ):
        return None
    try:
        return _write_training_event(
            root=root, predicted_label=predicted_label, probability=probability,
            event_id=event_id, source=source, detection_time=detection_time,
            frames=frames, classification_diff=classification_diff,
            cutout_rect=cutout_rect, cutout_size=cutout_size, frame_rate=frame_rate,
            detected_line=detected_line, frame_range=frame_range,
            media_directories=media_directories,
            minimum_probability=minimum_probability if collection is None else None,
            collection=collection, model_info=model_info,
        )
    except Exception:
        if reserved:
            ml_collection_policy.release_low_score(root, collection["observation_night"], collection["event_key"])
        raise


def _write_training_event(*, root, predicted_label, probability, event_id, source,
                          detection_time, frames, classification_diff, cutout_rect,
                          cutout_size, frame_rate, detected_line, frame_range,
                          media_directories, minimum_probability, collection, model_info):

    label = "meteor" if predicted_label == "meteor" else "not_meteor"
    pending = root / "pending" / label
    pending.mkdir(parents=True, exist_ok=True)
    safe_event_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in event_id)
    final_dir = pending / safe_event_id
    if final_dir.exists():
        final_dir = pending / f"{safe_event_id}_{uuid.uuid4().hex[:8]}"
    temp_dir = pending / f".{final_dir.name}.tmp-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=False)
    renamed = False

    try:
        diff = _gray(_extract_square(classification_diff, cutout_rect, cutout_size))
        crops, temporal_bgr, time_of_peak = build_temporal_representations(
            frames, cutout_rect, cutout_size
        )
        if not cv2.imwrite(str(temp_dir / "diff.png"), diff):
            raise IOError("Could not write diff.png")
        if not cv2.imwrite(str(temp_dir / "temporal_rgb.png"), temporal_bgr):
            raise IOError("Could not write temporal_rgb.png")
        if not cv2.imwrite(str(temp_dir / "time_of_peak.png"), time_of_peak):
            raise IOError("Could not write time_of_peak.png")
        _write_clip(temp_dir / "clip.mp4", crops, frame_rate)

        metadata = {
            "schema_version": SCHEMA_VERSION,
            "event_id": final_dir.name,
            "review_status": "pending",
            "predicted_label": label,
            "meteor_probability": probability,
            "minimum_save_probability": minimum_probability,
            "collection": collection,
            "model": dict(model_info) if model_info else None,
            "source": str(source),
            "detection_time": detection_time.isoformat(),
            "created_at": _utc_now(),
            "frame_rate": float(frame_rate),
            "frame_count": len(crops),
            "frame_range": [int(v) for v in frame_range],
            "cutout_rect": [int(v) for v in cutout_rect],
            "cutout_size": int(cutout_size),
            "detected_line": [[int(v) for v in point] for point in detected_line],
            "detected_media": {
                "directories": {
                    label_name: str(Path(directory).expanduser().resolve())
                    for label_name, directory in (media_directories or {}).items()
                    if label_name in ("meteor", "not_meteor") and str(directory).strip()
                },
                "filename_prefix": safe_event_id,
            } if media_directories else None,
            "artifacts": {
                "diff": "diff.png",
                "temporal_rgb": "temporal_rgb.png",
                "time_of_peak": "time_of_peak.png",
                "clip": "clip.mp4",
            },
        }
        with (temp_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
        temp_dir.rename(final_dir)
        renamed = True
        _append_jsonl(root / "export_log.jsonl", {
            "action": "export", "timestamp": _utc_now(), "event_id": final_dir.name,
            "predicted_label": label, "path": str(final_dir),
        })
        return str(final_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        if renamed:
            shutil.rmtree(final_dir, ignore_errors=True)
        raise


def pending_events(root_dir: str, *, cancel: Optional[threading.Event] = None) -> List[Path]:
    root = Path(root_dir).expanduser().resolve()
    events: List[Path] = []
    for label in ("meteor", "not_meteor"):
        folder = root / "pending" / label
        if folder.exists():
            with os.scandir(folder) as entries:
                for entry in entries:
                    if cancel is not None and cancel.is_set():
                        raise InterruptedError("レビュー読み込みを中断しました")
                    if not entry.name.startswith(".") and entry.is_dir():
                        events.append(Path(entry.path))
    return sorted(events, key=lambda p: p.name)


def load_metadata(event_dir: Path) -> Dict[str, Any]:
    with (event_dir / "metadata.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _detected_media_metadata(
    media_directories: Dict[str, str], filename_prefix: str
) -> Dict[str, Any]:
    return {
        "directories": {
            label_name: str(Path(directory).expanduser().resolve())
            for label_name, directory in media_directories.items()
            if label_name in ("meteor", "not_meteor") and str(directory).strip()
        },
        "filename_prefix": filename_prefix,
    }


def _event_media_prefix(event_id: str) -> str:
    parts = str(event_id).rsplit("_", 1)
    if len(parts) == 2 and len(parts[1]) == 8 and all(
        character in "0123456789abcdefABCDEF" for character in parts[1]
    ):
        return parts[0]
    return str(event_id)


def _has_detected_media_metadata(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    directories = value.get("directories")
    if not isinstance(directories, dict) or not str(value.get("filename_prefix", "")).strip():
        return False
    return all(str(directories.get(label, "")).strip() for label in ("meteor", "not_meteor"))


def _available_media_destination(path: Path) -> Path:
    if not path.exists():
        return path
    candidate = path.with_name(f"{path.stem}_{uuid.uuid4().hex[:8]}{path.suffix}")
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{uuid.uuid4().hex[:8]}{path.suffix}")
    return candidate


def _apply_media_moves(
    moves: Iterable[Tuple[Path, Path]], *, rename_conflicts: bool
) -> List[Dict[str, str]]:
    """Move files as a group and roll back completed moves if one fails."""
    completed: List[Dict[str, str]] = []
    try:
        for source, destination in moves:
            if not source.is_file() or source.resolve() == destination.resolve():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if not rename_conflicts:
                    raise FileExistsError(str(destination))
                destination = _available_media_destination(destination)
            shutil.move(str(source), str(destination))
            completed.append({"from": str(source), "to": str(destination)})
    except Exception:
        rollback_errors = []
        for item in reversed(completed):
            moved = Path(item["to"])
            original = Path(item["from"])
            try:
                if moved.exists() and not original.exists():
                    original.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(moved), str(original))
            except Exception as exc:
                rollback_errors.append(str(exc))
        if rollback_errors:
            raise RuntimeError(
                f"メディア移動に失敗し、ロールバックにも失敗しました: {'; '.join(rollback_errors)}"
            )
        raise
    return completed


def _move_detected_media(
    metadata: Dict[str, Any], source_label: str, destination_label: str
) -> List[Dict[str, str]]:
    detected_media = metadata.get("detected_media")
    if not isinstance(detected_media, dict):
        return []
    directories = detected_media.get("directories")
    prefix = str(detected_media.get("filename_prefix", "")).strip()
    if not isinstance(directories, dict) or not prefix:
        return []
    source_dir = Path(str(directories.get(source_label, ""))).expanduser()
    destination_dir = Path(str(directories.get(destination_label, ""))).expanduser()
    if not str(directories.get(source_label, "")).strip() or not str(
        directories.get(destination_label, "")
    ).strip():
        return []
    if source_dir.resolve() == destination_dir.resolve() or not source_dir.is_dir():
        return []

    matching_files = _matching_media(source_dir, prefix)
    return _apply_media_moves(
        ((path, destination_dir / path.name) for path in sorted(matching_files)),
        rename_conflicts=True,
    )


def review_event(
    event_dir: Path,
    root_dir: str,
    reviewed_label: str,
    media_directories: Optional[Dict[str, str]] = None,
    *, batch_id: Optional[str] = None,
) -> Dict[str, Any]:
    label = "meteor" if reviewed_label == "meteor" else "not_meteor"
    root = Path(root_dir).expanduser().resolve()
    metadata = load_metadata(event_dir)
    predicted = metadata.get("predicted_label", event_dir.parent.name)
    predicted = "meteor" if predicted == "meteor" else "not_meteor"
    if media_directories and not _has_detected_media_metadata(metadata.get("detected_media")):
        metadata["detected_media"] = _detected_media_metadata(
            media_directories,
            _event_media_prefix(str(metadata.get("event_id") or event_dir.name)),
        )
    destination_parent = root / "reviewed" / label
    destination_parent.mkdir(parents=True, exist_ok=True)
    destination = destination_parent / event_dir.name
    if destination.exists():
        destination = destination_parent / f"{event_dir.name}_{uuid.uuid4().hex[:8]}"

    media_moves = _move_detected_media(metadata, predicted, label) if label != predicted else []
    original_metadata = dict(metadata)
    metadata.update({
        "review_status": "reviewed",
        "reviewed_label": label,
        "was_misclassified": label != predicted,
        "reviewed_at": _utc_now(),
        "detected_media_moves": media_moves,
    })
    record = {
        "action": "review", "timestamp": _utc_now(), "event_id": destination.name,
        "predicted_label": predicted, "reviewed_label": label,
        "was_misclassified": label != predicted,
        "from": str(event_dir), "to": str(destination),
        "media_moves": media_moves,
    }
    if batch_id:
        metadata["review_batch_id"] = batch_id
        record["batch_id"] = batch_id
    try:
        with (event_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
        shutil.move(str(event_dir), str(destination))
        _append_jsonl(root / "review_log.jsonl", record)
    except Exception:
        if destination.exists() and not event_dir.exists():
            shutil.move(str(destination), str(event_dir))
        if event_dir.exists():
            with (event_dir / "metadata.json").open("w", encoding="utf-8") as handle:
                json.dump(original_metadata, handle, ensure_ascii=False, indent=2)
        _apply_media_moves(
            ((Path(item["to"]), Path(item["from"])) for item in reversed(media_moves)),
            rename_conflicts=False,
        )
        raise
    return record


def sync_reviewed_media(root_dir: str, media_directories: Dict[str, str], *,
                        cancel: Optional[threading.Event] = None) -> int:
    """Move saved detection files for already-reviewed misclassifications."""
    root = Path(root_dir).expanduser().resolve()
    moved_count = 0
    if root not in _sync_skip_cache and len(_sync_skip_cache) >= 4:
        _sync_skip_cache.clear()
    skip_cache = _sync_skip_cache.setdefault(root, {})
    for reviewed_folder_label in ("meteor", "not_meteor"):
        reviewed_dir = root / "reviewed" / reviewed_folder_label
        if not reviewed_dir.is_dir():
            continue
        for event_dir in sorted(reviewed_dir.iterdir()):
            # Never interrupt a move/metadata/log transaction midway through.
            if cancel is not None and cancel.is_set():
                return moved_count
            if not event_dir.is_dir() or event_dir.name.startswith("."):
                continue
            marker = event_dir / "metadata.json"
            identity = _file_identity(marker)
            if skip_cache.get(marker) == identity:
                continue
            metadata = load_metadata(event_dir)
            if metadata.get("review_status") != "reviewed":
                skip_cache[marker] = identity
                continue
            predicted_raw = metadata.get("predicted_label")
            reviewed_raw = metadata.get("reviewed_label", reviewed_folder_label)
            if predicted_raw not in ("meteor", "not_meteor") or reviewed_raw not in (
                "meteor", "not_meteor"
            ):
                skip_cache[marker] = identity
                continue
            if predicted_raw == reviewed_raw or "detected_media_moves" in metadata:
                skip_cache[marker] = identity
                continue

            detected_media = metadata.get("detected_media")
            if not _has_detected_media_metadata(detected_media):
                detected_media = _detected_media_metadata(
                    media_directories,
                    _event_media_prefix(str(metadata.get("event_id") or event_dir.name)),
                )

            original_metadata = dict(metadata)
            metadata["detected_media"] = detected_media
            media_moves = _move_detected_media(metadata, predicted_raw, reviewed_raw)
            if not media_moves:
                continue
            metadata["detected_media_moves"] = media_moves
            try:
                with (event_dir / "metadata.json").open("w", encoding="utf-8") as handle:
                    json.dump(metadata, handle, ensure_ascii=False, indent=2)
                _append_jsonl(root / "review_log.jsonl", {
                    "action": "sync_reviewed_media",
                    "timestamp": _utc_now(),
                    "event_id": event_dir.name,
                    "review_timestamp": metadata.get("reviewed_at"),
                    "media_moves": media_moves,
                })
            except Exception:
                with (event_dir / "metadata.json").open("w", encoding="utf-8") as handle:
                    json.dump(original_metadata, handle, ensure_ascii=False, indent=2)
                _apply_media_moves(
                    ((Path(item["to"]), Path(item["from"])) for item in reversed(media_moves)),
                    rename_conflicts=False,
                )
                raise
            moved_count += len(media_moves)
            skip_cache[marker] = _file_identity(marker)
    return moved_count


def detection_date(event_dir: Path) -> Optional[str]:
    """Return the local capture date (YYYY-MM-DD) recorded for an event."""
    try:
        value = str(load_metadata(event_dir).get("detection_time", "")).strip()
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not value:
        return None
    try:
        # ``fromisoformat`` accepts both naive timestamps and explicit offsets.
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        # Older exports occasionally contain a sortable date prefix only.
        compact = value[:10]
        return compact if len(compact) == 10 else None


def skip_event(event_dir: Path, root_dir: str, reason: str = "single", *,
               batch_id: Optional[str] = None) -> Dict[str, Any]:
    """Hide a pending event persistently without assigning a training label.

    Skipped samples are moved out of ``pending`` so reopening the reviewer does
    not show them again.  They remain intact under ``skipped`` and every move is
    appended to the persistent review log, making the action reversible.
    """
    event_dir = Path(event_dir).expanduser().resolve()
    root = Path(root_dir).expanduser().resolve()
    metadata = load_metadata(event_dir)
    original_metadata = dict(metadata)
    predicted = str(metadata.get("predicted_label", event_dir.parent.name))
    predicted = "meteor" if predicted == "meteor" else "not_meteor"
    destination_parent = root / "skipped" / predicted
    destination_parent.mkdir(parents=True, exist_ok=True)
    destination = destination_parent / event_dir.name
    if destination.exists():
        destination = destination_parent / f"{event_dir.name}_{uuid.uuid4().hex[:8]}"

    timestamp = _utc_now()
    metadata.update({
        "review_status": "skipped",
        "skipped_at": timestamp,
        "skip_reason": str(reason or "single"),
    })
    record = {
        "action": "skip",
        "timestamp": timestamp,
        "event_id": destination.name,
        "predicted_label": predicted,
        "detection_date": detection_date(destination),
        "reason": str(reason or "single"),
        "from": str(event_dir),
        "to": str(destination),
    }
    if batch_id:
        metadata["skip_batch_id"] = batch_id
        record["batch_id"] = batch_id
    try:
        with (event_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
        shutil.move(str(event_dir), str(destination))
        _append_jsonl(root / "review_log.jsonl", record)
    except Exception:
        if destination.exists() and not event_dir.exists():
            shutil.move(str(destination), str(event_dir))
        if event_dir.exists():
            with (event_dir / "metadata.json").open("w", encoding="utf-8") as handle:
                json.dump(original_metadata, handle, ensure_ascii=False, indent=2)
        raise
    return record


def undo_skip(record: Dict[str, Any], root_dir: str) -> Optional[Path]:
    """Restore one still-skipped event to its original pending directory."""
    source = Path(record.get("to", ""))
    destination = Path(record.get("from", ""))
    if not source.exists() or destination.exists():
        return None
    metadata = load_metadata(source)
    original_metadata = dict(metadata)
    metadata["review_status"] = "pending"
    for key in ("skipped_at", "skip_reason", "skip_batch_id"):
        metadata.pop(key, None)
    try:
        with (source / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        _append_jsonl(Path(root_dir).expanduser() / "review_log.jsonl", {
            "action": "undo_skip", "timestamp": _utc_now(),
            "event_id": destination.name,
            "reverts_skip_timestamp": record.get("timestamp"),
            "from": str(source), "to": str(destination),
        })
    except Exception:
        if destination.exists() and not source.exists():
            shutil.move(str(destination), str(source))
        if source.exists():
            with (source / "metadata.json").open("w", encoding="utf-8") as handle:
                json.dump(original_metadata, handle, ensure_ascii=False, indent=2)
        raise
    return destination


def undo_review(record: Dict[str, Any], root_dir: str) -> Optional[Path]:
    source = Path(record.get("to", ""))
    destination = Path(record.get("from", ""))
    if not source.exists() or destination.exists():
        return None
    metadata = load_metadata(source)
    reviewed_metadata = dict(metadata)
    media_moves = metadata.get("detected_media_moves", record.get("media_moves", []))
    if not isinstance(media_moves, list):
        media_moves = []
    restored_media_moves = _apply_media_moves(
        ((Path(item["to"]), Path(item["from"])) for item in reversed(media_moves)),
        rename_conflicts=False,
    )
    metadata["review_status"] = "pending"
    for key in ("reviewed_label", "was_misclassified", "reviewed_at", "review_batch_id"):
        metadata.pop(key, None)
    metadata.pop("detected_media_moves", None)
    try:
        with (source / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        _append_jsonl(Path(root_dir).expanduser() / "review_log.jsonl", {
            "action": "undo", "timestamp": _utc_now(), "event_id": destination.name,
            "reverts_review_timestamp": record.get("timestamp"), "from": str(source),
            "to": str(destination),
        })
    except Exception:
        if destination.exists() and not source.exists():
            shutil.move(str(destination), str(source))
        if source.exists():
            with (source / "metadata.json").open("w", encoding="utf-8") as handle:
                json.dump(reviewed_metadata, handle, ensure_ascii=False, indent=2)
        _apply_media_moves(
            ((Path(item["to"]), Path(item["from"])) for item in reversed(restored_media_moves)),
            rename_conflicts=False,
        )
        raise
    return destination


def _read_history(log_path, cancel):
    records = {"review": [], "skip": []}
    timestamps = {"review": {}, "skip": {}}
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if cancel is not None and cancel.is_set():
                raise InterruptedError("レビュー読み込みを中断しました")
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(record, dict):
                continue
            action = record.get("action")
            if not isinstance(action, str):
                continue
            if action in records:
                active = records[action]
                index = len(active)
                active.append(record)
                try:
                    timestamps[action].setdefault(record.get("timestamp"), []).append(index)
                except TypeError:
                    pass  # Legacy malformed timestamps retain last-match semantics.
            elif action in ("undo", "undo_skip"):
                kind = "review" if action == "undo" else "skip"
                reverted = record.get("reverts_review_timestamp" if kind == "review"
                                      else "reverts_skip_timestamp")
                active = records[kind]
                try:
                    bucket = timestamps[kind].get(reverted, [])
                    if bucket:
                        active[bucket.pop()] = None
                except TypeError:
                    for index in range(len(active) - 1, -1, -1):
                        if active[index] is not None and active[index].get("timestamp") == reverted:
                            active[index] = None
                            break
    return tuple([(record, Path(record.get("to", ""))) for record in records[kind]
                  if record is not None] for kind in ("review", "skip"))


def _surviving_history(records, cancel):
    # Historical logs contain many deleted/moved samples. Scan each parent
    # directory once instead of issuing one stat per historical record.
    names = {}
    result = []
    for record, path in records:
        if cancel is not None and cancel.is_set():
            raise InterruptedError("レビュー読み込みを中断しました")
        parent = path.parent
        if not path.name:
            exists = path.exists()
        else:
            if parent not in names:
                try:
                    with os.scandir(parent) as entries:
                        names[parent] = {entry.name for entry in entries
                                         if not entry.is_symlink() or Path(entry.path).exists()}
                except FileNotFoundError:
                    names[parent] = set()
                except OSError:
                    names[parent] = None
            inventory = names[parent]
            exists = path.exists() if inventory is None else path.name in inventory
        if exists:
            result.append(record)
    return result


def undoable_history(root_dir: str, *, cancel: Optional[threading.Event] = None,
                     strict: bool = False
                     ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Reuse an unchanged log, while rechecking actual directory inventories."""
    log_path = Path(root_dir).expanduser().resolve() / "review_log.jsonl"
    try:
        identity = _file_identity(log_path)
        with _history_cache_lock:
            cached = _history_cache.get(log_path)
            if cached is not None and cached[0] == identity:
                records = cached[1]
            else:
                records = _read_history(log_path, cancel)
                if _file_identity(log_path) == identity:
                    if len(_history_cache) >= 4:
                        _history_cache.clear()
                    _history_cache[log_path] = (identity, records)
        return tuple(_surviving_history(group, cancel) for group in records)
    except FileNotFoundError:
        return [], []
    except InterruptedError:
        raise
    except OSError:
        if strict:
            raise
        return [], []


def undoable_reviews(root_dir: str) -> List[Dict[str, Any]]:
    """Return active review records, including earlier app sessions."""
    return undoable_history(root_dir)[0]


def undoable_skips(root_dir: str) -> List[Dict[str, Any]]:
    """Return active skip records, including earlier app sessions."""
    return undoable_history(root_dir)[1]
