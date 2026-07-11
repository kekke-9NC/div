"""Export and review detector events as a training-oriented dataset."""

from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


SCHEMA_VERSION = 1
_log_lock = threading.Lock()


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
    background = np.median(stack, axis=0).astype(np.int16)
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
) -> Optional[str]:
    """Atomically export one pending event and return its directory."""
    label = "meteor" if predicted_label == "meteor" else "not_meteor"
    root = Path(root_dir).expanduser().resolve()
    pending = root / "pending" / label
    pending.mkdir(parents=True, exist_ok=True)
    safe_event_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in event_id)
    final_dir = pending / safe_event_id
    if final_dir.exists():
        final_dir = pending / f"{safe_event_id}_{uuid.uuid4().hex[:8]}"
    temp_dir = pending / f".{final_dir.name}.tmp-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=False)

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
            "meteor_probability": float(probability),
            "source": str(source),
            "detection_time": detection_time.isoformat(),
            "created_at": _utc_now(),
            "frame_rate": float(frame_rate),
            "frame_count": len(crops),
            "frame_range": [int(v) for v in frame_range],
            "cutout_rect": [int(v) for v in cutout_rect],
            "cutout_size": int(cutout_size),
            "detected_line": [[int(v) for v in point] for point in detected_line],
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
        _append_jsonl(root / "export_log.jsonl", {
            "action": "export", "timestamp": _utc_now(), "event_id": final_dir.name,
            "predicted_label": label, "path": str(final_dir),
        })
        return str(final_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def pending_events(root_dir: str) -> List[Path]:
    root = Path(root_dir).expanduser().resolve()
    events: List[Path] = []
    for label in ("meteor", "not_meteor"):
        folder = root / "pending" / label
        if folder.exists():
            events.extend(p for p in folder.iterdir() if p.is_dir() and not p.name.startswith("."))
    return sorted(events, key=lambda p: p.name)


def load_metadata(event_dir: Path) -> Dict[str, Any]:
    with (event_dir / "metadata.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def review_event(event_dir: Path, root_dir: str, reviewed_label: str) -> Dict[str, Any]:
    label = "meteor" if reviewed_label == "meteor" else "not_meteor"
    root = Path(root_dir).expanduser().resolve()
    metadata = load_metadata(event_dir)
    predicted = metadata.get("predicted_label", event_dir.parent.name)
    destination_parent = root / "reviewed" / label
    destination_parent.mkdir(parents=True, exist_ok=True)
    destination = destination_parent / event_dir.name
    if destination.exists():
        destination = destination_parent / f"{event_dir.name}_{uuid.uuid4().hex[:8]}"

    metadata.update({
        "review_status": "reviewed",
        "reviewed_label": label,
        "was_misclassified": label != predicted,
        "reviewed_at": _utc_now(),
    })
    with (event_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    shutil.move(str(event_dir), str(destination))
    record = {
        "action": "review", "timestamp": _utc_now(), "event_id": destination.name,
        "predicted_label": predicted, "reviewed_label": label,
        "was_misclassified": label != predicted,
        "from": str(event_dir), "to": str(destination),
    }
    _append_jsonl(root / "review_log.jsonl", record)
    return record


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


def skip_event(event_dir: Path, root_dir: str, reason: str = "single") -> Dict[str, Any]:
    """Hide a pending event persistently without assigning a training label.

    Skipped samples are moved out of ``pending`` so reopening the reviewer does
    not show them again.  They remain intact under ``skipped`` and every move is
    appended to the persistent review log, making the action reversible.
    """
    event_dir = Path(event_dir).expanduser().resolve()
    root = Path(root_dir).expanduser().resolve()
    metadata = load_metadata(event_dir)
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
    with (event_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    shutil.move(str(event_dir), str(destination))
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
    _append_jsonl(root / "review_log.jsonl", record)
    return record


def undo_skip(record: Dict[str, Any], root_dir: str) -> Optional[Path]:
    """Restore one still-skipped event to its original pending directory."""
    source = Path(record.get("to", ""))
    destination = Path(record.get("from", ""))
    if not source.exists() or destination.exists():
        return None
    metadata = load_metadata(source)
    metadata["review_status"] = "pending"
    for key in ("skipped_at", "skip_reason"):
        metadata.pop(key, None)
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
    return destination


def undo_review(record: Dict[str, Any], root_dir: str) -> Optional[Path]:
    source = Path(record.get("to", ""))
    destination = Path(record.get("from", ""))
    if not source.exists() or destination.exists():
        return None
    metadata = load_metadata(source)
    metadata["review_status"] = "pending"
    for key in ("reviewed_label", "was_misclassified", "reviewed_at"):
        metadata.pop(key, None)
    with (source / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))
    _append_jsonl(Path(root_dir).expanduser() / "review_log.jsonl", {
        "action": "undo", "timestamp": _utc_now(), "event_id": destination.name,
        "reverts_review_timestamp": record.get("timestamp"), "from": str(source),
        "to": str(destination),
    })
    return destination


def undoable_reviews(root_dir: str) -> List[Dict[str, Any]]:
    """Return still-active review records, including reviews from earlier app sessions."""
    log_path = Path(root_dir).expanduser() / "review_log.jsonl"
    if not log_path.exists():
        return []
    active: List[Dict[str, Any]] = []
    try:
        with log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue
                if record.get("action") == "review":
                    active.append(record)
                elif record.get("action") == "undo":
                    reverted = record.get("reverts_review_timestamp")
                    for index in range(len(active) - 1, -1, -1):
                        if active[index].get("timestamp") == reverted:
                            active.pop(index)
                            break
    except OSError:
        return []
    return [record for record in active if Path(record.get("to", "")).exists()]


def undoable_skips(root_dir: str) -> List[Dict[str, Any]]:
    """Return active skip records, including skips made in earlier sessions."""
    log_path = Path(root_dir).expanduser() / "review_log.jsonl"
    if not log_path.exists():
        return []
    active: List[Dict[str, Any]] = []
    try:
        with log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue
                if record.get("action") == "skip":
                    active.append(record)
                elif record.get("action") == "undo_skip":
                    reverted = record.get("reverts_skip_timestamp")
                    for index in range(len(active) - 1, -1, -1):
                        if active[index].get("timestamp") == reverted:
                            active.pop(index)
                            break
    except OSError:
        return []
    return [record for record in active if Path(record.get("to", "")).exists()]
