"""Metadata-only selection and cancellable, reversible bulk dataset review."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import threading
from typing import Callable
import uuid

import ml_training_data


@dataclass(frozen=True)
class ReviewFilter:
    threshold: float = 0.1
    comparison: str = "below"
    detection_date: str = ""
    predicted_label: str = ""

    def __post_init__(self):
        try:
            value = float(self.threshold)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("流星確率の閾値は0〜1で指定してください。") from exc
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("流星確率の閾値は0〜1で指定してください。")
        if self.comparison not in ("below", "above"):
            raise ValueError("比較条件が不正です。")
        if self.predicted_label not in ("", "meteor", "not_meteor"):
            raise ValueError("予測ラベルが不正です。")
        if self.detection_date:
            from datetime import date
            date.fromisoformat(self.detection_date)
        object.__setattr__(self, "threshold", value)


@dataclass(frozen=True)
class Candidate:
    path: Path
    metadata_digest: str


@dataclass
class Selection:
    criteria: ReviewFilter
    candidates: list[Candidate] = field(default_factory=list)
    unreadable: int = 0
    scanned: int = 0
    cancelled: bool = False


@dataclass
class BatchResult:
    records: list[dict] = field(default_factory=list)
    changed: int = 0
    cancelled: bool = False
    error: str = ""


def _pending_metadata(path: Path, root: Path):
    # Never follow a stale selection into reviewed/skipped or another dataset.
    if path.parent not in (root / "pending/meteor", root / "pending/not_meteor"):
        raise ValueError("未確認データ以外は一括処理できません。")
    if path.is_symlink() or path.resolve() != path or (path / "metadata.json").is_symlink():
        raise ValueError("リンク先のデータは一括処理できません。")
    raw = (path / "metadata.json").read_bytes()
    metadata = json.loads(raw)
    if not isinstance(metadata, dict) or metadata.get("review_status", "pending") != "pending":
        raise ValueError("このデータは未確認ではありません。")
    return metadata, hashlib.sha256(raw).hexdigest()


def select_events(root_dir: str, criteria: ReviewFilter, cancel: threading.Event,
                  progress: Callable[[int, int], None] = lambda _done, _total: None) -> Selection:
    root = Path(root_dir).expanduser().resolve()
    result = Selection(criteria)
    events = ml_training_data.pending_events(str(root))
    for index, path in enumerate(events):
        if cancel.is_set():
            result.cancelled = True
            break
        try:
            meta, digest = _pending_metadata(path, root)
            raw_probability = meta["meteor_probability"]
            if isinstance(raw_probability, bool):
                raise ValueError("Invalid probability")
            probability = float(raw_probability)
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("Invalid probability")
            matches = (probability <= criteria.threshold if criteria.comparison == "below"
                       else probability >= criteria.threshold)
            if criteria.predicted_label:
                matches &= meta.get("predicted_label", path.parent.name) == criteria.predicted_label
            if criteria.detection_date:
                from datetime import datetime
                day = datetime.fromisoformat(str(meta.get("detection_time", "")).replace("Z", "+00:00")).date().isoformat()
                matches &= day == criteria.detection_date
            if matches:
                result.candidates.append(Candidate(path, digest))
        except (OSError, ValueError, TypeError, KeyError, OverflowError):
            result.unreadable += 1
        result.scanned += 1
        if index % 100 == 0 or index + 1 == len(events):
            progress(index + 1, len(events))
    return result


def apply_selection(root_dir: str, selection: Selection, action: str,
                    cancel: threading.Event, media_directories=None,
                    progress: Callable[[int, int], None] = lambda _done, _total: None) -> BatchResult:
    if action not in ("skip", "meteor", "not_meteor"):
        raise ValueError("一括処理の分類が不正です。")
    if selection.cancelled:
        raise ValueError("検索が中断されたため、もう一度対象を確認してください。")
    root = Path(root_dir).expanduser().resolve()
    result = BatchResult()
    batch_id = uuid.uuid4().hex
    reason = f"probability:{selection.criteria.comparison}:{selection.criteria.threshold:g}"
    for index, candidate in enumerate(selection.candidates):
        if cancel.is_set():
            result.cancelled = True
            break
        try:
            _metadata, digest = _pending_metadata(candidate.path, root)
        except (OSError, ValueError):
            digest = None
        if digest != candidate.metadata_digest:
            result.changed += 1
        else:
            try:
                if action == "skip":
                    record = ml_training_data.skip_event(candidate.path, str(root), reason, batch_id=batch_id)
                else:
                    record = ml_training_data.review_event(candidate.path, str(root), action,
                                                          media_directories, batch_id=batch_id)
                result.records.append(record)
            except Exception as exc:
                result.error = f"{candidate.path.name}: {exc}"
                break
        if index % 20 == 0 or index + 1 == len(selection.candidates):
            progress(index + 1, len(selection.candidates))
    return result


def latest_batch_records(records: list[dict]) -> list[dict]:
    batches = [record for record in records if record.get("batch_id")]
    if not batches:
        return []
    latest = max(batches, key=lambda record: record.get("timestamp", ""))["batch_id"]
    return sorted((record for record in batches if record["batch_id"] == latest),
                  key=lambda record: record.get("timestamp", ""))


def undo_batch(root_dir: str, records: list[dict], cancel: threading.Event,
               progress: Callable[[int, int], None] = lambda _done, _total: None) -> BatchResult:
    result = BatchResult()
    for index, record in enumerate(reversed(records)):
        if cancel.is_set():
            result.cancelled = True
            break
        try:
            undo = ml_training_data.undo_skip if record["action"] == "skip" else ml_training_data.undo_review
            if undo(record, root_dir) is not None:
                result.records.append(record)
            else:
                result.changed += 1
        except Exception as exc:
            result.error = f"{record.get('event_id', '')}: {exc}"
            break
        if index % 20 == 0 or index + 1 == len(records):
            progress(index + 1, len(records))
    return result
