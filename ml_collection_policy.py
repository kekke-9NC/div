"""Model-relative collection of review candidates and persistent nightly quotas."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from datetime import timedelta
from pathlib import Path


DEFAULT_BOUNDARY_FRACTION = 0.30
LOW_SCORE_SAMPLE_RATE = 0.001
LOW_SCORE_NIGHTLY_LIMIT = 100
POLICY_VERSION = "model_relative_v1"


def sample_value(event_key: str) -> float:
    # Stable sampling prevents a restart or a new model from rerolling events.
    digest = hashlib.sha256((POLICY_VERSION + event_key).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def collection_decision(*, probability, decision_threshold, boundary_fraction,
                        source, detection_time, frame_range, cutout_rect):
    for name, value in (("probability", probability),
                        ("decision_threshold", decision_threshold),
                        ("boundary_fraction", boundary_fraction)):
        if isinstance(value, bool) or not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
            raise ValueError(f"{name} must be between 0 and 1")
    probability, threshold, fraction = map(float, (probability, decision_threshold, boundary_fraction))
    # The observation night crosses midnight; use the event time, not wall time.
    night = (detection_time - timedelta(hours=12)).date().isoformat()
    identity = json.dumps([str(source), detection_time.isoformat(), list(frame_range),
                           list(cutout_rect)], ensure_ascii=False, separators=(",", ":"))
    event_key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    lower = threshold * (1 - fraction)
    if probability >= threshold:
        reason = "predicted_meteor"
    elif fraction > 0 and probability >= lower:
        reason = "decision_boundary"
    elif sample_value(event_key) < LOW_SCORE_SAMPLE_RATE:
        reason = "low_score_sample"
    else:
        return None
    return {
        "policy_version": POLICY_VERSION,
        "reason": reason,
        "decision_threshold": threshold,
        "boundary_fraction": fraction,
        "boundary_lower_probability": lower,
        "low_score_sample_rate": LOW_SCORE_SAMPLE_RATE,
        "low_score_nightly_limit": LOW_SCORE_NIGHTLY_LIMIT,
        "observation_night": night,
        "event_key": event_key,
    }


def reserve_low_score(root: Path, night: str, event_key: str) -> bool:
    """Share the cap across workers, processes, restarts, and model versions.

    Reserve before writing media. An interrupted process conservatively consumes
    a slot, so a crash cannot cause the nightly limit to be exceeded.
    """
    root.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(root / "collection_quota.sqlite3"), timeout=30)
    try:
        connection.execute("CREATE TABLE IF NOT EXISTS low_score_slots "
                           "(night TEXT, event_key TEXT, PRIMARY KEY (night, event_key))")
        connection.execute("BEGIN IMMEDIATE")
        count = connection.execute("SELECT COUNT(*) FROM low_score_slots WHERE night = ?", (night,)).fetchone()[0]
        if count >= LOW_SCORE_NIGHTLY_LIMIT:
            connection.rollback()
            return False
        cursor = connection.execute("INSERT OR IGNORE INTO low_score_slots VALUES (?, ?)", (night, event_key))
        connection.commit()
        return cursor.rowcount == 1
    finally:
        connection.close()


def release_low_score(root: Path, night: str, event_key: str) -> None:
    connection = sqlite3.connect(str(root / "collection_quota.sqlite3"), timeout=30)
    try:
        with connection:
            connection.execute("DELETE FROM low_score_slots WHERE night = ? AND event_key = ?", (night, event_key))
    finally:
        connection.close()
