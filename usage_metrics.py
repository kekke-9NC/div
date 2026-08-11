"""Small, thread-safe JSONL logger for AI processing usage metrics."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import threading
from typing import Any, Mapping, Optional

import config


_LOG_LOCK = threading.Lock()


def _as_non_negative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def record_usage(
    operation: str,
    *,
    elapsed_seconds: float = 0.0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: Optional[int] = None,
    status: str = "success",
    source: str = "",
    model: str = "",
    error: str = "",
    metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Append one usage event and return the serialized record.

    Token values are deliberately kept as zero when a backend does not expose
    token accounting (for example, the deterministic cloud heuristic or the
    local astrometric solver).  The elapsed time is still recorded.
    """
    input_count = _as_non_negative_int(input_tokens)
    output_count = _as_non_negative_int(output_tokens)
    total_count = (
        _as_non_negative_int(total_tokens)
        if total_tokens is not None
        else input_count + output_count
    )
    record: dict[str, Any] = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "operation": str(operation),
        "status": str(status or "unknown"),
        "elapsed_seconds": round(max(0.0, float(elapsed_seconds or 0.0)), 3),
        "input_tokens": input_count,
        "output_tokens": output_count,
        "total_tokens": total_count,
    }
    if source:
        record["source"] = str(source)
    if model:
        record["model"] = str(model)
    if error:
        record["error"] = str(error)[:1000]
    if metadata:
        record["metadata"] = dict(metadata)

    path = Path(config.AI_USAGE_LOG_PATH).expanduser()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _LOG_LOCK:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
    except OSError:
        # Usage logging must never interrupt calibration or monitoring.
        pass
    return record
