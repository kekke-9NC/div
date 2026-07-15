"""Shared media start-time discovery helpers.

Filesystem creation time is authoritative. Embedded media metadata and the
recorder's YYYYMMDD/HH/MM path are fallbacks for filesystems without birth
time support.
"""

import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Tuple


def _filesystem_creation_time(path: str) -> Optional[datetime]:
    try:
        stat_result = os.stat(path)
    except OSError:
        return None
    timestamp = getattr(stat_result, "st_birthtime", None)
    if timestamp is None and os.name == "nt":
        timestamp = stat_result.st_ctime
    if timestamp and timestamp > 0:
        return datetime.fromtimestamp(timestamp)
    return None


def _embedded_creation_time(path: str) -> Optional[datetime]:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format_tags=creation_time",
                "-of", "json", path,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        if result.returncode != 0:
            return None
        value = (json.loads(result.stdout or "{}").get("format", {}).get("tags", {})
                 .get("creation_time"))
        if not value:
            return None
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone().replace(tzinfo=None)
        return parsed
    except (OSError, ValueError, TypeError, json.JSONDecodeError, subprocess.TimeoutExpired):
        return None


def _path_time(path: str) -> Optional[datetime]:
    source = Path(path)
    parts = source.parts
    for index, part in enumerate(parts):
        if not re.fullmatch(r"(?:19|20)\d{6}", part):
            continue
        if index + 1 >= len(parts) or not re.fullmatch(r"\d{1,2}", parts[index + 1]):
            continue
        # Legacy RTSP files use MM.mp4.  The disk-backed NoiseTwin pipeline
        # uses MM_SS.mp4 so reconnects within the same minute cannot
        # overwrite one another; retain those seconds in the fallback time.
        segment_match = re.match(r"(\d{2})(?:_(\d{2}))?", source.stem)
        if not segment_match:
            continue
        try:
            return datetime.strptime(
                f"{part}{int(parts[index + 1]):02d}"
                f"{segment_match.group(1)}{segment_match.group(2) or '00'}",
                "%Y%m%d%H%M%S",
            )
        except ValueError:
            continue

    compact_match = re.search(r"((?:19|20)\d{12})", source.stem)
    if compact_match:
        try:
            return datetime.strptime(compact_match.group(1), "%Y%m%d%H%M%S")
        except ValueError:
            pass
    separated_match = re.search(r"((?:19|20)\d{6})[_-](\d{6})", source.stem)
    if separated_match:
        try:
            return datetime.strptime("".join(separated_match.groups()), "%Y%m%d%H%M%S")
        except ValueError:
            pass
    return None


def get_media_start_time(path: str) -> Tuple[Optional[datetime], str]:
    """Return media start time and the source used to determine it."""
    created = _filesystem_creation_time(path)
    if created is not None:
        return created, "ファイル作成時刻"
    embedded = _embedded_creation_time(path)
    if embedded is not None:
        return embedded, "動画メタデータ creation_time"
    inferred = _path_time(path)
    if inferred is not None:
        return inferred, "ファイル階層・名前"
    return None, "取得不可"


def first_media_start_time(paths: Iterable[str]) -> Tuple[Optional[datetime], str, Optional[str]]:
    """Return the first existing media's start time, preserving input order."""
    for path in paths:
        if not os.path.isfile(path):
            continue
        timestamp, source = get_media_start_time(path)
        if timestamp is not None:
            return timestamp, source, path
    return None, "取得不可", None


def local_timezone_label() -> str:
    return datetime.now().astimezone().tzname() or "LOCAL"
