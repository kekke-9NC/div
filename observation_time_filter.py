"""Filter date-rooted video archives to astronomical-night observation times."""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import sun_times


DATE_RE = re.compile(r"^\d{8}$")


def date_from_root(path: Path) -> Optional[date]:
    if not DATE_RE.match(path.name):
        return None
    try:
        return datetime.strptime(path.name, "%Y%m%d").date()
    except ValueError:
        return None


def datetime_from_archive_path(video_path: Path, date_root: Path) -> Optional[datetime]:
    day = date_from_root(date_root)
    if day is None:
        return None
    try:
        relative = video_path.relative_to(date_root)
    except ValueError:
        return None
    parts = relative.parts
    if len(parts) < 2:
        return None
    try:
        hour = int(parts[0])
        minute_match = re.match(r"^(\d{1,2})", video_path.stem)
        if minute_match is None:
            return None
        minute = int(minute_match.group(1))
        return datetime(day.year, day.month, day.day, hour, minute)
    except (TypeError, ValueError):
        return None


def astronomical_window(day: date, lat: float, lon: float):
    times = sun_times.get_sun_times(lat, lon, when=day)
    dawn = times.get("astro_dawn")
    dusk = times.get("astro_dusk")
    # The legacy NOAA helper returns a next-day date for morning events in UTC+ timezones.
    # Archive filtering is calendar-day based, so retain the calculated local clock time.
    if dawn is not None:
        dawn = datetime.combine(day, dawn.time())
    if dusk is not None:
        dusk = datetime.combine(day, dusk.time())
    return dawn, dusk


def filter_date_root_videos(
    date_root: Path, videos: Sequence[Path], lat: float, lon: float
) -> Tuple[List[Path], Dict[str, object]]:
    day = date_from_root(date_root)
    if day is None:
        return list(videos), {"applied": False}
    dawn, dusk = astronomical_window(day, lat, lon)
    if dawn is None or dusk is None:
        return list(videos), {"applied": False, "reason": "薄明時刻を計算できません"}
    included, unknown = [], []
    for video in videos:
        timestamp = datetime_from_archive_path(video, date_root)
        if timestamp is None:
            unknown.append(video)
        elif timestamp <= dawn or timestamp >= dusk:
            included.append(video)
    # Unknown filenames are retained rather than silently losing user data.
    included.extend(unknown)
    included.sort()
    return included, {
        "applied": True,
        "date": day.isoformat(),
        "astro_dawn": dawn,
        "astro_dusk": dusk,
        "input_count": len(videos),
        "included_count": len(included),
        "unknown_count": len(unknown),
    }
