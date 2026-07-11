"""Responsive discovery of videos selected through the GUI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set

import observation_time_filter


ProgressCallback = Callable[[str], None]


def _scan_directory(
    root: Path,
    extensions: Set[str],
    cancel_flag=None,
) -> List[Path]:
    videos: List[Path] = []
    for current_root, dirs, files in os.walk(root):
        if cancel_flag is not None and cancel_flag.is_set():
            break
        dirs.sort()
        for filename in sorted(files):
            if Path(filename).suffix.lower() in extensions:
                videos.append(Path(current_root) / filename)
    return videos


def discover_sources(
    selected_paths: Sequence[str],
    extensions: Iterable[str],
    *,
    twilight_filter_enabled: bool,
    latitude: float,
    longitude: float,
    progress_callback: Optional[ProgressCallback] = None,
    cancel_flag=None,
) -> List[Dict[str, object]]:
    """Enumerate selected sources outside the Tk thread and return pipeline entries."""
    extension_set = {str(ext).lower() for ext in extensions}
    sources: List[Dict[str, object]] = []
    seen = set()
    total_selected = len(selected_paths)

    for index, raw_path in enumerate(selected_paths, start=1):
        if cancel_flag is not None and cancel_flag.is_set():
            break
        path = Path(raw_path)
        if progress_callback:
            progress_callback(f"動画を走査中 ({index}/{total_selected}): {path}")
        if path.is_dir():
            found = _scan_directory(path, extension_set, cancel_flag)
            original_count = len(found)
            if twilight_filter_enabled:
                found, info = observation_time_filter.filter_date_root_videos(
                    path, found, latitude, longitude
                )
                if info.get("applied") and progress_callback:
                    dawn = info["astro_dawn"].strftime("%H:%M")
                    dusk = info["astro_dusk"].strftime("%H:%M")
                    progress_callback(
                        f"{path.name}: 天文薄明フィルタ {dawn}以前 / {dusk}以後 "
                        f"({original_count}本 → {len(found)}本)"
                    )
            candidates = found
        elif path.is_file() and path.suffix.lower() in extension_set:
            candidates = [path]
        else:
            candidates = []

        for candidate in candidates:
            try:
                key = str(candidate.resolve())
            except OSError:
                key = str(candidate)
            if key not in seen:
                seen.add(key)
                sources.append({"path": str(candidate), "is_rtsp": False})

    return sources
