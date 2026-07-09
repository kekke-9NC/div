from typing import List


SOURCE_PRIORITY_DEFAULT = ("periodic", "rtsp", "folder")


def normalize_source_priority(priority) -> List[str]:
    """保存値を、利用可能な3つの入力種別を含む順序へ正規化する。"""
    normalized = []
    for item in priority or ():
        if item in SOURCE_PRIORITY_DEFAULT and item not in normalized:
            normalized.append(item)
    normalized.extend(item for item in SOURCE_PRIORITY_DEFAULT if item not in normalized)
    return normalized


def select_source_by_priority(
    priority,
    periodic_enabled: bool,
    has_rtsp: bool,
    has_folder: bool,
):
    """有効な入力のうち、優先順位が最も高い種別を1つ返す。"""
    active = {
        "periodic": periodic_enabled,
        "rtsp": has_rtsp,
        "folder": has_folder,
    }
    for source_type in normalize_source_priority(priority):
        if active[source_type]:
            return source_type
    return None


def should_enable_start(
    is_running: bool,
    cancel_flag_set: bool,
    periodic_enabled: bool,
    folder_paths: List[str],
    rtsp_urls: List[str],
    periodic_time_limit_enabled: bool,
    start_hour: str = "0",
    start_min: str = "0",
    end_hour: str = "0",
    end_min: str = "0",
) -> bool:
    """Determine whether the Start button should be enabled.

    Rules:
      - If a cancel has been requested (cancel_flag_set==True) we allow the Start
        button to be pressed again so the user can re-start — this matches the
        requested UX.
      - If the system is running normally (is_running==True) Start is disabled.
      - Otherwise Start is enabled only if periodic is enabled or there are
        folder paths or RTSP urls configured, and (if periodic time-limit is
        enabled) the start/end spinboxes parse as integers.
    """
    # Immediately allow start if cancel has been requested
    if cancel_flag_set:
        return True

    # If already running (and not cancelled) do not allow start
    if is_running:
        return False

    can_start = (periodic_enabled or bool(folder_paths) or bool(rtsp_urls))

    if periodic_enabled and periodic_time_limit_enabled:
        try:
            int(start_hour); int(start_min); int(end_hour); int(end_min)
        except Exception:
            return False

    return bool(can_start)
