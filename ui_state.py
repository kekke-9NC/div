from typing import List


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
