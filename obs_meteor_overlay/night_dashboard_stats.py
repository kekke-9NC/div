#!/usr/bin/env python3
"""Publish corrected hourly counts and compact observing-night context for OBS."""

from __future__ import annotations

import argparse
from datetime import date, datetime, time as datetime_time, timedelta, timezone
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from legacy_detection_mask import default_settings_path
from reviewed_events import ReviewedMeteorCatalog
from sun_times import get_sun_times

STOP = False
_MOON_CACHE = (None, None)


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _clock(settings: dict, prefix: str):
    try:
        hour = int(settings[f"rtsp_{prefix}_hour"])
        minute = int(settings[f"rtsp_{prefix}_minute"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return datetime_time(hour, minute)


def _night_window(now: datetime):
    if now.time() < datetime_time(6, 0):
        start = datetime.combine(now.date() - timedelta(days=1), datetime_time(18, 0))
        end = datetime.combine(now.date(), datetime_time(6, 0))
    else:
        start = datetime.combine(now.date(), datetime_time(18, 0))
        end = datetime.combine(now.date() + timedelta(days=1), datetime_time(6, 0))
    return start, end


def _scheduled_window(now: datetime, settings: dict):
    if not _truthy(settings.get("rtsp_time_limit_enabled", False)):
        return None, None
    start_clock = _clock(settings, "start")
    end_clock = _clock(settings, "end")
    if start_clock is None or end_clock is None:
        return None, None
    crosses_midnight = end_clock <= start_clock
    if crosses_midnight:
        start_date = now.date() - timedelta(days=1) if now.time() < end_clock else now.date()
        end_date = start_date + timedelta(days=1)
    else:
        start_date = now.date() + timedelta(days=1) if now.time() >= end_clock else now.date()
        end_date = start_date
    return datetime.combine(start_date, start_clock), datetime.combine(end_date, end_clock)


def _moon_details(now_local: datetime):
    """Use Astropy's bundled ephemeris; cache the result for 15 minutes."""
    global _MOON_CACHE
    now_epoch = now_local.timestamp()
    bucket = int(now_epoch // 900)
    if _MOON_CACHE[0] == bucket:
        return _MOON_CACHE[1]
    try:
        from astropy import units as u
        from astropy.coordinates import GeocentricTrueEcliptic, get_body
        from astropy.time import Time
        from astropy.utils import iers

        iers.conf.auto_download = False
        instant = Time(now_local.astimezone(timezone.utc))
        moon = get_body("moon", instant).transform_to(GeocentricTrueEcliptic(equinox=instant))
        sun = get_body("sun", instant).transform_to(GeocentricTrueEcliptic(equinox=instant))
        elongation = float((moon.lon - sun.lon).wrap_at(360 * u.deg).deg) % 360.0
        illumination = (1.0 - math.cos(math.radians(elongation))) / 2.0
        phase_index = int(((elongation + 22.5) % 360.0) // 45.0)
        phase_names = (
            "新月", "満ちていく月", "上弦", "十三夜ごろ",
            "満月", "欠けていく月", "下弦", "有明月",
        )
        details = {
            "phase_name": phase_names[phase_index],
            "age_days": round(elongation / 360.0 * 29.530588853, 1),
            "illumination_percent": int(round(illumination * 100)),
        }
    except Exception as exc:
        details = {"phase_name": "計算待ち", "age_days": None,
                   "illumination_percent": None, "error": str(exc)}
    _MOON_CACHE = (bucket, details)
    return details


def dashboard_payload(now_local: datetime, events, settings: dict):
    now = now_local.replace(tzinfo=None)
    window_start, window_end = _night_window(now)
    night_bins = [0] * 12
    for event in events:
        detected = event.detected
        if window_start <= detected < window_end and detected <= now:
            index = int((detected - window_start).total_seconds() // 3600)
            if 0 <= index < len(night_bins):
                night_bins[index] += 1

    observation_start, observation_end = _scheduled_window(now, settings)
    active_night_start = observation_start or window_start
    offset = now_local.utcoffset()
    offset_hours = int(round(offset.total_seconds() / 3600)) if offset else 0
    latitude = longitude = None
    try:
        latitude = float(settings["observation_latitude"])
        longitude = float(settings["observation_longitude"])
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            latitude = longitude = None
    except (KeyError, TypeError, ValueError):
        pass

    sunset_epoch = None
    if latitude is not None and longitude is not None:
        try:
            sunset = get_sun_times(latitude, longitude, when=active_night_start.date(),
                                   tz_offset_hours=offset_hours).get("sunset")
            if sunset is not None:
                sunset_epoch = sunset.timestamp()
        except (ValueError, OverflowError):
            pass

    moon = _moon_details(now_local)
    start_epoch = observation_start.timestamp() if observation_start else None
    end_epoch = observation_end.timestamp() if observation_end else None
    sunset_day = active_night_start.date()
    return {
        "updated_at_epoch": now_local.timestamp(),
        "classification_basis": "dataset_review_then_prediction",
        "night_window_start_epoch": window_start.timestamp(),
        "night_window_end_epoch": window_end.timestamp(),
        "night_hour_labels": [f"{hour:02d}" for hour in (*range(18, 24), *range(0, 6))],
        "night_bins": night_bins,
        "night_count": sum(night_bins),
        "night_label": f"{sunset_day.month:02d}/{sunset_day.day:02d}夜",
        "observation_schedule_enabled": observation_start is not None,
        "observation_start_epoch": start_epoch,
        "observation_end_epoch": end_epoch,
        "sunset_epoch": sunset_epoch,
        "moon_phase_name": moon["phase_name"],
        "moon_age_days": moon["age_days"],
        "moon_illumination_percent": moon["illumination_percent"],
    }


def atomic_json(path: Path, payload: dict):
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", mode="w",
                                     encoding="utf-8", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meteor-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--legacy-settings", type=Path, default=default_settings_path())
    parser.add_argument("--parent-pid", type=int)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    catalog = ReviewedMeteorCatalog(args.legacy_settings, args.meteor_dir)

    def stop(*_):
        global STOP
        STOP = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        os.nice(10)
    except OSError:
        pass
    last_error = None
    while not STOP:
        if args.parent_pid:
            try:
                os.kill(args.parent_pid, 0)
            except ProcessLookupError:
                break
        started = time.monotonic()
        try:
            events = catalog.refresh()
            payload = dashboard_payload(datetime.now().astimezone(), events, catalog.settings)
            atomic_json(args.output_dir / "night_dashboard.json", payload)
            last_error = None
        except Exception as exc:
            if str(exc) != last_error:
                print(f"[OBS dashboard] {exc}", file=sys.stderr, flush=True)
                last_error = str(exc)
        while not STOP and time.monotonic() - started < max(2.0, args.poll_seconds):
            time.sleep(0.25)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
