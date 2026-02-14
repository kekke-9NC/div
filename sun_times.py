"""Compute sunrise/sunset and astronomical twilight times for a given location/date.

This implements the NOAA sunrise/sunset algorithm (approximate) and returns
local naive datetimes for sunrise, sunset, astronomical dawn (start) and
astronomical dusk (end). Timezone is approximated by longitude (offset hours = round(lon/15)).

The outputs are naive datetimes in the local timezone (no tzinfo).
"""
from __future__ import annotations
import math
from datetime import date, datetime, timedelta
from typing import Optional, Tuple, Dict


def _day_of_year(d: date) -> int:
    return d.timetuple().tm_yday


def _normalize_angle_deg(angle: float) -> float:
    return angle % 360.0


def _deg_to_rad(d: float) -> float:
    return math.radians(d)


def _rad_to_deg(r: float) -> float:
    return math.degrees(r)


def _calc_time_utc(is_rise: bool, lat: float, lon: float, zenith: float, when: date) -> Optional[float]:
    # Based on NOAA algorithm: returns time in UTC hours (0-24) or None when sun doesn't rise/set
    N = _day_of_year(when)
    lng_hour = lon / 15.0

    # approximate time
    if is_rise:
        t = N + ((6 - lng_hour) / 24.0)
    else:
        t = N + ((18 - lng_hour) / 24.0)

    # Sun's mean anomaly
    M = (0.9856 * t) - 3.289

    # Sun's true longitude
    L = M + (1.916 * math.sin(_deg_to_rad(M))) + (0.020 * math.sin(_deg_to_rad(2 * M))) + 282.634
    L = _normalize_angle_deg(L)

    # Sun's right ascension
    RA = _rad_to_deg(math.atan(0.91764 * math.tan(_deg_to_rad(L))))
    RA = _normalize_angle_deg(RA)

    # RA needs to be in the same quadrant as L
    L_quadrant = (math.floor(L / 90.0)) * 90.0
    RA_quadrant = (math.floor(RA / 90.0)) * 90.0
    RA = RA + (L_quadrant - RA_quadrant)

    RA = RA / 15.0  # convert to hours

    # Sun's declination
    sinDec = 0.39782 * math.sin(_deg_to_rad(L))
    cosDec = math.cos(math.asin(sinDec))

    # Sun's local hour angle
    cosH = (math.cos(_deg_to_rad(zenith)) - (sinDec * math.sin(_deg_to_rad(lat)))) / (cosDec * math.cos(_deg_to_rad(lat)))

    if cosH > 1:
        return None  # sun never rises on this location (on the specified date)
    if cosH < -1:
        return None  # sun never sets

    if is_rise:
        H = 360.0 - _rad_to_deg(math.acos(cosH))
    else:
        H = _rad_to_deg(math.acos(cosH))

    H = H / 15.0

    # local mean time of rising/setting
    T = H + RA - (0.06571 * t) - 6.622

    # UTC time
    UT = T - lng_hour
    UT = UT % 24.0
    return UT


def _utc_hours_to_local_datetime(utc_hours: float, when: date, tz_offset_hours: int) -> datetime:
    # Round via total seconds to avoid invalid values such as second=60 caused
    # by floating-point error around hh:mm:59.999...
    total_seconds = int(round((utc_hours % 24.0) * 3600.0))
    dt_utc = datetime(when.year, when.month, when.day) + timedelta(seconds=total_seconds)
    return dt_utc + timedelta(hours=tz_offset_hours)


def get_sun_times(lat: float, lon: float, when: Optional[date] = None, tz_offset_hours: Optional[int] = None) -> Dict[str, Optional[datetime]]:
    """Return a dict with keys: sunrise, sunset, astro_dawn, astro_dusk.

    Each value is a naive datetime in local time (tzinfo=None) or None if not applicable.
    tz_offset_hours: if None, approximate timezone as round(lon/15).
    """
    if when is None:
        when = date.today()
    if tz_offset_hours is None:
        tz_offset_hours = int(round(lon / 15.0))

    results: Dict[str, Optional[datetime]] = {'sunrise': None, 'sunset': None, 'astro_dawn': None, 'astro_dusk': None}

    # official sunrise/sunset uses zenith 90.833 degrees
    sunrise_utc = _calc_time_utc(True, lat, lon, 90.833, when)
    sunset_utc = _calc_time_utc(False, lat, lon, 90.833, when)

    if sunrise_utc is not None:
        results['sunrise'] = _utc_hours_to_local_datetime(sunrise_utc, when, tz_offset_hours)
    if sunset_utc is not None:
        results['sunset'] = _utc_hours_to_local_datetime(sunset_utc, when, tz_offset_hours)

    # astronomical twilight: zenith 108 degrees (sun 18° below horizon)
    astro_dawn_utc = _calc_time_utc(True, lat, lon, 108.0, when)
    astro_dusk_utc = _calc_time_utc(False, lat, lon, 108.0, when)
    if astro_dawn_utc is not None:
        results['astro_dawn'] = _utc_hours_to_local_datetime(astro_dawn_utc, when, tz_offset_hours)
    if astro_dusk_utc is not None:
        results['astro_dusk'] = _utc_hours_to_local_datetime(astro_dusk_utc, when, tz_offset_hours)

    return results


def pretty_print(lat: float, lon: float, when: Optional[date] = None, tz_offset_hours: Optional[int] = None) -> None:
    times = get_sun_times(lat, lon, when=when, tz_offset_hours=tz_offset_hours)
    print(f"Sun times for lat={lat}, lon={lon} on {when or date.today()}")
    def fmt(k):
        v = times.get(k)
        return v.strftime('%Y-%m-%d %H:%M:%S') if v else 'N/A'

    print(f"  Sunrise:          {fmt('sunrise')}")
    print(f"  Sunset:           {fmt('sunset')}")
    print(f"  Astronomical dawn:{fmt('astro_dawn')}")
    print(f"  Astronomical dusk:{fmt('astro_dusk')}")


if __name__ == '__main__':
    # quick demo: Kyoto (35N, 135E)
    pretty_print(35.0, 135.0)


def compute_night_period(lat: float, lon: float, when: Optional[date] = None) -> Dict[str, Optional[datetime]]:
    """Compute suggested start/end times for nightly monitoring.

    Returns dict with keys 'start' and 'end' as naive datetimes in local time or None.
    start = midpoint between sunset and astronomical dusk (on 'when' date)
    end   = midpoint between sunrise and astronomical dawn on the next day
    """
    if when is None:
        when = date.today()

    today_times = get_sun_times(lat, lon, when=when)
    tomorrow_times = get_sun_times(lat, lon, when=when + timedelta(days=1))

    def midpoint(a: Optional[datetime], b: Optional[datetime]) -> Optional[datetime]:
        if a is None or b is None:
            return None
        # ensure a <= b; if not, swap
        if a > b:
            a, b = b, a
        return a + (b - a) / 2

    start = midpoint(today_times.get('sunset'), today_times.get('astro_dusk'))
    # for end, use tomorrow's astro_dawn and sunrise
    end = midpoint(tomorrow_times.get('sunrise'), tomorrow_times.get('astro_dawn'))

    return {'start': start, 'end': end}
