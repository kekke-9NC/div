"""Simple location utility: try to get location via IP geolocation, fallback to default.

This uses a public IP geolocation endpoint (ip-api.com) via urllib to avoid extra deps.
If anything fails, returns the default coordinates (lat=35.0, lon=135.0).
"""
from __future__ import annotations
import urllib.request
import urllib.error
import json
from typing import Tuple


DEFAULT_LAT = 35.0
DEFAULT_LON = 135.0


def get_current_location(timeout: int = 5) -> Tuple[float, float]:
    """Attempt to get current location (latitude, longitude).

    Returns (lat, lon). On any error or unexpected response returns the default
    coordinates (35.0, 135.0).
    """
    try:
        url = "http://ip-api.com/json/?fields=status,message,lat,lon"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)

        if isinstance(data, dict) and data.get("status") == "success":
            lat = float(data.get("lat", DEFAULT_LAT))
            lon = float(data.get("lon", DEFAULT_LON))
            return lat, lon
        else:
            return DEFAULT_LAT, DEFAULT_LON
    except Exception as e:
        print(f"location_utils: failed to get location: {e}")
        return DEFAULT_LAT, DEFAULT_LON
