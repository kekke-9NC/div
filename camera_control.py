import json
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests


class CameraControlError(Exception):
    pass


def camera_base_url_from_rtsp(rtsp_url: str) -> str:
    parsed = urlparse((rtsp_url or "").strip())
    if not parsed.hostname:
        return ""
    return f"http://{parsed.hostname}"


@dataclass
class CameraControlClient:
    base_url: str
    timeout: float = 5.0

    def __post_init__(self):
        self.base_url = (self.base_url or "").strip().rstrip("/")
        if not self.base_url:
            raise CameraControlError("Camera URL is empty.")
        if not self.base_url.startswith(("http://", "https://")):
            self.base_url = "http://" + self.base_url
        self.session = requests.Session()

    def get_image_adjustment(self) -> Dict[str, Any]:
        return self._request("GET", "/action/getImageAdjustment")

    def get_image_adjustment_ex(self) -> Dict[str, Any]:
        return self._request("GET", "/action/getImageAdjustmentEx")

    def set_image_adjustment(self, values: Dict[str, Any]) -> Dict[str, Any]:
        if not values:
            return {"code": 0}
        return self._request("POST", "/action/setImageAdjustment", values)

    def set_image_adjustment_ex(self, values: Dict[str, Any]) -> Dict[str, Any]:
        if not values:
            return {"code": 0}
        return self._request("POST", "/action/setImageAdjustmentEx", values)

    def _request(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = self.base_url + path
        try:
            if method.upper() == "GET":
                response = self.session.get(url, timeout=self.timeout)
            else:
                response = self.session.post(url, json=payload or {}, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise CameraControlError(f"Camera API request failed: {exc}") from exc

        try:
            data = response.json()
        except ValueError:
            try:
                data = json.loads(response.content.decode("utf-8", errors="replace"))
            except Exception as exc:
                raise CameraControlError("Camera API returned non-JSON response.") from exc

        if not isinstance(data, dict):
            raise CameraControlError("Camera API returned unexpected response.")
        code = data.get("code", 0)
        if code not in (0, "0", None):
            raise CameraControlError(f"Camera API returned code={code}.")
        return data
