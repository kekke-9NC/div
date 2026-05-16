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

    IMAGE_ADJUSTMENT_KEYS = [
        "hue", "brightness", "sharpness", "contrast", "saturation", "gamma",
        "antiDIS", "blc_level", "max_exposure", "max_a_gain", "antiFog",
        "frameTurbo_pro", "sceneMode", "AE_strategy_mode", "auto_exposureEx",
        "exposure_time", "exposure_time_max", "auto_gain_mode", "auto_DGain_max",
        "auto_AGain_max", "max_sys_gain", "manual_AGain_enable", "manual_AGain",
        "manual_DGain_enable", "manual_DGain", "ai_isp", "auto_awb", "awb_red",
        "awb_green", "awb_blue", "awb_auto_mode", "awb_style_red",
        "awb_style_green", "awb_style_blue", "rotate",
    ]
    IMAGE_ADJUSTMENT_EX_KEYS = [
        "scene_mode", "auto_iris", "color_black", "flip", "hlc_enable",
        "infr_day_h", "infr_day_m", "infr_detect_mode", "infr_night_h",
        "infr_night_m", "mirror", "wdr_sensor", "wdr_level",
        "wdr_level_sensor", "power_freq", "sens_day_to_night",
        "sens_night_to_day", "blc_level", "ircut_level", "ldr_level",
        "led_control_mode", "lamp_type", "led_level", "ir_level",
        "led_control_avail", "led_control", "low_farme_rate",
        "noiseReduction", "anti_flicker", "_2DNR_level", "lens_correction",
        "byLDC_XOffset", "byLDC_YOffset", "byLDC_Ratio",
    ]

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
        current = self.get_image_adjustment()
        payload = {key: current[key] for key in self.IMAGE_ADJUSTMENT_KEYS if key in current}
        payload.update(values)
        return self._request("POST", "/action/setImageAdjustment", payload)

    def set_image_adjustment_ex(self, values: Dict[str, Any]) -> Dict[str, Any]:
        if not values:
            return {"code": 0}
        current = self.get_image_adjustment_ex()
        payload = {key: current[key] for key in self.IMAGE_ADJUSTMENT_EX_KEYS if key in current}
        payload.update(values)
        return self._request("POST", "/action/setImageAdjustmentEx", payload)

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
