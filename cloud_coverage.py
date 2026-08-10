"""Cloud-fraction estimation for the automatic camera-model builder.

The preferred path is the same OpenAI-compatible VLM endpoint already used by
the application (LM Studio with ``qwen/qwen3-vl-4b``).  A deterministic image
heuristic remains available so the monitor can still operate when the VLM is
not running; its result is explicitly labelled as a fallback in the returned
record.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, asdict
import io
import json
import re
from typing import Any, Callable, Optional

import cv2
import numpy as np
import requests


@dataclass(frozen=True)
class CloudClassification:
    cloud_fraction: float
    source: str
    confidence: float
    raw_text: str = ""
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_base_url(url: str) -> str:
    value = (url or "http://localhost:1234/v1").strip().rstrip("/")
    if value.endswith("/api/v1"):
        value = value[:-7] + "/v1"
    if not value.endswith("/v1"):
        value += "/v1"
    return value


def _jpeg_data_uri(image: np.ndarray) -> str:
    if image is None:
        raise ValueError("image is required")
    frame = image
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    height, width = frame.shape[:2]
    scale = min(1.0, 960.0 / max(width, height))
    if scale < 1.0:
        frame = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        raise ValueError("could not encode image")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def _heuristic_cloud_fraction(image: np.ndarray) -> CloudClassification:
    """Estimate cloud cover without a model, conservatively and repeatably.

    Night cameras have no reliable absolute brightness threshold.  The
    heuristic therefore combines local contrast with colour/brightness and
    only claims a clear sky when the upper sky is consistently dark and
    spatially smooth.  It is deliberately a fallback, not a replacement for
    the VLM result.
    """
    frame = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    height, width = frame.shape[:2]
    top = frame[: max(1, round(height * 0.86)), :]
    gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY).astype(np.float32)
    smooth = cv2.GaussianBlur(gray, (0, 0), 9)
    local = np.abs(gray - smooth)
    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV).astype(np.float32)
    luminance = np.percentile(gray, 75)
    texture = np.percentile(local, 75)
    # Bright, broad patches and strong local structure are cloud candidates.
    bright = np.clip((gray - max(28.0, luminance * 0.72)) / 45.0, 0.0, 1.0)
    structured = np.clip((local - max(3.0, texture * 0.55)) / 18.0, 0.0, 1.0)
    low_saturation = np.clip((52.0 - hsv[:, :, 1]) / 52.0, 0.0, 1.0)
    score = 0.58 * bright + 0.30 * structured + 0.12 * low_saturation * bright
    # Ignore the lower obstruction band and obvious saturated artificial lights.
    score[: max(1, round(score.shape[0] * 0.04)), :] *= 0.75
    fraction = float(np.clip(np.mean(score > 0.48), 0.0, 1.0))
    confidence = float(np.clip(0.35 + abs(fraction - 0.5) * 0.5, 0.35, 0.7))
    return CloudClassification(fraction, "heuristic", confidence)


def _parse_fraction(text: str) -> Optional[float]:
    value = (text or "").strip()
    candidates = []
    try:
        payload = json.loads(value)
        if isinstance(payload, dict):
            for key in ("cloud_fraction", "cloud_cover", "cloud_amount", "fraction"):
                if key in payload:
                    candidates.append(payload[key])
                    break
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    match = re.search(r"(?:cloud[_ -]?(?:fraction|cover|amount)|fraction)\s*[=:：]\s*([0-9]+(?:\.[0-9]+)?)\s*(%|割)?", value, re.I)
    if match:
        candidates.append(match.group(1) + (match.group(2) or ""))
    for item in candidates:
        try:
            raw = str(item).strip().lower()
            if raw.endswith("%"):
                number = float(raw[:-1]) / 100.0
            elif raw.endswith("割"):
                number = float(raw[:-1]) / 10.0
            else:
                number = float(raw)
                if number > 1.0:
                    number /= 100.0
            if 0.0 <= number <= 1.0:
                return number
        except (TypeError, ValueError):
            continue
    return None


def _classify_with_openai_compatible(
    image: np.ndarray,
    *,
    url: str,
    model_id: str,
    api_key: str = "",
    timeout: float = 45.0,
    status_callback: Optional[Callable[[str], None]] = None,
) -> CloudClassification:
    if status_callback:
        status_callback(f"雲量判定中: {model_id}")
    endpoint = _normalize_base_url(url) + "/chat/completions"
    prompt = (
        "You classify cloud cover in a fixed wide-angle night-sky camera image. "
        "Ignore stars, the lens hood, buildings, trees and artificial lights. "
        "Return JSON only with cloud_fraction (0.0 to 1.0), confidence (0.0 to 1.0), "
        "and one short reason. cloud_fraction means the fraction of usable sky "
        "covered by opaque or translucent clouds."
    )
    payload = {
        "model": model_id or "qwen/qwen3-vl-4b",
        "temperature": 0.0,
        "max_tokens": 180,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": _jpeg_data_uri(image)}},
            ],
        }],
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
    response.raise_for_status()
    body = response.json()
    text = body["choices"][0]["message"]["content"]
    if isinstance(text, list):
        text = " ".join(str(part.get("text", part)) for part in text)
    text = str(text)
    fraction = _parse_fraction(text)
    if fraction is None:
        raise ValueError(f"VLM response did not contain cloud_fraction: {text[:240]}")
    confidence_match = re.search(r"confidence\s*[=:：]\s*([0-9]+(?:\.[0-9]+)?)", text, re.I)
    confidence = float(confidence_match.group(1)) if confidence_match else 0.75
    if confidence > 1.0:
        confidence /= 100.0
    return CloudClassification(float(fraction), "qwen-vlm", float(np.clip(confidence, 0.0, 1.0)), text)


def classify_cloud_fraction(
    image: np.ndarray,
    *,
    backend: str = "lmstudio_qwen3_5_2b",
    lm_studio_url: str = "http://localhost:1234/v1",
    lm_studio_model_id: str = "qwen/qwen3-vl-4b",
    lm_studio_api_key: str = "",
    timeout: float = 45.0,
    status_callback: Optional[Callable[[str], None]] = None,
) -> CloudClassification:
    """Classify one frame, falling back to the deterministic estimator."""
    use_vlm = str(backend or "").lower() in {
        "lmstudio_qwen3_5_2b", "lmstudio", "qwen", "qwen3_vl_4b", "local_qwen3_vl_4b"
    }
    if use_vlm and lm_studio_url and lm_studio_model_id:
        try:
            return _classify_with_openai_compatible(
                image, url=lm_studio_url, model_id=lm_studio_model_id,
                api_key=lm_studio_api_key, timeout=timeout,
                status_callback=status_callback,
            )
        except Exception as exc:
            fallback = _heuristic_cloud_fraction(image)
            return CloudClassification(
                fallback.cloud_fraction, fallback.source, fallback.confidence,
                error=f"VLM unavailable: {exc}",
            )
    return _heuristic_cloud_fraction(image)
