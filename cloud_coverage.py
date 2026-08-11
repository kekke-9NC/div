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
import json
import re
import time
from typing import Any, Callable, Optional

import cv2
import numpy as np
import requests

from usage_metrics import record_usage


@dataclass(frozen=True)
class CloudClassification:
    cloud_fraction: float
    source: str
    confidence: float
    raw_text: str = ""
    error: str = ""
    sky_visibility: str = "unknown"
    blocked_fraction: float = 0.0
    reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    elapsed_seconds: float = 0.0

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


def _extract_json_payload(text: str) -> dict[str, Any]:
    value = (text or "").strip().replace("```json", "").replace("```", "").strip()
    try:
        payload = json.loads(value)
        if isinstance(payload, dict):
            return payload
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    # Qwen sometimes adds one short sentence before/after an otherwise valid
    # object.  Parse the first balanced-looking JSON object as a fallback.
    start, end = value.find("{"), value.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(value[start:end + 1])
            if isinstance(payload, dict):
                return payload
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    return {}


def _coerce_unit_interval(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        raw = str(value).strip().lower()
        if raw.endswith("%"):
            number = float(raw[:-1]) / 100.0
        elif raw.endswith("割"):
            number = float(raw[:-1]) / 10.0
        else:
            number = float(raw)
            if number > 1.0:
                number /= 100.0
        return float(number) if 0.0 <= number <= 1.0 else None
    except (TypeError, ValueError):
        return None


def _parse_fraction(text: str) -> Optional[float]:
    value = (text or "").strip()
    payload = _extract_json_payload(value)
    for key in ("cloud_fraction", "cloud_cover", "cloud_amount", "fraction"):
        if key in payload:
            parsed = _coerce_unit_interval(payload[key])
            if parsed is not None:
                return parsed
    match = re.search(r"[\"']?(?:cloud[_ -]?(?:fraction|cover|amount)|fraction)[\"']?\s*[=:：]\s*[\"']?([0-9]+(?:\.[0-9]+)?)[\"']?\s*(%|割)?", value, re.I)
    if match:
        return _coerce_unit_interval(match.group(1) + (match.group(2) or ""))
    return None


def _parse_confidence(text: str) -> float:
    payload = _extract_json_payload(text)
    parsed = _coerce_unit_interval(payload.get("confidence"))
    if parsed is not None:
        return parsed
    match = re.search(r"[\"']?confidence[\"']?\s*[=:：]\s*[\"']?([0-9]+(?:\.[0-9]+)?)[\"']?", text or "", re.I)
    parsed = _coerce_unit_interval(match.group(1)) if match else None
    return float(parsed if parsed is not None else 0.75)


def _classify_with_openai_compatible(
    image: np.ndarray,
    *,
    url: str,
    model_id: str,
    api_key: str = "",
    timeout: float = 45.0,
    status_callback: Optional[Callable[[str], None]] = None,
) -> CloudClassification:
    started = time.perf_counter()
    if status_callback:
        status_callback(f"雲量判定中: {model_id}")
    endpoint = _normalize_base_url(url) + "/chat/completions"
    system_prompt = (
        "You are a conservative meteorological night-sky assessor for one fixed "
        "1920x1080 ultra-wide camera. The lens has strong distortion. Your job is "
        "only to estimate the fraction of usable sky obscured by opaque or "
        "translucent clouds. Do not identify meteors or stars. "
        "Ignore individual stars, star trails, sensor hot pixels, JPEG noise, "
        "vignetting, broad exposure gradients, lens dirt, the lens hood, buildings, "
        "trees and artificial lights. Treat a broad soft gray/white structure that "
        "hides stars as cloud. Use the visible upper sky and exclude roughly the "
        "lowest 14 percent of the image and any non-sky obstruction. "
        "Mentally divide the usable sky into an 8 by 6 grid, judge each cell, and "
        "average the clear/partly-cloudy/overcast cells by area. Thin clouds count "
        "when they noticeably reduce star visibility. If uncertain, round the cloud "
        "fraction upward: this is a safety gate for astrometric calibration. "
        "A value below 0.10 is allowed only when at least 90 percent of usable sky "
        "is convincingly clear. Return one JSON object only; no markdown and no "
        "extra prose."
    )
    prompt = (
        "Analyze this current camera frame. Return exactly this schema: "
        '{"cloud_fraction":0.00,"confidence":0.00,"sky_visibility":"clear|mixed|overcast",'
        '"blocked_fraction":0.00,"reason":"short reason"}. '
        "cloud_fraction and blocked_fraction must be numbers from 0.0 to 1.0."
    )
    payload = {
        "model": model_id or "qwen/qwen3-vl-4b",
        "temperature": 0.0,
        "max_tokens": 220,
        "messages": [{
            "role": "system",
            "content": system_prompt,
        }, {
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
    usage = body.get("usage") or {}
    input_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
    output_tokens = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
    total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens) or 0)
    text = body["choices"][0]["message"]["content"]
    if isinstance(text, list):
        text = " ".join(str(part.get("text", part)) for part in text)
    text = str(text)
    fraction = _parse_fraction(text)
    if fraction is None:
        raise ValueError(f"VLM response did not contain cloud_fraction: {text[:240]}")
    payload_data = _extract_json_payload(text)
    confidence = _parse_confidence(text)
    blocked = _coerce_unit_interval(payload_data.get("blocked_fraction")) or 0.0
    visibility = str(payload_data.get("sky_visibility", "unknown")).strip().lower() or "unknown"
    reason = str(payload_data.get("reason", "")).strip()[:240]
    return CloudClassification(
        float(fraction), "qwen-vlm", float(np.clip(confidence, 0.0, 1.0)), text,
        sky_visibility=visibility, blocked_fraction=blocked, reason=reason,
        input_tokens=input_tokens, output_tokens=output_tokens,
        total_tokens=total_tokens, elapsed_seconds=time.perf_counter() - started,
    )


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
    started = time.perf_counter()
    use_vlm = str(backend or "").lower() in {
        "lmstudio_qwen3_5_2b", "lmstudio", "qwen", "qwen3_vl_4b", "local_qwen3_vl_4b"
    }
    if use_vlm and lm_studio_url and lm_studio_model_id:
        try:
            result = _classify_with_openai_compatible(
                image, url=lm_studio_url, model_id=lm_studio_model_id,
                api_key=lm_studio_api_key, timeout=timeout,
                status_callback=status_callback,
            )
            record_usage(
                "cloud_classification",
                elapsed_seconds=result.elapsed_seconds or (time.perf_counter() - started),
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                total_tokens=result.total_tokens,
                source=result.source,
                model=lm_studio_model_id,
                metadata={"backend": backend},
            )
            return result
        except Exception as exc:
            fallback = _heuristic_cloud_fraction(image)
            result = CloudClassification(
                fallback.cloud_fraction, fallback.source, fallback.confidence,
                error=f"VLM unavailable: {exc}",
                elapsed_seconds=time.perf_counter() - started,
            )
            record_usage(
                "cloud_classification",
                elapsed_seconds=result.elapsed_seconds,
                status="fallback",
                source=result.source,
                model=lm_studio_model_id,
                error=result.error,
                metadata={"backend": backend},
            )
            return result
    result = _heuristic_cloud_fraction(image)
    elapsed = time.perf_counter() - started
    result = CloudClassification(
        **{**result.as_dict(), "elapsed_seconds": elapsed}
    )
    record_usage(
        "cloud_classification",
        elapsed_seconds=elapsed,
        source=result.source,
        model=lm_studio_model_id,
        metadata={"backend": backend},
    )
    return result
