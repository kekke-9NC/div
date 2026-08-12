"""Built-in and user-editable meteor-shower catalogue.

The radiant-analysis UI keeps a local copy of this catalogue so users can
add, edit, or remove entries without changing application source code.  The
values are representative catalogue positions for shower association, not a
replacement for a full orbit solution.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import config
import meteor_radiant_analysis as radiant_analysis
from meteor_radiant_analysis import MeteorShower


CATALOG_VERSION = 1
CATALOG_FILENAME = "meteor_shower_catalog.json"

# Commonly used minor-shower reference positions.  These are the 2026 IMO
# working-list values at or near maximum; radiant drift is intentionally
# handled as an editable catalogue field rather than hidden in the matcher.
EXTRA_SHOWERS: Tuple[MeteorShower, ...] = (
    MeteorShower("GUM", "こぐま座γ流星群", (1, 10), (1, 18), (1, 22), 228.0, 67.0, category="小流星群"),
    MeteorShower("ACE", "ケンタウルス座α流星群", (1, 31), (2, 8), (2, 20), 211.0, -58.0, category="小流星群"),
    MeteorShower("PPU", "とも座π流星群", (4, 15), (4, 24), (4, 28), 110.0, -45.0, category="小流星群"),
    MeteorShower("ELY", "こと座η流星群", (5, 3), (5, 11), (5, 14), 291.0, 43.0, category="小流星群"),
    MeteorShower("JBO", "6月うしかい座流星群", (6, 22), (6, 22), (7, 2), 221.0, 48.0, category="小流星群"),
    MeteorShower("JPE", "7月ペガスス座流星群", (7, 1), (7, 10), (7, 20), 347.0, 11.0, category="小流星群"),
    MeteorShower("GDR", "7月γりゅう座流星群", (7, 25), (7, 28), (7, 31), 280.0, 51.0, category="小流星群"),
    MeteorShower("ERI", "エリダヌス座η流星群", (7, 31), (8, 7), (8, 19), 41.0, -11.0, category="小流星群"),
    MeteorShower("SLY", "9月りゅう座流星群", (9, 10), (9, 13), (10, 8), 113.0, 56.0, category="小流星群"),
    MeteorShower("DSX", "ろくぶんぎ座昼間流星群", (9, 20), (10, 1), (10, 6), 156.0, -2.0, category="小流星群"),
    MeteorShower("OCT", "10月きりん座流星群", (10, 5), (10, 6), (10, 6), 164.0, 79.0, category="小流星群"),
    MeteorShower("EGE", "ふたご座ε流星群", (10, 14), (10, 18), (10, 27), 102.0, 27.0, category="小流星群"),
    MeteorShower("LMI", "こじし座流星群", (10, 19), (10, 24), (10, 27), 162.0, 37.0, category="小流星群"),
    MeteorShower("NTA", "おうし座北流星群", (10, 20), (11, 12), (12, 10), 58.0, 22.0, category="小流星群"),
    MeteorShower("AMO", "いっかくじゅう座α流星群", (11, 15), (11, 22), (11, 25), 117.0, 1.0, category="小流星群"),
    MeteorShower("NOO", "11月オリオン座流星群", (11, 13), (11, 28), (12, 6), 91.0, 16.0, category="小流星群"),
    MeteorShower("PHO", "ほうおう座流星群", (12, 1), (12, 2), (12, 5), 8.0, -27.0, category="小流星群"),
    MeteorShower("PUP", "とも座・らしんばん座流星群", (12, 1), (12, 7), (12, 15), 123.0, -45.0, category="小流星群"),
    MeteorShower("MON", "いっかくじゅう座流星群", (12, 1), (12, 9), (12, 19), 100.0, 8.0, category="小流星群"),
    MeteorShower("HYD", "みずへび座σ流星群", (12, 3), (12, 9), (12, 20), 125.0, 2.0, category="小流星群"),
    MeteorShower("COM", "かみのけ座流星群", (12, 4), (12, 23), (1, 30), 164.0, 29.0, category="小流星群"),
)

MAJOR_CODES = {
    "QUA", "LYR", "ETA", "SDA", "PER", "ORI", "LEO", "GEM", "URS",
}


def catalogue_path(path: Optional[str] = None) -> Path:
    return Path(path or os.path.join(config.EXE_DIR, CATALOG_FILENAME)).expanduser().resolve()


def catalogue_source_label(path: Optional[str] = None) -> str:
    """Describe whether the app is using its built-in or saved catalogue."""
    target = catalogue_path(path)
    return "内蔵カタログ" if not target.is_file() else f"保存済みカタログ（{target.name}）"


def default_catalogue() -> List[MeteorShower]:
    entries: List[MeteorShower] = []
    seen = set()
    for shower in (*radiant_analysis.METEOR_SHOWERS, *EXTRA_SHOWERS):
        if shower.code in seen:
            continue
        seen.add(shower.code)
        category = "大流星群" if shower.code in MAJOR_CODES else shower.category
        entries.append(MeteorShower(
            shower.code,
            shower.name,
            shower.active_start,
            shower.peak,
            shower.active_end,
            shower.radiant_ra_deg,
            shower.radiant_dec_deg,
            shower.match_limit_deg,
            category,
        ))
    return entries


def shower_to_record(shower: MeteorShower) -> Dict[str, Any]:
    return {
        "code": shower.code,
        "name": shower.name,
        "active_start": list(shower.active_start),
        "peak": list(shower.peak),
        "active_end": list(shower.active_end),
        "radiant_ra_deg": shower.radiant_ra_deg,
        "radiant_dec_deg": shower.radiant_dec_deg,
        "match_limit_deg": shower.match_limit_deg,
        "category": shower.category,
    }


def record_to_shower(record: Dict[str, Any]) -> MeteorShower:
    def date_pair(key: str) -> Tuple[int, int]:
        value = record.get(key)
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{key} must be [month, day]")
        month, day = int(value[0]), int(value[1])
        if not 1 <= month <= 12 or not 1 <= day <= 31:
            raise ValueError(f"invalid {key}: {value}")
        try:
            datetime(2024, month, day)
        except ValueError as exc:
            raise ValueError(f"invalid {key}: {value}") from exc
        return month, day

    code = str(record.get("code", "")).strip().upper()
    name = str(record.get("name", "")).strip()
    if not code or not name:
        raise ValueError("code and name are required")
    ra = float(record["radiant_ra_deg"])
    dec = float(record["radiant_dec_deg"])
    limit = float(record.get("match_limit_deg", 12.0))
    if not math.isfinite(ra):
        raise ValueError("radiant_ra_deg must be finite")
    if not -90.0 <= dec <= 90.0:
        raise ValueError("radiant_dec_deg must be between -90 and 90")
    if not 0.0 < limit <= 90.0:
        raise ValueError("match_limit_deg must be between 0 and 90")
    return MeteorShower(
        code=code,
        name=name,
        active_start=date_pair("active_start"),
        peak=date_pair("peak"),
        active_end=date_pair("active_end"),
        radiant_ra_deg=ra % 360.0,
        radiant_dec_deg=dec,
        match_limit_deg=limit,
        category=str(record.get("category", "小流星群")) or "小流星群",
    )


def load_catalogue(path: Optional[str] = None) -> List[MeteorShower]:
    target = catalogue_path(path)
    if not target.is_file():
        return default_catalogue()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        records = payload.get("showers", payload) if isinstance(payload, (dict, list)) else []
        if not isinstance(records, list):
            raise ValueError("catalogue showers must be a list")
        loaded = [record_to_shower(item) for item in records if isinstance(item, dict)]
        return loaded or default_catalogue()
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return default_catalogue()


def save_catalogue(showers: Sequence[MeteorShower], path: Optional[str] = None) -> str:
    target = catalogue_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CATALOG_VERSION,
        "note": "代表放射点による候補判定用。完全な軌道解ではありません。",
        "showers": [shower_to_record(shower) for shower in showers],
    }
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)
    return str(target)


def date_text(value: Tuple[int, int]) -> str:
    return f"{value[0]:02d}/{value[1]:02d}"


def shower_summary(shower: MeteorShower) -> str:
    return (
        f"{shower.code}  {shower.name}  |  {shower.category}  |  "
        f"RA {shower.radiant_ra_deg:.1f}° / Dec {shower.radiant_dec_deg:+.1f}°  |  "
        f"活動 {date_text(shower.active_start)}〜{date_text(shower.active_end)}"
    )
