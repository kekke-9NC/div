"""Per-launch choices for saved RTSP work; deferral never changes saved data."""
from datetime import datetime
import json
from pathlib import Path


_FRESH_STARTS = {}


def pending_jobs(root):
    jobs = []
    for marker in sorted((Path(root) / ".schedule_jobs").glob("*/job.json")):
        job = json.loads(marker.read_text(encoding="utf-8"))
        if job["state"] != "complete":
            jobs.append(job)
    return jobs


def pending_recordings(root):
    """Find disk-backed detection/transform work without opening video decoders."""
    root = Path(root)
    pending = set()
    for path in root.rglob("*.mp4"):
        if ".work" in path.parts or ".schedule_cache" in path.parts:
            continue
        if path.name.startswith("chunk_") and "_innovation" not in path.name:
            if not Path(str(path) + ".analysis_consumed.json").exists():
                pending.add(path)
        elif ".noise_twin_spool" in path.parts and "_innovation" not in path.name:
            pending.add(path)
        elif path.name.endswith("_innovation.mp4"):
            video = path.with_name(path.name.replace("_innovation.mp4", ".mp4"))
            marker = Path(str(video) + ".preprocessed.json")
            data = json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else {}
            if video.exists() and not data.get("analyzed"):
                pending.add(video)
    for suffix in (".preprocessed.json", ".analysis_pending.json"):
        for marker in root.rglob("*.mp4" + suffix):
            if ".schedule_cache" in marker.parts or ".work" in marker.parts:
                continue
            data = json.loads(marker.read_text(encoding="utf-8"))
            video = Path(str(marker)[:-len(suffix)])
            if video.exists() and not data.get("analyzed") and data.get("method") in (
                    "temporal_mean", "temporal_mean_analysis"):
                pending.add(video)
    return sorted(pending)


def defer_previous(root, jobs, when=None):
    root = Path(root).resolve()
    # Snapshot file identities so a file created after the choice is always
    # eligible, even when it belongs to the same night's schedule directory.
    recordings = frozenset(path.resolve() for path in root.rglob("*.mp4"))
    _FRESH_STARTS[root] = (when or datetime.now(), frozenset(job["id"] for job in jobs), recordings)


def fresh_started_at(root):
    choice = _FRESH_STARTS.get(Path(root).resolve())
    return choice[0] if choice else None


def deferred_job_ids(root):
    choice = _FRESH_STARTS.get(Path(root).resolve())
    return choice[1] if choice else frozenset()


def is_deferred_recording(path):
    if not _FRESH_STARTS:
        return False
    resolved = Path(path).resolve()
    return any(resolved in choice[2] for choice in _FRESH_STARTS.values())


def clear_choices():
    _FRESH_STARTS.clear()
