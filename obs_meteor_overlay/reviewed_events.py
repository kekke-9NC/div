"""Read legacy dataset reviews without changing dataset or detection files."""
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import time

STAMP = re.compile(r'^(\d{8}_\d{9})_(?:not_)?meteor_')


def identity(value: str) -> str:
    name = Path(value).name
    if name.endswith(('.mp4', '.txt')):
        name = Path(name).stem
    name = re.sub(r'(?:_[0-9a-fA-F]{8})+$', '', name)
    for suffix in ('_summary', '_full_denoised', '_full', '_info'):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return re.sub(r'(?:_[0-9a-fA-F]{8})+$', '', name)


def event_time(value: str) -> datetime | None:
    match = STAMP.match(Path(value).name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), '%Y%m%d_%H%M%S%f')
    except ValueError:
        return None


@dataclass(frozen=True)
class MeteorEvent:
    key: str
    detected: datetime
    video: Path | None


class ReviewedMeteorCatalog:
    def __init__(self, settings_path: Path, meteor_dir: Path):
        self.settings_path = settings_path
        self.default_meteor_dir = meteor_dir
        self.settings = {}
        self.cache = {}
        self.reviews = {}
        self.review_root = None
        self.next_review_refresh = 0.0
        self.events = []

    def _path(self, key: str, fallback: Path) -> Path:
        configured = self.settings.get(key)
        path = Path(configured).expanduser() if isinstance(configured, str) and configured.strip() else fallback
        if not path.is_absolute():
            path = self.settings_path.parent / path
        return path.resolve()

    def _refresh_reviews(self, root: Path) -> None:
        if root == self.review_root and time.monotonic() < self.next_review_refresh:
            return
        if root != self.review_root:
            self.cache = {}
        cache, reviews = {}, {}
        # The current reviewed folders are authoritative. Reading only the
        # append-only log would keep obsolete labels after Undo or Skip.
        for label in ('meteor', 'not_meteor'):
            folder = root / 'reviewed' / label
            for path in folder.glob('*/metadata.json'):
                if path.parent.name.startswith('.'):
                    continue
                old = self.cache.get(path)
                try:
                    stat = path.stat()
                    fingerprint = (stat.st_mtime_ns, stat.st_size)
                    data = old[1] if old and old[0] == fingerprint else json.loads(path.read_text(encoding='utf-8'))
                    if not isinstance(data, dict):
                        raise ValueError(f'Invalid dataset metadata: {path}')
                except (OSError, ValueError):
                    if old is None:
                        # Do not publish a count from a partly saved new review.
                        raise
                    fingerprint, data = old
                cache[path] = (fingerprint, data)
                if data.get('review_status') != 'reviewed' or data.get('reviewed_label') not in ('meteor', 'not_meteor'):
                    continue
                media = data.get('detected_media')
                prefix = media.get('filename_prefix') if isinstance(media, dict) else None
                key = identity(str(prefix or data.get('event_id') or path.parent.name))
                detected = event_time(key)
                if detected is None:
                    continue
                previous = reviews.get(key)
                rank = str(data.get('reviewed_at') or '')
                if previous is None or rank > previous[2]:
                    reviews[key] = (data['reviewed_label'], detected, rank)
        self.cache, self.reviews, self.review_root = cache, reviews, root
        self.next_review_refresh = time.monotonic() + 5

    def refresh(self) -> list[MeteorEvent]:
        if self.settings_path.exists():
            settings = json.loads(self.settings_path.read_text(encoding='utf-8'))
            if not isinstance(settings, dict):
                raise ValueError('Invalid legacy settings')
            self.settings = settings
        meteor = self._path('meteor_save_path', self.default_meteor_dir)
        non_meteor = self._path('not_meteor_save_path', meteor.parent / 'not_meteor')
        root = self._path('ml_training_data_root', meteor.parent / 'ml_training_data')
        self._refresh_reviews(root)
        videos = {}
        for label, folder in (('meteor', meteor), ('not_meteor', non_meteor)):
            for path in folder.glob('*_meteor_*.mp4'):
                media_stem = re.sub(r'(?:_[0-9a-fA-F]{8})+$', '', path.stem)
                if media_stem.endswith(('_full', '_full_denoised')) or not path.is_file():
                    continue
                key = identity(path.name)
                detected = event_time(key)
                if detected is None:
                    continue
                effective_label = self.reviews.get(key, (label,))[0]
                if effective_label != 'meteor':
                    continue
                current = videos.get(key)
                # Prefer summaries; filename probability never determines the
                # corrected label and moved non-meteor filenames remain valid.
                priority = (media_stem.endswith('_summary'), label == 'meteor')
                if current is None or priority > current[0]:
                    videos[key] = (priority, MeteorEvent(key, detected, path))
        events = {key: value[1] for key, value in videos.items()}
        for key, (label, detected, _) in self.reviews.items():
            if label == 'meteor' and key not in events:
                # A reviewed meteor still counts if its original video has
                # been archived. The dataset itself is the reviewed evidence.
                events[key] = MeteorEvent(key, detected, None)
        self.events = sorted(events.values(), key=lambda event: event.detected)
        return self.events
