#!/usr/bin/env python3
"""Prepare a bounded, reviewed and masked replay list away from live updates."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import signal
import time
from broadcast_timing import atomic_json
from event_timing import EventTiming
from latest_meteor_gif import convert_raw_zoom_to_gif, convert_saved_zoom_to_gif, convert_to_gif, fallback_sources
from legacy_detection_mask import LegacyDetectionMask, default_settings_path
from reviewed_events import ReviewedMeteorCatalog

STOP = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--meteor-dir', type=Path, required=True)
    parser.add_argument('--legacy-settings', type=Path, default=default_settings_path())
    parser.add_argument('--ffmpeg', required=True)
    parser.add_argument('--parent-pid', type=int)
    args = parser.parse_args()
    os.nice(10)
    def stop(*_):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    mask = LegacyDetectionMask(args.legacy_settings)
    catalog = ReviewedMeteorCatalog(args.legacy_settings, args.meteor_dir)
    timing = EventTiming(args.output_dir)
    cache = args.output_dir / 'recent_cache'
    cache.mkdir(exist_ok=True)
    seen = set()
    try:
        seen = {e['event'] for e in json.loads((args.output_dir / 'recent_meteors.json').read_text())['events']}
    except (OSError, ValueError, KeyError):
        pass
    while not STOP:
        try:
            if args.parent_pid:
                os.kill(args.parent_pid, 0)
            all_events = catalog.refresh()
            accepted = {event.key for event in all_events}
            events = [e for e in reversed(all_events) if e.video and e.detected <= datetime.now()][:10]
            seen.update(event.key for event in events)
            mask.refresh()
            if not mask.revision:
                raise ValueError('Mask unavailable')
            entries, missing, keep = [], [], set()
            for event in events:
                revision = hashlib.sha256(f'{event.key}:{mask.revision}:{event.video.stat().st_mtime_ns}'.encode()).hexdigest()[:20]
                gif = cache / f'{revision}.gif'
                keep.add(gif.name)
                entry = {'event': event.key, 'detected_at_epoch': event.detected.timestamp(), **timing.resolve(event.video)}
                entry['image'] = f'recent_cache/{gif.name}' if gif.is_file() else None
                entries.append(entry)
                if not gif.is_file():
                    missing.append((event.video, gif))
            # Publish removals from dataset corrections immediately, before rendering.
            atomic_json(args.output_dir / 'recent_meteors.json', {'updated_at_epoch': time.time(),
                        'mask_revision': mask.revision, 'events': entries, 'excluded_events': sorted(seen - accepted),
                        'classification_basis': 'dataset_review_then_prediction'})
            if missing:
                source, output = missing[0]
                raw = convert_raw_zoom_to_gif(source, source.parent, output, args.ffmpeg, mask)
                if raw is None:
                    full, zoom, zoom_mask = fallback_sources(source, source.parent, mask)
                    if mask.mask is not None or '_full' in full.stem:
                        convert_saved_zoom_to_gif(full, source, source.parent, output, args.ffmpeg, zoom_mask)
                    else:
                        convert_to_gif(zoom, output, args.ffmpeg, 180, 180, zoom_mask)
            # Keep two generations briefly while browsers finish a page transition.
            for path in cache.glob('*.gif'):
                if path.name not in keep and time.time() - path.stat().st_mtime > 180:
                    path.unlink(missing_ok=True)
        except ProcessLookupError:
            break
        except Exception as exc:
            print(f'[recent meteors] retry: {type(exc).__name__}', flush=True)
        for _ in range(3):
            if STOP:
                break
            time.sleep(1)


if __name__ == '__main__':
    main()
