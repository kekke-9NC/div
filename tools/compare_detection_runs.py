"""Compare offline legacy detection runs exactly, optionally including media.

Each run directory contains detections.json from the full legacy pipeline.
Output paths and detection counters may differ; event evidence must not.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


FIELDS = ('source_video', 'center', 'probability', 'label', 'timestamp',
          'start_frame', 'end_frame', 'detection_start_frame',
          'detection_end_frame', 'detection_line')


def signature(item):
    return json.dumps({key: item.get(key) for key in FIELDS}, sort_keys=True)


def video_digest(path):
    import cv2
    options = [cv2.CAP_PROP_N_THREADS, 2] if hasattr(cv2, 'CAP_PROP_N_THREADS') else []
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, options)
    if not capture.isOpened():
        raise ValueError(f'Cannot read video: {path}')
    digest, count = hashlib.sha256(), 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            digest.update(str((frame.shape, str(frame.dtype))).encode())
            digest.update(memoryview(frame))
            count += 1
    finally:
        capture.release()
    if not count:
        raise ValueError(f'Empty video: {path}')
    return count, digest.hexdigest()


def compare(before, after, media=False, fixture=None):
    a = json.loads((before / 'detections.json').read_text())
    b = json.loads((after / 'detections.json').read_text())
    groups = []
    for items in (a, b):
        grouped = defaultdict(list)
        for item in items:
            grouped[signature(item)].append(item)
        groups.append(grouped)
    result = {'before_count': len(a), 'after_count': len(b),
              'exact_detections': sorted(map(signature, a)) == sorted(map(signature, b)),
              'exact_artifact_keys': True, 'compared_images': 0,
              'compared_videos': 0, 'compared_video_frames': 0,
              'compared_arrays': 0, 'compared_info': 0, 'media_mismatches': []}
    if result['exact_detections']:
        for key, old_group in groups[0].items():
            for old, new in zip(old_group, groups[1][key]):
                if set(old['saved_paths']) != set(new['saved_paths']):
                    result['exact_artifact_keys'] = False
                if not media:
                    continue
                for kind in set(old['saved_paths']) & set(new['saved_paths']):
                    left, right = Path(old['saved_paths'][kind]), Path(new['saved_paths'][kind])
                    if left.is_dir() and right.is_dir():
                        pairs = [(file, right / file.relative_to(left))
                                 for file in left.rglob('*') if file.is_file()]
                    else:
                        pairs = [(left, right)]
                    for p, q in pairs:
                        suffix = p.suffix.lower()
                        if suffix in ('.png', '.jpg', '.jpeg', '.tiff', '.npy', '.npz'):
                            exact = p.read_bytes() == q.read_bytes()
                            counter = 'compared_arrays' if suffix in ('.npy', '.npz') else 'compared_images'
                        elif suffix == '.mp4':
                            frames, digest = video_digest(p)
                            new_frames, new_digest = video_digest(q)
                            exact = (frames, digest) == (new_frames, new_digest)
                            counter = 'compared_videos'
                            result['compared_video_frames'] += frames
                        elif kind == 'info' and suffix == '.txt':
                            # These lines contain the deliberately isolated output destinations.
                            normalize = lambda path: [line for line in path.read_text().splitlines()
                                                       if not line.startswith('Saved ')]
                            exact = normalize(p) == normalize(q)
                            counter = 'compared_info'
                        else:
                            continue
                        result[counter] += 1
                        if not exact:
                            result['media_mismatches'].append({'kind': kind, 'before': str(p), 'after': str(q)})
    if fixture:
        from tools.validate_short_flash_recall import match_events
        events = json.loads(fixture.read_text())
        detections = [dict(item, minute=int(Path(item['source_video']).stem)) for item in b]
        result['recall'] = match_events(events, detections)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--media', action='store_true')
    parser.add_argument('--fixture', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = compare(args.before, args.after, args.media, args.fixture)
    payload = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload)
    return 0 if (result['exact_detections'] and result['exact_artifact_keys']
                 and not result['media_mismatches']
                 and not result.get('recall', {}).get('missing')) else 1


if __name__ == '__main__':
    raise SystemExit(main())
