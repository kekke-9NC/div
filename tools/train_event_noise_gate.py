"""Train the lightweight event filter on reviewed chronological clips.

The full classifier supplies the capture-date split. Native reviewed inputs
are preferred over their encoded previews. The 29 confirmed events are training
and regression examples; separate validation/test dates remain independent.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from event_boost_model import EventBoostClassifier, export_boosted_trees
from event_noise_gate import GATE_PREPROCESS_VERSION, extract_noise_gate_features
from train_event_boost import reviewed_events
from train_universal_meteor import metrics_for, read_clip


def gate_vector(event, cache):
    directory = Path(event['directory'])
    metadata = event['metadata']
    identity = event['clip_sha256'] + metadata.get('native_gate_frames_sha256', '')
    destination = cache / (hashlib.sha256(identity.encode()).hexdigest() + '.npy')
    native = metadata.get('native_gate_frames_npz')
    if native:
        path = (directory / native).resolve()
        if path.parent != directory.resolve():
            raise ValueError('Native gate frames must be inside their reviewed bundle')
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata.get('native_gate_frames_sha256'):
            raise ValueError(f'Native gate checksum mismatch: {path}')
    if destination.exists():
        return np.load(destination, allow_pickle=False)
    if native:
        with np.load(path, allow_pickle=False) as archive:
            frames = list(archive['frames'])
    else:
        frames = read_clip(directory / 'clip.mp4')
    vector = extract_noise_gate_features(frames, frame_rate=event['frame_rate'])
    np.save(destination, vector, allow_pickle=False)
    return vector


def clean_json(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    return value


def train(args):
    full = Path(args.full_model)
    records = {row['clip_sha256']: row for row in json.loads(full.with_suffix('.evaluation.json').read_text())}
    metadata = json.loads(Path(str(full) + '.meta.json').read_text())
    full_threshold = metadata['decision_threshold']
    events, _ = reviewed_events(Path(args.reviewed_root))
    # Newly reviewed cloud negatives must enter training too. Restricting
    # training negatives to old auxiliary examples ignores later corrections.
    by_hash = {event['clip_sha256']: event for event in events
               if event['clip_sha256'] in records}
    for root in args.extra_reviewed_root:
        extra, _ = reviewed_events(Path(root))
        for event in extra:
            previous = by_hash.get(event['clip_sha256'])
            if previous is not None and previous['label'] != event['label']:
                raise ValueError('Conflicting reviewed filter labels')
            if event['clip_sha256'] not in records and event['night'] not in metadata['training_nights']:
                raise ValueError('New examples must be from the declared training dates')
            by_hash[event['clip_sha256']] = event
    events = list(by_hash.values())
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        x = np.stack(list(pool.map(lambda event: gate_vector(event, cache), events)))
    y = np.array([event['label'] for event in events])
    groups = np.array([records.get(event['clip_sha256'], {}).get('split', 'training') for event in events])
    train_ids = np.flatnonzero(groups == 'training')
    validation_ids = np.flatnonzero(groups == 'validation')
    test_ids = np.flatnonzero(groups == 'test')
    accepted_validation = [i for i in validation_ids if y[i] == 1
                           and records[events[i]['clip_sha256']]['probability'] >= full_threshold]
    if not accepted_validation or len(np.unique(y[train_ids])) != 2:
        raise ValueError('Both training classes and accepted validation meteors are required')
    # Give reviewed native corrections equal weight on both sides, including
    # the camera's faint confirmed meteors, not just its background examples.
    weights = np.array([30. if events[i]['metadata'].get('native_gate_frames_npz') else 1.
                        for i in train_ids])
    candidates, checkpoints = [], []
    with threadpool_limits(limits=args.threads):
        for leaves, iterations in ((15, 200), (15, 400), (31, 300)):
            estimator = HistGradientBoostingClassifier(
                max_leaf_nodes=leaves, max_iter=iterations, min_samples_leaf=3,
                l2_regularization=3., learning_rate=.06, early_stopping=False,
                random_state=20261011,
            )
            estimator.fit(x[train_ids], y[train_ids], sample_weight=weights)
            probabilities = estimator.predict_proba(x)[:, 1]
            # Preserve training meteors and the full model's validation true
            # positives, with a 10% score margin. Test scores do not set this.
            calibration = np.r_[probabilities[train_ids][y[train_ids] == 1],
                                probabilities[accepted_validation]]
            threshold = min(.95, float(calibration.min()) * .9)
            checkpoint = export_boosted_trees([estimator], x.shape[1])
            checkpoint.update(gate_preprocess_version=GATE_PREPROCESS_VERSION,
                              gate_threshold=threshold, gate_brightness_min=None)
            actual = EventBoostClassifier(checkpoint).predict_proba(x)
            np.testing.assert_allclose(actual, probabilities, atol=1e-10, rtol=0)
            row = {'leaves': leaves, 'iterations': iterations, 'threshold': threshold}
            for name, ids in (('training', train_ids), ('validation', validation_ids), ('test', test_ids)):
                row[name] = metrics_for(y[ids], actual[ids], threshold)
            row['reviewed_negatives_rejected'] = int(sum(actual[y == 0] < threshold))
            candidates.append(row)
            checkpoints.append(checkpoint)
            print(json.dumps(clean_json(row)), flush=True)
    # Sensitivity is fixed by calibration; validation negatives decide how
    # much non-meteor work can safely be rejected. Test metrics are excluded.
    selected = max(range(len(candidates)), key=lambda i: (
        candidates[i]['validation']['tn'], -candidates[i]['iterations']))
    checkpoint = checkpoints[selected]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    report = {
        'created_at': datetime.now(timezone.utc).isoformat(), 'reviewed_events': len(events),
        'selected': selected, 'candidates': candidates, 'gate_threshold': checkpoint['gate_threshold'],
        'gate_preprocess_version': GATE_PREPROCESS_VERSION, 'brightness_min': None,
        'weights_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'full_model_sha256': hashlib.sha256(full.read_bytes()).hexdigest(),
        'split_source': str(full.with_suffix('.evaluation.json')), 'full_model_threshold': full_threshold,
        'dataset_sha256': hashlib.sha256(json.dumps([
            [event['clip_sha256'], event['label'], event['night'], event['frame_rate'],
             event['metadata'].get('native_gate_frames_sha256')] for event in events
        ]).encode()).hexdigest(),
        'extra_reviewed_roots': args.extra_reviewed_root,
        'selection_policy': 'maximum validation non-meteor rejection after preserving training meteors and validation full-model true positives with 0.9 score margin; held-out test not consulted',
        'confirmed29_role': 'native training and regression examples; not independent test data',
    }
    Path(str(output) + '.meta.json').write_text(json.dumps(clean_json(report), indent=2, allow_nan=False))
    print('SAVED', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-model', default='models/meteor_event_transformer_boost_v2.pth')
    parser.add_argument('--reviewed-root', default='ml_training_data/reviewed')
    parser.add_argument('--extra-reviewed-root', action='append', required=True)
    parser.add_argument('--cache', default='ml_training_data/event_noise_gate_cache_v1')
    parser.add_argument('--output', default='models/auxiliary/meteor_event_noise_gate_v1.pth')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    train(parser.parse_args())
