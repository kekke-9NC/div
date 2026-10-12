"""Compare a reviewed-event classifier cascade with its previous release."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from event_boost_model import EventBoostClassifier
from train_event_boost import reviewed_events, features_for_event, embedding_identity
from train_universal_meteor import metrics_for
from tools.train_event_noise_gate import gate_vector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-model', type=Path, required=True)
    parser.add_argument('--baseline-model', type=Path, required=True)
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--reviewed-root', type=Path, default=Path('ml_training_data/reviewed'))
    parser.add_argument('--extra-reviewed-root', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.full_model.with_suffix('.evaluation.json').read_text())
    meta = json.loads(Path(str(args.full_model) + '.meta.json').read_text())
    old_meta = json.loads(Path(str(args.baseline_model) + '.meta.json').read_text())
    events = {}
    for root in [args.reviewed_root, *args.extra_reviewed_root]:
        reviewed, _ = reviewed_events(root)
        for event in reviewed:
            if event['clip_sha256'] in events and events[event['clip_sha256']]['label'] != event['label']:
                raise ValueError('Conflicting reviewed labels')
            events.setdefault(event['clip_sha256'], event)
    ordered = [events[row['clip_sha256']] for row in records]
    if any(event['label'] != row['label'] for event, row in zip(ordered, records)):
        raise ValueError('Reviewed labels changed after training')
    cache = Path('ml_training_data/event_boost_cache_v2')
    universal = Path('ml_training_data/universal_cache_v1')
    def features(event):
        base = features_for_event(event, universal, cache)
        embedding = np.load(cache / 'dinov2_v1' / (embedding_identity(event) + '.npy'), allow_pickle=False)
        return np.r_[base, embedding]
    with ThreadPoolExecutor(max_workers=4) as pool:
        x = np.stack(list(pool.map(features, ordered)))
        gate_x = np.stack(list(pool.map(lambda e: gate_vector(e, Path('ml_training_data/event_noise_gate_cache_v1')), ordered)))
    old = EventBoostClassifier(torch.load(args.baseline_model, weights_only=True, map_location='cpu')).predict_proba(x)
    gate = torch.load(args.gate, weights_only=True, map_location='cpu')
    gate_p = EventBoostClassifier(gate).predict_proba(gate_x)
    full = np.array([row['probability'] for row in records])
    filtered = np.where(gate_p < gate['gate_threshold'], 0., full)
    y = np.array([row['label'] for row in records])
    old_records = json.loads(args.baseline_model.with_suffix('.evaluation.json').read_text())
    old_hashes = {row['clip_sha256'] for row in old_records}
    report = dict(gate_threshold=gate['gate_threshold'],
                  model_sha256=hashlib.sha256(args.full_model.read_bytes()).hexdigest(),
                  gate_sha256=hashlib.sha256(args.gate.read_bytes()).hexdigest(),
                  baseline_sha256=hashlib.sha256(args.baseline_model.read_bytes()).hexdigest())
    groups = {split: np.array([row['split'] == split for row in records]) for split in ('training', 'validation', 'test')}
    groups['newly_reviewed'] = np.array([row['clip_sha256'] not in old_hashes for row in records])
    for name, mask in groups.items():
        report[name] = dict(
            events=int(mask.sum()), baseline=metrics_for(y[mask], old[mask], old_meta['decision_threshold']),
            full=metrics_for(y[mask], full[mask], meta['decision_threshold']),
            cascade=metrics_for(y[mask], filtered[mask], meta['decision_threshold']),
            baseline_true_positives_lost=int(np.sum(mask & (y == 1) & (old >= old_meta['decision_threshold']) & (filtered < meta['decision_threshold']))),
            full_true_positives_lost_by_gate=int(np.sum(mask & (y == 1) & (full >= meta['decision_threshold']) & (filtered < meta['decision_threshold']))),
            background_rejected=int(np.sum(mask & (y == 0) & (gate_p < gate['gate_threshold']))))
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
