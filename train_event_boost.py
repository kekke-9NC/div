"""Reproducible supervised EventBoost training with separate validation/test nights.

Example: .venv-mac/bin/python train_event_boost.py
Only explicitly reviewed labels are used. Pending/skipped predictions are never
treated as ground truth. The exported deployment weights are the exact model
evaluated on the untouched test set, with no refit on validation or test data.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from event_boost_model import (ARCHITECTURE_NAME, PREPROCESS_VERSION,
                               EventBoostClassifier, export_boosted_trees,
                               extract_event_features, extract_motion_features)
from train_universal_meteor import scan_events, prepare_event, metrics_for, read_clip
from event_transformer import (EventImageEncoder, ENCODER_ID, ENCODER_REVISION,
                               ARCHITECTURE_NAME as TRANSFORMER_ARCHITECTURE_NAME)


def reviewed_events(root):
    events, seen, duplicates = [], {}, []
    for event in scan_events(root):
        metadata = event["metadata"]
        if (metadata.get("review_status") != "reviewed"
                or metadata.get("reviewed_label") != event["label_name"]):
            raise ValueError(f"Unconfirmed or inconsistent label: {event['directory']}")
        digest = hashlib.sha256((Path(event["directory"]) / "clip.mp4").read_bytes()).hexdigest()
        if digest in seen:
            previous = seen[digest]
            if previous["label"] != event["label"]:
                raise ValueError(f"Conflicting duplicate labels: {event['directory']}")
            duplicates.append(event["directory"])
            continue
        event["clip_sha256"] = digest
        seen[digest] = event
        events.append(event)
    return events, duplicates


def representation_path(event, universal_cache):
    """Use recorded native inference inputs when present, otherwise the reviewed clip."""
    name = event["metadata"].get("native_inputs_npz")
    if name:
        path = Path(event["directory"]) / name
        if path.parent.resolve() != Path(event["directory"]).resolve():
            raise ValueError("Native input must be inside its reviewed event bundle")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != event["metadata"].get("native_inputs_sha256"):
            raise ValueError(f"Native input checksum mismatch: {path}")
        return path
    return prepare_event(event, universal_cache)


def embedding_identity(event):
    return event["metadata"].get("native_inputs_sha256", event["clip_sha256"])


def features_for_event(event, universal_cache, feature_cache):
    # Include input geometry/FPS and representation identity in the cache key.
    identity = json.dumps([PREPROCESS_VERSION, event["clip_sha256"],
                           event["detected_line"], event["metadata"].get("cutout_rect"),
                           event["frame_rate"], event["metadata"].get("native_inputs_sha256")], sort_keys=True)
    key = hashlib.sha256(identity.encode()).hexdigest()
    destination = feature_cache / f"{key}.npy"
    if destination.is_file():
        return np.load(destination, allow_pickle=False)
    path = representation_path(event, universal_cache)
    with np.load(path, allow_pickle=False) as data:
        representation = extract_event_features(data["image"].astype(np.float32) / 65535.,
                                        data["kymograph"].astype(np.float32) / 65535.,
                                        data["features"])
        motion = data["motion_features"] if "motion_features" in data else None
    if motion is None:
        motion = extract_motion_features(read_clip(Path(event["directory"]) / "clip.mp4"),
                                        frame_rate=event["frame_rate"])
    vector = np.concatenate([representation, motion])
    feature_cache.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.npy")
    np.save(temporary, vector, allow_pickle=False)
    os.replace(temporary, destination)
    return vector


def choose_validation_split(events, seed):
    labels = np.array([e["label"] for e in events])
    nights = np.array([e["night"] for e in events])
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    candidates = list(folds.split(np.zeros(len(events)), labels, nights))
    # Selection uses only labels/group sizes, never any model performance.
    return min(candidates, key=lambda pair: abs(labels[pair[1]].mean() - labels.mean())
               + abs(len(pair[1]) / len(events) - .2))


def choose_threshold(labels, probabilities, maximum_threshold=None):
    candidates = np.unique(np.r_[.5, probabilities])
    if maximum_threshold is not None:
        candidates = np.unique(np.r_[candidates[candidates <= maximum_threshold], maximum_threshold])
    # Maximize validation F1, then recall; the test set is not consulted.
    return max(candidates, key=lambda t: (
        2 * np.sum((probabilities >= t) & (labels == 1)) / max(
            1, np.sum(probabilities >= t) + np.sum(labels == 1)),
        np.sum((probabilities >= t) & (labels == 1)), float(t)))


def train(args):
    if not np.isfinite(args.correction_weight) or args.correction_weight <= 0:
        raise ValueError("correction_weight must be positive and finite")
    started = time.monotonic()
    events, duplicates = reviewed_events(Path(args.data_root))
    by_hash = {event["clip_sha256"]: event for event in events}
    for root in args.extra_reviewed_root:
        extra, extra_duplicates = reviewed_events(Path(root))
        duplicates.extend(extra_duplicates)
        for event in extra:
            previous = by_hash.get(event["clip_sha256"])
            if previous is not None:
                if previous["label"] != event["label"]:
                    raise ValueError(f"Conflicting duplicate labels: {event['directory']}")
                duplicates.append(event["directory"])
                continue
            by_hash[event["clip_sha256"]] = event
            events.append(event)
    print(f"Confirmed events: {len(events)}; duplicates excluded: {len(duplicates)}", flush=True)
    feature_cache = Path(args.feature_cache)
    universal_cache = Path(args.universal_cache)
    features = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        vectors = pool.map(lambda e: features_for_event(e, universal_cache, feature_cache), events)
        for index, vector in enumerate(vectors, 1):
            features.append(vector)
            if index % 200 == 0 or index == len(events):
                print(f"features {index}/{len(events)}; elapsed {time.monotonic()-started:.1f}s", flush=True)
    x = np.stack(features)
    encoder_checkpoint = None
    if not args.no_image_encoder:
        if not Path(args.encoder_checkpoint).is_file():
            from transformers import Dinov2Model
            pretrained = Dinov2Model.from_pretrained(
                ENCODER_ID, revision=ENCODER_REVISION, use_safetensors=True)
            encoder_path = Path(args.encoder_checkpoint)
            encoder_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"config": pretrained.config.to_dict(),
                        "state_dict": pretrained.state_dict()}, encoder_path)
            del pretrained
        encoder_checkpoint = torch.load(args.encoder_checkpoint, map_location="cpu", weights_only=True)
        torch.set_num_threads(args.threads)
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        encoder = EventImageEncoder(encoder_checkpoint, device)
        embedding_root = Path(args.feature_cache) / "dinov2_v1"
        embedding_root.mkdir(parents=True, exist_ok=True)
        embeddings = []
        for start in range(0, len(events), 32):
            batch = events[start:start + 32]
            paths = [embedding_root / (embedding_identity(e) + ".npy") for e in batch]
            missing = [i for i, path in enumerate(paths) if not path.is_file()]
            if missing:
                images = []
                for i in missing:
                    with np.load(representation_path(batch[i], universal_cache), allow_pickle=False) as data:
                        images.append(data["image"].astype(np.float32) / 65535.)
                vectors = encoder.encode(images)
                for i, vector in zip(missing, vectors):
                    np.save(paths[i], vector, allow_pickle=False)
            embeddings.extend(np.load(path, allow_pickle=False) for path in paths)
            if start % 320 == 0:
                print(f"Transformer embeddings {start + len(batch)}/{len(events)}", flush=True)
        x = np.concatenate([x, np.stack(embeddings)], axis=1)
        del encoder
    y = np.array([e["label"] for e in events])
    test_nights = set(args.test_nights or [])
    is_test = lambda e: e["night"] in test_nights if test_nights else e["night"] >= args.test_from
    development = np.array([i for i, e in enumerate(events) if not is_test(e)])
    test = np.array([i for i, e in enumerate(events) if is_test(e)])
    if not len(test) or not len(development):
        raise ValueError("Test cutoff must produce nonempty development and test sets")
    validation_pool = np.array([i for i in development
                               if not args.validation_before or events[i]["night"] < args.validation_before])
    _, val_local = choose_validation_split([events[i] for i in validation_pool], args.seed)
    validation = validation_pool[val_local]
    training = np.array(sorted(set(development) - set(validation)))
    preserved = np.zeros(len(events), dtype=bool)
    preservation = None
    if args.preserve_model:
        previous_path = Path(args.preserve_model)
        previous_meta = json.loads(Path(str(previous_path) + '.meta.json').read_text())
        previous = EventBoostClassifier(torch.load(previous_path, map_location='cpu', weights_only=True))
        previous_probabilities = previous.predict_proba(x)
        # Only reviewed positives from development enter calibration. Held-out
        # test scores never choose an operating threshold or a candidate.
        preserved[development] = (y[development] == 1) & (
            previous_probabilities[development] >= previous_meta['decision_threshold'])
        preservation = dict(model_sha256=hashlib.sha256(previous_path.read_bytes()).hexdigest(),
                            threshold=previous_meta['decision_threshold'],
                            development_positives=int(preserved.sum()))
        print('Preserve baseline reviewed meteors', preservation, flush=True)
    def calibrated_threshold(estimator_probabilities):
        bound = float(estimator_probabilities[preserved].min()) * .99 if preserved.any() else None
        return float(choose_threshold(y[validation], estimator_probabilities[validation], bound))
    for name, indices in (("training", training), ("validation", validation), ("test", test)):
        if len(np.unique(y[indices])) != 2:
            raise ValueError(f"{name} must contain both classes")
        print(name, len(indices), dict(Counter(y[indices].tolist())), flush=True)
    # Fixed small search, entirely on the validation nights.
    candidates, estimators = [], []
    with threadpool_limits(limits=args.threads):
        for leaves, iterations, regularization in ((15, 300, 3.), (15, 600, 8.),
                                                  (31, 400, 5.), (7, 600, 3.)):
            estimator = HistGradientBoostingClassifier(
                max_iter=iterations, max_leaf_nodes=leaves, learning_rate=.05,
                l2_regularization=regularization, min_samples_leaf=15,
                early_stopping=False, random_state=args.seed)
            weights = np.array([args.correction_weight if events[i]["metadata"].get("was_misclassified") else 1.
                                for i in training])
            estimator.fit(x[training], y[training], sample_weight=weights)
            all_probabilities = estimator.predict_proba(x)[:, 1]
            probabilities = all_probabilities[validation]
            threshold = calibrated_threshold(all_probabilities)
            metrics = metrics_for(y[validation], probabilities, threshold)
            estimators.append(estimator)
            candidates.append({"parameters": {"leaves": leaves, "iterations": iterations,
                                               "regularization": regularization},
                               "validation_metrics": metrics})
            print("candidate", json.dumps(candidates[-1]), flush=True)
        # Also consider averaging the fixed candidates, selected by validation only.
        ensemble_all = np.mean([e.predict_proba(x)[:, 1] for e in estimators], axis=0)
        ensemble_p = ensemble_all[validation]
        ensemble_t = calibrated_threshold(ensemble_all)
        candidates.append({"parameters": {"ensemble": list(range(len(estimators)))},
                           "validation_metrics": metrics_for(y[validation], ensemble_p, ensemble_t)})
        linear_estimators = []
        if encoder_checkpoint:
            for c in (.001, .01, .1, 1.):
                estimator = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1000))
                estimator.fit(x[training], y[training], logisticregression__sample_weight=weights)
                all_probabilities = estimator.predict_proba(x)[:, 1]
                probabilities = all_probabilities[validation]
                threshold = calibrated_threshold(all_probabilities)
                candidates.append({"parameters": {"linear_C": c},
                                   "validation_metrics": metrics_for(y[validation], probabilities, threshold)})
                linear_estimators.append(estimator)
                print("candidate", json.dumps(candidates[-1]), flush=True)
        selected = max(range(len(candidates)), key=lambda i: (
            candidates[i]["validation_metrics"]["f1"],
            candidates[i]["validation_metrics"]["pr_auc"]))
        linear_chosen = []
        if selected > len(estimators):
            chosen = []
            linear_chosen = [linear_estimators[selected - len(estimators) - 1]]
        else:
            chosen = estimators if selected == len(estimators) else [estimators[selected]]
        checkpoint = export_boosted_trees(chosen, x.shape[1])
        if encoder_checkpoint:
            checkpoint["architecture"] = TRANSFORMER_ARCHITECTURE_NAME
            checkpoint["encoder"] = encoder_checkpoint
            checkpoint["linear_members"] = []
            for estimator in linear_chosen:
                scaler, linear = estimator.steps[0][1], estimator.steps[1][1]
                checkpoint["linear_members"].append({
                    "mean": torch.from_numpy(scaler.mean_), "scale": torch.from_numpy(scaler.scale_),
                    "coef": torch.from_numpy(linear.coef_[0]), "intercept": float(linear.intercept_[0])})
        portable = EventBoostClassifier(checkpoint)
        # Export must preserve predictions before we write a deployment artifact.
        expected = np.mean([e.predict_proba(x)[:, 1] for e in chosen + linear_chosen], axis=0)
        actual = portable.predict_proba(x)
        export_error = float(np.max(np.abs(expected - actual)))
        if export_error > (2e-6 if linear_chosen else 1e-10):
            raise RuntimeError(f"Export prediction mismatch: {export_error}")
        threshold = candidates[selected]["validation_metrics"]["threshold"]
        if preserved.any() and np.any(actual[preserved] < threshold):
            raise RuntimeError('Export lost a baseline reviewed development meteor')
        test_metrics = metrics_for(y[test], actual[test], threshold)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    report = {
        "architecture": checkpoint["architecture"], "preprocess_version": PREPROCESS_VERSION,
        "feature_count": int(x.shape[1]), "class_names": ["not_meteor", "meteor"],
        "meteor_class_index": 1, "decision_threshold": float(threshold),
        "input_resize": [160, 160], "mean": [0., 0., 0.], "std": [1., 1., 1.],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_events": len(training), "validation_events": len(validation),
        "test_events": len(test), "reviewed_events": len(events),
        "duplicate_events_excluded": duplicates, "seed": args.seed,
        "split_policy": "disjoint_capture_dates; validation_only_model_and_threshold_selection; no_refit",
        "test_from": args.test_from, "selected_candidate": selected,
        "explicit_test_nights": sorted(test_nights), "validation_before": args.validation_before,
        "extra_reviewed_roots": args.extra_reviewed_root,
        "correction_weight": args.correction_weight,
        "preserved_baseline": preservation,
        "candidates": candidates, "validation_metrics": metrics_for(y[validation], actual[validation], threshold),
        "test_metrics": test_metrics, "export_max_absolute_error": export_error,
        "camera_policy": "camera_date_source_previous_prediction_excluded_from_features",
        "trained_on_all_reviewed_events": False,
        "pretrained_encoder": ENCODER_ID if encoder_checkpoint else None,
        "pretrained_encoder_revision": ENCODER_REVISION if encoder_checkpoint else None,
        "dataset_sha256": hashlib.sha256(json.dumps([
            [e["clip_sha256"], e["label"], e["night"], e["frame_rate"],
             e.get("detected_line"), e["metadata"].get("native_inputs_sha256")]
            for e in events
        ], sort_keys=True).encode()).hexdigest(),
    }
    for name, indices in (("training", training), ("validation", validation), ("test", test)):
        report[f"{name}_nights"] = sorted({events[i]["night"] for i in indices})
    Path(f"{output}.meta.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = []
    for name, indices in (("training", training), ("validation", validation), ("test", test)):
        for i in indices:
            manifest.append({"event": Path(events[i]["directory"]).name,
                             "clip_sha256": events[i]["clip_sha256"], "night": events[i]["night"],
                             "label": int(y[i]), "split": name, "probability": float(actual[i])})
    output.with_suffix(".evaluation.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    print("SAVED", str(output), "test", json.dumps(test_metrics), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="ml_training_data/reviewed")
    parser.add_argument("--extra-reviewed-root", action="append", default=[], help="Additional explicitly reviewed event bundles")
    parser.add_argument("--universal-cache", default="ml_training_data/universal_cache_v1")
    parser.add_argument("--feature-cache", default="ml_training_data/event_boost_cache_v2")
    parser.add_argument("--output", default="models/meteor_event_transformer_boost_v1.pth")
    parser.add_argument("--test-from", default="2026-10-08")
    parser.add_argument("--test-nights", nargs="+", help="Explicit held-out capture dates; other dates enter development")
    parser.add_argument("--validation-before", help="Keep recently reviewed dates in training, not validation")
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--correction-weight", type=float, default=1., help="Weight explicitly corrected human labels more heavily")
    parser.add_argument('--preserve-model', help='Preserve this model\'s reviewed development true positives while calibrating; held-out test excluded')
    parser.add_argument("--encoder-checkpoint", default="ml_training_data/dinov2_small_encoder.pth")
    parser.add_argument("--no-image-encoder", action="store_true",
                        help="Diagnostic ablation: train without the pretrained image Transformer")
    train(parser.parse_args())
