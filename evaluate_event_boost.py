"""Compare the deployed EventBoost weights and prior models on identical events."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from event_boost_model import EventBoostClassifier
from event_transformer import EventImageEncoder, ARCHITECTURE_NAME as TRANSFORMER_ARCHITECTURE_NAME
from train_universal_meteor import prepare_event
from train_event_boost import reviewed_events, features_for_event
from train_universal_meteor import EventDataset, evaluate, metrics_for
from universal_meteor_model import MeteorFusionUniversal


def compare(args):
    torch.set_num_threads(4)
    path = Path(args.model)
    metadata = json.loads(Path(str(path) + ".meta.json").read_text())
    manifest = json.loads(path.with_suffix(".evaluation.json").read_text())
    test_records = {e["clip_sha256"]: e for e in manifest if e["split"] == "test"}
    events, _ = reviewed_events(Path(args.data_root))
    events = [e for e in events if e["clip_sha256"] in test_records]
    if len(events) != metadata["test_events"]:
        raise ValueError("Reviewed data changed since training; regenerate the evaluation")
    for event in events:
        if event["label"] != test_records[event["clip_sha256"]]["label"]:
            raise ValueError("Reviewed labels changed since training")
    y = np.array([e["label"] for e in events])
    x = np.stack([features_for_event(e, Path(args.universal_cache), Path(args.feature_cache))
                  for e in events])
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["architecture"] == TRANSFORMER_ARCHITECTURE_NAME:
        encoder = EventImageEncoder(checkpoint["encoder"], "mps" if torch.backends.mps.is_available() else "cpu")
        images = []
        for event in events:
            with np.load(prepare_event(event, Path(args.universal_cache)), allow_pickle=False) as data:
                images.append(data["image"].astype(np.float32) / 65535.)
        embeddings = np.concatenate([encoder.encode(images[start:start + 32]) for start in range(0, len(images), 32)])
        x = np.concatenate([x, embeddings], axis=1)
    portable = EventBoostClassifier(checkpoint)
    p = portable.predict_proba(x)
    recorded = np.array([test_records[e["clip_sha256"]]["probability"] for e in events])
    np.testing.assert_allclose(p, recorded, atol=1e-12, rtol=0)
    report = {"test_events": len(events), "test_nights": metadata["test_nights"], "models": {}}
    probabilities = {path.stem: p}
    report["models"][path.stem] = {
        "metrics": metrics_for(y, p, metadata["decision_threshold"]),
        "weights_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "training_overlap_nights": [],
    }
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    loader = DataLoader(EventDataset(events, Path(args.universal_cache), False),
                        batch_size=32, num_workers=0)
    for name in ("meteor_fusion_universal_v1", "meteor_fusion_universal_v2"):
        old_path = path.parent / f"{name}.pth"
        old_meta = json.loads(Path(str(old_path) + ".meta.json").read_text())
        network = MeteorFusionUniversal().to(device)
        network.load_state_dict(torch.load(old_path, map_location="cpu", weights_only=True))
        labels, prediction = evaluate(network, loader, device)
        np.testing.assert_array_equal(labels, y)
        probabilities[name] = prediction
        report["models"][name] = {
            "metrics": metrics_for(y, prediction, old_meta["decision_threshold"]),
            "weights_sha256": hashlib.sha256(old_path.read_bytes()).hexdigest(),
            "training_overlap_nights": sorted(set(old_meta["training_nights"]) & set(metadata["test_nights"])),
        }
    for name, prediction in probabilities.items():
        model_report = report["models"][name]
        model_report["per_night"] = {}
        for night in metadata["test_nights"]:
            indices = np.array([i for i, e in enumerate(events) if e["night"] == night])
            model_report["per_night"][night] = metrics_for(y[indices], prediction[indices],
                                                                       model_report["metrics"]["threshold"])
    output = path.with_suffix(".comparison.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/meteor_event_transformer_boost_v1.pth")
    parser.add_argument("--data-root", default="ml_training_data/reviewed")
    parser.add_argument("--universal-cache", default="ml_training_data/universal_cache_v1")
    parser.add_argument("--feature-cache", default="ml_training_data/event_boost_cache_v2")
    compare(parser.parse_args())
