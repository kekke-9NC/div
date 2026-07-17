"""Train MeteorFusionUniversal on reviewed ML event bundles."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

from universal_meteor_model import (
    ARCHITECTURE_NAME,
    FEATURE_COUNT,
    IMAGE_SIZE,
    KYMO_HEIGHT,
    KYMO_WIDTH,
    PREPROCESS_VERSION,
    MeteorFusionUniversal,
    augment_universal_inputs,
    build_universal_inputs,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def scan_events(root: Path) -> List[Dict]:
    events = []
    for label_name, label in (("not_meteor", 0), ("meteor", 1)):
        for directory in sorted((root / label_name).iterdir()):
            metadata_path = directory / "metadata.json"
            clip_path = directory / "clip.mp4"
            if not directory.is_dir() or not metadata_path.exists() or not clip_path.exists():
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            events.append(
                {
                    "directory": str(directory),
                    "label": label,
                    "label_name": label_name,
                    "source": str(metadata.get("source", directory.name)),
                    "night": str(metadata.get("detection_time", directory.name))[:10],
                    "frame_rate": float(metadata.get("frame_rate") or 15.0),
                    "detected_line": metadata.get("detected_line"),
                    "metadata": metadata,
                }
            )
    return events


def cache_path(cache_root: Path, event: Dict) -> Path:
    identity = f"{PREPROCESS_VERSION}:{event['directory']}"
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()
    return cache_root / digest[:2] / f"{digest}.npz"


def read_clip(path: Path) -> List[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_N_THREADS, 1)
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    finally:
        capture.release()
    if not frames:
        raise IOError(f"Could not decode {path}")
    return frames


def prepare_event(event: Dict, cache_root: Path) -> Path:
    destination = cache_path(cache_root, event)
    if destination.exists():
        return destination
    directory = Path(event["directory"])
    frames = read_clip(directory / "clip.mp4")
    image, kymograph, features = build_universal_inputs(
        frames,
        rect=None,
        detected_line=_local_line(event["detected_line"], event["metadata"]),
        frame_rate=event["frame_rate"],
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        image=(np.clip(image, 0.0, 1.0) * 65535.0).astype(np.uint16),
        kymograph=(np.clip(kymograph, 0.0, 1.0) * 65535.0).astype(np.uint16),
        features=features.astype(np.float32),
    )
    os.replace(temporary, destination)
    return destination


def _local_line(detected_line, metadata):
    if not detected_line:
        return None
    rect = metadata.get("cutout_rect")
    if not rect or len(rect) != 4:
        return detected_line
    x1, y1, x2, y2 = (float(v) for v in rect)
    sx = 256.0 / max(1.0, x2 - x1)
    sy = 256.0 / max(1.0, y2 - y1)
    return [
        [(float(point[0]) - x1) * sx, (float(point[1]) - y1) * sy]
        for point in detected_line
    ]


class EventDataset(Dataset):
    def __init__(self, events: Sequence[Dict], cache_root: Path, augment: bool):
        self.events = list(events)
        self.cache_root = cache_root
        self.augment = augment

    def __len__(self):
        return len(self.events)

    def __getitem__(self, index):
        event = self.events[index]
        path = prepare_event(event, self.cache_root)
        with np.load(path) as data:
            image = torch.from_numpy(data["image"].astype(np.float32) / 65535.0)
            kymograph = torch.from_numpy(data["kymograph"].astype(np.float32) / 65535.0)
            features = torch.from_numpy(data["features"].astype(np.float32))
        if self.augment:
            image, kymograph, features = augment_universal_inputs(
                image, kymograph, features
            )
        return image, kymograph, features, torch.tensor(event["label"], dtype=torch.float32)


def choose_split(events: Sequence[Dict], seed: int, holdout_year: str = ""):
    if holdout_year:
        train_indices = [
            index
            for index, event in enumerate(events)
            if not event["night"].startswith(holdout_year)
        ]
        validation_indices = [
            index
            for index, event in enumerate(events)
            if event["night"].startswith(holdout_year)
        ]
        if not train_indices or not validation_indices:
            raise ValueError(f"holdout year has no usable split: {holdout_year}")
        return np.asarray(train_indices), np.asarray(validation_indices)
    labels = np.asarray([event["label"] for event in events])
    groups = np.asarray([event["night"] for event in events])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    candidates = list(splitter.split(np.zeros(len(events)), labels, groups))
    target = float(labels.mean())
    train_indices, validation_indices = min(
        candidates,
        key=lambda pair: abs(float(labels[pair[1]].mean()) - target)
        + abs(len(pair[1]) / len(events) - 0.2),
    )
    return train_indices, validation_indices


def metrics_for(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> Dict:
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / max(1, len(labels))),
        "precision": float(tp / max(1, tp + fp)),
        "recall": float(tp / max(1, tp + fn)),
        "specificity": float(tn / max(1, tn + fp)),
        "f1": float(2 * tp / max(1, 2 * tp + fp + fn)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "tp": int(tp),
        "fn": int(fn),
        "fp": int(fp),
        "tn": int(tn),
    }


def threshold_for_recall(
    labels: np.ndarray, probabilities: np.ndarray, target_recall: float
) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    valid = np.where(recall[:-1] >= target_recall)[0]
    if not len(valid):
        return 0.5
    best = valid[np.argmax(precision[:-1][valid])]
    return float(thresholds[best])


def evaluate(model, loader, device):
    model.eval()
    labels, probabilities = [], []
    with torch.inference_mode():
        for image, kymograph, features, target in loader:
            logits = model(
                image.to(device),
                kymograph.to(device),
                features.to(device),
            )
            probabilities.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            labels.extend(target.numpy().astype(np.int64).tolist())
    return np.asarray(labels), np.asarray(probabilities)


def train(args):
    set_seed(args.seed)
    torch.set_num_threads(max(1, min(args.threads, os.cpu_count() or 1)))
    device = select_device(args.device)
    root = Path(args.data_root)
    cache_root = Path(args.cache_root)
    output = Path(args.output)
    events = scan_events(root)
    if not events:
        raise RuntimeError(f"No reviewed events under {root}")
    print(f"device={device} events={len(events)} labels={Counter(e['label_name'] for e in events)}")

    print("Preparing camera-normalized cache...")
    started = time.time()
    for index, event in enumerate(events, 1):
        prepare_event(event, cache_root)
        if index % 100 == 0 or index == len(events):
            print(f"cache {index}/{len(events)} elapsed={time.time() - started:.1f}s")

    if args.train_all:
        train_indices = np.arange(len(events))
        validation_indices = np.asarray([], dtype=np.int64)
    else:
        train_indices, validation_indices = choose_split(
            events, args.seed, holdout_year=args.holdout_year
        )
    training = [events[index] for index in train_indices]
    validation = [events[index] for index in validation_indices]
    print(
        "split",
        f"train={len(training)} {Counter(e['label_name'] for e in training)} nights={len(set(e['night'] for e in training))}",
        f"validation={len(validation)} {Counter(e['label_name'] for e in validation)} nights={len(set(e['night'] for e in validation))}",
    )

    train_loader = DataLoader(
        EventDataset(training, cache_root, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    validation_loader = None
    if validation:
        validation_loader = DataLoader(
            EventDataset(validation, cache_root, augment=False),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            persistent_workers=args.workers > 0,
        )

    model = MeteorFusionUniversal().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"parameters={parameter_count:,}")
    positive = sum(event["label"] for event in training)
    negative = len(training) - positive
    positive_weight = torch.tensor(
        [negative / max(1, positive)], dtype=torch.float32, device=device
    )
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.03
    )

    best_state = None
    best_pr_auc = -1.0
    best_epoch = 0
    patience = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0
        epoch_started = time.time()
        for image, kymograph, features, target in train_loader:
            image = image.to(device)
            kymograph = kymograph.to(device)
            features = features.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(image, kymograph, features)
            loss = criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(target)
            count += len(target)
        scheduler.step()

        row = {
            "epoch": epoch,
            "loss": total_loss / max(1, count),
            "seconds": time.time() - epoch_started,
        }
        if validation_loader is not None:
            labels, probabilities = evaluate(model, validation_loader, device)
            default_metrics = metrics_for(labels, probabilities, 0.5)
            tuned_threshold = threshold_for_recall(
                labels, probabilities, args.target_recall
            )
            tuned_metrics = metrics_for(labels, probabilities, tuned_threshold)
            row["default"] = default_metrics
            row["target_recall"] = tuned_metrics
        history.append(row)
        if validation_loader is None:
            print(
                f"epoch={epoch:02d} loss={row['loss']:.4f} "
                f"sec={row['seconds']:.1f}"
            )
            best_epoch = epoch
            best_state = copy.deepcopy(
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
            )
        else:
            print(
                f"epoch={epoch:02d} loss={row['loss']:.4f} "
                f"pr_auc={default_metrics['pr_auc']:.4f} auc={default_metrics['roc_auc']:.4f} "
                f"f1={default_metrics['f1']:.4f} recall={default_metrics['recall']:.4f} "
                f"sec={row['seconds']:.1f}"
            )
            if default_metrics["pr_auc"] > best_pr_auc + 1e-4:
                best_pr_auc = default_metrics["pr_auc"]
                best_epoch = epoch
                best_state = copy.deepcopy(
                    {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    }
                )
                patience = 0
            else:
                patience += 1
                if patience >= args.patience:
                    print(f"early_stop epoch={epoch}")
                    break

    if best_state is None:
        raise RuntimeError("Training produced no model")
    model.load_state_dict(best_state)
    labels = np.asarray([], dtype=np.int64)
    probabilities = np.asarray([], dtype=np.float32)
    threshold = 0.5
    final_metrics = {}
    default_metrics = {}
    reference_metadata = {}
    if validation_loader is not None:
        labels, probabilities = evaluate(model, validation_loader, device)
        threshold = threshold_for_recall(labels, probabilities, args.target_recall)
        final_metrics = metrics_for(labels, probabilities, threshold)
        default_metrics = metrics_for(labels, probabilities, 0.5)
    elif args.reference_metadata:
        reference_path = Path(args.reference_metadata)
        reference_metadata = json.loads(reference_path.read_text(encoding="utf-8"))
        threshold = float(reference_metadata.get("decision_threshold", 0.5))
        final_metrics = reference_metadata.get("validation_metrics", {})
        default_metrics = reference_metadata.get("validation_metrics_at_0_5", {})

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output)
    metadata = {
        "architecture": ARCHITECTURE_NAME,
        "preprocess_version": PREPROCESS_VERSION,
        "input_resize": [IMAGE_SIZE, IMAGE_SIZE],
        "kymograph_size": [KYMO_HEIGHT, KYMO_WIDTH],
        "feature_count": FEATURE_COUNT,
        "class_names": ["not_meteor", "meteor"],
        "meteor_class_index": 1,
        "decision_threshold": threshold,
        "target_recall": args.target_recall,
        "training_events": len(training),
        "validation_events": len(validation),
        "training_nights": sorted(set(event["night"] for event in training)),
        "validation_nights": sorted(set(event["night"] for event in validation)),
        "best_epoch": best_epoch,
        "parameter_count": parameter_count,
        "validation_metrics": final_metrics,
        "validation_metrics_at_0_5": default_metrics,
        "history": history,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "camera_policy": "camera_name_date_source_excluded; temporal_median_mad_whitening",
        "trained_on_all_reviewed_events": bool(args.train_all),
        "split_policy": (
            "all_reviewed_events_after_model_selection"
            if args.train_all
            else f"holdout_year:{args.holdout_year}"
            if args.holdout_year
            else "stratified_group_kfold_by_night"
        ),
    }
    if reference_metadata:
        metadata["model_selection_metrics_source"] = str(args.reference_metadata)
    metadata_path = Path(f"{output}.meta.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if validation:
        predictions_path = output.with_suffix(".validation.jsonl")
        with predictions_path.open("w", encoding="utf-8") as handle:
            for event, label, probability in zip(validation, labels, probabilities):
                handle.write(
                    json.dumps(
                        {
                            "event": event["directory"],
                            "night": event["night"],
                            "source": event["source"],
                            "label": int(label),
                            "probability": float(probability),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    print(f"saved={output}")
    print(json.dumps(final_metrics, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", default="ml_training_data/reviewed"
    )
    parser.add_argument(
        "--cache-root", default="ml_training_data/universal_cache_v1"
    )
    parser.add_argument("--output", default="models/meteor_fusion_universal_v1.pth")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--target-recall", type=float, default=0.97)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--holdout-year",
        default="",
        help="Diagnostic only: reserve every event whose night starts with this year.",
    )
    parser.add_argument(
        "--train-all",
        action="store_true",
        help="Train the final deployment model on every reviewed event.",
    )
    parser.add_argument(
        "--reference-metadata",
        default="",
        help="Selection-run metadata providing the deployment threshold and metrics.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
