# MeteorFusion Universal v1

## Purpose

Binary classification of detector candidates into `meteor` and `not_meteor`
without using camera names, source paths, dates, or predictions from an older
model.

## Deployment artifact

- Weights: `meteor_fusion_universal_v1.pth`
- Metadata: `meteor_fusion_universal_v1.pth.meta.json`
- Architecture: `meteor_fusion_universal_v1`
- Parameters: 250,849
- Training data: 2,622 reviewed events from 91 nights
- Classes: 1,173 meteor / 1,449 not_meteor

## Inputs

The app builds three inputs from the unaveraged monochrome event clip:

1. Five-channel robust temporal response image.
2. Track-aligned longitudinal/transverse space-time image.
3. Twelve dimensionless temporal and morphology features.

Each event is whitened with its temporal median and temporal MAD. Camera name,
recording year, file path, and previous model probability are excluded.

## Model selection evaluation

Model selection used a night-grouped holdout of 529 events from 15 nights.
These events were not used to fit the selection model.

At the recall-oriented threshold `0.1052513868`:

- PR-AUC: 0.9859
- ROC-AUC: 0.9895
- Recall: 0.9702
- Precision: 0.9306
- Specificity: 0.9422
- F1: 0.9500
- Confusion matrix: TP 228 / FN 7 / FP 17 / TN 277

At threshold `0.5`:

- Recall: 0.9532
- Precision: 0.9451
- Specificity: 0.9558
- F1: 0.9492

The deployment weights were subsequently retrained for 32 epochs using all
2,622 reviewed events. The holdout metrics above describe the model-selection
run, not a direct evaluation of the final all-data weights.

## Unseen-camera diagnostic

A separate diagnostic model was trained with all 2026 events excluded, then
tested on the 212 events from 2026:

- PR-AUC: 0.9342
- ROC-AUC: 0.9693
- Recall-oriented result: TP 50 / FN 1 / FP 23 / TN 138

The threshold for this diagnostic was selected on that diagnostic set, so its
confusion matrix is evidence of separability rather than an unbiased estimate
of a production threshold. A genuinely new third-party camera is still needed
for final external validation.

## Runtime behavior

The app uses the model automatically when it is the selected model. Legacy
three-channel `.pth` models remain loadable and use the legacy image path.
The model settings panel displays the recommended threshold.
