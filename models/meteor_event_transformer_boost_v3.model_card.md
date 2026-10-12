# Meteor event classifier V3 — 2026-10-11

## Intended use

Selectable event classifier for the legacy Meteor Detector UI. The V3 model and its foreground gate are installed alongside V2. V2 remains selected in the running app while the OBS broadcast continues.

## Training data and split

- 6,164 unique reviewed events after duplicate removal; 4,934 training, 1,019 validation, 211 held-out evaluation events.
- Includes 308 recently confirmed non-meteor events from 2026-10-11 and 60 reviewed hard negatives from the retraining set.
- Capture dates 2026-10-08 and 2026-10-09 were held out from main-model training. Validation selection used only earlier validation dates; 2026-10-10 and 2026-10-11 were in training.
- Decision threshold: 0.401088284131, selected on validation.

## Held-out evaluation versus V2

The same 211-event evaluation set was used for both versions. The main classifier retained all 176/176 meteors in both versions. V3 reduced non-meteor false positives from 9 to 8 and increased true negatives from 26 to 27.

- V2: validation TP 580, FN 11, FP 7, TN 421; held-out TP 176, FN 0, FP 9, TN 26.
- V3: validation TP 581, FN 10, FP 8, TN 420; held-out TP 176, FN 0, FP 8, TN 27.

The separately trained foreground gate retained all 176/176 held-out meteors. Its held-out non-meteor rejections increased from 0 to 2 of 35. This set is small and has only 35 non-meteors; future recordings can differ.

## Deployment

Model: `meteor_event_transformer_boost_v3.pth`
Gate: `auxiliary/meteor_event_noise_gate_v2.pth`
Release ID: `20261011-v3`

Both files and metadata are installed under the legacy app's model directory. The running app still uses V2 so the broadcast is not interrupted. After the broadcast, click **Update** in Settings → **Flow classification model**, select V3, and apply it. No app restart is required for model selection.
