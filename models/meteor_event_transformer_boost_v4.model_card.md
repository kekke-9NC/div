# Meteor event classifier V4 — 2026-10-11

## Intended use

V4 is a newly trained event classifier artifact for the legacy Meteor Detector UI. It is saved beside the earlier models; it has not been selected or installed as the app's active model.

## Training data and method

- 6,506 unique explicitly reviewed events after excluding 19 duplicate clips.
- 5,276 training events, 1,019 validation events, and 211 held-out comparison events.
- Data sources: `ml_training_data/reviewed` and `ml_training_data/experiments/retrain_20261011/hard_negatives`.
- The 2026-10-08 and 2026-10-09 capture dates were excluded from fitting and threshold selection. They are the same fixed comparison dates used for V3, so these results are a regression comparison, not a fresh independent test.
- The fixed DINOv2-small encoder and event/motion features were combined with an ensemble of four gradient-boosted tree models. The ensemble was selected using validation data only; the decision threshold is 0.4694749847.
- Dataset SHA-256: `655ba7f0591f8fbf16c5b40f0a8a38bd82a89692d076a1d509d48130b5f7541e`.

## Evaluation

Validation: 579/591 meteors retained (12 missed), and 5/428 non-meteors classified as meteors.

Fixed 211-event comparison set: 176/176 meteors retained, with 7/35 non-meteors classified as meteors. V3 on the same examples retained 176/176 meteors and classified 8/35 non-meteors as meteors. The set is small and has been used in earlier model development; it does not establish performance on future nights or changing conditions.

## Files

- Weights: `meteor_event_transformer_boost_v4.pth`
- Training metadata: `meteor_event_transformer_boost_v4.pth.meta.json`
- Per-event evaluation records: `meteor_event_transformer_boost_v4.evaluation.json`
