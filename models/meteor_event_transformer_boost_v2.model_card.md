# Meteor event classifier V2 — 2026-10-11

## Intended use

Legacy Tk meteor detection, with reviewed corrections from capture dates 2026-10-10 and 2026-10-11. This is an event classifier, not a plate solve model. It requires chronological frames and the detector’s focused activity window.

## Model and inputs

An unchanged pretrained DINOv2-small encoder produces two image-view embeddings. A supervised histogram gradient boosted tree classifier combines these with event brightness, temporal image/kymograph and motion features. The encoder weights are embedded, so runtime does not download a model. Only the supervised classifier and the separate lightweight foreground filter were trained here.

Neither classifier uses wall clock, date, camera identifier, file name, or the previous predicted label as an input. The same classifier handles the evening and dawn. Labels come from reviewed bundles, never pending predictions or assumptions that a particular hour cannot contain meteors.

## Main classifier training

- 5,852 unique reviewed examples: 5,792 canonical examples and 60 manually inspected native negative corrections. Nineteen canonical duplicate clips excluded.
- Canonical examples from capture dates October 10/11: 389 (230/159); all placed in training. Including extra native negatives: 449 (270/179).
- Training: 4,622; validation: 1,019; held-out test: 211, from capture dates October 8/9. Capture dates are disjoint. No refit on validation or test.
- Decision threshold: 0.4404476142134689, selected on validation.
- Validation: TP580, FN11, FP7, TN421.
- Test: TP176, FN0, FP9, TN26. Test precision 95.14%; non-meteor specificity 74.29%. The test is small and class imbalanced; this is not a claim of zero false positives in deployment.
- Portable tree export reproduces training probabilities exactly.

## Lightweight foreground filter

A separately trained tree filter uses 74 features of chronological 64-pixel grayscale crops, including temporal activity and movement. It runs before the full image encoder and media export. Its native reviewed examples include 131 negatives and all 29 previously confirmed meteors. Twenty-nine positives have the same weight as native negatives. Threshold calibration preserves every training meteor and every validation true positive of the full classifier, with a 0.9 score margin. Candidate selection uses validation negatives only; test data do not select the filter.

The 29 confirmed meteors are now training/regression examples. Their successful detection is a regression check, not an independent accuracy estimate. The cascade loses zero main-classifier true positives on training, validation and held-out test.

## Runtime and saved media

The existing coarse/fine proposal thresholds, frame rate, input pixels, mask and meteor media options are preserved. Exact dense-component scoring, bounded overlapping frame reuse and parallel filter evaluation reduce work. Models without foreground-filter metadata retain their former classification path. A filter evaluation failure bypasses filtering; a corrupt filter checkpoint refuses the switch and leaves the working model intact.

Rejected foreground candidates keep the original source recording and an atomic JSON record with source frame range and position, rather than thousands of event media exports. For candidates classified by the full model with meteor score below 0.01, training clip, crop media/images and information are retained, while full-frame and summary videos are omitted. Every meteor retains all enabled media outputs. Save settings themselves are not disabled.

## Provenance and limitations

Metadata and the evaluation manifest record dataset/weight fingerprints, date splits and predictions. Native input fingerprints include frame rate and geometry. Extra correction bundles remain local under `ml_training_data/experiments/retrain_20261011`; they do not alter the user's canonical reviewed labels. Benchmark evidence is under `obs_meteor_overlay/validation/retrain_realtime_20261011`.

Review notes on native negative bundles describe the chronological frame inspection. Original 20:30 false predictions include aircraft and stationary brightness/compression events; a twelve-event prediction count was never a verified count of twelve meteors. The separate 04:30 prediction has not been visually confirmed as a meteor.

New-camera, unseen weather, very faint and unusual meteors can still be rejected. Real-time timing is measured on selected recordings on this Mac with the legacy settings and outputs, excluding RTSP capture and Tk repaint. No universal worst-case timing or perfect recall guarantee is made.

DINOv2 license and attribution: `THIRD_PARTY_NOTICES.md` and `assets/licenses/dinov2/LICENSE`.
