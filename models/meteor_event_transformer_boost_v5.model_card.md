# Meteor event transformer boost v5

Trained 2026-10-13 from 7,393 deduplicated reviewed events, including the latest cloud/noise corrections. Fixed embedded DINOv2-small weights supply image features; a newly trained 600-iteration boosted-tree head supplies the primary prediction. No network model download is required.

The primary threshold is **0.17177035153000328**. Calibration preserves v4's 3,743 reviewed development true positives with a score margin. Date-disjoint held-out nights 2026-10-08/09 remain outside training and threshold selection. The auxiliary 74-feature foreground filter is embedded by relative path and verified checksum; its threshold is **0.00840118635499679**.

The cascade retains all 176 held-out meteors and all v4 development true positives. Newly reviewed examples have 413 TP, 0 FN and 0 FP, but are training examples. Historical validation FP increases from 5 to 19 and held-out FP from 7 to 8; improved sensitivity and fitting recent corrections do not demonstrate universal cloud generalization.

Foreground rejections are retained in a separate audit JSON when info saving is enabled. They do not count as normal detected events. Every accepted meteor retains the configured media exports. The model sidecar declares the one-time migration from bundled v1–v4; deliberate later choices and custom models remain supported.

See [validation and runtime comparison](../docs/cloud_noise_v5_20261013.md) and the sidecar/evaluation manifest for data fingerprints, capture-date splits and export verification.
