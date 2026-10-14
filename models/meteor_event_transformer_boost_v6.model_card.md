# Meteor event transformer boost v6

Trained on 2026-10-14 from 7,979 unique explicitly reviewed event bundles, including the reviewed ML dataset and the existing hard-negative experiment set. Twenty-three duplicate clips were excluded. Pending and skipped events were not used as labels.

The split contains 6,749 training events, 1,019 validation events, and 211 test events. The classifier uses the fixed embedded DINOv2-small encoder and a 400-iteration, 31-leaf boosted-tree head. It preserves 3,873 v5 development true positives during threshold selection.

The primary decision threshold is **0.06657051318251248**. Validation metrics are F1 **0.9695**, precision **0.9453**, recall **0.9949** (588 TP, 3 FN, 34 FP, 394 TN). The held-out metrics are F1 **0.9644**, precision **0.9312**, recall **1.0000** (176 TP, 0 FN, 13 FP, 22 TN).

The companion foreground gate was retrained on the same versioned dataset. Its threshold is **0.016243367690646587**. It is linked by relative path and checksum in the model sidecar.

The held-out nights are 2026-10-08 and 2026-10-09, matching v5's test cohort. On this shared cohort, v6 has 13 false positives versus v5's 9. Those results support a direct comparison but are not an independent new benchmark for v6.

See the Japanese training and pending-sample comparison at ../docs/meteor_event_transformer_boost_v6_20261014.md, the sidecar at meteor_event_transformer_boost_v6.pth.meta.json, and the evaluation manifest at meteor_event_transformer_boost_v6.evaluation.json.
