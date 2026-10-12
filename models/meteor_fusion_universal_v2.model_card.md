# MeteorFusion Universal v2

## Deployment artifact

- Weights: `meteor_fusion_universal_v2.pth`
- Metadata: `meteor_fusion_universal_v2.pth.meta.json`
- Architecture: `meteor_fusion_universal_v1` (same input contract as v1)
- Training data: 4,070 reviewed events, including 393 corrected misclassifications
- Classes: `not_meteor` / `meteor`
- Parameters: 250,849

## Training and selection

v2 was fine-tuned from `meteor_fusion_universal_v1.pth` using reviewed events,
then retrained on all reviewed events after model selection. The selection run
held out every event from 2026, leaving 721 completely unseen validation
events.

At the recall-oriented operating threshold `0.0568525419`:

- PR-AUC: 0.9982
- ROC-AUC: 0.9965
- Recall: 0.9907
- Precision: 0.9953
- Specificity: 0.9931
- F1: 0.9930
- Confusion matrix: TP 427 / FN 4 / FP 2 / TN 288

The app reads this threshold from the model metadata instead of applying the
legacy fixed `0.5` threshold.
