import numpy as np

from train_event_boost import choose_threshold


def test_recall_bound_keeps_baseline_positive_even_when_f1_prefers_rejection():
    labels = np.array([1, 1, 0, 0, 0])
    probabilities = np.array([.9, .2, .3, .4, .5])
    assert choose_threshold(labels, probabilities) > .2
    threshold = choose_threshold(labels, probabilities, maximum_threshold=.198)
    assert threshold <= .198
    assert np.all(probabilities[labels == 1] >= threshold)


def test_unbounded_threshold_preserves_default_f1_selection():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([.1, .2, .8, .9])
    assert choose_threshold(labels, probabilities) == .8


def test_gate_audit_does_not_count_background_as_an_event_or_hide_meteors():
    from video_processing import _reportable_detections
    meteor = dict(label='meteor', probability=.9)
    rejected = dict(label='not_meteor', classification_stage='foreground_gate')
    regular = dict(label='not_meteor', probability=.2)
    protected = dict(label='meteor', classification_stage='foreground_gate')
    items = [meteor, rejected, regular, protected]
    assert _reportable_detections(items) == items
    assert _reportable_detections(items, True) == [meteor, regular, protected]
