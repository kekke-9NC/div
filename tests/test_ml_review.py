import json
from unittest import mock

import pytest

import ml_review


def test_meteor_prediction_highlights_meteor_confirmation():
    assert ml_review.confirmation_button_styles("meteor") == (
        ml_review.PREDICTED_CONFIRM_STYLE,
        ml_review.DEFAULT_CONFIRM_STYLE,
    )


def test_not_meteor_prediction_highlights_not_meteor_confirmation():
    assert ml_review.confirmation_button_styles("not_meteor") == (
        ml_review.DEFAULT_CONFIRM_STYLE,
        ml_review.PREDICTED_CONFIRM_STYLE,
    )


def test_unknown_prediction_does_not_highlight_confirmation():
    assert ml_review.confirmation_button_styles(None) == (
        ml_review.DEFAULT_CONFIRM_STYLE,
        ml_review.DEFAULT_CONFIRM_STYLE,
    )


@pytest.mark.parametrize("sign", [1, -1])
def test_overflowing_json_probability_displays_unknown_and_continues_preview(tmp_path, sign):
    (tmp_path / "metadata.json").write_text(json.dumps({
        "predicted_label": "not_meteor", "meteor_probability": sign * 10**400,
        "detection_time": "2026-10-12T23:00:00", "source": "source.mp4",
    }))
    window = mock.Mock()
    window.current_event = tmp_path
    window.events = [tmp_path]
    window.index = 0
    window._fit_photo.return_value = None
    with mock.patch.object(ml_review.cv2, "VideoCapture"):
        ml_review.TrainingDataReviewWindow._show_current(window)
    detail = window.detail_var.set.call_args.args[0]
    assert "流星確率: 不明" in detail
    assert "日時: 2026-10-12T23:00:00" in detail
    window._play_next_frame.assert_called_once_with()
