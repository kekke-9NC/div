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
