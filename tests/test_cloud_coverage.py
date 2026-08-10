from unittest import mock

import numpy as np

from cloud_coverage import CloudClassification, classify_cloud_fraction


def test_heuristic_classification_is_bounded():
    result = classify_cloud_fraction(np.zeros((120, 200, 3), dtype=np.uint8), backend="disabled")
    assert result.source == "heuristic"
    assert 0.0 <= result.cloud_fraction <= 1.0


def test_qwen_openai_compatible_response_is_parsed():
    response = mock.Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "choices": [{"message": {"content": '{"cloud_fraction": 0.07, "confidence": 0.91}'}}]
    }
    with mock.patch("cloud_coverage.requests.post", return_value=response) as post:
        result = classify_cloud_fraction(
            np.zeros((60, 100, 3), dtype=np.uint8), backend="lmstudio_qwen3_5_2b",
            lm_studio_url="http://localhost:1234/v1", lm_studio_model_id="qwen/qwen3-vl-4b",
        )
    assert result.source == "qwen-vlm"
    assert result.cloud_fraction == 0.07
    post.assert_called_once()
