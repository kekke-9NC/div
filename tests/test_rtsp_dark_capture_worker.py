import numpy as np

import rtsp_dark_capture_worker


def test_build_fixed_pattern_excludes_known_timestamp_regions():
    height, width = 100, 200
    samples = []
    for index in range(11):
        frame = np.full((height, width), 40, dtype=np.uint8)
        frame[2:7, 3:45] = 120 + index
        frame[95:99, 150:195] = 100 + index
        frame[45:55, 90:110] = 70
        samples.append(frame)

    _pattern, correction, _target = rtsp_dark_capture_worker.build_fixed_pattern(samples)

    assert np.count_nonzero(correction[:9, :50]) == 0
    assert np.count_nonzero(correction[94:, 140:]) == 0
    assert np.count_nonzero(correction[45:55, 90:110]) > 0


def test_choose_sample_locations_is_reproducible_and_spans_clips():
    counts = [100, 200, 300]

    first = rtsp_dark_capture_worker.choose_sample_locations(counts, 90, random_seed=7)
    second = rtsp_dark_capture_worker.choose_sample_locations(counts, 90, random_seed=7)

    assert first == second
    assert len(first) == 90
    assert {video_index for video_index, _frame_index in first} == {0, 1, 2}
    assert all(0 <= frame_index < counts[video_index]
               for video_index, frame_index in first)
