from unittest.mock import patch
import numpy as np
import component_statistics
import cv2


def test_mask_equality_matches_numpy_without_changing_or_copying_inputs():
    rng = np.random.default_rng(849)
    for shape in ((0,8),(8,8),(1080,1920)):
        left = rng.integers(0,256,shape,dtype=np.uint8)
        right = left.copy()
        assert component_statistics.same_uint8_pixels(left,right)
        assert component_statistics.same_uint8_pixels(left,left)
        for index in (0,left.size//2,left.size-1):
            if not left.size:
                continue
            right.flat[index] ^= np.uint8(1)
            assert not component_statistics.same_uint8_pixels(left,right)
            right.flat[index] ^= np.uint8(1)
        np.testing.assert_array_equal(left,right)
        for a,b in ((left[:,::2],right[:,::2]),(left[::-1],right[::-1]),
                    (left.astype(np.float32),right.astype(np.float32)),
                    (left.astype(bool),right.astype(bool)),(left,right.reshape(-1))):
            assert component_statistics.same_uint8_pixels(a,b)==np.array_equal(a,b)
        with patch.object(component_statistics,'_load',return_value=None):
            assert component_statistics.same_uint8_pixels(left,right)==np.array_equal(left,right)
    assert not component_statistics.same_uint8_pixels(np.array([np.nan]),np.array([np.nan]))
    assert component_statistics.same_uint8_pixels(np.array([0.]),np.array([-0.]))


def test_native_scores_and_peaks_match_exact_numpy_reference_and_keep_inputs():
    rng = np.random.default_rng(521)
    for count in (3, 400, 4096):
        raw_labels = rng.integers(0, count, (130, 180), dtype=np.int32)
        raw_values = rng.integers(0, 256, (130, 180), dtype=np.uint8)
        for labels, values in ((raw_labels, raw_values), (raw_labels[::2, ::2], raw_values[::2, ::2])):
            eligible = np.arange(1, count, 3)
            original_labels, original_values = labels.copy(), values.copy()
            keep = np.zeros(count, bool)
            keep[eligible] = True
            pixels = keep[labels.ravel()]
            selected_labels, selected_values = labels.ravel()[pixels], values.ravel()[pixels]
            expected = np.bincount(selected_labels, weights=selected_values, minlength=count)
            peaks = np.zeros(count, np.uint8)
            np.maximum.at(peaks, selected_labels, selected_values)
            actual = component_statistics.aggregate(labels, values, eligible, count)
            assert actual is not None
            np.testing.assert_array_equal(actual[0], expected)
            np.testing.assert_array_equal(actual[1], peaks)
            np.testing.assert_array_equal(labels, original_labels)
            np.testing.assert_array_equal(values, original_values)


def test_unavailable_native_or_unsupported_dtype_keeps_reference_path():
    labels, values = np.zeros((4, 4), np.int32), np.zeros((4, 4), np.uint8)
    with patch.object(component_statistics, '_load', return_value=None):
        assert component_statistics.aggregate(labels, values, [0], 1) is None
    assert component_statistics.aggregate(labels, values.astype(np.float32), [0], 1) is None


def test_integer_area_crop_matches_opencv_all_pixels_and_rounding_cases():
    rng = np.random.default_rng(979)
    for sy in (1, 2, 3, 4, 5, 6, 8):
        for sx in (1, 2, 3, 4, 5, 6, 8):
            frame = rng.integers(0, 256, (64*sy+20, 64*sx+20), dtype=np.uint8)
            integral = cv2.integral(frame, sdepth=cv2.CV_32S)
            for x, y in ((0, 0), (3, 7)):
                expected = cv2.resize(frame[y:y+64*sy, x:x+64*sx], (64, 64), interpolation=cv2.INTER_AREA)
                actual = component_statistics.area_crop(integral, x, y, sx, sy)
                assert actual is not None
                np.testing.assert_array_equal(actual, expected)
    assert component_statistics.area_crop(integral, -1, 0, 2, 2) is None


def test_fractional_160_area_crop_matches_opencv_all_pixels():
    rng = np.random.default_rng(25160)
    for trial in range(256):
        frame = rng.integers(0, 256, (175, 181), dtype=np.uint8)
        if trial < 8:
            frame = np.where(frame < trial*32, 0, 255).astype(np.uint8)
        integral = cv2.integral(frame, sdepth=cv2.CV_32S)
        x, y = trial % 21, trial % 15
        expected = cv2.resize(frame[y:y+160, x:x+160], (64, 64), interpolation=cv2.INTER_AREA)
        actual = component_statistics.area_crop_160(integral, x, y)
        assert actual is not None
        np.testing.assert_array_equal(actual, expected)
    assert component_statistics.area_crop_160(integral, -1, 0) is None


def test_batched_native_crops_match_every_frame_and_own_each_cached_image():
    rng = np.random.default_rng(6425)
    frames = list(rng.integers(0,256,(25,530,550),dtype=np.uint8))
    integrals = [cv2.integral(frame,sdepth=cv2.CV_32S) for frame in frames]
    for width,height in ((160,160),(128,128),(256,384),(512,64)):
        actual = component_statistics.area_crops(integrals,3,7,width,height)
        assert actual is not None
        for frame,crop in zip(frames,actual):
            np.testing.assert_array_equal(crop,cv2.resize(frame[7:7+height,3:3+width],(64,64),interpolation=cv2.INTER_AREA))
            assert crop.flags.owndata
    assert component_statistics.area_crops(integrals,3,7,163,160) is None


def test_native_small_temporal_median_matches_numpy_every_length_and_ties():
    rng = np.random.default_rng(3225)
    for length in range(1,33):
        for maximum in (1,255):
            values = rng.integers(0,maximum+1,(length,64,64),dtype=np.uint8)
            original = values.copy()
            actual = component_statistics.small_temporal_median(values)
            assert actual.dtype == np.float64
            np.testing.assert_array_equal(actual,np.median(values,axis=0))
            np.testing.assert_array_equal(original,values)


def test_parallel_native_median_matches_large_numpy_even_and_odd_histories():
    from temporal_statistics import temporal_median
    rng = np.random.default_rng(964540)
    for length in (24,25):
        values = rng.integers(0,256,(length,540,960),dtype=np.uint8)
        original = values.copy()
        np.testing.assert_array_equal(temporal_median(values),np.median(values,axis=0))
        np.testing.assert_array_equal(values,original)
