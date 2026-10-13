from unittest.mock import patch,call
import pytest
import video_io


def test_rtsp_thread_ceiling_is_shared_and_restored_after_nested_failure():
    with patch.object(video_io.cv2,'getNumThreads',return_value=18),patch.object(video_io.cv2,'setNumThreads') as setter:
        with video_io.rtsp_opencv_budget():
            assert video_io._budget_users == 1
            with pytest.raises(RuntimeError),video_io.rtsp_opencv_budget():
                assert video_io._budget_users == 2
                raise RuntimeError('recorder failed')
            assert setter.call_args_list == [call(4)]
        assert setter.call_args_list == [call(4),call(18)]
        assert video_io._budget_users == 0 and video_io._original_threads is None


def test_rtsp_budget_does_not_raise_a_smaller_user_budget():
    with patch.object(video_io.cv2,'getNumThreads',return_value=2),patch.object(video_io.cv2,'setNumThreads') as setter:
        with video_io.rtsp_opencv_budget():
            pass
        assert setter.call_args_list == [call(2),call(2)]
