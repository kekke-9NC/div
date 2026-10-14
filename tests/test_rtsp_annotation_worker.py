from concurrent.futures import Future
from datetime import datetime,timedelta
from multiprocessing import shared_memory
from unittest.mock import Mock,patch

import numpy as np

from rtsp_annotation_worker import AnnotationWorker


def test_shared_annotation_matches_actual_sky_pixels_and_reclaims_worker_memory():
    from obs_meteor_overlay import sky_overlay
    payload = dict(model_type='fixed-camera-stg-poly',width=1920,height=1080,
                   polynomial_degree=2,stg_parameters=[.1,-.2,.05,np.log(700),np.log(700),960,540],
                   correction_coefficients=[[0,0]]*6,support_grid=[[1]*8]*5,
                   reference_datetime='2026-10-13T00:00:00')
    original = np.random.default_rng(80).integers(0,256,(1080,1920,3),dtype=np.uint8)
    worker = AnnotationWorker()
    try:
        for seconds,mask in ((0,None),(500,np.full((54,96),255,np.uint8))):
            when = datetime(2026,10,13)+timedelta(seconds=seconds)
            before = original.copy()
            layer = sky_overlay.render_layer(payload,when,mask,.45)
            assert np.any(layer[:,:,3])
            alpha = layer[:,:,3:4].astype(np.float32)/255
            expected = np.rint(original*(1-alpha)+layer[:,:,:3][:,:,::-1]*alpha).astype(np.uint8)
            actual = worker.render(original,payload,when,mask,.45)
            np.testing.assert_array_equal(actual,expected)
            np.testing.assert_array_equal(original,before)
            assert not worker.torch_loaded
        name = worker.memory.name
    finally:
        worker.close()
        worker.close()
    assert worker.pool is None and worker.memory is None
    try:
        leaked = shared_memory.SharedMemory(name=name)
    except FileNotFoundError:
        pass
    else:
        leaked.close()
        raise AssertionError('annotation slot leaked')
    assert worker.render(original,payload,when,mask,.45) is None


def test_failed_annotation_worker_requests_local_fallback_and_cannot_restart():
    failed = Future()
    failed.set_exception(RuntimeError('worker lost'))
    pool = Mock()
    pool.submit.return_value = failed
    worker = AnnotationWorker()
    frame = np.zeros((64,96,3),np.uint8)
    with patch('rtsp_annotation_worker.ProcessPoolExecutor',return_value=pool):
        assert worker.render(frame,{},datetime.now(),None,.45) is None
        assert worker.render(frame,{},datetime.now(),None,.45) is None
        assert worker.failed and pool.submit.call_count == 1
    worker.close()
    pool.shutdown.assert_called_once_with(wait=True,cancel_futures=True)
