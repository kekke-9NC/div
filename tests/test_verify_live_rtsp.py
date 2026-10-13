from datetime import datetime
import io

from tools.verify_live_rtsp import LiveMeter,RedactedLog,summarize_timeline


def test_live_window_clock_counts_progress_without_retained_ack_tail(tmp_path):
    meter = LiveMeter()
    path = tmp_path/'chunk_000000000.mp4'
    meter.receive(path,dict(start_time='2026-10-13T19:00:00',frame_count=25,fps=25,size=[1920,1080]))
    class Capture:
        started_at = datetime(2026,10,13,19)
        fps = 25
    meter.process(Capture(),19,False)
    sample = meter.sample(0)
    assert sample['processed_media_seconds'] == .8
    assert sample['lag_seconds'] == .2
    assert sample['consumed_chunks'] == 0
    meter.receive(path,dict(start_time='2026-10-13T19:00:00',frame_count=25,fps=25,size=[1920,1080]))
    assert len(meter.chunks)==1


def test_catch_up_requires_receiving_and_keeps_monotonic_wall_interval():
    rows = [dict(analysis_seconds=0,capture_stopped=False,lag_seconds=60,processed_media_seconds=0),
            dict(analysis_seconds=20,capture_stopped=False,lag_seconds=40,processed_media_seconds=40),
            dict(analysis_seconds=60,capture_stopped=False,lag_seconds=2,processed_media_seconds=118),
            dict(analysis_seconds=70,capture_stopped=False,lag_seconds=2,processed_media_seconds=128)]
    result = summarize_timeline(rows,60)
    assert result['catch_up_seconds']==60
    assert result['catch_up_media_seconds_per_wall_second']==118/60
    assert result['steady_samples']==1
    assert result['steady_p95_lag_seconds']==2
    rows[2]['capture_stopped']=rows[3]['capture_stopped']=True
    assert not summarize_timeline(rows,60)['caught_up_while_receiving']


def test_url_redaction_buffers_split_print_writes_before_logging():
    handle = io.StringIO()
    logger = RedactedLog(handle)
    logger.write('connect: rtsp://user:')
    logger.write('secret@camera/stream\n')
    assert handle.getvalue()=='connect: rtsp://[redacted]\n'
