from datetime import datetime

from tools.benchmark_realtime_rtsp import historical_replay_now
from rtsp_schedule_products import schedule_window


def test_recorded_replay_finishes_its_saved_night_without_opening_tonights_job():
    saved = (datetime(2026,10,12,23,30,21),datetime(2026,10,13,5))
    inputs = [dict(start_time='2026-10-13T02:18:23',frame_count=15000,fps=25)]
    now = historical_replay_now(saved,inputs)
    assert now == datetime(2026,10,13,5,0,1)
    assert schedule_window(now,17,36,5,0) is None
    assert schedule_window(datetime(2026,10,13,18),17,36,5,0) is not None


def test_replay_without_products_uses_the_latest_recorded_frame_clock():
    inputs = [dict(start_time='2026-10-13T02:18:23',frame_count=1500,fps=25),
              dict(start_time='2026-10-13T03:20:23',frame_count=1500,fps=25)]
    assert historical_replay_now(None,inputs) == datetime(2026,10,13,3,21,24)
