import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('observer','tools/observe_running_rtsp.py')
observer=importlib.util.module_from_spec(spec);spec.loader.exec_module(observer)


def test_observation_is_read_only_and_excludes_open_temps(tmp_path):
    observer.ROOT=tmp_path
    session=tmp_path/'.rtsp_live/day/hour/session';session.mkdir(parents=True)
    raw=session/'chunk_000000001.mp4';raw.write_bytes(b'closed')
    marker=Path(str(raw)+'.capture.json');marker.write_text(json.dumps({'frame_count':12,'fps':25,'start_time':'2026-10-13T04:00:00'}))
    (session/'chunk_000000002_temp_1.mp4').write_bytes(b'open')
    (session/'chunk_000000003.mp4').write_bytes(b'no timing')
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    result=observer.snapshot()
    assert len(result['entries'])==1
    assert len(result['incomplete'])==1
    assert result['entries'][str(raw.relative_to(tmp_path))]['seconds']==.48
    assert before=={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


def test_compare_counts_durable_ack_and_exact_active_session(tmp_path):
    observer.ROOT=tmp_path
    relative='.rtsp_live/old/hour/session/chunk_1.mp4'
    row=dict(seconds=.48,frames=12,fps=25,start_time='2026-10-13T04:00:00',consumed=False)
    old=dict(monotonic=1,observed_at='start',scan_seconds=.1,entries={relative:row},incomplete=[])
    current='.rtsp_live/today/hour/current'
    after=dict(monotonic=2,observed_at='end',scan_seconds=.1,entries={relative:dict(row,consumed=True),current+'/chunk_2.mp4':dict(row,seconds=.52)},incomplete=[])
    with patch.object(observer,'archived_sources',return_value=(set(),0)):
        result=observer.compare(old,after,current)
    assert result['acknowledged_media_seconds']==.48
    assert result['newly_recorded_media_seconds']==.52
    assert abs(result['pending_seconds_change']-.04)<1e-10
    assert result['new_session_ack_count']==0
    assert result['snapshot_wall_bounds']==[.9,1.1]
