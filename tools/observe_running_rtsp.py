"""Read-only observations of durable acknowledgements; never imports the app."""
import os
import json
import time
import struct
from datetime import datetime
from pathlib import Path

ROOT = None
OUT = None


def snapshot():
    began = time.monotonic()
    entries, sessions, incomplete = {}, {}, []
    for directory, _dirs, names in os.walk(ROOT/'.rtsp_live'):
        names = set(names)
        chunks = [name for name in names if name.startswith('chunk_') and name.endswith('.mp4')
                  and '_temp_' not in name and '_innovation' not in name]
        if not chunks:
            continue
        folder = Path(directory)
        session_key = str(folder.relative_to(ROOT))
        total = pending = acknowledged = 0.
        latest = None
        for name in chunks:
            try:
                data = json.loads((folder/(name+'.capture.json')).read_text())
                seconds = data['frame_count']/data['fps']
                if seconds <= 0:
                    raise ValueError('Invalid closed capture duration')
            except (OSError,ValueError,KeyError,ZeroDivisionError):
                incomplete.append(str((folder/name).relative_to(ROOT)))
                continue
            consumed = name+'.analysis_consumed.json' in names
            relative = str((folder/name).relative_to(ROOT))
            entries[relative] = dict(seconds=seconds,frames=data['frame_count'],fps=data['fps'],
                                     start_time=data['start_time'],consumed=consumed)
            total += seconds
            if consumed:
                acknowledged += seconds
            else:
                pending += seconds
            latest = max(latest or data['start_time'],data['start_time'])
        sessions[session_key] = dict(closed_chunks=sum(key.startswith(session_key+'/') for key in entries),
                                    closed_raw_seconds=total,pending_raw_seconds=pending,
                                    surviving_ack_seconds=acknowledged,latest_closed_clock=latest)
    ended = time.monotonic()
    return dict(observed_at=datetime.now().isoformat(),monotonic=(began+ended)/2,
                scan_seconds=ended-began,entries=entries,sessions=sessions,incomplete=incomplete)


def finalized_mp4(path):
    # Same read-only top-level atom validation used by the running archiver.
    try:
        size=path.stat().st_size
        found=set()
        with path.open('rb') as stream:
            offset=0
            while offset+8<=size:
                stream.seek(offset)
                length,kind=struct.unpack('>I4s',stream.read(8))
                header=8
                if length==1:
                    length=struct.unpack('>Q',stream.read(8))[0]
                    header=16
                elif length==0:
                    length=size-offset
                if length<header or offset+length>size:
                    return False
                found.add(kind)
                offset += length
        return offset==size and b'moov' in found and b'mdat' in found
    except (OSError,ValueError,struct.error):
        return False


def archived_sources():
    found = set()
    archives = 0
    for directory, _dirs, names in os.walk(ROOT):
        if Path(directory).name.startswith('.'):
            _dirs[:] = []
            continue
        _dirs[:] = [name for name in _dirs if not name.startswith('.')]
        for name in names:
            if not name.endswith('.mp4.capture.json') or '_minute_' not in name:
                continue
            marker = Path(directory)/name
            archive = Path(str(marker)[:-len('.capture.json')])
            try:
                if not archive.is_file() or not finalized_mp4(archive):
                    continue
                data = json.loads(marker.read_text())
                found.update(data.get('sources',()))
                archives += 1
            except (OSError,ValueError):
                continue
    return found,archives


def compare(before, after, active_session=None):
    sources,archive_count = archived_sources()
    consumed,unknown = {},[]
    for relative,old in before['entries'].items():
        if old['consumed']:
            continue
        new = after['entries'].get(relative)
        if new is not None:
            if new['consumed']:
                consumed[relative] = dict(old,proof='surviving_analysis_consumed_marker')
        elif relative in sources and not (ROOT/relative).exists():
            consumed[relative] = dict(old,proof='deleted_only_after_ack_and_retained_archive')
        else:
            unknown.append(relative)
    wall = after['monotonic']-before['monotonic']
    media = sum(row['seconds'] for row in consumed.values())
    grouped = {}
    for relative,row in consumed.items():
        session = str(Path(relative).parent)
        group = grouped.setdefault(session,dict(consumed_chunks=0,consumed_media_seconds=0.))
        group['consumed_chunks'] += 1
        group['consumed_media_seconds'] += row['seconds']
    new = {key:row for key,row in after['entries'].items() if key not in before['entries']}
    # Recorder is still on this day's session, whose raw files have not yet
    # reached analysis. Their completed capture durations measure arrivals.
    live_new = {key:row for key,row in new.items()
                if active_session and str(Path(key).parent) == active_session}
    b_pending = sum(row['seconds'] for row in before['entries'].values() if not row['consumed'])
    a_pending = sum(row['seconds'] for row in after['entries'].values() if not row['consumed'])
    return dict(start=before['observed_at'],end=after['observed_at'],wall_seconds=wall,
                scan_seconds=[before['scan_seconds'],after['scan_seconds']],
                snapshot_wall_bounds=[wall-(before['scan_seconds']+after['scan_seconds'])/2,
                                      wall+(before['scan_seconds']+after['scan_seconds'])/2],
                incomplete_counts=[len(before['incomplete']),len(after['incomplete'])],
                active_session=active_session,
                acknowledged_chunks=len(consumed),acknowledged_media_seconds=media,
                processing_speed=media/wall,processing_ratio=None if not media else wall/media,
                faster_than_realtime=media>wall,by_session=grouped,
                newly_recorded_closed_chunks=len(live_new),newly_recorded_media_seconds=sum(x['seconds'] for x in live_new.values()),
                pending_seconds_before=b_pending,pending_seconds_after=a_pending,
                pending_seconds_change=a_pending-b_pending,
                unknown_disappeared_inputs=unknown,retained_archives_examined=archive_count,
                new_session_ack_count=sum(1 for path in consumed
                                         if active_session and str(Path(path).parent) == active_session),
                acknowledgement_source='Consumer completed detection/save window, retaining recent finer-detection tail; not decoder read-ahead',
                limitations=['Read-only chunk acknowledgement observation, not full backlog completion',
                             'A fixed recent tail and queued window boundaries make short interval rates fluctuate',
                             'A disappeared chunk is counted only if its retained archive source mapping exists; assumes no external deletion'],
                acknowledged_inputs=consumed)


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--active-session', help='Session path relative to root')
    parser.add_argument('--before')
    args = parser.parse_args()
    ROOT = args.root.resolve()
    OUT = args.output.resolve()
    if OUT == ROOT or ROOT in OUT.parents:
        parser.error('Observations must be saved outside the recording root')
    if Path(args.name).name != args.name or (args.before and Path(args.before).name != args.before):
        parser.error('Snapshot names must be plain filenames')
    OUT.mkdir(parents=True, exist_ok=True)
    value = snapshot()
    path=OUT/(args.name+'.json')
    path.write_text(json.dumps(value,indent=2))
    print(json.dumps({k:v for k,v in value.items() if k!='entries'},indent=2))
    if args.before:
        result=compare(json.loads((OUT/(args.before+'.json')).read_text()),value,args.active_session)
        (OUT/(args.name+'-comparison.json')).write_text(json.dumps(result,indent=2))
        print(json.dumps({k:v for k,v in result.items() if k!='acknowledged_inputs'},indent=2))
