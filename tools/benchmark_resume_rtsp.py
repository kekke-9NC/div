"""Measure startup resume on a byte copy of the complete saved RTSP state.

The GUI resume handler and RTSP analysis entry point are production code.
Only the modal choice and network recorder are replaced. No new frames enter.
Model loading, warmup and copying are outside the reported processing timer.
"""
import contextlib
import hashlib
import json
from pathlib import Path
import queue
import shutil
import threading
import time


def rebase(value, source, target):
    if isinstance(value, str):
        return value.replace(str(source), str(target))
    if isinstance(value, list):
        return [rebase(item, source, target) for item in value]
    if isinstance(value, dict):
        return {key: rebase(item, source, target) for key, item in value.items()}
    return value


def fingerprint(path):
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino]


def run_resume(args, settings, options, noise, correction, mask):
    import config
    import file_utils
    import gui_processing
    import rtsp_live
    import rtsp_schedule_products as products
    import rtsp_startup
    import video_processing
    import torch

    source = args.resume_root.resolve()
    root = args.output / 'rtsp'
    if source == root or source in root.parents or root in source.parents:
        raise ValueError('Snapshot output must be separate from saved recordings')
    pending = rtsp_startup.pending_recordings(source)
    if not pending or any(not path.name.startswith('chunk_') for path in pending):
        raise ValueError('This verification requires pending raw continuous RTSP chunks')
    if noise.get('enabled') or noise.get('temporal_mean_frames'):
        raise ValueError('Raw resume verification cannot substitute preprocessing')
    files = sorted(path for path in source.rglob('*') if path.is_file())
    if any(path.is_symlink() for path in source.rglob('*')):
        raise ValueError('Saved state must not contain symlinks')
    originals = {str(path.relative_to(source)): fingerprint(path) for path in files}
    inputs = []
    for path in pending:
        data = json.loads(Path(str(path)+'.capture.json').read_text())
        inputs.append(dict(relative=str(path.relative_to(source)),
                           sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **data))
    (args.output/'snapshot_originals.json').write_text(json.dumps(originals))
    (args.output/'resume_inputs.json').write_text(json.dumps(inputs, indent=2))
    print(f'Copying full resume state: {len(files)} files; {sum(p.stat().st_size for p in files)/1e9:.2f} GB', flush=True)
    shutil.copytree(source, root)
    for path in root.rglob('*.json'):
        data = json.loads(path.read_text())
        updated = rebase(data, source, root)
        if data != updated:
            path.write_text(json.dumps(updated, ensure_ascii=False, indent=2))
    for path in (root/'.schedule_jobs').glob('*/job.json'):
        data = json.loads(path.read_text())
        if data['options'].get('output_dir'):
            data['options']['output_dir'] = str(args.output/'schedule_products')
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    if any(hashlib.sha256((root/item['relative']).read_bytes()).hexdigest()!=item['sha256'] for item in inputs):
        raise ValueError('Pending recording copy differs from original')
    config.RTSP_SAVE_ROOT = str(root)
    targets = {str(root/item['relative']): item for item in inputs}
    duration = sum(item['frame_count']/item['fps'] for item in inputs)
    consumed, sessions, publications, detections, failures = {}, [], [], [], []
    lock = threading.RLock()
    cancel = threading.Event()
    restored = []
    original_detect = video_processing.create_line_video_clips
    original_index = video_processing._SavedEventIndex
    original_consume = rtsp_live.RealtimeRunner._consume
    original_publish = rtsp_live.MinuteArchiver._publish
    original_recorder = file_utils.save_rtsp_video_segments
    detection_done = None
    completion_reason = None
    started = None

    def consume(runner, path):
        original_consume(runner, path)
        if str(path) in targets:
            with lock:
                consumed[str(path)] = time.perf_counter()-started

    def detect(*values, **kwargs):
        capture = kwargs['save_options']['_rtsp_live_capture']
        begin = time.perf_counter()
        result = original_detect(*values, **kwargs)
        with lock:
            sessions.append(dict(first=str(capture.first_path.relative_to(root)),
                                 seconds=time.perf_counter()-begin,
                                 decoded_frames=capture.total_input_frames, eof=capture.eof))
        return result

    class ObservedIndex(original_index):
        def append(self, item):
            super().append(item)
            detections.append(dict(item))

    def publish(archiver, paths):
        metadata = [(str(path), dict(rtsp_live.timing(path))) for path in paths]
        original_publish(archiver, paths)
        with lock:
            publications.extend(metadata)

    def status(message):
        print(message, flush=True)
        if 'エラー' in message or '再試行予定' in message or '未連結の録画を保持' in message:
            failures.append(message)

    def recorder(*_values, **_kwargs):
        nonlocal detection_done, completion_reason
        while not cancel.wait(.5):
            elapsed = time.perf_counter()-started
            with lock:
                count = len(consumed)
                media = sum(targets[path]['frame_count']/targets[path]['fps'] for path in consumed)
                ended = bool(sessions) and all(row['eof'] for row in sessions)
                complete = count == len(targets) and ended
                archived = {path for path, _data in publications if path in targets}
            jobs = [json.loads(path.read_text()) for path in (root/'.schedule_jobs').glob('*/job.json')]
            job_states = {job['id']: job['state'] for job in jobs}
            if complete and detection_done is None:
                detection_done = elapsed
            snapshot = dict(elapsed_seconds=elapsed, consumed_chunks=count,
                            total_chunks=len(targets), consumed_media_seconds=media,
                            input_seconds=duration, detection_done_seconds=detection_done,
                            archived_pending_chunks=len(archived), jobs=job_states,
                            failures=failures[-5:])
            rtsp_live.write_json(args.output/'progress.json', snapshot)
            if complete and len(archived)==len(targets) and all(value=='complete' for value in job_states.values()):
                completion_reason = 'all_detection_and_products_complete'
                return
            if detection_done is not None and elapsed-detection_done>120:
                completion_reason = 'products_not_complete_after_detection_plus_120s'
                return
            if elapsed > args.timeout:
                completion_reason = 'timeout'
                return

    for owner, name, replacement in (
        (video_processing,'create_line_video_clips',detect),
        (video_processing,'_SavedEventIndex',ObservedIndex),
        (rtsp_live.RealtimeRunner,'_consume',consume),
        (rtsp_live.MinuteArchiver,'_publish',publish),
        (file_utils,'save_rtsp_video_segments',recorder)):
        restored.append((owner,name,getattr(owner,name)))
        setattr(owner,name,replacement)
    app = object.__new__(gui_processing.ProcessingMixin)
    app._startup_analysis_ready = False
    app.progress_queue = queue.Queue()
    app.append_log = status
    choice_observations = []
    def choose(jobs, recordings):
        choice_observations.append(dict(jobs=len(jobs),recordings=len(recordings),choice='resume'))
        return 'resume'
    app._prompt_startup_analysis = choose
    clock = tuple(int(settings.get(key,fallback)) for key,fallback in (
        ('rtsp_start_hour',17),('rtsp_start_minute',0),('rtsp_end_hour',5),('rtsp_end_minute',0)))
    try:
        with (args.output/'pipeline.log').open('w') as log, contextlib.redirect_stdout(log):
            started = time.perf_counter()
            if not app._prepare_startup_analysis():
                raise RuntimeError('Actual startup resume handler failed')
            file_utils.rtsp_save_and_process_thread_target(
                rtsp_url='rtsp://no-network-resume',save_root=str(root),
                segment_duration=config.RTSP_SEGMENT_DURATION,scan_interval=config.RTSP_SCAN_INTERVAL,
                progress_callback=lambda value: status(value[0]) if str(value[0]).startswith('[自動作成]') else None,
                mask=mask,meteor_save_path=str(args.output/'meteor'),not_meteor_save_path=str(args.output/'not_meteor'),
                cancel_flag=cancel,save_options=options,interval=float(settings.get('interval',1)),
                duration=float(settings.get('duration',1)),summary_video_config=settings.get('summary_video_config'),
                time_limit_enabled=bool(settings.get('rtsp_time_limit_enabled',True)),
                start_hour=clock[0],start_minute=clock[1],end_hour=clock[2],end_minute=clock[3],
                max_workers=int(settings.get('concurrency',4)),dark_frame=correction,
                notify_on_detection=False,noise_twin_options=noise,rtsp_fps=config.RTSP_FPS,
                schedule_product_options={})
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
    finally:
        cancel.set()
        products.stop_background_products()
        for owner,name,original in reversed(restored):
            setattr(owner,name,original)
    seconds = time.perf_counter()-started
    changes = [relative for relative,stamp in originals.items()
               if not (source/relative).exists() or fingerprint(source/relative)!=stamp]
    hashes_ok = all(hashlib.sha256((source/item['relative']).read_bytes()).hexdigest()==item['sha256'] for item in inputs)
    archived = {path:data for path,data in publications if path in targets}
    published_frames = sum(data['frame_count'] for data in archived.values())
    final_jobs = [json.loads(path.read_text()) for path in (root/'.schedule_jobs').glob('*/job.json')]
    alive = [thread.name for thread in threading.enumerate()
             if thread.name.startswith(('RTSP','rtsp-'))]
    result = dict(seconds=seconds,input_seconds=duration,input_frames=sum(item['frame_count'] for item in inputs),
                  detection_seconds=detection_done,processing_ratio=None if detection_done is None else detection_done/duration,
                  end_to_end_ratio=seconds/duration,completion_reason=completion_reason,
                  startup_choices=choice_observations,sessions=sessions,consumed_chunks=len(consumed),
                  total_chunks=len(targets),archived_pending_chunks=len(archived),archived_pending_frames=published_frames,
                  jobs=[dict(id=job['id'],state=job['state'],outputs=job['outputs']) for job in final_jobs],
                  failures=failures,original_changed_files=changes,original_pending_hashes_unchanged=hashes_ok,
                  threads_alive=alive,installed_app_started=False,network_recorder_replaced=True)
    result['detection_valid'] = (detection_done is not None and len(consumed)==len(targets)
                                 and all(row['eof'] for row in sessions) and not alive
                                 and not changes and hashes_ok and published_frames==result['input_frames'])
    result['faster_than_video'] = result['detection_valid'] and detection_done < duration
    result['full_resume_complete'] = (result['detection_valid'] and all(job['state']=='complete' for job in final_jobs))
    (args.output/'publications.json').write_text(json.dumps(publications,indent=2))
    (args.output/'consumed.json').write_text(json.dumps(consumed,indent=2))
    return result,detections
