"""Isolated recorded-input replay for benchmark_legacy_detection --realtime.

Only the recorder is replaced: the app's RTSP entry point, continuous capture,
0.25-second proposal clock, classifier, media saving, minute archiver and
scheduled-product worker run normally. Inputs and app settings are read-only;
all consumption markers, archives and outputs belong to a new replay root.
"""
import contextlib
from datetime import datetime, timedelta
import json
from pathlib import Path
import resource
import shutil
import subprocess
import threading
import time


def run_replay(args, settings, options, noise, correction, mask):
    import cProfile
    import numpy as np
    import torch
    import config
    import file_utils
    import media_time
    import rtsp_live
    import rtsp_schedule_products as products
    import rtsp_segments
    import video_processing
    import moving_point_detector
    import short_flash_detector
    import model
    import image_processing

    root = args.output / 'rtsp'
    root.mkdir()
    config.RTSP_SAVE_ROOT = str(root)
    paths, inputs, sessions = [], [], []
    remaining = args.replay_seconds
    for source in args.sources:
        if args.replay_seconds and remaining <= 0:
            break
        if source.is_dir():
            original = sorted(source.glob('chunk_*.mp4'))
            selected = []
            for path in original:
                if '_innovation' in path.name:
                    continue
                data = json.loads(Path(str(path) + '.capture.json').read_text())
                if args.replay_seconds and remaining <= 0:
                    break
                selected.append((path, data))
                remaining -= data['frame_count'] / data['fps']
        else:
            source = source.resolve()
            started, _ = media_time.get_media_start_time(str(source))
            if started is None:
                raise ValueError(f'Input has no recording clock: {source}')
            directory = root / '.rtsp_live' / f'session_{len(sessions):04d}'
            directory.mkdir(parents=True)
            listing = directory / 'segments.csv'
            command = [shutil.which('ffmpeg') or 'ffmpeg', '-nostdin', '-v', 'error',
                '-i', str(source)]
            if args.replay_seconds:
                command += ['-t', str(remaining)]
            subprocess.run(command + ['-map', '0:v:0', '-an', '-c', 'copy', '-f', 'segment',
                '-segment_time', str(config.RTSP_SEGMENT_DURATION), '-reset_timestamps', '1',
                '-segment_list', str(listing), '-segment_list_type', 'csv',
                str(directory / 'chunk_%09d.mp4')], check=True)
            import csv
            selected = []
            for row in csv.reader(listing.read_text().splitlines()):
                path = directory / Path(row[0]).name
                fps, count, size = rtsp_segments._properties(path)
                selected.append((path, dict(start_time=(started + timedelta(seconds=float(row[1]))).isoformat(),
                    fps=fps, frame_count=count, size=list(size))))
                remaining -= count / fps
            listing.unlink()  # Timing below is authoritative; no source file is changed.
        if not selected:
            continue
        directory = root / '.rtsp_live' / f'session_{len(sessions):04d}'
        directory.mkdir(parents=True, exist_ok=True)
        rtsp_live.write_json(directory / '.session.json', {'start_time': selected[0][1]['start_time']})
        sessions.append(directory)
        for index, (source, data) in enumerate(selected):
            path = directory / f'chunk_{index:09d}.mp4'
            if source != path:
                shutil.copy2(source, path)
            rtsp_live.write_json(str(path) + '.capture.json', data)
            paths.append(path)
            inputs.append(dict(source=str(source), replay=str(path), **data))
    if not paths:
        raise ValueError('No replay inputs selected')
    duration = sum(item['frame_count']/item['fps'] for item in inputs)
    if noise.get('enabled') or noise.get('temporal_mean_frames'):
        raise ValueError('Preprocessed replay requires matching innovation/mean inputs')

    product_options = products.processing_options({} if args.no_products else settings.get('rtsp_schedule_products', {}))
    clock = tuple(int(settings.get(key, fallback)) for key, fallback in (
        ('rtsp_start_hour', 17), ('rtsp_start_minute', 0),
        ('rtsp_end_hour', 5), ('rtsp_end_minute', 0)))
    schedule_window = None
    if products.enabled(product_options):
        start = min(datetime.fromisoformat(item['start_time']) for item in inputs)
        end = max(datetime.fromisoformat(item['start_time']) + timedelta(seconds=item['frame_count']/item['fps'])
                  for item in inputs)
        # Resume the saved job's full clock, as the actual startup workflow
        # does. Compressing a whole night into the fixture's short time span
        # would invent hundreds of extra timelapse samples.
        saved_windows = []
        for marker in (args.settings.resolve().parent / 'rtsp/.schedule_jobs').glob('*/job.json'):
            saved = json.loads(marker.read_text(encoding='utf-8'))
            low, high = datetime.fromisoformat(saved['start']), datetime.fromisoformat(saved['end'])
            if low <= start and high >= end:
                saved_windows.append((low, high))
        schedule_window = max(saved_windows) if saved_windows else products.schedule_window(start, *clock)
    if schedule_window is not None:
        start, end = schedule_window
        job_dir = root / '.schedule_jobs' / 'replay'
        job_dir.mkdir(parents=True)
        job = dict(id='replay', start=start.isoformat(), end=end.isoformat(),
                   options=products.normalize_options(product_options), state='recording',
                   archives=[], events=[], samples={}, outputs={})
        if mask is not None:
            np.save(job_dir / 'mask.npy', mask)
            job['mask_path'] = str(job_dir / 'mask.npy')
        rtsp_live.write_json(job_dir / 'job.json', job)

    (args.output / 'replay_inputs.json').write_text(json.dumps(inputs, indent=2))
    cancel = threading.Event()
    finished, errors, detections = [], [], []
    original_detect = video_processing.create_line_video_clips
    original_index = video_processing._SavedEventIndex
    original_recorder = file_utils.save_rtsp_video_segments
    source_context = {}
    worker_profiles = {}
    profile_lock = threading.Lock()
    restored = []
    product_keys = []
    if hasattr(products.ScheduleProducts, '_timelapse_key'):
        original_key = products.ScheduleProducts._timelapse_key
        def observe_key(service, job):
            key = original_key(service, job)
            product_keys.append(dict(key=key, options=dict(job['options']),
                archives=service._source_revision(job),
                samples=[(index, when, service._source_stamp(service.cache.root /
                    (job['id'] + f'_sample_{int(index):06d}') / 'frame.png'))
                    for index, when in sorted(job['samples'].items())]))
            return key
        restored.append((products.ScheduleProducts, '_timelapse_key', original_key))
        products.ScheduleProducts._timelapse_key = observe_key
    if args.profile:
        def install_profile(owner, name):
            original = getattr(owner, name)
            def wrapped(*values, **kwargs):
                thread = threading.current_thread()
                if not (thread.name.startswith('rtsp-') or thread.name in ('RTSPProposalPipeline', 'RTSPWindowPipeline', 'RTSPScheduleProducts')):
                    return original(*values, **kwargs)
                key = (thread.ident, name)
                with profile_lock:
                    profiler = worker_profiles.setdefault(key, cProfile.Profile())
                profiler.enable()
                try:
                    return original(*values, **kwargs)
                finally:
                    profiler.disable()
            restored.append((owner, name, original))
            setattr(owner, name, wrapped)
        for owner, name in [(moving_point_detector, 'detect_moving_point_tracks'),
                            (short_flash_detector.ShortFlashDetector, 'consume'),
                            (image_processing, 'detect_lines'),
                            (video_processing, '_process_finer_detection_worker'),
                            (video_processing, '_coarse_proposals'),
                            (image_processing._WindowExtrema, 'reduce'),
                            (rtsp_live.SegmentCapture, 'read'),
                            (products.ScheduleProducts, 'tick'),
                            (model, 'predict_noise_gate')]:
            install_profile(owner, name)

    class ObservedIndex(original_index):
        def append(self, item):
            super().append(item)
            detections.append(dict(source=source_context.get('source'), detection=item))

    def detect(*values, **kwargs):
        source_context['source'] = kwargs.get('source', values[0] if values else None)
        profiler = cProfile.Profile() if args.profile else None
        try:
            if profiler:
                profiler.enable()
            result = original_detect(*values, **kwargs)
            finished.append(source_context['source'])
            return result
        except Exception as exc:
            errors.append(repr(exc))
            raise
        finally:
            if profiler:
                profiler.disable()
                profiler.dump_stats(str(args.output / f'continuous-{len(finished):04d}.pstats'))

    started = time.perf_counter()
    completed = False
    def recorder(*_values, **_kwargs):
        nonlocal completed
        while not cancel.wait(.05):
            if errors:
                raise RuntimeError(errors)
            consumed = all(not path.exists() or Path(str(path)+'.analysis_consumed.json').exists() for path in paths)
            jobs = list((root / '.schedule_jobs').glob('*/job.json'))
            products_done = all(json.loads(path.read_text())['state'] == 'complete' for path in jobs)
            if consumed and len(finished) == len(sessions) and products_done:
                completed = True
                return
            if time.perf_counter() - started > args.timeout:
                raise TimeoutError('Continuous replay did not drain before the timeout')

    video_processing.create_line_video_clips = detect
    video_processing._SavedEventIndex = ObservedIndex
    file_utils.save_rtsp_video_segments = recorder
    before = resource.getrusage(resource.RUSAGE_SELF)
    try:
        with (args.output / 'pipeline.log').open('w') as log, contextlib.redirect_stdout(log):
            # These are the actual GUI entry point's arguments. Incoming network
            # wait is replaced by the already captured spool; analysis is unchanged.
            file_utils.rtsp_save_and_process_thread_target(
                rtsp_url='rtsp://isolated-recording-replay', save_root=str(root),
                segment_duration=config.RTSP_SEGMENT_DURATION, scan_interval=config.RTSP_SCAN_INTERVAL,
                progress_callback=lambda value: print(value[0]) if str(value[0]).startswith('[自動作成]') else None, mask=mask,
                meteor_save_path=str(args.output/'meteor'), not_meteor_save_path=str(args.output/'not_meteor'),
                cancel_flag=cancel, save_options=options, interval=float(settings.get('interval', 1)),
                duration=float(settings.get('duration', 1)), summary_video_config=settings.get('summary_video_config'),
                time_limit_enabled=bool(settings.get('rtsp_time_limit_enabled', True)),
                start_hour=clock[0], start_minute=clock[1], end_hour=clock[2], end_minute=clock[3],
                max_workers=int(settings.get('concurrency', 4)), dark_frame=correction,
                notify_on_detection=False, noise_twin_options=noise, rtsp_fps=config.RTSP_FPS,
                schedule_product_options=product_options)
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
    finally:
        cancel.set()
        products.stop_background_products()
        video_processing.create_line_video_clips = original_detect
        video_processing._SavedEventIndex = original_index
        file_utils.save_rtsp_video_segments = original_recorder
        for owner, name, original in restored:
            setattr(owner, name, original)
        for index, ((_thread, name), profiler) in enumerate(worker_profiles.items()):
            profiler.dump_stats(str(args.output / f'worker-{name}-{index:03d}.pstats'))
    seconds = time.perf_counter()-started
    (args.output / 'product_keys.json').write_text(json.dumps(product_keys, indent=2))
    after = resource.getrusage(resource.RUSAGE_SELF)
    items = [dict(row['detection'], source_video=str(Path(row['source']).relative_to(root))) for row in detections]
    return dict(seconds=seconds, input_seconds=duration, processing_ratio=seconds/duration,
                target_ratio=.25, meets_target=seconds <= duration*.25, faster_than_realtime=seconds < duration,
                completed=completed, sessions=len(sessions), chunks=len(paths),
                profile_enabled=args.profile, media_enabled=not args.no_save,
                input_frames=sum(item['frame_count'] for item in inputs),
                detection_interval=config.RTSP_LIVE_DETECT_INTERVAL, scheduled_products=product_options,
                schedule_window=None if schedule_window is None else [value.isoformat() for value in schedule_window],
                errors=errors, meteor=sum(item.get('label') == 'meteor' for item in items),
                background_gate=sum(item.get('classification_stage') == 'foreground_gate' for item in items),
                peak_rss_bytes=after.ru_maxrss,
                cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime), items
