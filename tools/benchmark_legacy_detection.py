"""Benchmark a legacy code snapshot with saved UI settings and isolated outputs.

Use compare_detection_runs.py to check probabilities, geometry and media.
Limited runs stop coarse proposal windows only; finer lookahead is unchanged.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-root', type=Path, required=True)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--windows', type=int, default=0)
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--model', type=Path, help='Override the selected model without editing UI settings')
    parser.add_argument('--concurrency', type=int, default=1)
    parser.add_argument('--realtime', action='store_true',
                        help='Replay through the real continuous RTSP runner, archiver and scheduled products')
    parser.add_argument('--live', action='store_true',help='Use a real RTSP connection and the original recorder')
    parser.add_argument('--stream-url',help='Unauthenticated loopback RTSP URL for the equal-speed live test')
    parser.add_argument('--live-seconds',type=float,default=600)
    parser.add_argument('--backlog-seconds',type=float,default=60)
    parser.add_argument('--replay-seconds', type=float, default=0,
                        help='Select this many seconds of existing spool input (0 selects all)')
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--no-products', action='store_true', help='Diagnostic replay without scheduled products')
    parser.add_argument('--opencv-threads', type=int, default=0, help='Diagnostic OpenCV worker budget override')
    parser.add_argument('sources', type=Path, nargs='*')
    args = parser.parse_args()
    if args.windows < 0 or args.concurrency < 1:
        parser.error('Windows must be nonnegative and concurrency must be positive')
    if args.replay_seconds < 0 or args.timeout <= 0 or (args.realtime and args.windows):
        parser.error('Replay seconds must be nonnegative, timeout positive; realtime cannot limit proposal windows')
    if args.live and (args.realtime or args.sources or args.windows or args.no_save or args.profile):
        parser.error('Live mode requires full saving and cannot combine with replay, sources, windows or profiles')
    if not args.live and not args.sources:
        parser.error('Recorded replay needs input sources')
    if args.live and not 0 <= args.backlog_seconds < args.live_seconds:
        parser.error('Live backlog delay must be shorter than capture duration')
    if args.live and not args.stream_url:
        parser.error('Live verification needs an explicit loopback RTSP test stream')
    if args.output.exists():
        parser.error('Output must be a new directory')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True)
    sys.path.insert(0, str(args.code_root.resolve()))
    for key in ('METEOR_NATIVE_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ.setdefault(key, '4')
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    settings = json.loads(args.settings.read_text())
    if args.model:
        settings['selected_model_path'] = str(args.model.resolve())
    import cv2
    import numpy as np
    import torch
    import config
    import model
    import video_processing
    from obs_meteor_overlay.legacy_detection_mask import LegacyDetectionMask
    if args.opencv_threads:
        cv2.setNumThreads(args.opencv_threads)

    config.RTSP_SAVE_ROOT = str(args.settings.resolve().parent / 'rtsp')
    config.TEMP_CLIP_DIR = str(args.output / 'temporary')
    config.LIGHTEN_BLEND_CACHE_DIR = str(args.output / 'lighten_cache')
    mapping = {
        'min_line_length': 'MIN_LINE_LENGTH', 'border_size': 'BORDER_SIZE',
        'duplicate_thresh': 'DUPLICATE_DETECTION_THRESHOLD',
        'meteor_prob': 'METEOR_PROBABILITY_THRESHOLD',
        'finer_window_sec': 'FINER_DETECT_WINDOW_SECONDS',
        'finer_comp_step': 'FINER_COMPOSITE_STEP',
        'finer_min_length': 'FINER_DETECT_MIN_LENGTH',
        'finer_padding_sec': 'FINER_DETECT_PADDING_SECONDS',
        'finer_cutout_size': 'FINER_CUTOUT_SIZE',
        'cutout_size': 'CUTOUT_SIZE',
        'airplane_dur_thresh': 'AIRPLANE_DURATION_THRESHOLD',
        'airplane_frame_thresh': 'AIRPLANE_FRAME_THRESHOLD',
        'tracking_dist_thresh': 'TRACKING_DISTANCE_THRESHOLD',
        'max_clip_dur': 'MAX_CLIP_DURATION', 'clip_dur_sec': 'CLIP_DURATION_SECONDS',
        'rtsp_scale_lower': 'RTSP_SCALE_LOWER', 'rtsp_scale_upper': 'RTSP_SCALE_UPPER',
    }
    for key, name in mapping.items():
        if key in settings.get('advanced_settings', {}):
            setattr(config, name, type(getattr(config, name))(
                float(settings['advanced_settings'][key])))
    preset = (config.RTSP_PRESET_CLEAR_SKY if settings.get('rtsp_preset') == 'clear'
              else config.RTSP_PRESET_CLOUDY)
    for key, value in preset.items():
        if key != 'name':
            setattr(config, 'RTSP_' + key.upper(), value)
    mask_reader = LegacyDetectionMask(args.settings)
    mask_reader.refresh()
    ok, message = model.reload_model(settings['selected_model_path'])
    if not ok:
        raise RuntimeError(message)
    if settings.get('use_plate_solve'):
        raise ValueError('This benchmark does not construct a WCS; disable plate solving in a settings copy')
    timestamp = settings.get('full_video_timestamp', {})
    options = dict(settings.get('save_options', {}))
    options.update(
        ml_training_export_enabled=settings.get('ml_training_export_enabled', False),
        ml_training_data_root=str(args.output / 'ml_training_data'),
        ml_training_min_probability=settings.get('ml_training_min_probability', config.DEFAULT_ML_TRAINING_MIN_PROBABILITY),
        auto_video_mask_enabled=settings.get('auto_video_mask_enabled', False),
        auto_video_mask_cache_dir=str(args.output / 'auto_mask_cache'),
        full_video_timestamp_enabled=timestamp.get('enabled', config.FULL_VIDEO_TIMESTAMP_ENABLED),
        full_video_timestamp_position=timestamp.get('position', config.FULL_VIDEO_TIMESTAMP_POSITION),
        full_video_timestamp_size_percent=float(timestamp.get('size_percent', config.FULL_VIDEO_TIMESTAMP_SIZE_PERCENT)),
    )
    if args.no_save:
        options.update({key: False for key in ('video', 'full_video', 'denoised_full_video',
                                              'cutout', 'full', 'composite', 'info',
                                              'summary', 'ml_training_export_enabled')})
    noise = dict(enabled=settings.get('noise_twin_enabled', False),
                 model_path=settings.get('noise_twin_model_path', ''),
                 require_validated=True, temporal_mean_frames=int(settings.get('temporal_mean_frames', 0)))
    correction = None
    if settings.get('apply_rtsp_dark'):
        with np.load(args.settings.with_name('rtsp_dark_frame.npz'), allow_pickle=False) as archive:
            correction = (archive['fixed_correction'].astype(np.int16) if 'fixed_correction' in archive
                          else archive['dark_frame'].astype(np.uint8))
    if args.windows:
        from itertools import islice
        original = video_processing.image_processing.create_diff_images
        def limited(*values, **kwargs):
            yield from islice(original(*values, **kwargs), args.windows)
        video_processing.image_processing.create_diff_images = limited
    model.predict_meteor_event([np.zeros((256, 256, 3), np.uint8) for _ in range(8)],
                              cutout_rect=(0, 0, 256, 256), detected_line=((100, 100), (120, 120)), frame_rate=25)
    if str(model.device) == 'mps':
        torch.mps.synchronize()
    runtime = dict(device=str(model.device), torch_threads=torch.get_num_threads(),
                   opencv_threads=cv2.getNumThreads(), concurrency=args.concurrency,
                   model=model.get_active_model_info(), threshold=model.get_meteor_decision_threshold(),
                   windows=args.windows, save_options=options, mask_revision=mask_reader.revision,
                   settings_sha256=hashlib.sha256(args.settings.read_bytes()).hexdigest(),
                   model_sha256=hashlib.sha256(Path(settings['selected_model_path']).read_bytes()).hexdigest(),
                   code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in args.code_root.glob('*.py')},
                   sources=[str(p.resolve()) for p in args.sources])
    runtime['workflow'] = 'live_rtsp' if args.live else 'continuous_rtsp' if args.realtime else 'archive_files'
    (args.output / 'runtime.json').write_text(json.dumps(runtime, indent=2))
    def process(source):
        items = video_processing.create_line_video_clips(
            str(source.resolve()), interval=float(settings.get('interval', 1)),
            duration=float(settings.get('duration', 1)), min_length=config.MIN_LINE_LENGTH,
            mask=mask_reader.mask, save_options=options,
            meteor_save_path=str(args.output / 'meteor'),
            not_meteor_save_path=str(args.output / 'not_meteor'),
            summary_video_config=settings.get('summary_video_config'),
            fixed_pattern_correction=correction, noise_twin_options=noise)
        for item in items:
            item['source_video'] = str(source.resolve())
        return items
    if args.realtime or args.live:
        if args.live:
            from verify_live_rtsp import run_live
            result, items = run_live(args,settings,options,noise,correction,mask_reader.mask)
        else:
            from benchmark_realtime_rtsp import run_replay
            result, items = run_replay(args, settings, options, noise, correction, mask_reader.mask)
        (args.output / 'detections.json').write_text(json.dumps(items, indent=2,
            default=lambda x: x.item() if hasattr(x, 'item') else str(x)))
        (args.output / 'result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result))
        return
    import cProfile
    profiler = cProfile.Profile()
    cpu_before = resource.getrusage(resource.RUSAGE_SELF)
    started = time.perf_counter()
    with (args.output / 'pipeline.log').open('w') as log, contextlib.redirect_stdout(log):
        if args.profile:
            profiler.enable()
        if args.concurrency == 1:
            items = [item for source in args.sources for item in process(source)]
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                items = [item for group in executor.map(process, args.sources) for item in group]
        if str(model.device) == 'mps':
            torch.mps.synchronize()
        if args.profile:
            profiler.disable()
            profiler.dump_stats(str(args.output / 'profile.pstats'))
    seconds = time.perf_counter() - started
    cpu_after = resource.getrusage(resource.RUSAGE_SELF)
    (args.output / 'detections.json').write_text(json.dumps(items, indent=2,
        default=lambda x: x.item() if hasattr(x, 'item') else str(x)))
    result = dict(seconds=seconds, meteor=sum(x['label'] == 'meteor' for x in items),
                  not_meteor=sum(x['label'] != 'meteor' for x in items),
                  peak_rss_bytes=cpu_after.ru_maxrss,
                  cpu_seconds=cpu_after.ru_utime + cpu_after.ru_stime - cpu_before.ru_utime - cpu_before.ru_stime)
    (args.output / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
