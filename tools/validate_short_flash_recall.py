"""Run the real legacy pipeline offline on the visually confirmed recording.

No live capture, settings writes, training exports or notification sounds.
Example:
  python tools/validate_short_flash_recall.py --recording-root /path/to/rtsp \
    --settings /path/to/app_settings.json --output /tmp/recall-validation
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('METEOR_NATIVE_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')


def match_events(fixture, detections):
    matched, missing = [], []
    for event in fixture['events']:
        x, y, width, height = event['roi']
        hits = [d for d in detections if d['minute'] == event['minute']
                and d['label'] == 'meteor'
                and d['start_frame'] <= event['last_frame'] + 2
                and d['end_frame'] >= event['first_frame'] - 2
                and x <= d['center'][0] <= x + width
                and y <= d['center'][1] <= y + height]
        (matched if hits else missing).append(event['id'])
    return {'expected': len(fixture['events']), 'matched': matched, 'missing': missing}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording-root', type=Path, required=True)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, default=ROOT / 'tests/fixtures/short_flash_20261011.json')
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--baseline', action='store_true')
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())
    fixture = json.loads(args.fixture.read_text())
    output = args.output.expanduser().resolve()
    # The benchmark must not write into the active app's media directories.
    for key in ('meteor_save_path', 'not_meteor_save_path', 'ml_training_data_root'):
        if settings.get(key):
            live = Path(settings[key]).expanduser().resolve()
            if output == live or live in output.parents:
                parser.error('output must be outside the live media directories')
    output.mkdir(parents=True, exist_ok=True)
    import cv2
    import torch
    cv2.setNumThreads(1)
    torch.backends.mps.is_available = lambda: False
    torch.cuda.is_available = lambda: False
    import config
    import model
    import video_processing
    from obs_meteor_overlay.legacy_detection_mask import LegacyDetectionMask
    ok, message = model.reload_model(settings['selected_model_path'])
    if not ok:
        raise RuntimeError(message)
    mask_reader = LegacyDetectionMask(args.settings)
    mask_reader.refresh()
    config.RTSP_SAVE_ROOT = str(args.recording_root.expanduser().resolve())
    config.RTSP_PARALLEL_ENABLED = args.parallel
    config.SHORT_FLASH_DETECT_ENABLED = not args.baseline
    config.TEMP_CLIP_DIR = str(output / 'temporary')
    preset = config.RTSP_PRESET_CLOUDY if settings.get('rtsp_preset') == 'cloudy' else config.RTSP_PRESET_CLEAR_SKY
    for key, value in preset.items():
        if key != 'name':
            setattr(config, 'RTSP_' + key.upper(), value)
    detections = []
    timings = []
    save_options = {key: False for key in ('video', 'cutout', 'full', 'composite',
                    'info', 'summary', 'full_video', 'denoised_full_video', 'ml_training_export_enabled')}
    save_options['video_decoder_threads'] = 1
    for minute in fixture['minutes']:
        source = args.recording_root / fixture['recording_date'] / fixture['hour'] / f'{minute:02d}.mp4'
        started = time.monotonic()
        with (output / f'{minute:02d}.log').open('w') as log, contextlib.redirect_stdout(log):
            items = video_processing.create_line_video_clips(
                str(source), mask=mask_reader.mask, save_options=save_options,
                meteor_save_path=str(output / 'meteor'), not_meteor_save_path=str(output / 'not_meteor'),
                notify_on_detection=False,
            )
        for item in items:
            item['minute'] = minute
            item['timestamp'] = str(item['timestamp'])
            detections.append(item)
        timings.append({'minute': minute, 'seconds': time.monotonic() - started})
        (output / 'detections.json').write_text(json.dumps(detections, indent=2, default=float))
        summary = match_events(fixture, detections)
        summary.update(timings=timings, model=model.get_active_model_info(),
                       threshold=model.get_meteor_decision_threshold(),
                       baseline=args.baseline, parallel=args.parallel)
        (output / 'summary.json').write_text(json.dumps(summary, indent=2, default=float))
        print(f'{minute:02d}.mp4: {len(items)} saved decisions, {timings[-1]["seconds"]:.1f}s', flush=True)
    print(f'Confirmed recording: {len(summary["matched"])}/{summary["expected"]}; missing {summary["missing"]}')
    return bool(summary['missing'])


if __name__ == '__main__':
    raise SystemExit(main())
