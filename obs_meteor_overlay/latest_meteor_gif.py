#!/usr/bin/env python3
"""Keep the newest Meteor Detector event available to an OBS browser source."""

from __future__ import annotations

import argparse
from datetime import datetime, time as datetime_time, timedelta
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from zipfile import BadZipFile
from pathlib import Path

from legacy_detection_mask import LegacyDetectionMask, default_settings_path


STOP = False
EVENT_STAMP_RE = re.compile(r"^(\d{8}_\d{9})_meteor_")


def stop_handler(_signum: int, _frame: object) -> None:
    global STOP
    STOP = True


def event_id(path: Path) -> str:
    name = path.name
    for suffix in ("_summary.mp4", "_full_denoised.mp4", "_full.mp4", "_info.txt", ".mp4"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def event_datetime(path: Path) -> datetime | None:
    match = EVENT_STAMP_RE.match(event_id(path))
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S%f")
    except ValueError:
        return None


def newest_summary(meteor_dir: Path) -> Path | None:
    candidates = [
        path
        for path in meteor_dir.glob("*_meteor_*_summary.mp4")
        if path.is_file() and not path.name.startswith(".")
    ]
    if not candidates:
        return None
    # Detector filenames begin with local YYYYMMDD_HHMMSSmmm, so lexical order
    # is the event order and does not depend on copied-file modification times.
    return max(candidates, key=lambda path: path.name)


def newest_event_video(meteor_dir: Path) -> Path | None:
    summary = newest_summary(meteor_dir)
    if summary is not None:
        return summary
    candidates = [
        path
        for path in meteor_dir.glob("*_meteor_*.mp4")
        if path.is_file()
        and not path.name.endswith(("_full.mp4", "_full_denoised.mp4", "_summary.mp4"))
    ]
    return max(candidates, key=lambda path: path.name) if candidates else None


def zoom_source_for(source: Path, meteor_dir: Path) -> Path:
    """Prefer the square cutout clip for the enlarged meteor panel."""
    raw_request = raw_zoom_request(source, meteor_dir)
    if raw_request is not None:
        return raw_request[0]
    cutout = meteor_dir / f"{event_id(source)}.mp4"
    return cutout if cutout.is_file() else source


def raw_zoom_request(source: Path, meteor_dir: Path, require_video: bool = True) -> tuple[Path, int, int, float, float] | None:
    """Read the original video and detection coordinates saved with an event."""
    info_path = meteor_dir / f"{event_id(source)}_info.txt"
    try:
        text = info_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    source_match = re.search(r"^Source:\s*(.+)$", text, re.MULTILINE)
    range_match = re.search(r"^Frame Range \(Clip\):\s*(\d+)\s*-\s*(\d+)$", text, re.MULTILINE)
    center_match = re.search(
        r"^Detected Line Center \(px\):\s*\(([-+]?\d+(?:\.\d+)?),\s*([-+]?\d+(?:\.\d+)?)\)$",
        text,
        re.MULTILINE,
    )
    if not source_match or not range_match or not center_match:
        return None
    raw_source = Path(source_match.group(1).strip())
    if require_video and not raw_source.is_file():
        return None
    return (
        raw_source,
        int(range_match.group(1)),
        int(range_match.group(2)),
        float(center_match.group(1)),
        float(center_match.group(2)),
    )


def probe_video_geometry(ffmpeg: str, source: Path) -> tuple[int, int, float]:
    ffprobe = Path(ffmpeg).with_name("ffprobe")
    try:
        result = subprocess.run(
            [
                str(ffprobe),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate",
                "-of",
                "default=nw=1:nk=1",
                str(source),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        values = result.stdout.strip().splitlines()
        width, height = int(values[0]), int(values[1])
        numerator, denominator = values[2].split("/", 1)
        fps = float(numerator) / float(denominator)
        if width > 0 and height > 0 and fps > 0:
            return width, height, fps
    except (OSError, ValueError, IndexError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass
    return 1920, 1080, 25.0


def render_gif(source, output, ffmpeg, video_filter, mask=None, input_options=(), output_options=()):
    """Mask at original resolution, before any crop, scaling or palette."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="latest_meteor_", suffix=".gif", dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
    mask_png = temporary.with_suffix(".png")
    try:
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                   *input_options, "-i", str(source)]
        if mask is not None and mask.mask is not None:
            width, height, _ = probe_video_geometry(ffmpeg, source)
            mask.write_png(mask_png, width, height)
            command += ["-i", str(mask_png), "-filter_complex",
                        "[0:v]format=gbrp[video];[1:v]format=gbrp[mask];"
                        "[video][mask]blend=all_mode=multiply:repeatlast=1," + video_filter]
        else:
            command += ["-vf", video_filter]
        command += [*output_options, "-an", "-loop", "0", str(temporary)]
        subprocess.run(command, check=True, timeout=90)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
        mask_png.unlink(missing_ok=True)


def convert_raw_to_gif(event_source, meteor_dir, output, ffmpeg, mask=None, zoom=False):
    request = raw_zoom_request(event_source, meteor_dir)
    if request is None:
        return None
    raw_source, clip_start, clip_end, center_x, center_y = request
    if not stable_file(raw_source, 0.25):
        return None
    source_width, source_height, fps = probe_video_geometry(ffmpeg, raw_source)
    padding_frames = max(1, int(round(fps)))
    start_frame = max(0, clip_start - padding_frames)
    duration_seconds = max(0.25, (clip_end + padding_frames - start_frame + 1) / fps)
    if zoom:
        crop_size = min(256, source_width, source_height)
        x = max(0, min(source_width - crop_size, int(round(center_x - crop_size / 2))))
        y = max(0, min(source_height - crop_size, int(round(center_y - crop_size / 2))))
        video_filter = (f"crop={crop_size}:{crop_size}:{x}:{y},"
                        "tpad=start_mode=clone:start_duration=0.25:stop_mode=clone:stop_duration=0.5,"
                        "fps=12,scale=180:180:flags=lanczos,")
    else:
        video_filter = ("tpad=start_mode=clone:start_duration=0.5:stop_mode=clone:stop_duration=1.0,"
                        "fps=12,scale=320:180:force_original_aspect_ratio=decrease:flags=lanczos,"
                        "pad=320:180:(ow-iw)/2:(oh-ih)/2:color=black,")
    video_filter += "split[s0][s1];[s0]palettegen=max_colors=192[p];[s1][p]paletteuse=dither=sierra2_4a"
    render_gif(raw_source, output, ffmpeg, video_filter, mask,
               input_options=("-ss", f"{start_frame / fps:.3f}"),
               output_options=("-t", f"{duration_seconds:.3f}"))
    return raw_source


def convert_raw_zoom_to_gif(event_source, meteor_dir, output, ffmpeg, mask=None):
    return convert_raw_to_gif(event_source, meteor_dir, output, ffmpeg, mask, zoom=True)


def fallback_sources(source, meteor_dir, mask):
    full = meteor_dir / f"{event_id(source)}_full.mp4"
    cutout = meteor_dir / f"{event_id(source)}.mp4"
    if mask.mask is None:
        return full if full.is_file() else source, cutout if cutout.is_file() else source, mask
    if not full.is_file():
        raise ValueError("同じマスクを適用するための元映像・フルサイズ映像がありません")
    # Re-crop the full video rather than guessing how an old saved cutout
    # was cropped (legacy border margins/cutout sizes can differ).
    return full, full, mask


def convert_saved_zoom_to_gif(full, event_source, meteor_dir, output, ffmpeg, mask):
    request = raw_zoom_request(event_source, meteor_dir, require_video=False)
    if request is None:
        convert_to_gif(full, output, ffmpeg, 180, 180, mask)
        return
    _, _, _, center_x, center_y = request
    width, height, _ = probe_video_geometry(ffmpeg, full)
    size = min(256, width, height)
    x = max(0, min(width - size, int(round(center_x - size / 2))))
    y = max(0, min(height - size, int(round(center_y - size / 2))))
    video_filter = (
        f"crop={size}:{size}:{x}:{y},"
        "tpad=start_mode=clone:start_duration=0.25:stop_mode=clone:stop_duration=0.5,"
        "fps=12,scale=180:180:flags=lanczos,split[s0][s1];"
        "[s0]palettegen=max_colors=192[p];[s1][p]paletteuse=dither=sierra2_4a"
    )
    render_gif(full, output, ffmpeg, video_filter, mask)


def event_files(meteor_dir: Path) -> list[Path]:
    """Return one completed video per detected event for the night count."""
    candidates: dict[str, Path] = {}
    for path in meteor_dir.glob("*_meteor_*.mp4"):
        if not path.is_file() or path.name.endswith(("_full.mp4", "_full_denoised.mp4")):
            continue
        key = event_id(path)
        existing = candidates.get(key)
        if existing is None or path.name.endswith("_summary.mp4"):
            candidates[key] = path
    return list(candidates.values())


def night_window(now: datetime) -> tuple[datetime, datetime]:
    if now.time() < datetime_time(6, 0):
        start = datetime.combine(now.date() - timedelta(days=1), datetime_time(18, 0))
        end = datetime.combine(now.date(), datetime_time(6, 0))
    else:
        start = datetime.combine(now.date(), datetime_time(18, 0))
        end = datetime.combine(now.date() + timedelta(days=1), datetime_time(6, 0))
    return start, end


def night_count(meteor_dir: Path, now: datetime) -> tuple[int, datetime, datetime]:
    start, end = night_window(now)
    count = sum(
        1
        for path in event_files(meteor_dir)
        if (stamp := event_datetime(path)) is not None and start <= stamp < end and stamp <= now
    )
    return count, start, end


def write_hourly_statistics(meteor_dir: Path, output: Path, now: datetime) -> None:
    """Count unique saved events in the rolling hour, in six ten-minute bins.

    Filename timestamps are local camera times, just as for the night count.
    The left edge is exclusive and the right edge inclusive so adjacent bins
    partition the hour exactly, including events detected at the current time.
    """
    end = now.timestamp()
    start = end - 3600
    bins = [0] * 6
    for path in event_files(meteor_dir):
        stamp = event_datetime(path)
        if stamp is None:
            continue
        epoch = stamp.timestamp()
        if start < epoch <= end:
            # An exact boundary belongs to the interval ending at that boundary.
            index = min(5, max(0, int((epoch - start - 0.000001) / 600)))
            bins[index] += 1
    payload = {
        "updated_at_epoch": end,
        "window_start_epoch": start,
        "window_end_epoch": end,
        "last_hour_count": sum(bins),
        "bin_minutes": 10,
        "bins": bins,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, output)


def stable_file(path: Path, wait_seconds: float) -> bool:
    try:
        first = path.stat()
        if first.st_size <= 0:
            return False
        time.sleep(wait_seconds)
        second = path.stat()
    except FileNotFoundError:
        return False
    return first.st_size == second.st_size and first.st_mtime_ns == second.st_mtime_ns


def convert_to_gif(source, output, ffmpeg, width, height, mask=None):
    video_filter = (
        "tpad=start_mode=clone:start_duration=0.5:stop_mode=clone:stop_duration=1.0,"
        f"fps=12,scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
        "split[s0][s1];[s0]palettegen=max_colors=192[p];[s1][p]paletteuse=dither=sierra2_4a"
    )
    render_gif(source, output, ffmpeg, video_filter, mask)


def write_metadata(
    metadata_path: Path,
    source: Path | None,
    zoom_source: Path | None,
    count: int,
    window_start: datetime,
    window_end: datetime,
    mask_revision: str | None,
    mask_enabled: bool,
) -> None:
    detected = event_datetime(source) if source is not None else None
    payload = {
        "source": str(source) if source is not None else None,
        "zoom_source": str(zoom_source) if zoom_source is not None else None,
        "event": event_id(source) if source is not None else None,
        "detected_at": detected.strftime("%Y/%m/%d %H:%M:%S") if detected else None,
        "detected_at_date": detected.strftime("%Y/%m/%d") if detected else None,
        "detected_at_clock": detected.strftime("%H:%M:%S") if detected else None,
        "detected_at_epoch": detected.timestamp() if detected else None,
        "tonight_count": count,
        "tonight_window": f"{window_start.strftime('%m/%d %H:%M')}–{window_end.strftime('%m/%d %H:%M')}",
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mask_revision": mask_revision,
        "mask_enabled": mask_enabled,
    }
    temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, metadata_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meteor-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--legacy-settings", type=Path, default=default_settings_path())
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args()

    gif_path = args.output_dir / "latest_meteor.gif"
    zoom_gif_path = args.output_dir / "latest_meteor_zoom.gif"
    metadata_path = args.output_dir / "latest_meteor.json"
    last_rendered = None
    last_metadata = None
    mask = LegacyDetectionMask(args.legacy_settings)
    exported_revision = None
    last_mask_error = None
    display_source = display_zoom = None
    display_mask_revision = None
    display_mask_enabled = False
    if metadata_path.exists():
        try:
            previous = json.loads(metadata_path.read_text(encoding="utf-8"))
            display_source = Path(previous["source"]) if previous.get("source") else None
            display_zoom = Path(previous["zoom_source"]) if previous.get("zoom_source") else None
            display_mask_revision = previous.get("mask_revision")
            display_mask_enabled = bool(previous.get("mask_enabled", False))
        except (OSError, ValueError):
            pass

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OBS meteor overlay] watching {args.meteor_dir}", flush=True)

    while not STOP:
        # Publish counts independently of GIF conversion and mask availability.
        # A conversion failure must not freeze the rolling observation window.
        try:
            write_hourly_statistics(args.meteor_dir, args.output_dir / "hourly_stats.json", datetime.now())
        except (OSError, ValueError) as exc:
            print(f"[OBS meteor overlay] statistics update failed: {exc}", file=sys.stderr, flush=True)
        try:
            mask.refresh()
            if mask.revision != exported_revision:
                mask.write_png(args.output_dir / "detection_mask.png")
                mask.write_state(args.output_dir / "mask_state.json")
                exported_revision = mask.revision
                print(f"[OBS meteor overlay] legacy mask: {mask.mask_path} (enabled={mask.mask is not None})", flush=True)
            last_mask_error = None
        except (OSError, ValueError, EOFError, BadZipFile) as exc:
            if str(exc) != last_mask_error:
                print(f"[OBS meteor overlay] mask sync failed; retaining previous mask: {exc}", file=sys.stderr, flush=True)
                last_mask_error = str(exc)
        # Do not publish an unmasked first frame while enabled settings are
        # temporarily unreadable. Retry after the legacy UI has finished saving.
        if not mask.revision:
            time.sleep(max(0.5, args.poll_seconds))
            continue
        now = datetime.now()
        count, window_start, window_end = night_count(args.meteor_dir, now)
        source = newest_event_video(args.meteor_dir)
        render_key = (source.name, mask.revision) if source is not None else None
        if source is not None and render_key != last_rendered:
            if stable_file(source, min(0.5, max(0.1, args.poll_seconds / 2))):
                next_full = args.output_dir / ".next_full.gif"
                next_zoom = args.output_dir / ".next_zoom.gif"
                try:
                    print(f"[OBS meteor overlay] updating: {source.name}", flush=True)
                    full_raw = convert_raw_to_gif(source, args.meteor_dir, next_full, args.ffmpeg, mask)
                    zoom_raw = convert_raw_zoom_to_gif(source, args.meteor_dir, next_zoom, args.ffmpeg, mask)
                    if full_raw is None or zoom_raw is None:
                        full, zoom, zoom_mask = fallback_sources(source, args.meteor_dir, mask)
                        if not stable_file(full, 0.25) or not stable_file(zoom, 0.25):
                            raise OSError("流星映像を保存中です")
                        if full_raw is None:
                            convert_to_gif(full, next_full, args.ffmpeg, 320, 180, mask)
                        if zoom_raw is None:
                            if mask.mask is not None:
                                convert_saved_zoom_to_gif(zoom, source, args.meteor_dir, next_zoom, args.ffmpeg, zoom_mask)
                            else:
                                convert_to_gif(zoom, next_zoom, args.ffmpeg, 180, 180, zoom_mask)
                            zoom_raw = zoom
                    os.replace(next_full, gif_path)
                    os.replace(next_zoom, zoom_gif_path)
                    display_source, display_zoom = source, zoom_raw
                    display_mask_revision = mask.revision
                    display_mask_enabled = mask.mask is not None
                    last_rendered = render_key
                except (OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                    print(f"[OBS meteor overlay] update failed: {exc}", file=sys.stderr, flush=True)
                finally:
                    next_full.unlink(missing_ok=True)
                    next_zoom.unlink(missing_ok=True)
        source, current_zoom_source = display_source, display_zoom
        metadata_snapshot = json.dumps(
            {
                "source": str(source) if source is not None else None,
                "event": event_id(source) if source is not None else None,
                "zoom_source": str(current_zoom_source) if current_zoom_source is not None else None,
                "detected_at": (
                    event_datetime(source).strftime("%Y/%m/%d %H:%M:%S")
                    if source is not None and event_datetime(source) is not None
                    else None
                ),
                "tonight_count": count,
                "mask_revision": display_mask_revision,
                "tonight_window": f"{window_start.strftime('%m/%d %H:%M')}–{window_end.strftime('%m/%d %H:%M')}",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if metadata_snapshot != last_metadata:
            write_metadata(
                metadata_path,
                source,
                current_zoom_source,
                count,
                window_start,
                window_end,
                display_mask_revision,
                display_mask_enabled,
            )
            last_metadata = metadata_snapshot
        time.sleep(max(0.5, args.poll_seconds))

    print("[OBS meteor overlay] stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
