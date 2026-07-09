import argparse
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np


def log(message: str) -> None:
    print(message, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture an averaged RTSP dark frame.")
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preview", required=True)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    log(f"STATUS connecting {args.url}")

    cap = cv2.VideoCapture(args.url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
    if not cap.isOpened():
        log("ERROR RTSPストリームを開けません。")
        return 2

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log(f"STATUS opened {width}x{height}")

    for i in range(max(0, args.warmup)):
        cap.read()
        if (i + 1) == args.warmup:
            log(f"STATUS warmup {i + 1}/{args.warmup}")

    frames = []
    attempts = 0
    max_attempts = max(args.frames * 4, args.frames + 10)
    started = time.time()

    while len(frames) < args.frames and attempts < max_attempts:
        attempts += 1
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame.astype(np.float32))
            if len(frames) == 1 or len(frames) % 5 == 0 or len(frames) == args.frames:
                log(f"PROGRESS {len(frames)} {args.frames}")
        else:
            time.sleep(0.05)

    cap.release()

    if len(frames) < 10:
        log(f"ERROR 十分なフレームを取得できませんでした ({len(frames)} frames)。")
        return 3

    dark_frame = np.clip(np.mean(frames, axis=0), 0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(
        args.output,
        dark_frame=dark_frame,
        source_url=np.array(args.url),
        frames=np.array(len(frames)),
        created_at=np.array(datetime.now().isoformat(timespec="seconds")),
    )
    cv2.imwrite(args.preview, dark_frame)

    elapsed = time.time() - started
    log(
        "RESULT "
        f"frames={len(frames)} elapsed={elapsed:.2f} "
        f"shape={dark_frame.shape[1]}x{dark_frame.shape[0]} "
        f"mean={float(dark_frame.mean()):.2f} min={int(dark_frame.min())} max={int(dark_frame.max())}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
