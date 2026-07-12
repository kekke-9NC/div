import threading
import queue
import os
import time
from typing import List, Dict, Any, Optional, Callable

import network_copy
import video_processing
import config


def compute_worker_plan(requested_workers: int, logical_cpus: Optional[int] = None) -> Dict[str, int]:
    """Allocate one CPU budget across videos, OpenCV, and nested finer detection."""
    cpus = max(1, int(logical_cpus or os.cpu_count() or 1))
    reserved_for_ui = 2 if cpus >= 8 else 1 if cpus >= 3 else 0
    usable = max(1, cpus - reserved_for_ui)
    video_workers = max(1, min(int(requested_workers or 1), usable, 6))
    per_video_budget = max(1, usable // video_workers)
    return {
        "logical_cpus": cpus,
        "reserved_for_ui": reserved_for_ui,
        "usable_cpus": usable,
        "video_workers": video_workers,
        "download_workers": min(2, video_workers),
        "opencv_threads": max(1, min(3, per_video_budget)),
        # OpenCV/FFmpeg otherwise opens 16 H.264 decoder threads *and* a
        # similarly-sized helper pool for every VideoCapture on this Mac.
        # Two decoder threads per concurrently processed 1080p video retained
        # throughput in measurements while avoiding ~100 native threads.
        "decoder_threads": 2 if video_workers > 1 else max(2, min(4, per_video_budget)),
        # Avoid multiplying full-frame serialization across every video worker.
        "finer_workers": (
            max(1, min(2, per_video_budget)) if video_workers == 1 else 1
        ),
    }


class _ProgressLimiter:
    """Keep high-volume archive messages from starving Tk's event loop."""
    def __init__(self, callback, total: int, min_interval: float = 0.20):
        self.callback = callback
        self.total = total
        self.min_interval = min_interval
        self.lock = threading.Lock()
        self.last_progress_time = 0.0
        self.last_progress = 0
        self.last_heartbeat_time = 0.0
        self.not_meteor_count = 0

    def message(self, payload):
        message, value = payload
        if isinstance(message, str):
            if message.startswith("ダウンロード完了:"):
                return
            if message.startswith("検出 ") and ": not_meteor " in message:
                with self.lock:
                    self.not_meteor_count += 1
                return
            if message.lstrip().startswith("-> Not Meteor:"):
                return
            if message.lstrip().startswith("-> ") and "Summary:" not in message:
                return
        self.callback(payload)

    def heartbeat(self, processed: int, active: int, queued: int, pending: int):
        """Emit a low-frequency proof-of-life while long videos are scanning."""
        now = time.monotonic()
        with self.lock:
            if now - self.last_heartbeat_time < 10.0:
                return
            self.last_heartbeat_time = now
        self.callback((
            f"処理状況: 完了 {processed}/{self.total}, "
            f"動画処理中 {active}, 読込待ち {queued}, 未準備 {pending}",
            None,
        ))

    def finish(self):
        with self.lock:
            count = self.not_meteor_count
        if count:
            self.callback((f"非流星候補: {count}件（個別ログは省略）", None))

    def progress(self, processed: int):
        now = time.monotonic()
        with self.lock:
            if processed < self.total and now - self.last_progress_time < self.min_interval:
                self.last_progress = processed
                return
            self.last_progress_time = now
            self.last_progress = processed
        self.callback((None, (processed, self.total)))


def run_pipeline(
    sources: List[Dict[str, Any]],
    max_workers: int,
    interval: float,
    duration: float,
    mask: Optional[Any],
    global_wcs_info: Optional[Dict[str, Any]],
    plate_solve_mask: Optional[Any],
    meteor_save_path: str,
    not_meteor_save_path: str,
    cancel_flag: threading.Event,
    progress_callback,
    save_options: Dict[str, bool] = None,
    summary_video_config: Optional[List[Dict[str, Any]]] = None,
    tmp_root: Optional[str] = None,
    status_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    fixed_pattern_correction: Optional[Any] = None,
):
    """Run a 2-stage pipeline: download workers and processing workers.

    - max_workers: number of downloaders and number of processors.
    - progress_callback: callable(message, value)
    - cancel_flag: threading.Event to signal cancellation
    """
    if not sources:
        return

    worker_plan = compute_worker_plan(max_workers)
    max_workers = worker_plan["video_workers"]
    download_worker_count = worker_plan["download_workers"]
    save_options = dict(save_options or {})
    save_options["finer_detection_workers"] = worker_plan["finer_workers"]
    save_options["video_decoder_threads"] = worker_plan["decoder_threads"]
    try:
        # OpenCV otherwise creates a full CPU pool inside every video worker.
        import cv2
        cv2.setNumThreads(worker_plan["opencv_threads"])
    except Exception:
        pass
    progress = _ProgressLimiter(progress_callback, len(sources))
    progress_callback((
        "並列処理構成: "
        f"動画={max_workers}, 詳細検出/動画={worker_plan['finer_workers']}, "
        f"OpenCV={worker_plan['opencv_threads']}, デコーダ/動画={worker_plan['decoder_threads']}, "
        f"UI予約={worker_plan['reserved_for_ui']}コア",
        None,
    ))

    # simple shared index for downloaders
    src_lock = threading.Lock()
    src_index = {'i': 0}
    downloaders_done = threading.Event()
    pipeline_done = threading.Event()

    # Keep only a small rolling window; thousands of queued jobs add no throughput.
    task_q: queue.Queue = queue.Queue(maxsize=max_workers * 3)

    # track processor busy state so the UI can show which slots are active
    processors_busy = [False] * max_workers

    def get_next_source():
        with src_lock:
            i = src_index['i']
            if i >= len(sources):
                return None
            src_index['i'] += 1
            return sources[i]

    # downloader will try up to retry_count times with exponential backoff
    retry_count = 3
    retry_backoff = 2.0

    def downloader_thread_fn(tid: int):
        while not cancel_flag.is_set():
            s = get_next_source()
            if s is None:
                break
            path = s.get('path')
            attempt = 0
            last_exc = None
            while attempt < retry_count and not cancel_flag.is_set():
                try:
                    local_path, tmp_dir = network_copy.ensure_local_copy(path, tmp_root=tmp_root, cancel_flag=cancel_flag)
                    # push to queue for processors (allow cancellation while waiting for space)
                    item = {'local_path': local_path, 'tmp_dir': tmp_dir, 'source': s}
                    if cancel_flag.is_set():
                        # cleanup if cancelled before enqueue
                        try:
                            network_copy.cleanup_tempdir(tmp_dir)
                        except Exception:
                            pass
                        break
                    # try until put succeeds or cancel set
                    while not cancel_flag.is_set():
                        try:
                            task_q.put(item, timeout=0.2)
                            break
                        except queue.Full:
                            continue
                    if cancel_flag.is_set():
                        # cancelled while waiting to enqueue; cleanup temp and exit retry loop
                        try:
                            network_copy.cleanup_tempdir(tmp_dir)
                        except Exception:
                            pass
                        break
                    try:
                        progress.message((f"ダウンロード完了: {os.path.basename(path)}", None))
                    except Exception:
                        pass
                    last_exc = None
                    break
                except network_copy.CancelledCopy:
                    # cancel requested during copy; log and exit this downloader
                    try:
                        progress.message((f"ダウンロードをキャンセル: {path}", None))
                    except Exception:
                        pass
                    return
                except Exception as e:
                    last_exc = e
                    attempt += 1
                    try:
                        progress.message((f"ダウンロード失敗 (試行 {attempt}/{retry_count}): {path} ({e})", None))
                    except Exception:
                        pass
                    if attempt < retry_count:
                        time.sleep(retry_backoff ** attempt)

            if last_exc is not None:
                # final failure after retries
                try:
                    progress.message((f"ダウンロード失敗: {path} ({last_exc})", None))
                except Exception:
                    pass
        # downloader exits

    def processor_thread_fn(tid: int):
        while True:
            if cancel_flag.is_set() and downloaders_done.is_set() and task_q.empty():
                break
            try:
                item = task_q.get(timeout=1.0)
            except queue.Empty:
                # check if all downloaders are done and queue is empty
                if downloaders_done.is_set() and task_q.empty():
                    break
                else:
                    continue

            if item is None:
                task_q.task_done()
                break

            local_path = item.get('local_path')
            tmp_dir = item.get('tmp_dir')
            src = item.get('source')

            try:
                if cancel_flag.is_set():
                    # Still acknowledge every queued item.  Exiting here used
                    # to leave unfinished_tasks non-zero and task_q.join()
                    # could hang forever during cancellation.
                    continue
                # call the existing processing function
                processors_busy[tid] = True
                video_processing.create_line_video_clips(
                    source=local_path,
                    is_rtsp=False,
                    interval=interval,
                    duration=duration,
                    min_length=config.MIN_LINE_LENGTH,
                    mask=mask,
                    progress_callback=progress.message,
                    meteor_save_path=meteor_save_path,
                    not_meteor_save_path=not_meteor_save_path,
                    use_plate_solve=(global_wcs_info is not None),
                    global_wcs_info=global_wcs_info,
                    plate_solve_mask=plate_solve_mask,
                    cancel_flag=cancel_flag,
                    save_options=save_options or {},
                    summary_video_config=summary_video_config,
                    fixed_pattern_correction=fixed_pattern_correction,
                )
            except Exception as e:
                try:
                    progress.message((f"処理エラー ({local_path}): {e}", None))
                except Exception:
                    pass
            finally:
                processors_busy[tid] = False
                # update processed count and notify progress
                try:
                    with src_lock:
                        if 'processed' not in src_index:
                            src_index['processed'] = 0
                        src_index['processed'] += 1
                        processed = src_index['processed']
                        total = len(sources)
                    # send a progress update: (None, (processed, total))
                    try:
                        progress.progress(processed)
                    except Exception:
                        pass
                except Exception:
                    pass
                # cleanup temp dir if any
                try:
                    network_copy.cleanup_tempdir(tmp_dir)
                except Exception:
                    pass
                task_q.task_done()

    # optional status updater thread
    def status_updater_fn():
        # keep reporting until everything is done or cancel
        while not cancel_flag.is_set() and not pipeline_done.is_set():
            try:
                qsz = task_q.qsize()
                with src_lock:
                    pending = max(0, len(sources) - src_index['i'])
                    processed = int(src_index.get('processed', 0))
                # copy busy list
                busy_copy = list(processors_busy)
                progress.heartbeat(
                    processed, sum(1 for busy in busy_copy if busy), qsz, pending
                )
                if status_callback:
                    try:
                        status_callback({'download_queue_size': qsz, 'pending_sources': pending, 'processors_busy': busy_copy})
                    except Exception:
                        pass
                time.sleep(0.5)
            except Exception:
                time.sleep(0.5)

    status_thread = None
    if status_callback:
        status_thread = threading.Thread(target=status_updater_fn, daemon=True)
        status_thread.start()

    # start downloaders and processors
    downloaders = [
        threading.Thread(target=downloader_thread_fn, args=(i,), name=f"video-download-{i}")
        for i in range(download_worker_count)
    ]
    processors = [
        threading.Thread(target=processor_thread_fn, args=(i,), name=f"video-process-{i}")
        for i in range(max_workers)
    ]

    for t in processors:
        t.start()
    for t in downloaders:
        t.start()

    # wait for downloaders to finish
    for t in downloaders:
        t.join()
    downloaders_done.set()

    # wait for queue to be processed
    task_q.join()

    # processors will exit when queue empty and all downloads done
    for t in processors:
        t.join()

    pipeline_done.set()
    progress.finish()

    # stop status thread
    if status_thread:
        try:
            status_thread.join(timeout=1.0)
        except Exception:
            pass
