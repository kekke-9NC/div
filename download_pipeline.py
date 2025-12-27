import threading
import queue
import os
import time
from typing import List, Dict, Any, Optional, Callable

import network_copy
import video_processing
import config


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
):
    """Run a 2-stage pipeline: download workers and processing workers.

    - max_workers: number of downloaders and number of processors.
    - progress_callback: callable(message, value)
    - cancel_flag: threading.Event to signal cancellation
    """
    if not sources:
        return

    # simple shared index for downloaders
    src_lock = threading.Lock()
    src_index = {'i': 0}

    # Allow a larger buffer so downloaders can queue more items without
    # overwhelming processors; user requested max_workers * 20.
    task_q: queue.Queue = queue.Queue(maxsize=max_workers * 20)

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
                        progress_callback((f"ダウンロード完了: {os.path.basename(path)}", None))
                    except Exception:
                        pass
                    last_exc = None
                    break
                except network_copy.CancelledCopy:
                    # cancel requested during copy; log and exit this downloader
                    try:
                        progress_callback((f"ダウンロードをキャンセル: {path}", None))
                    except Exception:
                        pass
                    return
                except Exception as e:
                    last_exc = e
                    attempt += 1
                    try:
                        progress_callback((f"ダウンロード失敗 (試行 {attempt}/{retry_count}): {path} ({e})", None))
                    except Exception:
                        pass
                    if attempt < retry_count:
                        time.sleep(retry_backoff ** attempt)

            if last_exc is not None:
                # final failure after retries
                try:
                    progress_callback((f"ダウンロード失敗: {path} ({last_exc})", None))
                except Exception:
                    pass
        # downloader exits

    def processor_thread_fn(tid: int):
        while not cancel_flag.is_set():
            try:
                item = task_q.get(timeout=1.0)
            except queue.Empty:
                # check if all downloaders are done and queue is empty
                with src_lock:
                    done = src_index['i'] >= len(sources)
                if done and task_q.empty():
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
                # call the existing processing function
                processors_busy[tid] = True
                video_processing.create_line_video_clips(
                    local_path, False, interval, duration, config.MIN_LINE_LENGTH, mask, None, progress_callback,
                    meteor_save_path, not_meteor_save_path, (global_wcs_info is not None), global_wcs_info,
                    plate_solve_mask, config.RTSP_BUFFER_DURATION, cancel_flag, save_options or {}, False, summary_video_config
                )
            except Exception as e:
                try:
                    progress_callback((f"処理エラー ({local_path}): {e}", None))
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
                        progress_callback((None, (processed, total)))
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
        while not cancel_flag.is_set():
            try:
                qsz = task_q.qsize()
                with src_lock:
                    pending = max(0, len(sources) - src_index['i'])
                # copy busy list
                busy_copy = list(processors_busy)
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
    downloaders = [threading.Thread(target=downloader_thread_fn, args=(i,), daemon=True) for i in range(max_workers)]
    processors = [threading.Thread(target=processor_thread_fn, args=(i,), daemon=True) for i in range(max_workers)]

    for t in processors:
        t.start()
    for t in downloaders:
        t.start()

    # wait for downloaders to finish
    for t in downloaders:
        t.join()

    # wait for queue to be processed
    task_q.join()

    # processors will exit when queue empty and all downloads done
    for t in processors:
        t.join()

    # stop status thread
    if status_thread:
        try:
            status_thread.join(timeout=0.1)
        except Exception:
            pass
