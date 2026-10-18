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
        return {"total_sources": 0, "failures": []}

    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")

    # simple shared index for downloaders
    src_lock = threading.Lock()
    src_index = {'i': 0}
    failure_records = []

    def record_failure(source_path, stage, error):
        with src_lock:
            failure_records.append({
                "path": source_path,
                "stage": stage,
                "error": str(error),
            })

    # Allow a larger buffer so downloaders can queue more items without
    # overwhelming processors; user requested max_workers * 20.
    task_q: queue.Queue = queue.Queue(maxsize=max_workers * 20)
    downloaders_done = threading.Event()

    # track processor busy state so the UI can show which slots are active
    processors_busy = [False] * max_workers

    def get_next_source():
        with src_lock:
            i = src_index['i']
            if i >= len(sources):
                return None
            src_index['i'] += 1
            return sources[i]

    def discard_task(item):
        try:
            if isinstance(item, dict):
                network_copy.cleanup_tempdir(item.get('tmp_dir'))
        except Exception:
            pass
        finally:
            task_q.task_done()

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
                    enqueued = False
                    while not cancel_flag.is_set():
                        try:
                            task_q.put(item, timeout=0.2)
                            enqueued = True
                            break
                        except queue.Full:
                            continue
                    if not enqueued:
                        # Cancelled while waiting to enqueue; no processor owns this temp copy.
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
                record_failure(path, "download", last_exc)
                try:
                    progress_callback((f"ダウンロード失敗: {path} ({last_exc})", None))
                except Exception:
                    pass
        # downloader exits

    def processor_thread_fn(tid: int):
        while True:
            if cancel_flag.is_set():
                try:
                    item = task_q.get_nowait()
                except queue.Empty:
                    break
                discard_task(item)
                continue

            try:
                item = task_q.get(timeout=1.0)
            except queue.Empty:
                # Sources can be assigned before their downloads finish. Wait
                # until every downloader has exited before declaring the queue done.
                if downloaders_done.is_set() and task_q.empty():
                    break
                else:
                    continue

            if item is None:
                task_q.task_done()
                break

            if cancel_flag.is_set():
                discard_task(item)
                continue

            local_path = item.get('local_path')
            tmp_dir = item.get('tmp_dir')
            src = item.get('source')

            try:
                # call the existing processing function
                processors_busy[tid] = True
                video_processing.create_line_video_clips(
                    local_path, False, interval, duration, config.MIN_LINE_LENGTH, mask, None, progress_callback,
                    meteor_save_path, not_meteor_save_path, (global_wcs_info is not None), global_wcs_info,
                    plate_solve_mask, config.RTSP_BUFFER_DURATION, cancel_flag, save_options or {}, False,
                    summary_video_config, raise_on_error=True
                )
            except Exception as e:
                record_failure(src.get('path') if isinstance(src, dict) else local_path, "processing", e)
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
    status_stop_event = threading.Event()
    def status_updater_fn():
        # Keep reporting while this run is active. A completed run does not set
        # the shared cancel flag, so use a run-local stop event as well.
        while not cancel_flag.is_set() and not status_stop_event.is_set():
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
            except Exception:
                pass
            status_stop_event.wait(0.5)

    status_thread = None
    if status_callback:
        status_thread = threading.Thread(target=status_updater_fn, daemon=True)
        status_thread.start()

    # start downloaders and processors
    downloaders = [threading.Thread(target=downloader_thread_fn, args=(i,), daemon=True) for i in range(max_workers)]
    processors = [threading.Thread(target=processor_thread_fn, args=(i,), daemon=True) for i in range(max_workers)]

    try:
        for t in processors:
            t.start()
        for t in downloaders:
            t.start()

        # wait for downloaders to finish
        for t in downloaders:
            t.join()
        downloaders_done.set()

        # A downloader may have enqueued just as a processor observed cancellation
        # and exited. With producers stopped, discard any such remaining work safely.
        if cancel_flag.is_set():
            while True:
                try:
                    item = task_q.get_nowait()
                except queue.Empty:
                    break
                discard_task(item)

        # wait for queue to be processed
        task_q.join()

        # processors will exit when queue empty and all downloads done
        for t in processors:
            t.join()
    finally:
        status_stop_event.set()
        if status_thread:
            status_thread.join()

    with src_lock:
        return {
            "total_sources": len(sources),
            "failures": list(failure_records),
        }
